import json
import os
import time
import re
import random
from datetime import datetime
from typing import Optional, Callable
from collections import Counter

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from loguru import logger
from torch.optim.optimizer import Optimizer
from torch_geometric.data import Data, Dataset, Batch, HeteroData
from torch_geometric.nn.conv import GCNConv, SAGEConv, HeteroConv
from torch_geometric.loader import DataLoader

from core.conf import ROOT_DIR, EMB_DIM, SPIQA_DIR, WRITE_DIR
from core.output import IndexFileManager, FilesManager
from core.prompt import ImageInfo
from core.train import info_nce_loss


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def run():
    set_seed(42)
    train_data_path = os.path.join(SPIQA_DIR, "train_val/SPIQA_train.json")
    val_data_path = os.path.join(SPIQA_DIR, "train_val/SPIQA_val.json")
    test_data_path = os.path.join(SPIQA_DIR, "test-A/SPIQA_testA.json")
    file_tag = datetime.now().strftime("%Y%m%d%H%M")
    write_dir = os.path.join(WRITE_DIR, f"graph_rag_train_{file_tag}")
    dataset_dir = "/home/chenyinbo/dataset/graph-rag-output/graph_dataset"
    model_path = "/home/chenyinbo/dataset/graph-rag-output/best_gnn_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingAlignmentGNN(['image', 'text'],
                                  [('image', 'be_reference', 'text'), ('text', 'reference', 'image')],
                                  {'image': EMB_DIM, 'text': EMB_DIM}, EMB_DIM).to(device)
    if not os.path.exists(model_path):
        train_loader = DataLoader(
            dataset=GraphDataset(train_data_path, os.path.join(dataset_dir, "train"), os.path.join(dataset_dir, "train_images"),
                                 os.path.join(dataset_dir, "train_texts"), os.path.join(ROOT_DIR, "train_hetero")),
            batch_size=128, shuffle=True, num_workers=4)
        val_loader = DataLoader(
            dataset=GraphDataset(val_data_path, os.path.join(dataset_dir, "val"), os.path.join(dataset_dir, "val_images"),
                                 os.path.join(dataset_dir, "val_texts"), os.path.join(ROOT_DIR, "val_hetero")),
            batch_size=128, shuffle=False, num_workers=4)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3, amsgrad=True)
        _train_and_validate(model, train_loader, val_loader, opt, 100, 0.07)
    else:
        model.load_state_dict(torch.load(model_path))

    _run_index(model, test_data_path, dataset_dir, write_dir, "indices")
    _run_search(model, test_data_path, dataset_dir, write_dir, file_tag, "indices")


class EmbeddingAlignmentGNN(nn.Module):
    def __init__(self, node_types: list, edge_types: list, input_dims: dict, hidden_dim: int):
        super(EmbeddingAlignmentGNN, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. 为不同节点类型定义各自的投影层
        self.projections = nn.ModuleDict({
            node_type: nn.Linear(input_dims[node_type], hidden_dim, bias=False)
            for node_type in node_types
        })

        # 2. 定义第一层异质卷积
        # HeteroConv 会为每种边类型维护一个独立的 SAGEConv
        self.conv1 = HeteroConv({
            edge_type: SAGEConv(hidden_dim, hidden_dim)
            for edge_type in edge_types
        }, aggr='sum')  # 不同边类型的消息通过 sum 聚合

        # 3. 定义第二层异质卷积
        self.conv2 = HeteroConv({
            edge_type: SAGEConv(hidden_dim, hidden_dim)
            for edge_type in edge_types
        }, aggr='sum')

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_dict, edge_index_dict):
        # x_dict: {'image': x_img, 'text': x_txt}
        # edge_index_dict: {('image', 'to', 'text'): edge_index, ...}

        # --- 步骤 1: 投影到统一维度 ---
        out_dict = {}
        for node_type, x in x_dict.items():
            out_dict[node_type] = self.projections[node_type](x)

        # 保存用于残差连接的 identity
        identity_dict = {k: v for k, v in out_dict.items()}

        # --- 步骤 2: 第一层卷积 ---
        out_dict = self.conv1(out_dict, edge_index_dict)
        out_dict = {k: self.act(self.dropout(v)) for k, v in out_dict.items()}

        # --- 步骤 3: 残差连接与第二层卷积 ---
        # 这里的残差是加在投影后的特征上（因为维度一致）
        out_dict = {k: v + identity_dict[k] for k, v in out_dict.items()}

        out_dict = self.conv2(out_dict, edge_index_dict)

        # --- 步骤 4: 归一化 ---
        out_dict = {k: F.normalize(v, dim=1) for k, v in out_dict.items()}

        return out_dict



class GraphDataset(Dataset):

    def __init__(self,
                 data_path: str,
                 questions_images_dir: str,
                 images_dir: str,
                 texts_dir: str,
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.data_path = data_path
        self.questions_images_dir = questions_images_dir
        self.images_dir = images_dir
        self.texts_dir = texts_dir
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.file_names = [f"data_{idx}.pt" for idx in range(len(json.load(f)))]
        logger.debug(f"len file_names: {len(self.file_names)}")
        super().__init__(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.file_names

    def download(self):
        pass

    def process(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, (paper_id, paper) in enumerate(sorted(data.items())):
            target_file_path = os.path.join(self.processed_dir, f"data_{i}.pt")
            if os.path.exists(target_file_path):
                continue
            logger.debug(f"processing paper {i} {paper_id}")
            if not os.path.exists(os.path.join(self.texts_dir, f"data_{i}.pt")):
                logger.warning(f"data_{i}.pt not found")
                continue

            questions_images_data = torch.load(os.path.join(self.questions_images_dir, f"data_{i}.pt"),
                                               weights_only=False, map_location="cpu")
            texts_data = torch.load(os.path.join(self.texts_dir, f"data_{i}.pt"), weights_only=False,
                                    map_location="cpu")
            images_data = torch.load(os.path.join(self.images_dir, f"data_{i}.pt"), weights_only=False,
                                     map_location="cpu")

            images_info = []
            for image_name, image_detail in paper["all_figures"].items():
                images_info.append(ImageInfo(
                    type="image/png",
                    name=image_name,
                    path=os.path.join(self.images_dir, paper_id, image_name),
                    caption=image_detail["caption"]))

            question_image_name_pair = []
            for qa in paper["qa"]:
                question_image_name_pair.append([qa["question"], qa["reference"]])

            reference_images_name = [qa["reference"] for qa in paper["qa"]]
            graph = _make_hetero_graph(paper_id, texts_data["texts"], texts_data["texts_embedding"],
                                images_data["all_images_name"],
                                images_data["all_images_embedding"], reference_images_name,
                                questions_images_data["questions_embedding"],
                                images_info)
            torch.save(graph, target_file_path)

    def len(self):
        return len(self.file_names)

    def get(self, idx):
        if not os.path.exists(os.path.join(self.processed_dir, f"data_{idx}.pt")):
            return torch.load(os.path.join(self.processed_dir, f"data_{idx + 1}.pt"), weights_only=False,
                              map_location="cpu")
        data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"), weights_only=False, map_location="cpu")
        return data



def validate_one_epoch(model, loader, temperature):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for graph_data in loader:
            batch_loss = 0
            graph_data = graph_data.to(model.device)
            x_updated = model(graph_data.x_dict, graph_data.edge_index_dict)
            node_nums = 0
            for i in range(len(graph_data.paper_id)):
                gi = graph_data[i]
                images_indices = gi.reference_images_idx + node_nums
                images_embedding = x_updated["image"][images_indices]
                questions_embedding = model.projections["text"](gi.questions_embedding)
                loss = info_nce_loss(questions_embedding, images_embedding, temperature)
                batch_loss += loss
                node_nums += graph_data[i].x_dict["image"].size(0)
            total_loss += batch_loss.item()
    return total_loss / len(loader)


def _train_and_validate(model: EmbeddingAlignmentGNN, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer,
                        epochs: int, temperature: float = 0.07):
    best_val_loss = float("inf")
    not_improve = 0
    train_losses, val_losses = [], []

    logger.info("Calculating initial loss (Epoch 0)...")
    # 初始状态下，我们假设 train_loss 和 val_loss 差不多，或者只算一次 val 作为基准
    initial_loss = validate_one_epoch(model, val_loader, temperature)
    train_losses.append(initial_loss)  # 用验证集初始值占位
    val_losses.append(initial_loss)
    logger.info(f"Epoch 0/{epochs}, Initial Validation Loss: {initial_loss:.4f}")

    for epoch in range(epochs):

        model.train()
        train_loss = 0
        for batch_idx, graph_data in enumerate(train_loader):
            optimizer.zero_grad()
            start = time.time()
            batch_loss = 0
            graph_data = graph_data.to(model.device)
            x_updated = model(graph_data.x_dict, graph_data.edge_index_dict)
            node_nums = 0
            for i in range(len(graph_data.paper_id)):
                gi = graph_data[i]
                images_indices = gi.reference_images_idx + node_nums
                images_embedding = x_updated["image"][images_indices]
                questions_embedding = model.projections["text"](gi.questions_embedding)
                loss = info_nce_loss(questions_embedding, images_embedding, temperature)
                batch_loss += loss
                node_nums += graph_data[i].x_dict["image"].size(0)
            batch_loss.backward()
            logger.debug(f"batch: {batch_idx + 1}, loss: {batch_loss:.4f} {time.time() - start}")
            train_loss += batch_loss.item()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"epoch {epoch + 1}/{epochs}, loss: {avg_train_loss:.4f}")
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, graph_data in enumerate(val_loader):
                start = time.time()
                batch_loss = 0
                graph_data = graph_data.to(model.device)
                x_updated = model(graph_data.x_dict, graph_data.edge_index_dict)
                node_nums = 0
                for i in range(len(graph_data.paper_id)):
                    gi = graph_data[i]
                    images_indices = gi.reference_images_idx + node_nums
                    images_embedding = x_updated["image"][images_indices]
                    questions_embedding = model.projections["text"](gi.questions_embedding)
                    loss = info_nce_loss(questions_embedding, images_embedding, temperature)
                    batch_loss += loss
                    node_nums += graph_data[i].x_dict["image"].size(0)
                val_loss += batch_loss.item()
                logger.debug(f"batch: {batch_idx + 1}, loss: {batch_loss:.4f} {time.time() - start}")
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logger.info(f"validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "/home/chenyinbo/dataset/graph-rag-output/best_gnn_model.pth")
            logger.info(f"saved best model with Validation Loss: {best_val_loss:.4f}")
            not_improve = 0
        else:
            not_improve += 1
            if not_improve >= 2:
                logger.info(f"train {not_improve} epoch no improve, early stop")
                break

    # 打印最终的 Loss 列表，方便以后手动绘图
    logger.info(f"Final Train Losses: {train_losses}")
    logger.info(f"Final Val Losses: {val_losses}")

    # 绘制更专业的图表
    plt.style.use('seaborn-v0_8-muted')  # 使用一个漂亮的主题
    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(0, len(train_losses))
    ax.plot(x, train_losses, 'b-o', label='Train Loss', markersize=4)
    ax.plot(x, val_losses, 'r-s', label='Val Loss', markersize=4)

    # 找到并标注最小验证损耗点
    min_val_idx = val_losses.index(min(val_losses))
    ax.annotate(f'Best: {val_losses[min_val_idx]:.4f}',
                xy=(x[min_val_idx], val_losses[min_val_idx]),
                xytext=(x[min_val_idx], val_losses[min_val_idx] + 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Learning Curves (Embedding Alignment)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def _run_index(model: EmbeddingAlignmentGNN, test_data_path: str, dataset_dir: str, write_dir: str, indices_dir: str):
    if not os.path.exists(os.path.join(write_dir, indices_dir)):
        os.makedirs(os.path.join(write_dir, indices_dir))
    im = IndexFileManager(write_dir, indices_dir)

    test_loader = DataLoader(dataset=GraphDataset(test_data_path, os.path.join(dataset_dir, "test_a"), os.path.join(dataset_dir, "test_a_images"),
                                 os.path.join(dataset_dir, "test_a_texts"), os.path.join(ROOT_DIR, "test_a_hetero")),
        batch_size=16, shuffle=True, num_workers=4)
    for batch_idx, graph_data in enumerate(test_loader):
        graph_data = graph_data.to(model.device)
        x_updated = model(graph_data.x_dict, graph_data.edge_index_dict)
        text_node_nums, image_node_nums = 0, 0
        for i in range(len(graph_data.paper_id)):
            gi = graph_data[i]
            paper_id = gi.paper_id
            texts = gi.texts
            images_info = gi.all_images_info
            texts_indices = gi.texts_index + text_node_nums
            images_indices = gi.all_images_index + image_node_nums
            texts_embedding = x_updated["text"][texts_indices].cpu().detach().numpy()
            images_embedding = x_updated["image"][images_indices].cpu().detach().numpy()
            text_node_nums += gi.x_dict["text"].size(0)
            image_node_nums += gi.x_dict["image"].size(0)

            id_to_element = dict()
            indices = []
            index = faiss.IndexIDMap(faiss.IndexFlatL2(EMB_DIM))
            curr = 0
            for j, text in enumerate(texts):
                idx = j + curr
                indices.append(idx)
                id_to_element[idx] = {"type": "text", "data": text}
            index.add_with_ids(texts_embedding, np.array(indices))
            curr += len(texts)
            im.write_texts_index(paper_id, index)
            logger.info(f"write {paper_id} texts index finish")

            index = faiss.IndexIDMap(faiss.IndexFlatL2(EMB_DIM))
            indices = []
            for j, image_info in enumerate(images_info):
                idx = j + curr
                indices.append(idx)
                id_to_element[str(idx)] = {"type": "image", "data": image_info.model_dump()}
            index.add_with_ids(images_embedding, np.array(indices))
            im.write_images_index(paper_id, index)
            logger.info(f"write {paper_id} images index finish")

            im.write_id_to_element(paper_id, id_to_element)
            logger.info(f"write {paper_id} id_to_element json finish")


def _run_search(model: EmbeddingAlignmentGNN, test_data_path: str, dataset_dir: str, write_dir: str, file_tag: str, indices_dir):
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"current tag {file_tag}")

    fm = FilesManager(write_dir, file_tag)
    progress = fm.read_curr_progress()
    im = IndexFileManager(write_dir, indices_dir)

    curr = 0
    score = dict()

    for j, (paper_id, paper) in enumerate(sorted(test_data.items())):
        id_to_element = im.read_id_to_element(paper_id)
        images_index = im.read_images_index(paper_id)

        qs_embeddings = torch.load(os.path.join(dataset_dir, "test_a", f"data_{j}.pt"), weights_only=False, map_location="cpu")["questions_embedding"].to(model.device)
        with torch.no_grad():
            qs_embeddings = model.projections["text"](qs_embeddings).cpu().detach().numpy()
        images_distances, images_find_indices = images_index.search(qs_embeddings, 1)


        for i, qa in enumerate(paper["qa"]):
            curr += 1
            if curr < progress.curr_total_count:
                logger.info(f"skip qa {curr}")
                continue

            progress.curr_total_count += 1
            logger.info(f"compute current qa {progress.curr_total_count} ...")

            images_info = [ImageInfo(**id_to_element[str(idx)]["data"]) for idx in images_find_indices[i] if idx >= 0]
            if images_info[0].name == qa["reference"]:
                progress.true_image_count += 1
            logger.info(f"target image {qa['reference']}, predict image {images_info[0].name}")
            logger.info(f"acc {progress.true_image_count / progress.curr_total_count}")

    score["RetAcc"] = round(progress.true_image_count / progress.curr_total_count, 4)
    logger.info(f"score: {score}")
    fm.write_metric(score)


def _make_hetero_graph(paper_id, texts, texts_embedding, all_images_name, all_images_embedding, reference_images_name,
                       questions_embedding,
                       all_images_info) -> HeteroData:
    x_dict = {"image": all_images_embedding, "text": texts_embedding}

    text_to_image = []
    image_to_text = []

    for i, text in enumerate(texts):
        counter = count_figure_table(text)
        for j, image_name in enumerate(all_images_name):
            try:
                _, figure_table_and_num, _ = image_name.split("-")
            except Exception:
                logger.warning(f"image name {image_name} not found")
                continue
            if figure_table_and_num.startswith("Figure"):
                is_figure = True
                num = figure_table_and_num[6:]
            else:
                is_figure = False
                num = figure_table_and_num[5:]
            if is_figure:
                if (("Figure", num) in counter) or ("Fig.", num) in counter:
                    text_to_image.append([i, j])
                    image_to_text.append([j, i])
            else:
                if ("Table", num) in counter:
                    text_to_image.append([i, j])
                    image_to_text.append([j, i])

    # index = faiss.IndexIDMap(faiss.IndexFlatL2(EMB_DIM))
    # indices = [i for i in range(len(texts))]
    # index.add_with_ids(texts_embedding.cpu().detach().numpy(), np.array(indices))
    # texts_distances, texts_find_indices = index.search(all_images_embedding.cpu().detach().numpy(), 5)
    # for i in range(len(texts_find_indices)):
    #     for idx in texts_find_indices[i]:
    #         if idx != -1:
    #             text_to_image.append([idx, len(texts) + idx])
    #             image_to_text.append([len(texts) + idx, idx])

    edge_index_dict = {
        ("text", "reference", "image"): torch.tensor(text_to_image, dtype=torch.long).t(),
        ("image", "be_reference", "text"): torch.tensor(image_to_text, dtype=torch.long).t(),
    }

    data = HeteroData()
    for node_type, x in x_dict.items():
        data[node_type].x = x
    for edge_type, edge_index in edge_index_dict.items():
        data[edge_type].edge_index = edge_index

    data.paper_id = paper_id
    data.all_images_index = torch.tensor([i for i in range(len(all_images_name))], dtype=torch.long)
    data.texts_index = torch.tensor([i for i in range(len(texts))], dtype=torch.long)
    name_to_idx = {name: i for i, name in enumerate(all_images_name)}
    data.reference_images_idx = torch.tensor([name_to_idx[name] for name in reference_images_name], dtype=torch.long)
    data.questions_embedding = questions_embedding
    data.texts = texts
    data.all_images_info = all_images_info

    return data


def count_figure_table(text: str) -> dict:
    pattern = (
        r'\b'
        r'(Fig(?:ure)?s?\.?|Tables?)'  # group 1: Figure / Fig. / Table
        r'(?:\s*([1-9]\d*))?'  # group 2: 编号（可选，只捕获数字）
        r'(?:[a-z]|\([a-z]\))?'  # a / (a)，但不捕获
        r'\b'
    )
    matches = re.findall(pattern, text)
    return dict(Counter(matches))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    run()

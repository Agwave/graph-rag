import os
import random
import json
import time
import re
from dataclasses import dataclass
from typing import Optional, Callable, Literal
from datetime import datetime
from collections import Counter

import faiss
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset, HeteroData
from torch_geometric.nn import SAGEConv, GINConv, GCNConv, LayerNorm, HeteroConv
from loguru import logger
from pydantic import BaseModel, Field

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

SPIQA_DIR = os.getenv("SPIQA_DIR",
                      "/home/chenyinbo/.cache/huggingface/hub/datasets--google--spiqa/snapshots/1774b71511f029b82089a069d75328f25fbf0705")
WRITE_DIR = os.getenv("WRITE_DIR", "/home/chenyinbo/dataset/graph-rag-output")
BERT_MODEL_DIR = os.getenv("BERT_MODEL_DIR", "./models/bert-base-uncased")
API_MODEL = os.getenv("API_MODEL", "qwen-vl-max-2025-08-13")
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", "./models/clip-vit-base-patch32")
ROOT_DIR = os.getenv("ROOT_DIR", "/home/chenyinbo/dataset/graph-rag-output/graph_dataset/root")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "qwen2.5-vl-embedding")
EMB_DIM = int(os.getenv("EMB_DIM", "1024"))
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-c450e178a972467d93b282e218c1dfba")


def run():
    cfg = ExpConfig(mode="fast")
    set_seed(cfg.seed)

    train_data_path = os.path.join(SPIQA_DIR, "train_val/SPIQA_train.json")
    curr_time = datetime.now().strftime("%Y%m%d%H%M")

    dataset_dir = "/home/chenyinbo/dataset/graph-rag-output/graph_dataset"
    model_path = "/home/chenyinbo/dataset/graph-rag-output/best_gnn_model_fast.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingAlignmentGNN(['image', 'text'],
                                  [
                                      ('image', 'be_reference', 'text'), ('text', 'reference', 'image'),
                                      # ('text', 'before', 'text'), ('text', 'after', 'image'),
                                      ('text', 'semantic', 'image'), ('image', 'be_semantic', 'text'),
                                  ],
                                  {'image': EMB_DIM, 'text': EMB_DIM}, EMB_DIM).to(device)
    model_q = QuestionEncoder(EMB_DIM).to(device)
    if not os.path.exists(model_path):
        train_loader = DataLoader(
            dataset=build_dataset_subset(GraphDataset(train_data_path, os.path.join(dataset_dir, "train"),
                                                      os.path.join(dataset_dir, "train_images"),
                                                      os.path.join(dataset_dir, "train_texts"),
                                                      os.path.join(ROOT_DIR, "train_hetero_reference_semantic")), cfg),
            batch_size=cfg.batch_size, shuffle=True, num_workers=4)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-3, amsgrad=True)
        train(model, model_q, train_loader, opt, get_num_epochs(cfg), 0.07)
    else:
        model.load_state_dict(torch.load(model_path))


@dataclass
class ExpConfig:
    mode: Literal["fast", "mini", "full"] = "fast"

    # ---- 数据 ----
    fast_num_samples: int = 2000  # fast-dev 用多少样本
    mini_num_samples: int = 10000  # mini-dev

    # ---- 训练 ----
    fast_epochs: int = 15
    mini_epochs: int = 15
    full_epochs: int = 100

    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 42

    # ---- debug ----
    log_grad: bool = True
    clip_grad: float = 1.0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset_subset(dataset, cfg: ExpConfig):
    if cfg.mode == "fast":
        return Subset(dataset, range(min(len(dataset), cfg.fast_num_samples)))
    elif cfg.mode == "mini":
        return Subset(dataset, range(min(len(dataset), cfg.mini_num_samples)))
    else:
        return dataset


def get_num_epochs(cfg: ExpConfig):
    if cfg.mode == "fast":
        return cfg.fast_epochs
    elif cfg.mode == "mini":
        return cfg.mini_epochs
    else:
        return cfg.full_epochs


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


class QuestionEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, q):
        return F.normalize(self.net(q), dim=1)


class ImageInfo(BaseModel):
    type: str = Field("image/png", description="image type")
    name: str = Field("", description="name of the image")
    path: str = Field("", description="image path")
    caption: str = Field("", description="image caption")


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


def _make_hetero_graph(paper_id, texts, texts_embedding, all_images_name, all_images_embedding, reference_images_name,
                       questions_embedding,
                       all_images_info) -> HeteroData:
    x_dict = {"image": all_images_embedding, "text": texts_embedding}

    text_before_text = []
    text_after_text = []
    for i in range(len(texts)-1):
        text_before_text.append([i, i+1])
        text_after_text.append([i+1, i])

    text_reference_image = []
    image_be_reference_text = []

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
                    text_reference_image.append([i, j])
                    image_be_reference_text.append([j, i])
            else:
                if ("Table", num) in counter:
                    text_reference_image.append([i, j])
                    image_be_reference_text.append([j, i])

    text_semantic_image = []
    image_be_semantic_text = []
    index = faiss.IndexIDMap(faiss.IndexFlatL2(EMB_DIM))
    indices = [i for i in range(len(texts))]
    index.add_with_ids(texts_embedding.cpu().detach().numpy(), np.array(indices))
    texts_distances, texts_find_indices = index.search(all_images_embedding.cpu().detach().numpy(), 5)
    for i in range(len(texts_find_indices)):
        for idx in texts_find_indices[i]:
            if idx != -1:
                text_semantic_image.append([idx, i])
                image_be_semantic_text.append([i, idx])

    edge_index_dict = {
        ("text", "reference", "image"): torch.tensor(text_reference_image, dtype=torch.long).t(),
        ("image", "be_reference", "text"): torch.tensor(image_be_reference_text, dtype=torch.long).t(),
        # ("text", "before", "text"): torch.tensor(text_before_text, dtype=torch.long).t(),
        # ("text", "after", "text"): torch.tensor(text_after_text, dtype=torch.long).t(),
        ("text", "semantic", "image"): torch.tensor(text_semantic_image, dtype=torch.long).t(),
        ("image", "be_semantic", "text"): torch.tensor(image_be_semantic_text, dtype=torch.long).t(),
    }

    data = HeteroData()
    for node_type, x in x_dict.items():
        data[node_type].x = x
    for edge_type, edge_index in edge_index_dict.items():
        data[edge_type].edge_index = edge_index

    data.paper_id = paper_id
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


def train(model: EmbeddingAlignmentGNN, model_q: QuestionEncoder, train_loader: DataLoader, optimizer: Optimizer,
          epochs: int, temperature: float = 0.07):
    best_val_loss = float("inf")
    not_improve = 0
    losses = []
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
        losses.append(avg_train_loss)
        logger.info(f"epoch {epoch + 1}/{epochs}, loss: {avg_train_loss:.4f}")

    logger.info(f"losses: {losses}")
    plt.plot(losses, label="baseline")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curve")
    plt.legend()
    plt.show()
    logger.info("training finished")


def info_nce_loss(questions_embedding: torch.Tensor, images_embedding: torch.Tensor, temperature: float = 0.07):
    similarity_matrix = torch.matmul(questions_embedding, images_embedding.T) / temperature
    labels = torch.arange(len(questions_embedding)).to(questions_embedding.device)
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


if __name__ == '__main__':
    run()

import json
import os
import time
from typing import Optional, Callable

import faiss
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.nn.conv import GCNConv
from torch_geometric.loader import DataLoader

from core.conf import ROOT_DIR, EMB_DIM
from core.output import IndexFileManager, FilesManager
from core.prompt import ImageInfo
from core.train import info_nce_loss

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from loguru import logger
from torch.optim.optimizer import Optimizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class EmbeddingAlignmentGNN(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super(EmbeddingAlignmentGNN, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        with torch.no_grad():
            self.projection.weight.data = torch.eye(output_dim, input_dim) * 0.1
        self.conv = GCNConv(output_dim, output_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x, edge_index):
        x = self.projection(x)
        x = self.conv(x, edge_index)
        x = functional.normalize(x, dim=1)
        return x


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
            # self.file_names = [f"data_0.pt", f"data_1.pt"]
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
            logger.debug(f"processing paper {i} {paper_id}")

            target_file_path = os.path.join(self.processed_dir, f"data_{i}.pt")
            if os.path.exists(target_file_path):
                continue
            if not os.path.exists(os.path.join(self.texts_dir, f"data_{i}.pt")):
                logger.warning(f"data_{i}.pt not found")
                continue

            questions_images_data = torch.load(os.path.join(self.questions_images_dir, f"data_{i}.pt"), weights_only=False, map_location="cpu")
            texts_data = torch.load(os.path.join(self.texts_dir, f"data_{i}.pt"), weights_only=False, map_location="cpu")
            images_data = torch.load(os.path.join(self.images_dir, f"data_{i}.pt"), weights_only=False, map_location="cpu")

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
            graph = _make_graph(paper_id, texts_data["texts"], texts_data["texts_embedding"], images_data["all_images_name"],
                                images_data["all_images_embedding"], reference_images_name, questions_images_data["questions_embedding"],
                                images_info)
            torch.save(graph, target_file_path)

    def len(self):
        return len(self.file_names)

    def get(self, idx):
        if not os.path.exists(os.path.join(self.processed_dir, f"data_{idx}.pt")):
            return torch.load(os.path.join(self.processed_dir, f"data_{idx+1}.pt"), weights_only=False, map_location="cpu")
        data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"), weights_only=False, map_location="cpu")
        return data


def run(train_data_path: str, val_data_path: str, test_data_path: str, write_dir: str, file_tag: str):
    model_path = "output/best_gnn_model.pth"
    dataset_dir = "output/graph_dataset"
    model = EmbeddingAlignmentGNN(EMB_DIM, EMB_DIM).to("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        train_loader = DataLoader(
            dataset=GraphDataset(train_data_path, os.path.join(dataset_dir, "train"), os.path.join(dataset_dir, "train_images"),
                                 os.path.join(dataset_dir, "train_texts"), os.path.join(ROOT_DIR, "train")),
            batch_size=128, shuffle=True, num_workers=4)
        val_loader = DataLoader(
            dataset=GraphDataset(val_data_path, os.path.join(dataset_dir, "val"), os.path.join(dataset_dir, "val_images"),
                                 os.path.join(dataset_dir, "val_texts"), os.path.join(ROOT_DIR, "val")),
            batch_size=128, shuffle=False, num_workers=4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        _train_and_validate(model, train_loader, val_loader, opt, 100, 0.07)
    else:
        model.load_state_dict(torch.load(model_path))

    _run_index(model, test_data_path, dataset_dir, write_dir, "indices")
    _run_search(model, test_data_path, dataset_dir, write_dir, file_tag, "indices")


def _train_and_validate(model: EmbeddingAlignmentGNN, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer,
                        epochs: int, temperature: float = 0.07):
    best_val_loss = float("inf")
    not_improve = 0
    for epoch in range(epochs):

        model.train()
        train_loss = 0
        for batch_idx, graph_data in enumerate(train_loader):
            optimizer.zero_grad()
            start = time.time()
            batch_loss = 0
            graph_data: Data = graph_data.to(model.device)
            x_updated = model(graph_data.x, graph_data.edge_index)
            node_nums = 0
            for i in range(len(graph_data.paper_id)):
                gi = graph_data[i]
                images_indices = gi.reference_images_index + node_nums
                images_embedding = x_updated[images_indices]
                questions_embedding = model.projection(gi.questions_embedding)
                loss = info_nce_loss(questions_embedding, images_embedding, temperature)
                batch_loss += loss
                node_nums += graph_data[i].x.size(0)
            batch_loss.backward()
            logger.debug(f"batch: {batch_idx + 1}, loss: {batch_loss:.4f} {time.time() - start}")
            train_loss += batch_loss.item()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"epoch {epoch + 1}/{epochs}, loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, graph_data in enumerate(val_loader):
                start = time.time()
                batch_loss = 0
                graph_data: Data = graph_data.to(model.device)
                x_updated = model(graph_data.x, graph_data.edge_index)
                node_nums = 0
                for i in range(len(graph_data.paper_id)):
                    gi = graph_data[i]
                    images_indices = gi.reference_images_index + node_nums
                    images_embedding = x_updated[images_indices]
                    questions_embedding = model.projection(gi.questions_embedding)
                    loss = info_nce_loss(questions_embedding, images_embedding, temperature)
                    batch_loss += loss
                    node_nums += graph_data[i].x.size(0)
                val_loss += batch_loss.item()
                logger.debug(f"batch: {batch_idx + 1}, loss: {batch_loss:.4f} {time.time() - start}")
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "output/best_gnn_model.pth")
            logger.info(f"saved best model with Validation Loss: {best_val_loss:.4f}")
            not_improve = 0
        else:
            not_improve += 1
            if not_improve >= 10:
                logger.info("train 10 epoch no improve, early stop")
                break

    logger.info("training finished")


def _run_index(model: EmbeddingAlignmentGNN, test_data_path: str, dataset_dir: str, write_dir: str, indices_dir: str):
    if not os.path.exists(os.path.join(write_dir, indices_dir)):
        os.makedirs(os.path.join(write_dir, indices_dir))
    im = IndexFileManager(write_dir, indices_dir)

    test_loader = DataLoader(dataset=GraphDataset(test_data_path, os.path.join(dataset_dir, "test_a"), os.path.join(dataset_dir, "test_a_images"),
                                 os.path.join(dataset_dir, "test_a_texts"), os.path.join(ROOT_DIR, "test_a")),
        batch_size=16, shuffle=True, num_workers=4)
    for batch_idx, graph_data in enumerate(test_loader):
        graph_data: Data = graph_data.to(model.device)
        x_updated = model(graph_data.x, graph_data.edge_index)
        node_nums = 0
        for i in range(len(graph_data.paper_id)):
            gi = graph_data[i]
            paper_id = gi.paper_id
            texts = gi.texts
            images_info = gi.all_images_info
            texts_indices = gi.texts_index + node_nums
            images_indices = gi.all_images_index + node_nums
            texts_embedding = x_updated[texts_indices].cpu().detach().numpy()
            images_embedding = x_updated[images_indices].cpu().detach().numpy()
            node_nums += gi.x.size(0)

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
            qs_embeddings = model.projection(qs_embeddings).cpu().detach().numpy()
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


def _make_graph(paper_id, texts, texts_embedding, all_images_name, all_images_embedding, reference_images_name, questions_embedding,
                all_images_info) -> Data:
    x = torch.concat([texts_embedding, all_images_embedding], 0)

    edges = []
    for i in range(len(texts) - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    index = faiss.IndexIDMap(faiss.IndexFlatL2(EMB_DIM))
    indices = [i for i in range(len(texts))]
    index.add_with_ids(texts_embedding.cpu().detach().numpy(), np.array(indices))
    texts_distances, texts_find_indices = index.search(all_images_embedding.cpu().detach().numpy(), 5)
    for i in range(len(texts_find_indices)):
        for idx in texts_find_indices[i]:
            if idx != -1:
                edges.append([idx, len(texts) + i])
                edges.append([len(texts) + i, idx])
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    all_images_index = torch.tensor([len(texts) + i for i in range(len(all_images_name))], dtype=torch.long)
    texts_index = torch.tensor([i for i in range(len(texts))], dtype=torch.long)
    name_to_index = {name: i for i, name in enumerate(all_images_name)}
    reference_images_index = torch.tensor([name_to_index[name] + len(texts) for name in reference_images_name], dtype=torch.long)

    return Data(x, edge_index, paper_id=paper_id, all_images_index=all_images_index, texts_index=texts_index,
                reference_images_index=reference_images_index, questions_embedding=questions_embedding,
                texts=texts, all_images_info=all_images_info)

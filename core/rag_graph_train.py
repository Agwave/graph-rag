import json
import os
import time
from typing import Optional, Callable

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from loguru import logger
from openai import Client
from PIL import Image
from torch.optim.optimizer import Optimizer
from torch_geometric.data import Data, Dataset
from torch_geometric.nn.conv import GCNConv
from torch_geometric.loader import DataLoader
from transformers import CLIPModel, CLIPProcessor

from core.clip import init_model_and_processor, embedding_texts, embedding_images, trunk_by_paragraph
from core.conf import CLIP_MODEL_PATH, ROOT_DIR
from core.data import read_text_file
from core.llm import invoke_llm
from core.metric import create_coco_eval_file, score_compute
from core.output import IndexFileManager, FilesManager
from core.prompt import ImageInfo
from core.train import info_nce_loss


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
                 images_dir: str,
                 paragraphs_dir: str,
                 clip_model: CLIPModel,
                 clip_processor: CLIPProcessor,
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.data_path = data_path
        self.images_dir = images_dir
        self.paragraphs_dir = paragraphs_dir
        self.clip_model = clip_model
        self.clip_processor = clip_processor
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
        idx = 0
        for paper_id, paper in sorted(data.items()):
            logger.debug(f"processing paper {paper_id}")

            target_file_path = os.path.join(self.processed_dir, f"data_{idx}.pt")
            if os.path.exists(target_file_path):
                idx += 1
                continue

            paragraphs_file_path = os.path.join(self.paragraphs_dir, f"{paper_id}.txt")
            text = read_text_file(paragraphs_file_path)
            texts = trunk_by_paragraph(text)

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

            graph = _make_graph(self.clip_model, self.clip_processor, texts, images_info, question_image_name_pair, paper_id)
            torch.save(graph, target_file_path)
            idx += 1

    def len(self):
        return len(self.file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"), weights_only=False, map_location="cpu")
        return data


def run(client: Client, train_data_path: str, val_data_path: str, train_val_images_dir: str,
        test_data_path: str, test_images_dir: str, paragraphs_dir: str, write_dir: str, file_tag: str):
    model_path = "output/best_gnn_model.pth"
    model = EmbeddingAlignmentGNN(512, 512).to("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        clip_model, clip_processor = init_model_and_processor(CLIP_MODEL_PATH)
        train_loader = DataLoader(
            dataset=GraphDataset(train_data_path, train_val_images_dir, paragraphs_dir, clip_model, clip_processor,
                                 os.path.join(ROOT_DIR, "train")), batch_size=16, shuffle=True, num_workers=4)
        val_loader = DataLoader(
            dataset=GraphDataset(val_data_path, train_val_images_dir, paragraphs_dir, clip_model, clip_processor,
                                 os.path.join(ROOT_DIR, "val")), batch_size=16, shuffle=False, num_workers=4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        _train_and_validate(model, train_loader, val_loader, opt, 100, 0.07)
    else:
        model.load_state_dict(torch.load(model_path))

    _run_index(model, test_data_path, paragraphs_dir, test_images_dir, write_dir, "indices")
    _run_search(model, client, test_data_path, write_dir, file_tag, "indices")


def _train_and_validate(model: EmbeddingAlignmentGNN, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer,
                        epochs: int, temperature: float = 0.07):
    best_val_loss = float("inf")
    not_improve = 0
    for epoch in range(epochs):

        model.train()
        train_loss = 0
        for batch_idx, graph_data in enumerate(train_loader):
            start = time.time()
            graph_data: Data = graph_data.to(model.device)
            optimizer.zero_grad()
            x_updated = model(graph_data.x, graph_data.edge_index)
            images_embedding = x_updated[graph_data.images_index]
            questions_embedding = model.projection(graph_data.questions_embedding)
            loss = info_nce_loss(questions_embedding, images_embedding, temperature)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            logger.debug(f"batch: {batch_idx + 1}, loss: {loss.item():.4f} {time.time() - start}")
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"epoch {epoch + 1}/{epochs}, loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, graph_data in enumerate(val_loader):
                graph_data: Data = graph_data.to(model.device)
                x_updated = model(graph_data.x, graph_data.edge_index)
                images_embedding = x_updated[graph_data.images_index]
                questions_embedding = model.projection(graph_data.questions_embedding)
                loss = info_nce_loss(questions_embedding, images_embedding, temperature)
                val_loss += loss.item()
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


def _run_index(model: EmbeddingAlignmentGNN, test_data_path: str, paragraphs_dir: str, images_dir: str, write_dir: str,
               indices_dir: str):
    if not os.path.exists(os.path.join(write_dir, indices_dir)):
        os.makedirs(os.path.join(write_dir, indices_dir))
    im = IndexFileManager(write_dir, indices_dir)
    clip_model, clip_processor = init_model_and_processor(CLIP_MODEL_PATH)
    test_loader = DataLoader(dataset=GraphDataset(
        test_data_path, images_dir, paragraphs_dir, clip_model, clip_processor, os.path.join(ROOT_DIR, "test_a")),
        batch_size=1, shuffle=True, num_workers=0)
    for batch_idx, graph_data in enumerate(test_loader):
        graph_data: Data = graph_data.to(model.device)
        x_updated = model(graph_data.x, graph_data.edge_index)
        logger.info(f"texts len {len(graph_data.texts)}")
        for i in range(len(graph_data.paper_id)):
            paper_id = graph_data.paper_id[i]
            texts = graph_data.texts[i]
            images_info = graph_data.images_info[i]
            batch_mask = (graph_data.batch == i)
            texts_indices = graph_data.texts_index[batch_mask[graph_data.texts_index]]
            images_indices = graph_data.images_index[batch_mask[graph_data.images_index]]
            texts_embedding = x_updated[texts_indices].cpu().detach().numpy()
            images_embedding = x_updated[images_indices].cpu().detach().numpy()

            id_to_element = dict()
            indices = []
            index = faiss.IndexIDMap(faiss.IndexFlatL2(512))
            curr = 0
            for j, text in enumerate(texts):
                idx = j + curr
                indices.append(idx)
                id_to_element[idx] = {"type": "text", "data": text}
            index.add_with_ids(texts_embedding, np.array(indices))
            curr += len(texts)
            im.write_texts_index(paper_id, index)
            logger.info(f"write {paper_id} texts index finish")

            index = faiss.IndexIDMap(faiss.IndexFlatL2(512))
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


def _run_search(model: EmbeddingAlignmentGNN, client: Client, test_data_path: str, write_dir: str, file_tag: str,
                indices_dir):
    clip_model, clip_processor = init_model_and_processor(CLIP_MODEL_PATH)
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"current tag {file_tag}")

    fm = FilesManager(write_dir, file_tag)
    progress = fm.read_curr_progress()
    im = IndexFileManager(write_dir, indices_dir)
    for paper_id, paper in sorted(test_data.items()):
        id_to_element = im.read_id_to_element(paper_id)

        texts_index = im.read_texts_index(paper_id)
        images_index = im.read_images_index(paper_id)
        qs = [qa["question"] for qa in paper["qa"]]
        qs_embeddings = embedding_texts(clip_model, clip_processor, qs)
        qs_embeddings = model.projection(qs_embeddings).cpu().detach().numpy()
        texts_distances, texts_find_indices = texts_index.search(qs_embeddings, 5)
        images_distances, images_find_indices = images_index.search(qs_embeddings, 5)

        for i, qa in enumerate(paper["qa"]):
            if i < progress.curr_total_count:
                logger.info(f"skip qa {i+1}")
                continue

            progress.curr_total_count += 1
            logger.info(f"compute current qa {progress.curr_total_count} ...")

            images_info = [ImageInfo(**id_to_element[str(idx)]["data"]) for idx in images_find_indices[i]]
            texts = [id_to_element[str(idx)]["data"] for idx in texts_find_indices[i]]
            paragraphs = "\n\n---\n\n".join(texts)
            try:
                answer = invoke_llm(client, qa["question"], paragraphs, images_info)
            except Exception as e:
                logger.warning(f"invoke_llm failed: {e}")
                progress.except_count += 1
                fm.write_curr_progress(progress)
                continue

            logger.info(f"images_info {images_info}")
            logger.info(f"question: {qa['question']}")
            logger.info(f"gt_answer: {qa['answer']}")
            logger.info(f"pred_answer: {answer}")

            d = {
                "id": f"{paper_id}_{i}",
                "question": qa["question"],
                "pred_answer": answer,
                "gt_answer": qa["answer"],
            }

            fm.write_gene_line(d)
            fm.write_curr_progress(progress)

    pred_answers, gt_answers = [], []
    for line in fm.read_gene_file():
        data = json.loads(line.strip())
        pred_answers.append(data["pred_answer"])
        gt_answers.append(data["gt_answer"])

    create_coco_eval_file(fm.pred_file_path, fm.gnth_file_path, pred_answers, gt_answers)
    score = score_compute(fm.pred_file_path, fm.gnth_file_path, metrics=["METEOR", "ROUGE_L", "CIDEr", "BERTScore"])
    score["RetAcc"] = round(progress.find_true_image_count / progress.curr_qa_count, 4)
    logger.info(f"score: {score}")
    fm.write_metric(score)


def _make_graph(clip_model: CLIPModel, clip_processor: CLIPProcessor, texts: list[str],
                images_info: list[ImageInfo], question_image_name_pair: list[list[str]], paper_id: str) -> Data:
    images = []
    for image_info in images_info:
        images.append(Image.open(image_info.path).convert("RGB"))
    texts_feature = embedding_texts(clip_model, clip_processor, texts)
    images_feature = embedding_images(clip_model, clip_processor, images)
    x = torch.concat([texts_feature, images_feature], 0)
    x.to("cpu")

    edges = []
    for i in range(len(texts) - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    index = faiss.IndexIDMap(faiss.IndexFlatL2(512))
    indices = [i for i in range(len(texts))]
    index.add_with_ids(texts_feature.cpu().detach().numpy(), np.array(indices))
    texts_distances, texts_find_indices = index.search(images_feature.cpu().detach().numpy(), 5)
    for i in range(len(texts_find_indices)):
        for idx in texts_find_indices[i]:
            if idx != -1:
                edges.append([idx, len(texts) + i])
                edges.append([len(texts) + i, idx])
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    questions_embedding = embedding_texts(clip_model, clip_processor, [p[0] for p in question_image_name_pair])
    images_index = torch.tensor([len(texts) + i for i in range(len(images))], dtype=torch.long)
    texts_index = torch.tensor([i for i in range(len(texts))], dtype=torch.long)

    return Data(x, edge_index, images_index=images_index, texts_index=texts_index,
                questions_embedding=questions_embedding, images_info=images_info, texts=texts, paper_id=paper_id)

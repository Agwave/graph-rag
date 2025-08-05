import json
import os
from typing import Optional, Callable

import faiss
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from openai import Client
from PIL import Image
from torch.optim.optimizer import Optimizer
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from transformers import CLIPModel, CLIPProcessor

from core.clip import init_model_and_processor, embedding_texts, embedding_image, trunk_by_paragraph
from core.conf import CLIP_MODEL_PATH
from core.data import read_text_file
from core.llm import invoke_llm
from core.metric import create_coco_eval_file, score_compute
from core.output import IndexFileManager, FilesManager
from core.prompt import ImageInfo
from core.train import info_nce_loss


class EmbeddingAlignmentGNN(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super(EmbeddingAlignmentGNN, self).__init__()
        self.image_branch = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
        self.text_branch = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
        self.clip_model, self.clip_processor = init_model_and_processor(CLIP_MODEL_PATH)

    def get_texts_embedding(self, texts: list[str]):
        texts_embeddings = embedding_texts(self.clip_model, self.clip_processor, texts)
        aligned_texts_embedding = self.text_branch(texts_embeddings)
        return aligned_texts_embedding

    def get_images_embedding(self, images: list[Image.Image]):
        images_embedding = embedding_image(self.clip_model, self.clip_processor, images)
        aligned_images_embedding = self.image_branch(images_embedding)
        return aligned_images_embedding

    def forward(self, data: Data):
        ## 这里需要做的是过了gnn之后，使用图片索引获取到向量，然后与问题向量进行对比学习和损失计算
        pass


class GraphData(Data):

    def __init__(self,
                 x: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 image_index: Optional[torch.Tensor] = None,
                 questions_embedding: Optional[torch.Tensor] = None,
                 **kwargs
                 ):
        super().__init__(x=x, edge_index=edge_index, image_index=image_index, questions_embedding=questions_embedding, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index" or key == "image_index":
            return self.x.size(0)
        return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return 1
        else:
            return 0


class GraphDataset(Dataset):

    def __init__(self,
                 data_path: str,
                 images_dir: str,
                 paragraphs_dir: str,
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.data_path = data_path
        self.images_dir = images_dir
        self.paragraphs_dir = paragraphs_dir

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [f"data_{idx}.pt" for idx in range(len(data))]

    def download(self):
        pass

    def process(self):
        clip_model, clip_processor = init_model_and_processor(CLIP_MODEL_PATH)
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        idx = 0
        for paper_id, paper in data.items():
            paragraphs_file_path = os.path.join(self.paragraphs_dir, f"{paper_id}.txt")
            text = read_text_file(paragraphs_file_path)
            texts = trunk_by_paragraph(text)
            logger.info(f"paragraphs: {text[:100]}...")

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

            graph = _make_graph(clip_model, clip_processor, texts, images_info, question_image_name_pair)
            torch.save(graph, os.path.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
        return data


def run(client: Client, train_data_path: str, val_data_path: str, train_val_images_dir,
        test_data_path: str, test_images_dir: str, paragraphs_dir: str, write_dir: str, file_tag: str):
    model_path = "output/best_gnn_model.pth"
    model = EmbeddingAlignmentGNN(512, 512).to("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        train_loader = DataLoader(dataset=GraphDataset(train_data_path, train_val_images_dir, paragraphs_dir), batch_size=2,
                                  shuffle=True, num_workers=1)
        val_loader = DataLoader(dataset=GraphDataset(val_data_path, train_val_images_dir, paragraphs_dir), batch_size=2,
                                shuffle=False, num_workers=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        _train_and_validate(model, train_loader, val_loader, opt, 20, 0.07)
    else:
        model.load_state_dict(torch.load(model_path))

    _run_index(model, test_data_path, paragraphs_dir, test_images_dir, write_dir, "indices")
    _run_search(model, client, test_data_path, write_dir, file_tag, "indices")


def _train_and_validate(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer,
                        epochs: int, temperature: float = 0.07):
    best_val_loss = float("inf")
    not_improve = 0
    for epoch in range(epochs):

        model.train()
        train_loss = 0
        for batch_idx, graph_data in enumerate(train_loader):
            graph_data: GraphData = graph_data
            optimizer.zero_grad()
            x_updated = model(graph_data)
            images_embedding = x_updated[graph_data.image_index]
            loss = info_nce_loss(graph_data.questions_embedding, images_embedding, temperature)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            break  # TODO delete
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"epoch {epoch + 1}/{epochs}, loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, graph_data in enumerate(val_loader):
                graph_data: GraphData = graph_data
                x_updated = model(graph_data)
                images_embedding = x_updated[graph_data.image_index]
                loss = info_nce_loss(graph_data.questions_embedding, images_embedding, temperature)
                val_loss += loss.item()
                break  # TODO delete
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

        break  # TODO delete

    logger.info("training finished")



def _run_index(model: EmbeddingAlignmentGNN, test_data_path: str, paragraphs_dir: str, images_dir: str, write_dir: str,
               indices_dir: str):
    test_loader = DataLoader(dataset=GraphDataset(test_data_path, images_dir, paragraphs_dir), batch_size=2, shuffle=False,
                             num_workers=1)
    for batch_idx, graph_data in enumerate(test_loader):
        pass

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    if not os.path.exists(os.path.join(write_dir, indices_dir)):
        os.makedirs(os.path.join(write_dir, indices_dir))

    im = IndexFileManager(write_dir, indices_dir)
    for paper_id, paper in test_data.items():
        id_to_element = dict()
        curr = 0

        index = faiss.IndexIDMap(faiss.IndexFlatL2(512))
        paragraphs_file_path = os.path.join(paragraphs_dir, f"{paper_id}.txt")
        text = read_text_file(paragraphs_file_path)
        texts = trunk_by_paragraph(text)
        logger.info(f"paragraphs: {text[:100]}...")

        text_embeddings = model.get_texts_embedding(texts).cpu().detach().numpy()
        indices = []
        for i, text in enumerate(texts):
            idx = i + curr
            indices.append(idx)
            id_to_element[idx] = {"type": "text", "data": text}
        index.add_with_ids(text_embeddings, np.array(indices))
        curr += len(texts)
        im.write_texts_index(paper_id, index)
        logger.info(f"write {paper_id} texts index finish")

        index = faiss.IndexIDMap(faiss.IndexFlatL2(512))
        images = []
        images_info = []
        for image_name, image_detail in paper["all_figures"].items():
            img = Image.open(os.path.join(images_dir, paper_id, image_name)).convert("RGB")
            images.append(img)
            images_info.append(ImageInfo(
                type="image/png",
                name=image_name,
                path=os.path.join(images_dir, paper_id, image_name),
                caption=image_detail["caption"]).model_dump()
                               )

        image_embeddings = model.get_images_embedding(images).cpu().detach().numpy()
        indices = []
        for i, image_info in enumerate(images_info):
            idx = i + curr
            indices.append(idx)
            id_to_element[str(idx)] = {"type": "image", "data": image_info}
        index.add_with_ids(image_embeddings, np.array(indices))
        im.write_images_index(paper_id, index)
        logger.info(f"write {paper_id} images index finish")

        im.write_id_to_element(paper_id, id_to_element)
        logger.info(f"write {paper_id} id_to_element json finish")
        break  # TODO delete


def _run_search(model: EmbeddingAlignmentGNN, client: Client, test_data_path: str, write_dir: str, file_tag: str,
                indices_dir):
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"current tag {file_tag}")

    fm = FilesManager(write_dir, file_tag)
    skip_qa_count = fm.read_skip_count()

    curr_qa_count = 0
    find_true_image_count = 0
    im = IndexFileManager(write_dir, indices_dir)
    for paper_id, paper in test_data.items():
        with open(os.path.join(write_dir, indices_dir, f"{paper_id}.json"), "r", encoding="utf-8") as f:
            id_to_element = json.load(f)

        texts_index = im.read_texts_index(paper_id)
        images_index = im.read_images_index(paper_id)
        qs = [qa["question"] for qa in paper["qa"]]
        qs_embeddings = model.get_texts_embedding(qs).cpu().detach().numpy()
        texts_distances, texts_find_indices = texts_index.search(qs_embeddings, 3)
        images_distances, images_find_indices = images_index.search(qs_embeddings, 1)

        for i, qa in enumerate(paper["qa"]):
            curr_qa_count += 1
            if skip_qa_count > 0:
                skip_qa_count -= 1
                logger.info(f"skip qa {curr_qa_count}")
                continue

            logger.info(f"compute current qa {curr_qa_count} ...")

            images_info = [ImageInfo(**id_to_element[str(idx)]["data"]) for idx in images_find_indices[i]]
            if images_info[0].name == qa["reference"]:
                logger.info(f"success to find target image | find {images_info[0].name} | target {qa['reference']}")
                find_true_image_count += 1
            else:
                logger.info(f"fail to find target image | find {images_info[0].name} | target {qa['reference']}")
            logger.info(f"total num {curr_qa_count} | success find num {find_true_image_count}")

            texts = [id_to_element[str(idx)]["data"] for idx in texts_find_indices[i]]
            paragraphs = "\n\n---\n\n".join(texts)
            try:
                answer = invoke_llm(client, qa["question"], paragraphs, images_info)
            except Exception as e:
                logger.warning(f"invoke_llm failed: {e}")
                fm.write_skip_count(curr_qa_count)
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
            fm.write_skip_count(curr_qa_count)

        break  # TODO delete

    pred_answers, gt_answers = [], []
    for line in fm.read_gene_file():
        data = json.loads(line.strip())
        pred_answers.append(data["pred_answer"])
        gt_answers.append(data["gt_answer"])

    create_coco_eval_file(fm.pred_file_path, fm.gnth_file_path, pred_answers, gt_answers)
    score = score_compute(fm.pred_file_path, fm.gnth_file_path, metrics=["METEOR", "ROUGE_L", "CIDEr", "BERTScore"])
    score["RetAcc"] = round(find_true_image_count / curr_qa_count, 4)
    logger.info(f"score: {score}")
    fm.write_metric(score)


def _make_graph(clip_model: CLIPModel, clip_processor: CLIPProcessor, texts: list[str],
                images_info: list[ImageInfo], question_image_name_pair: list[list[str]]) -> Data:
    images = []
    for image_info in images_info:
        images.append(Image.open(image_info.path).convert("RGB"))
    texts_feature = embedding_texts(clip_model, clip_processor, texts)
    images_feature = embedding_image(clip_model, clip_processor, images)
    x = torch.concat([texts_feature, images_feature], 0)

    edges = []
    for i in range(len(texts) - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    for j, image_info in enumerate(images_info):
        caption_title = image_info.name.split("-")[1]
        alpha, numeric = _split_alpha_numeric_loop(caption_title)
        possible_caption_title = [alpha + " " + numeric, alpha[:3] + ". " + numeric]
        for i, text in enumerate(texts):
            has_caption_title = False
            for title in possible_caption_title:
                if title in text:
                    has_caption_title = True
                    break
            if has_caption_title:
                edges.append([i, j + len(texts)])
                edges.append([j + len(texts), i])
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    questions_embedding = embedding_texts(clip_model, clip_processor, [p[0] for p in question_image_name_pair])
    image_index = torch.tensor([p[1] for p in question_image_name_pair], dtype=torch.long)

    return Data(x, edge_index, image_index=image_index, questions_embedding=questions_embedding)


def _split_alpha_numeric_loop(text):
    for i, char in enumerate(text):
        if char.isdigit():
            return text[:i], text[i:]

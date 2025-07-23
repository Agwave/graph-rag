import json
import os

import faiss
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from openai import Client
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer

from core.clip import init_model_and_processor, embedding_texts, embedding_image, trunk_by_paragraph
from core.data import read_text_file
from core.llm import invoke_llm
from core.metric import create_coco_eval_file, score_compute
from core.output import IndexFileManager, FilesManager
from core.prompt import ImageInfo


def run(client: Client, train_data_path: str, val_data_path: str, train_val_image_dir,
        test_data_path: str, test_image_dir: str, paragraphs_dir: str, write_dir: str, file_tag: str):
    model_path = "output/best_alignment_model.pth"
    model = EmbeddingAlignmentMLP(512, 512)
    if not os.path.exists(model_path):
        train_loader = DataLoader(ImageQuestionDataset(train_data_path, train_val_image_dir), batch_size=64,
                                  shuffle=True, num_workers=4)
        val_loader = DataLoader(ImageQuestionDataset(val_data_path, train_val_image_dir), batch_size=64, shuffle=True,
                                num_workers=4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        _train_and_validate(model, train_loader, val_loader, opt, 20, 0.07)
    else:
        model.load_state_dict(torch.load(model_path))

    _run_index(model, test_data_path, paragraphs_dir, test_image_dir, write_dir, "indices")
    _run_search(model, client, test_data_path, write_dir, file_tag, "indices")


class EmbeddingAlignmentMLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(EmbeddingAlignmentMLP, self).__init__()
        self.image_branch = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
        self.text_branch = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
        clip_model_name = "models/clip-vit-base-patch32"
        self.clip_model, self.clip_processor = init_model_and_processor(clip_model_name)

    def get_texts_embedding(self, texts: list[str]):
        texts_embeddings = embedding_texts(self.clip_model, self.clip_processor, texts)
        aligned_texts_embedding = self.text_branch(texts_embeddings)
        return aligned_texts_embedding

    def get_images_embedding(self, images: list[Image.Image]):
        images_embedding = embedding_image(self.clip_model, self.clip_processor, images)
        aligned_images_embedding = self.image_branch(images_embedding)
        return aligned_images_embedding

    def forward(self, questions: list[str], images: list[Image.Image]):
        questions_embedding = embedding_texts(self.clip_model, self.clip_processor, questions)
        images_embedding = embedding_image(self.clip_model, self.clip_processor, images)
        aligned_questions_embedding = self.text_branch(questions_embedding)
        aligned_images_embedding = self.image_branch(images_embedding)
        return aligned_questions_embedding, aligned_images_embedding


class ImageQuestionDataset(Dataset):

    def __init__(self, data_path: str, images_dir: str):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.length = 0
        self.data = []
        for paper_id, paper in data.items():
            self.length += len(paper["qa"])
            for qa in paper["qa"]:
                self.data.append((qa["question"], os.path.join(images_dir, paper_id, qa["reference"])))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        question, image_path = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        return question, image


def _info_nce_loss(questions_embedding: torch.Tensor, images_embedding: torch.Tensor, temperature: float = 0.07):
    similarity_matrix = torch.matmul(questions_embedding, images_embedding.T) / temperature
    logit_positive_pairs = torch.diag(similarity_matrix)
    exp_logits = torch.exp(similarity_matrix)
    sum_exp_logits = exp_logits.sum(dim=1)
    loss = - (logit_positive_pairs - torch.log(sum_exp_logits)).mean()
    return loss


def _train_and_validate(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer,
                        epochs: int, temperature: float = 0.07):
    best_val_loss = float("inf")
    not_improve = 0
    for epoch in range(epochs):

        model.train()
        train_loss = 0
        for batch_idx, (questions, images) in enumerate(train_loader):
            optimizer.zero_grad()
            qs_emb, is_emb = model(questions, images)
            loss = _info_nce_loss(qs_emb, is_emb, temperature)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"epoch {epoch + 1}/{epochs}, loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (questions, images) in enumerate(val_loader):
                qs_emb, is_emb = model(questions, images)
                loss = _info_nce_loss(qs_emb, is_emb, temperature)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "output/best_alignment_model.pth")
            logger.info(f"saved best model with Validation Loss: {best_val_loss:.4f}")
            not_improve = 0
        else:
            not_improve += 1
            if not_improve >= 10:
                logger.info("train 10 epoch no improve, early stop")
                break

    logger.info("training finished")


def _run_index(model: EmbeddingAlignmentMLP, test_data_path: str, paragraphs_dir: str, images_dir: str, write_dir: str,
               indices_dir):

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

        text_embeddings = model.get_texts_embedding(texts).cpu().numpy()
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

        image_embeddings = model.get_images_embedding(images).cpu().numpy()
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


def _run_search(model: EmbeddingAlignmentMLP,client: Client, test_data_path: str, write_dir: str, file_tag: str, indices_dir):
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
        qs_embeddings = model.get_images_embedding(qs).cpu().numpy()
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

        break # TODO delete

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

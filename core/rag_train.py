import json
import os

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from loguru import logger
from openai import Client
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from transformers import CLIPModel, CLIPProcessor

from core.clip import init_model_and_processor, embedding_texts, embedding_images, trunk_by_paragraph
from core.conf import CLIP_MODEL_PATH
from core.data import read_text_file
from core.llm import invoke_llm
from core.metric import create_coco_eval_file, score_compute
from core.output import IndexFileManager, FilesManager
from core.prompt import ImageInfo
from core.train import info_nce_loss


def run(client: Client, train_data_path: str, val_data_path: str, train_val_images_dir: str,
        test_data_path: str, test_images_dir: str, paragraphs_dir: str, write_dir: str, file_tag: str):
    clip_model, clip_processor = init_model_and_processor(CLIP_MODEL_PATH)
    for param in clip_model.parameters():
        param.requires_grad = False
    model_path = "output/best_alignment_model.pth"
    model = EmbeddingAlignmentMLP(512, 512).to(clip_model.device)
    if not os.path.exists(model_path):
        train_loader = DataLoader(ImageQuestionDataset(train_data_path, train_val_images_dir), batch_size=16,
                                  shuffle=True, num_workers=8,
                                  collate_fn=lambda x: _custom_collate_fn(x, clip_processor))
        val_loader = DataLoader(ImageQuestionDataset(val_data_path, train_val_images_dir), batch_size=16, shuffle=True,
                                num_workers=8, collate_fn=lambda x: _custom_collate_fn(x, clip_processor))
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        _train_and_validate(model, clip_model, train_loader, val_loader, opt, 100, 0.07)
    else:
        model.load_state_dict(torch.load(model_path))

    _run_index(model, clip_model, clip_processor, test_data_path, paragraphs_dir, test_images_dir, write_dir, "indices")
    _run_search(model, clip_model, clip_processor, client, test_data_path, write_dir, file_tag, "indices")


class EmbeddingAlignmentMLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(EmbeddingAlignmentMLP, self).__init__()
        self.image_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.text_projection = nn.Linear(input_dim, output_dim, bias=False)
        with torch.no_grad():
            self.image_projection.weight.data = torch.eye(output_dim, input_dim) * 0.1
            self.text_projection.weight.data = torch.eye(output_dim, input_dim) * 0.1

    def get_texts_embedding(self, texts_embeddings: torch.Tensor) -> torch.Tensor:
        aligned_texts_embedding = self.text_projection(texts_embeddings)
        aligned_texts_embedding = functional.normalize(aligned_texts_embedding, dim=1)
        return aligned_texts_embedding

    def get_images_embedding(self, images_embedding: torch.Tensor) -> torch.Tensor:
        aligned_images_embedding = self.image_projection(images_embedding)
        aligned_images_embedding = functional.normalize(aligned_images_embedding, dim=1)
        return aligned_images_embedding

    def forward(self, questions_embedding: torch.Tensor, images_embedding: torch.Tensor):
        aligned_questions_embedding = self.text_projection(questions_embedding)
        aligned_images_embedding = self.image_projection(images_embedding)
        aligned_questions_embedding = functional.normalize(aligned_questions_embedding, dim=1)
        aligned_images_embedding = functional.normalize(aligned_images_embedding, dim=1)
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


def _custom_collate_fn(batch, clip_processor):
    questions = [item[0] for item in batch]
    images = [item[1] for item in batch]
    questions_embedding = clip_processor(text=questions, return_tensors="pt", padding=True, truncation=True)
    images_embedding = clip_processor(images=images, return_tensors="pt")
    return questions_embedding, images_embedding


def _train_and_validate(model: nn.Module, clip_model: CLIPModel, train_loader: DataLoader,
                        val_loader: DataLoader, optimizer: Optimizer, epochs: int, temperature: float = 0.07):
    torch.autograd.set_detect_anomaly(True)
    best_val_loss = float("inf")
    not_improve = 0
    for epoch in range(epochs):

        model.train()
        train_loss = 0
        for batch_idx, (questions_input, images_input) in enumerate(train_loader):
            optimizer.zero_grad()
            questions_input = {k: v.to(clip_model.device) for k, v in questions_input.items()}
            images_input = {k: v.to(clip_model.device) for k, v in images_input.items()}
            questions_embedding = clip_model.get_text_features(**questions_input)
            questions_embedding = functional.normalize(questions_embedding, dim=1)
            images_embedding = clip_model.get_image_features(**images_input)
            images_embedding = functional.normalize(images_embedding, dim=1)

            qs_emb, is_emb = model(questions_embedding, images_embedding)
            loss = info_nce_loss(qs_emb, is_emb, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            logger.debug(f"batch: {batch_idx + 1}, loss: {loss.item()}")
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"epoch {epoch + 1}/{epochs}, loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (questions_input, images_input) in enumerate(val_loader):
                questions_input = {k: v.to(clip_model.device) for k, v in questions_input.items()}
                images_input = {k: v.to(clip_model.device) for k, v in images_input.items()}
                questions_embedding = clip_model.get_text_features(**questions_input)
                questions_embedding = functional.normalize(questions_embedding, dim=1)
                images_embedding = clip_model.get_image_features(**images_input)
                images_embedding = functional.normalize(images_embedding, dim=1)

                qs_emb, is_emb = model(questions_embedding, images_embedding)
                loss = info_nce_loss(qs_emb, is_emb, temperature)
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


def _run_index(model: EmbeddingAlignmentMLP, clip_model: CLIPModel, clip_processor: CLIPProcessor, test_data_path: str,
               paragraphs_dir: str, images_dir: str, write_dir: str, indices_dir: str):
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    if not os.path.exists(os.path.join(write_dir, indices_dir)):
        os.makedirs(os.path.join(write_dir, indices_dir))

    im = IndexFileManager(write_dir, indices_dir)
    model.eval()
    for paper_id, paper in test_data.items():
        id_to_element = dict()
        curr = 0

        index = faiss.IndexIDMap(faiss.IndexFlatL2(512))
        paragraphs_file_path = os.path.join(paragraphs_dir, f"{paper_id}.txt")
        text = read_text_file(paragraphs_file_path)
        texts = trunk_by_paragraph(text)
        logger.info(f"paragraphs: {text[:100]}...")

        texts_embedding = embedding_texts(clip_model, clip_processor, texts)
        with torch.no_grad():
            texts_embedding = model.get_texts_embedding(texts_embedding).cpu().detach().numpy()
        indices = []
        for i, text in enumerate(texts):
            idx = i + curr
            indices.append(idx)
            id_to_element[idx] = {"type": "text", "data": text}
        index.add_with_ids(texts_embedding, np.array(indices))
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
                caption=image_detail["caption"]).model_dump())

        images_embedding = embedding_images(clip_model, clip_processor, images)
        with torch.no_grad():
            images_embedding = model.get_images_embedding(images_embedding).cpu().detach().numpy()
        indices = []
        for i, image_info in enumerate(images_info):
            idx = i + curr
            indices.append(idx)
            id_to_element[str(idx)] = {"type": "image", "data": image_info}
        index.add_with_ids(images_embedding, np.array(indices))
        im.write_images_index(paper_id, index)
        logger.info(f"write {paper_id} images index finish")

        im.write_id_to_element(paper_id, id_to_element)
        logger.info(f"write {paper_id} id_to_element json finish")


def _run_search(model: EmbeddingAlignmentMLP, clip_model: CLIPModel, clip_processor: CLIPProcessor, client: Client,
                test_data_path: str, write_dir: str, file_tag: str, indices_dir: str):
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"current tag {file_tag}")

    fm = FilesManager(write_dir, file_tag)
    skip_qa_count = fm.read_skip_count()

    curr_qa_count = 0
    find_true_image_count = 0
    im = IndexFileManager(write_dir, indices_dir)
    model.eval()
    for paper_id, paper in test_data.items():
        with open(os.path.join(write_dir, indices_dir, f"{paper_id}.json"), "r", encoding="utf-8") as f:
            id_to_element = json.load(f)

        texts_index = im.read_texts_index(paper_id)
        images_index = im.read_images_index(paper_id)
        qs = [qa["question"] for qa in paper["qa"]]
        qs_embedding = embedding_texts(clip_model, clip_processor, qs)
        with torch.no_grad():
            qs_embedding = model.get_texts_embedding(qs_embedding).cpu().detach().numpy()
        texts_distances, texts_find_indices = texts_index.search(qs_embedding, 3)
        images_distances, images_find_indices = images_index.search(qs_embedding, 1)

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
            #
            # texts = [id_to_element[str(idx)]["data"] for idx in texts_find_indices[i]]
            # paragraphs = "\n\n---\n\n".join(texts)
            # try:
            #     answer = invoke_llm(client, qa["question"], paragraphs, images_info)
            # except Exception as e:
            #     logger.warning(f"invoke_llm failed: {e}")
            #     fm.write_skip_count(curr_qa_count)
            #     continue
            #
            # logger.info(f"images_info {images_info}")
            # logger.info(f"question: {qa['question']}")
            # logger.info(f"gt_answer: {qa['answer']}")
            # logger.info(f"pred_answer: {answer}")
            #
            # d = {
            #     "id": f"{paper_id}_{i}",
            #     "question": qa["question"],
            #     "pred_answer": answer,
            #     "gt_answer": qa["answer"],
            # }

            # fm.write_gene_line(d)
            fm.write_skip_count(curr_qa_count)

    # pred_answers, gt_answers = [], []
    # for line in fm.read_gene_file():
    #     data = json.loads(line.strip())
    #     pred_answers.append(data["pred_answer"])
    #     gt_answers.append(data["gt_answer"])
    #
    # create_coco_eval_file(fm.pred_file_path, fm.gnth_file_path, pred_answers, gt_answers)
    # score = score_compute(fm.pred_file_path, fm.gnth_file_path, metrics=["METEOR", "ROUGE_L", "CIDEr", "BERTScore"])
    # score["RetAcc"] = round(find_true_image_count / curr_qa_count, 4)
    # logger.info(f"score: {score}")
    # fm.write_metric(score)

import json
import os

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from loguru import logger
from openai import Client
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer

from core.embedding import embedding_texts, embedding_images
from core.conf import EMB_MODEL_NAME, LLM_API_KEY, EMB_DIM
from core.output import IndexFileManager, FilesManager
from core.prompt import ImageInfo
from core.train import info_nce_loss


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

COUNT = 30000000


async def run(client: Client, train_data_path: str, val_data_path: str, train_val_images_dir: str,
        test_data_path: str, test_images_dir: str, paragraphs_dir: str, write_dir: str, file_tag: str):
    model_path = "output/best_alignment_model.pth"
    model = EmbeddingAlignmentMLP(EMB_DIM, EMB_DIM)
    model = model.to(model.device)
    if not os.path.exists(model_path):
        dataset_dir = "output/rag_dataset"
        train_loader = DataLoader(
            await ImageQuestionDataset.create(train_data_path, train_val_images_dir, os.path.join(dataset_dir, "train")),
            batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: _custom_collate_fn(x))
        val_loader = DataLoader(
            await ImageQuestionDataset.create(val_data_path, train_val_images_dir, os.path.join(dataset_dir, "val")),
            batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: _custom_collate_fn(x))
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        _train_and_validate(model, train_loader, val_loader, opt, 20, 0.07)
    else:
        model.load_state_dict(torch.load(model_path))

    await _run_index(model, test_data_path, test_images_dir, write_dir, "indices")
    await _run_search(model, test_data_path, write_dir, file_tag, "indices")


class EmbeddingAlignmentMLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(EmbeddingAlignmentMLP, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.image_projection = nn.Linear(input_dim, output_dim, bias=False, dtype=torch.float32)
        self.text_projection = nn.Linear(input_dim, output_dim, bias=False, dtype=torch.float32)
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

    def __init__(self, target_dir: str, length):
        self.target_dir = target_dir
        self.length = length

    @classmethod
    async def create(cls, data_path: str, images_dir: str, target_dir: str):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sorted_data = sorted(data.items())

        for i, (paper_id, paper) in enumerate(sorted_data[:COUNT]):
            if os.path.exists(os.path.join(target_dir, f"data_{i}.pt")):
                continue
            questions = [qa["question"] for qa in paper["qa"]]
            images_path = [os.path.join(images_dir, paper_id, qa["reference"]) for qa in paper["qa"]]

            questions_embedding = torch.from_numpy(
                await embedding_texts(EMB_MODEL_NAME, LLM_API_KEY, questions)
            )
            images_embedding = torch.from_numpy(
                await embedding_images(EMB_MODEL_NAME, LLM_API_KEY, images_path)
            )

            torch.save({"paper_id": paper_id, "images_embedding": images_embedding, "questions_embedding": questions_embedding}, os.path.join(target_dir, f"data_{i}.pt"))
            logger.debug(f"saved data_{i}.pt")

        return cls(target_dir, len(sorted_data[:COUNT]))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.target_dir, f"data_{idx}.pt"), weights_only=False, map_location="cpu")
        return data


def _custom_collate_fn(batch):
    return batch


def _train_and_validate(model: nn.Module, train_loader: DataLoader,
                        val_loader: DataLoader, optimizer: Optimizer, epochs: int, temperature: float = 0.07):
    torch.autograd.set_detect_anomaly(True)
    best_val_loss = float("inf")
    not_improve = 0
    for epoch in range(epochs):

        model.train()
        train_loss = 0
        for batch_idx, papers_input in enumerate(train_loader):
            batch_loss = 0
            for paper_input in papers_input:
                paper_input["questions_embedding"] = paper_input["questions_embedding"].to(model.device).float()
                paper_input["images_embedding"] = paper_input["images_embedding"].to(model.device).float()
                optimizer.zero_grad()
                qs_emb, is_emb = model(paper_input["questions_embedding"], paper_input["images_embedding"])
                loss = info_nce_loss(qs_emb, is_emb, temperature)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                batch_loss += loss.item()
            logger.debug(f"batch: {batch_idx + 1}, loss: {batch_loss:.4f}")
            train_loss += batch_loss
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"epoch {epoch + 1}/{epochs}, loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, papers_input in enumerate(val_loader):
                for paper_input in papers_input:
                    paper_input["questions_embedding"] = paper_input["questions_embedding"].to(model.device).float()
                    paper_input["images_embedding"] = paper_input["images_embedding"].to(model.device).float()
                    qs_emb, is_emb = model(paper_input["questions_embedding"], paper_input["images_embedding"])
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


async def _run_index(model: EmbeddingAlignmentMLP, test_data_path: str, images_dir: str, write_dir: str, indices_dir: str):
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    if not os.path.exists(os.path.join(write_dir, indices_dir)):
        os.makedirs(os.path.join(write_dir, indices_dir))

    im = IndexFileManager(write_dir, indices_dir)
    model.eval()
    count = 0
    for i, (paper_id, paper) in enumerate(sorted(test_data.items())):
        id_to_element = dict()
        curr = 0

        index = faiss.IndexIDMap(faiss.IndexFlatL2(EMB_DIM))
        images_info = []
        for image_name, image_detail in paper["all_figures"].items():
            images_info.append(ImageInfo(
                type="image/png",
                name=image_name,
                path=os.path.join(images_dir, paper_id, image_name),
                caption=image_detail["caption"]))

        data = torch.load(os.path.join("output/rag_dataset/test-A-all_figures", f"data_{i}.pt"), weights_only=False, map_location="cpu")
        image_embeddings = data["all_figures_embedding"].to(model.device).float()
        with torch.no_grad():
            images_embedding = model.get_images_embedding(image_embeddings).cpu().detach().numpy()

        indices = []
        for i, image_info in enumerate(images_info):
            idx = i + curr
            indices.append(idx)
            id_to_element[str(idx)] = {"type": "image", "data": image_info.model_dump()}
        index.add_with_ids(images_embedding, np.array(indices))
        im.write_images_index(paper_id, index)
        logger.info(f"write {paper_id} images index finish")

        im.write_id_to_element(paper_id, id_to_element)
        logger.info(f"write {paper_id} id_to_element json finish")

        count += 1
        if count >= COUNT:
            break


async def _run_search(model: EmbeddingAlignmentMLP, test_data_path: str, write_dir: str, file_tag: str, indices_dir: str):
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"current tag {file_tag}")

    fm = FilesManager(write_dir, file_tag)
    progress = fm.read_curr_progress()
    im = IndexFileManager(write_dir, indices_dir)
    model.eval()
    curr = 0
    count = 0
    score = dict()
    for i, (paper_id, paper) in enumerate(sorted(test_data.items())):
        with open(os.path.join(write_dir, indices_dir, f"{paper_id}.json"), "r", encoding="utf-8") as f:
            id_to_element = json.load(f)

        images_index = im.read_images_index(paper_id)
        data = torch.load(os.path.join("output/rag_dataset/test-A", f"data_{i}.pt"), weights_only=False, map_location="cpu")
        qs_embeddings = data["questions_embedding"].to(model.device).float()
        with torch.no_grad():
            qs_embedding = model.get_texts_embedding(qs_embeddings).cpu().detach().numpy()
        images_distances, images_find_indices = images_index.search(qs_embedding, 3)

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

        count += 1
        if count >= COUNT:
            break

    score["RetAcc"] = round(progress.true_image_count / progress.curr_total_count, 4)
    logger.info(f"score: {score}")
    fm.write_metric(score)

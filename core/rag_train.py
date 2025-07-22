import json
import os

import torch
import torch.nn as nn
from loguru import logger
from openai import Client
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer

from core.clip import init_model_and_processor, embedding_texts, embedding_image


def run(client: Client, train_data_path: str, val_data_path: str, train_val_image_dir,
        test_data_path: str, test_image_dir, paragraphs_dir: str, images_dir: str, write_dir: str, file_tag: str):
    model_path = "output/best_alignment_model.pth"
    model = EmbeddingAlignmentMLP(512, 512)
    if not os.path.exists(model_path):
        train_loader = DataLoader(ImageQuestionDataset(train_data_path, train_val_image_dir), batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(ImageQuestionDataset(val_data_path, train_val_image_dir), batch_size=64, shuffle=True, num_workers=4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        train_and_validate(model, train_loader, val_loader, opt, 20, 0.07)
    else:
        model.load_state_dict(torch.load(model_path))




class EmbeddingAlignmentMLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(EmbeddingAlignmentMLP, self).__init__()
        self.image_branch = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
        self.question_branch = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
        clip_model_name = "models/clip-vit-base-patch32"
        self.clip_model, self.clip_processor = init_model_and_processor(clip_model_name)

    def forward(self, questions: list[str], images: list[Image.Image]):
        questions_embedding = embedding_texts(self.clip_model, self.clip_processor, questions)
        images_embedding = embedding_image(self.clip_model, self.clip_processor, images)
        aligned_questions_embedding = self.question_branch(questions_embedding)
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


def info_nce_loss(questions_embedding: torch.Tensor, images_embedding: torch.Tensor, temperature: float = 0.07):
    similarity_matrix = torch.matmul(questions_embedding, images_embedding.T) / temperature
    logit_positive_pairs = torch.diag(similarity_matrix)
    exp_logits = torch.exp(similarity_matrix)
    sum_exp_logits = exp_logits.sum(dim=1)
    loss = - (logit_positive_pairs - torch.log(sum_exp_logits)).mean()
    return loss


def train_and_validate(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, epochs: int, temperature: float = 0.07):
    best_val_loss = float("inf")
    not_improve = 0
    for epoch in range(epochs):

        model.train()
        train_loss = 0
        for batch_idx, (questions, images) in enumerate(train_loader):
            optimizer.zero_grad()
            qs_emb, is_emb = model(questions, images)
            loss = info_nce_loss(qs_emb, is_emb, temperature)
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

import torch
import torch.nn.functional as functional


def info_nce_loss(questions_embedding: torch.Tensor, images_embedding: torch.Tensor, temperature: float = 0.07):
    similarity_matrix = torch.matmul(questions_embedding, images_embedding.T) / temperature
    labels = torch.arange(len(questions_embedding)).to(questions_embedding.device)
    loss = functional.cross_entropy(similarity_matrix, labels)
    return loss

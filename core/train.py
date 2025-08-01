import torch


def info_nce_loss(questions_embedding: torch.Tensor, images_embedding: torch.Tensor, temperature: float = 0.07):
    similarity_matrix = torch.matmul(questions_embedding, images_embedding.T) / temperature
    logit_positive_pairs = torch.diag(similarity_matrix)
    exp_logits = torch.exp(similarity_matrix)
    sum_exp_logits = exp_logits.sum(dim=1)
    loss = - (logit_positive_pairs - torch.log(sum_exp_logits)).mean()
    return loss

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from loguru import logger


def init_model_and_processor(model_name: str) -> (CLIPModel, CLIPProcessor):
    model = CLIPModel.from_pretrained(model_name, device_map="auto")
    processor = CLIPProcessor.from_pretrained(model_name)
    logger.info("init_model_and_processor success")
    return model, processor


def embedding_texts(model: CLIPModel, processor: CLIPProcessor, texts: list[str]) -> torch.Tensor:
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs)
    return text_embeddings


def embedding_image(model: CLIPModel, processor: CLIPProcessor, images: list[Image.Image]) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)
    return image_embeddings


def trunk_by_paragraph(text: str) -> list[str]:
    texts = text.split("\n\n")
    return texts


def _id_to_text(texts: list[str], curr: int) -> dict[int, str]:
    res = dict()
    for i, text in enumerate(texts):
        res[i + curr] = text
    return res
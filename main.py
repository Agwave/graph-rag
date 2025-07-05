import torch

from loguru import logger
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


def run():
    model_name = "openai/clip-vit-base-patch32"
    model, processor = _init_model_and_processor(model_name)
    text = "I am a doctor"
    text_embeddings = _embedding_text(model, processor, text)
    logger.info(text)
    logger.info(f"embeddings shape: {text_embeddings.shape}")
    logger.info(f"embeddings example: {text_embeddings[0, :5]}")
    dummy_image = Image.new('RGB', (224, 224), color='red')
    image_embeddings = _embedding_image(model, processor, dummy_image)
    logger.info(f"embeddings shape: {image_embeddings.shape}")
    logger.info(f"embeddings example: {image_embeddings[0, :5]}")



def _init_model_and_processor(model_name: str) -> (CLIPModel, CLIPProcessor):
    model = CLIPModel.from_pretrained(model_name, device_map="auto")
    processor = CLIPProcessor.from_pretrained(model_name)
    logger.info("init_model_and_processor success")
    return model, processor


def _embedding_text(model: CLIPModel, processor: CLIPProcessor, text: str) -> torch.Tensor:
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs)
    return text_embeddings


def _embedding_image(model: CLIPModel, processor: CLIPProcessor, image: Image.Image) -> torch.Tensor:
    inputs = processor(images=[image], return_tensors="pt")
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)
    return image_embeddings


if __name__ == '__main__':
    run()

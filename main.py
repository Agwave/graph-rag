import os

import faiss
import numpy as np
import torch

from loguru import logger
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


def run():
    texts_dir = "datasets/spiqa/SPIQA_train_val_test-A_extracted_paragraphs"
    images_dir = "datasets/spiqa/train_val/SPIQA_train_val_Images"
    model_name = "models/clip-vit-base-patch32"

    model, processor = _init_model_and_processor(model_name)
    curr = 0

    search_text = None
    search_embeddings = None
    search_index = None
    id_to_text = dict()
    index = faiss.IndexIDMap(faiss.IndexFlatL2(512))
    for file_name in os.listdir(texts_dir)[:10]:
        text = _read_text_file(os.path.join(texts_dir, file_name))
        texts = _trunk_by_paragraph(text)
        logger.info(f"before trunk: {text[:100]}")
        logger.info(f"after trunk (first text): {texts[0]}")

        text_embeddings = _embedding_texts(model, processor, texts).cpu().numpy()
        indices = []
        for i, text in enumerate(texts):
            idx = i + curr
            indices.append(idx)
            id_to_text[idx] = text
        index.add_with_ids(text_embeddings, np.array(indices))

        search_embeddings = text_embeddings[0:1, :]
        search_text = texts[0]
        search_index = curr

        curr += len(texts)

    distances, find_indices = index.search(search_embeddings, 1)
    logger.info(f"search | index {search_index} | text {search_text[:200]}")
    index_id = find_indices[0][0]
    distance = distances[0][0]
    find_text = id_to_text[index_id]
    logger.info(f"find | index {index_id} | text {find_text} | distance {distance}")

    search_path = None
    search_embeddings = None
    search_index = None
    id_to_image_path = dict()
    for image_dir in os.listdir(images_dir)[:10]:
        images = []
        paths = []
        for file_name in os.listdir(os.path.join(images_dir, image_dir))[:10]:
            img = Image.open(os.path.join(images_dir, image_dir, file_name)).convert("RGB")
            images.append(img)
            paths.append(os.path.join(image_dir, file_name))
        logger.info(f"images count: {len(images)}")

        image_embeddings = _embedding_image(model, processor, images).cpu().numpy()
        indices = []
        for i, path in enumerate(paths):
            idx = i + curr
            indices.append(idx)
            id_to_image_path[idx] = path
        index.add_with_ids(image_embeddings, np.array(indices))

        search_embeddings = image_embeddings[0:1, :]
        search_path = paths[0]
        search_index = curr

        curr += len(paths)

    distances, find_indices = index.search(search_embeddings, 1)
    logger.info(f"search | index {search_index} | image {search_path}")
    index_id = find_indices[0][0]
    distance = distances[0][0]
    find_image_path = id_to_image_path[index_id]
    logger.info(f"find | index {index_id} | image {find_image_path} | distance {distance}")

    file_name = "my_rag_index.faiss"
    faiss.write_index(index, file_name)
    logger.info(f"written index to {file_name}")

    faiss.read_index("my_rag_index.faiss")
    logger.info(f"loaded index from {file_name}")


def _init_model_and_processor(model_name: str) -> (CLIPModel, CLIPProcessor):
    model = CLIPModel.from_pretrained(model_name, device_map="auto")
    processor = CLIPProcessor.from_pretrained(model_name)
    logger.info("init_model_and_processor success")
    return model, processor


def _embedding_texts(model: CLIPModel, processor: CLIPProcessor, texts: list[str]) -> torch.Tensor:
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs)
    return text_embeddings


def _embedding_image(model: CLIPModel, processor: CLIPProcessor, images: list[Image.Image]) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)
    return image_embeddings


def _trunk_by_paragraph(text: str) -> list[str]:
    texts = text.split("\n\n")
    return texts


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _id_to_text(texts: list[str], curr: int) -> dict[int, str]:
    res = dict()
    for i, text in enumerate(texts):
        res[i + curr] = text
    return res


if __name__ == '__main__':
    run()

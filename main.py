import os

import faiss
import numpy as np
import torch

from loguru import logger
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


def run():
    dir_ = ("/home/chenyinbo/.cache/huggingface/hub/datasets--google--spiqa/snapshots/"
            "1774b71511f029b82089a069d75328f25fbf0705/SPIQA_train_val_test-A_extracted_paragraphs")

    model_name = "openai/clip-vit-base-patch32"
    model, processor = _init_model_and_processor(model_name)
    search_text = None
    search_embeddings = None
    search_index = None
    id_to_text = dict()
    curr = 0
    index = faiss.IndexIDMap(faiss.IndexFlatL2(512))
    for file_name in os.listdir(dir_)[:10]:
        text = _read_text_file(os.path.join(dir_, file_name))
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

    file_name = "my_rag_index.faiss"
    faiss.write_index(index, file_name)
    logger.info(f"written index to {file_name}")

    loaded_index = faiss.read_index("my_rag_index.faiss")
    logger.info(f"loaded index from {file_name}")

    distances, find_indices = loaded_index.search(search_embeddings, 1)
    logger.info(f"search | index {search_index} | text {search_text[:200]}")
    index_id = find_indices[0][0]
    distance = distances[0][0]
    find_text = id_to_text[index_id]
    logger.info(f"find | index {index_id} | text {find_text} | distance {distance}")


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


def _embedding_image(model: CLIPModel, processor: CLIPProcessor, image: Image.Image) -> torch.Tensor:
    inputs = processor(images=[image], return_tensors="pt")
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

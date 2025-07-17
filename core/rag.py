import json
import os

import faiss
import numpy as np
from loguru import logger
from openai import Client
from PIL import Image

from core.clip import init_model_and_processor, trunk_by_paragraph, embedding_texts, embedding_image
from core.data import read_text_file
from core.output import FilesManager


def run(client: Client, test_data_path: str, paragraphs_dir: str, images_dir: str, write_dir: str, output_tag: str):
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    _run_index(test_data_path, paragraphs_dir, images_dir, write_dir)





def _run_search(client: Client, test_data_path: str, paragraphs_dir: str, images_dir: str, write_dir: str, file_tag: str):
    model_name = "models/clip-vit-base-patch32"
    model, processor = init_model_and_processor(model_name)

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"current tag {file_tag}")

    fm = FilesManager(write_dir, file_tag)
    skip_qa_count = fm.read_skip_count()

    curr_qa_count = 0
    for paper_id, paper in test_data.items():
        index = faiss.read_index(os.path.join(write_dir, "index", f"{paper_id}.faiss"))
        with open(os.path.join(write_dir, "index", f"{paper_id}.json"), "r", encoding="utf-8") as f:
            id_to_element = json.load(f)

        qs = [qa["question"] for qa in paper["qa"]]
        qs_embeddings = embedding_texts(model, processor, qs)

        for emb, qa in zip(qs_embeddings, paper["qa"]):
            curr_qa_count += 1
            if skip_qa_count > 0:
                skip_qa_count -= 1
                logger.info(f"skip qa {curr_qa_count}")
                continue

            logger.info(f"compute current qa {curr_qa_count} ...")
            distances, find_indices = index.search(emb, 10)

            texts = []
            image = None

            for idx in find_indices:
                if len(texts) >= 3 and image is not None:
                    continue
                ele = id_to_element[idx]
                if ele["type"] == "text":
                    pass


def _run_index(test_data_path: str, paragraphs_dir: str, images_dir: str, write_dir: str):
    model_name = "models/clip-vit-base-patch32"
    model, processor = init_model_and_processor(model_name)
    indices_dir = "indices"

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    if not os.path.exists(os.path.join(write_dir, indices_dir)):
        os.makedirs(os.path.join(write_dir, indices_dir))

    for paper_id, paper in test_data.items():
        index = faiss.IndexIDMap(faiss.IndexFlatL2(512))
        id_to_element = dict()
        curr = 0

        paragraphs_file_path = os.path.join(paragraphs_dir, f"{paper_id}.txt")
        text = read_text_file(paragraphs_file_path)
        texts = trunk_by_paragraph(text)
        logger.info(f"paragraphs: {text[:100]}...")

        text_embeddings = embedding_texts(model, processor, texts).cpu().numpy()
        indices = []
        for i, text in enumerate(texts):
            idx = i + curr
            indices.append(idx)
            id_to_element[idx] = {"type": "text", "data": text}
        index.add_with_ids(text_embeddings, np.array(indices))
        curr += len(texts)

        images = []
        image_names = []
        for file_name in os.listdir(os.path.join(images_dir, paper_id)):
            img = Image.open(os.path.join(images_dir, paper_id, file_name)).convert("RGB")
            images.append(img)
            image_names.append(file_name)
        logger.info(f"images count: {len(images)}")

        image_embedings = embedding_image(model, processor, image_names).cpu().numpy()
        indices = []
        for i, image_name in enumerate(image_names):
            idx = i + curr
            indices.append(idx)
            id_to_element[idx] = {"type": "image", "data": image_name}
        index.add_with_ids(image_embedings, np.array(indices))

        faiss.write_index(index, os.path.join(write_dir, indices_dir, f"{paper_id}.faiss"))
        logger.info(f"write index to index/{paper_id}.faiss finish")
        with open(os.path.join(write_dir, indices_dir, f"{paper_id}.json"), "w", encoding="utf-8") as f:
            json.dump(id_to_element, f, ensure_ascii=False, indent=4)
        logger.info(f"write id_to_element to index/{paper_id}.json finish")

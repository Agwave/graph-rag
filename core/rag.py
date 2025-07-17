import json
import os

import faiss
import numpy as np
from loguru import logger
from openai import Client
from PIL import Image

from core.clip import init_model_and_processor, trunk_by_paragraph, embedding_texts, embedding_image
from core.data import read_text_file


def run(client: Client, test_data_path: str, paragraphs_dir: str, images_dir: str, write_dir: str, output_tag: str):
    model_name = "models/clip-vit-base-patch32"

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_a_data = json.load(f)
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    logger.info(f"current tag {output_tag}")

    skip_qa_count = 0
    skip_qa_filename = os.path.join(write_dir, f"skip_qa_{output_tag}.txt")
    if os.path.exists(skip_qa_filename):
        with open(skip_qa_filename, "r", encoding="utf-8") as f:
            skip_qa_count = int(f.read())

    gen_qa_filename = os.path.join(write_dir, f"gen_qa_{output_tag}.jsonl")


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

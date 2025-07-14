import json
import os
from datetime import datetime

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from loguru import logger
from openai import OpenAI
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from core.conf import show_env, SPIQA_DIR, WRITE_DIR
from core.metric import create_coco_eval_file, score_compute
from core.qwen import run as qwen_run, ImageInfo


def run():
    show_env()
    client = OpenAI(api_key="sk-c450e178a972467d93b282e218c1dfba",
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    test_data_path = os.path.join(SPIQA_DIR, "test-A/SPIQA_testA.json")
    paragraphs_dir = os.path.join(SPIQA_DIR, "SPIQA_train_val_test-A_extracted_paragraphs")
    images_dir = os.path.join(SPIQA_DIR, "test-A/SPIQA_testA_Images_224px")

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_a_data = json.load(f)
    if not os.path.exists(WRITE_DIR):
        os.makedirs(WRITE_DIR)

    now = datetime.now()
    curr_time = now.strftime("%Y%m%d%H%M")
    logger.info(f"current tag {curr_time}")

    skip_qa_count = 0
    skip_qa_filename = os.path.join(WRITE_DIR, f"skip_qa_{curr_time}.txt")
    if os.path.exists(skip_qa_filename):
        with open(skip_qa_filename, "r", encoding="utf-8") as f:
            skip_qa_count = int(f.read())

    gen_qa_filename = os.path.join(WRITE_DIR, f"gen_qa_{curr_time}.jsonl")

    curr_qa_count = 0
    for paper_id, paper in test_a_data.items():

        images = []
        for image_name, image_detail in paper["all_figures"].items():
            images.append(ImageInfo(
                type="image/png",
                path=os.path.join(images_dir, paper_id, image_name),
                caption=image_detail["caption"],
            ))

        paragraphs = _read_text_file(os.path.join(paragraphs_dir, f"{paper_id}.txt"))
        for i, qa in enumerate(paper["qa"]):
            curr_qa_count += 1
            if skip_qa_count > 0:
                skip_qa_count -= 1
                logger.info(f"skip qa {curr_qa_count}")
                continue

            logger.info(f"compute current qa {curr_qa_count} ...")

            try:
                answer = qwen_run(client, qa["question"], paragraphs, images)
            except Exception as e:
                logger.warning(f"qwen_run failed: {e}")
                with open(skip_qa_filename, "w", encoding="utf-8") as f:
                    f.write(str(curr_qa_count))
                continue

            logger.info(f"paragraphs len: {len(paragraphs)}")
            logger.info(f"images {images}")
            logger.info(f"question: {qa['question']}")
            logger.info(f"gt_answer: {qa['answer']}")
            logger.info(f"pred_answer: {answer}")

            d = {
                "id": f"{paper_id}_{i}",
                "question": qa["question"],
                "pred_answer": answer,
                "gt_answer": qa["answer"],
            }

            with open(gen_qa_filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

            with open(skip_qa_filename, "w", encoding="utf-8") as f:
                f.write(str(curr_qa_count))


    pred_answers, gt_answers = [], []
    with open(gen_qa_filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            pred_answers.append(data["pred_answer"])
            gt_answers.append(data["gt_answer"])

    pred_path = os.path.join(WRITE_DIR, f"pred_{curr_time}.json")
    gt_path = os.path.join(WRITE_DIR, f"gt_{curr_time}.json")
    create_coco_eval_file(pred_path, gt_path, pred_answers, gt_answers)
    score = score_compute(pred_path, gt_path, metrics=["Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "BERTScore"])

    metric_path = os.path.join(WRITE_DIR, f"metric_{curr_time}.json")
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(score, f, ensure_ascii=False)


def _run_embedding_texts_images():
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


def _read_parquet(path: str, n_rows: int) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    first_iter_batch = pf.iter_batches(batch_size=n_rows)
    first_batch = next(first_iter_batch)
    df_head = first_batch.to_pandas()
    return df_head.head(n_rows)


if __name__ == '__main__':
    run()

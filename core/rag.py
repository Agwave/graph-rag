import asyncio
import json
import os

import faiss
import numpy as np
from loguru import logger
from numpy.ma.core import count
from openai import Client

from core.embedding import embedding_texts, embedding_images
from core.preprocess import trunk_by_paragraph
from core.conf import EMB_DIM, EMB_MODEL_NAME, LLM_API_KEY
from core.data import read_text_file
from core.llm import invoke_llm
from core.metric import create_coco_eval_file, score_compute
from core.output import FilesManager, IndexFileManager
from core.prompt import ImageInfo


def run(client: Client, test_data_path: str, paragraphs_dir: str, images_dir: str, write_dir: str, file_tag: str):
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    asyncio.run(_run_index(test_data_path, paragraphs_dir, images_dir, write_dir, "indices"))
    asyncio.run(_run_search(client, test_data_path, write_dir, file_tag, "indices"))


async def _run_search(client: Client, test_data_path: str, write_dir: str, file_tag: str, indices_dir):
    score = dict()

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"current tag {file_tag}")

    fm = FilesManager(write_dir, file_tag)
    progress = fm.read_curr_progress()
    im = IndexFileManager(write_dir, indices_dir)
    curr = 0
    for paper_id, paper in sorted(test_data.items()):
        with open(os.path.join(write_dir, indices_dir, f"{paper_id}.json"), "r", encoding="utf-8") as f:
            id_to_element = json.load(f)

        texts_index = im.read_texts_index(paper_id)
        images_index = im.read_images_index(paper_id)
        qs = [qa["question"] for qa in paper["qa"]]
        qs_embeddings = await embedding_texts(EMB_MODEL_NAME, LLM_API_KEY, qs)
        texts_distances, texts_find_indices = texts_index.search(qs_embeddings, 3)
        images_distances, images_find_indices = images_index.search(qs_embeddings, 3)

        for i, qa in enumerate(paper["qa"]):
            curr += 1
            if curr < progress.curr_total_count:
                logger.info(f"skip qa {curr}")
                continue

            progress.curr_total_count += 1
            logger.info(f"compute current qa {progress.curr_total_count} ...")

            images_info = [ImageInfo(**id_to_element[str(idx)]["data"]) for idx in images_find_indices[i] if idx >= 0]
            if images_info[0].name == qa["reference"]:
                progress.true_image_count += 1
            logger.info(f"target image {qa['reference']}, predict image {images_info[0].name}")
            logger.info(f"acc {progress.true_image_count / progress.curr_total_count}")

    #         texts = [id_to_element[str(idx)]["data"] for idx in texts_find_indices[i] if idx >= 0]
    #         paragraphs = "\n\n---\n\n".join(texts)
    #         try:
    #             answer = invoke_llm(client, qa["question"], paragraphs, images_info)
    #         except Exception as e:
    #             logger.warning(f"invoke_llm failed: {e}")
    #             progress.except_count += 1
    #             fm.write_curr_progress(progress)
    #             continue
    #
    #         logger.info(f"pred_true {progress.true_image_count}, llm_except {progress.except_count}, "
    #                     f"total {progress.curr_total_count}")
    #         logger.info(f"question: {qa['question']}")
    #         logger.info(f"gt_answer: {qa['answer']}")
    #         logger.info(f"pred_answer: {answer}")
    #
    #         d = {
    #             "id": f"{paper_id}_{i}",
    #             "question": qa["question"],
    #             "pred_answer": answer,
    #             "gt_answer": qa["answer"],
    #         }
    #
    #         fm.write_gene_line(d)
    #         fm.write_curr_progress(progress)
    #
    # pred_answers, gt_answers = [], []
    # for line in fm.read_gene_file():
    #     data = json.loads(line.strip())
    #     pred_answers.append(data["pred_answer"])
    #     gt_answers.append(data["gt_answer"])
    #
    # create_coco_eval_file(fm.pred_file_path, fm.gnth_file_path, pred_answers, gt_answers)
    # score = score_compute(fm.pred_file_path, fm.gnth_file_path, metrics=["METEOR", "ROUGE_L", "CIDEr", "BERTScore"])
    score["RetAcc"] = round(progress.true_image_count / progress.curr_total_count, 4)
    logger.info(f"score: {score}")
    fm.write_metric(score)


async def _run_index(test_data_path: str, paragraphs_dir: str, images_dir: str, write_dir: str, indices_dir):
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    if not os.path.exists(os.path.join(write_dir, indices_dir)):
        os.makedirs(os.path.join(write_dir, indices_dir))
    else:
        return

    im = IndexFileManager(write_dir, indices_dir)
    count = 0
    for paper_id, paper in sorted(test_data.items()):
        id_to_element = dict()
        curr = 0

        index = faiss.IndexIDMap(faiss.IndexFlatIP(EMB_DIM))
        # paragraphs_file_path = os.path.join(paragraphs_dir, f"{paper_id}.txt")
        # text = read_text_file(paragraphs_file_path)
        # texts = trunk_by_paragraph(text)
        # logger.info(f"paragraphs: {len(texts)}")
        #
        # text_embeddings = await embedding_texts(EMB_MODEL_NAME, LLM_API_KEY, texts)
        # indices = []
        # for i, text in enumerate(texts):
        #     idx = i + curr
        #     indices.append(idx)
        #     id_to_element[idx] = {"type": "text", "data": text}
        # index.add_with_ids(text_embeddings, np.array(indices))
        # curr += len(texts)
        im.write_texts_index(paper_id, index)
        # logger.info(f"write {paper_id} texts index finish")

        index = faiss.IndexIDMap(faiss.IndexFlatIP(EMB_DIM))
        images_info = []
        for image_name, image_detail in paper["all_figures"].items():
            images_info.append(ImageInfo(
                type="image/png",
                name=image_name,
                path=os.path.join(images_dir, paper_id, image_name),
                caption=image_detail["caption"])
            )

        image_embeddings = await embedding_images(EMB_MODEL_NAME, LLM_API_KEY, images_info)

        indices = []
        for i, image_info in enumerate(images_info):
            idx = i + curr
            indices.append(idx)
            id_to_element[str(idx)] = {"type": "image", "data": image_info.model_dump()}
        index.add_with_ids(image_embeddings, np.array(indices))
        im.write_images_index(paper_id, index)
        logger.info(f"write {paper_id} images index finish")

        im.write_id_to_element(paper_id, id_to_element)
        logger.info(f"write {paper_id} id_to_element json finish")

        count += 1
        if count >= 50:
            break

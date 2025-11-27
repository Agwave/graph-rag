import asyncio
import json
import os

import faiss
import numpy as np
from loguru import logger

import torch
from core.conf import EMB_DIM
from core.output import FilesManager, IndexFileManager
from core.prompt import ImageInfo


def run(test_data_path: str, images_dir: str, write_dir: str, file_tag: str):
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    asyncio.run(_run_index(test_data_path, images_dir, write_dir, "indices"))
    asyncio.run(_run_search(test_data_path, write_dir, file_tag, "indices"))


async def _run_search(test_data_path: str, write_dir: str, file_tag: str, indices_dir):
    score = dict()

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"current tag {file_tag}")

    fm = FilesManager(write_dir, file_tag)
    progress = fm.read_curr_progress()
    im = IndexFileManager(write_dir, indices_dir)
    curr = 0
    for i, (paper_id, paper) in enumerate(sorted(test_data.items())):
        with open(os.path.join(write_dir, indices_dir, f"{paper_id}.json"), "r", encoding="utf-8") as f:
            id_to_element = json.load(f)

        images_index = im.read_images_index(paper_id)
        data = torch.load(os.path.join("output/rag_dataset/test-A", f"data_{i}.pt"), weights_only=False, map_location="cpu")
        qs_embeddings = data["questions_embedding"].numpy()
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

    score["RetAcc"] = round(progress.true_image_count / progress.curr_total_count, 4)
    logger.info(f"score: {score}")
    fm.write_metric(score)


async def _run_index(test_data_path: str, images_dir: str, write_dir: str, indices_dir):
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    if not os.path.exists(os.path.join(write_dir, indices_dir)):
        os.makedirs(os.path.join(write_dir, indices_dir))
    else:
        return

    im = IndexFileManager(write_dir, indices_dir)
    for i, (paper_id, paper) in enumerate(sorted(test_data.items())):
        id_to_element = dict()

        index = faiss.IndexIDMap(faiss.IndexFlatIP(EMB_DIM))
        images_info = []
        for image_name, image_detail in paper["all_figures"].items():
            images_info.append(ImageInfo(
                type="image/png",
                name=image_name,
                path=os.path.join(images_dir, paper_id, image_name),
                caption=image_detail["caption"])
            )
        data = torch.load(os.path.join("output/rag_dataset/test-A-all_figures", f"data_{i}.pt"), weights_only=False, map_location="cpu")
        image_embeddings = data["all_figures_embedding"].numpy()

        indices = []
        for j, image_info in enumerate(images_info):
            idx = j
            indices.append(idx)
            id_to_element[str(idx)] = {"type": "image", "data": image_info.model_dump()}
        index.add_with_ids(image_embeddings, np.array(indices))
        im.write_images_index(paper_id, index)
        logger.info(f"write {paper_id} images index finish")

        im.write_id_to_element(paper_id, id_to_element)
        logger.info(f"write {paper_id} id_to_element json finish")

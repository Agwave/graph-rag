import json
import os

from loguru import logger
from openai import Client

from core.llm import invoke_llm_find_image
from core.prompt import ImageInfo


def run(client: Client, test_data_path: str, images_dir: str):
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    curr_qa_count = 0
    correct = 0
    for paper_id, paper in test_data.items():

        images_info = []
        for image_name, image_detail in paper["all_figures"].items():
            images_info.append(ImageInfo(
                type="image/png",
                name=image_name,
                path=os.path.join(images_dir, paper_id, image_name),
                caption=image_detail["caption"],
            ))

        for i, qa in enumerate(paper["qa"]):
            logger.info(f"compute current qa {curr_qa_count} ...")

            try:
                answer = invoke_llm_find_image(client, qa["question"], images_info)
            except Exception as e:
                logger.warning(f"qwen_run failed: {e}")
                continue

            logger.info(f"images_info {images_info}")
            logger.info(f"question: {qa['question']}")
            logger.info(f"pred_answer: {answer}")
            logger.info(f"reference: {qa['reference']}")

            if answer == qa["reference"]:
                correct += 1
            curr_qa_count += 1
            logger.info(f"pred_true {correct}, total {curr_qa_count}, acc {correct}/{curr_qa_count}")

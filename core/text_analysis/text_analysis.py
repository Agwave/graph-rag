import json
import os
import re
from collections import Counter

from loguru import logger


SPIQA_DIR = os.getenv("SPIQA_DIR", "/home/chenyinbo/.cache/huggingface/hub/datasets--google--spiqa/snapshots/1774b71511f029b82089a069d75328f25fbf0705")
WRITE_DIR = os.getenv("WRITE_DIR", "/home/chenyinbo/dataset/graph-rag-output")
BERT_MODEL_DIR = os.getenv("BERT_MODEL_DIR", "./models/bert-base-uncased")
API_MODEL = os.getenv("API_MODEL", "qwen-vl-max-2025-08-13")
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", "./models/clip-vit-base-patch32")
ROOT_DIR = os.getenv("ROOT_DIR", "/home/chenyinbo/dataset/graph-rag-output/graph_dataset/root")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "qwen2.5-vl-embedding")
EMB_DIM = int(os.getenv("EMB_DIM", "1024"))
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-c450e178a972467d93b282e218c1dfba")


def run():
    with open(os.path.join(SPIQA_DIR, "train_val/SPIQA_train.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    lack_count = 0
    empty_num_count = 0
    for i, (paper_id, paper) in enumerate(sorted(data.items())):
        logger.info(f"---------{i}---------")
        logger.info(f"paper: {paper_id}")
        image_names = [image_name for image_name in paper["all_figures"].keys()]
        logger.info(f"image_names: {image_names}")
        with open(os.path.join(SPIQA_DIR, "SPIQA_train_val_test-A_extracted_paragraphs", f"{paper_id}.txt"), "r", encoding="utf-8") as f:
            text = f.read()
        counter = count_figure_table(text)
        for key, value in counter.items():
            if key[1] == "":
                empty_num_count += 1
                break
        logger.info(f"counter: {counter}")
        counter_sum = sum(counter.values())
        logger.info(f"image_len: {len(image_names)}, all_count: {counter_sum}")
        if len(image_names) > counter_sum:
            lack_count += 1
            logger.warning(f"image_len: {len(image_names)}, all_count: {counter_sum}")

        for image_name in image_names:
            _, num_png = image_name.rsplit("-", maxsplit=1)
            num, _ = num_png.split(".", maxsplit=1)
            if num != "1":
                logger.warning(f"second image num not 1, image_name {image_name}")

    logger.info(f"lack_count: {lack_count}, empty_num_count: {empty_num_count}")


def count_figure_table(text: str) -> dict:
    """
    统计文本中 Figure / figure / Table / table 的出现次数
    """
    pattern = (
        r'\b'
        r'(?:Fig(?:ure)?s?\.?|Tables?)'          # Figure / Fig. / Table（含复数）
        r'(?:\s*[1-9]\d*(?:[a-z]|\([a-z]\))?)?'  # 编号 + a / (a)（整体可选）
        r'\b'
    )

    pattern = (
        r'\b'
        r'(Fig(?:ure)?s?\.?|Tables?)'  # group 1: Figure / Fig. / Table
        r'(?:\s*([1-9]\d*))?'  # group 2: 编号（可选，只捕获数字）
        r'(?:[a-z]|\([a-z]\))?'  # a / (a)，但不捕获
        r'\b'
    )
    matches = re.findall(pattern, text)
    return dict(Counter(matches))


if __name__ == '__main__':
    run()

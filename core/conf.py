import os

from loguru import logger


SPIQA_DIR = os.getenv("SPIQA_DIR", "/data/datasets/spiqa")
WRITE_DIR = os.getenv("WRITE_DIR", "./output")
BERT_MODEL_DIR = os.getenv("BERT_MODEL_DIR", "/data/models/bert-base-uncased")
API_MODEL = os.getenv("API_MODEL", "qwen-vl-max-2025-04-08")
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", "/data/models/clip-vit-base-patch32")


def show_env():
    logger.info(f"SPIQA_DIR: {SPIQA_DIR}")
    logger.info(f"WRITE_DIR: {WRITE_DIR}")
    logger.info(f"BERT_MODEL_DIR: {BERT_MODEL_DIR}")

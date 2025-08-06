import os

from loguru import logger


SPIQA_DIR = os.getenv("SPIQA_DIR", "./datasets/spiqa")
WRITE_DIR = os.getenv("WRITE_DIR", "./output")
BERT_MODEL_DIR = os.getenv("BERT_MODEL_DIR", "./models/bert-base-uncased")
API_MODEL = os.getenv("API_MODEL", "qwen-vl-max-2025-04-08")
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", "./models/clip-vit-base-patch32")
ROOT_DIR = os.getenv("ROOT_DIR", "./output/dataset")


def show_env():
    logger.info(f"SPIQA_DIR: {SPIQA_DIR}")
    logger.info(f"WRITE_DIR: {WRITE_DIR}")
    logger.info(f"BERT_MODEL_DIR: {BERT_MODEL_DIR}")
    logger.info(f"API_MODEL: {API_MODEL}")
    logger.info(f"CLIP_MODEL_PATH: {CLIP_MODEL_PATH}")
    logger.info(f"ROOT_DIR: {ROOT_DIR}")

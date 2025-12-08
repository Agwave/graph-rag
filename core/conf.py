import os

from loguru import logger


SPIQA_DIR = os.getenv("SPIQA_DIR", "/home/chenyinbo/.cache/huggingface/hub/datasets--google--spiqa/snapshots/1774b71511f029b82089a069d75328f25fbf0705")
WRITE_DIR = os.getenv("WRITE_DIR", "./output")
BERT_MODEL_DIR = os.getenv("BERT_MODEL_DIR", "./models/bert-base-uncased")
API_MODEL = os.getenv("API_MODEL", "qwen-vl-max-2025-08-13")
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", "./models/clip-vit-base-patch32")
ROOT_DIR = os.getenv("ROOT_DIR", "./output/graph_dataset/root")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "qwen2.5-vl-embedding")
EMB_DIM = int(os.getenv("EMB_DIM", "1024"))
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-c450e178a972467d93b282e218c1dfba")


def show_env():
    logger.info(f"SPIQA_DIR: {SPIQA_DIR}")
    logger.info(f"WRITE_DIR: {WRITE_DIR}")
    logger.info(f"BERT_MODEL_DIR: {BERT_MODEL_DIR}")
    logger.info(f"API_MODEL: {API_MODEL}")
    logger.info(f"CLIP_MODEL_PATH: {CLIP_MODEL_PATH}")
    logger.info(f"ROOT_DIR: {ROOT_DIR}")
    logger.info(f"EMB_DIM: {EMB_DIM}")

import os

from loguru import logger


SPIQA_DIR = os.getenv("SPIQA_DIR", "/app/graph-rag/datasets/spiqa")
WRITE_DIR = os.getenv("WRITE_DIR", "/app/graph-rag/output")


def show_env():
    logger.info(f"SPIQA_DIR: {SPIQA_DIR}")
    logger.info(f"WRITE_DIR: {WRITE_DIR}")

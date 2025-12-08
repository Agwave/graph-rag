import asyncio
import os
import json
from idlelib.window import add_windows_to_menu

import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from core.conf import SPIQA_DIR, EMB_MODEL_NAME, LLM_API_KEY
from core.data import read_text_file, chunk_text
from core.embedding import embedding_images, embedding_texts
from core.rag_train import ImageQuestionDataset
from core.prompt import ImageInfo


train_data_path = os.path.join(SPIQA_DIR, "train_val/SPIQA_train.json")
val_data_path = os.path.join(SPIQA_DIR, "train_val/SPIQA_val.json")
test_data_path = os.path.join(SPIQA_DIR, "test-A/SPIQA_testA.json")
train_val_images_dir = os.path.join(SPIQA_DIR, "train_val/SPIQA_train_val_Images")
test_images_dir = os.path.join(SPIQA_DIR, "test-A/SPIQA_testA_Images")
train_val_test_texts_dir = os.path.join(SPIQA_DIR, "SPIQA_train_val_test-A_extracted_paragraphs")
dataset_dir = "output/graph_dataset"


async def f():
    # train_loader = DataLoader(
    #     await ImageQuestionDataset.create(train_data_path, train_val_images_dir, os.path.join(dataset_dir, "train")),
    #     batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: _custom_collate_fn(x))
    # val_loader = DataLoader(
    #     await ImageQuestionDataset.create(val_data_path, train_val_images_dir, os.path.join(dataset_dir, "val")),
    #     batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: _custom_collate_fn(x))
    # test_loader = DataLoader(
    #     await ImageQuestionDataset.create(test_data_path, test_images_dir, os.path.join(dataset_dir, "test-A")),
    #     batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: _custom_collate_fn(x))
    # await emb_test_all_images(test_data_path, test_images_dir, os.path.join(dataset_dir, "test-A-all_figures"))
    await emb_all_texts(train_data_path, train_val_test_texts_dir, os.path.join(dataset_dir, "train_texts"))
    # await emb_all_texts(val_data_path, train_val_test_texts_dir, os.path.join(dataset_dir, "val_texts"))
    # await emb_all_texts(test_data_path, train_val_test_texts_dir, os.path.join(dataset_dir, "test_a_texts"))
    # data = torch.load(os.path.join("output/rag_dataset/train_texts", f"data_0.pt"), weights_only=False,
    #                   map_location="cpu")
    # print(data["paper_id"])
    # print(len(data["texts"]), data["texts"][0])
    # print(data["texts_embedding"].size(), data["texts_embedding"])


def _custom_collate_fn(batch):
    return batch

def run():
    asyncio.run(f())


async def emb_test_all_images(data_path: str, images_dir: str, target_dir: str):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, (paper_id, paper) in enumerate(sorted(data.items())):
        if os.path.exists(os.path.join(target_dir, f"data_{i}.pt")):
            continue
        images_path, images_info = [], []
        for image_name, image_detail in paper["all_figures"].items():
            images_info.append(ImageInfo(
                type="image/png",
                name=image_name,
                path=os.path.join(images_dir, paper_id, image_name),
                caption=image_detail["caption"])
            )
            images_path.append(os.path.join(images_dir, paper_id, image_name))

        images_embedding = torch.from_numpy(
            await embedding_images(EMB_MODEL_NAME, LLM_API_KEY, images_path)
        )
        torch.save({"paper_id": paper_id, "all_figures_embedding": images_embedding}, os.path.join(target_dir, f"data_{i}.pt"))
        logger.debug(f"saved data_{i}.pt")


async def emb_all_texts(data_path: str, texts_dir: str, target_dir: str):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, (paper_id, paper) in enumerate(sorted(data.items())):
        if os.path.exists(os.path.join(target_dir, f"data_{i}.pt")):
            continue
        text_path = os.path.join(texts_dir, f"{paper_id}.txt")
        text = read_text_file(text_path)
        texts = chunk_text(text)
        logger.debug(f"text example: {texts[:1]}")
        try:
            texts_embedding = torch.from_numpy(
                await embedding_texts(EMB_MODEL_NAME, LLM_API_KEY, texts)
            )
        except Exception as e:
            logger.warning(e)
            continue
        torch.save({"paper_id": paper_id, "texts": texts, "texts_embedding": texts_embedding}, os.path.join(target_dir, f"data_{i}.pt"))
        logger.debug(f"saved data_{i}.pt")

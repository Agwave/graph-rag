import os
import json

import torch
import asyncio
from loguru import logger

from core.conf import EMB_DIM, LLM_API_KEY, EMB_MODEL_NAME, SPIQA_DIR
from core.embedding import embedding_images


train_data_path = os.path.join(SPIQA_DIR, "train_val/SPIQA_train.json")
val_data_path = os.path.join(SPIQA_DIR, "train_val/SPIQA_val.json")
test_data_path = os.path.join(SPIQA_DIR, "test-A/SPIQA_testA.json")
train_val_images_dir = os.path.join(SPIQA_DIR, "train_val/SPIQA_train_val_Images")
test_images_dir = os.path.join(SPIQA_DIR, "test-A/SPIQA_testA_Images")
train_val_test_texts_dir = os.path.join(SPIQA_DIR, "SPIQA_train_val_test-A_extracted_paragraphs")


async def create(data_path: str, images_dir: str, target_dir: str, source_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sorted_data = sorted(data.items())

    for i, (paper_id, paper) in enumerate(sorted_data):
        if os.path.exists(os.path.join(target_dir, f"data_{i}.pt")):
            continue

        all_images_name = list(paper["all_figures"].keys())
        all_images_embedding = torch.zeros((len(all_images_name), EMB_DIM))
        name_to_index = {name: i for i, name in enumerate(all_images_name)}

        reference_images_name = [qa["reference"] for qa in paper["qa"]]
        assert len(reference_images_name) == len(set(reference_images_name))
        reference_images_index = [name_to_index[name] for name in reference_images_name]
        data = torch.load(os.path.join(source_dir, f"data_{i}.pt"), weights_only=False, map_location="cpu")
        all_images_embedding[reference_images_index] = data["images_embedding"]

        if len(reference_images_name) != len(all_images_name):
            other_images_name = [name for name in all_images_name if name not in reference_images_name]
            other_images_index = [name_to_index[name] for name in other_images_name]
            other_images_path = [os.path.join(images_dir, paper_id, name) for name in other_images_name]
            other_images_embedding = torch.from_numpy(
                await embedding_images(EMB_MODEL_NAME, LLM_API_KEY, other_images_path)
            )
            all_images_embedding[other_images_index] = other_images_embedding
            logger.warning(f"no equal len, data_{i}.pt {reference_images_index}")

        torch.save(
            {"paper_id": paper_id, "all_images_name": all_images_name, "all_images_embedding": all_images_embedding},
            os.path.join(target_dir, f"data_{i}.pt"),
        )
        logger.debug(f"saved data_{i}.pt")


async def create_test(data_path: str, images_dir: str, target_dir: str, source_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sorted_data = sorted(data.items())

    for i, (paper_id, paper) in enumerate(sorted_data):
        if os.path.exists(os.path.join(target_dir, f"data_{i}.pt")):
            continue
        all_images_name = list(paper["all_figures"].keys())
        data = torch.load(os.path.join(source_dir, f"data_{i}.pt"), weights_only=False, map_location="cpu")
        assert len(all_images_name) == len(data["all_figures_embedding"])

        reference_images_name = [qa["reference"] for qa in paper["qa"]]
        assert len(reference_images_name) == len(all_images_name)
        torch.save(
            {"paper_id": paper_id, "all_images_name": all_images_name, "all_images_embedding": data["all_figures_embedding"]},
            os.path.join(target_dir, f"data_{i}.pt"),
        )
        logger.debug(f"saved data_{i}.pt")


def run_graph_dataset_store():
    # asyncio.run(create(train_data_path, train_val_images_dir, os.path.join("output/graph_dataset", "train_images"), os.path.join("output/graph_dataset", "train")))
    # asyncio.run(create(val_data_path, train_val_images_dir, os.path.join("output/graph_dataset", "val_images"), os.path.join("output/graph_dataset", "val")))
    asyncio.run(create(test_data_path, test_images_dir, os.path.join("output/graph_dataset", "test_a_images"), os.path.join("output/graph_dataset", "test_a")))



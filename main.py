import os
from datetime import datetime

from openai import OpenAI

from core.conf import show_env, SPIQA_DIR, WRITE_DIR
from core.cot import run as cot_run
from core.full import run as full_run
from core.rag import run as rag_run
from core.rag_train import run as rag_train_run
from core.rag_graph_train import run as rag_graph_train_run


def run():
    show_env()
    client = OpenAI(api_key="sk-c450e178a972467d93b282e218c1dfba",
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    train_data_path = os.path.join(SPIQA_DIR, "train_val/SPIQA_train.json")
    val_data_path = os.path.join(SPIQA_DIR, "train_val/SPIQA_val.json")
    test_data_path = os.path.join(SPIQA_DIR, "test-A/SPIQA_testA.json")
    paragraphs_dir = os.path.join(SPIQA_DIR, "SPIQA_train_val_test-A_extracted_paragraphs")
    test_images_dir = os.path.join(SPIQA_DIR, "test-A/SPIQA_testA_Images_224px")
    train_val_images_dir = os.path.join(SPIQA_DIR, "train_val/SPIQA_train_val_Images")
    now = datetime.now()
    curr_time = now.strftime("%Y%m%d%H%M")
    # full_run(client, test_data_path, paragraphs_dir, test_images_dir, os.path.join(WRITE_DIR, f"full_{curr_time}"), f"full_{curr_time}")
    # cot_run(client, test_data_path, test_images_dir)
    # rag_run(client, test_data_path, paragraphs_dir, test_images_dir, os.path.join(WRITE_DIR, f"rag_{curr_time}"), f"rag_{curr_time}")
    rag_train_run(client, train_data_path, val_data_path, train_val_images_dir, test_data_path, test_images_dir,
                  paragraphs_dir, os.path.join(WRITE_DIR, f"rag_train_{curr_time}"), f"rag_train_{curr_time}")
    # rag_graph_train_run(client, train_data_path, val_data_path, train_val_images_dir, test_data_path, test_images_dir,
    #                     paragraphs_dir, os.path.join(WRITE_DIR, f"graph_rag_train_{curr_time}"), curr_time)


if __name__ == '__main__':
    run()

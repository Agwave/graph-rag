import os
from datetime import datetime

from openai import OpenAI

from core.conf import show_env, SPIQA_DIR, WRITE_DIR
from core.full import run as full_run
from core.rag import run as rag_run

def run():
    show_env()
    client = OpenAI(api_key="sk-c450e178a972467d93b282e218c1dfba",
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    test_data_path = os.path.join(SPIQA_DIR, "test-A/SPIQA_testA.json")
    paragraphs_dir = os.path.join(SPIQA_DIR, "SPIQA_train_val_test-A_extracted_paragraphs")
    images_dir = os.path.join(SPIQA_DIR, "test-A/SPIQA_testA_Images_224px")
    now = datetime.now()
    curr_time = now.strftime("%Y%m%d%H%M")
    # full_run(client, test_data_path, paragraphs_dir, images_dir, os.path.join(WRITE_DIR, f"full_{curr_time}"), f"full_{curr_time}")
    rag_run(client, test_data_path, paragraphs_dir, images_dir, os.path.join(WRITE_DIR, f"rag_{curr_time}"), f"rag_{curr_time}")


if __name__ == '__main__':
    run()

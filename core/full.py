import json
import os

from loguru import logger
from openai import Client

from core.data import read_text_file
from core.llm import invoke_llm
from core.metric import create_coco_eval_file, score_compute
from core.output import FilesManager
from core.prompt import ImageInfo


def run(client: Client, test_data_path: str, paragraphs_dir: str, images_dir: str, write_dir: str, file_tag: str):
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"current tag {file_tag}")
    fm = FilesManager(os.path.join(write_dir, file_tag), file_tag)
    progress = fm.read_curr_progress()
    curr = 0
    for paper_id, paper in test_data.items():

        images_info = []
        for image_name, image_detail in paper["all_figures"].items():
            images_info.append(ImageInfo(
                type="image/png",
                name=image_name,
                path=os.path.join(images_dir, paper_id, image_name),
                caption=image_detail["caption"],
            ))

        paragraphs = read_text_file(os.path.join(paragraphs_dir, f"{paper_id}.txt"))
        for i, qa in enumerate(paper["qa"]):
            curr += 1
            if curr < progress.curr_total_count:
                logger.info(f"skip qa {curr}")
                continue

            progress.curr_total_count += 1
            logger.info(f"compute current qa {progress.curr_total_count} ...")

            try:
                answer = invoke_llm(client, qa["question"], paragraphs, images_info)
            except Exception as e:
                logger.warning(f"qwen_run failed: {e}")
                progress.except_count += 1
                fm.write_curr_progress(progress)
                continue

            logger.info(f"paragraphs len: {len(paragraphs)}")
            logger.info(f"images_info {images_info}")
            logger.info(f"question: {qa['question']}")
            logger.info(f"gt_answer: {qa['answer']}")
            logger.info(f"pred_answer: {answer}")

            d = {
                "id": f"{paper_id}_{i}",
                "question": qa["question"],
                "pred_answer": answer,
                "gt_answer": qa["answer"],
            }

            fm.write_gene_line(d)
            fm.write_curr_progress(progress)

    pred_answers, gt_answers = [], []
    for line in fm.read_gene_file():
        data = json.loads(line.strip())
        pred_answers.append(data["pred_answer"])
        gt_answers.append(data["gt_answer"])

    create_coco_eval_file(fm.pred_file_path, fm.gnth_file_path, pred_answers, gt_answers)
    score = score_compute(fm.pred_file_path, fm.gnth_file_path, metrics=["METEOR", "ROUGE_L", "CIDEr", "BERTScore"])
    logger.info(f"score: {score}")
    fm.write_metric(score)

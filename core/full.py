import base64
import json
import os
from datetime import datetime

from loguru import logger
from openai import Client
from pydantic import BaseModel, Field

from core.conf import API_MODEL
from core.data import read_text_file
from core.metric import create_coco_eval_file, score_compute
from core.output import FilesManager

PROMPT = """You are given a question, paragraphs from a scientific paper, a few input images, and a caption corresponding to each input image. \
Please answer the question based on the paper, input images and corresponding captions. \
Question: <question>. Output in the following json format: {"Answer": "Direct Answer to the Question"}. \n"""


class ImageInfo(BaseModel):
    type: str = Field("image/png", description="image type")
    path: str = Field("", description="image path")
    caption: str = Field("", description="image caption")


def run(client: Client, test_data_path: str, paragraphs_dir: str, images_dir: str, write_dir: str, file_tag: str):
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_a_data = json.load(f)

    logger.info(f"current tag {file_tag}")

    fm = FilesManager(write_dir, file_tag)
    skip_qa_count = fm.read_skip_count()

    curr_qa_count = 0
    for paper_id, paper in test_a_data.items():

        images = []
        for image_name, image_detail in paper["all_figures"].items():
            images.append(ImageInfo(
                type="image/png",
                path=os.path.join(images_dir, paper_id, image_name),
                caption=image_detail["caption"],
            ))

        paragraphs = read_text_file(os.path.join(paragraphs_dir, f"{paper_id}.txt"))
        for i, qa in enumerate(paper["qa"]):
            curr_qa_count += 1
            if skip_qa_count > 0:
                skip_qa_count -= 1
                logger.info(f"skip qa {curr_qa_count}")
                continue

            logger.info(f"compute current qa {curr_qa_count} ...")

            try:
                answer = _invoke_llm(client, qa["question"], paragraphs, images)
            except Exception as e:
                logger.warning(f"qwen_run failed: {e}")
                fm.write_skip_count(skip_qa_count)
                continue

            logger.info(f"paragraphs len: {len(paragraphs)}")
            logger.info(f"images {images}")
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
            fm.write_skip_count(skip_qa_count)

        break # TODO delete


    pred_answers, gt_answers = [], []
    for line in fm.read_gene_file():
        data = json.loads(line.strip())
        pred_answers.append(data["pred_answer"])
        gt_answers.append(data["gt_answer"])

    create_coco_eval_file(fm.pred_file_path, fm.gnth_file_path, pred_answers, gt_answers)
    score = score_compute(fm.pred_file_path, fm.gnth_file_path, metrics=["METEOR", "ROUGE_L", "CIDEr", "BERTScore"])
    fm.write_metric(score)


def _invoke_llm(client: Client, question: str, paragraphs: str, images: list[ImageInfo]):
    content = _content(question, paragraphs, images)
    completion = client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ]
    )
    response = completion.choices[0].message.content
    try:
        answer = _extract_answer(response)
    except Exception as e:
        logger.warning(f"extract answer failed: {e}, response: {response}")
        return response
    return answer


def _content(question: str, paragraphs: str, images: list[ImageInfo]) -> list[dict]:
    res = [{"type": "text", "text": PROMPT.replace("<question>", question)},
           {"type": "text", "text": f"Paragraphs from the paper: {paragraphs}"}]
    for i, image in enumerate(images):
        res.append({"type": "text", "text": f"Image {i}:"})
        image_encoded = _encode_image_to_base6(image.path)
        res.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_encoded}"}})
        res.append({"type": "text", "text": f"Caption {i}: {image.caption}"})
    return res


def _encode_image_to_base6(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_answer(text: str) -> str:
    json_start = text.find("```json")
    json_end = text.rfind("```")
    if json_start == -1 or json_end == -1:
        logger.warning(f"_extract_json failed: {text}")
        return json.loads(text.strip())["Answer"]
    answer = json.loads(text[json_start+7:json_end].strip())["Answer"]
    return answer

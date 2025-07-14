import base64
import json

from loguru import logger
from openai import Client
from pydantic import BaseModel, Field


PROMPT = "You are given a question, paragraphs from a scientific paper, a few input images, and a caption corresponding to each input image. \
Please answer the question based on the paper, input images and corresponding captions. \
Question: <question>. Output in the following format: {'Answer': 'Direct Answer to the Question'}. \n"


class ImageInfo(BaseModel):
    type: str = Field("image/png", description="image type")
    path: str = Field("", description="image path")
    caption: str = Field("", description="image caption")


def run(client: Client, question: str, paragraphs: str, images: list[ImageInfo]):
    content = _content(question, paragraphs, images)
    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ]
    )
    response = completion.choices[0].message.content
    answer = _extract_answer(response)
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
        return text
    answer = json.loads(text[json_start+7:json_end].strip())["Answer"]
    return answer

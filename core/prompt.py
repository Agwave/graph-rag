import base64

from pydantic import BaseModel, Field


PROMPT = """You are given a question, paragraphs from a scientific paper, a few input images, and a caption corresponding to each input image. \
Please answer the question based on the paper, input images and corresponding captions. \
Question: <question>. Output in the following json format: {"Answer": "Direct Answer to the Question"}. \n"""


FIND_IMAGE_PROMPT = """You are given a question and a few input images from a scientific paper, \
Please find the most helpful image to answer the following question. \
Please only output the image name. Such as, 1611.04684v1-Table5-1.png \
Question: <question>.\n"""


FIND_IMAGE_ANSWER_PROMPT = """You are given a question, paragraphs, a few input images from a scientific paper. \
Please first find the most helpful image to answer the following question, and then answer the question.

Question: <question>. 

Output in the following json format: {"ImageName": "1611.04684v1-Table5-1.png", "Answer": "Direct Answer to the Question"}. \
ImageName field is image name, Answer field is the answer to the question."""


class ImageInfo(BaseModel):
    type: str = Field("image/png", description="image type")
    name: str = Field("", description="name of the image")
    path: str = Field("", description="image path")
    caption: str = Field("", description="image caption")


def build_model_content(question: str, paragraphs: str, images_info: list[ImageInfo]) -> list[dict]:
    res = [{"type": "text", "text": PROMPT.replace("<question>", question)},
           {"type": "text", "text": f"Paragraphs from the paper: {paragraphs}"}]
    for i, image in enumerate(images_info):
        res.append({"type": "text", "text": f"Image {i}:"})
        image_encoded = _encode_image_to_base64(image.path)
        res.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_encoded}"}})
        # res.append({"type": "text", "text": f"Caption {i}: {image.caption}"})
    return res


def build_find_image_content(question: str, images_info: list[ImageInfo]) -> list[dict]:
    res = [{"type": "text", "text": FIND_IMAGE_PROMPT.replace("<question>", question)}]
    for i, image in enumerate(images_info):
        res.append({"type": "text", "text": f"Image Name {image.name}:"})
        image_encoded = _encode_image_to_base64(image.path)
        res.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_encoded}"}})
        # res.append({"type": "text", "text": f"Caption {i}: {image.caption}"})
    return res


def build_find_image_answer_content(question: str, paragraphs: str, images_info: list[ImageInfo]) -> list[dict]:
    res = [{"type": "text", "text": FIND_IMAGE_ANSWER_PROMPT.replace("<question>", question)},
           {"type": "text", "text": f"Paragraphs from the paper: {paragraphs}"}]
    for i, image in enumerate(images_info):
        res.append({"type": "text", "text": f"Image Name {image.name}:"})
        image_encoded = _encode_image_to_base64(image.path)
        res.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_encoded}"}})
    return res


def _encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
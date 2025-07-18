import json

from loguru import logger
from openai import Client

from core.conf import API_MODEL
from core.prompt import build_model_content, ImageInfo


def invoke_llm(client: Client, question: str, paragraphs: str, images_info: list[ImageInfo]):
    content = build_model_content(question, paragraphs, images_info)
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
        answer = extract_answer(response)
    except Exception as e:
        logger.warning(f"extract answer failed: {e}, response: {response}")
        return response
    return answer


def extract_answer(text: str) -> str:
    json_start = text.find("```json")
    json_end = text.rfind("```")
    if json_start == -1 or json_end == -1:
        logger.warning(f"_extract_json failed: {text}")
        return json.loads(text.strip())["Answer"]
    answer = json.loads(text[json_start+7:json_end].strip())["Answer"]
    return answer
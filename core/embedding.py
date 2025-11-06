import dashscope
import asyncio

import numpy as np
from loguru import logger

from core.prompt import encode_image_to_base64


async def embedding_texts(model_name, api_key, texts):
    semaphore = asyncio.Semaphore(3)
    tasks = []
    for text in texts:
        tasks.append(invoke_llm_embedding(semaphore, model_name, api_key, [{"text": text}]))
    res = await asyncio.gather(*tasks, return_exceptions=True)
    return np.array(res)


async def embedding_images(model_name, api_key, images_path):
    semaphore = asyncio.Semaphore(3)
    tasks = []
    for path in images_path:
        image_encoded = encode_image_to_base64(path)
        tasks.append(invoke_llm_embedding(semaphore, model_name, api_key,
                                          [{"image": f"data:image/png;base64,{image_encoded}"}]))
    res = await asyncio.gather(*tasks, return_exceptions=True)
    return np.array(res)


async def invoke_llm_embedding(semaphore, model_name, api_key, model_input):
    async with semaphore:
        resp = await dashscope.AioMultiModalEmbedding.call(
            model=model_name,
            api_key=api_key,
            input=model_input,
        )
        fail_count = 0
        while resp["status_code"] != 200 and fail_count < 5:
            logger.warning(f"resp status_code not 200: {resp}")
            await asyncio.sleep(2 ** fail_count)
            fail_count += 1
            resp = await dashscope.AioMultiModalEmbedding.call(
                model=model_name,
                api_key=api_key,
                input=model_input,
            )
        if fail_count >= 5:
            raise
        return resp["output"]["embeddings"][0]["embedding"]

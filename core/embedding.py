import dashscope
import asyncio

import numpy as np
from torch.fx.passes.pass_manager import logger

from core.prompt import encode_image_to_base64


async def embedding_texts(model_name, api_key, texts):
    semaphore = asyncio.Semaphore(3)
    tasks = []
    for text in texts:
        tasks.append(invoke_llm_embedding(semaphore, model_name, api_key, [{"text": text}]))
    res = await asyncio.gather(*tasks, return_exceptions=True)
    return np.array(res)


async def embedding_images(model_name, api_key, images_info):
    semaphore = asyncio.Semaphore(3)
    tasks = []
    for image_info in images_info:
        image_encoded = encode_image_to_base64(image_info.path)
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
        if resp["status_code"] != 200:
            logger.warning(f"resp status_code not 200: {resp}")
            raise
        return resp["output"]["embeddings"][0]["embedding"]

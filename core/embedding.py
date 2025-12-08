import dashscope
import asyncio

import numpy as np
from loguru import logger

from core.conf import EMB_DIM
from core.prompt import encode_image_to_base64


async def embedding_texts(model_name, api_key, texts):
    semaphore = asyncio.Semaphore(5)
    tasks = []
    for text in texts:
        tasks.append(invoke_llm_embedding(semaphore, model_name, api_key, [{"text": text}]))
    res = await asyncio.gather(*tasks)
    # logger.debug(f"embedding_texts res type {[type(r) for r in res]}")
    # logger.debug(f"embedding_texts res len {[len(r) for r in res]}")
    return np.array(res, dtype=np.float32)


async def embedding_images(model_name, api_key, images_path):
    semaphore = asyncio.Semaphore(5)
    tasks = []
    for path in images_path:
        image_encoded = encode_image_to_base64(path)
        tasks.append(invoke_llm_embedding(semaphore, model_name, api_key,
                                          [{"image": f"data:image/png;base64,{image_encoded}"}]))
    res = await asyncio.gather(*tasks, return_exceptions=True)
    # logger.debug(f"embedding_images res type {[type(r) for r in res]}")
    # logger.debug(f"embedding_images res len {[len(r) for r in res]}")
    return np.array(res, dtype=np.float32)


async def invoke_llm_embedding(semaphore, model_name, api_key, model_input, timeout=30):
    fail_count = 0

    while fail_count <= 1:
        try:
            # 使用 asyncio.wait_for 设置超时
            async with semaphore:
                resp = await asyncio.wait_for(
                    dashscope.AioMultiModalEmbedding.call(model=model_name, api_key=api_key, input=model_input),
                    timeout  # 超时限制
                )

            # 检查响应内容是否符合预期
            if resp["status_code"] == 200 and isinstance(resp["output"]["embeddings"][0]["embedding"], list) and len(
                    resp["output"]["embeddings"][0]["embedding"]) == EMB_DIM:
                return resp["output"]["embeddings"][0]["embedding"]
            else:
                logger.warning(f"Invalid response structure or embedding length: {resp}")

        except asyncio.TimeoutError:
            # 捕获超时异常并进行重试
            logger.warning(f"Request timed out on attempt {fail_count + 1}")
        except Exception as e:
            # 捕获其他异常并进行重试
            logger.error(f"Error occurred during embedding call: {e}")

        # 超时或失败后重试
        await asyncio.sleep(2 ** fail_count)
        fail_count += 1

    # 如果超过最大重试次数，抛出异常
    logger.error(f"Exceeded maximum retry attempts ({fail_count})")
    raise Exception("Failed to get valid response after multiple attempts.")

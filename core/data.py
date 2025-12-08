import re

import pandas as pd
import pyarrow.parquet as pq


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_parquet(path: str, n_rows: int) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    first_iter_batch = pf.iter_batches(batch_size=n_rows)
    first_batch = next(first_iter_batch)
    df_head = first_batch.to_pandas()
    return df_head.head(n_rows)


def chunk_text(text: str, max_tokens=100):
    paras = text.split("\n\n")
    results = []

    for para in paras:
        if len(para.split()) <= max_tokens:
            if para:
                results.append(para)
        else:
            results.extend(chunk_text_by_punc(para, max_tokens))

    return results

def chunk_text_by_punc(text: str, max_tokens: int = 100, punc_pattern: str = r"[.!?;]"):
    words = text.split()
    buffer = []
    res = []
    overflow = False

    for word in words:
        buffer.append(word)
        if len(word) > max_tokens:
            overflow = True
        if overflow and re.search(punc_pattern, word):
            res.append(" ".join(buffer))
            buffer.clear()
            overflow = False

    if buffer:
        res.append(" ".join(buffer))
    return res

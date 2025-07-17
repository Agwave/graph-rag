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

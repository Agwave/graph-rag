import pyarrow.parquet as pq
import pandas as pd





def read_first_n_rows_efficiently(file_path, n=5):
    """
    高效地读取 Parquet 文件的前 'n' 行，避免加载整个文件。

    Args:
        file_path (str): Parquet 文件路径。
        n (int): 要读取的行数。

    Returns:
        pandas.DataFrame: 包含前 'n' 行的 DataFrame。
    """
    try:
        parquet_file = pq.ParquetFile(file_path)

        # 使用 iter_batches() 来迭代读取记录批次
        # batch_size 参数可以控制每个批次的大小，但这里我们主要关注只读取最开始的批次
        rows_read = 0
        all_batches = []

        # 获取文件的总行数，如果文件行数小于 n，则直接读取所有行
        total_rows_in_file = parquet_file.metadata.num_rows

        if total_rows_in_file <= n:
            # 如果文件总行数小于或等于 n，则安全地读取整个文件
            # 这种情况下，其实和 parquet_file.read().to_pandas() 类似，但更明确
            print(f"文件总行数 ({total_rows_in_file}) 小于等于请求行数 ({n})，将读取所有行。")
            table = parquet_file.read()
            return table.to_pandas()

        # 否则，只读取前 n 行
        for batch in parquet_file.iter_batches():
            current_batch_df = batch.to_pandas()
            all_batches.append(current_batch_df)
            rows_read += len(current_batch_df)
            if rows_read >= n:
                break  # 达到或超过 n 行后停止读取

        # 将读取到的批次合并并取前 n 行
        combined_df = pd.concat(all_batches).head(n)
        print(f"成功高效地读取了 '{file_path}' 的前 {n} 行。")
        return combined_df

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        return None
    except Exception as e:
        print(f"读取 Parquet 文件时发生错误：{e}")
        return None


# --- 示例用法 ---
if __name__ == "__main__":

    large_dummy_file_path = '/home/chenyinbo/.cache/huggingface/hub/datasets--MMDocIR--MMDocIR_Train_Dataset/snapshots/059e7a30e87429698eaead28c30b1613dcf915c8/parquet/ArxivQA_filter.parquet'

    # 使用高效方法读取前 10 行
    first_rows_efficient = read_first_n_rows_efficiently(large_dummy_file_path, n=10)

    if first_rows_efficient is not None:
        print("\n高效读取的前 10 行数据：")
        print(first_rows_efficient)

    print("\n" + "=" * 30 + "\n")



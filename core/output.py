import json
import os


class FilesManager:

    def __init__(self, write_dir: str, file_tag: str):
        self.write_dir = write_dir
        self.file_tag = file_tag
        self.skip_file_path = os.path.join(write_dir, f"skip_{file_tag}.txt")
        self.gene_file_path = os.path.join(write_dir, f"gene_{file_tag}.txt")
        self.pred_file_path = os.path.join(write_dir, f"pred_{file_tag}.json")
        self.gnth_file_path = os.path.join(write_dir, f"gnth_{file_tag}.json")
        self.metric_file_path = os.path.join(write_dir, f"metric_{file_tag}.json")
        if not os.path.exists(self.write_dir):
            os.makedirs(self.write_dir)

    def read_skip_count(self) -> int:
        skip_qa_count = 0
        if os.path.exists(self.skip_file_path):
            with open(self.skip_file_path, "r", encoding="utf-8") as f:
                skip_qa_count = int(f.read())
        return skip_qa_count

    def write_skip_count(self, skip_qa_count: str):
        with open(self.skip_file_path, "w", encoding="utf-8") as f:
            f.write(str(skip_qa_count))

    def read_gene_file(self) -> iter:
        with open(self.gene_file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line

    def write_gene_line(self, data: dict):
        with open(self.gene_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def write_metric(self, score: dict):
        with open(self.metric_file_path, "w", encoding="utf-8") as f:
            json.dump(score, f, ensure_ascii=False)

import json
import os

import faiss


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


class IndexFileManager:

    def __init__(self, write_dir: str, indices_dir: str):
        self.write_dir = write_dir
        self.indices_dir = indices_dir

    def read_images_index(self, paper_id: str) -> faiss.Index:
        return faiss.read_index(os.path.join(self.write_dir, self.indices_dir, f"{paper_id}_images.faiss"))

    def write_images_index(self, paper_id: str, index: faiss.Index):
        faiss.write_index(index, os.path.join(self.write_dir, self.indices_dir, f"{paper_id}_images.faiss"))

    def read_texts_index(self, paper_id: str) -> faiss.Index:
        return faiss.read_index(os.path.join(self.write_dir, self.indices_dir, f"{paper_id}_texts.faiss"))

    def write_texts_index(self, paper_id: str, index: faiss.Index):
        faiss.write_index(index, os.path.join(self.write_dir, self.indices_dir, f"{paper_id}_texts.faiss"))

    def read_id_to_element(self, paper_id: str) -> dict:
        with open(os.path.join(self.write_dir, self.indices_dir, f"{paper_id}.json"), "r", encoding="utf-8") as f:
            id_to_element = json.load(f)
        return id_to_element

    def write_id_to_element(self, paper_id: str, id_to_element: dict):
        with open(os.path.join(self.write_dir, self.indices_dir, f"{paper_id}.json"), "w", encoding="utf-8") as f:
            json.dump(id_to_element, f, ensure_ascii=False, indent=4)


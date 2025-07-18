import json

from bert_score import BERTScorer
from loguru import logger
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocotools.coco import COCO

from core.conf import BERT_MODEL_DIR


def score_compute(pycocoeval_pred_file_path: str, pycocoeval_gt_file_path: str, metrics: list[str]) -> dict[str, float]:
    coco = COCO(pycocoeval_gt_file_path)
    coco_result = coco.loadRes(pycocoeval_pred_file_path)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()
    res = dict()
    for metric in metrics:
        if metric == "BERTScore":
            continue
        res[metric] = coco_eval.eval[metric]
    if "BERTScore" not in metrics:
        logger.info(f"score: {res}")
        return res

    with open(pycocoeval_pred_file_path, "r", encoding="utf-8") as f:
        pred = json.load(f)
    with open(pycocoeval_gt_file_path, "r", encoding="utf-8") as f:
        gt = json.load(f)["annotations"]
    assert len(pred) == len(gt)
    scorer = BERTScorer(model_type=BERT_MODEL_DIR, num_layers=9)
    ps, gs = [], []
    for p, g in zip(pred, gt):
        assert p["image_id"] == g["image_id"]
        ps.append(p["caption"])
        gs.append(g["caption"])
    _, _, sc = scorer.score(ps, gs)
    avg_bert_f1 = sc.mean().item()
    res["BERTScore"] = avg_bert_f1

    return res


def create_coco_eval_file(pred_path: str, gt_path: str, pred_answers: list[str], gt_answers: list[str]):
    assert len(pred_answers) == len(gt_answers)
    counter = 0
    pycocoeval_like_pred = []
    pycocoeval_like_gt = []
    for pred, gt in zip(pred_answers, gt_answers):
        pycocoeval_like_pred.append({"image_id": counter, "caption": pred})
        pycocoeval_like_gt.append({"image_id": counter, "id": counter, "caption": gt, "category_id": None})
        counter += 1
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pycocoeval_like_pred, f, ensure_ascii=False)
    images = []
    for gt in pycocoeval_like_gt:
        images.append({"id": gt["image_id"]})
    gt_write = dict(annotations=pycocoeval_like_gt, images=images, info={}, licenses=[], categories=[])
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_write, f, ensure_ascii=False)
    logger.info(f"create_coco_eval_file success")



class COCOEvalCap:
    def __init__(self, coco, coco_res):
        self.eval_imgs = []
        self.eval = {}
        self.img_to_eval = {}
        self.coco = coco
        self.coco_res = coco_res
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        img_ids = self.params['image_id']
        gts = {}
        res = {}
        for imgId in img_ids:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.coco_res.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        scorers = [
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.set_eval(sc, m)
                    self.set_img_to_eval_imgs(scs, gts.keys(), m)
            else:
                self.set_eval(score, method)
                self.set_img_to_eval_imgs(scores, gts.keys(), method)
        self.set_eval_imgs()

    def set_eval(self, score, method):
        self.eval[method] = score

    def set_img_to_eval_imgs(self, scores, img_ids, method):
        for imgId, score in zip(img_ids, scores):
            if not imgId in self.img_to_eval:
                self.img_to_eval[imgId] = {}
                self.img_to_eval[imgId]["image_id"] = imgId
            self.img_to_eval[imgId][method] = score

    def set_eval_imgs(self):
        self.eval_imgs = [eval_ for imgId, eval_ in self.img_to_eval.items()]

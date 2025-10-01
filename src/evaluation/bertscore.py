import evaluate
import numpy as np
from .base import BaseScoreEvaluator


class BertScoreEvaluator(BaseScoreEvaluator):
    fields = ["bert.p", "bert.r", "bert.f1"]

    def __init__(self, bert_model="roberta-large"):
        super().__init__(name="bertscore")
        self.scorer = evaluate.load("bertscore", keep_in_memory=True)
        self.bert_model = bert_model

    def score(self, dataset):
        res = self.scorer.compute(
            predictions=dataset["predictions"],
            references=dataset["references"],
            model_type=self.bert_model,
            lang="en",
        )
        results = {
            "bert.f1": res["f1"],
            "bert.p": res["precision"],
            "bert.r": res["recall"],
        }

        return results

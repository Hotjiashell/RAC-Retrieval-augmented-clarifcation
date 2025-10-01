import evaluate
import numpy as np
from .base import BaseScoreEvaluator


class BleuScoreEvaluator(BaseScoreEvaluator):
    fields = ["bleu"]

    def __init__(self):
        super().__init__("bleu")
        self.scorer = evaluate.load("sacrebleu", keep_in_memory=True)

    def score(self, dataset):
        results = {self.name: []}
        for pred, ref in zip(dataset["predictions"], dataset["references"]):
            score = self.scorer.compute(predictions=[pred], references=[ref])[
                "score"] / 100.0
            results[self.name].append(score)
        return results

# class BleuScoreEvaluator(BaseScoreEvaluator):
#     fields = ["bleu"]

#     def __init__(self):
#         super().__init__("bleu")
#         self.scorer = evaluate.load("sacrebleu", keep_in_memory=True)

#     def score(self, dataset):
#         results = {}
#         results[self.name] = (
#             self.scorer.compute(
#                 predictions=dataset["predictions"], references=dataset["references"]
#             )["score"]
#             / 100.0
#         )
#         return results

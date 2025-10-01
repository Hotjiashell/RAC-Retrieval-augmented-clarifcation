import evaluate
import numpy as np
from .base import BaseScoreEvaluator


class RougeScoreEvaluator(BaseScoreEvaluator):
    fields = ["rouge.rouge1", "rouge.rouge2", "rouge.rougeLsum", "rouge.meanrouge"]

    def __init__(self):
        super().__init__("rouge")
        self.scorer = evaluate.load("rouge",keep_in_memory=True)


    def score(self, dataset):
        results = self.scorer.compute(
            predictions=dataset["predictions"],
            references=dataset["references"],
            use_aggregator=False,
        )
        
        results["meanrouge"] = []
        for i in range(len(results["rouge1"])):
            results["meanrouge"].append(
                (results["rouge1"][i] + results["rouge2"][i] + results["rougeLsum"][i])
                / 3.0
            )

        results = {f"rouge.{k}": v for k, v in results.items()}

        return {k: results[k] for k in self.fields}


from .base import BaseScoreEvaluator
from .rouge import RougeScoreEvaluator
from .sacrebleu import BleuScoreEvaluator
from .meteor import MeteorScoreEvaluator
from .bertscore import BertScoreEvaluator
from .utils import get_scorers

import numpy as np
from scipy.stats import ttest_rel


class CollectionScoreEvaluator(BaseScoreEvaluator):
    def __init__(self, metrics_names, **kwargs):
        super().__init__("Collection")
        self.scorers = get_scorers(metrics_names, **kwargs)

    def score(self, dataset):
        scores = {}
        for scorer in self.scorers:
            scores.update(scorer.score(dataset))
        return scores

    def compute_mean(self, scores):
        # Handle case where scores is a list of dicts (e.g., max_scores_list)
        if isinstance(scores, list) and all(isinstance(s, dict) for s in scores):
            mean_scores = {}
            for metric in scores[0].keys():
                mean_scores[metric] = float(
                    np.mean([s[metric] for s in scores]))
            return mean_scores

        # Fallback to raw scorer outputs
        mean_scores = {}
        for scorer in self.scorers:
            mean_scores.update(scorer.compute_mean(scores))
        return mean_scores

    def ttest(self, scores_a, scores_b):
        """
        Perform paired t-tests and compute effect sizes between two sets
        of per-item scores (list of dicts, one dict per reference).
        """
        assert len(scores_a) == len(
            scores_b), "Both score lists must be same length"
        metrics = scores_a[0].keys()
        results = {}

        for metric in metrics:
            vals_a = np.array([s[metric] for s in scores_a])
            vals_b = np.array([s[metric] for s in scores_b])

            # Paired t-test
            t_stat, p_val = ttest_rel(vals_a, vals_b)

            # Effect size (Cohen's d for paired data)
            diff = vals_a - vals_b
            cohen_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan

            results[metric] = {
                "mean_a": float(vals_a.mean()),
                "mean_b": float(vals_b.mean()),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "cohen_d": float(cohen_d)
            }

        return results

# class CollectionScoreEvaluator(BaseScoreEvaluator):
#     def __init__(self, metrics_names, **kwargs):
#         super().__init__("Collection")
#         self.scorers = get_scorers(metrics_names, **kwargs)

#     def score(self, dataset):
#         scores = {}
#         for scorer in self.scorers:
#             scores.update(scorer.score(dataset))
#         return scores

#     def compute_mean(self, scores):
#         mean_scores = {}
#         for scorer in self.scorers:
#             mean_scores.update(scorer.compute_mean(scores))
#         return mean_scores

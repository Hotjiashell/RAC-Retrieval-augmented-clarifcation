from .bertscore import BertScoreEvaluator
from .meteor import MeteorScoreEvaluator
from .rouge import RougeScoreEvaluator
from .sacrebleu import BleuScoreEvaluator


def get_scorers(metrics_names, **kwargs):
    scorers = []
    for metric_name in metrics_names:
        if metric_name == "rouge":
            scorers.append(RougeScoreEvaluator(**kwargs))
        elif metric_name == "bleu":
            scorers.append(BleuScoreEvaluator(**kwargs))
        elif metric_name == "meteor":
            scorers.append(MeteorScoreEvaluator(**kwargs))
        elif metric_name == "bertscore":
            scorers.append(BertScoreEvaluator(**kwargs))
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    return scorers

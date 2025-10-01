# eval-LLM
A small framework to optimize coding and application of evaluation metrics for LLM generated content metrics against groundtruth, the goal is to apply this for different baselines, and add dynamically metrics that I want according to the goal of the evaluation and type of model


## Installation

```bash
pip install evaluate numpy torch
```

## Usage example

```python
from .collectionEvaluator import CollectionScoreEvaluator

# Your test dataset (prediction-reference pairs)
dataset = {
    "predictions": [
        "The cat sat on the mat.",
        "The quick brown fox."
    ],
    "references": [
        "The cat is sitting on the mat.",
        "A quick brown fox jumps over the lazy dog."
    ]
}

# Create evaluator
evaluator = CollectionScoreEvaluator(["rouge", "bleu", "meteor", "bertscore"])

# Compute all scores
scores = evaluator.score(dataset)

# Print raw results
print("All Scores:\n", scores)

# Compute mean score for each metric
means = evaluator.compute_mean(scores)
print("\nMean Scores:\n", means)

```

## Add new metrics

To add a new metric, simply create a subclass of `BaseScoreEvaluator`. Here's an example for a dummy metric:

```python
class DummyScoreEvaluator(BaseScoreEvaluator):
    def __init__(self):
        super().__init__("dummy")

    def score(self, dataset):
        self.scores = [1.0 for _ in dataset["predictions"]]
        return {"dummy": self.scores}
```

Then register it in `utils/get_scorers()`:

```python
elif metric == "dummy":
    scorers.append(DummyScoreEvaluator())
```
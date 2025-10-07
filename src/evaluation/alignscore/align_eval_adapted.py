
"""
AlignScore Evaluation Script for Clarifying Question Generation
---------------------------------------------------------------

Evaluates DPO or SFT model predictions using the AlignScore metric.
It cleans clarifying questions, aligns them with reference passages,
and reports omission/hallucination/faithfulness statistics.

Usage:
    python align_eval_adapted.py --source dpo --dataset path/to/data.json --ckpt_path path/to/AlignScore-large.ckpt
"""

import os
import re
import argparse
import torch
import pandas as pd
import spacy
from tqdm import tqdm
from alignscore import AlignScore


# ============================
# 🔹 Clarifying Question Cleaning
# ============================

QUESTION_PATTERNS = [
    r"^Are you looking(?: for| to| about)?\s*",
    r"^Can you please specify\s*",
    r"^Do you want to know\s*",
    r"^Would you like to know\s*",
    r"^What specific (information|aspect)\s*",
    r"This clarifying question.*",
    r"^Which (one|version|type|writer|role|year|season|party|symbol|location|time|grenade|ride|Weasley).*:?[\s]*",
    r"^In which [\w\s]+:?[\s]*",
    r"^As of [\w\s]+:?[\s]*",
    r"^For which [\w\s]+:?[\s]*",
    r"^In terms of what:?[\s]*",
]

def clean_question_text(text: str) -> str:
    """Remove interrogative templates and keep only content-bearing parts."""
    if not isinstance(text, str):
        return ""
    for pattern in QUESTION_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip(" ?")


def extract_content_from_question(lst, nlp):
    """Extract content tokens from a list containing one question string."""
    if not isinstance(lst, list) or not lst:
        return []
    text = clean_question_text(lst[0])
    doc = nlp(text)
    content_tokens = [
        token.text for token in doc
        if token.pos_ not in {"PRON", "DET", "AUX", "PART", "CCONJ", "SCONJ", "PUNCT"}
    ]
    return [" ".join(content_tokens)] if content_tokens else []


# ============================
# 🔹 Passage Cleaning
# ============================

def clean_passage_text(raw_text):
    """Clean passage formatting artifacts."""
    if not isinstance(raw_text, str):
        return raw_text
    if raw_text.startswith("'[") and raw_text.endswith("]'"):
        raw_text = raw_text[2:-2]
    matches = re.findall(r"'# (.*?)'(?:,|$)", raw_text)
    return " ".join(m.strip() for m in matches)


# ============================
# 🔹 AlignScore Scorer Class
# ============================

class SimpleAlignScorer:
    """Wrapper around AlignScore for hallucination and omission detection."""
    def __init__(self, model="roberta-large", ckpt_path=None, batch_size=8, device="cpu"):
        self.model = AlignScore(
            model=model,
            batch_size=batch_size,
            device=device,
            ckpt_path=ckpt_path,
            evaluation_mode="nli_sp",
        )

    def score(self, sources, predictions):
        """Compute AlignScore-based faithfulness metrics."""
        results = []
        for src, pred in zip(sources, predictions):
            srcs = [src] if isinstance(src, str) else src
            omission_scores = self.model.score(contexts=srcs, claims=[pred] * len(srcs))
            omission_probs = [float(s) for s in omission_scores]
            has_omission = any(s < 0.5 for s in omission_probs)
            merged_src = " ".join(srcs)
            halluc_score = float(self.model.score(contexts=[merged_src], claims=[pred])[0])
            has_hallucination = halluc_score < 0.5

            results.append({
                "prediction": pred,
                "align.omission_scores": omission_probs,
                "align.hallucination_score": halluc_score,
                "align.omission": int(has_omission),
                "align.hallucination": int(has_hallucination),
                "align.omission_hallucination": int(has_omission and has_hallucination),
                "align.ok": int(not has_omission and not has_hallucination),
                "align.score": min(omission_probs + [halluc_score])
            })
        return results


# ============================
# 🔹 Main Execution
# ============================

def main():
    parser = argparse.ArgumentParser(description="AlignScore evaluation for DPO or SFT predictions.")
    parser.add_argument("--source", choices=["dpo", "sft"], required=True,
                        help="Choose prediction source: 'dpo' or 'sft'")
    parser.add_argument("--dataset", required=True, help="Path to JSON dataset file (JSON Lines format)")
    parser.add_argument("--ckpt_path", required=True, help="Path to AlignScore checkpoint (.ckpt)")
    parser.add_argument("--output", default="results.csv", help="Path to save the evaluation results (CSV)")
    args = parser.parse_args()

    prediction_column = args.source if args.source != "sft" else "fine_tuned"
    print(f"[INFO] Using predictions from column: '{prediction_column}'")

    print("[INFO] Loading SpaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print("[INFO] Loading dataset...")
    df = pd.read_json(args.dataset, lines=True)

    if "Passage" not in df.columns or prediction_column not in df.columns:
        raise ValueError(f"Dataset must contain 'Passage' and '{prediction_column}' columns.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    scorer = SimpleAlignScorer(model="roberta-large", batch_size=64,
                               device=device, ckpt_path=args.ckpt_path)

    print("[INFO] Scoring examples...")
    all_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring rows"):
        ground_truth = row["Passage"]
        raw_preds = row[prediction_column]

        # Clean predictions
        if isinstance(raw_preds, list):
            preds = [clean_question_text(p) for p in raw_preds if clean_question_text(p).strip()]
        else:
            preds = [clean_question_text(raw_preds)]

        if not preds:
            continue

        scores = scorer.score([ground_truth] * len(preds), preds)
        for i, s in enumerate(scores):
            s.update({
                "example_id": idx,
                "ground_truth": ground_truth,
                "prediction_id": i,
            })
            all_rows.append(s)

    # Combine and compute averages
    scores_df = pd.DataFrame(all_rows)
    best_per_example = scores_df.loc[
        scores_df.groupby("example_id")["align.score"].idxmax()
    ].reset_index(drop=True)

    print("\n--- AlignScore Averages (Best Prediction per Example) ---")
    print(best_per_example[["align.omission", "align.hallucination", "align.ok", "align.score"]].mean())

    print(f"\n[INFO] Saving detailed results to: {args.output}")
    scores_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

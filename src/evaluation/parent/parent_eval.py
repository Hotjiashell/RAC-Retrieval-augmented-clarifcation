import re
import spacy
import pandas as pd
from nltk.tokenize import word_tokenize
from typing import List, Set, Dict, Any
from datasets import Dataset
from parent import parent  # assumes PARENT metric implementation is available


# Load spaCy model once
nlp = spacy.load("en_core_web_sm")


def extract_entities_noun_chunks(text: str) -> Set[str]:
    """Extract named entities, noun chunks, and simple subject-verb-object facts."""
    doc = nlp(text)
    extracted = set()

    # Named entities
    for ent in doc.ents:
        if ent.label_ in [
            "PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY",
            "EVENT", "NORP", "CARDINAL", "ORDINAL"
        ]:
            extracted.add(ent.text.strip())

    # Noun chunks
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        if len(chunk_text.split()) > 1:
            extracted.add(chunk_text)

    # Subject–verb–object triples
    for sent in doc.sents:
        subject, verb, obj = "", "", ""
        for token in sent:
            if "subj" in token.dep_:
                subject = token.text
            if token.pos_ == "VERB":
                verb = token.lemma_
            if "obj" in token.dep_:
                obj = token.text
        if subject and verb and obj:
            extracted.add(f"{subject} {verb} {obj}")

    # Regex-based year capture
    for year, suffix in re.findall(r"\b(1[5-9]\d{2}|20\d{2}|2100)(s?)\b", text):
        extracted.add(year + suffix)

    return extracted


class ParentScorer:
    """Wrapper for computing PARENT metric on text datasets."""
    fields = ["parent.p", "parent.r", "parent.f1"]

    def __init__(self, lower: bool = False):
        self.lower = lower
        self.empty_examples = []

    def format_parent_text(self, dataset) -> List[List]:
        table = []
        for i, (topic, passages) in enumerate(zip(dataset["topic"], dataset["Passage"])):
            combined = f"{topic} {' '.join(passages)}"
            facts = extract_entities_noun_chunks(combined)
            res = []
            for fact in facts:
                tokens = word_tokenize(fact.lower() if self.lower else fact)
                res.append((["fact"], tokens))
            if not res:
                self.empty_examples.append({"index": i, "topic": topic})
                res.append((["fact"], ["dummy"]))
            table.append(res)
        return table

    def score(self, dataset) -> Dict[str, Any]:
        predictions = [word_tokenize(t + p[0]) for t, p in zip(dataset["topic"], dataset["prediction"])]

        references = [
            [word_tokenize(ref) for ref in subref]
            for subref in dataset.get("references", [[""] * len(dataset)])
        ]

        sources = self.format_parent_text(dataset)

        parent_p, parent_r, parent_f1 = parent(
            predictions, references, sources, avg_results=False,
            use_tqdm=True, n_jobs=-1, lambda_weight=1.0,
        )

        return {"parent.p": parent_p, "parent.r": parent_r, "parent.f1": parent_f1}


def load_input_json(path: str, source: str) -> pd.DataFrame:
    """
    Load a JSON dataset with 'ground-truth', 'Passage', and prediction fields ('dpo' or 'sft').
    If source is 'sft', rename the internal column to 'fine_tuned' for clarity.
    """
    df = pd.read_json(path, lines=False)
    df["topic"] = df["ground-truth"]

    if source not in df.columns:
        raise ValueError(f"Source '{source}' not found in JSON. Available columns: {list(df.columns)}")

    # Rename for internal consistency
    if source == "sft":
        df["fine_tuned"] = df["sft"]
        df["prediction"] = df["fine_tuned"]
    else:
        df["prediction"] = df[source]

    return df


def main(dataset_path: str, source: str):
    print(f"Loading dataset from {dataset_path} (source={source})...")
    df = load_input_json(dataset_path, source)

    dataset = Dataset.from_dict(df)
    scorer = ParentScorer(lower=True)
    results = scorer.score(dataset)

    mean_p = sum(results["parent.p"]) / len(results["parent.p"])
    mean_r = sum(results["parent.r"]) / len(results["parent.r"])
    mean_f1 = sum(results["parent.f1"]) / len(results["parent.f1"])

    print("\n=== PARENT Metric Results ===")
    print(f"Precision (mean): {mean_p:.4f}")
    print(f"Recall    (mean): {mean_r:.4f}")
    print(f"F1        (mean): {mean_f1:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute PARENT metric for QA datasets.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input dataset JSON file.")
    parser.add_argument("--source", type=str, default="dpo", choices=["dpo", "sft"],
                        help="Select which prediction field to use: 'dpo' or 'sft'. If 'sft', uses internal name 'fine_tuned'.")
    args = parser.parse_args()
    main(args.dataset, args.source)

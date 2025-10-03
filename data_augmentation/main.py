import argparse
import yaml
import os

from indexing import passage_chunking, sparse_indexing
from retrieval import sparse_retrieval, dense_retrieval


def main():
    parser = argparse.ArgumentParser(description="Indexing & Retrieval Pipeline")
    parser.add_argument(
        "--stage", type=str, required=True,
        choices=["passages", "sparse-index", "dense-index", "sparse-retrieval", "dense-retrieval"],
        help="Which stage of the pipeline to run"
    )
    args = parser.parse_args()

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["settings"]["dataset_name"]
    dataset_mode = config["settings"]["dataset_mode"]
    level_type   = config["settings"]["level_type"]
    index_type   = config["settings"]["indexation_type"]
    base_dir     = config["paths"]["base_dir"]

    # =========================
    # Passage Chunking
    # =========================
    if args.stage == "passages":
        from indexing import retrieve_relevant_docs  # lazy import if heavy
        from data_loader import load_datasets, load_relevance
        from preprocessing import clean_datasets

        datasets = clean_datasets(load_datasets(config))
        relevance = load_relevance(config)

        results_df = retrieve_relevant_docs(
            datasets[dataset_mode],
            dataset_name,
            relevance[f"{dataset_mode}_relevance"],
            level_type,
            config
        )
        passage_chunking(results_df, dataset_name, dataset_mode, level_type, index_type, base_dir)
        print("✅ Passages generated.")

    # =========================
    # Sparse Indexing
    # =========================
    elif args.stage == "sparse-index":
        from data_loader import load_passage_results
        results_df = load_passage_results(dataset_name, dataset_mode, level_type, base_dir)
        sparse_indexing(results_df, dataset_name, dataset_mode, level_type, index_type, base_dir)
        print("✅ Sparse indexes built.")

    # =========================
    # Dense Indexing
    # =========================
    elif args.stage == "dense-index":
        from indexing import build_dense_index
        build_dense_index(dataset_name, dataset_mode, level_type, index_type, base_dir)
        print("✅ Dense indexes built.")

    # =========================
    # Sparse Retrieval
    # =========================
    elif args.stage == "sparse-retrieval":
        from data_loader import load_datasets
        from preprocessing import clean_datasets
        datasets = clean_datasets(load_datasets(config))
        sparse_retrieval(datasets[dataset_mode], dataset_name, dataset_mode, level_type, index_type, base_dir, top_k=100)

    # =========================
    # Dense Retrieval
    # =========================
    elif args.stage == "dense-retrieval":
        from data_loader import load_datasets
        from preprocessing import clean_datasets
        datasets = clean_datasets(load_datasets(config))
        dense_retrieval(datasets[dataset_mode], dataset_name, dataset_mode, level_type, index_type, base_dir, top_k=5)


if __name__ == "__main__":
    main()


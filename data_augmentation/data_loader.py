import pandas as pd
import os

def safe_read(path, func, **kwargs):
    try:
        return func(path, **kwargs)
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return pd.DataFrame()

def load_datasets(config):
    base = config["datasets"]["base_dir"]
    datasets = {
        "Qulac": safe_read(os.path.join(base, config["datasets"]["qulac"]), pd.read_json),
        "ClariQ_train": safe_read(os.path.join(base, config["datasets"]["clariq_train"]), pd.read_json),
        "ClariQ_dev": safe_read(os.path.join(base, config["datasets"]["clariq_dev"]), pd.read_json),
        "ClariQfkw_train": safe_read(os.path.join(base, config["datasets"]["clariqfkw_train"]), pd.read_json),
        "ClariQfkw_dev": safe_read(os.path.join(base, config["datasets"]["clariqfkw_dev"]), pd.read_json)
    }
    return datasets

def load_relevance(config):
    base = config["datasets"]["base_dir"]
    rel = {
        "ClariQ_train": pd.read_csv(os.path.join(base, config["relevance_files"]["clariq_train"]), sep=" ", names=["facet_id", "zero", "doc_id", "relevance"]),
        "ClariQ_dev": pd.read_csv(os.path.join(base, config["relevance_files"]["clariq_dev"]), sep=" ", names=["facet_id", "zero", "doc_id", "relevance"]),
        "Qulac": pd.read_csv(os.path.join(base, config["relevance_files"]["qulac"]), sep=" ", names=["facet_id", "zero", "doc_id", "relevance"])
    }
    return rel

# 🔍 Qulac and ClariQ augmentation 

This framework provides a modular pipeline for dataset preparation, passage segmentation, indexing, and retrieval using Pyserini.

It supports:

✅ Multiple datasets (ClariQ, Qulac, ClariQ-FKw, etc.)

✅ Sparse retrieval (Lucene BM25)

✅ Dense retrieval (FAISS + ColBERT/TCT-ColBERT encoder)

✅ Configurable pipeline via a single config.yaml


## ⚙️ Installation

```bash
cd query-retrieval-framework
pip install -r requirements.txt

#Install Pyserini (required for indexing & retrieval):

pip install pyserini


# Install Java (for Lucene indexing):

sudo apt install openjdk-21-jdk
```

## 📝 Configuration (config.yaml)

All dataset paths, environment variables, and pipeline settings are stored in config.yaml.

Example:

```yaml
environment:
  CUDA_DEVICE_ORDER: "PCI_BUS_ID"
  CUDA_VISIBLE_DEVICES: "0,1,2"
  HTTP_PROXY: "192.168.0.100:3128"
  HTTPS_PROXY: "192.168.0.100:3128"
  HF_HOME: "/data/user/.cache/"
  JAVA_HOME: "~/java/jdk21"

datasets:
  base_dir: "/home/user/datasets"
  qulac: "query_rewriting_outputs/Qulac_rewritten.json"
  clariq_train: "query_rewriting_outputs/ClariQ_train_rewritten.json"
  clariq_dev: "query_rewriting_outputs/ClariQ_dev_rewritten.json"
  clariqfkw_train: "query_rewriting_outputs/ClariQfkw_train_rewritten.json"
  clariqfkw_dev: "query_rewriting_outputs/ClariQfkw_dev_rewritten.json"

relevance_files:
  clariq_train: "ClariQ/data/train.qrel"
  clariq_dev: "ClariQ/data/dev.qrel"
  qulac: "qulac/data/qulac/faceted.qrel"

settings:
  dataset_name: "ClariQn"
  dataset_mode: "train"
  level_type: "full"       # options: full, level 1, level 2, level 3, level 4
  indexation_type: "clustered"   # flat / clustered
  retrieval_type: "dense"        # sparse / dense
  duplicate: false
```

## 🚀 Usage

Run the pipeline from main.py.

1. Environment Setup + Data Loading
python main.py --stage setup



2. Passage chunking
```python
python main.py --stage passages
python main.py --stage sparse-index
python main.py --stage sparse-retrieval
```

3.  Passage Indexing


Sparse Indexing
python main.py --stage sparse-index

Dense Indexing
python main.py --stage dense-index


4. Retrieval
Dense Retrieval
python main.py --stage dense-retrieval

Sparse Retrieval
python main.py --stage sparse-retrieval


Results are stored in:

JSONL outputs with top-k retrieved passages

📊 Example Output (Dense Retrieval)

Each entry in the results JSONL contains:
```json
{
  "topic_id": "123",
  "question": "What is the capital of France?",
  "rewritten_query": "capital city of France",
  "docid": ["DOC123", "DOC456"],
  "Passage": ["Paris is the capital of France...", "France's largest city is Paris..."]
}
```
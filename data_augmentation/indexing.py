import os
import json
import pandas as pd
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
import subprocess
from doc_split import segment_document
from utils import clean_wikipedia_text


def passage_chunking(results, dataset_name, dataset_mode, level_type, indexation_type):
    """Split retrieved documents into passages and save to JSONL files."""
    base_dir = f"/data/user/passages/{dataset_name}/{dataset_mode}/{level_type}"
    os.makedirs(base_dir, exist_ok=True)

    for facet_id in tqdm(results.get('facet_id', [])[:], desc="Processing Facets"):
        if indexation_type == "clustered":
            output_file = f"{base_dir}/{facet_id}/facet_passages_overlap_window_text.jsonl"
        else:
            output_file = f"{base_dir}/FLAT/facet_passages_overlap_window_text_{facet_id}.jsonl"

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "a", encoding="utf-8") as file:
            retrieved_docs = results.loc[results['facet_id'] == facet_id, 'retrieved_docs'].values
            if len(retrieved_docs) > 0:
                for doc in retrieved_docs[0]:
                    try:
                        content = json.loads(doc.raw()).get('contents', '')
                        content = clean_wikipedia_text(content)
                        chunks = segment_document(content)
                        for i, chunk in enumerate(chunks):
                            passage = {
                                'id': f"{doc.docid()}_P_{i}",
                                'text': chunk,
                                'contents': chunk,
                                'facet_id': str(facet_id)
                            }
                            file.write(json.dumps(passage) + "\n")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON for facet_id {facet_id}")


def sparse_indexing(results, dataset_name, dataset_mode, level_type, indexation_type):
    """Build Lucene (sparse) indexes from passage files."""
    for facet_id in tqdm(results.get('facet_id', [])[:], desc="Indexing Facets"):
        out_dir = f"/data/user/indexes/SPARSE/{dataset_name}/{dataset_mode}/{level_type}/{indexation_type}/{facet_id}"
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", f"/data/user/passages/{dataset_name}/{dataset_mode}/{level_type}/{facet_id}/",
            "--index", out_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]
        subprocess.run(cmd)


def retrieve_relevant_docs(dataset, dataset_name, dataset_relev_judgement, level_type, config, duplicate=False):
    """Retrieve relevant docs from indexes specified in config.yaml."""

    # Read index paths from config
    if dataset_name.lower() == "qulac":
        index_paths = config["indexes"]["qulac"]
        facet_id_col = "topic_facet_id"
    else:
        index_paths = config["indexes"]["default"]
        facet_id_col = "topic_id"

    searchers = [LuceneSearcher(path) for path in index_paths]
    results = []

    for topic_id in dataset['topic_id'].unique():
        local_topic = dataset[dataset['topic_id'] == topic_id]
        num_relev = 0
        Docs = {}

        for facet_id in local_topic[facet_id_col].unique():
            relev_docs = dataset_relev_judgement.loc[dataset_relev_judgement['facet_id'] == facet_id]

            # Select relevant documents by level
            if level_type == "full":
                relev_docs = relev_docs.loc[relev_docs['relevance'].isin([1, 2, 3, 4])]
            else:
                lvl = int(level_type.split()[-1])  # e.g. "level 2" -> 2
                relev_docs = relev_docs.loc[relev_docs['relevance'] == lvl]

            relevant_doc_ids = relev_docs['doc_id'].values
            num_relevant_docs = len(relevant_doc_ids)
            num_relev += num_relevant_docs
            print(f"Facet {facet_id}: {num_relevant_docs} relevant docs")

            seen_contents = set()
            num_retrieved = 0
            for doc_id in relevant_doc_ids:
                for searcher in searchers:
                    doc = searcher.doc(doc_id)
                    if doc is not None:
                        try:
                            content = json.loads(doc.raw()).get('contents', '')
                        except Exception:
                            content = doc.raw()
                        if not duplicate:
                            content_hash = hash(content)
                            if content_hash not in seen_contents:
                                num_retrieved += 1
                                Docs[doc_id] = doc
                                seen_contents.add(content_hash)
                        else:
                            num_retrieved += 1
                            Docs[doc_id] = doc

        results.append((topic_id, num_relev, num_retrieved, Docs))

    results = list(map(lambda x: (x[0], x[1], x[2], list(x[3].values())), results))
    return pd.DataFrame(results, columns=['facet_id', 'num_relevant_docs', 'num_retrieved_relevant_docs', 'retrieved_docs'])

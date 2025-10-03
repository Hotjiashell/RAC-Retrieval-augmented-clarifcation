import os
import json
from tqdm import tqdm
from collections import Counter
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from pyserini.encode import TctColBertQueryEncoder


def dense_retrieval(dataset, dataset_name, dataset_mode, level_type, indexation_type, top_k=5):
    """Run dense retrieval with FAISS and write results to JSONL + run file."""

    run_file_path = f"{dataset_name}_{dataset_mode}_{level_type}_dense_{indexation_type}_top{top_k}.run"
    output_file = f"./{dataset_name}_DENSE_{dataset_mode}_{level_type}_{indexation_type}_top{top_k}.jsonl"

    encoder = TctColBertQueryEncoder('castorini/tct_colbert-v2-hnp-msmarco')
    unique_results = set()

    with open(run_file_path, "w") as run_file:
        for idx, topic_id in enumerate(dataset['topic_id'].unique()):
            topic = dataset[dataset['topic_id'] == topic_id]['initial_request'].values[0]
            local_clar = dataset[dataset['topic_id'] == topic_id]
            unique_questions = local_clar[['rewritten_query', 'question']].drop_duplicates()

            if indexation_type == "flat":
                index_path = f"/data/user/indexes/SPARSE/{dataset_name}/{dataset_mode}/{level_type}/{indexation_type}/"
                searcher = FaissSearcher(
                    f"/data/user/indexes/DENSE/{dataset_name}/{dataset_mode}/{level_type}/{indexation_type}/",
                    encoder
                )
            else:  # clustered
                index_path = f"/data/user/indexes/SPARSE/{dataset_name}/{dataset_mode}/{level_type}/{indexation_type}/{topic_id}/"
                searcher = FaissSearcher(
                    f"/data/user/indexes/DENSE/{dataset_name}/{indexation_type}/{topic_id}/",
                    encoder
                )

            searcher_doc = LuceneSearcher(index_path)

            for row in tqdm(unique_questions.itertuples(index=False), desc=f"Dense topic {topic_id}"):
                rewritten_query = row.rewritten_query
                question = row.question

                hits = searcher.search(rewritten_query, k=top_k)

                topic_content, topic_hits, topic_hits_scores = [], [], []
                for rank, hit in enumerate(hits):
                    content = json.loads(searcher_doc.doc(hit.docid).raw()).get('contents', '')
                    topic_content.append(content)
                    topic_hits.append(hit.docid)
                    topic_hits_scores.append(hit.score)

                    # also log in TREC run format
                    run_file.write(f"{topic_id} Q0 {hit.docid} {rank+1} {hit.score:.4f} dense\n")

                unique_results.add(
                    (topic_id, topic, question, rewritten_query,
                     tuple(topic_hits), tuple(topic_hits_scores), tuple(topic_content))
                )

    # write jsonl
    with open(output_file, "w", encoding="utf-8") as file:
        for topic_id, topic, question, rewritten_query, docid, score, content in unique_results:
            if content:
                file.write(json.dumps({
                    "topic_id": str(topic_id),

from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from dotenv import load_dotenv
from generation.decoding import MixtureDecoder
from data_utils import (
    PassageClarificationPromptFormatter,
    FacetClarificationPromptFormatter,
    DPOPassagePrefencePromptFormatter,
)
from generation.decoding import MixtureDecoder
from generation.parallel_gen import extract_response
from evaluation.collectionEvaluator import CollectionScoreEvaluator
from utils import setup_env, is_ampere_gpu

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset
from datasets import DatasetDict
import json
import os
import numpy as np
import pandas as pd
from data_utils import PassageClarificationPromptFormatter
import logging
import gc
import pickle


# def generate_clarifications_batch(
#     batch_dataset,
#     tokenizer,
#     finetuned_model
# ):
#     """
#     Generate clarifications for a batch of examples using batch inference.
#     """
#     generation_config = GenerationConfig(
#         do_sample=False,
#         max_new_tokens=35,
#         repetition_penalty=1.0,
#         num_return_sequences=1,
#         use_cache=True,
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     clarifications_list = []

#     convs = batch_dataset["conv"]
#     weak_convs = batch_dataset["weak_conv"]
#     topics = batch_dataset["topic"]
#     passages = batch_dataset["Passage"]
#     ground_truths = batch_dataset["question"]

#     input_texts = tokenizer.apply_chat_template(
#         convs, tokenize=False, add_generation_prompt=True
#     )

#     input_encodings = tokenizer(
#         input_texts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#     ).to(device)
#     fine_tuned_output = finetuned_model.generate(
#         input_ids=input_encodings["input_ids"],
#         attention_mask=input_encodings["attention_mask"],
#         generation_config=generation_config,
#     )

#     decoded_fine_tuned_output = tokenizer.batch_decode(
#         fine_tuned_output,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=True,
#     )
#     for i in range(len(convs)):
#         clarification = {
#             "ground-truth": [ground_truths[i]],
#             "fine_tuned": [],
#         }

#         prompt = {
#             "topic": [topics[i]],
#             "Passage": [passages[i]],
#         }
#         clarification["fine_tuned"].append([
#             extract_response(decoded_fine_tuned_output[i])]
#         )

#         clarifications_list.append({
#             "clarifications": clarification,
#             "prompt": prompt
#         })

#     return clarifications_list


def generate_clarifications_batch(
    batch_dataset,
    tokenizer,
    finetuned_model,
    num_variants=1, chat=True,max_new_tokens=32
):
    """
    Generate clarifications for a batch of examples using batch inference,
    producing multiple variant generations per example.
    """

    generation_config = GenerationConfig(
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.0,
        num_return_sequences=1,  # We handle repetition manually
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clarifications_list = []

    convs = batch_dataset["conv"]
    topics = batch_dataset["topic"]
    passages = batch_dataset["Passage"]
    ground_truths = batch_dataset["question"]

    if chat:
        input_texts = tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True
        )
    else:
        input_texts = convs

    input_encodings = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    # Repeat inputs for num_variants generations per example
    input_ids = input_encodings["input_ids"].repeat_interleave(
        num_variants, dim=0)
    attention_mask = input_encodings["attention_mask"].repeat_interleave(
        num_variants, dim=0)
    with torch.inference_mode():
        fine_tuned_output = finetuned_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

        decoded_fine_tuned_output = tokenizer.batch_decode(
            fine_tuned_output,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        # Group outputs back into list of predictions per example
        for i in range(len(convs)):
            start_idx = i * num_variants
            end_idx = start_idx + num_variants
            predictions = [extract_response(decoded_fine_tuned_output[j], skip_special_tokens=False, chat=chat)
                           for j in range(start_idx, end_idx)
                           ]

            clarification = {
                "ground-truth": [ground_truths[i]],
                "fine_tuned": predictions,  # list[str]
            }

            prompt = {
                "topic": [topics[i]],
                "Passage": [passages[i]],
            }

            clarifications_list.append({
                "clarifications": clarification,
                "prompt": prompt
            })

    return clarifications_list


def evaluate_metrics(
    cfg,
    finetuned_model,
    tokenizer,
    dataset_path,
    train_test_split=None,
):
    """
    Evaluate the model using the given metrics
    """
    result_df = pd.read_json(dataset_path, lines=True, orient="records")
    data_formatter = PassageClarificationPromptFormatter(
        response_template=cfg.model.response_template
    )
    if cfg.model.chat:
        result_df = result_df.apply(data_formatter.format_dataset, axis=1)
    else:
        result_df = result_df.apply(
            data_formatter.format_dataset_no_chat, axis=1)
    dataset = Dataset.from_pandas(
        result_df[["conv", "weak_conv", "question", "topic", "Passage"]]
    )

    test_dataset = dataset

    # output_path = "sft_" + cfg.dataset.preference_dataset
    output_path = cfg.dataset.full_sft_eval_dataset

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    clarifications_all = []
    batch_size = cfg.mixture.bs

    num_samples = len(test_dataset)
    print(test_dataset[0])
    with open(output_path, "w") as f:
        pass

    for start in tqdm(range(0, num_samples, batch_size), desc="Generating clarifications (batched)"):
        # for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch = test_dataset.select(range(start, end))

        clarifications_batch = generate_clarifications_batch(
            batch,
            tokenizer=tokenizer,
            finetuned_model=finetuned_model,
            num_variants=cfg.mixture.num_variants,
            chat=cfg.model.chat,
            max_new_tokens=cfg.run.max_new_tokens
        )

        # clarifications_all.extend(clarifications_batch)

        if cfg.mode == "generate-full-sft":
            clarifications_all = []
            for clarification in clarifications_batch:
                clarification = clarification["clarifications"]
                ground_truth = clarification["ground-truth"][0]
                fine_tuned = clarification["fine_tuned"]

                clarifications_all.append(
                    {
                        "ground-truth": ground_truth,
                        "fine_tuned": fine_tuned,
                    }
                )

            with open(output_path, "a") as f:
                for entry in clarifications_all:
                    f.write(json.dumps(entry) + "\n")

        else:
            logging.info(f"Please select the correct mode")
            return None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return None


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to run the script
    """
    # Setup environment
    setup_env(cfg)

    # Load model and tokenizer
    if cfg.mode == "generate-full-sft":
        cfg.model.load_directory = cfg.model.load_directory + "_full_sft"

    evaluator = CollectionScoreEvaluator(
        ["rouge", "bleu", "meteor", "bertscore"])
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.load_directory)
    tokenizer.padding_side = "left"
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.load_directory,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({"pad_token": cfg.model.pad_token})
    #     tokenizer.add_special_tokens(
    #         {"additional_special_tokens": [cfg.model.response_template]}
    #     )
    #     finetuned_model.resize_token_embeddings(len(tokenizer))
    finetuned_model.eval()
    mean_scores = evaluate_metrics(
        cfg,
        finetuned_model,
        tokenizer,
        cfg.dataset.eval_dataset,
        train_test_split=True,
    )

    dataset_path = cfg.dataset.full_sft_eval_dataset
    # dataset_path = "sft_" + cfg.dataset.preference_dataset
    cfg.dataset.metrics = "sft_" + cfg.dataset.metrics

    # Load dataset
    dataset = pd.read_json(dataset_path, lines=True, orient="records")

    # Evaluate
    # comparison_results = {}
    # file_results = {}
    # for column in dataset.columns[1:]:
    #     print(f"Processing column: {column}")
    #     references = dataset.iloc[:, 0].tolist()
    #     predictions = dataset[column].tolist()
    #     data_dict = {"references": references, "predictions": predictions}
    #     results = evaluator.score(data_dict)
    #     means = evaluator.compute_mean(results)
    #     file_results[column] = means

    # # Organize results into DataFrame
    # organized_results = []
    # for column, metrics in file_results.items():
    #     row = {"File": os.path.basename(dataset_path), "Column": column}
    #     row.update(metrics)
    #     organized_results.append(row)

    # results_df = pd.DataFrame(organized_results)

    # print(results_df)
    # for file, columns in comparison_results.items():
    #     for column, metrics in columns.items():
    #         row = {"File": file, "Column": column}
    #         row.update(metrics)
    #         organized_results.append(row)

    # results_df = pd.DataFrame(organized_results)

    # print(results_df)

    # os.makedirs(os.path.dirname(cfg.dataset.metrics), exist_ok=True)

    # with open(f"{cfg.dataset.metrics}.pkl", "wb") as f:
    #     pickle.dump(results_df, f)

    # with open(f"{cfg.dataset.metrics}.txt", "w") as f:
    #     f.write(results_df.to_string(index=False))

    max_scores_list = []

    for _, row in dataset.iterrows():
        reference = row["ground-truth"]
        predictions = row["fine_tuned"]

        # Build sub-dataset with each prediction paired to the same reference
        sub_dataset = {
            "predictions": predictions,
            "references": [reference] * len(predictions)
        }

        # Get scores for each prediction
        scores = evaluator.score(sub_dataset)  # dict: metric -> list of scores

        # Take the max score per metric for this item
        max_scores = {metric: max(score_list)
                      for metric, score_list in scores.items()}
        max_scores_list.append(max_scores)

    # Compute mean of the max scores across all items
    mean_scores = evaluator.compute_mean(max_scores_list)

    # Organize results into DataFrame
    results_df = pd.DataFrame([mean_scores])
    results_df.insert(0, "File", os.path.basename(dataset_path))
    results_df.insert(1, "Column", "max_candidates")

    print("\nFinal Results:\n", results_df)

    # Save results
    os.makedirs(os.path.dirname(cfg.dataset.metrics), exist_ok=True)

    # with open(f"{cfg.dataset.metrics}.pkl", "wb") as f:
    #     pickle.dump(results_df, f)

    with open(f"{cfg.dataset.metrics}.txt", "w") as f:
        f.write(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

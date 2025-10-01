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
from generation.parallel_gen import extract_response
import gc


def generate_clarifications_batch(
    batch_dataset,
    model,
    tokenizer,
    finetuned_model,
    mixture_decoder,
    dpo_model,
    num_variants=1,
    chat=True,
    max_new_tokens=32
):
    """
    Generate clarifications for a batch of examples using batch inference.
    """
   # Number of candidate generations per input

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
    weak_convs = batch_dataset["weak_conv"]
    topics = batch_dataset["topic"]
    passages = batch_dataset["Passage"]
    ground_truths = batch_dataset["question"]

    # input_texts = tokenizer.apply_chat_template(
    #     convs, tokenize=False, add_generation_prompt=True
    # )

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
        if dpo_model is not None:
            dpo_output = dpo_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
            decoded_dpo_output = tokenizer.batch_decode(
                dpo_output,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

        for i in range(len(convs)):
            start_idx = i * num_variants
            end_idx = start_idx + num_variants
            predictions = [
                extract_response(decoded_dpo_output[j],
                                 skip_special_tokens=False, chat=chat)
                for j in range(start_idx, end_idx)
            ]
            clarification = {
                "ground-truth": [ground_truths[i]],
                "dpo": predictions,
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
    model,
    tokenizer,
    dataset_path,
    dpo_model=None,
    mixture=None,
    train_test_split=None,
):
    """
    Evaluate the model using the given metrics
    """
    # rouge = evaluate.load("rouge")
    # bleu = evaluate.load("bleu")
    # exact_match = evaluate.load("exact_match")
    # meteor = evaluate.load("meteor")
    # bertscore = evaluate.load("bertscore")

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
        result_df[["conv", "weak_conv", "question", "topic", "Passage"]])
    # test_dataset = dataset

    if train_test_split is not None:
        train_test_split = dataset.train_test_split(
            test_size=cfg.dataset.train_test_split, seed=cfg.dataset.seed
        )
        test_dataset = train_test_split["test"]

        print(test_dataset[0])
    else:
        test_dataset = dataset
    mixture_decoder = None
    if mixture is not None:
        pass
        # mixture_decoder = MixtureDecoder(
        #     model=finetuned_model,
        #     unconditional_model=model,
        #     mixture_alpha=cfg.mixture.alpha,
        #     mixture_mode="hard",
        # )

    if dpo_model is not None:
        output_path = cfg.dataset.dpo_dataset
    else:
        output_path = cfg.dataset.preference_dataset

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    clarifications_all = []
    batch_size = cfg.mixture.dpo_bs
    test_dataset = test_dataset
    # test_dataset = test_dataset.select(range(20))
    num_samples = len(test_dataset)
    print(test_dataset[0])
    with open(output_path, "w") as f:
        pass
    # for start in range(0, num_samples, batch_size):
    #     end = min(start + batch_size, num_samples)
    #     batch = test_dataset.select(range(start, end))

    #     clarifications_batch = generate_clarifications_parallel(
    #         batch,
    #         model=model,
    #         tokenizer=tokenizer,
    #         finetuned_model=finetuned_model,
    #         dpo_model=dpo_model,
    #         mixture_decoder=mixture_decoder,
    #         num_workers=20,
    #     )
    for start in tqdm(range(0, num_samples, batch_size), desc="Generating clarifications (batched)"):
        # for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch = test_dataset.select(range(start, end))

        clarifications_batch = generate_clarifications_batch(
            batch,
            model=model,
            tokenizer=tokenizer,
            finetuned_model=finetuned_model,
            dpo_model=dpo_model,
            mixture_decoder=mixture_decoder,
            num_variants=cfg.mixture.num_variants,
            chat=cfg.model.chat,
            max_new_tokens=cfg.run.max_new_tokens
        )

        # clarifications_all.extend(clarifications_batch)

        if cfg.mode == "generate":

            preference_data = []

            for clarification in clarifications_batch:

                topic = clarification["prompt"]["topic"][0]
                Passage = clarification["prompt"]["Passage"][0]
                clarification = clarification["clarifications"]
                ground_truth = clarification["ground-truth"][0]
                fine_tuned = clarification["fine_tuned"][0]
                chosen = clarification["ground-truth"][0]
                rejected = clarification["mixture"][0]

                preference_data.append(
                    {
                        "topic": topic,
                        "Passage": Passage,
                        "ground-truth": ground_truth,
                        "fine_tuned": fine_tuned,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                )

            with open(output_path, "a") as f:
                for entry in preference_data:
                    f.write(json.dumps(entry) + "\n")

        elif cfg.mode == "generate-dpo":
            clarifications_all = []
            for clarification in clarifications_batch:
                clarification = clarification["clarifications"]
                ground_truth = clarification["ground-truth"][0]
                dpo = clarification["dpo"]

                clarifications_all.append(
                    {
                        "ground-truth": ground_truth,
                        "dpo": dpo,
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
    tokenizer = AutoTokenizer.from_pretrained(cfg.dpo.dpo_directory)
    tokenizer.padding_side = "left"

    # model = AutoModelForCausalLM.from_pretrained(
    #     cfg.model.name,
    #     device_map="balanced",
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    # )
    # finetuned_model = AutoModelForCausalLM.from_pretrained(
    #     cfg.model.load_directory,
    #     device_map="balanced",
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    # )

    dpo_model = AutoModelForCausalLM.from_pretrained(
        cfg.dpo.dpo_directory,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({"pad_token": cfg.model.pad_token})
    #     tokenizer.add_special_tokens(
    #         {"additional_special_tokens": [cfg.model.response_template]}
    #     )
    # model.resize_token_embeddings(len(tokenizer))
    # finetuned_model.resize_token_embeddings(len(tokenizer))
    # dpo_model.resize_token_embeddings(len(tokenizer))
    dpo_model.eval()

    mean_scores = evaluate_metrics(
        cfg,
        None,
        None,
        tokenizer,
        dataset_path=cfg.dataset.eval_dataset,
        dpo_model=dpo_model,
        mixture=True,
        train_test_split=None,
    )


if __name__ == "__main__":
    main()

import hydra
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,

)
from trl import (
    DPOTrainer,
    DPOConfig,

)
from data_utils import (
    PassageClarificationPromptFormatter,
    apply_chat_template,
)
from utils import setup_env, is_ampere_gpu
import pandas as pd
from datasets import Dataset
import gc
from dotenv import load_dotenv
import torch
import wandb


def dpo(cfg, model, ref_model, tokenizer, peft_config):

    attn_kwargs = {}
    if is_ampere_gpu():
        # attn_kwargs["attn_implementation"] = "flash_attention_2"
        # attn_kwargs["torch_dtype"] = torch.bfloat16
        attn_kwargs["torch_dtype"] = torch.float32

    data_formatter = PassageClarificationPromptFormatter(
        response_template=cfg.model.response_template
    )

    # result_df = pd.read_json(
    #     cfg.dataset.preference_dataset, lines=True, orient="records"
    # )

    result_df = pd.read_json(
        "/lustre/fswork/projects/rech/ize/uyy82al/jz-sync/llm_judging/pq3f_judge_finetuned_filtered.json", lines=True, orient="records"
    )
    if cfg.model.chat:
        result_df = result_df.apply(
            data_formatter.format_dataset_preference_tuning, axis=1)
    else:
        result_df = result_df.apply(
            data_formatter.format_dataset_preference_tuning_no_chat, axis=1)
    result_df['chosen'] = result_df["fine_tuned"]
    dataset = Dataset.from_pandas(result_df[["prompt", "chosen", "rejected"]])
    if cfg.model.chat:
        dataset = dataset.map(
            lambda example: apply_chat_template(example, tokenizer=tokenizer)
        )

    training_args = DPOConfig(
        output_dir=cfg.dpo.dpo_directory,
        per_device_train_batch_size=cfg.dpo.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.dpo.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.dpo.gradient_accumulation_steps,
        gradient_checkpointing=cfg.dpo.gradient_checkpointing,
        max_grad_norm=cfg.dpo.max_grad_norm,
        learning_rate=cfg.dpo.learning_rate,
        lr_scheduler_type=cfg.dpo.lr_scheduler_type,
        warmup_ratio=cfg.dpo.warmup_ratio,
        optim=cfg.dpo.optim,
        logging_steps=cfg.dpo.logging_steps,
        num_train_epochs=cfg.dpo.num_epochs,
        save_steps=cfg.dpo.save_steps,
        max_length=cfg.dpo.max_seq_length,
        bf16=cfg.dpo.bf16,
        save_total_limit=cfg.dpo.save_total_limit,
        beta=cfg.dpo.beta,
        auto_find_batch_size=True,
        report_to="wandb",
        run_name=cfg.wandb.run_name,
        loss_type=[cfg.dpo.loss_type, "sft"],
        loss_weights=[0.5, 0.5],

    )
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    return trainer


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to run the script
    """
    # Setup environment
    setup_env(cfg)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.load_directory)

    fine_tuned_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.load_directory,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    ref_fine_tuned_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.load_directory,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
    )

    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({"pad_token": cfg.model.pad_token})
    #     tokenizer.add_special_tokens(
    #         {"additional_special_tokens": [cfg.model.response_template]}
    #     )
    #     fine_tuned_model.resize_token_embeddings(len(tokenizer))
    #     ref_fine_tuned_model.resize_token_embeddings(len(tokenizer))

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    trainer = dpo(
        cfg,
        fine_tuned_model,
        ref_fine_tuned_model,
        tokenizer,
        peft_config=None,
    )

    trainer.train()

    # Save the model

    trainer.save_model(cfg.dpo.dpo_directory)
    tokenizer.save_pretrained(cfg.dpo.dpo_directory)


if __name__ == "__main__":
    main()

import hydra
from typing import Dict
from omegaconf import DictConfig
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import (
    SFTConfig,
    SFTTrainer,
)
from data_utils import (
    PassageClarificationPromptFormatter,
    UnconditionalClarificationPromptFormatter
)
from utils import setup_env
import pandas as pd
from datasets import Dataset
import gc
import torch

# wandb.ensure_configured()


def prepare_trainer(
    cfg: DictConfig, model, tokenizer, peft_config, response_template, mode="full_sft"
):
    """
    Prepare the trainer for the model
    """

    # print("WandB is enabled, initializing...")
    # # print(importlib.util.find_spec("wandb") is not None)

    # run = wandb.init(
    #     project=cfg.wandb.project,
    #     name=cfg.wandb.run_name,
    #     config=OmegaConf.to_container(cfg, resolve=True),
    #     mode=cfg.wandb.mode
    # )

    data_formatter = PassageClarificationPromptFormatter(
        response_template=response_template
    )

    if cfg.mode == "uncond_sft":
        data_formatter = UnconditionalClarificationPromptFormatter(
            response_template=response_template
        )
        cfg.model.save_directory = cfg.model.uncond_save_directory

    result_df = pd.read_json(cfg.dataset.fine_tune_path,
                             lines=True, orient="records")
    if cfg.model.chat:
        result_df = result_df.apply(data_formatter.format_dataset, axis=1)
    else:
        result_df = result_df.apply(
            data_formatter.format_dataset_no_chat, axis=1)
    # result_df["messages"] = result_df["text"]
    dataset = Dataset.from_pandas(result_df[["prompt", "completion"]])
    # result_df["messages"] = result_df.apply(
    #     lambda row: tokenizer.apply_chat_template(
    #         row["text"],
    #         tokenize=False,
    #         add_generation_prompt=False,
    #     ),
    #     axis=1,
    # )
    # dataset = dataset.map(
    #     lambda examples: formatting_prompts_func(examples, tokenizer), batched=True
    # )

    if mode == "dpo_sft":
        train_test_split = dataset.train_test_split(
            test_size=1.0 - cfg.dataset.train_test_split, seed=cfg.dataset.seed
        )
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]
    elif mode == "full_sft":
        train_dataset = dataset
        cfg.model.save_directory = cfg.model.save_directory + "_full_sft"
    elif mode == "uncond_sft":
        train_test_split = dataset.train_test_split(
            test_size=1.0 - cfg.dataset.train_test_split, seed=cfg.dataset.seed
        )

        train_dataset = train_test_split["train"]
        # train_test_split = train_dataset.train_test_split(
        #     test_size=1.0 - cfg.dataset.train_test_split, seed=cfg.dataset.seed
        # )
        # train_dataset = train_test_split["train"]

    training_args = SFTConfig(
        report_to="wandb",
        run_name=cfg.wandb.run_name,
        output_dir=cfg.model.save_directory,
        per_device_train_batch_size=cfg.model.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.model.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.model.gradient_accumulation_steps,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        max_grad_norm=cfg.model.max_grad_norm,
        lr_scheduler_type=cfg.model.lr_scheduler_type,
        optim=cfg.model.optim,
        warmup_ratio=cfg.model.warmup_ratio,
        logging_steps=cfg.model.logging_steps,
        num_train_epochs=cfg.model.num_epochs,
        learning_rate=cfg.model.learning_rate,
        logging_dir=cfg.model.logging_dir,
        save_steps=cfg.model.save_steps,
        max_length=cfg.model.max_seq_length,
        # bf16=cfg.model.bf16,

        # eval_steps=cfg.model.eval_steps,
        save_total_limit=cfg.model.save_total_limit,
        # load_best_model_at_end=True,
        label_names=["labels"],
        # fsdp=True,
        # accelerator_config=accelerate_config,
        # fsdp_strategy="full_shard",

        # gradient_checkpointing_kwargs={"use_reentrant": False},
        # ddp_find_unused_parameters=False,
        auto_find_batch_size=True,
        completion_only_loss=True,
    )

    # Access the train and test datasets

    # response_template = "\n<|assistant|>\n"
    # collator = DataCollatorForCompletionOnlyLM(
    #     response_template=data_formatter.response_template,
    #     tokenizer=tokenizer,
    #     mlm=False,
    # )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        # data_collator=collator,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    # run.finish()
    return trainer


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to run the script
    """
    # Setup environment
    setup_env(cfg)
    # if not cfg.wandb.disabled:

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(
        # "gpt2",
        cfg.model.name,
        # load_in_8bit=cfg.model.load_in_8bit,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": cfg.model.pad_token})
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [cfg.model.response_template]}
        )
        model.resize_token_embeddings(len(tokenizer))

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Prepare trainer
    trainer = prepare_trainer(
        cfg,
        model,
        tokenizer,
        peft_config=None,
        response_template=cfg.model.response_template,
        mode=cfg.mode,
    )

    trainer.train()

    # Save the model
    trainer.save_model(cfg.model.save_directory)
    tokenizer.save_pretrained(cfg.model.save_directory)


if __name__ == "__main__":
    main()

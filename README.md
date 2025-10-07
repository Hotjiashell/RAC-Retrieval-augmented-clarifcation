# RAC-Retrieval-augmented-clarifcation

Clarification questions help conversational search systems resolve ambiguous or underspecified queries. However, most prior approaches risk asking questions that are not grounded in the corpus, leading to misleading interactions. We propose RAC (Retrieval-Augmented Clarification), a framework for generating faithful clarifying questions.

<p align="center">
  <img src="assets/racexample.png" alt="Overview of RAC." width="600">
  <br>
  <em>Figure 1. Overview of RAC. Given an ambiguous user query, the system first retrieves the top-$k$ passages ((a) passage retrieval). A mixture of the fine-tuned model and the base model is then used to generate unfaithful clarifying questions. Both faithful and unfaithful clarifying questions are subsequently leveraged for preference optimization via the DPO algorithm ((b) training pipeline). During inference, the trained model directly generates faithful clarifying questions.</em>
</p>



In the following we will go through the execution of each component of RAC pipeline.

<p align="center">
  <img src="assets/pipelinerac.png" alt="RAC training and data generation Pipeline" width="600">
  <br>
  <em>Figure 2. Overview of RAC training pipeline.</em>
</p>




## 🚀 Installation & Environment Setup

```bash
# Create virtual environment
conda create -n rac_env python=3.10
conda activate rac_env

# Install dependencies
pip install -r requirements.txt
```

Login to huggingface

```bash
huggingface-cli login --token "..."
```

or export the env variable

```bash
export HF_TOKEN=""
```

## 📊 Fine-tuning

To fine-tune the model, replace `DATASET_NAME` with your chosen dataset.

* Open datasets are available in the `./data` folder.
* **Note**: Qulac and ClariQ augmented datasets are not included due to the ClueWeb09/12 license restrictions.
  However, if you have access to a ClueWeb index, you can use the data augmentation scripts provided in the `./data_augmentation` folder to reproduce the augmented datasets.

### Example command:


```bash

bash job_run_src.sh --finetune model.name=meta-llama/Llama-3.1-8B model.tokenizer_name=meta-llama/Llama-3.1-8B model.Qlora=False dataset.name=DATASET_NAME model.learning_rate=5e-5 dataset.fine_tune_path=../data/DATASET_NAME/DATASET_train.json dataset.eval_dataset=../data/DATASET_NAME/DATASET_dev.jsonl model.num_epochs=1 model.per_device_train_batch_size=32 model.bf16=True model.lr_scheduler_type=linear model.gradient_checkpointing=True model.gradient_accumulation_steps=2 mixture.num_variants=1 mixture.dpo_bs=32 run.gen_run_id=GEN_ID run.eval_run_id=RUN_ID model.chat=False

```

## Second finetuning of the $Uncond$ model

Second finetuning to train the model conditioned only on query.

```bash
bash job_run_src.sh --uncond model.name=meta-llama/Llama-3.1-8B model.tokenizer_name=meta-llama/Llama-3.1-8B model.Qlora=False dataset.name=DATASET_NAME model.learning_rate=5e-5 dataset.fine_tune_path=../data/DATASET_NAME/DATASET_train.json dataset.eval_dataset=../data/DATASET_NAME/DATASET_dev.jsonl  dataset.train_test_split=0.5 model.num_epochs=1 model.per_device_train_batch_size=32 model.bf16=True model.lr_scheduler_type=linear model.gradient_checkpointing=True model.gradient_accumulation_steps=2 mixture.num_variants=1 mixture.dpo_bs=32 run.gen_run_id=GEN_ID run.eval_run_id=RUN_ID model.chat=False dpo.untouched=0
```


## Noisy data generation (rejected data samples $Cq^-$)

Generate the noisy samples which will be used by the next step of the pipeline to train dpo 

```bash
bash job_run_src.sh --data-generation model.name=meta-llama/Llama-3.1-8B model.tokenizer_name=meta-llama/Llama-3.1-8B model.Qlora=False dataset.name=DATASET_NAME model.learning_rate=5e-5 dataset.fine_tune_path=../DATASET/DATASET_train.json dataset.eval_dataset=../DATASET/DATASET_dev.jsonl dataset.train_test_split=0.5 model.num_epochs=1 model.per_device_train_batch_size=32 model.bf16=True model.lr_scheduler_type=linear model.gradient_checkpointing=True model.gradient_accumulation_steps=2 mixture.alpha=0.7 dpo.per_device_train_batch_size=32 dpo.num_epochs=2 dpo.learning_rate=4e-6 dpo.beta=0.2 dpo.bf16=True dpo.max_grad_norm=1.0 dpo.warmup_ratio=0.1 dpo.loss_type="dpo" dpo.loss="dpo" mixture.bs=32 mixture.num_variants=1 mixture.dpo_bs=32 run.gen_run_id=GEN_ID run.eval_run_id=RUN_ID model.chat=False dpo.untouched=0
```



## Disclaimer

```bash


```

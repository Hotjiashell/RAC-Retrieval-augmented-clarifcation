# RAC-Retrieval-augmented-clarifcation



## 🚀 Installation & Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## 📊 Fine-tuning

To fine-tune the model, replace `DATASET_NAME` with your chosen dataset.

* Open datasets are available in the `./data` folder.
* **Note**: Qulac and ClariQ augmented datasets are not included due to the ClueWeb09/12 license restrictions.
  However, if you have access to a ClueWeb index, you can use the data augmentation scripts provided in the `./data_augmentation` folder to reproduce the augmented datasets.

### Example command:


```bash

bash job_run_src.sh model.name=meta-llama/Llama-3.1-8B model.tokenizer_name=meta-llama/Llama-3.1-8B model.Qlora=False dataset.name=DATASET_NAME model.learning_rate=5e-5 dataset.fine_tune_path=../DATASET/DATASET_train.json dataset.eval_dataset=../DATASET/DATASET_dev.jsonl dataset.train_test_split=0.5 model.num_epochs=1 model.per_device_train_batch_size=32 model.bf16=True model.lr_scheduler_type=linear model.gradient_checkpointing=True model.gradient_accumulation_steps=2 mixture.alpha=0.7 dpo.per_device_train_batch_size=32 dpo.num_epochs=2 dpo.learning_rate=4e-6 dpo.beta=0.2 dpo.bf16=True dpo.max_grad_norm=1.0 dpo.warmup_ratio=0.1 dpo.loss_type="dpo" dpo.loss="dpo" mixture.bs=32 mixture.num_variants=1 mixture.dpo_bs=32 run.gen_run_id=GEN_ID run.eval_run_id=RUN_ID model.chat=False dpo.untouched=0

```

## Noisy data generation (rejected data samples $Cq^-$)



## Disclaimer

```bash


```

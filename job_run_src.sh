#!/bin/bash
#SBATCH --job-name=gen-neg-job
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=h100 
#SBATCH --cpus-per-task=10
#SBATCH --time=4:00:00
#SBATCH --account=ize@h100


export WORK=/lustre/fswork/projects/rech/ize/uyy82al
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export HYDRA_FULL_ERROR=1

export TRANSFORMERS_OFFLINE=1


source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate few-shot-env
cd $WORK/jz-sync/src/

module load cuda


export HF_HOME='/lustre/fswork/projects/rech/ize/uyy82al/.cache/'
 
# activation du mode offline
export WANDB_MODE=offline

conda activate few-shot-env
which python
python -c "import wandb; print('✅ wandb is available:', wandb.__version__)"

run_sft=false
run_finetune=false
run_uncond_sft=false
run_data_gen=false
run_sft_data_gen=false
run_multi_data_gen=false
run_dpo_train=false
run_data_gen_dpo=false
run_multi_dpo_train=false
run_metrics=false
run_judge=false

extra_args=""


while [[ $# -gt 0 ]]; do
  case $1 in
    --sft) run_sft=true ;;
    --finetune) run_finetune=true ;;
    --uncond-sft) run_uncond_sft=true ;;
    --data-generation) run_data_gen=true ;;
    --dpo-training) run_dpo_train=true ;;
    --data-generation-sft) run_sft_data_gen=true ;;
    --data-generation-dpo) run_data_gen_dpo=true ;;
    --metrics) run_metrics=true ;;
    --multi-data-generation) run_multi_data_gen=true ;;
    --judge) run_judge=true ;;
    --multi-dpo-training) run_multi_dpo_train=true;;
    --all)
      run_finetune=true
      run_data_gen=true
      run_multi_data_gen=true
      run_dpo_train=true
      run_data_gen_dpo=true
      run_multi_dpo_train=true
      run_metrics=true
      ;;
   # --*)  # Unknown option
    #  echo "Unknown option: $1"
    #  ;;
    *)  # Assume it's a hydra argument, collect it
      extra_args+="$1 "
      ;;
  esac
  shift
done


# Run each step if enabled

if $run_sft; then
  echo "Running full SFT"
  python run_sft.py mode="full_sft" $extra_args
fi

if $run_finetune; then
  echo "Running fine-tuning..."
 # accelerate launch --multi_gpu --num_processes=4 generate_negative_facet.py $extra_args mode="train"
  python run_sft.py $extra_args mode="dpo_sft"
fi

if $run_uncond_sft; then
  echo "Running unconditional fine-tuning..."
  python run_sft.py $extra_args mode="uncond_sft"
fi

if $run_data_gen; then
  echo "Running data generation..."
  python negative_generation.py $extra_args mode="generate"
fi

if $run_multi_data_gen; then
  echo "Running multi data generation...."
  python multi_negative_generation.py $extra_args mode="generate"
fi

if $run_sft_data_gen; then
  echo "Running full sft data generation"
  python sft_generation.py $extra_args mode="generate-full-sft"
fi

if $run_dpo_train; then
  echo "Running DPO training..."
  python run_dpo.py $extra_args mode="dpo"
fi

if $run_multi_dpo_train; then
  echo "Running multi DPO training..."
  python run_mn-dpo.py $extra_args mode="multi_dpo"
fi


if $run_data_gen_dpo; then
  echo "Running data generation for DPO..."
  python dpo_generation.py $extra_args mode="generate-dpo"
fi



if $run_metrics; then
  echo "Running metrics computation..."
  python run_evaluation.py $extra_args mode="compute_metrics"
fi


echo extra_args: $extra_args

echo $(pwd)

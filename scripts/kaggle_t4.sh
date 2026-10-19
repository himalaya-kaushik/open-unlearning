#!/bin/bash
# TOFU unlearn + eval on Kaggle 2x T4 (sm75: no bf16, no flash-attn 2). See KAGGLE.md.
# Usage: bash scripts/kaggle_t4.sh <trainer> <forget_split> <holdout_split> <retain_split> [model] [experiment]
# Example: bash scripts/kaggle_t4.sh NPO forget10 holdout10 retain90


trainer=${1:?trainer}
forget_split=${2:?forget_split}
holdout_split=${3:?holdout_split}
retain_split=${4:?retain_split}
model=${5:-Llama-3.2-1B-Instruct}
experiment=${6:-unlearn/tofu/default.yaml}

# same batch setup as scripts/tofu_unlearn.sh: 4 x 4 accum x 2 GPUs = 32 effective
per_device_train_batch_size=4
gradient_accumulation_steps=4

task_name=tofu_${model}_${forget_split}_${trainer}
model_path=open-unlearning/tofu_${model}_full

# T4 overrides; the tokenizer comes from the TOFU model repo (not gated) instead of meta-llama
t4_model_args="model.model_args.attn_implementation=sdpa \
model.tokenizer_args.pretrained_model_name_or_path=${model_path}"

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"
echo ${task_name}: Unlearning ${model_path} using ${trainer}

# Unlearn: 2 GPUs, DeepSpeed ZeRO-3 fp16; fp32 weights loaded, DeepSpeed keeps fp16 copies + fp32 master
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/t4_config.yaml --main_process_port $MASTER_PORT \
src/train.py --config-name=unlearn.yaml \
experiment=${experiment} \
trainer=${trainer} \
task_name=${task_name} \
model=${model} \
forget_split=${forget_split} \
retain_split=${retain_split} \
model.model_args.pretrained_model_name_or_path=${model_path} \
${t4_model_args} \
model.model_args.torch_dtype=float32 \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
trainer.args.bf16=False \
trainer.args.bf16_full_eval=False \
++trainer.args.fp16=True \
trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
trainer.args.ddp_find_unused_parameters=true \
trainer.args.gradient_checkpointing=true || exit 1

# Eval: 1 GPU, fp16
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
experiment=eval/tofu/default.yaml \
forget_split=${forget_split} \
holdout_split=${holdout_split} \
model=${model} \
task_name=${task_name} \
model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
${t4_model_args} \
model.model_args.torch_dtype=float16 \
paths.output_dir=saves/unlearn/${task_name}/evals \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

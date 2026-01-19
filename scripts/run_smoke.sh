#!/usr/bin/env bash
set -euo pipefail
source /home/qa/data2/tmp/project/src/env.sh

# # conda 활성화 
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate rag_abstain

EXP="smoke_proposed"
OUT_DIR="${OUT_ROOT}/${EXP}"

TRAIN_PATH="${SPLIT_DIR}/proposed_train.jsonl"
EVAL_PATH="${SPLIT_DIR}/eval_val.jsonl"


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
deepspeed --num_gpus 4 /home/qa/data2/tmp/project/src/train.py \
  --model_name "${MODEL_NAME}" \
  --train_path "${TRAIN_PATH}" \
  --eval_path "${EVAL_PATH}" \
  --output_dir "${OUT_DIR}" \
  --max_seq_len 2048 \
  --max_contexts 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --num_train_epochs 0.01 \
  --logging_steps 5 \
  --save_steps 1000000 \
  --eval_steps 50 \
  --gradient_checkpointing \
  --fp16 \
  --deepspeed /home/qa/data2/tmp/project/src/ds_zero2_fp16.json


# 스모크: 아주 짧게(예: 50 step 정도)
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node=4 /home/qa/data2/tmp/project/src/train.py \
#   --model_name "${MODEL_NAME}" \
#   --train_path "${TRAIN_PATH}" \
#   --eval_path "${EVAL_PATH}" \
#   --output_dir "${OUT_DIR}" \
#   --max_seq_len 2048 \
#   --max_contexts 3 \
#   --per_device_train_batch_size 1 \
#   --gradient_accumulation_steps 4 \
#   --learning_rate 2e-4 \
#   --num_train_epochs 0.01 \
#   --logging_steps 5 \
#   --save_steps 1000000 \
#   --eval_steps 50 \
#   --gradient_checkpointing \
#   --fp16

# CUDA_VISIBLE_DEVICES=0 python /home/qa/data2/tmp/project/src/train.py \
#   --model_name "${MODEL_NAME}" \
#   --train_path "${TRAIN_PATH}" \
#   --eval_path "${EVAL_PATH}" \
#   --output_dir "${OUT_DIR}" \
#   --max_seq_len 2048 \
#   --max_contexts 3 \
#   --per_device_train_batch_size 1 \
#   --gradient_accumulation_steps 4 \
#   --learning_rate 2e-4 \
#   --num_train_epochs 0.01 \
#   --logging_steps 5 \
#   --save_steps 1000000 \
#   --eval_steps 50 \
#   --gradient_checkpointing \
#   --fp16


# 스모크 eval (val에서만)
python /home/qa/data2/tmp/project/src/eval.py \
  --model_or_ckpt "${OUT_DIR}" \
  --data_path "${EVAL_PATH}" \
  --out_path "${RES_ROOT}/${EXP}_val.json" \
  --max_contexts 3 \
  --max_new_tokens 64 \
  --temperature 0 \
  --batch_size 4

echo "[done] smoke test complete -> ${RES_ROOT}/${EXP}_val.json"

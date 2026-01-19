#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

#실행 폴더
export PYTHONPATH="/home/qa/data2/tmp/project/src:${PYTHONPATH:-}"

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

export TOKENIZERS_PARALLELISM=false

export SPLIT_DIR="/home/qa/data2/tmp/project/data/g60_seed42" #split한 경로

# 결과 저장
export RES_ROOT="/home/qa/data2/tmp/project/results"

# 모델
: "${MODEL_NAME:=Qwen/Qwen2.5-7B-Instruct}"
export MODEL_NAME

echo "[env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[env] SPLIT_DIR=$SPLIT_DIR"

echo "[env] RES_ROOT=$RES_ROOT"
echo "[env] MODEL_NAME=$MODEL_NAME"

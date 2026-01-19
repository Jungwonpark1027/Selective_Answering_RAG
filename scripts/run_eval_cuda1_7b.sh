#!/usr/bin/env bash
set -euo pipefail


# 경로
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

source "${PROJ_ROOT}/scripts/env.sh"

CUDA_VISIBLE_DEVICES=1     

EVAL_PY="${PROJ_ROOT}/src/eval.py"

CKPT_ROOT="${PROJ_ROOT}/ckpt/qwen25_7b"
DATA_ROOT="${PROJ_ROOT}/data"
OUT_ROOT="${PROJ_ROOT}/results/eval/qwen25_7b"
LOG_ROOT="${PROJ_ROOT}/results/logs"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

# 
SPLITS=(g60_seed4222)
SEEDS=(42 123 428)
VARIANTS=(baseline1 baselineB proposed)

# 
MAX_CTX=5
MAX_NEW_TOKENS=64
BATCH_SIZE=4
TEMPERATURE=0
TOP_P=0.9
SAVE_SAMPLES=200


: "${SKIP_IF_EXISTS:=1}"

echo "[env] CKPT_ROOT=${CKPT_ROOT}"
echo "[env] DATA_ROOT=${DATA_ROOT}"
echo "[env] OUT_ROOT=${OUT_ROOT}"
echo

for split in "${SPLITS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for v in "${VARIANTS[@]}"; do

      CKPT_DIR="${CKPT_ROOT}/${split}/${v}_seed${seed}"
      DATA_PATH="${DATA_ROOT}/${split}/eval_val.jsonl"

      OUT_JSON="${OUT_ROOT}/${split}_${v}_seed${seed}.json"
      ts="$(date +%Y%m%d_%H%M%S)"
      LOG_FILE="${LOG_ROOT}/eval_${split}_${v}_seed${seed}_${ts}.log"

      if [[ ! -d "${CKPT_DIR}" ]]; then
        echo "[skip] ckpt not found: ${CKPT_DIR}"
        continue
      fi

      if [[ "${SKIP_IF_EXISTS}" == "1" ]] && [[ -f "${OUT_JSON}" ]]; then
        echo "[skip] eval already exists: ${OUT_JSON}"
        continue
      fi

      echo "=================================================="
      echo "[eval] split=${split} variant=${v} seed=${seed}"
      echo "[eval] ckpt_dir=${CKPT_DIR}"
      echo "[eval] data_path=${DATA_PATH}"
      echo "[eval] out=${OUT_JSON}"
      echo "=================================================="

      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      python "${EVAL_PY}" \
        --model_or_ckpt "${CKPT_DIR}" \
        --data_path "${DATA_PATH}" \
        --out_path "${OUT_JSON}" \
        --max_contexts "${MAX_CTX}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --batch_size "${BATCH_SIZE}" \
        --save_samples "${SAVE_SAMPLES}" \
        --fp16 \
      2>&1 | tee "${LOG_FILE}"

      echo
    done
  done
done

echo "[done] eval sweep complete -> ${OUT_ROOT}"

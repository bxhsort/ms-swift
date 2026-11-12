#!/bin/bash
set -euo pipefail

# export WANDB_MODE=offline
# export WANDB_PROJECT=Unified_Reward_Model
# export WANDB_NAME=Qwen3_5VL_9B_test
# export WANDB_DIR=wandb


export WANDB_DIR="$PWD/wandb"

# MODEL=${MODEL:-/inspire/hdd3/project/agent-driven-world-model/public/deepggen/qblim/model_zoo/SkyJM-Edit-9B}
MODEL=${MODEL:-/inspire/hdd3/project/agent-driven-world-model/public/deepggen/hdd/model_zoo/Qwen3.5-9B}
DATASET=${DATASET:-/inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/ms-swift/Rebuttal/data_TRM_generative_en_messages.jsonl}
OUT=${OUT:-output/init_Rebuttal_Generative}

# Optional validation set:
#   VAL_DATASET=/path/to/val.jsonl bash scripts/train_qwen3_5_9b.sh
VAL_DATASET=${VAL_DATASET:-}

# Qwen3.5 uses token budgets from qwen_vl_utils:
# image pixels ~= IMAGE_MAX_TOKEN_NUM * 32 * 32.
# The official Qwen3.5 examples use 1024 image tokens and 128 video tokens/frame.
IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-1024}
VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-128}
FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-12}

# Default launch target: 4 * H200.
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

VAL_ARGS=()
if [[ -n "$VAL_DATASET" ]]; then
  VAL_ARGS=(--val_dataset "$VAL_DATASET")
else
  VAL_ARGS=(--split_dataset_ratio 0.005)
fi

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM="$IMAGE_MAX_TOKEN_NUM" \
VIDEO_MAX_TOKEN_NUM="$VIDEO_MAX_TOKEN_NUM" \
FPS_MAX_FRAMES="$FPS_MAX_FRAMES" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
swift sft \
  --model "$MODEL" \
  --dataset "$DATASET" \
  "${VAL_ARGS[@]}" \
  --template qwen3_5 \
  --tuner_type lora \
  --lora_rank 32 \
  --lora_alpha 64 \
  --target_modules all-linear \
  --add_non_thinking_prefix true \
  --loss_scale ignore_empty_think \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --max_length 8192 \
  --group_by_length true \
  --load_from_cache_file true \
  --dataset_num_proc 16 \
  --dataloader_num_workers 4 \
  --save_steps 200 \
  --eval_steps 200 \
  --save_total_limit 10 \
  --logging_steps 1 \
  --max_steps -1 \
  --output_dir "$OUT" \
  --save_only_model false \
  --report_to tensorboard \
  --deepspeed zero2

#!/bin/bash

MODEL=/inspire/hdd/project/deepgen/public/model_zoo/Qwen3-VL-8B-Instruct
DATASET=/inspire/qb-ilm/project/deepgen/wangdianyi-240107110022/BXH/RL_Data/EditScore-Reward-Data/parse_train_data_5_2k_editscore_Final.json
OUT=output/qwen3vl_8b_test_it2i

export WANDB_MODE=offline
export WANDB_PROJECT=Unified_Reward_Model
export WANDB_NAME=Qwen3VL_8B_testcode
export WANDB_DIR=wandb


export WANDB_DIR="$PWD/wandb"

#
# Qwen3-VL token env:
# image token ~= pixels / (32 * 32)
# 262144 / 1024 = 256
# 16384 / 1024 = 16
IMAGE_MAX_TOKEN_NUM=512 \
VIDEO_MAX_TOKEN_NUM=16 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --template qwen3_vl \
  --tuner_type lora \
  --lora_rank 32 \
  --lora_alpha 64 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --lr_scheduler_type cosine \
  --modules_to_save embed_tokens lm_head \
  --warmup_ratio 0.1 \
  --max_length 8192 \
  --dataset_num_proc 16 \
  --dataloader_num_workers 4 \
  --save_steps 100 \
  --logging_steps 1 \
  --max_steps -1 \
  --output_dir "$OUT" \
  --save_only_model false \
  --report_to wandb \
  --deepspeed zero2

#!/bin/bash

# MODEL=/inspire/hdd/project/deepgen/public/model_zoo/Qwen3-VL-8B-Instruct
# DATASET=/inspire/qb-ilm/project/deepgen/wangdianyi-240107110022/BXH/RL_Data/test_data/Image_Editing/train_edit.json
# OUT=output/qwen3vl_8b_test_code

# mkdir -p "$OUT/runs"

# IMAGE_MAX_TOKEN_NUM=256 \
# VIDEO_MAX_TOKEN_NUM=16 \
# NPROC_PER_NODE=4 \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# swift sft \
#   --model "$MODEL" \
#   --dataset "$DATASET" \
#   --template qwen3_vl \
#   --new_special_tokens \
#     '<Rubric>' '</Rubric>' \
#     '<Evaluation>' '</Evaluation>' \
#   --tuner_type lora \
#   --lora_rank 32 \
#   --lora_alpha 64 \
#   --target_modules all-linear \
#   --torch_dtype bfloat16 \
#   --num_train_epochs 10 \
#   --per_device_train_batch_size 8 \
#   --gradient_accumulation_steps 8 \
#   --learning_rate 2e-4 \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.1 \
#   --max_length 8192 \
#   --dataset_num_proc 16 \
#   --dataloader_num_workers 4 \
#   --save_steps 250 \
#   --logging_steps 1 \
#   --max_steps -1 \
#   --output_dir "$OUT" \
#   --save_only_model false \
#   --report_to tensorboard \
#   --logging_dir "$OUT/runs" \
#   --deepspeed zero2

#!/usr/bin/env bash
set -euo pipefail

BASE=/inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/ms-swift/best_weight/checkpoint-250-merged

RUN_DIR=/inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/RL_train/verl/checkpoints/CVPR_2027_Image_Editing_RL/Rebuttal_qwen3_5_9b_pairwise_image_edit_BT_Format_adjust_param

ADAPTER=$RUN_DIR/global_step_25_merged/lora_adapter
OUTPUT=$RUN_DIR/global_step_25_merged_hf

test -s "$ADAPTER/adapter_config.json"
test -s "$ADAPTER/adapter_model.safetensors"

swift export \
    --model "$BASE" \
    --adapters "$ADAPTER" \
    --merge_lora true \
    --output_dir "$OUTPUT"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/nas-files/yuhanchen/miniconda3/envs/lora/bin:$PATH"
export TMPDIR="/nas-files/yuhanchen/tmp"
export TMP="/nas-files/yuhanchen/tmp"
export TEMP="/nas-files/yuhanchen/tmp"

echo "=== SMOKE TEST: CFR-only RAFT training (15 steps) ==="
echo "Using: $(which accelerate)"
echo "Using: $(which python)"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch   --num_processes=2   --num_machines=1   --mixed_precision=fp16   src/train_DLoRAL.py   --train_cfr_only   --flow_estimator raft   --load_cfr   --resume_ckpt ./preset/models/checkpoints/model.pkl   --dataset_txt_paths ./data/reds_train.txt   --pexel_txt_path ./data/reds_train.txt   --deg_file_path params_pasd.yml   --output_dir ./runs/smoke_cfr_raft_v2   --pretrained_model_name_or_path ./preset_models/stable-diffusion-2-1-base   --pretrained_model_path_csd ./preset_models/stable-diffusion-2-1-base   --ram_path ./preset/models/ram_swin_large_14m.pth   --learning_rate 1e-4   --lr_scheduler cosine   --lr_warmup_steps 50   --max_train_steps 15   --train_batch_size 1   --gradient_accumulation_steps 4   --checkpointing_steps 100   --mixed_precision fp16   --allow_tf32   --enable_xformers_memory_efficient_attention   --gradient_checkpointing   --set_grads_to_none   --max_grad_norm 1.0   --seed 42   --resolution 512   --frames 14   --lambda_l2 1.0   --lambda_lpips 2.0   --lambda_consistency 10.0   --lambda_moe 2.0   --report_to tensorboard   --dataloader_num_workers 4 2>&1 | tee ./runs/smoke_cfr_raft_v2/smoke.log

echo "=== Smoke test done ==="

#!/bin/bash
 
export TRANSFORMERS_OFFLINE=1
include=localhost:3 # 设置显卡id
 
model_name_or_path="" # 模型名称
data_path="" # 训练的json
image_folder="" # 训练的图像数据
 
deepspeed --include $include /LLaVA/llava/train/train_mem.py \
    --lora_enable True --lora_r 8 --lora_alpha 16 --mm_projector_lr 2e-5 \
    --deepspeed /LLaMA-Factory/ds_config_zero3.json \
    --model_name_or_path $model_name_or_path  \
    --version v1 \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower /clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir /LLaMA-Factory/saves/llava/lora/train_$(date +'%Y-%m-%d-%H-%M-%S') \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 1 \
    --learning_rate 5e-05 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
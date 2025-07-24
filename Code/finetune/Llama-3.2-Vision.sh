#!/bin/bash

# LLaMA Factory training script with DeepSpeed
# File: train_llama3_vision_lora_deepspeed.sh

# Configuration
NUM_GPUS=4                          # Number of GPUs to use
MODEL_PATH=""
OUTPUT_DIR="saves/Llama-3.2-11B-Vision-Instruct/lora/train_$(date +'%Y-%m-%d-%H-%M-%S')"
DEEPSPEED_CONFIG=""  # Replace with your config path
TRAIN_SCRIPT="src/train.py"         # Correct training script path

# Verify files exist
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "Error: DeepSpeed config file not found at $DEEPSPEED_CONFIG"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found at $TRAIN_SCRIPT"
    echo "Please ensure you're in the LLaMA-Factory project root directory"
    exit 1
fi

# Launch training with DeepSpeed
deepspeed --num_gpus $NUM_GPUS $TRAIN_SCRIPT \
    --deepspeed $DEEPSPEED_CONFIG \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template mllama \
    --flash_attn auto \
    --dataset_dir data \
    --dataset  \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 5000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    --plot_loss \
    --trust_remote_code \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen \
    --optim adamw_torch \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0 \
    --lora_target "q_proj,k_proj,v_proj,o_proj"

# For Ampere GPUs (A100, etc.) consider using --bf16 instead of --fp16
# Add --overwrite_cache if you want to ignore cached datasets
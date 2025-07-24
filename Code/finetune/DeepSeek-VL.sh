# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model  \
    --model_type deepseek_vl \
    --template deepseek_vl \
    --train_type lora \
    --dataset  \
    --torch_dtype float16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-05 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --save_total_limit 5 \
    --save_steps 100 \
    --logging_steps 5 \
    --max_length 3072 \
    --output_dir  \
    --dataloader_num_workers 16 \
    --split_dataset_ratio 0
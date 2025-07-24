# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=3 \
swift sft \
    --model  \
    --model_type mplug_owl3 \
    --template mplug_owl3 \
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
    --max_length 2048 \
    --output_dir  \
    --dataloader_num_workers 16 \
    --split_dataset_ratio 0 \
    --attn_impl None \

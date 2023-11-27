#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=10001

MODEL="/mnt/petrelfs/lijingsong/MLLM/Models/Qwen-VL-Chat" #"Qwen/Qwen-VL-Chat"/"Qwen/Qwen-VL"  Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/mnt/petrelfs/lijingsong/MLLM/Datasets/sharegpt4v/llava_instruct_158k-qwen.json"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

PARTITION=llm4
JOB_NAME=qwen-158k

srun -p $PARTITION \
    --job-name=$JOB_NAME \
    --gres=gpu:$GPUS_PER_NODE \
    --kill-on-bad-exit=1 \
    --quotatype=reserved \
    torchrun $DISTRIBUTED_ARGS finetune.py \
        --model_name_or_path $MODEL \
        --data_path $DATA \
        --bf16 True \
        --fix_vit True \
        --output_dir $JOB_NAME \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate 1e-5 \
        --weight_decay 0.1 \
        --adam_beta2 0.95 \
        --warmup_ratio 0.01 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --report_to "none" \
        --model_max_length 2048 \
        --lazy_preprocess True \
        --use_lora \
        --deepspeed finetune/ds_config_zero2.json
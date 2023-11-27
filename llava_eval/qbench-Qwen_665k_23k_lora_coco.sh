#!/bin/bash

JOB_NAME=qwen-665k-23k-lora-coco

if [ "$1" = "dev" ]; then
    echo "Evaluating in 'dev' split."
elif [ "$1" = "test" ]; then
    echo "Evaluating in 'test' split."
else
    echo "Unknown split, please choose between 'dev' and 'test'."
    exit 1
fi

# python -m llava.eval.model_vqa_qbench \
#     --model-path /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/Experiments/Qwen-VL/qwen-665k-23k-lora-coco \
#     --image-folder /mnt/petrelfs/lijingsong/MLLM/Benchmark/qbench/images_llvisionqa/ \
#     --questions-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/qbench/llvisionqa_$1.json \
#     --answers-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/qbench/${JOB_NAME}_$1_answers.jsonl \
#     --conv-mode llava_v1 \
#     --lang en

# python /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/LLaVA/playground/data/eval/qbench/format_qbench.py \
#     --filepath /mnt/petrelfs/lijingsong/MLLM/Benchmark/qbench/${JOB_NAME}_$1_answers.jsonl

python /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/LLaVA/playground/data/eval/qbench/qbench_eval.py \
    --filepath /mnt/petrelfs/lijingsong/MLLM/Benchmark/qbench/${JOB_NAME}_$1_answers.jsonl

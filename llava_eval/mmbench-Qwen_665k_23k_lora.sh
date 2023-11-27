#!/bin/bash

JOB_NAME=qwen-665k-23k-lora

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/Experiments/Qwen-VL/$JOB_NAME \
    --question-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/mmbench/$SPLIT.tsv \
    --answers-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/mmbench/answers/$SPLIT/$JOB_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /mnt/petrelfs/lijingsong/MLLM/Benchmark/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/mmbench/$SPLIT.tsv \
    --result-dir /mnt/petrelfs/lijingsong/MLLM/Benchmark/mmbench/answers/$SPLIT \
    --upload-dir /mnt/petrelfs/lijingsong/MLLM/Benchmark/mmbench/answers_upload/$SPLIT \
    --experiment $JOB_NAME

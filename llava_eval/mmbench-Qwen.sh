#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"

python -m llava.eval.model_vqa_mmbench \
    --model-path /mnt/petrelfs/lijingsong/MLLM/Models/Qwen-VL-Chat \
    --question-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/mmbench/$SPLIT.tsv \
    --answers-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/mmbench/answers/$SPLIT/qwen.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /mnt/petrelfs/lijingsong/MLLM/Benchmark/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/mmbench/$SPLIT.tsv \
    --result-dir /mnt/petrelfs/lijingsong/MLLM/Benchmark/mmbench/answers/$SPLIT \
    --upload-dir /mnt/petrelfs/lijingsong/MLLM/Benchmark/mmbench/answers_upload/$SPLIT \
    --experiment qwen

python /mnt/petrelfs/lijingsong/MLLM/Benchmark/mmbench/mmbench_excel_test.py /mnt/petrelfs/lijingsong/MLLM/Benchmark/mmbench/answers_upload/$SPLIT/qwen.xlsx
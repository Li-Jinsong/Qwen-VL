#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

model_name=qwen_158k_lora

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
    --model-path /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/Experiments/Qwen-VL/qwen-158k-lora \
    --question-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/MME/llava_mme.jsonl \
    --image-folder /mnt/petrelfs/lijingsong/MLLM/Benchmark/MME/MME_Benchmark_release \
    --answers-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/MME/answers/${model_name}-${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &
done

wait

output_file=/mnt/petrelfs/lijingsong/MLLM/Benchmark/MME/answers/${model_name}.jsonl

> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /mnt/petrelfs/lijingsong/MLLM/Benchmark/MME/answers/${model_name}-${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

cd /mnt/petrelfs/lijingsong/MLLM/Benchmark/MME

python convert_answer_to_mme.py --experiment ${model_name}

cd eval_tool

python calculation.py --results_dir answers/${model_name}

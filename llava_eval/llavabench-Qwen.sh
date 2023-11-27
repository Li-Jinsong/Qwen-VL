#!/bin/bash

# python -m llava.eval.model_vqa \
#     --model-path /mnt/petrelfs/lijingsong/MLLM/Models/Qwen-VL-Chat \
#     --question-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/llava-bench-in-the-wild/questions.jsonl \
#     --image-folder /mnt/petrelfs/lijingsong/MLLM/Benchmark/llava-bench-in-the-wild/images \
#     --answers-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/llava-bench-in-the-wild/answers/qwen.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# mkdir -p /mnt/petrelfs/lijingsong/MLLM/Benchmark/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question /mnt/petrelfs/lijingsong/MLLM/Benchmark/llava-bench-in-the-wild/questions.jsonl \
    --context /mnt/petrelfs/lijingsong/MLLM/Benchmark/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        /mnt/petrelfs/lijingsong/MLLM/Benchmark/llava-bench-in-the-wild/answers_gpt4.jsonl \
        /mnt/petrelfs/lijingsong/MLLM/Benchmark/llava-bench-in-the-wild/answers/qwen.jsonl \
    --output \
        /mnt/petrelfs/lijingsong/MLLM/Benchmark/llava-bench-in-the-wild/reviews/qwen.jsonl

python llava/eval/summarize_gpt_review.py -f /mnt/petrelfs/lijingsong/MLLM/Benchmark/llava-bench-in-the-wild/reviews/qwen.jsonl

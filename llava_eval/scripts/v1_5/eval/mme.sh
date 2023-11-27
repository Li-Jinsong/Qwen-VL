!/bin/bash

python -u -m llava.eval.model_vqa_loader \
    --question-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/MME/llava_mme.jsonl \
    --image-folder /mnt/petrelfs/lijingsong/MLLM/Benchmark/MME/MME_Benchmark_release_version \
    --answers-file /mnt/petrelfs/lijingsong/MLLM/Benchmark/MME/answers/minigpt4-ours.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /mnt/petrelfs/lijingsong/MLLM/Benchmark/MME

python convert_answer_to_mme.py --experiment minigpt4-ours

cd eval_tool

python calculation.py --results_dir answers/minigpt4-ours
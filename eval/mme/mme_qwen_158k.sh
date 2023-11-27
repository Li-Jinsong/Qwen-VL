JOB_NAME=qwen_158k_lora

cd /mnt/petrelfs/lijingsong/MLLM/Benchmark/MME/eval_tool/

python qwen_eval.py \
    --checkpoint /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/Experiments/Qwen-VL/qwen-158k-lora \
    --job-name ${JOB_NAME}
python calculation.py --results_dir ${JOB_NAME}
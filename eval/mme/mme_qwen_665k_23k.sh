JOB_NAME=qwen_665k_23k_lora

cd /mnt/petrelfs/lijingsong/MLLM/Benchmark/MME/eval_tool/

python qwen_eval.py \
    --checkpoint /mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/Experiments/Qwen-VL/qwen-665k-23k-lora \
    --job-name ${JOB_NAME}
python calculation.py --results_dir ${JOB_NAME}
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data/kw/models/Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port  59899 \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --max_model_len 8000 \
    --tensor_parallel_size 1
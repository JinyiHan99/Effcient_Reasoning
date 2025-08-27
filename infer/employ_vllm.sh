CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model /data2/Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port  59824 \
    --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --max_model_len 10000 \
    --tensor_parallel_size 1
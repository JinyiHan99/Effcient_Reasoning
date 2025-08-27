## 启动ref server
CUDA_VISIBLE_DEVICES=7 screen -L -Logfile /mnt/data/kw/hjy/logs/train/0827/server.log python /mnt/data/kw/hjy/efficient-reasoning/lsrl/refer_server_hjy_fast_think.py

## 启动压缩模型
CUDA_VISIBLE_DEVICES=6 screen -L -Logfile /mnt/data/kw/hjy/logs/train/0827/vllm.log bash /mnt/data/kw/hjy/efficient-reasoning/vllm_employ.sh
CUDA_VISIBLE_DEVICES=4 screen -L -Logfile /mnt/data/kw/hjy/logs/train/0827/vllm_2.log bash /mnt/data/kw/hjy/efficient-reasoning/vllm_employ2.sh



## 启动训练脚本
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=5 screen -L -Logfile /mnt/data/kw/hjy/logs/train/0827/train.log python /mnt/data/kw/hjy/efficient-reasoning/fast_slow_example.py



CUDA_VISIBLE_DEVICES=0 python /mnt/data/kw/hjy/Effcient_Reasoning/code/examples/fast_slow_example_0822.py

CUDA_VISIBLE_DEVICES=1 bash /mnt/data/kw/hjy/Effcient_Reasoning/code/examples/vllm_employ2.sh

VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=2 python /mnt/data/kw/hjy/Effcient_Reasoning/code/examples/fast_slow_example_0822.py


CUDA_VISIBLE_DEVICES=6,7 screen -L -Logfile /data2/hanjin1/logs/efficient_reasoning/0823/infer.log python /data2/hanjin1/infer/hjy_run_batch.py
CUDA_VISIBLE_DEVICES=4,5 screen -L -Logfile /data2/hanjin1/logs/efficient_reasoning/0823/infer2.log python /data2/hanjin1/infer/hjy_run_batch2.py


CUDA_VISIBLE_DEVICES=6,7 screen -L -Logfile /data2/hanjin1/logs/efficient_reasoning/0823/eval.log python /data2/hanjin1/hjy_backup/inference/infer_code/eval_answer.py


CUDA_VISIBLE_DEVICES=0,1,2,3 screen -L -Logfile /data2/hanjin1/logs/efficient_reasoning/0825/infer.log python /data2/hanjin1/infer/hjy_run_batch.py
CUDA_VISIBLE_DEVICES=4,5,6,7 screen -L -Logfile /data2/hanjin1/logs/efficient_reasoning/0825/infer_v2.log python /data2/hanjin1/infer/hjy_run_batch2.py


screen -L -Logfile /data2/hanjin1/logs/efficient_reasoning/0825/infer.log python /data2/hanjin1/infer/run_batch_v3.py
screen -L -Logfile /data2/hanjin1/logs/efficient_reasoning/0825/infer_v2.log python /data2/hanjin1/infer/run_batch_v4.py








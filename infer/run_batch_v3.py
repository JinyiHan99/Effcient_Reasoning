import subprocess
import multiprocessing
import os

def run_infer(args):
    gpu_id, step = args
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    command = f"python /data2/hanjin1/hjy_backup/inference/infer_code/infer.py \
        --data_names gpqa aime24 math500 \
        --model_name /data2/hanjin1/ckp/auto_tool/Qwen2.5-7B-Instruct/0824_7B_compress_one_think/step_{step} \
        --save_path /data2/hanjin1/inference_results/0825/ \
        --prompt_type ckp"

    print(f"[GPU {gpu_id}] Running step {step}: {command}")
    completed_process = subprocess.run(command, shell=True, env=env)

    if completed_process.returncode != 0:
        print(f"[GPU {gpu_id}] Script failed with step {step}!!!")
    else:
        print(f"[GPU {gpu_id}] Script succeeded with step {step}!!!")

if __name__ == "__main__":
    # 需要运行的 step 列表
    steps = list(range(1000, 2600, 500))  # 1000,1500,2000,2500
    gpu_ids = list(range(4))  # GPU 0 ~ 4

    # 将 step 和 gpu 一一对应（假设 step 数量 <= GPU 数量）
    tasks = list(zip(gpu_ids, steps))

    with multiprocessing.Pool(processes=len(tasks)) as pool:
        pool.map(run_infer, tasks)

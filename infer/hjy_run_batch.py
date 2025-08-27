import subprocess

# for i in data_paths:
# for j in range(1000, 1100, 500):

# model_paths = ["/data2/Qwen/DeepSeek-R1-Distill-Qwen-7B","/data2/hanjin1/ckp/auto_tool/Qwen2.5-7B-Instruct/0723_7B_fast_slow_split_union_rule/step_500"]
# prompt_types = ['raw',"ckp"]

for i in range(500, 2100, 500):
    command = f"python /mnt/data/kw/hjy/Effcient_Reasoning/infer/infer_dp.py \
        --data_names gpqa aime24 math500 \
        --model_name /mnt/data/kw/hjy/ckp/0827_7B_compress_half/step_{i} \
        --save_path /mnt/data/kw/hjy/data/0827/ \
        --prompt_type ckp"
    print(f"Running: {command}")
    completed_process = subprocess.run(command, shell=True)
        # 检查执行情况
    if completed_process.returncode != 0:
        print(f"Script failed with params!!!")
    else:
        print(f"Script succeeded with params!!!")
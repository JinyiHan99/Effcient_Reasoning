import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import time
import json
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import ray
from typing import List, Dict, Any
import math

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_list_to_jsonl(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

DATA_PATH_MAP = {
    "gsm": "/data2/hanjin1/data/gsm/gsm8k_test_cleaned.jsonl",
    "aime24": "/data2/hanjin1/data/aime/aime24_test_cleaned.jsonl",
    "math500": "/data2/hanjin1/data/math/math500_test_cleaned.jsonl",
    "gpqa": "/data2/hanjin1/data/gpqa/gpqa_diamond_cleaned.jsonl",
    "aime25": "/data2/hanjin1/data/aime/aime2025_cleaned.jsonl",
}

def make_normal_prompt_fn_ckp(example, tokenizer):
    system_prompt = "You are a helpful AI assistant. A conversation takes place between the User and the Assistant. The User asks a question, and the Assistant solves it."
    policy_prompt = "Please help me solve this question. Wrap only the final answer in \\boxed{}."
    
    return tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'{policy_prompt}\nQuestion:{example["question"]}'}], 
        tokenize=False, add_generation_prompt=True)


def make_normal_prompt_fn(example, tokenizer):
    system_prompt = "You are a helpful AI assistant. A conversation takes place between the User and the Assistant. The User asks a question, and the Assistant solves it."
    content_prompt = "Please think step by step, and put your final answer in \\boxed{} \n"
    return tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'{content_prompt}\nQuestion:{example["question"]}'}], 
        tokenize=False, add_generation_prompt=True)

def make_gpqa_normal(example, tokenizer):
    system_prompt = "You are a helpful AI assistant. A conversation takes place between the User and the Assistant. The User asks a question, and the Assistant solves it."
    return tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Question:{example["question"]}'}], 
        tokenize=False, add_generation_prompt=True)

def make_gpqa_normal_ckp(example, tokenizer):
    system_prompt = "You are a helpful AI assistant. A conversation takes place between the User and the Assistant. The User asks a question, and the Assistant solves it."
    policy_prompt = "Please help me solve this question. Wrap only the final answer in \\boxed{}."
    question = example['question'].replace("Please write your final answer in the form of \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}","")
    return tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'{policy_prompt}\nQuestion:{question}'}], 
        tokenize=False, add_generation_prompt=True)

@ray.remote(num_gpus=1)
class SingleGPUWorker:
    """每个worker独占一个GPU，避免显存竞争"""
    def __init__(self, model_name: str, worker_id: int):
        self.worker_id = worker_id
        print(f"Worker {worker_id}: Loading model on dedicated GPU...")
        
        self.model = LLM(
            model=model_name,
            enable_chunked_prefill=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,  # 独占GPU时可以使用更高利用率
            enforce_eager=True,
            max_model_len=30000
        )
        self.tokenizer = self.model.get_tokenizer()
        print(f"Worker {worker_id}: Model loaded successfully!")
        
    def process_multiple_batches(self, batches_data: List[List[Dict]], 
                                prompt_type: str, data_name: str, 
                                sampling_params: Dict) -> List[Dict]:
        """处理多个批次的数据"""
        all_results = []
        
        # 选择prompt函数
        if prompt_type == "ckp":
            if data_name == "gpqa":
                make_prompt_fn = make_gpqa_normal_ckp
                print("prompt type: CKP+ GPQA")
            else:
                make_prompt_fn = make_normal_prompt_fn_ckp
                print("prompt type: CKP")
        else:
            if data_name == "gpqa":
                make_prompt_fn = make_gpqa_normal
                print("prompt type: RAW + GPQA")
            else:
                make_prompt_fn = make_normal_prompt_fn
                print("prompt type: RAW")

        
        vllm_sampling_params = SamplingParams(**sampling_params)
        
        for batch_data in batches_data:
            if not batch_data:
                continue
                
            # 创建prompts
            prompts = [make_prompt_fn(example, self.tokenizer) for example in batch_data]
            
            # 生成
            outputs = self.model.generate(prompts, vllm_sampling_params, use_tqdm=False)
            
            # 处理结果
            cur_results = []
            for j, output in enumerate(outputs):
                original_example = batch_data[j].copy()
                generated_text = output.outputs[0].text.replace("<|im_end|>", "").strip()
                original_example[f'{prompt_type}_model_answer'] = generated_text
                all_results.append(original_example)
                cur_results.append(original_example)
            if self.worker_id == 0:
                print(f"Question:\n{prompts[0]}\nAnswer:\n{cur_results[0][f'{prompt_type}_model_answer']}")
            # print(f"Question:\n{prompts[0]}\nAnswer:\n{cur_results[0][f'{prompt_type}_model_answer']}")
        
        return all_results

def split_data_for_gpu_workers(data: List[Dict], num_gpus: int, batch_size: int) -> List[List[List[Dict]]]:
    """为每个GPU worker分配数据批次"""
    # 首先创建所有批次
    all_batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        all_batches.append(batch)
    
    # 将批次分配给GPU workers
    worker_batches = [[] for _ in range(num_gpus)]
    for i, batch in enumerate(all_batches):
        worker_idx = i % num_gpus
        worker_batches[worker_idx].append(batch)
    
    return worker_batches

def main(args):
    # 检查GPU数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available!")
        return
    
    # 限制worker数量不超过GPU数量
    num_workers = min(args.num_workers, num_gpus)
    print(f"Using {num_workers} workers on {num_gpus} GPUs")
    
    # 初始化Ray
    if not ray.is_initialized():
        # 告诉Ray我们有多少GPU
        ray.init(num_gpus=num_gpus)
    
    # 创建GPU workers
    workers = []
    for i in range(num_workers):
        try:
            worker = SingleGPUWorker.remote(args.model_name, i)
            workers.append(worker)
            print(f"Created worker {i}")
        except Exception as e:
            print(f"Failed to create worker {i}: {e}")
    
    if not workers:
        print("No workers created successfully!")
        return
        
    print(f"Successfully created {len(workers)} workers!")
    
    # 采样参数
    sampling_params = {
        "temperature": 0.6,
        "max_tokens": 25000
    }
    
    # 处理每个数据集
    for data_name in args.data_names:
        data_path = DATA_PATH_MAP[data_name]
        start_time = time.time()
        
        print(f"========= Loading data from: {data_path}")
        data = read_jsonl(data_path)
        print(f"Data length: {len(data)}")
        
        # 准备保存路径
        model_name_short = args.model_name.split("/")[-1]
        dataset_name = os.path.basename(data_path).split('.')[0]
        save_data_path = os.path.join(args.save_path, f"{model_name_short}_{dataset_name}_results.jsonl")
        print(f"Results will be saved to: {save_data_path}")
        
        # 分配数据给workers
        worker_batches = split_data_for_gpu_workers(data, len(workers), args.batch_size)
        
        # 提交任务
        futures = []
        for worker_idx, worker in enumerate(workers):
            if worker_batches[worker_idx]:  # 确保有数据要处理
                future = worker.process_multiple_batches.remote(
                    worker_batches[worker_idx],
                    args.prompt_type,
                    data_name,
                    sampling_params
                )
                futures.append(future)
        
        print(f"Submitted tasks to {len(futures)} workers")
        
        # 收集结果
        all_results = []
        with tqdm(total=len(futures), desc=f"Processing {dataset_name}") as pbar:
            for future in futures:
                try:
                    worker_results = ray.get(future)
                    all_results.extend(worker_results)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error in worker: {e}")
                    pbar.update(1)
        
        # 保存结果
        save_list_to_jsonl(all_results, save_data_path)
        
        end_time = time.time()
        run_time = (end_time - start_time) / 3600
        print(f"\n========= Finished! Results saved to: {save_data_path}")
        print(f"Total running time: {run_time:.2f}h")
        print(f"Processed {len(all_results)} examples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-dedicated Ray parallel inference")
    parser.add_argument("--model_name", type=str, 
                       default="/data2/hanjin1/ckp/auto_tool/Qwen2.5-7B-Instruct/0715_7B_fast_slow_split_linear/step_900", 
                       help="Path to the model")
    parser.add_argument("--data_names", type=str, nargs='+', 
                       default=['aime24', "gpqa", "math500"], 
                       help="Data names to process")
    parser.add_argument("--save_path", type=str, 
                       default="/data2/hanjin1/hjy_backup/inference/0823/", 
                       help="Directory to save the results")
    parser.add_argument("--prompt_type", type=str, default="ckp", 
                       help="Prompt type")
    parser.add_argument("--num_workers", type=int, default=4, 
                       help="Number of GPU workers (max = num_gpus)")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for inference")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    try:
        main(args)
    finally:
        if ray.is_initialized():
            ray.shutdown()
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import time
import json
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse

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
        "gsm":"/data2/hanjin1/data/gsm/gsm8k_test_cleaned.jsonl",
        "aime24":"/data2/hanjin1/data/aime/aime24_test_cleaned.jsonl",
        "math500": "/data2/hanjin1/data/math/math500_test_cleaned.jsonl",
        "gpqa": "/data2/hanjin1/data/gpqa/gpqa_diamond_cleaned.jsonl",
        "aime25": "/data2/hanjin1/data/aime/aime2025_cleaned.jsonl",     
}

def make_normal_prompt_fn_ckp(example, tokenizer):
    # system_prompt='''You are a helpful AI assistant. The user asks a question, and the Assistant solves it.'''
    system_prompt =  "You are a helpful AI assistant. A conversation takes place between the User and the Assistant. The User asks a question, and the Assistant solves it."
    policy_prompt = "Please help me solve this question. Wrap only the final answer in \\boxed{}.",
    
    # question = example[question_key]
    return tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'{policy_prompt}\nQuestion:{example['question']}'}], 
            tokenize=False, add_generation_prompt=True)

def make_normal_prompt_fn(example, tokenizer):
    system_prompt =  "You are a helpful AI assistant. A conversation takes place between the User and the Assistant. The User asks a question, and the Assistant solves it."
    conent_prompt = "Please think step by step, and put your final answer in \\boxed{} \n"
    return tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'{conent_prompt}\nQuestion:{example['question']}'}], 
            tokenize=False, add_generation_prompt=True)

def make_gpqa_normal(example, tokenizer):
    system_prompt =  "You are a helpful AI assistant. A conversation takes place between the User and the Assistant. The User asks a question, and the Assistant solves it."
    return tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Question:{example['question']}'}], 
            tokenize=False, add_generation_prompt=True)

def compress_prompt(example, tokenizer):
    compression_prompt = """You are a helpful assistant. Here is a solution for a problem. Remove redundant or repetitive steps without changing the original logic, reasoning order, or core meaning. Requirements:
1. Do not add new reasoning steps or conclusions.
2. Do not change the original solution method or sequence.
3. Only compress or merge expressions to make the reasoning concise and clear.
4. Keep all original mathematical expressions or key formulas unchanged.
5. Output only the simplified solution, with no explanations, notes, or extra text.
6. The final answer is wrapped with \\boxed{}.
    """
    
    # prompts = []
    ans = example['model_answer']
    prompt = self.tokenizer.apply_chat_template([
            {"role": "system", "content": compression_prompt},
            {"role": "user", "content": "Here is the Solution:\n" + ans}], 
        tokenize=False, add_generation_prompt=True)
    return prompt
            

                



# --- 主函数 ---
def main(args):
    print("Loading model...")
    model = LLM(
        model=args.model_name, 
        enable_chunked_prefill=True, 
        tensor_parallel_size=1, 
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        max_model_len=30000
    )
    tokenizer = model.get_tokenizer()
    print("Model loaded successfully!")
   
    print(f"!!! All the data name:{args.data_names}")
    # for data_name in args.data_names:
        # data_path = DATA_PATH_MAP[data_name]
        start_time = time.time()
        # print(f"========= Loading data from: {data_path}")
    data = read_jsonl(args.data_path)
    print(f"=====The data length:{len(data)}")
    
    model_name_short = args.model_name.split("/")[-1]
    dataset_name = os.path.basename(data_path).split('.')[0]
    save_data_path = os.path.join(args.save_path, f"{model_name_short}_{dataset_name}_results.jsonl")
    print(f"========= Results will be saved to: {save_data_path}")
    
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens = 25000
    )

    batch_size = 16
    results_data = []
    
    question_key = "question"

    if args.prompt_type == "ckp":
        make_prompt_fn = make_normal_prompt_fn_ckp
        print("========= Using 'ckp' prompt function.")
    elif args.prompt_type == "compress":
        make_prompt_fn = compress_prompt
    else:
        make_prompt_fn = make_normal_prompt_fn
        print("========= Using 'normal' prompt function.")

    
    for i in tqdm(range(0, len(data), batch_size), desc=f"Processing {dataset_name}"):
        batch_data = data[i : i + batch_size]
        
        prompts = [make_prompt_fn(example, tokenizer) for example in batch_data]
        outputs = model.generate(prompts, sampling_params, use_tqdm=False)
        
        cur_ans = []
        for j, output in enumerate(outputs):
            original_example = batch_data[j]
            # --- 关键修正：清理生成的文本 ---
            # 移除可能残留在末尾的 stop token，让输出更干净
            generated_text = output.outputs[0].text.replace("<|im_end|>", "").strip()
            original_example[f'{args.prompt_type}_model_answer'] = generated_text
            cur_ans.append(generated_text)
            results_data.append(original_example)
        print(f"Question:\n{prompts[0]}\nAnswer:\n{cur_ans[0]}")

        save_list_to_jsonl(results_data, save_data_path)

        end_time = time.time()
        run_time = (end_time - start_time) / 3600
        print(f"\n========= Finished! Results saved to: {save_data_path}")
        print(f"====== Total running time: {run_time:.2f}h")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="None")
    parser.add_argument("--model_name", type=str, default="/data2/hanjin1/ckp/auto_tool/Qwen2.5-7B-Instruct/0715_7B_fast_slow_split_linear/step_900", help="Path to the model")
    # 默认数据路径现在指向您的 gsm8k 文件
    # parser.add_argument("--data_path", type=str, default='/data2/hanjin1/data/gpqa/gpqa_main.jsonl', help="Path to the GSM8K test file (.jsonl)")
    parser.add_argument("--data_path", type=str, default=['aime24',"gpqa","math500"], help="please specify the data name you used,['gsm','aime24','math500','gpqa']")
    parser.add_argument("--save_path", type=str, default="/data2/hanjin1/hjy_backup/inference/0823/", help="Directory to save the results")
    parser.add_argument("--prompt_type", type=str, default="ckp", help="Prompt type")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    main(args)
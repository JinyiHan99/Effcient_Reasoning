import random, os, sys, re,json
from config import tool_calls_gen, normal_gen, train_config, eval_sys_prompt, fast_slow_gen
import re
import copy,math, requests,time,uuid
from collections import defaultdict


from openai import OpenAI

prompt_template = '''
        ### Question:
        {Question}

        ### Ground Truth:
        {Ground_Truth}

        ### Answer:
        {Answer}
        '''
client = OpenAI(base_url="http://10.176.64.144:59827/v1/", api_key="dummy")

def llm_eval(ans, example):
    matches = re.findall(r'\\boxed\{(.*?)\}', ans)
    final_answer =  "\\boxed{" + matches[0] + "}" if matches else ans
    user_prompt = prompt_template.format(Question = example['question'], Ground_Truth = example['std'], Answer = final_answer)
    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[
            {"role": "system", "content": eval_sys_prompt['prompt']},
            {"role": "user", "content": user_prompt}],
        temperature=0.2,
        )
    res = response.choices[0].message.content
    try: 
        reward = float(res)

    except:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", res)
        if numbers:
        # 取最后一个数字并转换
            last_number = numbers[-1]
            reward = float(last_number)
        else:
        # 如果没有找到任何数字，返回0
            reward = 0.0
    finally:
    # 确保奖励在0-1之间
        reward = max(0.0, min(reward, 1.0))
    return reward


    


def format_fn(answer: str, item) -> float:
    # box_match = re.search(r'\\boxed\{.*?\}', answer)
    box_match = re.search(r'\\boxed\{.*?\}', answer)
    if not box_match:
        return -1.0
    return 1.0

from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

def choice_eval(model_answer, item):
    std_answer = item['std']
    std_match = re.search(r"\\boxed\{([A-Z])\}", std_answer)
    if not std_match:
        raise ValueError(f"标准答案格式错误: {std_answer}")
    std_choice = std_match.group(1)
    
    model_match = re.search(r"\\boxed\{([A-Z])\}", model_answer)
    if not model_match:
        return -1.0
    model_choice = model_match.group(1)

    return 1.0 if model_choice == std_choice else -1.0

def math_eval(answer, item):
    ground_truth_str = item['std']
    pattern = r'\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}'
    boxed_content_ans = re.findall(pattern, answer)

    if not boxed_content_ans:
        return -1.0
    final_ans_expr = "\\boxed{" + boxed_content_ans[-1] + "}"
    final_gt_expr = "\\boxed{" + ground_truth_str + "}"

    if final_ans_expr == final_gt_expr:
        return 1.0
    try:
        parsed_ans = parse(final_ans_expr)
        parsed_gt = parse(final_gt_expr)
        is_correct = verify(parsed_ans, parsed_gt)
        return 1.0 if is_correct else -1.0
    except Exception as e:
        return -1.0




def correct_fn(answer, item):
    if item['source'] == "gpqa":
        reward = choice_eval(answer, item)
    else:
        reward = math_eval(answer, item)
    return reward




def make_normal_prompt_fn(self, example):
    question = example['question']
    return self.tokenizer.apply_chat_template([
                {"role": "system", "content": fast_slow_gen['system_prompt']},
                {"role": "user", "content": f"{fast_slow_gen['policy_prompt']}\nQuestion:{question}"}], 
            tokenize=False, add_generation_prompt=True)

def make_rollout_prompt_fn(self, example, prompt_type):
    question = example['question']
    return self.tokenizer.apply_chat_template([
                {"role": "system", "content": fast_slow_gen['system_prompt']},
                {"role": "user", "content": f'{fast_slow_gen[prompt_type]}\nQuestion:{question}'}], 
            tokenize=False, add_generation_prompt=True)




# from lsrl import LSRL, RefServer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lsrl import LSRL, RefServer
model_path = "/mnt/data/kw/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
if 'ref' in sys.argv:
    RefServer(model_path).start()
    sys.exit(0)
    
from math_verify import parse, verify, ExprExtractionConfig

if __name__ == '__main__':
    # from datasets import load_dataset
    # dataset = load_dataset("meta-math/GSM8K_zh", "default", split="train")
    # QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question_zh'], dataset['answer'])]
    # with open("/data2/hanjin1/data/gsm/math/dapo_math.json",'r') as f:
    #     QAs = json.load(f)
    with open("/mnt/data/kw/hjy/data/mix-dapo-math-gpqa-main.json", 'r', encoding='utf-8') as file:
    # with open("/data2/hanjin1/hjy_backup/GRPO/data/mix_gsm_8k_math_4k.json",'r', encoding='utf-8') as file:
        QAs = json.load(file)
    print(f"训练的总长度：{len(QAs)}")
    # QAs = [{'Q': d['question'], 'A': d['answer'].split('####')[-1].strip()} for d in dataset]
    random.shuffle(QAs)

    lsrl = LSRL(model_path, epochs=1, train_data=QAs, rollout_num=9,  
                train_batch_size=3, gen_batch_size = 16,
                gen_update_steps=16, trainer='LSCPU', gen_temperature=1,
                gen_device=[2,3], ref_server="http://localhost:59887",
                lr=1e-6, 
                accum_steps=64, 
                swanlab_project="fast_slow_thinking",
                swanlab_name="0827_7B_compress_half",
                gen_max_tokens=4096,
                save_steps=500,
                save_path_fold ="/mnt/data/kw/hjy/ckp/0827_7B_compress_half",
                eval_llm_flag = False,
                genlog_filename = "/mnt/data/kw/hjy/logs/train/0827/7B_compress_half"
                )
    lsrl.add_reward(format_fn)
    if lsrl.eval_llm_flag==False:
        lsrl.add_reward(correct_fn)
        print("!!使用Rule来评估答案的正确性")
    else:
        lsrl.set_llm_eval_fn(llm_eval)
        print("!!!使用LLM 来评估答案的正确性")
        
    lsrl.set_policy_prompt_fn(make_normal_prompt_fn)

    lsrl.set_rollout_prompt_fn_normal(make_rollout_prompt_fn)
    ### 这里可以添加不同的prompt来让模型生成不同的
    # lsrl.set_rollout_prompt_fn_tool(make_tool_prompt_fn)

    lsrl.train()
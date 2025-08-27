
import re
import json

from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

def correct_fn(answer, ground_truth):
    ground_truth_str = str(ground_truth)
    pattern = r'\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}'
    boxed_content_ans = re.findall(pattern, answer)

    if not boxed_content_ans:
        return 0
    final_ans_expr = "\\boxed{" + boxed_content_ans[-1] + "}"
    if "\\boxed" not in ground_truth_str:
        final_gt_expr = "\\boxed{" + ground_truth_str + "}"
    else:
        final_gt_expr = ground_truth_str
    if final_ans_expr == final_gt_expr:
        return 1.0
    try:
        parsed_ans = parse(final_ans_expr)
        parsed_gt = parse(final_gt_expr)
        is_correct = verify(parsed_ans, parsed_gt)
        return 1.0 if is_correct else 0
    except Exception as e:
        return 0


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



from transformers import AutoTokenizer

def calculate_token_length(texts, tokenizer):
 
    token_lengths = []
    for t in texts:
        tokens = tokenizer.encode(t, add_special_tokens=False)
        token_lengths.append(len(tokens))
    return token_lengths




    

def cal_metrics(data_path, ans_key, std_key):
    model_path= "/data2/Qwen/DeepSeek-R1-Distill-Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data = read_jsonl(data_path)
    # print(len())
    hit = 1
    total = len(data)

    total_length = 0
    for example in data:
        ans = example[ans_key]
        std = example[std_key]
        ## cal the acc
        eval_res = correct_fn(ans, std)
        if eval_res > 0:
            hit += 1
        ## cal the token length
        ans_length = len(tokenizer.encode(ans, add_special_tokens=False))
        # import pdb
        # pdb.set_trace()
        total_length += ans_length
    
    acc = hit / total
    average_token_length = total_length / total
    print(f"{data_path}\n {total} {acc * 100:.2f}%  token_length: {average_token_length:.2f} ")

    # return acc, average_token_length
        

# data_path = "/data2/hanjin1/hjy_backup/inference/0823/DeepSeek-R1-Distill-Qwen-7B_aime24_test_cleaned_results.jsonl"
# data_path = "/data2/hanjin1/hjy_backup/inference/0823/DeepSeek-R1-Distill-Qwen-7B_math500_test_cleaned_results.jsonl"
# data_path = "/data2/hanjin1/hjy_backup/inference/DeepSeek-R1-Distill-Qwen-7B/DeepSeek-R1-Distill-Qwen-7B_math500_test_results.jsonl"
# data_path = "/data2/hanjin1/hjy_backup/inference/DeepSeek-R1-Distill-Qwen-7B/DeepSeek-R1-Distill-Qwen-7B_aime24_test_results.jsonl"
# data_path = "/data2/hanjin1/hjy_backup/inference/ckp_res/0711_1.5B_fast_slow_split_mix_3w_2.5w_origin/DeepSeek-R1-Distill-Qwen-7B_aime24_test_results.jsonl"

# data_path = "/data2/hanjin1/hjy_backup/inference/ckp_res/0723_DS_Distill_union_rule_3w_2.5w/step_500_aime24_test_results.jsonl"
# data_path = "/data2/hjy_backup/inference/ckp_res/0723_DS_Distill_union_rule_3w_2.5w/step_500_math500_test_results.jsonl"
# data_path = "/data2/hanjin1/hjy_backup/inference/ckp_res/0723_DS_Distill_union_rule_3w_2.5w/step_500_math500_test_results.jsonl"
# data_path = "/data2/hanjin1/hjy_backup/inference/0823/DeepSeek-R1-Distill-Qwen-7B_gpqa_diamond_cleaned_results.jsonl"

# data_paths =[ "/data2/hanjin1/hjy_backup/inference/0823/step_500_gpqa_diamond_cleaned_results_eval.jsonl",
# "/data2/hanjin1/hjy_backup/inference/0823/DeepSeek-R1-Distill-Qwen-7B_gpqa_diamond_cleaned_results_eval.jsonl"]
# for data_path in data_paths:

# data_path = "/data2/hanjin1/hjy_backup/inference/ckp_res/0711_1.5B_fast_slow_split_mix_3w_2.5w_origin/DeepSeek-R1-Distill-Qwen-7B_aime24_test_results.jsonl"
# data_paths = [
#     "//data2/hanjin1/inference_results/0825/v2/step_500_gpqa_diamond_cleaned_results.jsonl",
#     "/data2/hanjin1/inference_results/0825/v2/step_500_aime24_test_cleaned_results.jsonl",
#     "/data2/hanjin1/inference_results/0825/v2/step_500_math500_test_cleaned_results.jsonl"
# ]

# data_paths = [
#     "/data2/hanjin1/inference_results/0825/step_1500_gpqa_diamond_cleaned_results.jsonl"
# ]

for data_name in ["gpqa_diamond","aime24","math500"]:
    for i in range(500,2600, 500):
        # data_path = f"/data2/hanjin1/inference_results/0825/step_{i}_gpqa_diamond_cleaned_results.jsonl"
        # data_path =f"/data2/hanjin1/inference_results/0825/v2/step_{i}_aime24_test_cleaned_results.jsonl"
        if data_name =="gpqa_diamond":
            data_path = f"/data2/hanjin1/inference_results/0825/v2/step_{i}_{data_name}_cleaned_results.jsonl"
        else:
            data_path = f"/data2/hanjin1/inference_results/0825/v2/step_{i}_{data_name}_test_cleaned_results.jsonl"

        if i==500:
            cal_metrics(data_path, "ckp_model_answer", "std")
        else:
            cal_metrics(data_path, "model_answer", "std")

    



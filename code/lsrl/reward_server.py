import json, os, shutil, re, random, io, time, random, re, math
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import torch
from bottle import request
import bottle
import torch
import torch.nn as nn

from utils import json_to_bytes_list, bytes_list_to_json

class RewardServer:
    def __init__(self, model_path, host='0.0.0.0', port=59878):       
        self.app = bottle.Bottle()
        self.host = host
        self.port = port
        # self.init(model_path)
        
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = LLM(model_path,
            max_model_len=32768,
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.9
        )
        # allowed_tokens = ["Yes", "No"]
        # allowed_token_ids = [self.tokenizer.encode(token, add_special_tokens=False) for token in allowed_tokens]
        # self.Yes_tokens = Yes_tokens = allowed_token_ids[0]
        # No_tokens = allowed_token_ids[1]

        self.sampling_params = SamplingParams(
            temperature=0.6,
        )


    
    def get_reward(self, data):
        output, question = data.get('output', ''), data.get('question', '')
        
        def remove_explanation_prefix(text):
            explanation_pattern = r"^(```\s*)?Explanation:\s*"
            return re.sub(explanation_pattern, '', text)
        
        output = remove_explanation_prefix(output)

        def truncate_before_question(text):
            marker = "\n\nQuestion:\n"
            index = text.find(marker)
            if index != -1:
                return text[index + len(marker):]
            return text 
        result = truncate_before_question(question)

        def truncate_to_max_length(text, max_len=30000):
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
                text = self.tokenizer.decode(tokens)
            return text
        
        def calculate_sequence_probability(logprobs_list, token_ids):
            total_logprob = 0
            for i, token_id in enumerate(token_ids):
                if i < len(logprobs_list):
                    token_logprobs = logprobs_list[i]
                    if token_id in token_logprobs:
                        total_logprob += token_logprobs[token_id].logprob
                    else:
                        return float('-inf')
            return total_logprob

        prompt_description = "Given the following Question and the corresponding Answer provided by a model, you are required to assess whether the model is certain about its answer. If the model is certain about its answer, output 'Yes'. If the model is uncertain about its answer, output 'No'.\n\n"
        full_prompt = f"{prompt_description}Question:\n{result}\n\nModel's Answer:\n{output}"
        truncated_prompt = truncate_to_max_length(full_prompt)
        messages = [{"role": "user", "content": truncated_prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        output = self.model.generate(formatted_prompt, self.sampling_params)[0]
        generated_text = output.outputs[0].text.strip()
        generated_logprobs = output.outputs[0].logprobs
    
        Yes_logprob = calculate_sequence_probability(generated_logprobs, self.Yes_tokens)
        Yes_prob = math.exp(Yes_logprob)

        response = {
            "question": question,
            "certainty": generated_text,
            "certainty_probability": float(Yes_prob),
            "logprobs": {
                "token": generated_text,
                "probability": float(Yes_prob)
            },
            "reward": float(Yes_prob)
        }
        return response

    usr_prompt = '''
        ### Question:
        {Question}

        ### Ground Truth:
        {Ground_Truth}

        ### Answer:
        {Answer}
        '''
    def reward_correct(self, data):
        # eval_sys_prompt
        prompts = []
        for x in :
            question = [Message(role="system", content=x['sys_prompt']),Message(role="user", content=usr_prompt.format(Question = x['q'], Ground_Truth = x['std'], Answer = x['answer']))]
            prompts.append(question)
        scores = []
        responses = client.generate(prompts)
        for i, response in enumerate(responses):
            # print(f"Response: {response.content}")
            scores.append(response.content)
        final_scores = []
        for res in scores:
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
            final_scores.append(reward)
        return final_scores

            
    def run_server(self): 
        @self.app.route('/get_reward', method='POST')
        def get_reward():
            dd = request.body.read()
            data = bytes_list_to_json(dd)
            return self.get_reward(data)

        bottle.run(self.app, host=self.host, port=self.port, server='tornado')

    def start(self):
        self.run_server()

if __name__ == '__main__':
    RewardServer(model_path='/data2/Qwen/Qwen2.5-14B-Instruct').start()
    
    # class MyRS(RewardServer):
    #     def init(self, model_path):
    #         from transformers import AutoTokenizer
    #         from vllm import LLM, SamplingParams
    #         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    #         self.model = LLM(model_path,
    #             dtype='bfloat16', max_model_len=32768,
    #             tensor_parallel_size=1, enforce_eager=True
    #         )
    #         allowed_tokens = ["Yes", "No"]
    #         allowed_token_ids = [self.tokenizer.encode(token, add_special_tokens=False) for token in allowed_tokens]
    #         self.Yes_tokens = Yes_tokens = allowed_token_ids[0]
    #         No_tokens = allowed_token_ids[1]

    #         self.sampling_params = SamplingParams(
    #             temperature=0,
    #             max_tokens=1,
    #             stop="<|eot|>",
    #             logprobs=len(Yes_tokens + No_tokens),
    #             top_k=len(Yes_tokens + No_tokens),
    #             allowed_token_ids=Yes_tokens + No_tokens
    #         )

from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct, types
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
from .tool_executor import run
import math
import copy
import swanlab

from tqdm import tqdm
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from .utils import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

class LSTrainer:
    def __init__(self, model_patch):
        self.model = AutoModelForCausalLM.from_pretrained(model_patch, torch_dtype=torch.bfloat16)
        self.model.train()
    def backward(self, loss): loss.backward()
    def step(self): self.opt.step()
    def get_model(self): return self.model

class LSCPUTrainer(LSTrainer):
    def __init__(self, model_patch, lr=1e-6, accum_steps=16, grad_offload=True):
        super().__init__(model_patch)
        self.model.to('cuda')
        self.device = self.model.device
        self.model.gradient_checkpointing_enable()
        from .cpuadamw import CPUAdamW, DistributedCPUAdamW
        if dist.is_initialized(): CPUAdamW = DistributedCPUAdamW
        self.opt = CPUAdamW(self.model.parameters(), lr=lr, accum_steps=accum_steps, grad_offload=grad_offload)
        self.engine = self.model

class DeepSpeedTrainer(LSTrainer):
    def __init__(self, model_patch, ds_config=None, train_batch_size=3, lr=1e-6, accum_steps=16):
        super().__init__(model_patch)
        import deepspeed
        deepspeed.init_distributed()
        self.ds_config = self.get_default_ds_config() if ds_config is None else ds_config
        self.ds_config['train_micro_batch_size_per_gpu'] = train_batch_size
        self.ds_config['gradient_accumulation_steps'] = accum_steps
        self.ds_config['optimizer']['params']['lr'] = lr
        self.engine, _, _, _ = deepspeed.initialize(config=self.ds_config, model=self.model, 
                                                    model_parameters=self.model.parameters())
        self.device = self.engine.device
        self.opt = self.engine
    
    def get_model(self): return self.engine.module

    def backward(self, loss): self.engine.backward(loss)

    def get_default_ds_config(self):
        return {
            "optimizer": {
                "type": "AdamW",
                "params": { "lr": 1e-6 }
            },
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
                "stage3_gather_16bit_weights_on_model_save": True,
                "offload_optimizer": {"device": "cpu"}
            }
        }

def distbarrier():
    if dist.is_initialized(): dist.barrier()

class GenLogRecorder:
    def __init__(self, filename=None):
        self.base = filename or f"rl_log_{int(time.time())}"
    def parpare(self):
        self.md_file = open(f"{self.base}.md", 'w', encoding='utf-8')
        self.jsonl_file = open(f"{self.base}.jsonl", 'w', encoding='utf-8')
        self.md_file.write("# RL Training Log\n\n")       
    def log(self, iteration, question, samples, rewards):
        if not hasattr(self, 'md_file'): self.parpare()
        answers_texts = [x for x in samples]
        # tokens_lens = [len(x['token_ids']) for x in samples]
        ts = datetime.now().isoformat()
        self.md_file.write(f"## Iter {iteration}\n\n**Input:** {str(question)}\n\n")
        for i, (ans, reward) in enumerate(zip(answers_texts, rewards)):
            # parts = [f"total: {r['total']:.2f}"] + [f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
            #                                          for k, v in r.items() if k != 'total']
            # print(f"!!! 单个答案的得分情况：format:{reward['format_fn']} acc:{reward['correct_fn']} skill:{reward['skill']} cost:{reward['cost']} effiency:{reward['effiency']} total:{reward['total']}")
            
            parts = [f"total: {reward['total']:.2f}", 
                    f"format:{reward['format_fn']:.2f}",
                    f"acc:{reward['correct_fn']:.2f}",
                    f"skill:{reward['skill']:.2f}",
                    f"cost:{reward['cost']:.2f}",
                    f"effiency:{reward['tool_effiency_fn']:.2f}",
                    f"total:{reward['total']}:.2f"]
            self.md_file.write(f"### Answer {i} - {', '.join(parts)}```\n{ans}\n```\n\n")
        self.md_file.write("---\n\n")
        self.md_file.flush()
        self.jsonl_file.write(json.dumps({
            "iter": iteration, "Q": question, "anss": answers_texts, 
            "rewards": rewards
        }, ensure_ascii=False) + '\n')
        self.jsonl_file.flush()

class LSRL:
    def __init__(self, model_path, epochs=1, algorithm="GRPO", rollout_num=8, train_data=None, trainer='LSCPU',
                 gen_device=4, train_batch_size=2, gen_update_steps=16, save_steps=300, gen_batch_size=1,
                 beta=0.04, clip_param=0.2, compute_gen_logps=True, ref_server="http://localhost:59876",
                 gen_max_tokens=4096, gen_temperature=0.9, genlog_filename=None, save_path_fold = None,
                 swanlab_project = None, swanlab_name = None,eval_llm_flag=True, swanlab_key = None,
                 **kwargs):
        self.model_path = model_path
        self.gen_device = [gen_device] if isinstance(gen_device, int) else list(gen_device)
        # GPU device for generation, don't put it in CUDA_VISIBLE_DEVICES with deepspeed
        # TODO: add an assert to check gen_device is not in CUDA_VISIBLE_DEVICES
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.rollout_num = rollout_num
        self.train_data = train_data
        self.gen_batch_size = gen_batch_size
        self.epochs = epochs
        self.algorithm = algorithm
        self.save_path_fold = save_path_fold
        self.all_steps = epochs * len(train_data) * rollout_num // train_batch_size
        self.train_batch_size = train_batch_size
        assert rollout_num % train_batch_size == 0, "rollout_num must be divisible by train_batch_size"
        self.gen_update_steps = gen_update_steps
        self.save_steps = save_steps
        self.compute_gen_logps = compute_gen_logps
        self.generate_fn = None
        self.ref_server = ref_server
        self.gen_max_tokens = gen_max_tokens
        self.gen_temperature = gen_temperature
        self.beta = beta
        self.eval_llm_flag = eval_llm_flag
        self.clip_param = clip_param
        self.reward_fns = []
        self.swanlab_name = swanlab_name
        self.swanlab_project = swanlab_project
        self.swanlab_key = swanlab_key
        self.genlog_recorder = GenLogRecorder(genlog_filename) if genlog_filename else None

        if trainer == 'LSCPU':
            self.trainer = LSCPUTrainer(model_path, **kwargs)
        elif trainer == 'DeepSpeed':
            self.trainer = DeepSpeedTrainer(model_path, train_batch_size=train_batch_size, **kwargs)
        else:
            raise ValueError("Unsupported trainer type. Use 'LSCPU' or 'DeepSpeed'.")
    
    def add_reward(self, reward_fn):
        self.reward_fns.append(reward_fn)

    def set_rollout_prompt_fn_normal(self, user_fn): self._rollout_prompt_fn_normal = user_fn
    def set_rollout_prompt_fn_tool(self, user_fn): self._rollout_prompt_fn_tool = user_fn

    def set_policy_prompt_fn(self, user_fn): self._policy_prompt_fn = user_fn
    
    def set_llm_eval_fn(self, user_fn): self.__llm_eval_fn = user_fn

    def rollout_prompt_fn_normal(self, item): return self._rollout_prompt_fn_normal(self, item)
    def rollout_prompt_fn_tool(self, item): return self._rollout_prompt_fn_tool(self, item)


    def policy_prompt_fn(self, item): return self._policy_prompt_fn(self, item)
    def llm_eval_fn(self, answers, items): return self.__llm_eval_fn(answers, items)
    
    def get_batch(self):
        try:
            r = requests.get(f"{self.ref_server}/get").content
            if r == b'empty': return None
        except: return None
        dd = bytes_list_to_list(r)
        data = json.loads(dd[0]) 
        data['inputs'] = bytes_to_tensor(dd[1])
        data['rewards'] = bytes_to_tensor(dd[2])
        data['format'] = bytes_to_tensor(dd[3])
        data['acc'] = bytes_to_tensor(dd[4])
        data['cost'] =  bytes_to_tensor(dd[5])
        data['skill'] = bytes_to_tensor(dd[6])
        data['effiency'] = bytes_to_tensor(dd[7])
        data['refs'] = bytes_to_tensor(dd[8])
        if len(dd) == 10: 
            data['gen_logps'] = bytes_to_tensor(dd[9])
        return data
        
    def GRPO_step(self, model, batch):
        def get_per_token_logps(logits, input_ids):
            per_token_logps = [] # Use a loop to reduce memory peak.
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)
        
        prompt_length = batch['plen']
        inputs = batch['inputs'].to(self.device)
        #目标1: 计算format、acc、skill 的得分
        reward_format_acc_skill = (batch['acc'] + batch['format']).to(self.device)
        # advantages_acc = reward_format_acc_skill.unsqueeze(1)
        advantages_acc = (reward_format_acc_skill - reward_format_acc_skill.mean()) / (reward_format_acc_skill.std() + 1e-4)
        advantages_acc = advantages_acc.unsqueeze(1)
        #目标2: 计算cost的得分
        reward_cost = batch['cost'].to(self.device)
        advantages_cost = (reward_cost - reward_cost.mean()) /  (reward_cost.std() + 1e-4)
        advantages_cost = advantages_cost.unsqueeze(1)

        # advantages = batch['rewards'].to(self.device).unsqueeze(1)
        logits = model(inputs, use_cache=False).logits
        logits = logits[:, :-1, :]  
        input_ids = inputs[:, 1:]  
        per_token_logps = get_per_token_logps(logits, input_ids)
        per_token_logps = per_token_logps[:,prompt_length-1:]
        ref_per_token_logps = batch['refs'].to(per_token_logps.device)
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        completion_mask = (inputs[:, prompt_length:] != self.tokenizer.pad_token_id).int()
        if self.algorithm == "GRPO":
            if 'gen_logps' in batch:
                ratio = torch.exp(per_token_logps - batch['gen_logps'].to(self.device))
                clipped_ratio = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param)
                acc_per_token_loss = torch.min(ratio * advantages_acc, clipped_ratio * advantages_acc)
                cost_per_token_loss = torch.min(ratio * advantages_cost, clipped_ratio * advantages_cost)
                # per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
            else: 
                # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
                assert self.compute_gen_logps is False
            # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
            # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            print(f"!!!split 140")
            per_token_loss = -(acc_per_token_loss+cost_per_token_loss- self.beta * per_token_kl)
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        elif self.algorithm == "DAPO":
            if 'gen_logps' in batch:
                ratio = torch.exp(per_token_logps - batch['gen_logps'].to(self.device))
                clipped_ratio = torch.clamp(ratio, 1-self.clip_param-0.1, 1+self.clip_param)
                per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
            per_token_loss = -per_token_loss
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        else:
            raise NotImplementedError("Other RL algorithm is not supported!")
        return loss
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('trainer', None)
        return state
    ## V3版本:解决索引混乱的问题
    def generate_tool_insert(self, model, prompts, num, group_expressions, group_numbers):
        function_call_res = "The result of executing this funciton is:"
        stop_sentences = "</function>"
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            n=1,
            temperature=self.gen_temperature,
            max_tokens= 800,
            stop=stop_sentences,
            include_stop_str_in_output=True,
        )
        outputs = model.generate(prompts, sampling_params, use_tqdm=False)
        responses = [output.outputs[0].text for output in outputs]

        # 工具调用的最大次数，防止无限次调用下去
        if num > 8: 
            return responses

        recursive_ids = [
            i for i, c in enumerate(responses) if c.endswith(stop_sentences)
        ]

        if recursive_ids:
            recursive_prompts = []
            # 创建与recursive_prompts对应的表达式和数字列表
            recursive_expressions = []
            recursive_numbers = []

            for i in recursive_ids:
                # 执行工具调用
                try:
                    computed_result = run(responses[i], group_expressions[i], group_numbers[i])
                except Exception as e:
                    print(f"Error during tool execution: {e}")
                    computed_result = "Error in Tool Execution"

                # 将工具调用的结果融合到后续的推理过程中
                responses[i] += f"\n{function_call_res}{computed_result}"
                recursive_prompts.append(prompts[i] + responses[i])

                # 添加对应的表达式和数字
                recursive_expressions.append(group_expressions[i])
                recursive_numbers.append(group_numbers[i])

            # 递归调用时使用对应的子集数据
            rec_resps = self.generate_tool_insert(
                model, 
                recursive_prompts, 
                num+1, 
                recursive_expressions,  # 只传递需要继续处理的表达式
                recursive_numbers      # 只传递需要继续处理的数字
            )

            # 将递归结果合并回原始响应
            for idx, org_id in enumerate(recursive_ids):
                responses[org_id] += rec_resps[idx]

        return responses


    def generate_normal(self, vllm_gen, prompts, number):
        from vllm import SamplingParams
        sampling_params = SamplingParams(n = number, temperature = self.gen_temperature, 
                                         max_tokens = self.gen_max_tokens)
        voutputs = vllm_gen.generate(prompts, sampling_params, use_tqdm=False)
        answers = []
        for v in voutputs:
            for z in v.outputs: 
                # answers.append({'text':z.text, 'token_ids': z.token_ids})
                answers.append(z.text) #这里修改为只返回生成答案的文本，而不包括token_ids        
        # assert len(answers) == len(prompts) * self.rollout_num
        return answers

    def gen_worker(self, Q_data, Q_state_dict, gen_device, gen_rank=0):
        os.environ["CUDA_VISIBLE_DEVICES"] = f'{gen_device}'
        torch.cuda.set_device(0)
        print(f"Generation worker process uses GPU {gen_device}")
        from vllm import LLM, SamplingParams
        vllm_gen = LLM(model=self.model_path, enable_chunked_prefill=True, gpu_memory_utilization=0.8)
        gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

        def gen_samples(items):

            #:::::准备 正常生成答案的prompt以及答案
            gen_normal_num_single = math.floor(self.rollout_num / 3 *  1 )
            gen_prompts_normal = [self.rollout_prompt_fn_normal(x) for x in items]
            normal_answers = self.generate_normal(vllm_gen, gen_prompts_normal, gen_normal_num_single)
            print(f"!!!items的长度:{len(items)} normal answer的长度:{len(normal_answers)}")
           
            #::::: 准备 生成调用工具方式的prompt以及答案
            gen_tool_num_single = math.ceil(self.rollout_num / 3 * 2)
            tool_prompts = [self.rollout_prompt_fn_tool(x) for x in items]
            group_expressions = []
            group_numbers =[]
            for x in items:
                if x['gpt_flag']==1:
                    group_expressions.append(x['gpt'])
                    group_numbers.append(x['replace_number'])
                else:
                    group_expressions.append([])
                    group_numbers.append([])

            tool_prompts = tool_prompts * gen_tool_num_single
            group_expressions = group_expressions * gen_tool_num_single
            group_numbers = group_numbers * gen_tool_num_single
            tool_answers = self.generate_tool_insert(vllm_gen, tool_prompts, 0, group_expressions, group_numbers)
     
            answers = []
            group1 = [normal_answers[i:i+gen_normal_num_single] for i in range(0, len(normal_answers), gen_normal_num_single)]
            
      
            group2 = []
            for i in range(len(items)):  # 因为每组间隔是4，总共分4组
                temp =[]
                for j in range(i, len(tool_answers), len(items)):
                    temp.append(tool_answers[j])
                group2.append(temp)

            for g1, g2 in zip(group1, group2):
                answers.extend(g1 + g2) 
            # print(f"最终生成的答案长度:{len(answers)} 结果应该是items的长度✖️rollout {len(answers)==len(items)*self.rollout_num}")

            rewards = []
            #[hjy]ToDo:  使用api接口来判断模型回答的正误 输入：（answers, items） 返回 cur_acc_scores 
            cur_acc_scores = []
            if self.eval_llm_flag == True:
                cur_acc_scores = self.llm_eval_fn(answers, items)
                print(f"!!!cur_acc_scores_length:{len(cur_acc_scores)}")
            for i, ans in enumerate(answers):
                reward = {}
                for reward_fn in self.reward_fns: 
                    reward[reward_fn.__name__] = reward_fn(ans, items[i // self.rollout_num])
                
               
                if "correct_fn" not in self.reward_fns and len(cur_acc_scores)>0:
                    reward['correct_fn'] = cur_acc_scores[i]
                if "cost_and_skill_fn" in reward: #计算了cost 和 skill 的得分
                    temp =  copy.deepcopy(reward['cost_and_skill_fn'])
                    if reward['cost_and_skill_fn'][-1] ==0 and reward['correct_fn']<0:
                        reward['tool_effiency_fn'] = -0.3
                    elif reward['cost_and_skill_fn'][-1] >=6 and reward['correct_fn']<0:
                        reward['tool_effiency_fn'] = -0.2

                    elif reward['cost_and_skill_fn'][1] == 0 and reward['correct_fn']>0:
                        reward['tool_effiency_fn'] = 1    #没有使用工具，但是做对了
                    elif reward['cost_and_skill_fn'][1]<=0.5 and reward['correct_fn']>0:
                        reward['tool_effiency_fn'] = 1 #合理使用了工具在这里给出正向的奖励
                    else:
                        reward['tool_effiency_fn'] = 0
                        
                    reward['cost_and_skill_fn'] = reward['cost_and_skill_fn'][-2]
                    reward['skill'] = temp[0]
                    reward['cost'] = temp[1]
                else:  #reward 中不计算cost和skill的得分，只包括acc 和 format
                     reward['cost_and_skill_fn'] = 0
                     reward['skill'] = 0
                     reward['cost'] = 0
                     reward['tool_effiency_fn'] = 0

                reward['total'] = sum(reward.values())
                rewards.append(reward)
            policy_prompts = [self.policy_prompt_fn(x) for x in items]
            return {'prompts': policy_prompts, 'answers': answers, 'rewards': rewards}

        def QueueGetNowait(Q):
            try: return Q.get_nowait()
            except: return None

        curr_ver = -1
        def try_update_model():
            nonlocal curr_ver
            info = QueueGetNowait(Q_state_dict)
            if info is None: return
            ver, new_state_dict = info['ver'], info['sd']
            if ver > curr_ver: 
                curr_ver = ver
                print(f'[VLLM PROC {gen_rank}] recving new model ...')
                llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(new_state_dict.items())
                print(f'[VLLM PROC {gen_rank}] model updated')
            del new_state_dict
            
        rn = self.rollout_num
        tbsz = self.train_batch_size
        from torch.nn.utils.rnn import pad_sequence
        it = 0
        while True:
            items = QueueGetNowait(Q_data)
            if items is None: break
            if 'end' in items: 
                print('\nGeneration worker finished, sending end signal to ref server ...')
                data = [json.dumps({"end":1}).encode()] + [tensor_to_bytes(torch.tensor([0]))] * 4
                requests.post(f"{self.ref_server}/upload", data=make_bytes_list(data))            
                break
            it += 1
            if it % 2 == 0: try_update_model()
            tic = time.time()
            items = items['batch']
            samples = gen_samples(items)
            ### 在这里计算生成答案的token_ids
            ans_token_ids = self.tokenizer(samples['answers'], return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)['input_ids']
            # 如果生成的token过于长，这里直接抛弃掉
            if ans_token_ids.shape[1] > self.gen_max_tokens:
                print(f"[!!!Gen Warning] Too long!!  Shape:{ans_token_ids.shape}")
                continue
            if gen_rank == 0 and self.genlog_recorder:
                self.genlog_recorder.log(it, items[0], samples['answers'][:rn], samples['rewards'][:rn])
                # only log the first item in the batch for simplicity
          
            for i, _ in enumerate(items):
                prompt_ids = self.tokenizer(samples['prompts'][i], return_tensors="pt", add_special_tokens=False)["input_ids"]
                plen = prompt_ids.shape[1]
                # curr_ans_ids = [x['token_ids'] for x in samples['answers'][i*rn:(i+1)*rn]]
                curr_ans_ids = ans_token_ids[i*rn:(i+1)*rn]
                # curr_rewards = torch.tensor([x['total'] for x in samples['rewards'][i*rn:(i+1)*rn]], dtype=torch.float32)
                reward_slice = samples['rewards'][i * rn:(i + 1) * rn]
                keys = reward_slice[0].keys()
                reward_tensors_all = {
                    key: torch.tensor([x[key] for x in reward_slice], dtype=torch.float32)
                    for key in keys
                }
                curr_rewards = reward_tensors_all['total']
                cur_format_score = reward_tensors_all['format_fn']
                cur_acc_score = reward_tensors_all['correct_fn']
                cur_cost_score = reward_tensors_all['cost']
                cur_skill_score = reward_tensors_all['skill']
                cur_effiency_score = reward_tensors_all['tool_effiency_fn']

                print(f"!!!curr_ans_ids的形状:{curr_ans_ids.shape} curr_reward的形状:{curr_rewards.shape}")
                
                if i == 0: print(f'[GEN {gen_rank}]  time: {time.time()-tic:.2f}s    ', f'avg_rewards: {curr_rewards.mean().item():.2f}' )
                if curr_rewards.max() - curr_rewards.min() < 1e-4: continue
                curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)

                for ii in range(0, len(curr_ans_ids), tbsz):
                    sub_rewards = curr_rewards[ii:ii+tbsz]
                    sub_ans_ids = curr_ans_ids[ii:ii+tbsz]

                    sub_format = cur_format_score[ii:ii+tbsz]
                    sub_acc = cur_acc_score[ii:ii+tbsz]
                    sub_cost = cur_cost_score[ii:ii+tbsz]
                    sub_skill = cur_skill_score[ii:ii+tbsz]
                    sub_effiency = cur_effiency_score[ii:ii+tbsz]

                    tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                    output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=self.tokenizer.pad_token_id) 
                    Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
                    merged_ids = torch.cat([Qrep, output_ids], dim=1)
                    data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards),
                            tensor_to_bytes(sub_format),
                            tensor_to_bytes(sub_acc),
                            tensor_to_bytes(sub_cost),
                            tensor_to_bytes(sub_skill),
                            tensor_to_bytes(sub_effiency),
                            ]       

                    if self.compute_gen_logps:
                        zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                        zz = [xx.prompt_logprobs[plen:] for xx in zz]
                        gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                        data.append(tensor_to_bytes(gen_logps))

                    xdata = make_bytes_list(data)
                    requests.post(f"{self.ref_server}/upload", data=xdata)
                    print("Upload successfully!!")


    def start_gen_worker(self):
        print('\nSTART vLLM generation...\n')
        ctx = mp.get_context('spawn')
        self.Q_data = ctx.Queue()
        self.Q_state_dict = ctx.Queue()
        for epoch in range(self.epochs):
            items = list(self.train_data)
            for i in range(0, len(items), self.gen_batch_size):
                batch = items[i:i+self.gen_batch_size]
                self.Q_data.put({'batch': batch})
        self.Q_data.put({'end': 1}) 
        for it, gendevice in enumerate(self.gen_device):
            p = ctx.Process(target=self.gen_worker, args=(self.Q_data, self.Q_state_dict, gendevice, it))
            p.start()

    def train(self):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.rank == 0: self.start_gen_worker()

        self.device = self.trainer.device
        progress = range(1, self.all_steps+1)
        if self.rank == 0: progress = tqdm(progress)

        total_output_length = 0
   
        total_format = 0
        total_acc = 0
        total_cost = 0
        total_skill = 0
        total_effiency = 0
        total_num = 0
        # swanlab.login(key=self.swanlab_key)
        swanlab.init(project=self.swanlab_project, name=self.swanlab_name)
        for step in progress:
            batch = self.get_batch()
            start_time = time.time()
            while batch is None:
                if time.time() - start_time > 600: #如果等待超过10min，表明gen有问题或者整个训练已经结束，直接停止训练
                    print("[TRAIN]: no batch received in 10 minutes. Exiting...")
                    sys.exit(0)  # 或用 os._exit(1)
                print('[TRAIN] waiting for batch...'); time.sleep(5)
                batch = self.get_batch()
            if 'end' in batch: break
           
            ## 在这里使用swanlab统计情况

            if self.rank == 0:
                # print(f"batch中的内容:{batch.keys()}")
                batch_length = (batch['gen_logps'].shape[0] * batch['gen_logps'].shape[1])
                total_output_length += batch_length

                total_format += ( batch['format'] > 0).sum().item()
                total_acc += ( batch['acc'] > 0).sum().item()
                total_cost += ( batch['cost']).sum().item()
                total_skill += ( batch['skill']).sum().item()
                total_effiency += ( batch['effiency']).sum().item()
                total_num += batch['inputs'].shape[0]

                swanlab.log({"avg_token_lenght": float(total_output_length) / total_num,
                            "format_scores": float(total_format / total_num),
                            "acc_scores": float(total_acc) / total_num,
                            "cost_scores": float(total_cost) / total_num,
                            "skill_scores": float(total_skill) / total_num,
                            "effiency_scores": float(total_effiency) / total_num,
                    })
                print(f'!!!!get batch data successfully')

            tic = time.time()
            loss = self.GRPO_step(self.trainer.engine, batch)
            self.trainer.backward(loss)
            self.trainer.step()

            if self.rank == 0:
                progress.set_description(f"Loss: {loss.item():.6f}")
                print(f'[TRAIN] step: {step},  BATCH shape', batch['inputs'].shape, f'  time: {time.time()-tic:.2f}s')

            if step % self.gen_update_steps == 0:
                distbarrier()
                if self.rank == 0 and self.Q_state_dict.empty():
                    print('[TRAINING PROC] sending latest state_dict ...')
                    state_dict = self.trainer.get_model().state_dict()
                    for _ in range(len(self.gen_device)): self.Q_state_dict.put({'ver':step, 'sd': state_dict})
                    print('[TRAINING PROC] send state_dict ok!')

            if step % self.save_steps == 0:
                distbarrier()
                if self.rank == 0:
                    print('saving model')
                    save_name = f"{self.save_path_fold}/step_{step}"
                    state_dict = self.trainer.get_model().state_dict()
                    state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                    self.trainer.get_model().save_pretrained(save_name, state_dict=state_dict)
                    self.tokenizer.save_pretrained(save_name)
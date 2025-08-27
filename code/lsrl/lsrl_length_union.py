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
from itertools import chain

from tqdm import tqdm
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from .utils import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list, text_to_bytes

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
    def __init__(self, model_patch, ds_config=None, train_batch_size=2, lr=1e-6, accum_steps=16):
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
            parts = [
                    f"format:{reward['format_fn']:.2f}",
                    f"acc_score:{reward['correct_fn']:.2f}",
                    f"length_score:{reward['token_reward']:.2f}",
                    f"length:{reward['length']}"
                    ]
            self.md_file.write(f"### Answer {i} - {', '.join(parts)}```\n{ans}\n```\n\n")
        self.md_file.write("---\n\n")
        self.md_file.flush()
        self.jsonl_file.write(json.dumps({
            "iter": iteration, "Q": question, "ans": answers_texts, 
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

    def rollout_prompt_fn_normal(self, item, prompt_type): return self._rollout_prompt_fn_normal(self, item, prompt_type)
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
        data['length'] =  bytes_to_tensor(dd[5])
        data['refs'] = bytes_to_tensor(dd[6])
        data['gen_logps'] = bytes_to_tensor(dd[7])
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
        # #目标1: 计算format、acc的得分
        # reward_format_acc_skill = (batch['acc'] + batch['format']).to(self.device)
        # advantages_acc = reward_format_acc_skill.unsqueeze(1)
        # advantages_acc = (reward_format_acc_skill - reward_format_acc_skill.mean()) / (reward_format_acc_skill.std() + 1e-4)
        # advantages_acc = advantages_acc.unsqueeze(1)
        # #目标2: 计算length的得分
        # reward_length = batch['length'].to(self.device)
        # advantages_length = (reward_length - reward_length.mean()) /  (reward_length.std() + 1e-4)
        # advantages_length = advantages_length.unsqueeze(1)

        advantages = batch['rewards'].to(self.device).unsqueeze(1)
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
                # acc_per_token_loss = torch.min(ratio * advantages_acc, clipped_ratio * advantages_acc)
                # length_per_token_loss = torch.min(ratio * advantages_length, clipped_ratio * advantages_length)
                per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
            else: 
                # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
                assert self.compute_gen_logps is False
            per_token_loss = -(per_token_loss - self.beta * per_token_kl)
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            # per_token_loss = -(acc_per_token_loss+length_per_token_loss- self.beta * per_token_kl)
            # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        else:
            raise NotImplementedError("Other RL algorithm is not supported!")
        return loss
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('trainer', None)
        return state


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

            #ToDo:hjy  :::::准备 正常生成答案的prompt以及答案，这里的rollout应该包括各种各种类型的答案
            gen_normal_num_single = self.rollout_num // 3
            group_answers = []
            # #包括三种不同类型的prompt
            # for prompt_type in ["direct_answer","short_COT","long_COT"]:
            #     gen_prompts_normal = [self.rollout_prompt_fn_normal(x, prompt_type) for x in items]
            #     # print(f"prompt:{gen_prompts_normal[0]}")
            #     normal_answers = self.generate_normal(vllm_gen, gen_prompts_normal, gen_normal_num_single)
            #     group_answers.append(normal_answers)
            #     print(f"!!!items的长度:{len(items)} single normal answer的长度:{len(normal_answers)}")
            
     
            # group_chunks = [
            #     [lst[i:i + gen_normal_num_single] for i in range(0, len(lst), gen_normal_num_single)]
            #     for lst in group_answers
            # ]
            # temp= [
            #     list(chain.from_iterable(answer_group))
            #     for answer_group in zip(*group_chunks)
            # ]
            # answers = []
            # for ans in temp:
            #     answers.extend(ans)
          
            # print(f"!!! answer length:{len(answers)}")
                 # print(f"!!! answer length:{len(answers)}")
            # prompt_types = ["direct_answer", "short_COT", "long_COT"]
            # prompt_types = ["direct_answer", "short_COT", "long_COT"]
            prompt_types = ['policy_prompt','policy_prompt','policy_prompt']


    
            # Step 1: 为每个 item 和每种 prompt_type 生成对应的 prompt
            # 结构: prompts_by_type[type][item_idx] = [prompt] * gen_normal_num_single
            all_prompts = []
            prompt_to_item_index = []  # 记录每个 prompt 对应的原始 item 索引，用于后续重组

            for item_idx, item in enumerate(items):
                for prompt_type in prompt_types:
                    # 为当前 item 和 prompt_type 生成 gen_normal_num_single 个相同的 prompt
                    # 注意：这里每个 prompt 会被重复 gen_normal_num_single 次，因为要生成多个答案
                    prompt = self.rollout_prompt_fn_normal(item, prompt_type)
                    all_prompts.extend([prompt] * gen_normal_num_single)
                    prompt_to_item_index.extend([item_idx] * gen_normal_num_single)

            # Step 2: 一次性调用 generate_normal
            # 假设 generate_normal 接受一个 prompt 列表，返回对应长度的答案列表
            all_answers = self.generate_normal(vllm_gen, all_prompts, 1)

            # Step 3: 按原始 item 重组答案
            # 每个 item 应该有 rollout_num 个答案：来自三种类型，每种 gen_normal_num_single 个
            answers = [[] for _ in range(len(items))]

            for ans, item_idx in zip(all_answers, prompt_to_item_index):
                answers[item_idx].append(ans)

            # 展平为最终列表，保持顺序：item0 的 rollout_num 个答案，item1 的 ... 
            answers = list(chain.from_iterable(answers))

            print(f"!!! answer length: {len(answers)}")  # 应为 len(items) * self.rollout_num

            rewards = []
            cur_acc_scores = []
            if self.eval_llm_flag == True:
                for i, example in enumerate(items):
                    for j in range(self.rollout_num):
                        index = i*self.rollout_num + j
                        # print(f"!!!index:{index}")
                        cur_acc_scores.append(self.llm_eval_fn(answers[index], example))
                
                print(f"!!!cur_acc_scores_length:{len(cur_acc_scores)}")
            
            #计算每一个答案的长度
            answer_token_length = []
            for ans in answers:
                answer_token_length.append(len(self.tokenizer.encode(ans, add_special_tokens=False)))
            length_min = min(answer_token_length)
            length_max = max(answer_token_length)
            epsilon_l = 1e-8

            denom = (length_max - length_min + epsilon_l)
            #方法1: 线性归一化的线性奖励
            # token_length_rewards = [1 - (li - length_min) / denom for li in answer_token_length]
            
            ##方法2:
            # alpha = 0.5
            # token_length_rewards = []
            # for l in answer_token_length:
            #     norm = (l - length_min) / denom
            #     token_length_rewards.append(math.exp(-alpha * norm))
            
            for i, ans in enumerate(answers):
                reward = {}
                for reward_fn in self.reward_fns: 
                    reward[reward_fn.__name__] = reward_fn(ans, items[i // self.rollout_num])
                
                if "correct_fn" not in self.reward_fns and len(cur_acc_scores)>0:
                    reward['correct_fn'] = cur_acc_scores[i]
               

                length_reward = 0.5 - (answer_token_length[i] - length_min) / denom
                if reward['correct_fn']>0:  #只有答案正确的情况下，才去计算length的奖励得分
                    reward['token_reward'] = length_reward
                else:
                    # reward['token_reward'] = min(0, 0.5 - length_reward)
                    reward['token_reward'] = 0
                    
                # reward['total'] = sum(reward.values())
                reward['total'] = reward['correct_fn'] + reward['token_reward'] + reward['format_fn']
                reward['length'] = answer_token_length[i]


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
                print("!!!!上传到log文件中")
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
                cur_length_score = reward_tensors_all['token_reward']
                # print(f"!!!curr_ans_ids的形状:{curr_ans_ids.shape} curr_reward的形状:{curr_rewards.shape}")
                
                if i == 0: print(f'[GEN {gen_rank}]  time: {time.time()-tic:.2f}s    ', f'avg_rewards: {curr_rewards.mean().item():.2f}' )
                if curr_rewards.max() - curr_rewards.min() < 1e-4: continue
                curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)

                for ii in range(0, len(curr_ans_ids), tbsz):
                    sub_rewards = curr_rewards[ii:ii+tbsz]
                    sub_ans_ids = curr_ans_ids[ii:ii+tbsz]

                    sub_format = cur_format_score[ii:ii+tbsz]
                    sub_acc = cur_acc_score[ii:ii+tbsz]
                    sub_length = cur_length_score[ii:ii+tbsz]

                   
              

                    tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                    output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=self.tokenizer.pad_token_id) 
                    Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
                    merged_ids = torch.cat([Qrep, output_ids], dim=1)
                    data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards),
                            tensor_to_bytes(sub_format),
                            tensor_to_bytes(sub_acc),
                            tensor_to_bytes(sub_length),
                       
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
        # print("!!!准备vllm生成")
        if self.rank == 0: 
            self.start_gen_worker()

        self.device = self.trainer.device
        progress = range(1, self.all_steps+1)
        if self.rank == 0: progress = tqdm(progress)

        total_output_length = 0
   
        total_format = 0
        total_acc = 0
        # total_cost = 0
        total_ans_length = 0
        total_num = 0
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
                total_ans_length += ( batch['length']).sum().item()
                total_num += batch['inputs'].shape[0]

                swanlab.log({
                            "format_scores": float(total_format / total_num),
                            "acc_scores": float(total_acc) / total_num,
                            "answer_length": float(total_ans_length) / total_num,
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
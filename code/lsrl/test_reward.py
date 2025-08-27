import random, os, sys, re, time, requests, json, math

from utils import json_to_bytes_list
def correct_fn(answer, item):    
    return requests.post(f"http://127.0.0.1:59878/get_reward", 
                data=json_to_bytes_list({'output':answer, 
                                         'question':item})).json().get('reward', -1.0)

    
if __name__ == '__main__':
    with open('/data2/hanjin1/data/aime/aime24_test.jsonl', 'r', encoding='utf-8') as f:
        datas = [json.loads(line) for line in f.readlines()]
    # items = [x['prompt'][0]['content'] for x in datas]

    for data in datas:
        res = correct_fn(data['Problem'], data['Solution'])
        print(f"!!!here is the result:{res}\n\n")


   
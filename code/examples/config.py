tool_calls_gen = {
#     "system_prompt": """You are an intelligent AI assistant skilled in solving mathematical problems. You can leverage a set of predefined function tools to assist in computation and reasoning when necessary. Each function comes with a clear description, including its purpose, parameter meanings, and required format.
# When solving problems, follow these rules:
# 1. Understand and analyze the user's math question step by step.
# -During your reasoning, automatically decide whether a function should be used.
# -If the problem can be directly solved with logical steps, do not call a function.
# -If the problem needs calculation or a function can help reach the answer more clearly or efficiently, use the function.
# 2. When calling any function, Use <function> ... </function> tags to wrap the call. Inside, follow these rules:
# -Only use the system-provided functions.
# -Fill in all required parameters with correct names and data types (e.g., integers, floats).
# -Follow the function description exactly. Do not miss or rename any parameters.
# 3. Ensure reasoning efficiency and coherence:
#     -Avoid unnecessary or redundant function calls.
#     -Prefer reasoning paths that are shorter and more computationally efficient.
#     -If a conclusion can be logically derived without calling a function, do so directly.
# 4. Wrap your final answer in <answer></answer> tags, like <answer> Here is an answer at the end of the thinking process.</answer> 
# """,
#     "user_prompt":[{"name":"math.add","description":"Calculate the sum of two numbers.","parameters":{"type":"object","properties":{"a":{"type":"number","description":"The first addend."},"b":{"type":"number","description":"The second addend."}},"required":["a","b"]}},{"name":"math.subtract","description":"Calculate the difference between two numbers.","parameters":{"type":"object","properties":{"a":{"type":"number","description":"The minuend."},"b":{"type":"number","description":"The subtrahend."}},"required":["a","b"]}},{"name":"math.multiply","description":"Calculate the product of two numbers.","parameters":{"type":"object","properties":{"a":{"type":"number","description":"The first factor."},"b":{"type":"number","description":"The second factor."}},"required":["a","b"]}},{"name":"math.divide","description":"Calculate the quotient of dividing one number by another.","parameters":{"type":"object","properties":{"a":{"type":"number","description":"The dividend."},"b":{"type":"number","description":"The divisor. Must not be zero."}},"required":["a","b"]}}]
# }

# normal_gen = {
#     "system_prompt":'''You are a helpful AI assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
# Wrap your final answer in <answer></answer> tags, like <answer> Here is an answer at the end of the thinking process.</answer> '''
# }
    "system_prompt": """You are a smart AI assistant for solving math problems. You can use a set of predefined functions, each with a clear description of its purpose and parameters.
Follow these rules:
- Think step by step:
    - During your reasoning, automatically decide whether a function should be used.
    - If the problem can be directly solved with logical steps, do not call a function.
    - If the problem needs calculation or a function can help reach the answer more clearly or efficiently, use the function.
- When calling a function:
   - Wrap it in <function> ... </function> tags.
   - Use only the provided functions.
   - Fill in all required parameters exactly as described (correct names and types).
- Keep your reasoning efficient:
   - Prefer shorter, more computationally efficient solutions.
   - If a conclusion can be logically derived without calling a function, do so directly.
- Output the final answer inside \\boxed{}.
""",
    "user_prompt":[{"name":"math.add","description":"Calculate the sum of two numbers.","parameters":{"type":"object","properties":{"a":{"type":"number","description":"The first addend."},"b":{"type":"number","description":"The second addend."}},"required":["a","b"]}},{"name":"math.subtract","description":"Calculate the difference between two numbers.","parameters":{"type":"object","properties":{"a":{"type":"number","description":"The minuend."},"b":{"type":"number","description":"The subtrahend."}},"required":["a","b"]}},{"name":"math.multiply","description":"Calculate the product of two numbers.","parameters":{"type":"object","properties":{"a":{"type":"number","description":"The first factor."},"b":{"type":"number","description":"The second factor."}},"required":["a","b"]}},{"name":"math.divide","description":"Calculate the quotient of dividing one number by another.","parameters":{"type":"object","properties":{"a":{"type":"number","description":"The dividend."},"b":{"type":"number","description":"The divisor. Must not be zero."}},"required":["a","b"]}}]
}

normal_gen = {
    "system_prompt":'''You are a helpful AI assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
Put your final answer within \\boxed{}''',
    "direct_answer":"",
    "short_COT":"",
    "long_COT":"",
}

###这里补充不同的rollout prompt

fast_slow_gen = {
    "system_prompt": "You are a helpful AI assistant. A conversation takes place between the User and the Assistant. The User asks a question, and the Assistant solves it.",
    "policy_prompt": "Please help me solve this question. Wrap only the final answer in \\boxed{}.",

    "direct_answer": "Provide only the final answer to the question. Do not include any explanations or intermediate steps. Wrap the final answer in \\boxed{}. Keep the entire response under 100 tokens.",

   "short_COT": "Briefly think through the problem and show only the essential steps or logic. Conclude with the final answer wrapped in \\boxed{}. Keep the response under 500 tokens.",

   "long_COT": "Solve the problem step by step, explaining your reasoning and showing all relevant calculations. Justify each step clearly. At the end, write the final answer wrapped in \\boxed{}."

}

# fast_slow_gen = {
#     "system_prompt":'''You are a helpful AI assistant. The user asks a question, and the Assistant solves it.''',
#     "direct_answer":"Directly provide the final answer. Omit all reasoning and steps. \ Your response must only contain the answer enclosed in \\boxed{}. Limit your output to at most 100 tokens.",
#     "short_COT":"Please reason step by step, and put your final answer within \\boxed{}. Limit your response to at most 500 tokens.",
#     "long_COT":"Provide a comprehensive, step-by-step explanation. Break the problem down, explain the underlying principles, show all your work, and justify each step. \
#     After the detailed explanation, put your final answer within \\boxed{}. Limit your output to at most 2000 tokens."
# }

# fast_slow_gen = {
#     "system_prompt":'''You are a helpful AI assistant. The user asks a question, and the Assistant solves it.''',
#     "direct_answer": "Give only the final answer. Do not explain. Do not show steps. Only output the result inside \\boxed{}.",
#     "short_COT":"Think briefly and explain the main steps in several sentences. Then give the final answer inside \\boxed{}.",
#     "long_COT":"Provide a comprehensive, step-by-step explanation. Break the problem down, explain the underlying principles, show all your work, and justify each step. \
#         After the detailed explanation, state the final answer in the format \\boxed{}."
# }
train_config = {
    "rollout_num":9
}

eval_sys_prompt = {
    "prompt":'''Now, I want to test an AI assistant's ability to answer questions.
Below is a question, a ground truth answer, and a final answer generated by the AI assistant, which is wrapped in \\boxed{}.
Please rate the AI assistant's final answer according to the ground truth answer.
If you think the answer is correct, your output is 1; otherwise, your output is -1.
Your output is just -1 or 1.'''
}

import sys, signal, traceback
# from executor import _test
import operator,re, io

def str_to_number(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"无法将 '{s}' 转换为数字")

import ast

def extract_args(func_str):
    """
    解析函数调用字符串中的参数，返回 (位置参数列表, 关键字参数字典)
    示例：
        "math.add(1, 2)"        -> ([1, 2], {})
        "func(a=1, b='x')"      -> ([], {'a': 1, 'b': 'x'})
        "func(1, y='abc')"      -> ([1], {'y': 'abc'})
    """
    match = re.search(r'\((.*)\)', func_str)
    if not match:
        return [], {}

    args_str = match.group(1).strip()

    # 若为空直接返回空参数
    if not args_str:
        return [], {}

    try:
        # 使用 ast 解析为 tuple 表达式
        parsed = ast.parse(f"f({args_str})", mode='eval')
        call_node = parsed.body  # ast.Call
        if not isinstance(call_node, ast.Call):
            return [], {}
        
        pos_args = []
        for arg in call_node.args:
            pos_args.append(ast.literal_eval(arg))

        kw_args = {}
        for kw in call_node.keywords:
            kw_args[kw.arg] = ast.literal_eval(kw.value)

        return pos_args, kw_args
    except Exception as e:
        # raise ValueError(f"参数解析失败: {e}")
        return [],{}
        pass

def handler(signum, frame):
    raise TimeoutError("Code execution timed out")
# 数学操作映射表
math_ops = {
    "math.add": operator.add,
    "math.subtract": operator.sub,
    "math.multiply": operator.mul,
    "math.divide": operator.truediv,
}

def evaluate_expression(expressions, numbers):
    # print(f"使用表达式来解决这个问题:\nraw expression:\n{expressions}\nnumbers: {numbers}")

    keys = [chr(97 + i) for i in range(len(numbers))]  # a, b, c...
    value_map = dict(zip(keys, map(str, numbers)))

    for key, val in value_map.items():
        expressions = expressions.replace(key, val)
    expressions += "\nprint(r)"
    # print(f"\ncode表达式:\n{expressions}")

    output_capture = io.StringIO()
    sys.stdout = output_capture
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(1)
        exec(expressions, {}, {})
    except TimeoutError:
        return "Error! The Code Execution timeout!"
    except Exception as e:
        return f"Error! {type(e).__name__}: {str(e)}"
    finally:
        signal.alarm(0)
        sys.stdout = sys.__stdout__

    output = output_capture.getvalue().strip()
    return output if output else "Error! No output"

def run(input_string, expressions, numbers):
    match = re.search(r'<function>(.*?)</function>', input_string, re.DOTALL)
    if not match:
        return "Error! No complete function description."
    function_call_name = match.group(1)

    pos_args, kw_args = extract_args(function_call_name)
    result = None
    try:
        if any(op in function_call_name for op in math_ops):
            op_key = next(op for op in math_ops if op in function_call_name)
            if len(pos_args) == 2:
                result = math_ops[op_key](*pos_args)
            elif len(kw_args) == 2:
                result = math_ops[op_key](kw_args.get('a'), kw_args.get('b'))
            else:
                result = "Error! The number of parameters does not match!"
        elif "math.solve_with_python" in function_call_name:
            result = evaluate_expression(expressions, numbers)
        else:
            result = "Error! Unsupported function call: " + function_call_name
    except Exception:
        result = "Error!" + traceback.format_exc().strip().split("\n")[-1]

    return str(result)



# print(run("<function>math.add(a=3, 2)</function>", "", []))           # 3.0
# print(run("<function>math.multiply(2.5, 4)</function>", "", []))    # 10.0
# print(run("<function>math.subtract(a=10, b=3)</function>", "", [])) # 7.0
# print(run("<function>math.solve_with_python('question')</function>", "r = a*b", [2,4]))        
# print(run("<function>math.devide('question', 2)</function>", "", []))        # 2.5


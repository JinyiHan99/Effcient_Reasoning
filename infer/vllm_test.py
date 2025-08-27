from openai import OpenAI
import time

def compress_half_answers(port, retries=10, delay=2):
    """
    Compress answers by sending requests to a vLLM server with retry mechanism.
    
    :param port: Port of the vLLM server
    :param retries: Number of retry attempts if request fails
    :param delay: Delay in seconds between retries
    :return: List of compressed answers
    """
    print(f"!!!! 链接{port}的vllm")
    client = OpenAI(
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="EMPTY"
    )

    answers = ["how to keep healthy?", "how to keep healthy?"]
    compressed_answers = []

    for ans in answers:
        attempt = 0
        while attempt < retries:
            try:
                response = client.chat.completions.create(
                    model="/data2/Qwen/Qwen2.5-7B-Instruct",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": ans}
                    ],
                    max_tokens=2048,
                    temperature=0.01
                )
                # 兼容返回格式，取出内容
                content = (
                    response.choices[0].message.content
                    if hasattr(response.choices[0].message, "content")
                    else response.choices[0].message["content"]
                )
                compressed_answers.append(content.strip())
                break  # 成功则跳出重试循环
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed for answer '{ans}': {e}")
                if attempt < retries:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Failed after {retries} attempts. Skipping this answer.")
                    compressed_answers.append(None)

    return compressed_answers

ans2 = compress_half_answers(59824)
print(ans2)


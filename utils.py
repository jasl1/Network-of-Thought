import os
import json
import time
import openai
#import anthropic
from openai import OpenAI
from vllm import LLM, SamplingParams
from typing import Union, List
from dotenv import load_dotenv
load_dotenv()


def get_env_variable(): #var_name
    """Fetch environment variable or return None if not set."""
    
    client = OpenAI(api_key="Your_API_KEY")
    openai.organization = "org-XXX"
    return client

CALL_SLEEP = 1
clients = {}

def initialize_clients():
    """Dynamically initialize available clients based on environment variables."""
    try:
        client = get_env_variable()
        clients['gpt'] = client
        '''gpt_api_key = get_env_variable('GPT_API_KEY')
        gpt_base_url = get_env_variable('BASE_URL_GPT')
        if gpt_api_key and gpt_base_url:
            clients['gpt'] = OpenAI(base_url=gpt_base_url, api_key=gpt_api_key)
        '''

        #claude_api_key = get_env_variable('CLAUDE_API_KEY')
        #claude_base_url = get_env_variable('BASE_URL_CLAUDE')
        #if claude_api_key and claude_base_url:
        #    clients['claude'] = OpenAI(base_url=claude_base_url, api_key=claude_api_key)
        
        #if claude_api_key:
        #            clients['claude'] = anthropic.Anthropic(api_key=claude_api_key)
            
        '''
        deepseek_api_key = get_env_variable('DEEPSEEK_API_KEY')
        deepseek_base_url = get_env_variable('BASE_URL_DEEPSEEK')
        if deepseek_api_key and deepseek_base_url:
            clients['deepseek'] = OpenAI(base_url=deepseek_base_url, api_key=deepseek_api_key)

        deepinfra_api_key = get_env_variable('DEEPINFRA_API_KEY')
        deepinfra_base_url = get_env_variable('BASE_URL_DEEPINFRA')
        if deepinfra_api_key and deepinfra_base_url:
            clients['deepinfra'] = OpenAI(base_url=deepinfra_base_url, api_key=deepinfra_api_key)'''

        if not clients:
            print("No valid API credentials found. Exiting.")
            exit(1)

    except Exception as e:
        print(f"Error during client initialization: {e}")
        exit(1)

def initialize_open_source_llms():
    llama_model = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    trust_remote_code=True,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.49,
    enforce_eager=False,
    swap_space=8,
    max_num_seqs=128,
    dtype='bfloat16',
    disable_custom_all_reduce=False
    )

    mistral_model = LLM(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        trust_remote_code=True,
        gpu_memory_utilization=0.28,
        enforce_eager=False,
        swap_space=8,
        max_num_seqs=64,
        dtype='bfloat16',
        disable_custom_all_reduce=False,
        max_model_len=4096
    )

    return llama_model, mistral_model

initialize_clients()
#llama_model, mistral_model = initialize_open_source_llms()

def get_client(model_name):
    """Select appropriate client based on the given model name."""
    if 'gpt' in model_name or 'o1-'in model_name:
        client = clients.get('gpt') 
    elif 'claude' in model_name:
        client = clients.get('claude')
    elif 'deepseek' in model_name:
        client = clients.get('deepseek')
    elif any(keyword in model_name.lower() for keyword in ['llama', 'qwen', 'mistral', 'microsoft']):
        client = clients.get('deepinfra')
    else:
        raise ValueError(f"Unsupported or unknown model name: {model_name}")

    if not client:
        raise ValueError(f"{model_name} client is not available.")
    return client 

def read_prompt_from_file(filename):
    with open(filename, 'r') as file:
        prompt = file.read()
    return prompt

def read_data_from_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def parse_json(output):
    try:
        output = ''.join(output.splitlines())
        if '{' in output and '}' in output:
            start = output.index('{')
            end = output.rindex('}')
            output = output[start:end + 1]
        data = json.loads(output)
        return data
    except Exception as e:
        print("parse_json:", e)
        return None
    
def check_file(file_path):
    if os.path.exists(file_path):
        return file_path
    else:
        raise IOError(f"File not found error: {file_path}.")


def open_source_llm_call(query: Union[List, str], model_name: str):
    
    #sampling_params = SamplingParams(max_tokens=600, temperature=1.3, top_p=0.95, top_k=50, repetition_penalty=1.05) # hyper-parameter optimization
    sampling_params = SamplingParams(max_tokens=600, temperature=1)
    
    llm_model = llama_model if 'llama' in model_name else mistral_model

    for _ in range(3):
        try:
            completion = llm_model.generate(query, sampling_params)
            if isinstance(query, str):
                return completion[0].outputs[0].text
            elif isinstance(query, list):
                return [response.outputs[0].text for response in completion]   
        except Exception as e:
            print(f"Open-source llm call error: {e}")
            time.sleep(CALL_SLEEP)
    return ""

def open_source_llm_append(model_name, dialog_hist: List, query: str):
    dialog_hist.append({"role": "user", "content": query})
    resp = open_source_llm_call(query, model_name=model_name)
    dialog_hist.append({"role": "assistant", "content": resp})
    return resp, dialog_hist
    
        
def gpt_call(client, query: Union[List, str], model_name = "gpt-4o", temperature = 0):
    if isinstance(query, List):
        messages = query
    elif isinstance(query, str):
        messages = [{"role": "user", "content": query}]
        
    for _ in range(3):
        try:
            if 'o1-' in model_name:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )
            else:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature
                )
            resp = completion.choices[0].message.content
            return resp
        except Exception as e:
            print(f"GPT_CALL Error: {model_name}:{e}")
            time.sleep(CALL_SLEEP)
            continue
    return ""

def gpt_call_append(client, model_name, dialog_hist: List, query: str):
    dialog_hist.append({"role": "user", "content": query})
    resp = gpt_call(client, dialog_hist, model_name=model_name)
    dialog_hist.append({"role": "assistant", "content": resp})
    return resp, dialog_hist
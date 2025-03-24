from openai import OpenAI
import os
from . import statics

MODEL_FEE = {
    'o1-preview': {
        'prompt_fee': 0.015 * 7,
        'completion_fee': 0.06 * 7,
    },
    'o1-mini': {
        'prompt_fee': 0.003 * 7,
        'completion_fee': 0.012 * 7,
    },
    'gpt-4o': {
        'prompt_fee': 0.0025 * 7,
        'completion_fee': 0.01 * 7,
    },
    'gpt-4o-2024-11-20': {
        'prompt_fee': 0.0025 * 7,
        'completion_fee': 0.01 * 7,
    },
    'gpt-4o-2024-08-06': {
        'prompt_fee': 0.0025 * 7,
        'completion_fee': 0.01 * 7,
    },
    'gpt-4o-2024-05-13': {
        'prompt_fee': 0.005 * 7,
        'completion_fee': 0.015 * 7,
    },
    'gpt-4o-mini': {
        'prompt_fee': 0.15/1000 * 7,
        'completion_fee': 0.6/1000 * 7,
    },
    'gpt-4o-mini-2024-07-18': {
        'prompt_fee': 0.15/1000 * 7,
        'completion_fee': 0.6/1000 * 7,
    },
    'gpt-4-vision-preview': {
        'prompt_fee': 0.01 * 7,
        'completion_fee': 0.03 * 7,
    },
    'gpt-4-32k': {
        'prompt_fee': 0.06 * 7,
        'completion_fee': 0.12 * 7,
    },
    'gpt-4-turbo': {
        'prompt_fee': 0.01 * 7,
        'completion_fee': 0.03 * 7,
    },
    'gpt-4': {
        'prompt_fee': 0.03 * 7,
        'completion_fee': 0.06 * 7,
    },
    'gpt-4-32k': {
        'prompt_fee': 0.06 * 7,
        'completion_fee': 0.12 * 7,
    },
    'gpt-3.5-turbo': {
        'prompt_fee': 0.0005 * 7,
        'completion_fee': 0.0015 * 7,
    },
     'gpt-3.5-0125': {
        'prompt_fee': 0.0005 * 7,
        'completion_fee': 0.0015 * 7,
    },
    'gpt-3.5-turbo-instruct': {
        'prompt_fee': 0.0015 * 7,
        'completion_fee': 0.002 * 7,
    },
    'moonshot-v1-8k': {
        'prompt_fee': 0.012,
        'completion_fee': 0.012,
    },
    'moonshot-v1-32k': {
        'prompt_fee': 0.024,
        'completion_fee': 0.024,
    },
    'moonshot-v1-128k': {
        'prompt_fee': 0.06,
        'completion_fee': 0.06,
    },
    'yi-lightning': {
        'prompt_fee': 0.99/1000,
        'completion_fee': 0.99/1000,
    },
    'yi-large': {
        'prompt_fee': 20/1000,
        'completion_fee': 20/1000,
    },
    'yi-medium': {
        'prompt_fee': 2.5/1000,
        'completion_fee': 2.5/1000,
    },
    'yi-vision': {
        'prompt_fee': 6/1000,
        'completion_fee': 6/1000,
    },
    'yi-medium-200k': {
        'prompt_fee': 12/1000,
        'completion_fee': 12/1000,
    },
    'yi-spark': {
        'prompt_fee': 1/1000,
        'completion_fee': 1/1000,
    },
    'yi-large-rag': {
        'prompt_fee': 25/1000,
        'completion_fee': 25/1000,
    },
    'yi-large-fc': {
        'prompt_fee': 20/1000,
        'completion_fee': 20/1000,
    },
    'yi-large-turbo': {
        'prompt_fee': 12/1000,
        'completion_fee': 12/1000,
    },
    'deepseek-chat': {
        'prompt_fee': 2/1000,
        'completion_fee': 8/1000,
    },
    'deepseek-reasoner': {
        'prompt_fee': 4/1000,
        'completion_fee': 16/1000,
    },
    'deepseek-coder': {
        'prompt_fee': 1/1000,
        'completion_fee': 2/1000,
    },
    'step-1-8k': {
        'prompt_fee': 5/1000,
        'completion_fee': 20/1000,
    },
    'step-1-32k': {
        'prompt_fee': 15/1000,
        'completion_fee': 70/1000,
    },
    'step-1-128k': {
        'prompt_fee': 40/1000,
        'completion_fee': 200/1000,
    },
    'step-1-256k': {
        'prompt_fee': 95/1000,
        'completion_fee': 300/1000,
    },
     'step-1-flash': {
        'prompt_fee': 1/1000,
        'completion_fee': 4/1000,
    },
    'step-2-16k': {
        'prompt_fee': 38/1000,
        'completion_fee': 120/1000,
    },
    'step-1v-8k': {
        'prompt_fee': 5/1000,
        'completion_fee': 20/1000,
    },
    'step-1v-32k': {
        'prompt_fee': 15/1000,
        'completion_fee': 70/1000,
    },
    'step-1.5v-mini': {
        'prompt_fee': 8/1000,
        'completion_fee': 35/1000,
    },
    'grok-beta': {
        'prompt_fee': 5*7/1000,
        'completion_fee': 15*7/1000,
    },
    'qwen-max': {
        'prompt_fee': 0.02,
        'completion_fee': 0.06,
    },
    'qwen-plus': {
        'prompt_fee': 0.0008,
        'completion_fee': 0.002,
    },
    'qwen-turbo': {
        'prompt_fee': 0.0003,
        'completion_fee': 0.0006,
    },
    'qwen-long': {
        'prompt_fee': 0.0005,
        'completion_fee': 0.002,
    },
    'qwen2.5-72b-instruct': {
        'prompt_fee': 0.004,
        'completion_fee': 0.012,
    },
    'qwen2.5-32b-instruct': {
        'prompt_fee': 0.0035,
        'completion_fee': 0.007,
    },
    'qwen2.5-14b-instruct': {
        'prompt_fee': 0.002,
        'completion_fee': 0.006,
    },
    'qwen2.5-7b-instruct': {
        'prompt_fee': 0.001,
        'completion_fee': 0.002,
    },

}


class Request:
    def __init__(self, server="knowbox", model='', port=9091):
        self.model = model
        base = ''
        match server:
            case 'jieyue' | 'step':
                key = os.environ.get('JIEYUE_API_KEY')
                base = 'https://api.step.ai/v1'
                base = "https://api.stepfun.com/v1"
                if self.model == "":
                    self.model = "step-1-flash"
            case 'ali' | 'qwen':
                key = os.environ.get('QWEN_API_KEY')
                base = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
                if self.model == '':
                    self.model = 'qwen-turbo'
            case 'grok':
                key = os.environ.get('GROK_API_KEY')
                base = 'https://api.x.ai/v1'
                if self.model == '':
                    self.model="grok-beta"
            case 'tmove':
                key = os.environ.get('TMOVE_KEY')
                if self.model == '':
                    self.model = "gpt-3.5-turbo"
            case 'tmove_r':
                key = os.environ.get('TMOVE_R_KEY')
                if self.model == '':
                    self.model = "gpt-3.5-turbo"
            case 'knowbox':
                key = os.environ.get('KNOWBOX_KEY')
                base = "https://maxsj166proxy.xyz/v1"
                if self.model == '':
                    self.model = "gpt-3.5-turbo"
            case 'moonshot' | 'kimi':
                key = os.environ.get('KIMI_API_KEY')
                base = "https://api.moonshot.cn/v1"
                if self.model == '':
                    self.model = "moonshot-v1-8k"
            case 'yi':
                key = os.environ.get('LINGYI_API_KEY')
                base = "https://api.lingyiwanwu.com/v1"
                if self.model == '':
                    self.model = "yi-medium"
            case 'deepseek':
                key = os.environ.get('DEEPSEEK_API_KEY')
                base = 'https://api.deepseek.com'
                if self.model == '':
                    self.model = "deepseek-chat"
            case 'private_deepseek':
                key = os.environ.get('HUANG_KEY')
                base = 'http://121.225.97.127:18981/api/v1'
                self.model = "DeepSeek-R1-Q4_K_M"
            case 'tunnel':
                base = 'https://api.nuwaapi.com/v1'
                key = os.environ.get('TUNNEL_KEY')
            case _:
                key = os.environ.get('OPENAI_API_KEY')

        if base:
            self.client = OpenAI(
            api_key=key,
            base_url=base
        )
        else:
            self.client = OpenAI(
            api_key=key,
        )

        self.response_list = []

    def define_tools(self, tools: list):
        tool_list = statics.define_tools(self, tools)
        return {'tools': tool_list}

    def call(self, messages, **kwargs):
        if 'vision' or 'gpt-4o' not in self.model:
            messages = statics.remove_no_str_message(messages)
        else:
            statics.check_image_urls(messages[-1])

        params = {
            'model': self.model,
            'messages': messages,
        }
        if 'temp' in kwargs.keys():
            params["temperature"] = kwargs['temp']
        elif 'temperature' in kwargs.keys():
            params['temperature'] = kwargs['temperature']
        
        if 'max_tokens' in kwargs.keys():
            params["max_tokens"] = kwargs['max_tokens']

        if 'tools' in kwargs.keys():
            params['tools'] =kwargs['tools']

        if 'tool_choice' in kwargs.keys():
            params['tool_choice'] = kwargs['tool_choice']

        if 'logprobs' in kwargs.keys():
            params['logprobs'] = kwargs['logprobs']
            if 'top_logprobs' in kwargs.keys():
                top_logprobs = kwargs['logprobs']
            else:
                top_logprobs = None
            if params['logprobs'] and isinstance(top_logprobs, int) and 0 <= top_logprobs <= 20:
                params['top_logprobs'] = top_logprobs

        if 'response_format' in kwargs.keys():
            if kwargs['response_format']== 'json':
                params['response_format'] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**params)
        self.response_list.append(response)
        return self

    def read_response(self, position=-1):
        try:
            completed_message = self.response_list[position].choices[0].message
        except Exception as e:
            return e

        if not completed_message.content:
            call_message = []
            for result in completed_message.tool_calls:
                call_message.append(result.function.__dict__)
            record_message = self.response_list[position].choices[0].message.model_dump()
            record_message.pop('function_call')
        else:
            record_message = completed_message.content
            call_message = None

        return statics.read_response(completed_message.role,
                                     record_message,
                                     call_message)

    def dump_tool_call_msg(self, tool_msg='', position=-1):
        return {"role": "tool",
                "content": tool_msg,
                "tool_call_id": self.response_list[position].choices[0].message.tool_calls[0].id
                }

    def count_usage(self):
        prompt_tokens = 0
        completion_tokens = 0
        model_set = ''
        if self.model in \
                ['gpt-4-0125-preview', 'gpt-4-turbo-preview', 'gpt-4-1106-preview', 'gpt-4-vision-preview',
                 'gpt-4-vision-1106-preview',]:
            model_set = 'gpt-4-turbo'
        elif 'gpt-4-32k' in self.model:
            model_set = 'gpt-4-32k'
        # elif 'gpt-4' in self.model:
        #     model_set = 'gpt-4'
        elif 'gpt-3.5-turbo-instruct' in self.model:
            model_set = 'gpt-3.5-turbo-instruct'
        elif 'gpt-3.5-turbo' in self.model:
            model_set = 'gpt-3.5-turbo'
        else:
            model_set = self.model

        for response in self.response_list:
            try:
                usage = response.usage
                prompt_tokens += usage.prompt_tokens
                completion_tokens += usage.completion_tokens
            except Exception:
                continue

        statics.tokens2fee(model_set, MODEL_FEE, prompt_tokens, completion_tokens)


if __name__ == '__main__':
    a = Request(server='private_deepseek',  )
    test_messages = [
        {'role': 'user',
         'content': '扮演一只猫'}
    ]
    print(a.call(test_messages).read_response()['show_msg'])
    a.count_usage()

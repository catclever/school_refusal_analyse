# 在多个api服务中通用的静态方法
import requests
import random
import pprint


def prompt_assembler(prompt_template, *args):
    ret = prompt_template.format(*args)
    return ret
    
def kw_prompt_assembler(prompt_template, **kwargs):
    ret = prompt_template.format(**kwargs)
    return ret

def define_tools(request, tools: list):
    tool_list = []
    for tool in tools:
        params = {
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': tool.description,
                'parameters': {
                    'type': 'object',
                    'properties': {}
                },
                'required': []
            }
        }
        if 'gpt-4' in request.model:
            # gpt=4没有required这个参数，gpt-3.5 即使传了required也可能不返回……
            params['function'].pop('required')

        for param in tool.params:
            if 'name' in param.keys() and param['name']:
                property_info = {}
                if 'type' in param.keys() and param['type']:
                    property_info['type'] = param['type']
                else:
                    continue
                if 'enum' in param.keys() and isinstance(param['enum'], list):
                    property_info['enum'] = param['enum']
                elif 'description' in param.keys():
                    property_info['description'] = param['description']
                params['function']['parameters']['properties'][f"{param['name']}"] = property_info

                if 'required' in param.keys() and param['required'] and 'required' in params['function'].keys():
                    params['function']['required'].append(param['name'])

        tool_list.append(params)
    return tool_list


def read_response(role, raw_msg, thinkings=None, tool_calls=None):
    try:
        result = {
            'show_msg': None,
            'record_msg': None,
            'thinkings': thinkings,
            'call_msg': None
        }
        if not tool_calls:
            result['record_msg'] = {
                'role': role,
                'content': raw_msg
            }
            result['show_msg'] = raw_msg
        else:
            # 处理tool_calls的返回
            result['show_msg'] = tool_calls
            result['record_msg'] = raw_msg
            result['call_msg'] = tool_calls
        # print(result)
        return result
    except Exception as e:
        return e


def str_context(context, user_name='', assistant_name=''):
    str_context = ''
    for message in context:
        str_context += message['role']+'：'+message['content']+'\n'
    if user_name:
        str_context = str_context.replace('user', user_name)
    if assistant_name:
        str_context = str_context.replace('assistant', assistant_name)
    return str_context
        

def tokens2fee(model, fee_list, prompt_tokens, completion_tokens):
    print(f'prompt_tokens: {prompt_tokens}')
    print(f'completion_tokens: {completion_tokens}')
    if model in fee_list.keys():
        total_fee = (fee_list[model]['prompt_fee'] * prompt_tokens + fee_list[model][
            'completion_fee'] * completion_tokens) / 1000
        print(f'total_fee: {total_fee}')


def merge_adjacent_messages_with_same_role(messages):
    req_messages = messages.copy()
    for i in range(len(req_messages) - 1, 0, -1):
        if req_messages[i]['role'] == req_messages[i - 1]['role']:
            # 合并content字段
            req_messages[i - 1]['content'] += '\n' + req_messages[i]['content']
            # 现在可以删除当前的消息，因为它已经被合并到前一个消息中了
            del req_messages[i]
    return req_messages


def remove_no_str_message(messages):
    related_messages = messages.copy()
    for i in range(len(related_messages) - 1, 0, -1):
        if related_messages[i]['content'] and not isinstance(related_messages[i]['content'], str) :
            del related_messages[i]
    return related_messages


def is_image(url):
    try:
        # 发送HTTP GET请求
        response = requests.get(url)

        # 检查响应状态码 - 200表示请求成功
        if response.status_code == 200:
            # 检查内容类型
            content_type = response.headers['Content-Type'].lower()
            if 'image' in content_type:
                return True
            else:
                return False
        else:
            return False

    except requests.RequestException as e:
        print(f"请求出错：{e}")
        return False
    
def is_video(url):
    # 定义常见视频扩展名
    video_extensions = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm")
    # 检查 URL 是否以这些扩展名结尾
    return url.lower().endswith(video_extensions)


def check_image_urls(message):
    # 容错，避免用户发送图片链接无效时，模型会重新回复上一条消息(gpt)或按照空文件进行回复(glm)
    if isinstance(message['content'], list):
        image_list = [d for d in message['content'] if d.get('type') == 'image_url']
        for image in image_list:
            url = image['image_url']['url']
            if not url.startswith('http'):
                continue
            if not is_image(url):
                raise ValueError("Invalid image url")
        video_list = [d for d in message['content'] if d.get('type') == 'video_url']
        for video in video_list:
            url = video['video_url']['url']
            if not url.startswith('http'):
                continue
            if not is_video(url):
                raise ValueError("Invalid image url")
            
            
def take_the_microphone(dm, services_list=None):
    if services_list:
        return services_list
    else:
        return [random.choice(dm.talk.services).name]


def remove_system_prompt(messages):
    try:
        if messages[0]['role'] == 'system':
            messages.pop(0)
    except IndexError or KeyError:
        print('没有找到system消息！')
    return messages


def probe(item=None):
    # 调试用的方法，正式使用时注释掉所有probe
    if item:
        pprint.pprint(item)
    else:
        print('here')

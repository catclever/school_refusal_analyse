import paradigm
import statics
import json
from types import SimpleNamespace
import copy
import threading
import queue
from functools import partial


class QuietThread(threading.Thread):
    def run(self):
        try:
            # 原始的run方法内容，这里调用Thread本身的实现
            if self._target:
                self._target(*self._args, **self._kwargs)
        except Exception:
            # 这里可以选择记录日志或者什么都不做来忽略异常
            pass  # 如果你想要完全无视错误，只需要pass即可


class Service:
    def __init__(self, name, server, model='',timeout=0):
        self.name = name
        self.params = {}
        self.on_call_list = []
            
        self.request = paradigm.Request(server, model) if model else paradigm.Request(server)

    def set_params(self, params):
        self.params.update(params)

    def add_tools(self, tools: list):
        for tool in tools:
            self.on_call_list.append(tool)
        self.update_tool_params()

    def remove_tools(self, name_to_remove: list):
        for i in range(len(self.on_call_list) - 1, 0, -1):
            if self.on_call_list[i].name in name_to_remove:
                del self.on_call_list[i]
        self.update_tool_params()

    def update_tool_params(self):
        if hasattr(self.request, 'define_tools'):
            params = self.request.define_tools(self.on_call_list)
            self.set_params(params)

    def respond(self, messages, silent=False, show_name=None, **kwargs):
        if not messages:
            return
        # print(messages)
        if not show_name:
            show_name = self.name
        params = copy.deepcopy(self.params)
        for key, value in kwargs.items():
            params[key] = value
        self.request.call(messages, **params)
        result = self.request.read_response()
        if isinstance(result, Exception):
            print(show_name + ': An error occured.\n'+f"{result}")
        else:
            if not silent:
                print(show_name + ':' + f"{result['show_msg']}")
                print('\n')
            return result
        
    def queue_respond(self, messages, result_queue, order=0, reply_type='public', task=''):
        if task =='deal_recall':
            result = self.answer_with_func_msg(messages)
        else:
            result = self.respond(messages)

        if result['call_msg']:
            # 处理tool_call的消息
            reply_type = 'private'
            result_queue.put(SimpleNamespace(service=self.name,
                                             result_type='call',
                                             reply=result['call_msg'],
                                             order=order
                                             ))
            result_type = 'call_record'
        else:
            result_type = 'record'

        result_queue.put(SimpleNamespace(service=self.name,
                                         result_type=result_type,
                                         reply=result['record_msg'],
                                         order=order,
                                         task = task,
                                         reply_type=reply_type))


    def receive_recall(self, call_msg):
        result = []
        for msg in call_msg:
            hitted = list(filter(lambda func: func.name == msg['name'], self.on_call_list))
            args_dict = json.loads(msg['arguments'])
            args_dict['service'] = self
            result.append(hitted[0].recall(args_dict))
        return result

    def answer_with_func_msg(self, messages):
        defined_tool_choice = self.params['tool_choice']
        if not isinstance(self.params['tool_choice'], str):
            self.params['tool_choice'] = "none"
        result = self.respond(messages)
        self.params['tool_choice'] = defined_tool_choice
        return result


class Function:
    # 理论上应该是Tools类，但目前除了function外没有其他tool可用……
    def __init__(self, name, params: list, description='', recall=None):
        self.name = name
        self.params = params
        self.description = description
        self.recall_func = recall

    def recall(self, args_dict):
        if callable(self.recall_func):
            func_msg = self.recall_func(args_dict)
        else:
            args_dict.pop('service', None)
            func_msg = args_dict
        return func_msg


class Talker(Service):
    def __init__(self, service_info, system_prompt='', ):
        name = service_info.get('name')
        server = service_info.get('server')
        model = service_info.get('model')
        timeout = service_info.get('timeout')
        super().__init__(name, server, model, timeout)
        self.context = []
        if system_prompt:
            self.context.append({'role': 'system', 'content': system_prompt})

    def restart(self):
        new_context = []
        if self.context and self.context[0]['role'] == 'system':
            new_context.append(self.context[0])
        self.context = new_context

    def update_system_prompt(self, system_prompt):
        if not self.context:
            self.context.insert(0,{'role': 'system', 'content': system_prompt})
        elif self.context[0]['role'] == 'system':
            self.context[0]['content'] = system_prompt
        else:
            self.context.insert(0,{'role': 'system', 'content': system_prompt})

    def add(self, message):
        self.context.append(message)

    def send(self, content='', message=None, silent=False):
        if message:
            self.context.append(message)
        elif content:
            self.context.append({'role': 'user', 'content': content})
        # print(self.params)
        # print(self.context)

        result = self.respond(self.context, silent=silent)
        
        if result['record_msg']:
            self.context.append(result['record_msg'])
        if result['call_msg']:
            func_msg = self.receive_recall(result['call_msg'])
            n = 0
            if func_msg:
                for msg in func_msg:
                    try:
                        func_str = json.dumps(msg)
                        self.context.append(self.request.dump_tool_call_msg(tool_msg=func_str))
                        # TODO：如果有多个工具返回，目前是插入多条tool消息，不确定是否能正常处理，需要验证
                        n += 1
                    except Exception:
                        continue
            if n > 0:
                result = self.answer_with_func_msg(self.context)
                self.context.append(result['record_msg'])

    def read_context(self):
        result = ""
        for context in self.context:
            if context["role"] != "system":
                result += f"{context['role']}：{context['content']}\n"
        return result

class Agent(Service):
    def __init__(self, name, server,  model='', controller=None, **kwargs):
        super().__init__(name, server, model)
        self.properties = kwargs
        self.controller = controller
        self.working_threads = []
        self.stop_event = threading.Event()
        self.states = {}

    def respond(self, messages, silent=False, show_name=None, **kwargs):
        task_system_prompt = 0
        if 'guidance' in self.states.keys():
            if self.states['guidance']:
                messages = statics.remove_system_prompt(messages)
                messages.insert(0,{'role': 'system', 'content':self.states['guidance']})
                task_system_prompt = 1
                if 'keep_guidance' in self.states.keys():
                    if not self.states['guiance']:
                        self.states['guidance'] = ''
                else:
                     self.states['guidance'] = ''

        if task_system_prompt == 0:
            if 'system_prompt' in self.properties.keys():
                messages = statics.remove_system_prompt(messages)
                messages.insert(0,{'role': 'system', 'content':self.properties['system_prompt']})

        return super().respond(messages, silent, show_name, **kwargs)
    
    def prepare(self):
        if (processes:= self.properties.get('pres')):
            for process in processes:
                if callable(process):
                    process(self)

    def think(self, messages, result_key:str, silent=True, show_name=None, overwrite=False, **kwargs):
        result = self.respond(messages, silent, show_name, **kwargs)
        if overwrite:
            self.states[result_key] = []
        if result_key in self.states.keys():
            if isinstance(self.states[result_key], list):
                self.states[result_key].insert(0,result['show_msg'])
            else:
                self.states[result_key] = result['show_msg']
        else:
            self.states[result_key] = [result['show_msg']]
    
    def control(self):
        i = self.states.get('control_round', 0)
      
        while True:
            if self.stop_event.is_set():
                if 'control_round' in self.states.keys():
                    self.states['control_round'] = i
                break
            else:
                break_flag = self.controller(agent=self, control_round=i)
                # controoler用到的参数通过states及properties传递s
                # congtroller中直接向states中写入数据
                i += 1
                if break_flag:
                    break

    def activate(self, quiet=True):
        # 开始agent的自动流程
        self.stop_event.clear()
        if self.controller:
            # thread = threading.Thread(target=self.control)
            thread = QuietThread(target=self.control) if quiet else threading.Thread(target=self.control)
            self.working_threads.append(thread)
            thread.start()
            return thread

    def wait(self):
        for thread in self.working_threads:
            thread.join()
    
    def hang_up(self):
        # 停止agent的自动流程
        self.stop_event.set()  # 请求停止线程、
        for thread in self.working_threads:
            thread.join()
            
class Quest:
    def __init__(self, process, **kwargs):
        self.process = process
        self.params = kwargs

    def __call__(self, args_dict=None):
        if callable(self.process):
            return self.process(**self.params)
        # args_dict用于接受上一个after process的返回参数
        # 返回参数必须是一个dict
        # 需要接受args_dict或者返回参数的Quest必须重写__call__方法，用于处理args，或者拼装返回的字典

class Task:
    # Task类是multi_talk.Talk类的平替，没有上下文管理功能
    def __init__(self, services=[], ):
        self.services = []
       
        for service in services:
            self.create_service(service)
        # self.cuhaorrent_task_service = []
        self.threads = []
        self.result_queue = queue.Queue()

        self.process = [] # 一系列pipeline处理逻辑，每个task只能有一个pipeline

    def create_service(self, service_info):
        service_info = service_info.copy()
        try:
            name = service_info['name']
            server = service_info['server']
            service_info.pop('name')
            service_info.pop('server')
        except KeyError:
            print('缺少必要参数！')
        if 'agent' in service_info.keys():
            agent = service_info['agent']
            service_info.pop('agent')
        else:
            agent = False
        if 'model' in service_info.keys():
            model = service_info['model']
            service_info.pop('model')
        else:
            model = ''
        if agent:
            service = Agent(name, server, model, **service_info)
        else:
            service = Service(name, server, model)
        self.services.append(service)

    def get_service(self, service_name):
        a = list(filter(lambda service: service.name == service_name, self.services))
        return a[0]

    def set_all(self, **params):
        for service in self.services:
            service.set_params(params)

    def restart(self):
        new_records = []
        if self.records and self.records[0].msg['role'] == 'system':
            new_records.append(self.records[0])
        self.records = new_records

    def get_receivers(self, receivers=None):
        task_services = []
        if not receivers:
            task_services = self.services
        elif isinstance(receivers, list):
            for service in self.services:
                if service.name in receivers or service in receivers:
                    task_services.append(service)
        return task_services

    def assign(self, messages, receivers=None, task=''):
        task_services = self.get_receivers(receivers)
        for service in task_services: 
            self._task_thread(service, messages, task)
        return self.receive()
    
    def abs_assign(self, assign_list:list):
        for assign_message in assign_list:
            try:
                receiver = assign_message['receiver']
                if isinstance(receiver, str):
                    receiver = self.get_service(receiver)
                messages = assign_message['messages']
            except KeyError:
                continue
            
            if 'dealing_functions' in assign_message.keys():
                self._multi_task_thread(receiver, messages, assign_message['dealing_functions'])
            else:
                if 'task' in assign_message.keys():
                    task = assign_message['task']
                else:
                    task = ''
       
                self._task_thread(receiver, messages, task)
        
        return self.receive()

    def _task_thread(self, service:Service, messages, task='',):
        self.hang_up(service)
        thread = QuietThread(target=service.queue_respond, 
                                   args=(messages, self.result_queue,),
                                   kwargs={'task': task})
        thread.start()
        self.threads.append(thread)

    def _multi_task_thread(self, service:Service, start_messages:list, dealing_functions:list):
        self.hang_up(service)
        functions = [(partial(service.respond, start_messages, sllent=True))]
        for function in dealing_functions:
            functions.append(function)

        def execute(functions):
            params = None
            for function in functions:
                if params:
                    params = function(params)
                else:
                    function()
        
        thread = QuietThread(target=execute, args=(functions))
        thread.start()
        self.threads.append(thread)

    @staticmethod
    def hang_up(service:Service):
        if isinstance(service, Agent):
            service.hang_up()

    def receive(self):
        """处理返回消息的函数"""
        # 等待所有线程完成
        for thread in self.threads:
            thread.join()

        self.threads = []
        report = []
        recursion = False

        while not self.result_queue.empty():
            result = self.result_queue.get()
            if result.result_type == 'record':
                report.append({
                    'service': result.service,
                    'reply': result.reply,
                })
             
            elif result.result_type == 'call':
                service = self.get_service(result.service)
                func_msg = service.receive_recall(result.reply)
                if func_msg:
                    for msg in func_msg:
                        try:
                            self.assign(receivers=[result.service], task='deal_recall')
                            recursion = True
                        except Exception:
                            continue
        if recursion:
            return self.receive()
        elif report:
            return report
        
    def add_process(self, process:Quest):
        self.process.append(process)

    def execute_process(self):
        args_dict = {}
        for process in self.process:
            if isinstance(process, Quest):
                args_dict = process(args_dict)
                if not isinstance(args_dict, dict):
                    args_dict = {}
                if 'break_flag' in args_dict.keys():
                    if args_dict['break_flag']:
                        break


if __name__ == "__main__":
    a = Task([{"name":'我是个AI', "server":"zijie", "model" :"ep-20250114164317-rnvmj",}])
    test_messages = [
        {'role': 'user',
         'content': '扮演一只猫'}
    ]
    a.assign(test_messages)

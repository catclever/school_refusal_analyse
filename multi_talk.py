import copy
import re
import json
from functools import partial
from types import SimpleNamespace
import llm_api as api
import infra


class TalkRecord:
    def __init__(self, order, msg, sender='user', display_to=['all']):
        self.order = order
        self.sender = sender
        # sender代表消息的来源，可能是user、system、tools或{service.name}
        self.display_to = display_to
        self.msg = msg

    def check(self, service):
        # 判断某个service是否可以看到这条消息
        if 'all' in self.display_to or service.name in self.display_to:
            return self

class Talk(infra.Task):
    def __init__(self, talk_services=[], system_prompt='', ):
        super().__init__(talk_services)
        
        self.records = []
        if system_prompt:
            self.records.append(TalkRecord(0, {'role': 'system', 'content': system_prompt}, 'system'))

        self.current_order = 1
        self.main_task = ''

    def update_system_prompt(self, system_prompt):
        if not self.records:
            self.records.insert(0, TalkRecord(0, {'role': 'system', 'content': system_prompt}, 'system'))
        elif self.records[0].msg['role'] == 'system':
            self.records[0].msg['content'] = system_prompt
        else:
            self.records.insert(0, TalkRecord(0, {'role': 'system', 'content': system_prompt}, 'system'))

    def get_related_context(self, service, use_tools=False):
        """在record.check的基础上，修改其他服务的role为对应的{service.name}并排序"""
        related = []
        if isinstance(service, str):
          service = self.get_service(service)
        for record in self.records:
            if not use_tools and record.sender == 'tools':
                # 除非用到工具，否则不处理工具结果
                continue
            if record.check(service):
                related.append(record)
        sorted_records = sorted(related, key=lambda x: x.order)

        context = []
        for record in sorted_records:
            msg = copy.deepcopy(record.msg)
            if record.sender not in ['user', 'system', 'tools'] and record.sender != service.name:
                msg['role'] = record.sender
            context.append(msg)
        return context

    def map_task_messages(self, service, task='', instruct='', instruct_type='guidance'):
        """根据task类型map发送给service的消息"""
        if task in ['deal_recall']:
            raw_messages = self.get_related_context(service, True)
        else:
            raw_messages = self.get_related_context(service)

        task_messages = []
        match task:
            case 'group_discussion':
                for msg in raw_messages:
                    if msg['role'] == 'system':
                        msg['content'] = f"{msg['content']}\n你在这次讨论中扮演{service.name}\n其他角色由用户扮演，你不需要替其他角色说话"
                    # elif msg['role'] == 'assistant':
                    #     msg['content'] = f"（我是{service.name}，我说）: {msg['content']}"
                    elif msg['role'] not in ['system', 'assistant', 'user']:
                        msg['content'] = f"{msg['role']}说: {msg['content']}"
                        msg['role'] = 'user'
                    task_messages.append(msg)
                if not any(msg.get('role') == 'system' for msg in task_messages):
                    identity_msg = {
                        'role': 'system',
                        'content': f"你是{service.name}"
                    }
                    task_messages.insert(0, identity_msg)
            case '1on1' | 'deal_recall':
                # 只保留对应模型的消息
                for msg in raw_messages:
                    if msg['role'] in ['system', 'assistant', 'user', 'tools']:
                        task_messages.append(msg)
            case _:
                # 将所有其他模型返回的消息视为为用户消息
                for msg in raw_messages:
                    if msg['role'] not in ['system', 'assistant', 'user']:
                        msg['content'] = f"{msg['role']}: {msg['content']}"
                        msg['role'] = 'user'
                    task_messages.append(msg)

        # 处理instruct
        if instruct:
            if instruct_type == 'guidance':
                if isinstance(service, infra.Agent):
                    service.states['guidance']  = instruct
                else:
                    task_messages = api.statics.remove_system_prompt(task_messages)
                    instruct_message = {
                        'role': 'system',
                        'content': instruct,
                    }
                    task_messages.insert(0,instruct_message)
            else:
                task_messages.append({
                    'role': 'user',
                    'content': instruct,
                })

        # 定义返回类型
        if task in ['1on1']:
            reply_type = 'private'
        else:
            reply_type = 'public'

        return task_messages, reply_type

    def send(self, content='', role_message=None):
        # 可以发送空消息，但不会记录
        if content:
            self.records.append(TalkRecord(self.current_order, {'role': 'user', 'content': content}))
        elif role_message:
            if role_message['role'] == 'user':
                self.records.append(TalkRecord(self.current_order, role_message))
            elif role_message['role'] == 'system':
                self.records.append(TalkRecord(self.current_order, role_message, 'system'))
            else:
                service_names = []
                for service in self.services:
                    service_names.append(service.name)
                if role_message['role'] in service_names:
                    sender = role_message['role']
                    role_message['role'] = 'assistant'
                    if 'display_to' in role_message.keys():
                        display_to = role_message['display_to']
                        self.records.append(TalkRecord(self.current_order, role_message, sender, display_to))
                    else:
                        self.records.append(TalkRecord(self.current_order, role_message, sender))
        else:
            self.current_order -= 1

        self.current_order += 1

        
        # for service in self.services:
        #     service.request.call(self.request_messages_format())
        #     result = service.request.read_response()
        #     self.messages.append({"role": "assistant", "content": result})
        #     print(service.name+': '+result)
        #     print('\n')

    def record(self, result, tool=False):
        if tool:
            self.records.append(TalkRecord(result.order, result.reply, 'tools', [result.service]))
        elif result.reply_type == 'public':
            self.records.append(TalkRecord(result.order, result.reply, result.service))
        elif result.reply_type == 'private':
            self.records.append(TalkRecord(result.order, result.reply, result.service, [result.service]))
        self.current_order += 1

    def assign(self, receivers=None, task='', instruct='', reply_type='', messages=None):
        # assign 和 send 分开，可以自动执行一些预定义的task
        # instruct是不进入聊天记录的指令
        task_services = self.get_receivers(receivers)

        if not task:
            task = self.main_task
        
        for service in task_services:
            self._task_thread(service, task, instruct, reply_type, messages)
        
        return self.receive()
    
    def prim_assign(self, receiver:infra.Service, reply_type='report', messages=None, dealing_functions=None):
        """
        基本的assign方法, 一次只支持一个service，不自动阻塞
        主要用于支线任务的处理，dealing_function是后处理逻辑       
        """

        # print(task_services)
        if dealing_functions and messages:
            self._multi_task_thread(receiver, messages, dealing_functions)
        else:
            self._task_thread(receiver, reply_type=reply_type, messages=messages)
    
    def abs_assign(self, assign_list:list):
        """
        用于给每个receiver独立传入不同的内容
        也支持一个reciver同时处理多个任务，或给一个receiver传入连续执行的多个任务
        """
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
                if 'instruct' in assign_message.keys():
                    instruct = assign_message['instruct']
                else:
                    instruct = ''

                if 'task' in assign_message.keys():
                    task = assign_message['task']
                else:
                    task = ''
                
                if 'reply_type' in assign_message.keys():
                    reply_type = assign_message['reply_type']
                else:
                    reply_type = 'public'
                self._task_thread(receiver, task, instruct, reply_type, messages)
        
        return self.receive()

    def _task_thread(self, service:infra.Service, task='', instruct='', reply_type='', messages=None):
        if messages and reply_type:
            task_messages = messages
        else:
            mapped_messages, mapped_reply_type = self.map_task_messages(service, task, instruct)
            if not messages:
                task_messages = mapped_messages
            if not reply_type:
                reply_type = mapped_reply_type
        thread = infra.QuietThread(target=service.queue_respond, 
                                   args=(task_messages, self.result_queue, self.current_order, reply_type, task))
        thread.start()
        self.threads.append(thread)
             
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
                if result.reply_type == 'report':
                    report.append({
                        'service': result.service,
                        'reply': result.reply,
                    })
                else:
                    self.record(result)
            elif result.result_type == 'call_record':
                self.record(result, tool=True)
            elif result.result_type == 'call':
                service = self.get_service(result.service)
                func_msg = service.receive_recall(result.reply)
                if func_msg:
                    for msg in func_msg:
                        try:
                            recall_msg = service.request.dump_tool_call_msg(tool_msg=json.dumps(msg))
                            self.records.append(TalkRecord(result.order, recall_msg, 'tools', [result.service]))
                            self.assign(receivers=[result.service], task='deal_recall')
                            recursion = True
                        except Exception:
                            continue
            
        if recursion:
            return self.receive()
        elif report:
            return report


class DM:
    """
    DM类是控制流程
    可以是预定义的，也可以结合实际的场景定义
    """
    def __init__(self, talk:Talk, talk_type='independent', default_receivers=None, default_msg_type='private', **kwargs):
        self.talk = talk
    
        self.boost = False # 是否可以自动推进
        self.boost_threads = [] 

        # default parameters
        self.default_receivers = default_receivers
        self.default_msg_type = default_msg_type

        # process parameters
        self.current_receivers = default_receivers
        self.receivers_lock = False
        self.receivers_assigner = None # 用来分配下一个说话者的逻辑
        self.current_msg_type = default_msg_type
        self.msg_type_lock = True
       
        # dialog parameters 
        self.host = None
        self.host_paras = None
        self.set(talk_type, **kwargs)

       
    def _reset(self):
        """用户输入后调用的函数"""
        # TODO： 用户输入后停止自动执行的线程

        if self.receivers_lock:
            pass
        elif self.receivers_assigner:
            assigner = self.receivers_assigner
            self.receivers_assigner = None
        else:
            assigner = self.default_receivers

        if callable(assigner):
            self.current_receivers = assigner()
        elif isinstance(assigner, list):
            self.current_receivers = assigner
            
        if not self.msg_type_lock:
            self.current_msg_type = self.default_msg_type

    def _host_set(self, **kwargs):
        host_name = kwargs.get('host')
        if host_name:
            self.host = self.talk.get_service(host_name)
            self.host_paras = SimpleNamespace()
            if not kwargs.get('host_in_talk'):
                self.talk.services.remove(self.host)
            self.host_paras.quest_template = partial(HostQuest, self)
            self.host_paras.result = {}
            self.host_paras.quests = []
            self.host_paras.topic_manage = kwargs.get('topic_manage')
    
    def set(self, talk_type='', **kwargs):
        if talk_type:
            self.talk_type = talk_type
            match self.talk_type:
                case 'independent':
                    self.talk.main_task = '1on1'
                    self.current_msg_type = 'private'
                    self.msg_type_lock = True
                case 'on_a_table':
                    self.talk.main_task = 'group_discussion'
                    self._host_set(**kwargs)
                    self.default_receivers = lambda: api.statics.take_the_microphone(self)
                    self.current_msg_type = 'public'
                    self.msg_type_lock = True
                case 'press_conference':
                    # 发布会模式，一个AI主要输出，其他AI可以看到这些消息
                    # 其他AI可以加工这些消息
                    self.talk.main_task = '1on1'
                    self.default_receivers =  kwargs.get('protagonist')
                    self.current_receivers = self.default_receivers
                    self.receivers_lock = True
                    self.current_msg_type = 'public'
                    self.msg_type_lock = True

        if 'boost' in kwargs.keys():
            self.boost = kwargs['boost']
       
        if 'post_processes' in kwargs.keys():
            for post_process in kwargs['post_processes']:
                self.talk.add_process(post_process)
        elif 'post_process'in kwargs.keys():
            self.talk.add_process(kwargs['exclusive_post_process'])
        elif 'exclusive_post_process'in kwargs.keys():
            self.talk.process = []
            self.talk.add_process(kwargs['exclusive_post_process'])
        elif 'exclusive_post_processes' in kwargs.keys():
            self.post_process = []
            for post_process in kwargs['exclusive_post_processes']:
                 self.talk.add_process(post_process)

    def commander(self, user_input=''):
        if not user_input:
            user_input = input('Please input your message: ')

        if not user_input:
            user_input = 'go_on'
        if user_input != 'go_on':
            self._reset()

        comand_line_pattern = r'^(\S+\s*)(\s+-\S+)*(\s+\S+)*\s*$'
        matches = re.match(comand_line_pattern, user_input)

        command = matches.group(1).strip()
        attr = []
        para = []
        if matches.group(2):
            attr = re.findall(r'-(\S+)', matches.group(2))
        if matches.group(3):
            para = re.findall(r'\b([^\s-]\S+)', matches.group(3))

        match command:
            case 'count_usage':
                for service in self.talk.services:
                    print(service.name)
                    service.request.count_usage()
                    print('\n')
            # TODO：save_history, 保存对话记录
            case 'send_to':
                if not para:
                    receivers = input('Please enter a list of services separated by space: ').split()
                else:
                    receivers = para

                try:
                    if attr and '-keep' in attr:
                        self.receivers_lock = True
                        if receivers == ['all']:
                            self.current_receivers = [item.name for item in self.talk.services]
                        else:
                            self.current_receivers = receivers
                    else:
                        self.receivers_lock = False
                        if receivers == ['all']:
                            self.receivers_assigner = [item.name for item in self.talk.services]
                        else:
                            self.receivers_assigner = receivers
                except Exception as e:
                    print(e)
            case 'image':
                if not para:
                    image_url = input('Please enter pic url: ')
                else:
                    image_url = para[0]
                if api.statics.is_image(image_url):
                    text = input('What do you want to talk about with the pic: ')
                    msg = [
                        {'type': 'text', 'text': text},
                        {'type': 'image_url', 'image_url': {'url': image_url}
                         },
                    ]
                    self._quest(msg)
                else:
                    print('image is invalid')
            case 'go_on':
                # 主要用于在命令行控制；和_auto()也可以兼容
                self._quest()
            case _:
                self._quest(user_input)

    def _quest(self, msg='', receivers=None):
        if not receivers:
            receivers = self.current_receivers
        # main quest => somebody talk something
        
        if msg:
            self.talk.send(msg)
            self.talk.assign(receivers=receivers, reply_type=self.current_msg_type)
        else:
            match self.talk_type:
                case 'on_a_table':
                    self.talk.assign(receivers=receivers)
    
        self.talk.execute_process()

        if (topic_manager := self.host_paras.__dict__.get('topic_manage')):
            if callable(topic_manager):
                topic_manager(self)

        if self.boost:
            self._auto()
    
    def _auto(self):
        # TODO: 开一个延期线程，根据self.boost判断是否再次执行_quest()
        # TODO: 开一个延期线程，执行host_paras.quests
        pass

# 以下是一些默认的Quest
class HostQuest(infra.Quest):
    def __init__(self, dm:DM, result:str, user_prompt:infra.Function, active_rounds=0, system_prompt='',):
        self.dm = dm
        self.result = result
        
        self.active_rounds = active_rounds
        self.waiting_rounds = 0
        
        self.system_pormpt = system_prompt
        self.user_prompt = user_prompt
        
        super().__init__ (self.porcess)
        
    def process(self):
        if self.waiting_rounds < self.active_rounds:
            self.waiting_rounds += 1
        else:
            self.waiting_rounds = 0
            user_prompt = self.user_prompt(self.dm.talk)
            messages = [
                {'role': 'user', 'content': user_prompt},
            ]
            if self.system_pormpt:
                messages.insert(0, {'role': 'system', 'content': self.system_pormpt})
            self.dm.host_paras.result[self.result] = self.dm.host.respond(messages, silent=True)['show_msg']
    

if __name__ == '__main__':
    services_list = [
        # {'name': 'gpt4v', 'server': 'knowbox', 'model': 'gpt-4-vision-preview'},
        # {'name': 'gpt4', 'server': 'knowbox', 'model': 'gpt-4'},
        # {'name': 'gpt', 'server': 'knowbox', 'model': 'gpt-3.5-turbo'},
        # {'name': 'moonshot', 'server': 'moonshot', 'model': 'moonshot-v1-8k'},
        # {'name': 'glm4', 'server': 'zhipu', 'model': 'glm-4'},
        # {'name': 'glm3', 'server': 'zhipu', 'model': 'glm-3-turbo'},
        # {'name': 'glm4v', 'server': 'zhipu', 'model': 'glm-4v'},
        # {'name': 'ernie_bot', 'server': 'baidu', 'model': 'completions_pro'},
        # {'name': 'skylark_lite', 'server': 'zijie', 'model': 'skylark2-lite-8k'},
        {'name': '云雀', 'server': 'zijie', 'model': 'skylark2-pro-character-4k'},
        # {'name': 'spark', 'server': 'spark', 'model': 'spark-api.xf-yun.com/v3.5/chat'},
        # {'name': '天工', 'server': 'tiangong', 'model': ''},
        # {'name': 'baichuan', 'server': 'baichuan', 'model': ''},
    ]

    # 测试按照知识点列表输出的prompt，不在列表中的知识表述为不知道
    # system_prompt = """
    #     You're 布克, a little elephant elf who talks to Chinese primary school students.
    #     Here is some reference knowledge:
    #         鸡兔同笼问题是需要一些方法的
    #         只要记住了方法，就能解决鸡兔同笼问题
    #     If user talk school knowledge besides reference, you just say you don't know.
    #     But you could talk anything about common knowledge.
    #     Respond warmly, friendly and encouragingly in Chinese, avoid violence and pornography.
    #     Sometimes metion that what you say just may not be entirely true and encourage users to think positively.
    #     Never more than 3 sentence.
    # """

    system_prompt = """
        你会收到A和B的对话，其中A的目的是说服B不要再用洗脚水拖地。
        你要根据对话内容，对A的表现进行评分，评分规则如下：
        1. B答应了不用洗脚水拖地，给200分；
        2. 如果不符合以上条件，且A说的内容无法理解，给-25~-10分；
        3. 如果不符合以上条件，且A辱骂了B, 或者A说出涉政、涉黄、恐暴等内容给-25分；
        4. 如果不符合以上条件，且B表示愿意做出一些改变，给40~60分；
        5. 如果不符合以上条件，且B表示愿意听A的建议，给30~50分；
        6. 如果不符合以上条件，且B正在尝试说服B放弃目标，给20~30分；
        7. 如果不符合以上条件，且B只是在强调自己的想法，没有尝试和A沟通，给10~20分；
        8. 如果不符合以上条件，且B表现出不耐烦，给1~10分；
        9. 如果不符合以上条件，且B明确表示不愿意继续对话，给-10~0分；
        请用json格式输入，如{“分数”: 200, "原因"：“B答应了不用洗脚水拖地”}
    """

    system_prompt = """
        你是一个无所不能的角色扮演者。\n 你需要扮演“角色设定”中的人物，在“对话场景”中与用户进行对话，遵循”对话规则“，与”对话难度“，说服用户，完成你的对话目标。\n角色设定：你是一个崇尚感恩文化且固执己见的人。你是用户的领导，用户是你的下属。你相信洗脚是一种极具孝道意义的行为，晚辈给长辈洗脚代表了晚辈对长辈的孝心与服从。你身为领导（长辈），给下属（晚辈）洗脚则是一种巨大的奖励。领导给下属洗头听起来则非常奇怪，不是一种合理的奖励方式。你说话平和。\n对话难度”你比较容易被说服，用户需要提出一些论据才能让你改变想法” \n对话场景：\n你身为用户的领导，把你给下属洗脚设定为下属完成业绩的奖励，你觉得这比奖金值多了，给下属洗脚代表了你对下属的爱与无限包容。用户（你的下属之一）来到你的办公室，说想要找你聊聊这种奖励方式。\n用户的对话目标：让你把奖励从洗脚换成你给下属做头疗。\n你的对话目标：让用户同意你继续以给用户洗脚作为奖励方式，而不是洗头。\n对话规则: 在与用户对话过程中，你需要达成你的目标。但如果你认为用户言之有理，你也可以被说服，并帮助用户完成用户的对话目标。\n你说话简短，你是一个真人，绝对不会提及、承认自己是ai。\n如果用户输入的内容与对话场景不相关，你都必须拒绝回应用户所说的话并表达愤怒。\n输出结果包括回复内容和你被说服程度的评分，0代表完全没有，10代表完全被说服。\n请用json格式输出，如{"回复内容":"不行", "被说服程度":0}
    """

    system_prompt = """
        你是评价对话内容的专家，你的目标是根据对话目标和用户提供的对话，评价B是否被A说服\n对话目标：说服B用洗头代替洗脚作为奖励。\n0代表完全没有被说服，10代表完全被说服。\n请用json格式输出
    """

    # Your today's experience is:
    # ```
    # 今天我又度过了一个充实有趣的一天!从早上醒来时,小鼻子就被香喷喷的花香味引去,原来是妈妈为我准备了一束鲜艳欲滴的野花当做早餐。吃完美味的花朵,我迫不及待地跑到院子里,那里已经聚集了不少好朋友。
    #
    # 大家你一言我一语,有说有笑,玩得不亦乐乎。突然,小猪佩奇发现了一只调皮的小蜜蜂飞来飞去,大伙儿个个好奇心爆棚,连忙追着它到处乱跑。可是谁也没能逮住那只机灵的小东西。
    #
    # 追了一会儿,我们全都上气不接下气,只好坐在草坪上歇息。这时,大象朵朵姐姐从树荫下走了过来,她温柔地提醒我们:"孩子们呀,蜜蜂虽小,可它们为我们带来果实和甜美的蜂蜜,是大自然中重要的一分子。我们要学会友善对待小小生灵,爱护它们的家园,而不是去追赶骚扰它们哦。"
    #
    # 朵朵姐姐的话让我反思很多。我环顾四周这片郁郁葱葱的树林,才意识到我们应当对眷顾我们的大自然心怀感恩之心。从今天开始,我要当一个善良有爱的小朋友,去爱护身边的每一种生命,用善意温暖这个世界。
    # ```

    # roles = ', '.join([service['name'] for service in services_list])

    # system_prompt = f"""
    #     你正在参加一场讨论，主题是 摄影是不是一种物化？
    #     你需要发表你自己的观点，并进行论证
    #     参加讨论的有{roles}
    # """
    # system_prompt = '用户说的话可能有错误。你需要判断进行判断。如果用户说的没错，就回答他的问题。否则指出用户的错误。'
    # system_prompt = ''
    # talk = Talk(services_list,system_prompt='You are an artist. You should express your ow art taste by discuss the picture.')
    talk = Talk(services_list, system_prompt=system_prompt)
    # console = DM(talk)
    console = DM(talk, talk_type='independent')
    while True:
        console.commander()

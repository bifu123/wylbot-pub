'''
本脚本向您演示了一个并行模式 下的插件编写方式、脚本用几个函数分别收集了有关元龙居士的信息，最后交与LLM大模型综合推理
'''

import aiohttp
import asyncio
from models_load import *

# LLM调用所需参数
msg = 0 # 串行模式返回数值类型时需要
bot_nick_name = ""
user_nick_name = ""
user_state = ""
name_space = ""
source_id = ""
# LLM调用示例
# response_message = asyncio.run(chat_generic_langchain(bot_nick_name, user_nick_name, source_id, message, user_state, name_space))



################ 参数说明 #################
# priority:  插件的优先级，数值越小，越优先执行
# post_type: 来自onebot协议的类型
#             1. message: 消息事件
#             2. request: 请求事件
#             3. notice: 通知事件
#             4. meta_event: 元事件
# user_state: 当前用户（群）所处的状态
#             1. 聊天
#             2. 文档问答
#             3. 知识库问答
#             4. 网站问答
# data：      监听到的所有数据的json   
# block:      是否阻断拦截，如果为Ture，将会执行完当前函数就结束，不再往下一个函数执行
# name_space: 插件命名空间，用于精准定位到执行某一功能的一个或多个函数
# role:       拥有执行插件权限命名空间的微信号、微信群列表

# - 串行模式 serial：  所有函数结果会按照优先级执行，上一个函数结果是下一个函数的输入，最后一个函数的结果为最终结果。
# - 并行模式 parallel：所有函数结果会按照优先级执行，所有函数必须返回一个字符类型结果（可以是""），最后结果是所有函数的拼合。
# - 两种模式会最终在主程序中调用拼合，一并交给LLM推理。
# - 主函数中名必须与@装饰函数名一致。

# *** 插件是很消耗 Token 的


################# 主函数 ##################
def fun_my_plugin(name_space, function_type, post_type, user_state:list, priority, role=[], block=False):
    def decorator(func):
        func._name_space = name_space
        func._function_type = function_type
        func._post_type = post_type
        func._priority = priority
        func._user_state = user_state
        func._role = role
        func._block = block
        return func
    return decorator





################# 子函数 ##################
# 插件函数示例1
@fun_my_plugin(name_space="元龙居士", function_type="parallel", post_type="message", user_state=["聊天"], priority=3, role=["21122263971@chatroom","cbf_415135222"])
def fun_1(data):
    
    global bot_nick_name
    global user_nick_name
    global user_state
    global name_space
    global source_id
    
    # 从data中拿出发送推理所需参数（本插件非必须）
    message = data["message"]
    bot_nick_name = data["bot_nick_name"]
    user_nick_name = data["user_nick_name"]
    source_id = data["source_id"]
    user_state = data["user_state"]
    name_space = data["name_space"]
    
    # 必须返回字符结果
    msg = f"他今年45岁了"
    return msg

# 插件函数示例2
@fun_my_plugin(name_space="元龙居士", function_type="parallel", post_type="message", user_state=["聊天"], priority=4, role=["21122263971@chatroom","cbf_415135222"])
def fun_2(data):
    msg = f"元龙居士是一个养猪的人、他喜欢国学文化、创办了元龙山寨"
    return msg

# 插件函数示例3
@fun_my_plugin(name_space="元龙居士", function_type="parallel", post_type="message", user_state=["聊天"], priority=5, role=["21122263971@chatroom","cbf_415135222"])
def fun_3(data):
    msg = f"他居住的地方海拔1800米"
    return msg

# 插件函数示例4
@fun_my_plugin(name_space="元龙居士", function_type="parallel", post_type="message", user_state=["聊天"], priority=6, role=["21122263971@chatroom","cbf_415135222"], block=True)
def fun_4(data):
    msg = f"他的头发是白的"
    return msg








# 从内置模块导入
import os
import shutil
import json
import sys
from sys import argv
import re
import base64
import importlib.util
import inspect
import subprocess
import getpass
import time
import stat


# 从文件导入
from models_load import *
from send import *
from commands import *
from do_history import save_chat_history



# 文档加工
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader, PythonLoader 
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter # 分割文档
from langchain_community.vectorstores import Chroma # 量化文档数据库


# 链结构
from langchain.chains import RetrievalQA #链

# 语义检索
from langchain.schema.runnable import RunnableMap
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 站点地图
import xml.dom.minidom
import datetime
from urllib import request
from bs4 import BeautifulSoup


from pathlib import Path

# 异步函数
import asyncio
import aiohttp





# 加载 embedding 过程
def load_retriever(db_path, embedding):
    vectorstore_from_db = Chroma(
        persist_directory = db_path,         # Directory of db
        embedding_function = embedding       # Embedding model
    )
    retriever = vectorstore_from_db.as_retriever()
    return retriever

# 消息中是否包含文件
def is_upload_file(bot_id, BytesExtra):
    # 解码 Base64 数据
    compressed_data = base64.b64decode(BytesExtra)
    # 匹配路径
    match = re.search(bytes(f'{bot_id}.*\\..*', 'utf-8') + b'.*', compressed_data) # 匹配 wxid_a2qwn1yzj30722 及其之后的所有字符
    if match:
        try:
            file_path = match.group().decode() # 对于图片等会出错
            filename = os.path.basename(file_path)
            username = getpass.getuser() # 获取当前用户名
            full_path = rf'{file_receive_path}\{file_path}' # 构建文件路径
            full_path = os.path.normpath(full_path)  # 标准化路径，确保路径分隔符和大小写符合 Windows 的规范
            return full_path, filename
        except:
            return "nothing", "nothing"
    else:
        return "nothing", "nothing"

# 检查文件的函数
def check_file_extension(file_name, allowed_extensions):
    file_ext = file_name[file_name.rfind("."):].lower()
    return file_ext in allowed_extensions

# 定义下载文件的函数
def download_file(url: str, file_name: str, download_path: str, allowed_extensions):
    if check_file_extension(file_name, allowed_extensions):
        # 下载文件
        response = requests.get(url)

        if response.status_code == 200:
            # 检查下载目录是否存在，如果不存在则创建
            if not os.path.exists(download_path):
                os.makedirs(download_path)
            
            # 将文件保存到指定路径
            file_path = os.path.join(download_path, file_name)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            msg = f"文件成功保存: {file_path}"
        else:
            msg = f"文件上传失败： {response.status_code}"
    else:
        extensions_string = ", ".join(allowed_extensions)
        # msg = f"你上传的文件我将不会保存到服务器上，它只会保存在群文件里。我能为你保存这些文件类型：{extensions_string}"
        msg = ""
    return msg

# 定义移动文件的函数
def move_file(source_path, file_name, target_path):
    current_permissions = os.stat(source_path).st_mode # 获取文件的当前权限    
    new_permissions = current_permissions | stat.S_IWRITE # 取消只读属性   
    os.chmod(source_path, new_permissions) # 更改文件的权限  
    if not os.path.exists(target_path): # 如果目标路径不存在，则创建
        os.makedirs(target_path)    
    file_path = os.path.join(target_path, file_name) # 构建目标路径
    shutil.copyfile(source_path, file_path) # 复制文件、覆盖保存
    os.remove(source_path) # 删除原始文件
    current_permissions = os.stat(file_path).st_mode # 获取文件的当前权限
    os.chmod(file_path, new_permissions) # 更改文件的权限   


# 显示文件夹下所有文件的函数
def get_files_in_directory(directory):
    directory_path = Path(directory)
    files = []
    for item in directory_path.iterdir():
        if item.is_file():
            files.append(str(item.resolve()))  # 将文件的绝对路径添加到列表中
        elif item.is_dir():
            files.extend(get_files_in_directory(item))  # 递归获取子文件夹中的文件
    return files

# 从数据库中读取当前群的允许状态
def get_allow_state(group_id):
    # 读取群消息开关
    try:
       allow_state = get_allow_state_from_db(group_id)
       if allow_state == "on":
           chat_type_allow = ["private", "group", "group_at"]
       else:
           chat_type_allow = ["private", "group_at"]
    except:
        chat_type_allow = ["private", "group_at"]
    return chat_type_allow

# 匹配URL的函数
def get_urls(text):
    # 定义一个正则表达式模式，用于匹配URL
    url_pattern = r'https?://\S+'
    # 使用findall函数查找文本中所有匹配的URL
    urls = re.findall(url_pattern, text)
    # 如果找到了URL，则返回True，否则返回False
    if urls:
        encoded_urls = base64.b64encode(json.dumps(urls).encode()).decode()   
        return "yes", encoded_urls
    else:
        return "no", "nothing"

# 匹配图片的函数
def get_image(text):
    # 使用正则表达式进行匹配
    pattern = r'\[CQ:image,file=(.*?),subType=\d+,url=(.*?)\]'
    matches = re.findall(pattern, text)

    # 如果匹配成功，返回 URL 地址和 True
    if matches:
        img = matches[0][1]
        return "yes", img
    else:
        return "no", "nothing"

# 匹配命名空间命令
def get_name_space(text):
    pattern = r"::|：：[^:]+"
    matches = re.findall(pattern, text)
    if matches:
        return "yes", matches[0]
    else:
        return "no", "nothing"

# 加载插件、构建query的函数
def get_response_from_plugins(name_space_p, post_type_p, user_state_p:list, data, source_id):
    # 存储每个函数的结果
    try:
        message = data["message"]
    except:
        message = ""

    plugin_dir = 'plugins'


    results = []
    result_serial = None  # 初始值设为None
    result_parallel = ''  # 用于并行执行的结果串联
    
    # 遍历plugins目录下的所有文件
    for filename in os.listdir(plugin_dir):
        if filename.endswith('.py'):
            plugin_path = os.path.join(plugin_dir, filename)
            # 动态导入模块
            spec = importlib.util.spec_from_file_location("plugin_module", plugin_path)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)
            
            # 获取模块中的所有函数及其优先级
            functions_with_priority = [
                (
                    getattr(plugin_module, func),
                    getattr(plugin_module, func)._name_space,
                    getattr(plugin_module, func)._priority,
                    getattr(plugin_module, func)._function_type,
                    getattr(plugin_module, func)._post_type,
                    getattr(plugin_module, func)._user_state,
                    getattr(plugin_module, func)._role,
                    getattr(plugin_module, func)._block
                )
                for func in dir(plugin_module)
                if callable(getattr(plugin_module, func)) and hasattr(getattr(plugin_module, func), '_priority')
            ]

            
            # 根据优先级对函数进行排序
            functions_with_priority.sort(key=lambda x: x[2])
            

            # 依次执行函数
            for function, name_space, priority, function_type, post_type, user_state, role, block in functions_with_priority:
                # 判断function_type、post_type和user_state是否满足特定条件
                if function_type == "serial" and post_type == post_type_p and user_state_p[0] in user_state and name_space == name_space_p:
                    if source_id in role or role == []:
                        if result_serial is None:
                            result_serial = data
                            # # 如果result为None，则根据函数参数类型设定初始值
                            # if 'dict' in str(function.__annotations__.values()):
                            #     result_serial = data
                            # elif 'str' in str(function.__annotations__.values()):
                            #     result_serial = ''
                            # 可以根据其他可能的参数类型继续添加条件
                        result_serial = function(result_serial)
                        # 如果block=True，则结束循环，不再执行后续函数
                        if getattr(function, '_block', True):
                            break

                elif function_type == "parallel" and post_type == post_type_p and user_state_p[0] in user_state and name_space == name_space_p:
                    if source_id in role or role == []:
                        result_parallel += f"{function(data)}"
                        result_parallel += "\n"

                        # 如果block=True，则结束循环，不再执行后续函数
                        if getattr(function, '_block', True):
                            break

    
    
    # 将每个函数的结果存储起来
    if result_serial is not None or result_parallel != "":
        results.append(f"{result_parallel}" + "\n" + f"{result_serial}")
        # 将所有结果组合起来
        result = "\n".join(results)
        result = result.replace("None", "").replace("\n\n", "\n")
        # 准备问题（将从插件获取的结果与当前问题拼接成上下文供LLM推理)
        query = f"{result}" + f"\n{message}"
    else:
        # 准备问题（将从插件获取的结果与当前问题拼接成上下文供LLM推理)
        # query = """不管用户的问题是什么，都请输出：\n你没有权限访问命名空间：%s\n- 不要添加你的任何理解和推理\n- 不要添加任何其它的标点符号和空格\n- 不要添加""和''""" % name_space_p
        query = message
    # 输出结果
    print("=" * 50)
    print(f"插件请求结果：\n\n{query}\n")
    return query

# 获取当前用户状态
def get_user_state_from_db(user_id, source_id):
    # 获取当前用户状态
    user_state = get_user_state(user_id, source_id)
    if user_state is None:
        user_state = "聊天"
        switch_user_state(user_id, source_id, user_state)
    return user_state

# 从JSON文件获得所有自定义命令名称和主体
def get_custom_commands_from_json():
    try:
        command_names = [command['command_name'] for command in commands_json]
        return command_names, commands_json
    except:
        return [], {}

# 根据command_name获得自定义命令单条JSON
def get_custom_commands_single(command_name, commands_json):
    custom_commands_single = None
    for command in commands_json:
        if command['command_name'] == command_name:
            custom_commands_single = command
            break

    return custom_commands_single

# 匹配是否群聊@
def is_group_at(text):
    # 正则表达式模式
    pattern = r"@.+?\s"
    # 使用 re 模块的 search 函数进行匹配
    match = re.search(pattern, text)
    # 如果匹配成功，输出 True，否则输出 False
    if match:
        return match.group()
    else:
        return None
#**************** 消息处理 ********************************************
def message_action(data):
    print("\n", "=" * 20, "参数收集", "=" * 20)
    
    # 定义一个存储消息信息的字典
    message_info = {}  
    
    # 消息类型
    message_info["post_type"] = "text"
    
    # 机器人微信号
    bot_id = data["wxid"]
    message_info["bot_id"] = bot_id
    
    # 机器人昵称
    bot_nick_name = requests.get(http_url + "/api/accountbywxid?wxid=" + bot_id).json()["data"]["nickname"]
    message_info["bot_nick_name"] = bot_nick_name
    
    # 是否包含文件
    BytesExtra = data["data"][0]["BytesExtra"]
    is_file = is_upload_file(bot_id, BytesExtra)
    message_info["is_file"] = is_file
 
    
    # 获取取消息内容
    group_at_string = is_group_at(data["data"][0]["StrContent"])
    if group_at_string is not None:
        message_info["message"] = data["data"][0]["StrContent"].replace(group_at_string, "")
    else:
        message_info["message"] = data["data"][0]["StrContent"]
        
    
    # 判断聊天类型
    if "@chatroom" in data["data"][0]["StrTalker"]: # 群聊
        if f"@{bot_nick_name}" in data["data"][0]["StrContent"] :
            method = "group_at"
            at = "yes"
        else:
            method = "group"
            at = "no"          
        user_id = data["data"][0]["Sender"]
        source_id = data["data"][0]["StrTalker"]
        group_id = data["data"][0]["StrTalker"]

        group_name = requests.get(http_url + "/api/dbchatroom?wxid=" + user_id).json()
        print(group_name)

    else: # 私聊
        method = "private"
        at = "no"
        user_id = data["data"][0]["StrTalker"]
        source_id = data["data"][0]["StrTalker"]
        group_id = "no"
        group_name = "nothing"
    
    # 获取用户昵称
    user_nick_name = requests.get(http_url + "/api/accountbywxid?wxid=" + user_id).json()["data"]["nickname"]
    
    message_info["user_id"] = user_id
    message_info["user_nick_name"] = user_nick_name
    message_info["method"] = method
    message_info["at"] = at
    message_info["group_id"] = group_id
    message_info["source_id"] = source_id
    # message_info["group_name"] = group_name
    
    # 获取当前群允许的聊天类型
    chat_type_allow = get_allow_state(group_id)
    message_info["chat_type_allow"] = chat_type_allow
    
    # 获取用户状态
    user_state = get_user_state_from_db(message_info["user_id"], message_info["source_id"])
    message_info["user_state"] = user_state


    # 获取name_space
    name_space = get_user_name_space(message_info["user_id"], message_info["source_id"])
    message_info["name_space"] = name_space


    # 以 | 分割找出其中的命令
    command_parts = message_info["message"].split("|")
    command_name = command_parts[0]
    message_info["command_name"] = command_name
    message_info["command_parts"] = command_parts
    
    # 是否包含图片
    message_info["is_image"] = get_image(message_info["message"])
    
    # 是否包含URL
    message_info["is_url"] = get_urls(message_info["message"])
    
    # 是否包含命名空间命令
    is_name_space_command = get_name_space(message_info["message"])
    message_info["is_name_space"] = is_name_space_command

    # 当前用户锁状态
    current_lock_state = get_user_lock_state(user_id, source_id, user_state)
    message_info["current_lock_state"] = current_lock_state
    
    
    # 如果包含自定义命令
    custom_commands_list = get_custom_commands_from_json()
    message_info["custom_commands_name"] = custom_commands_list[0]


    message_info["detail"] = {
        "info": "unknown"
    }
    
    
    # 参数收集完毕，格式化输出                                                                 
    formatted_json = json.dumps(message_info, indent=4, ensure_ascii=False)
    print(formatted_json)
    print("当前状态：", user_state)

    data = message_info
    #****************** 参数收集完毕 *************************
    
    
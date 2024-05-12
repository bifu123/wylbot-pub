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
        file_path = match.group().decode()
        filename = os.path.basename(file_path)
        username = getpass.getuser() # 获取当前用户名
        full_path = rf'{file_receive_path}\{file_path}' # 构建文件路径
        full_path = os.path.normpath(full_path)  # 标准化路径，确保路径分隔符和大小写符合 Windows 的规范
        return full_path, filename
    else:
        return "nothing", "nothing"
\
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
        msg = f"你上传的文件我将不会保存到服务器上，它只会保存在群文件里。我能为你保存这些文件类型：{extensions_string}"
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
    # msg = f"文件成功保存: {file_path}"  
    # return msg


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
    pattern = r"::[^:]+"
    matches = re.findall(pattern, text)
    if matches:
        return "yes", matches[0]
    else:
        return "no", "nothing"

# 加载插件、构建query的函数
def get_response_from_plugins(name_space_p, post_type_p, user_state_p, data):
    # 存储每个函数的结果
    try:
        message = data["message"]
    except:
        message = ""

    plugin_dir = 'plugins'


    results = []
    # 遍历plugins目录下的所有文件
    for filename in os.listdir(plugin_dir):
        if filename.endswith('.py'):
            plugin_path = os.path.join(plugin_dir, filename)
            # 动态导入模块
            spec = importlib.util.spec_from_file_location("plugin_module", plugin_path)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)
            
            # 获取模块中的所有函数及其优先级
            functions_with_priority = [(getattr(plugin_module, func), getattr(plugin_module, func)._name_space, getattr(plugin_module, func)._priority, getattr(plugin_module, func)._function_type, getattr(plugin_module, func)._post_type, getattr(plugin_module, func)._user_state, getattr(plugin_module, func)._block) for func in dir(plugin_module) if callable(getattr(plugin_module, func)) and hasattr(getattr(plugin_module, func), '_priority')]
            
            # 根据优先级对函数进行排序
            functions_with_priority.sort(key=lambda x: x[2])
            
            result_serial = None  # 初始值设为None
            result_parallel = ''  # 用于并行执行的结果串联
            # 依次执行函数
            for function, name_space, priority, function_type, post_type, user_state, block in functions_with_priority:
                # 判断function_type、post_type和user_state是否满足特定条件
                if function_type == "serial" and post_type == post_type_p and user_state == user_state_p and name_space == name_space_p:
                    if result_serial is None:
                        # 如果result为None，则根据函数参数类型设定初始值
                        if 'dict' in str(function.__annotations__.values()):
                            result_serial = {}
                        elif 'str' in str(function.__annotations__.values()):
                            result_serial = ''
                        # 可以根据其他可能的参数类型继续添加条件
                    result_serial = function(data=result_serial)  # 将data作为参数传递给函数
                    # 如果block=True，则结束循环，不再执行后续函数
                    if getattr(function, '_block', True):
                        break
                elif function_type == "parallel" and post_type == post_type_p and user_state == user_state_p and name_space == name_space_p:
                    result_parallel += f"{function(data)}"
                    result_parallel += "\n"

                    # 如果block=True，则结束循环，不再执行后续函数
                    if getattr(function, '_block', True):
                        break
            
            # 将每个函数的结果存储起来
            results.append(f"{result_parallel}" + "\n" + f"{result_serial}")
    
    # 将所有结果组合起来
    result = "\n".join(results)
    result = result.replace("None", "").replace("\n\n", "\n")

    # 输出结果
    print("=" * 50)
    print(f"插件返回结果：\n\n{result}\n")
    # 准备问题（将从插件获取的结果与当前问题拼接成上下文供LLM推理)
    query = f"{result}" + f"\n{message}"
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
    message_info["post_type"] = "message"
    
    # 机器人微信号
    bot_id = data["wxid"]
    message_info["bot_id"] = bot_id
    
    # 机器人昵称
    bot_nickname = requests.get(http_url + "/api/accountbywxid?wxid=" + bot_id).json()["data"]["nickname"]
    message_info["bot_nickname"] = bot_nickname
    
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
        if f"@{bot_nickname}" in data["data"][0]["StrContent"] :
            chat_type = "group_at"
            at = "yes"
        else:
            chat_type = "group"
            at = "no"          
        user_id = data["data"][0]["Sender"]
        source_id = data["data"][0]["StrTalker"]
        group_id = data["data"][0]["StrTalker"]
    else: # 私聊
        chat_type = "private"
        at = "no"
        user_id = data["data"][0]["StrTalker"]
        source_id = data["data"][0]["StrTalker"]
        group_id = "no"
        
    message_info["user_id"] = user_id
    message_info["chat_type"] = chat_type
    message_info["at"] = at
    message_info["group_id"] = group_id
    message_info["source_id"] = source_id
    
    # 获取当前群允许的聊天类型
    chat_type_allow = get_allow_state(group_id)
    message_info["chat_type_allow"] = chat_type_allow
    
    # 获取用户状态
    user_state = get_user_state_from_db(message_info["user_id"], message_info["source_id"])
    message_info["user_state"] = user_state
    
    # 组装文件路径
    if chat_type in ("group_at", "group"):  
        user_data_path = os.path.join(data_path, "group_" + message_info["group_id"]) # 群文件路径名
    else:
        user_data_path = os.path.join(data_path, message_info["user_id"]) # 用户文件路径名
    user_db_path = os.path.join(db_path, message_info["user_id"])
    message_info["user_data_path"] = user_data_path
    message_info["user_db_path"] = user_db_path
    
    # 确定用户embedding数据库目录和文档目录
    if message_info["chat_type"] in ("group_at", "group"):
        embedding_data_path = os.path.join(data_path, "group_"+ group_id)
        embedding_db_path_tmp = os.path.join(db_path, "group_"+ group_id)
        embedding_db_path_tmp_site = os.path.join(db_path, "group_"+ group_id + "_site")
    elif message_info["chat_type"] == "private":
        embedding_data_path = user_data_path
        embedding_db_path_tmp = user_db_path
        embedding_db_path_tmp_site = user_db_path + "_site"
    message_info["embedding_data_path"] = embedding_data_path

    # 获取name_space
    name_space = get_user_name_space(message_info["user_id"], message_info["source_id"])
    message_info["name_space"] = name_space

    # 读取数据库当前source_id的embedding路径
    if get_path_by_source_id(message_info["source_id"]) is None:
        insert_into_db_path(message_info["source_id"], embedding_db_path_tmp, embedding_db_path_tmp_site) # 如果没有记录则插入
        embedding_db_path = embedding_db_path_tmp
        embedding_db_path_site = embedding_db_path_tmp_site
    else:
        embedding_db_path, embedding_db_path_site = get_path_by_source_id(message_info["source_id"]) # 如果存在则直接使用
    message_info["embedding_db_path"] = embedding_db_path
    message_info["embedding_db_path_site"] = embedding_db_path_site
    
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
    message_info["is_name_space_command"] = is_name_space_command

    # 当前用户锁状态
    current_lock_state = get_user_lock_state(user_id, source_id, user_state)
    message_info["current_lock_state"] = current_lock_state
    
    
    # 如果包含自定义命令
    custom_commands_list = get_custom_commands_from_json()
    message_info["custom_commands_name"] = custom_commands_list[0]
    
    
    # 参数收集完毕，格式化输出                                                                 
    formatted_json = json.dumps(message_info, indent=4, ensure_ascii=False)
    print(formatted_json)
    print("当前状态：", user_state)
    

    #****************** 参数收集完毕 *************************
    response_message_url = ""
    response_message_file = ""
    response_message_chat = ""
    
    # 如果包含URL但不包含图片，则启动URL解读
    if message_info["is_url"][0] == "yes" and message_info["is_image"][0] == "no":
        try:
            question = "请用中文对以上内容解读，并输出一个结论"
            if sys.platform.startswith('win'):
                # Windows 上的命令
                command = f"start cmd /c \"conda activate wylbot && python url_chat.py {message_info['is_url'][1]} {question} {chat_type} {message_info['user_id']} {group_id} {at} {source_id} {user_state} && exit\""
            elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
                # Linux 或 macOS 上的命令
                command = f"python url_chat.py {message_info['is_url'][1]} {question} {chat_type} {message_info['user_id']} {group_id} {at} {source_id} {user_state}; exit"
            # 执行命令
            process = subprocess.Popen(command, shell=True)
            # 持续尝试关闭窗口
            while True:
                # 等待进程结束
                if process.poll() is not None:
                    break
                
        except Exception as e:
            print(f"URL错误：{e}")
        response_message_url = ""

    # 如果包含文件
    if message_info["is_file"][0] != "nothing":
        source_path, file_name = message_info["is_file"]
        if check_file_extension(file_name, allowed_extensions) == True: # 如果文件扩展在允许的列表
            # 移动文件
            file_path_temp = f"{user_data_path}_chat_temp_{user_id}" # 构建临时文件路径
            while True:
                try:
                    move_file(rf"{source_path}", file_name, file_path_temp)
                    response_message_file = f"文件保存成功：{source_path}😊"
                    print(f"{user_state}处理文件成功")
                    break
                except Exception as e:
                    print(e)
                    print(f"{user_state}处理文件失败，重试中")
                    time.sleep(1) 
            # 文件解读
            if user_state not in ("文档问答", "知识库问答"):  # 如果状态非"文档问答", "知识库问答"，则则移动文件并启动文件解读
                question = "请用中文对以上内容分析，并输出一个结论" # 提示词            
                if sys.platform.startswith('win'): # 判断操作系统类型、打开新窗口执行命令
                    command = f"start cmd /c \"conda activate wylbot && python docs_chat.py {file_path_temp} {question} {chat_type} {user_id} {group_id} {at} {source_id} {user_state} && exit\""
                elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
                    command = f"gnome-terminal -- bash -c 'python docs_chat.py {file_path_temp} {question} {chat_type} {user_id} {group_id} {at} {source_id} {user_state}; exit'"
                subprocess.Popen(command, shell=True) # 执行命令  
                response_message_file = ""   
        else: # 如果文件扩展不在允许的列表、只删除文件并作提示
            # 删除文件
            while True:
                try:
                    current_permissions = os.stat(source_path).st_mode # 获取文件的当前权限    
                    new_permissions = current_permissions | stat.S_IWRITE # 取消只读属性   
                    os.chmod(source_path, new_permissions) # 更改文件的权限  
                    os.remove(source_path) # 删除原始文件
                    print(f"{user_state}删除文件成功")
                    # 提示
                    extensions_string = ", ".join(allowed_extensions) # 允许的文件类型
                    response_message_file = f"你上传的文件我不会保存，我能为你保存的文件类型：{extensions_string}😊" 
                    break
                except Exception as e:
                    print(e)
                    print(f"{user_state}删除文件失败，重试中")
                    time.sleep(1) 
 
    # 在允许回复的聊天类型中处理
    if message_info["chat_type"] in chat_type_allow and message_info["is_url"][0] == "no": 
        # 如果当前处于锁定状态
        if current_lock_state == 1:
            update_custom_command(message_info["message"], source_id, user_id, user_state, chat_type, group_id, at)
            # response_message_chat = ""
        else:
            # 切换命名空间命令
            if is_name_space_command[0] == "yes":
                delete_all_records(source_id, user_state, name_space) # 清空聊天历史
                name_space_command = message_info["message"].replace("::", "")
                switch_user_name_space(user_id, source_id, name_space_command)
                print(f"已切换到 【{name_space_command}】 命名空间")
                response_message_chat = f"已切换到 【{name_space_command}】 命名空间😊"

            # 其它命令和问答
            else:
                # 命令： /我的文档 
                if command_name == "/我的文档":
                    print("命令匹配！")
                    try:
                        all_file = get_files_in_directory(embedding_data_path)
                        files_str = "\n".join(all_file)  # 将文件列表转换为单一的字符串，每个文件路径占一行
                        if len(files_str) > 0:
                            if chat_type in ("group_at", "group"):
                                response_message_chat = "以下是你们的知识库文档：\n\n" + files_str + "\n\n如果要删除，请输使用删除命令： /删除文档|完整路径的文件名😊"
                            else:
                                response_message_chat = "以下是你的知识库文档：\n\n" + files_str + "\n\n如果要删除，请输使用删除命令： /删除文档|完整路径的文件名😊"
                        else:
                            response_message_chat = "你还没有文档，请先给我发送你的文档。😊"
                    except:
                        response_message_chat = "你还没有文档，请先给我发送你的文档。😊"

                # 命令： /删除文档 
                elif command_name == "/删除文档":
                    # 取得文件名
                    try:
                        file_path = command_parts[1]
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            response_message_chat = f"文件 '{file_path}' 已成功删除。注：聊天软件里的同名文档不会被清除，请手动删除😊"
                        else:
                            response_message_chat = f"文件 '{file_path}' 不存在，无法删除😊"
                    except:
                        response_message_chat = "命令错误😊"

                # 命令： /邀请1 
                elif command_name == "/邀请1":
                    try:
                        # 获取命令参数
                        tag_user_id = str(command_parts[1])
                        tag_source_id = source_id
                        tag_state = command_parts[2]
                        try:
                            tag_name_space = command_parts[3]
                        except:
                            tag_name_space = ""

                        # 改变对方状态:
                        switch_user_state(tag_user_id, tag_source_id, tag_state)

                        # 改变对方命名空间
                        if tag_name_space != "":    
                            switch_user_name_space(tag_user_id, tag_source_id, tag_name_space)
                            response_tag = f"【{user_id}】 邀请了你进入\n状态： 【{tag_state}】 \n命名空间：【{tag_name_space}】😊"
                        else:
                            response_tag = f"【{user_id}】 邀请了你进入\n状态： 【{tag_state}】😊"

                        response_message_chat =  f"已邀请😊"
                        # 给对方发送通知
                        try:
                            asyncio.run(answer_action(chat_type, tag_user_id, group_id, at, response_tag))
                        except:
                            pass
                    except Exception as e:
                        response_message_chat = f"邀请错误：{e}😊"

                # 命令： /清空文档 
                elif command_name == "/清空文档":
                    # 取得文件名
                    try:
                        if os.path.exists(user_data_path):
                            shutil.rmtree(user_data_path)
                            response_message_chat = f"文件 '{user_data_path}' 下所有文件已成功删除。注：聊天软件里的同名文档不会被清除，请手动删除😊"
                        else:
                            response_message_chat = f"文件夹 '{user_data_path}' 不存在，无法删除😊"
                    except:
                        response_message_chat = "命令错误😊"
                                
                # 命令： /量化文档 
                elif command_name == "/量化文档":
                    embedding_type = "file"
                    try:
                        # 判断操作系统类型
                        if sys.platform.startswith('win'):
                            # Windows 上的命令
                            command = f"start cmd /c \"conda activate wylbot && python new_embedding.py {embedding_data_path} {embedding_db_path} {source_id} {chat_type} {user_id} {group_id} {at} {embedding_type} && exit\""
                        elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
                            # Linux 或 macOS 上的命令
                            command = f"gnome-terminal -- bash -c 'python new_embedding.py {embedding_data_path} {embedding_db_path} {source_id} {chat_type} {user_id} {group_id} {at} {embedding_type}; exit'"   
                        # 执行命令
                        subprocess.Popen(command, shell=True)

                        response_message_chat = "正在量化，完成后另行通知，这期间你仍然可以使用你现在的文档知识库😊"
                    except Exception as e:
                        response_message_chat = f"量化失败：{e}😊"

                # 命令： /量化网站 
                elif command_name == "/量化网站":
                    embedding_type = "site"
                    site_url = base64.b64encode(json.dumps(command_parts[1]).encode()).decode()
                    try:
                        # 判断操作系统类型
                        if sys.platform.startswith('win'):
                            # Windows 上的命令
                            command = f"start cmd /c \"conda activate wylbot && python new_embedding.py {embedding_data_path} {embedding_db_path_site} {source_id} {chat_type} {user_id} {group_id} {at} {embedding_type} {site_url} && exit\""
                        elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
                            # Linux 或 macOS 上的命令
                            command = f"gnome-terminal -- bash -c 'python new_embedding.py {embedding_data_path} {embedding_db_path_site} {source_id} {chat_type} {user_id} {group_id} {at} {embedding_type} {site_url}; exit'"
                        # 执行命令
                        subprocess.Popen(command, shell=True)
                    except Exception as e:
                        print(f"URL错误：{e}")
                    response_message_chat = "这将需要很长、很长的时间...不过你可以问我些其它事😊"

                # 命令： /上传文档 
                elif command_name == "/上传文档":
                    # 取得文件名
                    response_message_chat = "请直接发送文档😊"

                # 命令： /文档问答 
                elif command_name == "/文档问答":
                    # 切换到 文档问答 状态
                    # 用数据库保存每个用户的状态
                    switch_user_state(user_id, source_id, "文档问答")
                    response_message_chat = "你己切换到 【文档问答】 状态。其它状态命令：\n/聊天\n/网站问答\n/知识库问答😊"
                
                # 命令： /网站问答 
                elif command_name == "/网站问答":
                    # 切换到 文档问答 状态
                    # 用数据库保存每个用户的状态
                    switch_user_state(user_id, source_id, "网站问答")
                    response_message_chat = "你己切换到 【网站问答】 状态。其它状态命令：\n/聊天\n/文档问答\n/知识库问答\n插件问答😊" 

                # 命令： /知识库问答 
                elif command_name == "/知识库问答":
                    # 切换到 文档问答 状态
                    # 用数据库保存每个用户的状态
                    switch_user_state(user_id, source_id, "知识库问答")
                    response_message_chat = "你己切换到 【知识库问答】 状态。其它状态命令：\n/聊天\n/文档问答\n/网站问答\n/插件问答😊"   

                # 命令： /聊天 
                elif command_name == "/聊天":
                    # 切换到 聊天 状态
                    # 用数据库保存每个用户的状态
                    switch_user_state(user_id, source_id, "聊天")
                    response_message_chat = "你己切换到 【聊天】 状态。其它状态命令：\n/网站问答\n/文档问答\n/知识库问答\n/插件问答😊" 

                # 命令： /插件问答
                elif command_name == "/插件问答":
                    switch_user_state(user_id, source_id, "插件问答")
                    response_message_chat = "你己切换到 【插件问答】 状态。其它状态命令：\n/聊天\n/网站问答\n/文档问答\n/知识库问答😊" 

                # 命令： /我的状态 
                elif command_name == "/我的状态":
                    # 从数据库中查找用户当前状态
                    user_state = get_user_state_from_db(user_id, source_id)
                    response_message_chat = f"【{user_state}】😊"
                
                # 命令： /我的命名空间 
                elif command_name == "/我的命名空间":
                    if name_space == "no":
                        response_message_chat = "你当前所在聊天对象中还没有插件，你可以创建插件，或用 ::命名空间 的命令切换到已有的插件命名空间😊"
                    else:
                        response_message_chat = "【" + name_space + "】😊"
                
                # 命令： /开启群消息 
                elif command_name == "/开启群消息":
                    try:
                        switch_allow_state(message_info["group_id"], "on")
                        response_message_chat = "现在不管谁说话，我都会在群里回答😊，如果嫌小的话多，你就发 /关闭群消息"
                    except Exception as e:
                        response_message_chat = f"群消息开启失败：{e}😊"

                # 命令： /关闭群消息 
                elif command_name == "/关闭群消息":
                    try:
                        switch_allow_state(message_info["group_id"], "off")
                        response_message_chat = "好的，小的先行告退，就不插嘴各位大人的聊天了，有需要时@我😊"
                    except Exception as e:
                        response_message_chat = f"群消息关闭失败：{e}😊"

                # 命令： /清空记录 
                elif command_name == "/清空记录":
                    try:
                        user_state = get_user_state_from_db(user_id, source_id)
                        delete_all_records(source_id, user_state, name_space)
                        response_message_chat = "消息已经清空😊"
                    except Exception as e:
                        response_message_chat = f"消息清空失败：{e}😊"
                
                # 命令： /{自定义命令}
                elif command_name in custom_commands_list[0]:
                    command_main = get_custom_commands_single(command_name, custom_commands_list[1])
                    print("自定义命令:",command_name)
                    do_custom_command(command_name, source_id, user_id, user_state, command_main, chat_type, group_id, at)
                    response_message_chat = ""

                # 和 LLM 对话
                else:
                    # 当状态为命令等待
                    if user_state == "命令等待":
                        update_custom_command(message_info["message"], source_id, user_id, user_state, chat_type, group_id, at) # 更新自定义命令
                        response_message_chat = ""
                    
                    # 当状态为知识库问答
                    elif user_state == "知识库问答":
                        if message_info["is_file"][0] == "nothing":
                            embedding, llm, llm_rag, must_use_llm_rag = get_models_on_request()
                            # 调用RAG
                            print(f"{user_state}:加载 {embedding_db_path} 的向量知识库...")
                            retriever = load_retriever(embedding_db_path, embedding)
                            # 准备问题
                            query = message_info["message"]
                            # 执行问答
                            response_message_chat = asyncio.run(run_chain(retriever, source_id, query, user_state, name_space))

                    # 当状态为插件问答
                    elif user_state == "插件问答":
                        query = get_response_from_plugins(name_space, message_info["post_type"], user_state, message_info)
                        # 执行问答
                        response_message_chat = asyncio.run(chat_generic_langchain(source_id, query, user_state, name_space))

                    # 当状态为网站问答
                    elif user_state == "网站问答":
                        embedding, llm, llm_rag, must_use_llm_rag = get_models_on_request()
                        # 调用RAG
                        print(f"加载 {embedding_db_path_site} 的向量知识库...")
                        retriever = load_retriever(embedding_db_path_site, embedding)
                        # 准备问题
                        query = message_info["message"]
                        # 执行问答
                        response_message_chat = asyncio.run(run_chain(retriever, source_id, query, user_state, name_space))       

                    # 文档问答。文档未经过分割向量化，直接发给LLM推理
                    elif user_state == "文档问答":
                        if message_info["is_file"][0] == "nothing":
                            # 准备问题
                            question = message_info["message"]
                            # 新开窗口执行问答
                            if sys.platform.startswith('win'):
                            # Windows 上的命令
                                command = f"start cmd /c \"conda activate wylbot && python docs_chat.py {embedding_data_path} {question} {chat_type} {user_id} {group_id} {at} {source_id} {user_state} && exit\""
                            elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
                                # Linux 或 macOS 上的命令
                                command = f"gnome-terminal -- bash -c 'python docs_chat.py {embedding_data_path} {question} {chat_type} {user_id} {group_id} {at} {source_id} {user_state}; exit'"
                            # 执行命令
                            subprocess.Popen(command, shell=True)

                    # 聊天。
                    else:
                        query = f'{message_info["message"]}'
                        response_message_chat = asyncio.run(chat_generic_langchain(source_id, query, user_state, name_space))

    if response_message_url is None:
        response_message_url = ""   
    if response_message_file is None:
        response_message_file = ""
    if response_message_chat is None:
        response_message_chat = ""    
             
    print(f"response_message_url:{response_message_url}")
    print(f"response_message_file:{response_message_file}")
    print(f"response_message_chat:{response_message_chat}")
    
    response_message = response_message_url + response_message_file + response_message_chat
    
    if response_message == "" or response_message is None:
        print("=" * 50, "\n",f"没有回复、无需发送消息")
    else:
        print("=" * 50, "\n",f"答案：{response_message}") 
        try: 
            asyncio.run(answer_action(chat_type, user_id, group_id, at, response_message))
        except Exception as e:
            print("=" * 50, "\n",f"发送消息错误：{e}")
        



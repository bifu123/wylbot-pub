import os
import sys
from sys import argv
import shutil
import requests
import json
import time
import base64

# 文档加工
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader, PythonLoader 

# 从文件导入
from send import *
from models_load import *
from do_history import save_chat_history

# 异步函数
import asyncio




print(f"接收到的参数：{sys.argv}")


# time.sleep(10000)

embedding_data_path = sys.argv[1]
# question = sys.argv[2]
question = json.loads(base64.b64decode(sys.argv[2]).decode())
chat_type = str(sys.argv[3])
user_id = str(sys.argv[4])
group_id = str(sys.argv[5])
at = str(sys.argv[6])
source_id = str(sys.argv[7])
user_state = str(sys.argv[8])
bot_nick_name = str(sys.argv[9])
user_nick_name = str(sys.argv[10])



print("*" * 40)
print(f"embedding_data_path:", embedding_data_path)
print(f"question:", question)
print(f"chat_type:", chat_type)
print(f"user_id:", user_id)
print(f"group_id:", group_id)
print(f"at:", at)
print(f"source_id:", source_id)
print(f"user_state:", user_state)
print(f"bot_nick_name:", bot_nick_name)
print(f"user_nick_name:", user_nick_name)
print("*" * 40)



# 文件夹加载器函数
def load_documents(data_path):
    print("正在加载" + data_path + "下的所有文档...")
    try:
        loader = DirectoryLoader(data_path, show_progress=True, use_multithreading=True)
        loaders = loader.load()
        # print(loaders)
        return loaders
    except Exception as e:
        print(e)
        return f"加载文档出错：{e}"


name_space = get_user_name_space(user_id, source_id)



# 清除原来的聊天历史
# delete_all_records(source_id, user_state, name_space)

query = f"{load_documents(embedding_data_path)}\n{question}"

# 插入记录
if chat_type == "group_at":
    query_message_insert = "@" + bot_nick_name + " " + query
else:
    query_message_insert = query
# do_chat_history(query_message_insert, source_id, bot_nick_name, query_message_insert, user_state, name_space)

asyncio.run(save_chat_history(source_id, bot_nick_name, query_message_insert, user_state, name_space))

try:
    response_message = asyncio.run(chat_generic_langchain(bot_nick_name, user_nick_name, source_id, query, user_state, name_space))
    # 如果是聊天状态，问答完成立即删除文件
    if user_state == "聊天":
        shutil.rmtree(embedding_data_path)
except Exception as e:
    response_message = f"错误：{e}"
    shutil.rmtree(embedding_data_path)



# 打印答案，发送消息
print("*" * 40)
print(f"答案： {response_message}")

# 发送消息
asyncio.run(answer_action(chat_type, user_id, group_id, at, response_message))

# 插入记录
if chat_type == "group_at":
    response_message_insert = "@" + user_nick_name + " " + response_message
else:
    response_message_insert = response_message
    # do_chat_history(response_message_insert, source_id, bot_nick_name, response_message_insert, user_state, name_space)

    asyncio.run(save_chat_history(source_id, bot_nick_name, response_message_insert, user_state, name_space))












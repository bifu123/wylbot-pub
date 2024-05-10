import os
import sys
from sys import argv
import shutil
import requests
import json
import time

# 文档加工
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader, PythonLoader 

# 从文件导入
from send import *
from models_load import *

# 异步函数
import asyncio




print(f"接收到的参数：{sys.argv}")


embedding_data_path = sys.argv[1]
question = sys.argv[2]
chat_type = str(sys.argv[3])
user_id = str(sys.argv[4])
group_id = str(sys.argv[5])
at = str(sys.argv[6])
source_id = str(sys.argv[7])
user_state = str(sys.argv[8])



print("*" * 40)
print(f"embedding_data_path:", embedding_data_path)
print(f"question:", question)
print(f"chat_type:", chat_type)
print(f"user_id:", user_id)
print(f"group_id:", group_id)
print(f"at:", at)
print(f"source_id:", source_id)
print(f"user_state:", user_state)
print("*" * 40)



# 文件夹加载器函数
def load_documents(data_path):
    print("正在加载" + data_path + "下的所有文档...")
    loader = DirectoryLoader(data_path, show_progress=True, use_multithreading=True)
    loaders = loader.load()
    print(loaders)
    return loaders


name_space = get_user_name_space(user_id, source_id)


# 调用通用聊天得出答案
try:
    # 清除原来的聊天历史
    delete_all_records(source_id, user_state, name_space)
    query = f"{load_documents(embedding_data_path)}\n{question}"
    response_message = asyncio.run(chat_generic_langchain(source_id, query, user_state, name_space)) 
    # 如果是聊天状态，问答完成立即删除文件
    if user_state == "聊天":
        shutil.rmtree(embedding_data_path)
except Exception as e:
    response_message = f"错误：{e}😊"
    shutil.rmtree(embedding_data_path)



# 打印答案，发送消息
print("*" * 40)
print(f"答案： {response_message}")
# 发送消息

asyncio.run(answer_action(chat_type, user_id, group_id, at, response_message))











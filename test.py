data = {
    "message": "你好",
    "chat_type_allow": [
        "private",
        "group_at"
    ],
    "user_id": "cbf_415135222",
    "chat_type": "private",
    "at": "no",
    "group_id": "no",
    "source_id": "cbf_415135222",
    "user_state": "插件问答",
    "user_data_path": "./data\\cbf_415135222",
    "user_db_path": "./chroma_db\\cbf_415135222",
    "embedding_data_path": "./data\\cbf_415135222",
    "name_space": "test",
    "embedding_db_path": "./chroma_db\\cbf_415135222",
    "embedding_db_path_site": "./chroma_db\\cbf_415135222_site",
    "command_name": "你好",
    "command_parts": [
        "你好"
    ],
    "is_image": [
        "no",
        "nothing"
    ],
    "is_url": [
        "no",
        "nothing"
    ],
    "is_name_space_command": [
        "no",
        "nothing"
    ],
    "current_lock_state": 0
}

# 异步函数
# import asyncio
# import aiohttp

# from send import answer_action
# asyncio.run(answer_action("private", "cbf_415135222", "no", "no", "你好"))

# import requests
# import json

# url = "http://192.168.66.29:8080/api/sendtxtmsg"
# data = {
#     "wxid": "cbf_415135222",
#     "content": "你好"
# }
# response = requests.post(url, json=data)
# print(response.text)
    
# # http://192.168.66.29:8080/api/sendtxtmsg?wxid=cbf_415135222&content=hello

from config import *
import requests
import json


user_id = "cbf_415135222"
nickname = requests.get(http_url + "/api/accountbywxid?wxid=" + user_id).json()["data"]["nickname"]

print(nickname)

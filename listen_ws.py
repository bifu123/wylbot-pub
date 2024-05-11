import threading
import websocket
import json
from config import ws_url, model_choice, must_use_llm_rag
from dal import *
from sqlite_helper import init_commands_table, init_models_table

# 初始化数据库命令表
init_commands_table()

# 初始化模型表
embedding = model_choice["embedding"]
llm = model_choice["llm"]
llm_rag = model_choice["llm_rag"]
init_models_table(embedding ,llm, llm_rag, must_use_llm_rag)

def on_message(ws, message):
    data = json.loads(message)
    handle_message(data)


def handle_message_thread(data):
    threading.Thread(target=handle_message, args=(data,)).start()

# def handle_notice_thread(data):
#     threading.Thread(target=handle_notice, args=(data,)).start()

def handle_message(data):
    # 处理私聊消息或群聊消息
    print("\n", "="*20, "Message","="*20)
    # formatted_json = json.dumps(data, indent=4, ensure_ascii=False)
    # print(formatted_json)
    print(data)
    if "😊" not in data["data"][0]["StrContent"]:
        message_action(data)

def on_error(ws, error):
    print("Error:", error)

def on_close(ws):
    # 连接关闭时重新连接
    print("Connection closed")
    ws.run_forever()

def on_open(ws):
    print("Connection established")

# 设置自动重连
def create_connection():
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

# 建立 WebSocket 连接
create_connection()

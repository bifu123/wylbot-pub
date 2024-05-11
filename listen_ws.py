import threading
import time
import websocket
import json
import pyautogui
from config import ws_url, model_choice, must_use_llm_rag
from dal import *
from sqlite_helper import init_commands_table, init_models_table

# 初始化数据库命令表
init_commands_table()

# 初始化模型表
embedding = model_choice["embedding"]
llm = model_choice["llm"]
llm_rag = model_choice["llm_rag"]
init_models_table(embedding, llm, llm_rag, must_use_llm_rag)

def press_enter_every_2_seconds():
    try:
        while True:
            # 模拟按下回车键
            pyautogui.press('enter')
            # print("enter")
            # 等待2秒
            time.sleep(2)
    except KeyboardInterrupt:
        print("程序已停止")

def on_message(ws, message):
    data = json.loads(message)
    handle_message(data)

def handle_message(data):
    print("\n", "=" * 20, "Message", "=" * 20)
    print(data)
    if "😊" not in data["data"][0]["StrContent"]:
        threading.Thread(target=message_action, args=(data,)).start()

def on_error(ws, error):
    print("Error:", error)

def on_close(ws):
    print("Connection closed")
    # 连接关闭时重新连接
    create_connection()

def on_open(ws):
    print("Connection established")

def create_connection():
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

# 启动 WebSocket 连接
create_connection()

# 启动按下回车键的线程
threading.Thread(target=press_enter_every_2_seconds).start()

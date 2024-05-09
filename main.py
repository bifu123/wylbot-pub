import json
import websocket

def on_message(ws, message):
    data = json.loads(message)
    print(data)

def on_open(ws):
    print("连接已打开")

def on_close(ws):
    print("连接已关闭")

# 创建WebSocket连接
ws = websocket.WebSocketApp('ws://127.0.0.1:8080/ws/generalMsg',
                            on_message=on_message,
                            on_open=on_open,
                            on_close=on_close)

# 运行WebSocket连接
ws.run_forever()
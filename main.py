import json
import websocket
import re
import base64
import getpass
import os




def is_file(bot_id, BytesExtra):
    # 解码 Base64 数据
    compressed_data = base64.b64decode(BytesExtra)
    print(compressed_data)
    # 匹配路径
    match = re.search(bytes(f'{bot_id}.*\\..*', 'utf-8') + b'.*', compressed_data) # 匹配 wxid_a2qwn1yzj30722 及其之后的所有字符，并且是一个文件路径,比如C:\Users\Administrator\Documents\WeChat Files\wxid_a2qwn1yzj30722\FileStorage\File\2024-05\notice.txt
    if match:
        file_path = match.group().decode()
        
        filename = os.path.basename(file_path)
        username = getpass.getuser() # 获取当前用户名
        full_path = rf'C:\Users\{username}\Documents\WeChat Files\{file_path}' # 构建文件路径
        full_path = os.path.normpath(full_path)  # 标准化路径，确保路径分隔符和大小写符合 Windows 的规范
        return full_path, filename
    else:
        return "nothing", "nothing"



    


def on_message(ws, message):
    data = json.loads(message)
    print("\n", "="*20, "Message","="*20)
    formatted_json = json.dumps(data, indent=4, ensure_ascii=False)
    print(formatted_json)
    
    bot_id = data["wxid"]
    BytesExtra = data["data"][0]["BytesExtra"]  
    print(is_file(bot_id, BytesExtra))



def on_open(ws):
    print("连接已打开")

def on_close(ws):
    print("连接已关闭")

# 创建WebSocket连接
ws = websocket.WebSocketApp('ws://192.168.66.29:8080/ws/generalMsg',
                            on_message=on_message,
                            on_open=on_open,
                            on_close=on_close)

# 运行WebSocket连接
ws.run_forever()

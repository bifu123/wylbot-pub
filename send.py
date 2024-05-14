import aiohttp
import asyncio
import requests
from config import http_url, chat_type_allow, admin_wxid


# 根据聊天类型发送消息的异步函数
async def answer_action(chat_type, user_id, group_id, at, response_message):
    # url
    url = http_url + "/api/sendtxtmsg"
    
    # 群中@
    if chat_type == "group_at": 
        nick_name = requests.get(http_url + "/api/accountbywxid?wxid=" + user_id).json()["data"]["nickname"]
        params = {
            "wxid": group_id,
            "content": f"@{nick_name} {response_message}", 
            "atlist": [user_id]
        }  
        
    # 私聊  
    elif chat_type == "private": 
        params = {
            "wxid": user_id, 
            "content": response_message
        } 
        
    # 群聊  
    elif chat_type == "group": 
        params = {
            "wxid": group_id,
            "content": response_message
        } 
        
    # 其它   
    else:
        params = {
            "wxid": admin_wxid, 
            "content": f"{user_id} 发送了未知类型消息"
        } 
  
    don_send_message = '''
    " [文字解释] 当然，我很高兴能为您解答问题。请问有什么需要我回答的问题？\n\n     [语音输出] 当然，我很高兴能为您解答问题。请问有什么需要我回答的问题？😊"
    '''
    # 发送消息
    if response_message == "" or response_message is None:
        pass
    else: 
        if chat_type in chat_type_allow:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=params) as response:
                    # 检查响应状态码
                    print(response.text)
                    if response.status == 200:
                        print("=" * 50, "\n消息已成功发送\n\n")
                    else:
                        print("=" * 50, "\n发送消息时出错:\n\n", await response.text())
        else:
            pass

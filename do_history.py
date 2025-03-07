from sqlite_helper import *
from config import *  
import sys     
import re

from datetime import datetime, timedelta



# 去除重复的昵称
def remove_repeated_nicknames(content, nick_name):
    # 使用格式化字符串将变量嵌入到正则表达式模式中
    pattern = rf'({re.escape(nick_name)})\1+'
    # 使用正则表达式替换，保留一个 nick_name
    result = re.sub(pattern, r'\1', content)
    return result



#  # 把从数据表的历史记录格式化成字典的格式
# def format_history(bot_nick_name, history):
#     system_prompt = {"user": "system", "content": f"你好，我的名字叫{bot_nick_name}，我会尽力解答大家的问题."}
#     result = []
#     result.append(system_prompt)
#     for item in history:
#         user = item[0]
#         content = item[1].replace("\u2005", " ")
#         time = item[2]
#         result.append({"user": user, "content": content, "time": time})
#     return result



from datetime import datetime, timedelta

def format_history(bot_nick_name, history):
    if not history:
        return []

    # 将字符串时间戳转换为 datetime 对象
    for i in range(len(history)):
        user, content, timestamp = history[i]
        history[i] = (user, content, datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S"))

    # 获取最早的时间并减去3秒
    first_time = history[0][2] - timedelta(seconds=3)

    # 系统提示
    system_prompt = {
        "user": "system",
        "content": f"你好，我的名字叫{bot_nick_name}，我会尽力解答大家的问题.",
        "time": first_time.strftime("%Y-%m-%d %H:%M:%S")
    }

    result = [system_prompt]

    for item in history:
        user = item[0]
        content = item[1].replace("\u2005", " ")
        time = item[2].strftime("%Y-%m-%d %H:%M:%S")

        message = {
            "user": user,
            "content": content,
            "time": time
        }

        result.append(message)

    return result




def get_chat_history(source_id, user_state, name_space, bot_nick_name, query): # 处理聊天记录

    history = fetch_chat_history(source_id, user_state, name_space) # 从数据库中提取source_id的聊天记录
    chat_history = format_history(bot_nick_name, history)
    history_size_now = sys.getsizeof(f"{chat_history}") + sys.getsizeof(f"{query}") # 如果超过预定字节大小，删除记录
    print("=" * 50)

    formatted_json = json.dumps(chat_history, indent=4, ensure_ascii=False)

    print(f"预计聊天记录大小：{history_size_now}\n聊天记录：\n{formatted_json}")
    
    while history_size_now > chat_history_size_set:
        delete_oldest_records(source_id, user_state, name_space) # 删除数据库中时间最旧的1条记录

        if chat_history and len(chat_history) > 1:
            chat_history = chat_history[1:]
            print("* 删除最旧一条记录")

        history = fetch_chat_history(source_id, user_state, name_space) # 从数据库中提取source_id的聊天记录
        chat_history = format_history(bot_nick_name, history)
        history_size_now = sys.getsizeof(f"{chat_history}") + sys.getsizeof(f"{query}")

    return chat_history




# 处理聊天记录
async def save_chat_history(source_id, user, content, user_state, name_space):
    # 去除重复的昵称we
    content = remove_repeated_nicknames(content, user)
    content = content.encode('utf-8').decode('utf-8')
    content = content.replace("\u2005", " ")
    history_size_now = sys.getsizeof(f"{content}")
    # 如果超过预定字节大小就放弃写入
    if not history_size_now > chat_history_size_set:
        # 插入当前数据表 source_id、query、result
        insert_chat_history(source_id, user, content.replace("😊", ""), user_state, name_space)
    else:
        print("记录过大，放弃写入")


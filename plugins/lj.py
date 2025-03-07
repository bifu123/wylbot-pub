

'''
本脚本为元龙机器人插件获取客人送来的礼金SQL查询结果
'''
import aiohttp
import asyncio
from models_load import *





# 主函数
def lj(name_space, function_type, post_type, user_state:list, priority, role=[], block=False):
    def decorator(func):
        func._name_space = name_space
        func._function_type = function_type
        func._post_type = post_type
        func._priority = priority
        func._user_state = user_state
        func._role = role
        func._block = block
        return func
    return decorator

# 全局变量，在发送给LLM推理时用到，如果
bot_nick_name = ""
user_nick_name = ""
user_state = ""
name_space = ""
source_id = ""

# 子函数示例1
@lj(name_space="礼金", function_type="serial", post_type="message", user_state=["聊天"], priority=0, role=["21122263971@chatroom","cbf_415135222"])
# 从用户请求中提取姓名
def lj_get_name(data={}): # 第一个函数的参数必须为字典类型
    
    global bot_nick_name
    global user_nick_name
    global user_state
    global name_space
    global source_id
    
    message = data["message"]
    bot_nick_name = data["bot_nick_name"]
    user_nick_name = data["user_nick_name"]
    source_id = data["source_id"]
    user_state = data["user_state"]
    name_space = data["name_space"]
    
    prompt = '''
    根据用户输入的内容中提取出包含的姓名。\n 
    - 你只能输出你得到的姓名，不要输出其他内容和推理过程，也不要@用户。
    - 如果你确定无法提取姓名，请输出"无法提取姓名"。\n
    - 用户输入：%s\n
    - 不要输出其他内容和推理过程，也不要添加标点符号，直接用中文输出：''' % data
    response_message = asyncio.run(chat_generic_langchain(bot_nick_name, user_nick_name, source_id, prompt, user_state, name_space))
    return response_message


@lj(name_space="礼金", function_type="serial", post_type="message", user_state=["聊天"], priority=1, role=["21122263971@chatroom","cbf_415135222"])
# 使用提取出的姓名构造SQL语句
def lj_make_sql(data): # 第一个函数的参数必须为字典类型
    import pymssql
    import json
    # # global msg
    if "无法提取姓名" not in data:
        try:
        # 构造SQL查询语句
            sql = '''
                SELECT a.[序号]
                    ,a.[姓名]
                    ,a.[金额]
                    ,a.[地址]
                    ,b.[备注]
                FROM [dbo].[结婚礼簿_明细] a left join [结婚礼簿_主表] b 
                on a.ExcelServerRCID = b.ExcelServerRCID WHERE a.[姓名] = %s
            ''' 
            # 连接数据库
            conn = pymssql.connect(server='192.168.66.6', database='my', user='sa', password='Shift962512', charset='utf8')
            # 创建游标
            cursor = conn.cursor()
            # 执行SQL语句
            cursor.execute(sql, (data,))
            # 获取查询结果
            result = cursor.fetchall()
            if result:
                rows = []
                for row in result:
                    msg = {"送礼人的序号":str(row[0]), "送礼人的姓名":row[1], "送礼人的金额":str(row[2]), "送礼人的地址":row[3], "备注":row[4]}
                    msg_json = json.dumps(msg, ensure_ascii=False)
                    print(f"结果JSON:{msg_json}")
                    rows.append(msg_json)
                return rows
            else:
                return ""
            # 关闭连接
            conn.close()
        except Exception as e:
            print(f"查询出错：{e}，姓名：{data}")
            return f"查询出错：{e}，姓名：{data}"
    else:
        return ""








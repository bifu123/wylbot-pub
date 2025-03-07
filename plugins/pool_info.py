'''
贫困户信息
'''
import aiohttp
import asyncio
from models_load import *


################# 主函数 ##################
def pool_info(name_space, function_type, post_type, user_state:list, priority, role=[], block=False):
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





################# 子函数 ##################
# 插件函数示例1
@pool_info(name_space="贫困户信息", function_type="parallel", post_type="message", user_state=["聊天"], priority=1)
def fun_get_info(data):
    # 查询数据库
    import os
    try:
        import pymssql
        print('pymssql导入成功')
    except:
        os.system('pip install pymssql')
        import pymssql
        print('pymssql导入成功')
 
    # 连接数据库
    server = '192.168.66.6'
    user = 'sa'
    password = 'Shift962512'
    database = 'my'
    chatset = 'utf8'

    conn = pymssql.connect(server, user, password, database, chatset)

    # 执行查询
    query = """
    SELECT [姓名]
        ,[性别]
        ,[证件号码]
        ,[与户主关系]
        ,[民族]
        ,[政治面貌]
        ,[文化程度]
        ,[在校生情况]
        ,[健康状况]
        ,[劳动技能]
        ,[务工状况]
        ,[务工时间]
        ,[是否现役军人]
        ,[是否参加大病保睑]
        ,[是否享受低保]
        ,[务工企业名称]
        ,[企业地址]
        ,[联系电话]
        ,[户主]
        ,[记事]
    FROM [dbo].[贫困户信息采集_明细] for json auto
    """

        # 创建游标
    cursor = conn.cursor()

    # 执行查询
    cursor.execute(query)

    # 获取查询结果
    json_result = cursor.fetchall()

    # 关闭连接
    conn.close()

    # 打印 JSON 结果
    print(json_result)


    return json_result

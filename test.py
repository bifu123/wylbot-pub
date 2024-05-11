from sqlite_helper import *
print(get_models_table())


print(f'您说今天的天气是：{p1_value}')



try:
    update_models_table('{p1_value}', '{p2_value}', '{p3_value}', {p4_value})
    response = '模型己更新！😊'
except Exception as e:
    print(e)
    response = f'模型己更新失败：{e}😊'
asyncio.run(answer_action(chat_type, user_id, group_id, at, '模型己更新！😊'))
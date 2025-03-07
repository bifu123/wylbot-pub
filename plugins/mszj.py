'''
用selenium爬取民生资金网站
'''
import aiohttp
import asyncio
from models_load import *



################# 主函数 ##################
def get_mszj(name_space, function_type, post_type, user_state, priority, role=[], block=False):
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


def get_mszj_cx(name_space, function_type, post_type, user_state:list, priority, role=[], block=False):
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


################ 子函数 ##################
# 插件函数示例1
@get_mszj(name_space="民生资金", function_type="parallel", post_type="message", user_state=["聊天"], priority=3)
def get_mszj(data):
    import time
    import os
    try:
        from selenium import webdriver
        from webdriver_manager.firefox import GeckoDriverManager
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.action_chains import ActionChains
        from selenium.webdriver.common.keys import Keys
    except:
        os.system('pip install -U selenium webdriver_manager')
        from selenium import webdriver
        from webdriver_manager.firefox import GeckoDriverManager
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.action_chains import ActionChains
        from selenium.webdriver.common.keys import Keys
    try:
        import pandas as pd
    except:
        os.system('pip install -U pandas')
        import pandas as pd

    def scrape_data(searchPeopleId, searchPeopleName):
        driver = None  # Initialize driver outside try block
        try:
            try:
                driver = webdriver.Firefox()
            except:
                driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())

            driver.get("http://www.gzyytech10.com:8892/ccmimpweb/weixinFundPeopleDetail?wxSysId=&wxUserId=cbf_415135222")

            name_input = driver.find_element(By.ID, "searchPeopleName")
            name_input.send_keys(searchPeopleName)

            id_input = driver.find_element(By.ID, "searchPeopleId")
            id_input.send_keys(searchPeopleId)

            confirm_button = driver.find_element(By.XPATH, "/html/body/section[1]/p/img")
            confirm_button.click()

            driver.maximize_window()

            time.sleep(2)

            def find_element(xpath):
                try:
                    load_more_button = driver.find_elements(By.XPATH, xpath)
                    return True, load_more_button
                except:
                    return False, None

            find_element_r = find_element("//p[contains(text(), '加载更多')]")

            if find_element_r[0]:
                print("加载更多按钮已找到")
                time.sleep(1)

            while True:
                if not find_element_r[1]:
                    break
                driver.execute_script("arguments[0].scrollIntoView();", find_element_r[1][0])
                driver.execute_script("arguments[0].click();", find_element_r[1][0])

                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                is_scroll_end = driver.execute_script(
                    "return (window.innerHeight + window.scrollY) >= document.body.offsetHeight;"
                )
                if is_scroll_end:
                    break
                if not find_element_r[0]:
                    break

            data = driver.find_element(By.CLASS_NAME, "findercar-brandcore").text.replace("","")
            if data == "":
                data = f"{searchPeopleName}没有得到补助\n\n"
            else:
                data = f"{searchPeopleName}得到的补助：\n{data}\n\n"
            print(data)
            return True, data
         
        except Exception as e:
            print("An error occurred:", e)
            return False, f"{searchPeopleName}没有得到补助\n\n"

        finally:
            if driver is not None:
                driver.quit()

    search_params = [
        {"searchPeopleId": "52263519610515341143", "searchPeopleName": "黄绍中"},
        {"searchPeopleId": "522635197612193412", "searchPeopleName": "文友发"},
        {"searchPeopleId": "522635195703043410", "searchPeopleName": "黄绍林"},
        {"searchPeopleId": "522635196307253437", "searchPeopleName": "艾林付"},
        {"searchPeopleId": "522635197007193414", "searchPeopleName": "罗康武"},
        {"searchPeopleId": "52263519580608344X", "searchPeopleName": "杨盛芝"},
        {"searchPeopleId": "522635197008033412", "searchPeopleName": "龙顺国"},
        {"searchPeopleId": "52263519910904323X", "searchPeopleName": "罗仕龙"}
    ]

    result_try = ""
    for params in search_params:
        success, message = scrape_data(params["searchPeopleId"], params["searchPeopleName"])
        result_try += message

    # 将结果写入到文件中
    with open("result.txt", "w", encoding="utf-8") as file:
        file.write(result_try.replace("None", ""))
    
    return "执行完成，写入：result.txt"


@get_mszj_cx(name_space="民生资金查询", function_type="parallel", post_type="message", user_state=["聊天"], priority=1, block=True)
def cx_mszj(data):
    # 打开文件
    with open('result.txt', 'r', encoding='utf-8') as file:
        # 读取文件内容并解码
        read_content = file.read()
    read_content = read_content.replace("None", "")
    return f"下面是所有贫困户得到的补助：\n\n{read_content}"



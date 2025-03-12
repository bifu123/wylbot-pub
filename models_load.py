import os
import sys
import requests
import json
# import re

from config import *
from sqlite_helper import *
# from mssql_helper import *
import do_history


# ollama模型
# from langchain_community.embeddings import OllamaEmbeddings # 量化文档
# from langchain_community.llms import Ollama #模型
from langchain_ollama import OllamaEmbeddings, ChatOllama

# cohere重排模型
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere

# gemini模型
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# 提示词模板
from langchain_core.prompts import ChatPromptTemplate

# 通义千问模型
from langchain_community.llms import Tongyi
import dashscope
# deepseek 模型
from langchain_deepseek import ChatDeepSeek # pip install -qU langchain-deepseek
from langchain_openai import ChatOpenAI

# 语义检索
from langchain.schema.runnable import RunnableMap
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# chatGLM3-6B 模型
from langchain_community.llms.chatglm3 import ChatGLM3


# kimi 模型
from langchain_community.llms.moonshot import Moonshot

# groq api 模型
from langchain_groq import ChatGroq

# deepseek 模型
from langchain_deepseek import ChatDeepSeek # pip install -qU langchain-deepseek

# chatGLM3-6B 模型
from langchain_community.llms.chatglm3 import ChatGLM3

# 异步函数
import asyncio


# 读取环境变量
from dotenv import load_dotenv #pip install python-dotenv

# 读取 .env 文件
load_dotenv()

############################# API KEY #################################
# 将各个在线模型 API key 加入环境变量
# os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
# os.environ['DASHSCOPE_API_KEY'] = DASHSCOPE_API_KEY
# os.environ["MOONSHOT_API_KEY"] = MOONSHOT_API_KEY
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY
# os.environ["COHERE_API_KEY"] = COHERE_API_KEY
############################# 量化模型 #################################
# 本地量化模型
embedding_ollama = OllamaEmbeddings(
    base_url = embedding_ollama_conf["base_url"], 
    model = embedding_ollama_conf["model"]
) 

############################# 语言模型 #################################
# 本地语言模型
# llm_ollama = Ollama(
#     base_url = llm_ollama_conf["base_url"], 
#     model = llm_ollama_conf["model"]
# )

llm_ollama = ChatOllama(
    base_url = llm_ollama_conf["base_url"], 
    model = llm_ollama_conf["model"],
    temperature = llm_ollama_conf["temperature"],
    num_predict = llm_ollama_conf["num_predict"]
)

# 在线语言模型 gemini
llm_gemini = ChatGoogleGenerativeAI(
    model = llm_gemini_conf["model"],
    temperature = llm_gemini_conf["temperature"]
) 
# 线上google量化模型
embedding_google = GoogleGenerativeAIEmbeddings(
    model = embedding_google_conf["model"]
) 

# # 在线语言模型 通义千问
# llm_tongyi = Tongyi(
#     model_name = llm_tongyi_conf["model_name"],
#     temperature = llm_tongyi_conf["temperature"],
#     streaming = llm_tongyi_conf["streaming"]
#     #enable_search = True
# ) 

# 更新
llm_tongyi = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model=llm_tongyi_conf["model_name"],  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    # other params...
)



# 在线语言模型 kimi
llm_kimi = Moonshot(
    model_name = llm_kimi_conf["model_name"],
    temperature = llm_kimi_conf["temperature"]
) 
# 在线语言模型 groq
llm_groq = ChatGroq(
    model_name = llm_groq_conf["model_name"],
    temperature = llm_groq_conf["temperature"]
)
# 在线语言模型 deepseek
llm_deepseek = ChatDeepSeek(
    model = llm_deepseek_conf["model"],
    temperature = llm_deepseek_conf["temperature"],
    max_tokens = llm_deepseek_conf["max_tokens"],
    timeout = llm_deepseek_conf["timeout"],
    max_retries = llm_deepseek_conf["max_retries"]
) 

# 本地语言模型 ChatGLM3
llm_chatGLM = ChatGLM3(
    endpoint_url = llm_chatGLM_conf["endpoint_url"],
    max_tokens = llm_chatGLM_conf["max_tokens"],
    top_p = llm_chatGLM_conf["top_p"]
) 


############################# 模型选择 #################################
# 读取数据库
def get_models_on_request():
    models = get_models_table()
    must_use_llm_rag = models["must_use_llm_rag"]
    # 选择量化模型
    if models["embedding"] == "ollama":
        embedding = embedding_ollama
    else:
        embedding = embedding_google

    # 选择聊天语言模型
    if models["llm"] == "ollama":
        llm = llm_ollama
    elif models["llm"] == "gemini": 
        llm = llm_gemini
    elif models["llm"] == "tongyi": 
        llm = llm_tongyi
    elif models["llm"] == "kimi": 
        llm = llm_kimi
    elif models["llm"] == "groq": 
        llm = llm_groq
    else:
        llm = llm_chatGLM

    # 选择知识库语言模型
    if models["llm_rag"] == "ollama":
        llm_rag = llm_ollama
    elif models["llm_rag"] == "gemini": 
        llm_rag = llm_gemini
    elif models["llm_rag"] == "tongyi": 
        llm_rag = llm_tongyi
    elif models["llm_rag"] == "kimi": 
        llm_rag = llm_kimi
    elif models["llm_rag"] == "groq": 
        llm_rag = llm_groq
    else:
        llm_rag = llm_chatGLM
    # llm_rag.temperature = 0.0
    return embedding, llm, llm_rag, must_use_llm_rag
    
############################# 模型方法 #################################

# # 去除重复的昵称
# def remove_repeated_nicknames(content, nick_name):
#     # 使用格式化字符串将变量嵌入到正则表达式模式中
#     pattern = rf'({re.escape(nick_name)})(?:\1)+'
#     # 使用正则表达式替换，保留一个 user_nick_name
#     result = re.sub(pattern, r'\1', content)
#     return result



# 向量检索聊天（执行向量链）
async def run_chain(bot_nick_name, user_nick_name, retriever, source_id, query, user_state="聊天", name_space="test", template_string=""):
    # 是否使用重排模型
    if must_rerank_embedding == 1:
        # llm = Cohere(temperature=0)
        # 对检索结果进行重排
        compressor = CohereRerank() 
        retriever = ContextualCompressionRetriever(
            base_compressor = compressor, 
            base_retriever = retriever # 未重排前的检索结果
        )
        
    embedding, llm, llm_rag, must_use_llm_rag = get_models_on_request()

    if query !="" and query is not None:
        print("=" * 50)
        print("当前使用的知识库LLM：", llm_rag)
        if template_string == "":
            template_string = f"""你的名字叫{bot_nick_name}，请根据文档内容用简体中文完整地回答问题，同时也善于从大家复杂的对话中分析每个人表达的意图，以下是我必须提醒你注意的：
            
## 注意分清楚是谁对谁说话
- 当消息包含有类似于“@群成员”, 并且不是“@{bot_nick_name}”，就表示用户在和另外一个用户说话。
- 如果消息中包含有“@{bot_nick_name}”这样的字符，表明用户是对你说，请你回答时在内容前加上“@{user_nick_name} ”。
- 如果你的回答是针对某个用户的，请在回答的内容前面加上“@用户 ”，这样用户会看到你的回答是针对他。
- 请用简体中文输出。Please output in Chinese. """ # 默认模板
        
        template_cn = """%s
        {context}
        {question}
        """ % template_string
        
        

        # 处理聊天记录
        chat_history = do_history.get_chat_history(source_id, user_state, name_space, bot_nick_name, query)
            
            
        # 由模板生成prompt
        prompt = ChatPromptTemplate.from_template(template_cn) 

        # 创建chain
        chain = RunnableMap({
            "context": lambda x: retriever.invoke(x),
            "question": RunnablePassthrough(),
            "chat_history": lambda x: chat_history  # 使用历史记录的步骤
        }) | prompt | llm_rag | StrOutputParser()
        
        # 执行问答
        request = f'{{"user":"{user_nick_name}", "content":"{query}"}}'
        try:
            response_message = chain.invoke(request)
        except Exception as e:
            response_message = f"LLM响应错误: {e}"
            print(f"LLM响应错误: {e}")
            
        # 返回结果
        return response_message + "😊"



# from langchain.chains import RetrievalQA
# async def run_chain(bot_nick_name, user_nick_name, retriever, source_id, query, user_state="聊天", name_space="test", template_string=""):
#     if must_rerank_embedding == 1:
#         compressor = CohereRerank()
#         retriever = ContextualCompressionRetriever(
#             base_compressor=compressor,
#             base_retriever=retriever
#         )
    
#     embedding, llm, llm_rag, must_use_llm_rag = get_models_on_request()
    
#     if query:
#         print("=" * 50)
#         print("当前使用的知识库LLM：", llm_rag)
        
#         if not template_string:
#             template_string = f"""
#             你的名字叫{bot_nick_name}，请根据文档内容用简体中文完整地回答问题，同时也善于从大家复杂的对话中分析每个人表达的意图，以下是我必须提醒你注意的：
            
#             ## 注意分清楚是谁对谁说话
#             - 当消息包含有类似于“@群成员”, 并且不是“@{bot_nick_name}”，就表示用户在和另外一个用户说话。
#             - 如果消息中包含有“@{bot_nick_name}”这样的字符，表明用户是对你说，请你回答时在内容前加上“@{user_nick_name} ”。
#             - 如果你的回答是针对某个用户的，请在回答的内容前面加上“@用户 ”，这样用户会看到你的回答是针对他。
#             - 请用简体中文输出。Please output in Chinese.
#             """
        
#         template_cn = f"""
#         {template_string}
#         {{chat_history}}
#         {{context}}
#         {{question}}
#         """
        
#         chat_history = do_history.get_chat_history(source_id, user_state, name_space, bot_nick_name, query)
        
#         chain = RetrievalQA.from_chain_type(llm_rag, retriever=retriever)
        
#         request = f'{{"user":"{user_nick_name}", "content":"{query}", "chat_history": "{chat_history}"}}'
#         try:
#             response_message_dict = chain.invoke({"query": "请用中文回答我：" + request, "chat_history": chat_history})
#             response_message = response_message_dict.get("result", "").strip()
#         except Exception as e:
#             response_message = f"LLM响应错误: {e}"
#             print(f"LLM响应错误: {e}")
        
#         return response_message + "😊"


# 通用聊天
async def chat_generic_langchain(bot_nick_name, user_nick_name, source_id, query, user_state="聊天", name_space="test", template_string=""):
    embedding, llm, llm_rag, must_use_llm_rag = get_models_on_request()
    if query !="" and query is not None:

        # 处理聊天记录
        chat_history = do_history.get_chat_history(source_id, user_state, name_space, bot_nick_name, query)
            
        if template_string == "":
            template_string = f"""你是一个热心的人，你的名字叫{bot_nick_name}，尽力为人们解答问题.同时也善于从大家复杂的对话中分析每个人表达的意图，以下是我必须提醒你注意的：
            
## 注意分清楚是谁对谁说话
- 当消息包含有类似于“@群成员”, 并且不是“@{bot_nick_name}”，就表示用户在和另外一个用户说话。
- 如果消息中包含有“@{bot_nick_name}”这样的字符，表明用户是对你说，请你回答时在内容前加上“@{user_nick_name} ”。
- 如果你的回答是针对某个用户的，请在回答的内容前面加上“@用户 ”，这样用户会看到你的回答是针对他。
- 请用简体中文输出。Please output in Chinese. """ # 默认模板
            
        # 由模板生成 prompt
        prompt = ChatPromptTemplate.from_template("""
            %s
            {chat_history}
            {question}
        """ % template_string)
        print("=" * 50)
        
        # 创建链，将历史记录传递给链
        if user_state != "聊天" and must_use_llm_rag == 1:
            chain = {
                "question": RunnablePassthrough(), 
                "chat_history": lambda x: chat_history,
            } | prompt | llm_rag | StrOutputParser()  
            print("当前使用的聊天LLM：", llm_rag)
        else:
            chain = {
                "question": RunnablePassthrough(), 
                "chat_history": lambda x: chat_history,
            } | prompt | llm | StrOutputParser()  
            print("当前使用的聊天LLM：", llm)

        # 执行问答
        request = f'{{"user":"{user_nick_name}", "content":"{query}"}}'
        try:
            response_message = chain.invoke(request)
            # 处理聊天记录 
            # await do_chat_history(chat_history, source_id, user_nick_name, query, user_state, name_space)
            # do_chat_history(chat_history, source_id, bot_nick_name, response_message, user_state, name_space)
        except Exception as e:
            response_message = f"LLM响应错误: {e}"
            print(f"LLM响应错误: {e}")
            
        return response_message + "😊"




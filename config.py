# ********************** 通用配置 *********************** 
# 管理员微信ID
admin_wxid = "cbf_415135222"
# 微信接收文件保存路径
file_receive_path = rf"D:\Documents\WeChat Files"  # PC微信：设置-文件管理 中查看
# 允许的聊天回复
chat_type_allow = [
    "private",    # 私聊回复
    "group_at",   # 只有群中 @ 才回复
    "group" ,     # 回复群中所有聊天，这样机器人什么人说话它都会接话
    # "Unkown" ,  # 其它类型聊天也回复
    ]
# 允许上传的文件类型
allowed_extensions = [
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".csv",
    ".pdf",
    ".md",
    ".html",
    ".txt",
    ".htm",
    ".py"
]
# wxbot Websocket 监听地址
ws_url = "ws://127.0.0.1:8080/ws/generalMsg"  
# wxbot http API接口地址
http_url = "http://127.0.0.1:8080" 
# 源文档路径 
data_path = "./data"
# 分割文档的块大小
chunk_size = 800
# 分割文档时连续部分的重叠区大小
chunk_overlap = 128
# 保存聊天记录的大小
chat_history_size_set = 8192 # 记录越大，每次发给大模型分析的数据越多，上下文越全面。但是会增加响应的时间，而且随着话题的多样复杂，会降低大模型分析的精准度
# 量化后数据保存路径
db_path = "./chroma_db"




# ******************** 线上模型api key ******************** 
DASHSCOPE_API_KEY  = "sk-7d48078fa897417c9cfa5cfa70d95f9a" # 通义千问 api key
MOONSHOT_API_KEY = "sk-iIiYG1GyHKH66c0Rag0PTH3UQzF20wZT14Pr0nOD6AX35FJk" # moonshot ai kimi api key 在这里申请: https://platform.moonshot.cn/console/api-keys
GROQ_API_KEY = "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" # GROQ API KEY 在这里申请: https://console.groq.com/keys




# ******************** 模型配置 ****************************
# 本地量化模型
embedding_ollama_conf = { 
    "base_url": "http://192.168.66.24:11434", 
    "model": "mofanke/dmeta-embedding-zh" # nomic-embed-text | mofanke/dmeta-embedding-zh
}
# goole量化模型
embedding_google_conf = { 
    "model": "models/embedding-001"
} 
# 本地语言模型 
llm_ollama_conf = { 
    "base_url": "http://192.168.66.26:11434",  
    "model": "llama3:latest" # qwen:7b | llama3:latest | cwchang/llama3-taide-lx-8b-chat-alpha1:latest
}
# 线上google gemini语言模型
llm_gemini_conf = { 
    "model": "gemini-pro",
    "temperature": 0.7
} 
# 线上 通义千问 语言模型
llm_tongyi_conf = { 
    "model_name": "qwen-plus", # qwen-max-longcontext | qwen-max |qwen-plus |roger/minicpm:latest
    "temperature": 0.7,
    "streaming": False
} 
# 线上 moonshot ai kimi 语言模型
llm_kimi_conf = { 
    "model_name": "moonshot-v1-128k",
    "temperature": 0.3
} 
# 线上 groq api 语言模型
llm_groq_conf = { 
    "model_name": "mixtral-8x7b-32768", # llama3-70b-8192 | mixtral-8x7b-32768
    "temperature": 0.3
} 




# ******************** 模型选择 ****************************
model_choice = {
    # 本地向量模型
    "embedding":"ollama", # embedding: ollama | google
    # 本地知识库模型
    "llm_rag": "groq", # llm: ollama | tongyi | kimi | groq 
    # 聊天模型
    "llm": "groq", # llm: ollama | tongyi | kimi | groq
}



# 是否强制非聊天状态时，一律使用本地知识库模型
must_use_llm_rag = False # True | False # 当文档较大时，建议设置为True。因为在线 API 模型在处理本地文档时Tocken大小有限制，二是考虑到费用问题，当然你可结合自身的需求和实际情况而定






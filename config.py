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
GOOGLE_API_KEY = "your GOOGLE_API_KEY" #gemini api key的申请地址：https://makersuite.google.com/app/prompts/new_freeform ，条件：拥有google帐号
DASHSCOPE_API_KEY  = "your DASHSCOPE_API_KEY" # 通义千问 api key
MOONSHOT_API_KEY = "your MOONSHOT_API_KEY" # moonshot ai kimi api key 在这里申请: https://platform.moonshot.cn/console/api-keys
GROQ_API_KEY = "your GROQ_API_KEY" # GROQ API KEY 在这里申请: https://console.groq.com/keys
# cohere 重排模型 API KEY
COHERE_API_KEY = "your COHERE_API_KEY" # 申请地址：https://dashboard.cohere.com/api-keys
QIANFAN_AK = "your QIANFAN_AK" # 千帆API




# ******************** 模型配置 ****************************
# 本地量化模型
embedding_ollama_conf = { 
    "base_url": "http://127.0.0.1:11434", 
    "model": "nomic-embed-text" # nomic-embed-text | mofanke/dmeta-embedding-zh
}
# goole量化模型
embedding_google_conf = { 
    "model": "models/embedding-001"
} 
## 线上google gemini语言模型
llm_gemini_conf = {
    "model": "gemini-pro",
    "temperature": 0.7
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
# 线上 百度 语言模型
llm_qianfan_conf = { 
    "model_name": "ERNIE-Lite-8K-0922", 
    "treaming": False
} 
# 本地 chatGLM3-6b
llm_chatGLM_conf = {
    "endpoint_url": "http://192.168.66.26:8000/v1/chat/completions",
    "max_tokens": 80000,
    "top_p": 0.9
}



# ******************** 模型选择 ****************************
model_choice = {
    # 本地向量模型
    "embedding":"ollama", # embedding: ollama | google
    # 本地知识库模型
    "llm_rag": "tongyi", # llm: ollama | gemini | tongyi | chatglm | kimi | groq | qianfa
    # 聊天模型
    "llm": "tongyi", # llm: ollama | gemini | tongyi | chatglm | kimi | groq | qianfa
}


# 是否强制非聊天状态时，一律使用本地知识库模型
must_use_llm_rag = 0 # 1 | 0 # 当文档较大时，建议设置为1。因为在线 API 模型在处理本地文档时Tocken大小有限制，二是考虑到费用问题，当然你可结合自身的需求和实际情况而定


# 是否使用重排向量模型
must_rerank_embedding = 0 # 0 | 1 







# wylbot 元龙机器人微信版
## 简介
wylot是一个以微信聊天界面作为语言模型与用户交互端的RAG应用。延续了QQ版YLBot的功能，并优化了很多地方。
## 宗旨和理念
- 让Ai参与网络聊天社交，让人与AI、AI与AI可以互动
- 作为真人网上社交的助手，具有辅助作用
- 与其它语言模型前端不同，它有私聊、群聊、分组等功能
## 免责申明
本应用旨在研究语言模型应用，它的核习来自wxbot。仅供学习参考，请勿用于违法犯罪活动，一切资源均来自网络，不提供任何破解服务。如需生产使用，请申请官方微信机器人。本开源项目诸多细节需要一定知识，如有疑问，请联系作者：QQ 415135222
## 怎样使用
### 登陆微信
详见：https://github.com/jwping/wxbot
### 创建环境变量
当前目录下新建文件.env，内容如下：
```
# deepseek api key 
DEEPSEEK_API_KEY = "你的 deepseek api key" # 申请地址：https://platform.deepseek.com/
# gemini api key 
GOOGLE_API_KEY = "你的 gemini api key" # gemini api key 的申请地址：https://makersuite.google.com/app/prompts/new_freeform ，条件：拥有google帐号
# 通义千问 api key
DASHSCOPE_API_KEY  = "你的 DASHSCOPE_API_KEY"
# moonshot ai kimi api key
MOONSHOT_API_KEY = "你的 moonshot ai kimi api key" # 在这里申请: https://platform.moonshot.cn/console/api-keys
# groq
GROQ_API_KEY = "你的 groq api key" # 在这里申请: https://console.groq.com/keys
# cohere 重排模型 API KEY
COHERE_API_KEY = "你的 cohere api key" # 申请地址：https://dashboard.cohere.com/api-keys
```
以上的KEY的并不全部都必须，使用哪个模型就填写哪个KEY

此外，还需要安装chrome和firefox
### 安装环境
- windows可以使用conda环境安装部署，linux不建议用
```bash
conda create -n wylbot python=3.11
git clone https://github.com/bifu123/wylbot
cd wylbot
conda activate wylbot
pip install requirements.txt
```
### 修改配置文件config.py
请根据文件中提示，结合你的实际情况修改，其中涉及了对接ollama的部分，如果不会，先补上这部分知识，可以参看我在b站上的视频，也可以加QQ群：222302526 ，也可以使用在线模型的API（可能会产生费用）

### 启动程序
```bash
python listen_ws.py
```
程序启动后，它会用socket监听wxbot的事件和消息，并根据这些事件和消息运行程序逻辑，作出响应。

### 操控命令
| 命令名称   | 作用                                       | 备注                                                 |
|------------|--------------------------------------------|------------------------------------------------------|
| /我的文档   | 列出当前用户保存在服务器上的所有私人文档或所在群中公共文档 | 自动判断用户所处环境加载不同文档路径               |
| /删除文档   | 删除某个文档                                 | 用法：/删除文档|要删除的文档完整路径                     |
| /量化文档   | 将当前用户私人文档或用户所在群的公共文档量化 | 量化时，已有向量库不受影响，量化好后无缝迁移到新向量库 |
| /上传文档   | 上传文档到服务器                             | 允许常规的文档，先切换到“文档问答”或“知识库问答”状态，然后直接向机器人或机器人所在群发送文档                                       |
| /文档问答   | 就当前用户的文档或其所在群的文档进行问答     | 答案仅限于文档内容中，将每次加载一个或多个文档以及问题发送到LLM，所以注意文档大小，否则TOKEN耗费巨大。      |
| /插件问答   | 加载执行插件返回的结果为上下文供LLM推理     | 插件可以自由扩展                                 |
| /聊天      | 使用大模型进行对话，不加载文档知识库        | 模型可以在config.py中设置更换                         |
| /我的状态   | 显示当前用户处理文档问答还是聊天状态         | 可以根据命令提示切换状态                             |
| /开启群消息 | 所有群成员的发言，机器人都会回答            | 可以用于活跃群或具有足够针对性的知识，但也会给群带来骚扰，请酌情使用 |
| /关闭群消息 | 关闭之后，在群中除非@它，否则机器人不会作任何反应 | 一般群的默认值即为群消息关闭                         |
| /清空记录 | 清空用户私有或者群中公共的问答历史记录 | 在聊天历史过多或话题繁杂而影响了机器人分析回复质量时使用                         |
| /我的命名空间 | 显示当前用户所处的插件命名空间 | 插件命名空间用于插件功能的隔离，精准定位到某个关键词下一个或多个插件                        |
| /帮助 | 提供帮助 | 显示github上的项目readme.md链接                        |

## 插件
插件是本系统最为强大的功能，可以自由扩展。理论上只要python能实现的功能都可以通过插件实现。它基于langchain的串行、并行链思想，但是为了避免大语言模型LLM推理和返回结果的不可预知的缺点，使用函数对与链类似的功能做到了精确控制，这是我极为推荐的地方。
[如何使用插件](plugin.md)

## 重要更新
- 2024/5/22
取消插件只能限定在“插件问答”状态中执行，现在任意状态都可以加载相应插件。





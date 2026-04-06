#服务器端：接口 -> langchain -> openai/ollama
#客户端：机器人、website
#接口：http，https，websocket

#服务器：
1.接口访问，python选型 -> fastapi
2./chat接口,post请求
3./add_urls从url中更新知识库
4./add_pdfs从PDF中学习
5./add_texts从text中学习

#人性化
1.用户输入 -> AI判断一下当前问题的情绪，分类
2.工具调用 ->agent判断使用哪个工具 -> 带着相关参数请求工具 -> 得到观察结果
3.内部可学习记忆
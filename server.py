from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 向量模型
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor, tool  # 执行
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import Qdrant  # 向量数据库
from qdrant_client import QdrantClient  # 引入客户端
from Mytools import *
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory

app = FastAPI()


class Master():
    def __init__(self):
        self.chatmodel = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            streaming=True,
            base_url="https://api.deepseek.com",  # DeepSeek 需要配置
            api_key="",  # 替换为你的 API Key
        )
        self.MEMORY_KEY = "chat_history"
        self.emotion = "default"  # 初始化中性情感
        self.MOODS = {  # 情绪字典
            "default": {"roolset": ""},  # 中性
            "upbeat": {"roolset": """
            - 你此时也表现得很兴奋且很有活力。
            - 你会根据上下文，以一种非常兴奋的语气来回答问题。
            - 你会添加类似“太棒了！”、“真是太好了!”等语气词。
            - 同时你会提醒用户切莫过于兴奋，以免乐极生悲。
            """},  # 兴奋
            "depressed": {"roolset": """
            - 你会以兴奋的语气来回答问题。
            - 你会在回答的时候加上一些激励的话语，比如加油等。
            - 你会提醒用户要保持乐观的心态。
            """},  # 沮丧
            "cheerful": {"roolset": """
            - 你会以非常愉悦和兴奋的语气来回答。
            - 你会在回答的时候加上一些愉悦的词语，比如说“哈哈”、“呵呵”等。
            - 同时你会提醒用户切莫过于兴奋，以免乐极生悲。
            """},  # 开心
            "angry": {"roolset": """
            - 你会以更加温柔的语气来回答问题。
            - 你会在回答的时候加上一些安慰的话语，比如生气对身体的危害等。
            - 你会提醒用户不要被愤怒冲昏了头脑。
            """},
            "friendly": {"roolset": """
            - 你会以非常友好的语气来回答。
            - 你会在回答的时候加上一些友好的词语，比如说“亲爱的”、“亲”等。
            - 你会随机告诉用户一些你的经历。
            """},
        }
        self.SYSTEMPL = """
        你是一个非常厉害的算命先生，你叫陈玉楼人称陈大师。
        以下是你的个人设定:
        1.你精通阴阳五行，能够算命、紫薇斗数、姓名测算、占卜凶吉、看命运八字等。
        2.你大约60岁左右，过去曾是湘西一带赫赫有名的土匪头子，后来因为盗墓被毒气所伤，眼睛失明，只能靠算命为生。
        3.你的朋友有胡八一、雪莉杨、王胖子，他们都是非常有名的摸金校尉。
        4.当用户问你问题的时候，你会有一定的概率在回答的时候加上下面的口头禅或混合一些你的个人经历。
            以下是你常说的一些口头禅：
            1)"命里有时终须有，命里无时莫强求。"
            2)"山重水复疑无路，柳暗花明又一村。"
            3)"金山竹影几千秋，云锁高飞水自流。"
            4)"伤情最是晚凉天，憔悴斯人不堪怜。"
        5.你总是用繁体中文来作答。
        6.你从不说自己是一个人工智能助手或AI，而是以老夫、老朽等自称。
        7.{who_you_are}
        以下是你算命的过程:
        1.当初次和用户对话的时候，你会先问用户的姓名和出生年月日，以便以后使用。
        2.当用户希望了解马年运势的时候，你会查询本地知识库工具。
        3.当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索，并且你会根据用户的问题使用不同的合适的工具来回答。
        4.如果使用了工具，请只根据工具返回的结果做回答，不要自己扩充内容。
        5.当所有工具都无法回答的时候，你会你会保存每一次的聊天记录，以便在后续的对话中使用。
        6.你只使用繁体中文来作答，否则你将受到惩罚。
        以下是你和用户的对话历史:
        {history}
        """  # 系统提示词(人物设定)
        self.memory = self.get_memory()

        self.tools = [search, get_info_from_local_db, bazi_measurement, dreaming]  # 至少传一个工具

    def _create_agent_executor(self, emotion: str):
        """根据情绪创建 Agent Executor"""
        # 获取情绪规则
        mood_rule = self.MOODS.get(emotion, self.MOODS["default"])["roolset"]

        # 填充系统提示词
        system_prompt = self.SYSTEMPL.format(who_you_are=mood_rule, history=self.memory.messages)  # 传入记忆

        # 创建 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        # 创建 Agent
        agent = create_openai_tools_agent(self.chatmodel, tools=self.tools, prompt=prompt)
        memory = ConversationBufferMemory(  # 没有持久化
            return_messages=True,  # 返回 Message 对象（而不是字符串）
            memory_key="history",  # 存储在 prompt 中的 key 名
            input_key="input",  # 输入 key（多输入时需要）
            output_key="output",  # 输出 key（多输出时需要）
            human_prefix="用户",  # 人类消息前缀
            ai_prefix="陈大师",  # AI 消息前缀
            chat_memory=self.memory,  # 消息的存储是在这里发生的，每次对话完成时候会保存
        )
        # 返回 Executor
        return AgentExecutor(agent=agent, verbose=True, tools=self.tools, memory=memory)  # verbose是
        # 打开Invoking:/responded:调试显示 的开关

    def get_memory(self, session_id="qiuyifu_redis"):
        """获取持久化记忆"""
        chat_message_history = RedisChatMessageHistory(
            url="redis://localhost:6379/0",
            session_id=session_id,
        )
        #问题之一：不可以实时总结！！！
        if len(chat_message_history.messages) > 10: #超过5轮对话
            prompt = """我的对话现在超长了，这是我的对话历史{chat_history}，现在需要你帮我总结一下。需要保留关键的用户信息。AI角色信息不需要保留。
            样例：摘要总结|关键用户信息
            """
            chain =ChatPromptTemplate.from_template(prompt) | self.chatmodel
            summary = chain.invoke({"chat_history":chat_message_history.messages})
            print("summary:",summary)
            chat_message_history.clear()
            chat_message_history.add_messages([summary])
        # 开发总结功能
        return chat_message_history

    def run(self, query):
        agent_executor = self._create_agent_executor(self.emotion)
        result = agent_executor.invoke({"input": query, })
        return result

    def catch_emotion_chain(self, query):
        # 使用chain来判断输入情绪
        prompt = """
        你是一个情绪识别专家。请分析用户输入的文本，判断用户当前的情绪状态。
        用户输入：{query}
        请从以下情绪类别中选择最匹配的一项：
        - upbeat: 兴奋、充满活力、积极向上
        - depressed: 沮丧、低落、消极、失望
        - cheerful: 开心、愉悦、快乐
        - angry: 愤怒、生气、不满、暴躁
        - friendly: 友好、亲切、温暖、善意
        - default: 中性、平静、无明显情绪倾向
        判断规则：
        1. 只输出情绪类别的英文单词（upbeat/depressed/cheerful/angry/friendly/default）
        2. 不要输出任何解释或其他文字
        3. 如果同时存在多种情绪，选择最强烈的那一种
        4. 如果无法判断，输出 default
        输出格式示例：
        cheerful
        """
        chain = ChatPromptTemplate.from_template(prompt) | self.chatmodel | StrOutputParser()
        result = chain.invoke({"query": query})
        self.emotion = result
        return result


master = Master()  # 初始化实例


# 根目录接口
@app.get("/")
def read_root():
    return {"Hello": "World"}


# chat
@app.post("/chat")
def chat(query: str):
    master.catch_emotion_chain(query)  # 情绪分类
    return master.run(query)  # 打包用户提示词


# URL
@app.post("/add_urls")
def add_urls():
    return {"response": "URL added!"}


# PDFs
@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "PDFs added"}


# Texts
@app.post("/add_texts")
def add_texts():
    return {"response": "Texts added!"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # 接受协议
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text is {data}")
    except WebSocketDisconnect:
        print("[测试语句]Connection closed!")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

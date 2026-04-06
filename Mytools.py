from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 向量模型
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor, tool  # 执行
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import Qdrant  # 向量数据库
from qdrant_client import QdrantClient  # 引入客户端
from langchain_core.output_parsers import JsonOutputParser
import requests #python请求包


cal_api_key = ""
@tool
def search(query):
    """只有需要了解实时信息或者不知道的时候才会使用这个搜索工具。"""
    serp = SerpAPIWrapper(serpapi_api_key="")  # ← 直接传入
    res = serp.run(query)
    print("实时搜索结果：", res)
    return res


@tool
def get_info_from_local_db(query):
    """只有回答有关2026年运势或者马年运势相关的问题时候，才会使用这个工具。"""
    client = Qdrant(
        QdrantClient(path="/langchain风水大师项目/local——qdrant"),
        "local_documents",  # 集合
        OpenAIEmbeddings(),
    )
    retriever = client.as_retriever(search_type="mmr")
    result = retriever.get_relevant_documents(query)
    return result


@tool
def bazi_measurement(query):
    """只有做八字排盘的时候才会使用这个工具，需要输入用户姓名和出生年月日，如果缺少用户姓名和出生年月日则不可用"""
    url = f"https://api.yuanfenju.com/index.php/v1/Bazi/paipan"
    prompt = ChatPromptTemplate.from_template("""
    你是一个参数查询助手，根据用户的输入内容找出相关的参数并按json格式返回，
    json字段如下：
    "api_key":"",
    "name":"姓名",
    "sex":"性别可以根据姓名自动判断 0男 1女",
    "type":"历类型默认1 0农历 1公历",
    "year":"出生年 例: 1988",
    "month":"出生月 例: 8",
    "day":"出生日 例: 7",
    "hours":"出生时 例: 14",
    "minute":"出生分 例: 30,如果不知道具体分，可以传数字 0",如果没有找到相关参数，则需要提醒用户告诉你这些内容，只返回数据结构，不要有其他评论。
    用户输入:{query}
    """)
    parser = JsonOutputParser()  #JSON解析器 #大模型输出的是字符串，虽然长得像JSON，但可能夹杂其他文字。JsonOutputParser会提取并验证其中的JSON内容，转换成可直接使用的字典格式。
    prompt = prompt.partial(format_instructions=parser.get_format_instructions()) #格式化指令
    print("bazi_paipan prompt:", prompt)
    chatmodel = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            streaming=True,
            base_url="https://api.deepseek.com",  # DeepSeek 需要配置
            api_key="",  # 替换为你的 API Key
        )
    chain = prompt | chatmodel | parser
    data = chain.invoke({"query":query})
    # print("bazi_paipan 查询结果:",data) #这个应该是从prompt->JSON的一条链路，Json为后续像测算官网请求做准备
    res = requests.post(url,data=data)
    if res.status_code == 200:
        print("res:", res)
        print("JSON:", res.json())
        try:
            return_string = f"八字为:{res.json()["data"]["bazi_info"]["bazi"]}"
            print("return_string:",return_string)
            return return_string
        except Exception as e:
            return "八字查询失败！可能你提供的信息不完整哦，需要出生年月日时以及姓名！"
    return "技术错误！请告诉用户稍后再试"


@tool
def dreaming(query:str):
    """只有用户需要解梦时，才会使用这个工具"""
    url = "https://api.yuanfenju.com/index.php/v1/Gongju/zhougong"
    prompt = """你需要将用户的输入进行关键词提取。
    1.用户输入内容多的话，可自行切词，然后用英文逗号分隔最多不超过10个关键词，
    比如："梦见,可爱的,婴儿"
    2.梦境关键字尽量精简，比如婴儿,火车。词语副词太多不一定可以匹配到。
    你的返回结果只有关键词。
    用户的输入为{keywords}
    """
    chatmodel = ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        streaming=True,
        base_url="https://api.deepseek.com",  # DeepSeek 需要配置
        api_key="",  # 替换为你的 API Key
    )
    chain = ChatPromptTemplate.from_template(prompt) | chatmodel | StrOutputParser()
    data = chain.invoke({"keywords":query})
    print("data:",data)
    res = requests.post(url, data={"api_key":cal_api_key,"title_zhougong":data})
    if res.status_code == 200:
        print("res:",res)
        print("JSON:",res.json())
        try:
            return_string = f"解梦结果为:{res.json()["data"]}"
            print("return_string:", return_string)
            return return_string
        except Exception as e:
            return "解梦查询失败！可能你提供的信息不完整哦！"
    return "技术错误！请告诉用户稍后再试"
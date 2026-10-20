import os
from email import message
from urllib import response

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from knowledge.processor.import_process import base

load_dotenv()

llm = ChatOpenAI(
    model="qwen-flash",
    temperature=0.0, # 控制随机性，0.0 表示完全确定性
    max_tokens=1024,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_API_BASE")
)

message =[
    SystemMessage(content="你是商品识别专家，只输出字符串"),
    HumanMessage(content="请从以下信息中识别商品名称：R12万用表如何使用?")
]

response = llm.invoke(message)
item_name = response.content.strip()
print(item_name) #R12万用表
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import os

load_dotenv()


api_key = os.getenv("DEEPSEEK_API_KEY")
llm = ChatDeepSeek(model="deepseek-v4-flash", api_key=api_key)
print(type(llm.invoke("Hello")))
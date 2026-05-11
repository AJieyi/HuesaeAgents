from dotenv import load_dotenv
import os
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from typing import List

# 加载 .env 中的环境变量（需提前在 .env 文件中设置 DEEPSEEK_API_KEY）
load_dotenv()

# 初始化 DeepSeek 模型
api_key = os.getenv("DEEPSEEK_API_KEY")
llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0,
    max_retries=2,
    api_key=api_key,
)

# 1. 定义输出的 Pydantic 模型
class Person(BaseModel):
    """人物信息"""
    name: str = Field(description="人物的姓名")
    age: int = Field(description="人物的年龄")
    hobbies: List[str] = Field(description="人物的兴趣爱好列表")
# 2. 绑定结构化输出
structured_llm = llm.with_structured_output(Person)
# 2. 绑定结构化输出
user_input = "张三今年28岁，他喜欢打篮球、阅读和旅行。"
prompt = f"从以下文本中提取人物信息，并以 JSON 格式输出：{user_input}"
result = structured_llm.invoke(prompt)


# 4. 输出结果
print(result)
print(f"姓名: {result.name}")
print(f"年龄: {result.age}")
print(f"爱好: {', '.join(result.hobbies)}")
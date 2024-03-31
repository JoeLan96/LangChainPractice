import os
import openai

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import find_dotenv, load_dotenv
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

openai_api_base="https://api.nextapi.fun/v1", # 注意，末尾要加 /v1
openai.default_headers = {"x-foo": "true"}



# 加载模型
llm = ChatOpenAI(model_name ="gpt-3.5-turbo")

# 设置提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])
# 创建 Chain
chain = prompt | llm
response = chain.invoke({"input": "how can langsmith help with testing?"})
print(type(response))
# langchain_core.messages.ai.AIMessage
print(response)


# 设置输出解析器
output_parser = StrOutputParser()

# 创建 Chain
chain2 = prompt | llm | output_parser

# 调用 Chain
response2 = chain2.invoke({"input": "how can langsmith help with testing?"})


print(type(response2))
# str
print(response2)



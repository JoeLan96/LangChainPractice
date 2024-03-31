import os
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser


from dotenv import find_dotenv, load_dotenv
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# 如果你需要通过代理端口访问，你需要如下配置
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'


chat_model = ChatOpenAI(model="gpt-3.5-turbo")
template = "Generate a list of 5 {text}.\n\n{format_instructions}"

chat_prompt = ChatPromptTemplate.from_template(template)
print(chat_prompt)

output_parser = CommaSeparatedListOutputParser()


chat_prompt = chat_prompt.partial(format_instructions=output_parser.get_format_instructions())
print(chat_prompt)

chain = chat_prompt | chat_model | output_parser

response = chain.invoke({"text": "colors"})
print(response)
# >> ['red', 'blue', 'green', 'yellow', 'orange']
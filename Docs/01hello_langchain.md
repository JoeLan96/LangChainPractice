## 01 Hello LangChain

### conda 创建虚拟环境
命令行执行
```cmd
conda create --name promptclpy38 python=3.8
```

安装必要的包new
```cmd
pip install  langchain==0.1.13 openai==1.14.2
```
package
langchain-0.1.13
langchain-community-0.0.29
langchain-core-0.1.33

colorama-0.4.6
distro-1.8.0
httpcore-1.0.2
httpx-0.26.0
langchain-openai-0.0.8
openai-1.14.2
tiktoken-0.5.2



安装可选包（使用环境变量配置API key）
```cmd
pip install python-dotenv
```

### 简单调用示例 (官方API，需要外网)
```python
import os
import openai

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import find_dotenv, load_dotenv
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# 如果你需要通过代理端口访问，你需要如下配置
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'



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
```

### 中转 API 示例（无需外网）
```python

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 1.加载文档 
loader = TextLoader("./data/log.txt", encoding="utf-8")
docs = loader.load()

# 2.拆分文档  （大文档才拆分）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_documents(docs)


# 3.向量化文档并创建检索器
db = FAISS.from_documents(texts, OpenAIEmbeddings(
    openai_api_base="https://api.nextapi.fun/v1", # 注意，末尾要加 /v1
    openai_api_key="ak-5q53TviefzOYW8xW2F0gFtS20T288sPWSD92YQpjqmSJ",
))
retriever = db.as_retriever()

# 4.创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is Carl."),
    ("user", "Answer the question based only on the following context:\n{context}\nQuestion: {question}")
])


# 5. 加载模型（此方法为中转API调用）
model = ChatOpenAI(
    model_name ="gpt-3.5-turbo",
    openai_api_base="https://api.nextapi.fun/v1", 
    openai_api_key="ak-5q53TviefzOYW8xW2F0gFtS20aO2gT288sPWSD92YQpjqmSJ",
)

# 设置输出转换器
output_parser = StrOutputParser()

# 开启内容并行装载（同时填充context和question）
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

chain = setup_and_retrieval | prompt | model | output_parser

result = chain.invoke("What it is your name? and where did harrison work?")
print(result)
```


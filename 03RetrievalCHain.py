import os
import openai
from dotenv import find_dotenv, load_dotenv
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# 如果你需要通过代理端口访问，你需要如下配置
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'


# 1.加载文档
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./LongText.md", encoding="utf-8")
docs = loader.load()
print(docs)

# 2.拆分文档
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)

split_docs = text_splitter.split_documents(docs)
print(len(docs[0].page_content))
print(len(split_docs))

sum = 0
for split_doc in split_docs:
    cal = len(split_doc.page_content)
    print(cal)
    sum += cal
print("sum = ", sum)

# 3.向量存储
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
vector = FAISS.from_documents(split_docs, embeddings)
print(vector)


## 测试

# 模型加载
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

# 提示模板创建
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# chain
document_chain = create_stuff_documents_chain(llm, prompt)

# chain
from langchain.chains import create_retrieval_chain

query = "What is the use case of AWS Serverless?"
similar_docs = vector.similarity_search_with_score(query, 3, include_metadata=True)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 测试
response = retrieval_chain.invoke({"input": "什么是 WTF Langchain？"})
print(response["answer"])

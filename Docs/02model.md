
## 02 模型

参考教程 [LangChain 中的 Concepts](https://python.langchain.com/docs/modules/model_io/concepts#models)

---

### 模型简介

Langchain所封装的模型分为两类：
- 大语言模型 (LLM)
- 聊天模型 (Chat Models)

在后续的内容中，为简化描述，我们将使用 `LLM` 来指代大语言模型。

Langchain的支持众多模型供应商，包括OpenAI、ChatGLM、HuggingFace等。本教程中，我们将以OpenAI为例，后续内容中提到的模型默认为OpenAI提供的模型。

Langchain的封装，比如，对OpenAI模型的封装，实际上是指的是对OpenAI API的封装。


### Langchain与OpenAI模型

参考OpenAI [Model endpoint compatibility](https://platform.openai.com/docs/models/model-endpoint-compatibility) 文档，gpt模型都归为了聊天模型，而davinci, curie, babbage, ada模型都归为了文本补全模型。

| ENDPOINT | MODEL NAME |
| -------- | ------- |
| /v1/chat/completions | gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613, gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613 |
| /v1/completions | (Legacy)	text-davinci-003, text-davinci-002, text-davinci-001, text-curie-001, text-babbage-001, text-ada-001, davinci, curie, babbage, ada |

Langchain提供接口集成不同的模型。为了便于切换模型，Langchain将不同模型抽象为相同的接口 `BaseLanguageModel`，并提供 `predict` 和 `predict_messages` 函数来调用模型。

当使用LLM时推荐使用predict函数，当使用聊天模型时推荐使用predict_messages函数。

### 与LLM的交互

与LLM的交互，我们需要使用 `langchain.llms` 模块中的 `OpenAI` 类。

```python
from langchain.llms import OpenAI

import os
os.environ['OPENAI_API_KEY'] = '您的有效OpenAI API Key'

llm = OpenAI(model_name="text-davinci-003")
response = llm.predict("What is AI?")
print(response)
```

你应该能看到类似如下输出：

```shell
AI (Artificial Intelligence) is a branch of computer science that deals with creating intelligent machines that can think, reason, learn, and problem solve. AI systems are designed to mimic human behavior and can be used to automate tasks or provide insights into data. AI can be used in a variety of fields, such as healthcare, finance, robotics, and more.
```

### 与聊天模型的交互

与聊天模型的交互，我们需要使用 `langchain.chat_models` 模块中的 `ChatOpenAI` 类。

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

import os
os.environ['OPENAI_API_KEY'] = '您的有效OpenAI API Key'

chat = ChatOpenAI(temperature=0)
response = chat.predict_messages([ 
  HumanMessage(content="What is AI?")
])
print(response)
```

你应该能看到类似如下输出：

```shell
content='AI, or Artificial Intelligence, refers to the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, problem-solving, perception, and language understanding. AI technology has the capability to drastically change and improve the way we work, live, and interact.' additional_kwargs={} example=False
```

通过以下代码我们查看一下 `response` 变量的类型：

```python
response.__class
```

可以看到，它是一个 `AIMessage` 类型的对象。

```shell
langchain.schema.messages.AIMessage
```

```shell

接下来我们使用 `SystemMessage` 指令来指定模型的行为。如下代码指定模型对AI一无所知，在回答AI相关问题时，回答“I don't know”。

```python
response = chat.predict_messages([
  SystemMessage(content="You are a chatbot that knows nothing about AI. When you are asked about AI, you must say 'I don\'t know'"),
  HumanMessage(content="What is deep learning?")
])
print(response)
```

你应该能看到类似如下输出：

```shell
content="I don't know." additional_kwargs={} example=False
```

### 3个消息类

Langchain框架提供了三个消息类，分别是 `AIMessage`、`HumanMessage` 和 `SystemMessage`。它们对应了OpenAI聊天模型API支持的不同角色 `assistant`、`user` 和 `system`。请参考 [OpenAI API文档 - Chat - Role](https://platform.openai.com/docs/api-reference/chat/create#chat/create-role)。

| Langchain类 | OpenAI角色 | 作用 |
| -------- | ------- | ------- |
| AIMessage | assistant | 模型回答的消息 |
| HumanMessage | user | 用户向模型的请求或提问 |
| SystemMessage | system | 系统指令，用于指定模型的行为 |
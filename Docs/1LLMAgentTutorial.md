# LLM Agent Tutorial

注：openai_api_key 已做替换处理。请替换成自己的中转 API key 。



## 1 环境配置

### 1.1 python 虚拟环境创建

```cmd
conda create --name promptclpy38 python=3.8
```

安装 langchain 和 openai

```cmd
pip install langchain

pip install langchain-openai

pip install faiss-cpu
```

如果上述默认安装版本不可用，可以使用指定版本安装

```cmd
pip install  langchain==0.1.13 openai==1.14.2
```

详细的 package 版本

- langchain-0.1.13
- langchain-community-0.0.29
- langchain-core-0.1.33
- colorama-0.4.6
- distro-1.8.0
- httpcore-1.0.2
- httpx-0.26.0
- langchain-openai-0.0.8
- openai-1.14.2
- tiktoken-0.5.2
- faiss-1.7.4
- chromadb-0.4.24

### 1.2 City Learn 环境配置

在这个示例教程中，我们使用 citylearn 作为交互的环境

下面说明 city Learn 的环境配置

首先进入到项目的目录，如在 ./citylearn-phase-2 目录下运行

```cmd
pip install -r requirements.txt
```

如果有其他包，按提示安装即可。

## 2 LLM Agent 构建

### 2.1 文档加载及向量化

```
!pwd
```

/root/AAAMyProjects/citylearn-phase-2-submission-version1.7

```python
# 加载文档
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/log30.txt", encoding="utf-8")
docs = loader.load()
print(docs)
print(len(docs[0].page_content))
```

当面临大文档时，直接将整个文档输入到 prompt 中会消耗大量的 token 。

通常的做法是先将文档拆分成小块，然后使用相关性检索，找到最相关的 n(默认n=4) 个块传入 prompt。

`Langchain` 的 `Embeddings` 类实现与文本嵌入模型进行交互的标准接口。 当前市场上有许多嵌入模型提供者（如OpenAI、Cohere、Hugging Face等）。

嵌入模型创建文本片段的向量表示。这意味着我们可以在向量空间中处理文本，并进行语义搜索等操作， 从而查找在向量空间中最相似的文本片段。

注，文本之间的相似度由其向量表示的欧几里得距离来衡量。欧几里得距离是最常见的距离度量方式，也称为L2范数。 对于两个n维向量a和b，欧几里得距离可以通过以下公式计算：

```math
math
d(a, b) = √((a₁ - b₁)² + (a₂ - b₂)² + ... + (aₙ - bₙ)²)
```



```python
# # 拆分文档
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 2000,
#     chunk_overlap  = 200,
#     length_function = len,
# )

# # 拆分文档存入变量 texts

# texts = text_splitter.split_documents(docs)
# print(len(docs[0].page_content))
# for split_doc in texts:
#   print(len(split_doc.page_content))
```

针对不同的内容（如：普通文本、MarkDown、代码、HTML、Token等），通常有不同的文档拆分拆分方法，具体可以参考 https://python.langchain.com/docs/modules/data_connection/document_transformers/

```
# for split_doc in texts:
#   print(split_doc)
```



```python
# 向量化文档并生成检索器
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

completed_db = FAISS.from_documents(docs, OpenAIEmbeddings(
    openai_api_base="https://api.nextapi.fun/v1", 
    openai_api_key="ak-5q53TviefzOYW8xWS20aO2gT288sPWSD92YQpjqmSJ",
))

# 生成检索器
AgentRetriever = completed_db.as_retriever()
```



```python
print(AgentRetriever)
```

tags=['FAISS', 'OpenAIEmbeddings'] vectorstore=<langchain_community.vectorstores.faiss.FAISS object at 0x7ff3ebaba220>

### 2.2 提示词模板创建

[参考教程](https://python.langchain.com/docs/modules/model_io/prompts/composition#chat-prompt-composition)

注意：我们注意使用 chat prompt

聊天提示词模板的内容组成主要包含以下部分：

- SystemMessage 系统消息，通常用于指定角色
- HumanMessage 人类消息，就是我们平常输入的内容
- AIMessage AI回复的消息，用作多轮交互

使用模板的好处是，我们将不变的内容一次写好，只需要加载变的内容就好

```python
# 创建提示词模板
from langchain_core.prompts import ChatPromptTemplate


role = """
# ⻆⾊
你是一位电力系统分析领域的专家，擅长电力系统仿真，电力系统调度决策。
你会根据当前的电力系统状态信息，给出电力系统的调度决策。

## 技能
### 技能⼀：认真学习知识库中的历史决策经验记录。
决策经验通常由一个observations数组和actions数组组成，其中observations表示状态，actions表示决策。
其中，observation_names 中的字符分别是 observations 数组中对应位置数值的 key，action_names 中的字符分别是 actions 数组中对应位置数值的key。

### 技能二：根据Agent 历史决策经验记录，为新的 observations 给出对应的 actions。
你将根据用户输入的状态信息，结合决策经验给出对应决策。

## 输出形式
你的输出形式为键值对，key 固定为 "actions"，value为决策的数组。

## 例子：
用户输入
observations: [5, 1, 24.66, 24.910639, 38.41596, 27.611464, 0.0, 54.625927, 116.84289, 0.0, 0.0, 143.32434, 1020.7561, 0.0, 0.40248835, 23.098652, 0.35683933, 0.0, 0.0, 0.2, 0.67788136, 0.02893, 0.02893, 0.02915, 0.02893, 1.1192156, 0.055682074, 3.0, 23.222221, 0, 24.278513, 0.18733284, 0.0, 0.0, 0.2, 0.18733284, 0.0, 0.0, 1.0, 24.444445, 0, 24.431562, 0.4220805, 0.0, 0.0, 0.2, 0.5631514, 0.5579055, 0.0, 2.0, 24.444445, 0] 
你的回复
actions: [-0.12341118, -0.30998343, 0.15703982, -0.14674222, -0.1908567, 0.21175343, -0.29970372, -0.18800199, 0.19280493]
"""

mes = """
"历史决策信息为：\n{context}\n 当前的状态信息为:{question}\n 请给出决策: "
"""

AgentPrompt = ChatPromptTemplate.from_messages([
    ("system", role),
    ("user", mes)
])
```

在 `mes = """ "历史决策信息为：\n{context}\n 当前的状态信息为:\n 请给出决策: " """`中，{context} 和 {question} 就是变量，可以用传参数的方式传递文字进去。

### 2.3 模型加载与 chain 的创建

这里我们使用的是 中转 API ，使用以下固定形式进行调研，model_name 可以切换模型，在创建密钥时可以查看。具体有：

- gpt-4gpt-4-0314
- gpt-4-0613
- gpt-4-32k
- gpt-4-32k-0314
- gpt-4-32k-0613
- gpt-4-turbo-preview
- gpt-4-1106-preview
- gpt-4-0125-preview
- gpt-3.5-turbo-0301
- gpt-3.5-turbo
- gpt-3.5-turbo-0613
- gpt-3.5-turbo-16k
- gpt-3.5-turbo-16k-0613
- gpt-3.5-turbo-1106
- gpt-3.5-turbo-0125
- text-embedding-ada-002
- text-embedding-3-large
- text-embedding-3-small

```python
# 加载模型

AgentModel = ChatOpenAI(
    model_name ="gpt-3.5-turbo",
    openai_api_base="https://api.nextapi.fun/v1", 
    openai_api_key="ak-5q53TviefzOYW8xWS20aO2gT288sPWSD92YQpjqmSJ",
)
```



```python
# 创建 chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# RunnableParallel 用于并行的进行内容填充，即 setup_and_retrieval 运行时，会并行给 context 和 question 赋值。
setup_and_retrieval = RunnableParallel(
    {"context": AgentRetriever, "question": RunnablePassthrough()}
)

text = """
observations: [5, 1, 24.66, 24.910639, 38.41596, 27.611464, 0.0, 54.625927, 116.84289, 0.0, 0.0, 143.32434, 1020.7561, 0.0, 0.40248835, 23.098652, 0.35683933, 0.0, 0.0, 0.2, 0.67788136, 0.02893, 0.02893, 0.02915, 0.02893, 1.1192156, 0.055682074, 3.0, 23.222221, 0, 24.278513, 0.18733284, 0.0, 0.0, 0.2, 0.18733284, 0.0, 0.0, 1.0, 24.444445, 0, 24.431562, 0.4220805, 0.0, 0.0, 0.2, 0.5631514, 0.5579055, 0.0, 2.0, 24.444445, 0]
"""

# 创建了一个转换提示词的 chain
prompt_value = (setup_and_retrieval | AgentPrompt).invoke(text)
print(prompt_value)
print("复制")
```



```python
# 创建完整的 chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


output_parser = StrOutputParser()

# 完整的 chain，类似函数式编程，将 检索器，提示词模板，模型，输出格式化依次串联起来。
# 相关性搜索setup_and_retrieval，到生成提示词AgentPrompt，到传入模型AgentModel，到格式化输出output_parser
chain = setup_and_retrieval | AgentPrompt | AgentModel | output_parser

# 第一次交互测试
text = """
observations: [5, 1, 24.66, 24.910639, 38.41596, 27.611464, 0.0, 54.625927, 116.84289, 0.0, 0.0, 143.32434, 1020.7561, 0.0, 0.40248835, 23.098652, 0.35683933, 0.0, 0.0, 0.2, 0.67788136, 0.02893, 0.02893, 0.02915, 0.02893, 1.1192156, 0.055682074, 3.0, 23.222221, 0, 24.278513, 0.18733284, 0.0, 0.0, 0.2, 0.18733284, 0.0, 0.0, 1.0, 24.444445, 0, 24.431562, 0.4220805, 0.0, 0.0, 0.2, 0.5631514, 0.5579055, 0.0, 2.0, 24.444445, 0]
"""

result = chain.invoke(text)
print(result)
```



```python
# 第2次交互
text = """
observations: [5, 11, 33.87, 40.085255, 27.676508, 35.020977, 328.87, 102.59606, 0.0, 172.22371, 539.94, 682.6962, 0.0, 749.1275, 0.45759925, 24.40689, 0.39833075, 1.3557849, 0.0, 0.105440885, -0.38306755, 0.02915, 0.05867, 0.02893, 0.02893, 1.6538339, 0.0, 1.0, 24.444445, 0, 24.998137, 0.78700435, 0.67789245, 0.0, 0.105440885, 0.60736823, 1.5132071, 0.0, 1.0, 25.0, 0, 23.333338, 1.5277264, 1.3557849, 0.0, 0.10990912, 1.2639599, 2.3398511, 0.20385616, 1.0, 23.333334, 0]
"""

chain = setup_and_retrieval | AgentPrompt | AgentModel | output_parser

result = chain.invoke(text)
print(result)
```



```python
# 第3次交互
text = """
observations: [5, 18, 38.12, 26.891977, 23.315088, 37.287792, 84.96, 0.0, 0.0, 108.99303, 608.73, 0.0, 0.0, 562.6354, 0.44940948, 28.699625, 0.7734966, 0.59355897, 0.0, 0.06735881, 0.7720484, 0.05867, 0.02893, 0.02893, 0.02893, 1.4401542, 0.0, 2.0, 25.555555, 0, 29.613392, 0.3967779, 0.29677948, 0.0, 0.06735881, 0.892815, 1.5099409, 0.18567085, 1.0, 25.0, 0, 26.106087, 0.74361974, 0.59355897, 0.0, 0.07228303, 0.90870994, 1.7653095, 0.0, 1.0, 24.444445, 0]
"""

result = chain.invoke(text)
print(result)
{
  "actions": [-0.22071165, -0.40284544, 0.09194043, -0.07699513, -0.22387356, 0.1765942, -0.3893789, -0.24881685, 0.1414268]
}
```

### 2.4 对模型输出进行格式处理，对接 citylearn环境的输入格式。

这一步是比较繁琐的，需要仔细查看环境的输入形式。

```python
import json
import numpy as np

class LLMAgent:
    def __init__(self, chain):
        self.chain = chain

    # predict 函数接受一个 observations 数组，返回一个 actions 数组
    def predict(self, observations):
        text = f"observations: {observations}"
        # 获取模型输出
        mid = self.chain.invoke(text)

        # 使用 json 解析模型输出
        data_list = json.loads(mid)['actions']
        # print(data_list)

        # 将输出转换为 numpy 数组
        data_array = np.array(data_list, dtype=np.float32)
        # print(data_array)
        return [data_array]

    # def predict2(self, observations):
    #     text = f"observations: {observations}"
    #     return self.chain.invoke(text)
```



```python
# 测试 LLMAgent 的输出是否正确

agent = LLMAgent(chain)

text = """
[5, 3, 23.9, 29.351833, 39.150482, 23.659443, 0.0, 259.82214, 82.855934, 0.0, 0.0, 196.69113, 987.9093, 0.0, 0.3694583, 22.22306, 0.33876896, 0.0, 0.0, 0.1759648, 0.68140554, 0.02893, 0.02915, 0.02915, 0.02893, 1.4583724, 0.057004254, 3.0, 22.222221, 0, 24.214113, 0.18461598, 0.0, 0.045191526, 0.1759648, 0.22423235, 0.0, 0.0, 1.0, 24.444445, 0, 24.444445, 0.41660294, 0.0, 0.0, 0.17743151, 0.41237712, 0.11927145, 0.0, 2.0, 24.444445, 0]
"""
res = agent.predict(text)
print(res)

```

[array([-0.12294388, -0.17461687,  0.15998137, -0.12203425, -0.13837194,
        0.28283915, -0.21346188, -0.21001118,  0.23139182], dtype=float32)]

```
print(type(res))
```

<class 'list'>

## 3 City Learn Environment 加载

备注：

75行 `agent = LLMAgent(chain)` 使用 LLM 创建的 Agent 替换 RL Agent

95行 `while num_steps<3:` 用于测试，正式运行请改为 `while True:` 但是目前不见这么做，特别消耗 Token

```python
import numpy as np
import time
import os
from citylearn.citylearn import CityLearnEnv

"""
This is only a reference script provided to allow you 
to do local evaluation. The evaluator **DOES NOT** 
use this script for orchestrating the evaluations. 
"""

# from agents.user_agent import SubmissionAgent
from rewards.user_reward import SubmissionReward

class WrapperEnv:
    """
    Env to wrap provide Citylearn Env data without providing full env
    Preventing attribute access outside of the available functions
    """
    def __init__(self, env_data):
        self.observation_names = env_data['observation_names']
        self.action_names = env_data['action_names']
        self.observation_space = env_data['observation_space']
        self.action_space = env_data['action_space']
        self.time_steps = env_data['time_steps']
        self.seconds_per_time_step = env_data['seconds_per_time_step']
        self.random_seed = env_data['random_seed']
        self.buildings_metadata = env_data['buildings_metadata']
        self.episode_tracker = env_data['episode_tracker']
    
    def get_metadata(self):
        return {'buildings': self.buildings_metadata}

def create_citylearn_env(config, reward_function):
    env = CityLearnEnv(config.SCHEMA, reward_function=reward_function)

    env_data = dict(
        observation_names = env.observation_names,
        action_names = env.action_names,
        observation_space = env.observation_space,
        action_space = env.action_space,
        time_steps = env.time_steps,
        random_seed = None,
        episode_tracker = None,
        seconds_per_time_step = None,
        buildings_metadata = env.get_metadata()['buildings']
    )
    print(f"observation_names : {env.observation_names}")
    print(f"action_names : {env.action_names}")

    print("----------------------------------------")
    # print(f"observation_space : {env.observation_space}")
    # print(f"action_space : {env.action_space}")

    wrapper_env = WrapperEnv(env_data)
    return env, wrapper_env

def update_power_outage_random_seed(env: CityLearnEnv, random_seed: int) -> CityLearnEnv:
    """Update random seed used in generating power outage signals.
    
    Used to optionally update random seed for stochastic power outage model in all buildings.
    Random seeds should be updated before calling :py:meth:`citylearn.citylearn.CityLearnEnv.reset`.
    """

    for b in env.buildings:
        b.stochastic_power_outage_model.random_seed = random_seed

    return env

def evaluate(config):
    print("Starting local evaluation")
    
    env, wrapper_env = create_citylearn_env(config, SubmissionReward)
    # print("Env Created")

    agent = LLMAgent(chain)

    observations = env.reset()
    print(f"observations_0: {observations[0]}")

    # print(env.observation_names)
    agent_time_elapsed = 0

    # step_start = time.perf_counter()
    actions = agent.predict(observations[0])
    print(f"actions_0: {actions[0]}")
    # agent_time_elapsed += time.perf_counter() - step_start

    episodes_completed = 0
    num_steps = 0
    interrupted = False
    episode_metrics = []
    try:
        while num_steps<3:
            
            ### This is only a reference script provided to allow you 
            ### to do local evaluation. The evaluator **DOES NOT** 
            ### use this script for orchestrating the evaluations. 

            observations, _, done, _ = env.step(actions)
            # print(f"observations_{num_steps + 1}: {observations[0]}")

            # print(done)
            if not done:
                # step_start = time.perf_counter()
                actions = agent.predict(observations[0])
                print(f"actions_{num_steps + 1}: {actions[0]}")

                # print(actions)
                # agent_time_elapsed += time.perf_counter()- step_start
            else:
                
                episodes_completed += 1
                metrics_df = env.evaluate_citylearn_challenge()
                episode_metrics.append(metrics_df)
                
                print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics_df}", )
                
                # Optional: Uncomment line below to update power outage random seed 
                # from what was initially defined in schema
                env = update_power_outage_random_seed(env, 90000)

                observations = env.reset()

                # step_start = time.perf_counter()
                actions = agent.predict(observations)
                # agent_time_elapsed += time.perf_counter()- step_start
                return metrics_df
            
            num_steps += 1
            if num_steps % 1000 == 0:
                print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

            if episodes_completed >= config.num_episodes:
                break
    

    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True
    
    if not interrupted:
        print("=========================Completed=========================")

    print(f"Total time taken by agent: {agent_time_elapsed}s")
max recoder init finished
history recoder init finished
```

## 5 开始评估 Start to Evalution

这里使用 sys 将 print 的输出内容从命令行重定向到指定文件`LLMevaluate.txt`中了，记得修改文件名

```python
import sys

class Config:
    data_dir = './data/'
    SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')
    num_episodes = 1
    
config = Config()



original_stdout = sys.stdout  # 保存标准输出流
with open('LLMevaluate.txt', 'w', encoding="utf-8") as f:
    sys.stdout = f  # 重定向标准输出流到文件
    evaluate(config)

sys.stdout = original_stdout  # 还原标准输出流
```
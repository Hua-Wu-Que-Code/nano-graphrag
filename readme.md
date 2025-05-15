<div align="center">
  <a href="https://github.com/gusye1234/nano-graphrag">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://assets.memodb.io/nano-graphrag-dark.png">
      <img alt="Shows the MemoDB logo" src="https://assets.memodb.io/nano-graphrag.png" width="512">
    </picture>
  </a>
  <p><strong>一个简单、易于修改的 GraphRAG 实现</strong></p>
  <p>
    <img src="https://img.shields.io/badge/python->=3.9.11-blue">
    <a href="https://pypi.org/project/nano-graphrag/">
      <img src="https://img.shields.io/pypi/v/nano-graphrag.svg">
    </a>
    <a href="https://codecov.io/github/gusye1234/nano-graphrag" > 
     <img src="https://codecov.io/github/gusye1234/nano-graphrag/graph/badge.svg?token=YFPMj9uQo7"/> 
 		</a>
    <a href="https://pepy.tech/project/nano-graphrag">
      <img src="https://static.pepy.tech/badge/nano-graphrag/month">
    </a>
  </p>
  <p>
  	<a href="https://discord.gg/sqCVzAhUY6">
      <img src="https://dcbadge.limes.pink/api/server/sqCVzAhUY6?style=flat">
    </a>
    <a href="https://github.com/gusye1234/nano-graphrag/issues/8">
       <img src="https://img.shields.io/badge/群聊-wechat-green">
    </a>
  </p>
</div>









😭 [GraphRAG](https://arxiv.org/pdf/2404.16130) 很强大，但官方[实现](https://github.com/microsoft/graphrag/tree/main)代码难以**阅读或修改**。

😊 本项目提供了一个**更小、更快、更简洁的 GraphRAG**，同时保留核心功能（见[基准测试](#benchmark)和[问题](#Issues)）。

🎁 除去 `tests` 和 prompt，`nano-graphrag` 仅约 **1100 行代码**。

👌 小巧但[**可移植**](#Components)（faiss、neo4j、ollama...）、[**异步**](#Async)且全类型注解。



> 如果你在寻找一个多用户、长期记忆的 RAG 方案，可以看看这个项目：[memobase](https://github.com/memodb-io/memobase) :)

## 安装

**推荐源码安装**

```shell
# 先克隆本仓库
cd nano-graphrag
pip install -e .
```

**PyPi 安装**

```shell
pip install nano-graphrag
```



## 快速开始

> [!TIP]
>
> **请在环境变量中设置 OpenAI API key：`export OPENAI_API_KEY="sk-..."`。** 

> [!TIP]
> 如果你使用 Azure OpenAI API，请参考 [.env.example](./.env.example.azure) 设置，然后通过 `GraphRAG(...,using_azure_openai=True,...)` 启用。

> [!TIP]
> 如果你使用 Amazon Bedrock API，请确保通过 `aws configure` 等命令正确设置凭证。然后通过如下方式启用：`GraphRAG(...,using_amazon_bedrock=True, best_model_id="us.anthropic.claude-3-sonnet-20240229-v1:0", cheap_model_id="us.anthropic.claude-3-haiku-20240307-v1:0",...)`。可参考[示例脚本](./examples/using_amazon_bedrock.py)。

> [!TIP]
>
> 如果你没有任何 key，可以参考这个[示例](./examples/no_openai_key_at_all.py)，使用 `transformers` 和 `ollama`。如需自定义 LLM 或 Embedding Model，请见[进阶用法](#Advances)。

下载狄更斯的《圣诞颂歌》：

```shell
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
```

使用如下 Python 代码：

```python
from nano_graphrag import GraphRAG, QueryParam

graph_func = GraphRAG(working_dir="./dickens")

with open("./book.txt") as f:
    graph_func.insert(f.read())

# 全局 graphrag 检索
print(graph_func.query("What are the top themes in this story?"))

# 局部 graphrag 检索（更好且更易扩展）
print(graph_func.query("What are the top themes in this story?", param=QueryParam(mode="local")))
```

下次用同一个 `working_dir` 初始化 `GraphRAG` 时，会自动加载所有上下文。

#### 批量插入

```python
graph_func.insert(["TEXT1", "TEXT2",...])
```

<details>
<summary> 增量插入</summary>

`nano-graphrag` 支持增量插入，不会重复计算或存储数据：

```python
with open("./book.txt") as f:
    book = f.read()
    half_len = len(book) // 2
    graph_func.insert(book[:half_len])
    graph_func.insert(book[half_len:])
```

> `nano-graphrag` 使用内容的 md5-hash 作为 key，因此不会有重复 chunk。
>
> 但每次插入时，图的社区会重新计算，社区报告也会重新生成。

</details>

<details>
<summary> 朴素 RAG</summary>

`nano-graphrag` 也支持朴素 RAG 的插入和查询：

```python
graph_func = GraphRAG(working_dir="./dickens", enable_naive_rag=True)
...
# 查询
print(rag.query(
      "What are the top themes in this story?",
      param=QueryParam(mode="naive")
)
```
</details>


### 异步

每个方法 `NAME(...)` 都有对应的异步方法 `aNAME(...)`

```python
await graph_func.ainsert(...)
await graph_func.aquery(...)
...
```

### 可用参数

`GraphRAG` 和 `QueryParam` 都是 Python 的 `dataclass`。用 `help(GraphRAG)` 和 `help(QueryParam)` 查看所有参数！或参考[进阶用法](#Advances)部分。



## 组件

可用组件如下：

| 类型            |                             说明                             |                       位置                       |
| :-------------- | :----------------------------------------------------------: | :-----------------------------------------------: |
| LLM             |                            OpenAI                            |                     内置实现                      |
|                 |                        Amazon Bedrock                        |                     内置实现                      |
|                 |                           DeepSeek                           |              [示例](./examples)                   |
|                 |                           `ollama`                           |              [示例](./examples)                   |
| Embedding       |                            OpenAI                            |                     内置实现                      |
|                 |                        Amazon Bedrock                        |                     内置实现                      |
|                 |                    Sentence-transformers                     |              [示例](./examples)                   |
| 向量数据库      | [`nano-vectordb`](https://github.com/gusye1234/nano-vectordb) |                     内置实现                      |
|                 |        [`hnswlib`](https://github.com/nmslib/hnswlib)        |         内置实现, [示例](./examples)              |
|                 |  [`milvus-lite`](https://github.com/milvus-io/milvus-lite)   |              [示例](./examples)                   |
|                 | [faiss](https://github.com/facebookresearch/faiss?tab=readme-ov-file) |              [示例](./examples)                   |
| 图存储          | [`networkx`](https://networkx.org/documentation/stable/index.html) |                     内置实现                      |
|                 |                [`neo4j`](https://neo4j.com/)                 | 内置实现([文档](./docs/use_neo4j_for_graphrag.md))|
| 可视化          |                           graphml                            |              [示例](./examples)                   |
| 分块            |                        按 token 大小                         |                     内置实现                      |
|                 |                       按文本分割器                           |                     内置实现                      |

- `内置实现` 表示我们在 `nano-graphrag` 内部实现了该组件。`示例` 表示我们在 [examples](./examples) 文件夹下有相关教程。

- 查看 [examples/benchmarks](./examples/benchmarks) 了解组件间的对比。
- **欢迎贡献更多组件。**

## 进阶用法



<details>
<summary>一些设置选项</summary>

- `GraphRAG(...,always_create_working_dir=False,...)` 可跳过目录创建步骤。如果你把所有组件都换成非文件存储可用此选项。

</details>



<details>
<summary>只查询相关上下文</summary>

`graph_func.query` 默认返回最终答案（非流式）。

如需集成到你的项目，可用 `param=QueryParam(..., only_need_context=True,...)`，只返回检索到的上下文，例如：
````
# 局部模式
-----Reports-----
```csv
id,	content
0,	# FOX News and Key Figures in Media and Politics...
1, ...
```
...

# 全局模式
----Analyst 3----
Importance Score: 100
Donald J. Trump: Frequently discussed in relation to his political activities...
...
````

你可以将这些上下文集成到自定义的提示中。

</details>

<details>
<summary>提示</summary>

`nano-graphrag` 的提示语来自 `nano_graphrag.prompt.PROMPTS` 字典对象。你可以修改或替换其中任何提示。

一些重要的提示：

- `PROMPTS["entity_extraction"]` 用于从文本块中提取实体和关系。
- `PROMPTS["community_report"]` 用于组织和总结图聚类的描述。
- `PROMPTS["local_rag_response"]` 是局部检索生成的系统提示模板。
- `PROMPTS["global_reduce_rag_response"]` 是全局检索生成的系统提示模板。
- `PROMPTS["fail_response"]` 是当与用户查询无关时的备用响应。

</details>

<details>
<summary>自定义分块</summary>


`nano-graphrag` 允许你自定义分块方法，查看[示例](./examples/using_custom_chunking_method.py)。

切换到内置的文本分割器分块方法：

```python
from nano_graphrag._op import chunking_by_seperators

GraphRAG(...,chunk_func=chunking_by_seperators,...)
```

</details>



<details>
<summary>LLM 函数</summary>

在 `nano-graphrag` 中，我们需要两种类型的 LLM，一种用于计划和响应，另一种用于摘要。默认情况下，前者是 `gpt-4o`，后者是 `gpt-4o-mini`。

你可以实现自己的 LLM 函数（参考 `_llm.gpt_4o_complete`）：

```python
async def my_llm_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
  # 弹出缓存 KV 数据库
  hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
  # 其余的 kwargs 用于调用 LLM，例如 `max_tokens=xxx`
	...
  # 调用你的 LLM
  response = await call_your_LLM(messages, **kwargs)
  return response
```

用以下代码替换默认设置：

```python
# 如有需要，调整最大令牌大小或最大异步请求数
GraphRAG(best_model_func=my_llm_complete, best_model_max_token_size=..., best_model_max_async=...)
GraphRAG(cheap_model_func=my_llm_complete, cheap_model_max_token_size=..., cheap_model_max_async=...)
```

你可以参考这个[示例](./examples/using_deepseek_as_llm.py)，使用 [`deepseek-chat`](https://platform.deepseek.com/api-docs/) 作为 LLM 模型

你可以参考这个[示例](./examples/using_ollama_as_llm.py)，使用 [`ollama`](https://github.com/ollama/ollama) 作为 LLM 模型

#### Json 输出

`nano-graphrag` 将使用 `best_model_func` 输出 JSON，参数为 `"response_format": {"type": "json_object"}`。但某些开源模型可能会生成不稳定的 JSON。 

`nano-graphrag` 引入了一个后处理接口，用于将响应转换为 JSON。该函数的签名如下：

```python
def YOUR_STRING_TO_JSON_FUNC(response: str) -> dict:
  "将字符串响应转换为 JSON"
  ...
```

通过 `GraphRAG(...convert_response_to_json_func=YOUR_STRING_TO_JSON_FUNC,...)` 传入你自己的函数。

例如，你可以参考 [json_repair](https://github.com/mangiucugna/json_repair) 修复 LLM 返回的 JSON 字符串。 
</details>



<details>
<summary>Embedding 函数</summary>

你可以用任何 `_utils.EmbedddingFunc` 实例替换默认的嵌入函数。

例如，默认使用 OpenAI 嵌入 API：

```python
@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = AsyncOpenAI()
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
```

用以下代码替换默认嵌入函数：

```python
GraphRAG(embedding_func=your_embed_func, embedding_batch_num=..., embedding_func_max_async=...)
```

你可以参考这个[示例](./examples/using_local_embedding_model.py)，使用 `sentence-transformer` 本地计算嵌入。
</details>


<details>
<summary>存储组件</summary>

你可以用自己的实现替换所有与存储相关的组件，`nano-graphrag` 主要使用三种存储：

**`base.BaseKVStorage` 存储键值对数据** 

- 默认使用磁盘文件存储作为后端。 
- `GraphRAG(.., key_string_value_json_storage_cls=YOURS,...)`

**`base.BaseVectorStorage` 索引嵌入**

- 默认使用 [`nano-vectordb`](https://github.com/gusye1234/nano-vectordb) 作为后端。
- 我们还内置了 [`hnswlib`](https://github.com/nmslib/hnswlib) 存储，查看这个[示例](./examples/using_hnsw_as_vectorDB.py)。
- 查看这个[示例](./examples/using_milvus_as_vectorDB.py)，实现 [`milvus-lite`](https://github.com/milvus-io/milvus-lite) 作为后端（在 Windows 上不可用）。
- `GraphRAG(.., vector_db_storage_cls=YOURS,...)`

**`base.BaseGraphStorage` 存储知识图谱**

- 默认使用 [`networkx`](https://github.com/networkx/networkx) 作为后端。
- 我们为图内置了 `Neo4jStorage`，查看这个[教程](./docs/use_neo4j_for_graphrag.md)。
- `GraphRAG(.., graph_storage_cls=YOURS,...)`

你可以参考 `nano_graphrag.base` 查看各组件的详细接口。
</details>



## 常见问题

查看 [FQA](./docs/FAQ.md)。

## 路线图

见 [ROADMAP.md](./docs/ROADMAP.md)。

## 贡献

`nano-graphrag` 欢迎任何形式的贡献。请在贡献前阅读[这篇文档](./docs/CONTRIBUTING.md)。



## 基准测试

- [英文基准测试](./docs/benchmark-en.md)
- [中文基准测试](./docs/benchmark-zh.md)
- [多跳 RAG 任务](https://github.com/yixuantt/MultiHop-RAG) 的[评估](./examples/benchmarks/eval_naive_graphrag_on_multi_hop.ipynb)笔记本



## 使用了 `nano-graphrag` 的项目

- [Medical Graph RAG](https://github.com/MedicineToken/Medical-Graph-RAG): 医疗数据的图 RAG
- [LightRAG](https://github.com/HKUDS/LightRAG): 简单快速的增强生成
- [fast-graphrag](https://github.com/circlemind-ai/fast-graphrag): 智能适应你的用例、数据和查询的 RAG
- [HiRAG](https://github.com/hhy-huang/HiRAG): 基于层次知识的增强生成

> 如果你的项目使用了 `nano-graphrag`，欢迎提交 PR，帮助更多人信任这个仓库❤️



## 问题

- `nano-graphrag` 没有实现 `GraphRAG` 的 `covariates` 特性。
- `nano-graphrag` 的全局检索实现与原版不同。原版采用类似 map-reduce 的方式将所有社区填充到上下文中，而 `nano-graphrag` 仅使用前 K 个重要且中心的社区（使用 `QueryParam.global_max_consider_community` 控制，默认 512 个社区）。


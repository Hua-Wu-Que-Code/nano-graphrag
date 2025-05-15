from nano_graphrag import GraphRAG, QueryParam
from examples.using_local_embedding_model import local_embedding  # 导入本地 embedding
from examples.using_deepseek_as_llm import deepseepk_model_if_cache  # 导入 deepseek LLM

# 初始化 GraphRAG，指定本地 embedding 和 deepseek LLM
rag = GraphRAG(
    working_dir="./dickens",
    enable_llm_cache=True,  # 启用 LLM 缓存，避免重复请求
    embedding_func=local_embedding,
    best_model_func=deepseepk_model_if_cache,
    cheap_model_func=deepseepk_model_if_cache,
)

# 读取文本       
with open("./zhuyuanzhang.txt") as f:
    rag.insert(f.read())

# 全局 graphrag 检索
# 检索范围：在整个知识库（所有插入的文本、所有实体、所有关系）中进行检索和推理。
# 适用场景：适合需要全局理解、综合分析的问题，比如“这本书的主题是什么？”、“全文有哪些主要人物？”等。
# 特点：速度较慢但信息全面，适合全局性问题。
print(rag.query("朱元璋父亲是谁？"))
#print("*"*100000)
# 局部 graphrag 检索（更好且更易扩展）
# 检索范围：先用向量检索找到与问题最相关的若干文本块（chunks），然后只在这些相关内容上做知识图谱推理和答案生成。
# 适用场景：适合针对某一细节、某一段落、某一实体的具体问题，比如“Bob在故事中做了什么？”、“第三章讲了什么？”等。
# 特点：速度更快，消耗更少，适合大规模数据和高并发场景，也更易于扩展和定制。
#print(rag.query("朱元璋他爸是谁？", param=QueryParam(mode="local")))
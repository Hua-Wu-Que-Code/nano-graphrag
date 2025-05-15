"""
本测试文件演示如何在 nano-graphrag 中结合 DeepSeek-Chat 作为 LLM（大语言模型）和本地 MiniLM-L6-v2 作为嵌入模型进行端到端测试。
"""
import os
import shutil
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer

# 导入 deepseek-chat 的 LLM 调用函数
from examples.using_deepseek_as_llm import deepseepk_model_if_cache

# 设置测试缓存目录
WORKING_DIR = "/Users/huawuque/Desktop/RAG实验/nano-graphrag/nano_graphrag_cache_local_embedding_TEST"

# 保证测试目录为空
if os.path.exists(WORKING_DIR):
    shutil.rmtree(WORKING_DIR)
os.makedirs(WORKING_DIR, exist_ok=True)

# 加载本地 MiniLM-L6-v2 模型作为 embedding
EMBED_MODEL = SentenceTransformer(
    "/Users/huawuque/Desktop/RAG实验/nano-graphrag/model/all-MiniLM-L6-v2",
    cache_folder=WORKING_DIR,
    device="cpu"
)

@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    """
    使用本地 MiniLM-L6-v2 对输入文本列表编码，返回归一化后的向量。
    """
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)

def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)

def test_insert_and_query():
    # 读取测试文本
    with open("./mock_data.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()

    # 初始化 GraphRAG，指定本地 embedding 和 deepseek-chat 作为 LLM
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        embedding_func=local_embedding,
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
    )
    # 插入文本并建立索引
    rag.insert(FAKE_TEXT)

    # 查询中文主题
    result = rag.query("请用中文总结这本书的主题", param=QueryParam(mode="local"))
    print(result)

if __name__ == "__main__":
    test_insert_and_query()
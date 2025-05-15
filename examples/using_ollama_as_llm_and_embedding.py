import os
import sys

# 将上级目录加入 sys.path，方便本地开发时导入 nano_graphrag 包
sys.path.append("..")
import logging
import ollama  # 本地大模型推理和 embedding 的 Python 客户端
import numpy as np
from nano_graphrag import GraphRAG, QueryParam  # 主类和查询参数
from nano_graphrag.base import BaseKVStorage    # KV 存储基类（用于缓存）
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs  # 工具函数

# 设置日志级别，nano-graphrag 的日志单独设置为 INFO，便于调试
logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# -------------------- Ollama LLM 和 embedding 配置 --------------------

# LLM 模型名称（需替换为你本地 Ollama 支持的模型名，如 "qwen2"、"llama3" 等）
MODEL = "qwen2.5:7b"

# embedding 模型名称及参数（需替换为你本地 Ollama 支持的 embedding 模型名）
EMBEDDING_MODEL = "nomic-embed-text:latest"
EMBEDDING_MODEL_DIM = 768
EMBEDDING_MODEL_MAX_TOKENS = 8192

# -------------------- Ollama LLM 异步调用函数（带缓存） --------------------
async def ollama_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    使用 Ollama 本地大模型进行对话，支持 system prompt、历史消息和缓存。
    如果传入 hashing_kv，则自动缓存和复用历史响应。
    """
    # 移除 Ollama 不支持的参数
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)

    ollama_client = ollama.AsyncClient()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 获取缓存对象（如果有）
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # 查询缓存
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    # 真正调用 Ollama 本地大模型
    response = await ollama_client.chat(model=MODEL, messages=messages, **kwargs)
    result = response["message"]["content"]

    # 写入缓存
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})
    return result

# -------------------- Ollama embedding 函数（本地向量化） --------------------
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL_DIM,
    max_token_size=EMBEDDING_MODEL_MAX_TOKENS,
)
async def ollama_embedding(texts: list[str]) -> np.ndarray:
    """
    使用 Ollama 本地 embedding 模型对文本列表编码，返回向量。
    """
    embed_text = []
    for text in texts:
        data = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        embed_text.append(data["embedding"])
    return embed_text

import os
import logging
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash

# 设置日志级别，方便调试和查看 nano-graphrag 的日志信息
logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# DeepSeek API 的密钥和模型名称
DEEPSEEK_API_KEY = "sk-41d81101b13c4bacb02ae5c06ced504a"
MODEL = "deepseek-chat"

# 异步的 DeepSeek LLM 调用函数，支持缓存
async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # 创建 DeepSeek 的异步 OpenAI 客户端
    openai_async_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"
    )
    messages = []
    # 如果有 system_prompt，则加入消息列表
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 获取缓存（如果有）
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)  # 加入历史消息
    messages.append({"role": "user", "content": prompt})  # 当前用户输入
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            # 命中缓存，直接返回
            return if_cache_return["return"]

    # 调用 DeepSeek 的 chat.completions 接口获取回复
    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # 将结果写入缓存（如果有）
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    return response.choices[0].message.content

# 工具函数：如果文件存在则删除（用于清理缓存文件）
def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)
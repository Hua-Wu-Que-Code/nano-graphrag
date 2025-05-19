import asyncio  # Python 标准库，支持异步编程（如 async/await、事件循环等）
import os       # Python 标准库，进行文件和目录操作（如创建目录、判断文件是否存在等）
import tiktoken  # OpenAI 的分词库，用于按 token 分割文本，支持多种 LLM 模型的分词规则

from dataclasses import asdict, dataclass, field  # Python 标准库，简化数据类的定义和实例转字典
from datetime import datetime  # Python 标准库，处理日期和时间
from functools import partial  # Python 标准库，生成带部分参数的函数
from typing import Callable, Dict, List, Optional, Type, Union, cast  # Python 标准库，类型注解和类型转换

# 导入 nano-graphrag 内部各类 LLM、Embedding、存储、操作与工具函数

# _llm.py：封装了多种大模型（OpenAI、Azure、Bedrock等）的异步调用接口和 embedding 接口
from ._llm import (
    amazon_bedrock_embedding,                # Amazon Bedrock 的 embedding 接口
    create_amazon_bedrock_complete_function, # 动态生成 Bedrock 对话函数的工厂
    gpt_4o_complete,                         # OpenAI gpt-4o 快捷对话接口
    gpt_4o_mini_complete,                    # OpenAI gpt-4o-mini 快捷对话接口
    openai_embedding,                        # OpenAI embedding 接口
    azure_gpt_4o_complete,                   # Azure OpenAI gpt-4o 快捷对话接口
    azure_openai_embedding,                  # Azure OpenAI embedding 接口
    azure_gpt_4o_mini_complete,              # Azure OpenAI gpt-4o-mini 快捷对话接口
)

# _op.py：核心操作函数，包括分块、实体抽取、社区报告、检索等
from ._op import (
    chunking_by_token_size,      # 按 token 数分块的分块函数
    extract_entities,            # 实体抽取主函数
    generate_community_report,   # 社区分析与报告生成函数
    get_chunks,                  # 文本分块主流程
    local_query,                 # 局部（local）检索主流程
    global_query,                # 全局（global）检索主流程
    naive_query,                 # naive（仅向量）检索主流程
)

# _storage.py：存储后端实现，包括 KV 存储、向量数据库、图数据库
from ._storage import (
    JsonKVStorage,           # 基于 JSON 文件的 KV 存储实现
    NanoVectorDBStorage,     # nano-vectordb 的向量存储实现
    NetworkXStorage,         # 基于 networkx 的图数据库实现
)

# _utils.py：工具函数和类型定义
from ._utils import (
    EmbeddingFunc,               # 嵌入函数类型定义
    compute_mdhash_id,           # 计算文本哈希 id 的工具
    limit_async_func_call,        # 限制异步函数最大并发数的装饰器
    convert_response_to_json,     # LLM 响应转 json 的工具
    always_get_an_event_loop,     # 获取或新建 asyncio 事件循环
    logger,                      # 项目统一日志对象
)

# base.py：基础类型和接口定义
from .base import (
    BaseGraphStorage,        # 图数据库存储基类
    BaseKVStorage,           # KV 存储基类
    BaseVectorStorage,       # 向量存储基类
    StorageNameSpace,        # 存储命名空间接口
    QueryParam,              # 检索参数数据类
)

@dataclass
# @dataclass是Python 3.7+ 标准库 dataclasses 提供的一个装饰器，用于简化数据类的定义。
# 自动为类生成常用的魔法方法，如 __init__()、__repr__()、__eq__() 等。
# 让你只需声明属性，Python 会自动帮你实现构造函数和属性赋值。
class GraphRAG:
    """
    GraphRAG 主类，负责管理文本插入、分块、实体抽取、知识图谱构建、向量存储、LLM 调用等全流程。
    支持多种后端、异步处理、缓存、增量插入、全局/局部/naive 检索等。
    """

    # 工作目录，所有缓存、索引、图谱等文件都存放于此
    # default_factory 的值是一个可执行的函数
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    # 是否启用局部（local）图谱检索
    enable_local: bool = True
    # 是否启用 naive RAG（仅向量检索，不用图谱）
    enable_naive_rag: bool = False

    # 文本分块相关参数
    ## Callable 是一个类型注解：用于表示一个可调用的对象，如函数
    chunk_func: Callable = chunking_by_token_size  # 分块函数,按照 token 进行划分
    chunk_token_size: int = 1200                   # 每块最大 token 数
    chunk_overlap_token_size: int = 100            # 块之间重叠 token 数
    tiktoken_model_name: str = "gpt-4o"            # 用于分词的模型名

    # 实体抽取相关参数
    entity_extract_max_gleaning: int = 1           # 实体抽取最大遍历次数
    entity_summary_to_max_tokens: int = 500        # 实体摘要最大 token 数

    # 图聚类相关参数
    graph_cluster_algorithm: str = "leiden"        # 图聚类算法
    max_graph_cluster_size: int = 10               # 最大聚类数
    graph_cluster_seed: int = 0xDEADBEEF           # 聚类随机种子

    node_embedding_algorithm: str = "node2vec"  # 节点嵌入算法名称，当前默认使用 node2vec
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,    # 嵌入向量的维度，决定每个节点的特征向量长度
            "num_walks": 10,       # 每个节点生成的随机游走（walk）次数，用于采样邻居信息
            "walk_length": 40,     # 每次随机游走的步数，决定采样的路径长度
            "num_walks": 10,       # （重复定义，建议去掉一处）每个节点的随机游走次数
            "window_size": 2,      # skip-gram 的窗口大小，影响上下文节点的范围
            "iterations": 3,       # skip-gram 训练的迭代次数
            "random_seed": 3,      # 随机种子，保证实验可复现
        }
    )

    # 社区报告 LLM 调用参数
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # 文本嵌入相关参数
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)  # 嵌入函数
    embedding_batch_num: int = 32                  # 嵌入批量大小
    embedding_func_max_async: int = 16             # 嵌入异步并发数
    query_better_than_threshold: float = 0.2       # 检索相关性阈值

    # LLM 相关参数
    using_azure_openai: bool = False               # 是否用 Azure OpenAI
    using_amazon_bedrock: bool = False             # 是否用 Amazon Bedrock
    best_model_id: str = "us.anthropic.claude-3-sonnet-20240229-v1:0"  # Bedrock 主力模型
    cheap_model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"  # Bedrock 经济模型
    best_model_func: callable = gpt_4o_complete    # 主力 LLM 函数
    best_model_max_token_size: int = 32768         # 主力 LLM 最大 token
    best_model_max_async: int = 16                 # 主力 LLM 并发数
    cheap_model_func: callable = gpt_4o_mini_complete  # 经济 LLM 函数
    cheap_model_max_token_size: int = 32768        # 经济 LLM 最大 token
    cheap_model_max_async: int = 16                # 经济 LLM 并发数

    # 实体抽取函数
    entity_extraction_func: callable = extract_entities

    # 存储后端相关参数
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage  # KV 存储类
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage    # 向量存储类
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)        # 向量存储参数
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage             # 图存储类
    enable_llm_cache: bool = True                                          # 是否启用 LLM 缓存

    # 扩展参数
    always_create_working_dir: bool = True         # 是否自动创建工作目录
    addon_params: dict = field(default_factory=dict)  # 额外参数
    convert_response_to_json_func: callable = convert_response_to_json      # LLM 响应转 json

    def __post_init__(self):
        """
        初始化 GraphRAG 实例，自动切换 LLM/Embedding 后端，创建存储目录和各类存储实例。
        """
        # 打印当前配置（debug）
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

        # 根据配置自动切换 Azure OpenAI
        if self.using_azure_openai:
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )

        # 根据配置自动切换 Amazon Bedrock
        if self.using_amazon_bedrock:
            self.best_model_func = create_amazon_bedrock_complete_function(self.best_model_id)
            self.cheap_model_func = create_amazon_bedrock_complete_function(self.cheap_model_id)
            self.embedding_func = amazon_bedrock_embedding
            logger.info(
                "Switched the default openai funcs to Amazon Bedrock"
            )

        # 自动创建工作目录
        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # 初始化各类存储实例（KV、向量、图谱等）
        
        # 全文档存储（原始文本，key 为 doc id）
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )
        # 文本分块存储（每个 chunk，key 为 chunk id）
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )
        # LLM 响应缓存（可选，提升推理效率，避免重复请求）
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )
        # 社区报告存储（每个社区的分析报告）
        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        # 知识图谱存储（chunk-实体-关系图）
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        # 包装 embedding_func，限制最大异步并发数
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        # 初始化实体向量数据库
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        # 初始化 chunk 向量数据库（仅 naive RAG 用）
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        # 包装 LLM 函数，自动注入缓存和并发控制
        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

    def insert(self, string_or_strings):
        """
        同步接口：插入文本（或文本列表），自动分块、嵌入、实体抽取、图谱构建等。
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        """
        同步接口：对知识库进行检索与问答，支持全局/局部/naive 模式。
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        """
        异步接口：根据 param.mode 选择不同的检索模式（local/global/naive）。
        """
        if param.mode == "local" and not self.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        if param.mode == "local":
            # 局部检索：先向量召回相关 chunk，再做图谱推理
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "global":
            # 全局检索：全量图谱推理
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "naive":
            # naive 检索：仅向量召回，不用图谱
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def ainsert(self, string_or_strings):
        """
        异步接口：插入文本，自动分块、嵌入、实体抽取、图谱构建等。
        支持增量插入（已存在的内容自动跳过）。
        """
        await self._insert_start()
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # ---------- new docs
            # 计算每个文档的哈希 id，构建新文档字典
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            # 过滤已存在的文档，只保留新增部分
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning(f"All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- chunking
            # 对新文档分块
            inserting_chunks = get_chunks(
                new_docs=new_docs,
                chunk_func=self.chunk_func,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
            )
            # 过滤已存在的 chunk，只保留新增部分
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # TODO: 目前社区分析不支持增量，直接清空重建
            await self.community_reports.drop()

            # ---------- extract/summary entity and upsert to graph
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await self.entity_extraction_func(
                inserting_chunks,
                knwoledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=asdict(self),
                using_amazon_bedrock=self.using_amazon_bedrock,
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- update clusterings of graph
            logger.info("[Community Report]...")
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            await generate_community_report(
                self.community_reports, self.chunk_entity_relation_graph, asdict(self)
            )

            # ---------- commit upsertings and indexing
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_start(self):
        """
        _insert_start 就是“插入前的准备/加锁钩子”，保证数据插入时各类存储后端都已准备好，防止并发冲突和数据不一致。
        在插入新数据（如文档、分块、实体等）到各类存储前，提前对相关存储做“准备工作”，
        比如加锁、索引准备等，确保数据插入过程的安全性和一致性。

        为什么需要这个函数？
        在多线程/多进程环境下，可能会有多个线程/进程同时尝试插入数据到同一个存储实例。
        这可能导致数据不一致、索引错误等问题。
        通过在插入前进行一些准备工作，可以确保数据插入的安全性和一致性。
        例如：
        1. 锁定存储实例，防止其他线程/进程同时插入数据。
        2. 准备索引，确保索引结构是最新的。
        3. 清理旧数据，确保存储实例的状态是干净的。
        4. 其他需要在插入前进行的操作。
        

        具体做了什么？
        1. 遍历所有需要在插入前进行处理的存储实例（目前只包含知识图谱存储）。
        2. 对每个存储实例调用 index_start_callback() 方法。
        3. index_start_callback() 方法会执行一些准备工作，比如加锁、索引准备等。
        4. 使用 asyncio.gather() 方法并发执行所有存储实例的 index_start_callback() 方法。
        5. 等待所有存储实例的 index_start_callback() 方法执行完成。
        
        为什么是异步的？
        存储后端可能是远程数据库、分布式存储等，准备动作可能很耗时，用异步可以提升效率。
        """
        tasks = []
        # 遍历需要在插入前进行处理的存储实例（目前只包含知识图谱存储）
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        """
        插入流程结束后的回调（如索引解锁、持久化等），支持多存储后端扩展。
        """
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        """
        检索流程结束后的回调（如缓存持久化等）。
        """
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
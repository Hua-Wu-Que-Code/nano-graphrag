from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar, List

import numpy as np

from ._utils import EmbeddingFunc

# ------------------- 查询参数定义 -------------------

@dataclass
class QueryParam:
    # 检索模式，可选 "local"（局部图谱）、"global"（全局社区）、"naive"（仅向量）
    mode: Literal["local", "global", "naive"] = "global"
    only_need_context: bool = False  # 是否只需要上下文，不需要最终答案
    response_type: str = "Multiple Paragraphs"  # LLM 输出格式
    level: int = 2  # 检索深度或社区层级
    top_k: int = 20  # 检索返回的候选数量

    # naive 检索相关参数
    naive_max_token_for_text_unit = 12000

    # local 检索相关参数
    local_max_token_for_text_unit: int = 4000  # 单个文本块最大 token 数
    local_max_token_for_local_context: int = 4800  # 局部上下文最大 token 数
    local_max_token_for_community_report: int = 3200  # 社区报告最大 token 数
    local_community_single_one: bool = False  # 是否只选一个社区

    # global 检索相关参数
    global_min_community_rating: float = 0  # 社区评分下限
    global_max_consider_community: float = 512  # 最大考虑社区数
    global_max_token_for_community_report: int = 16384  # 全局社区报告最大 token 数
    global_special_community_map_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )  # LLM 调用参数

# ------------------- 类型定义 -------------------

# 文本分块的结构定义
TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

# 单个社区的结构定义
SingleCommunitySchema = TypedDict(
    "SingleCommunitySchema",
    {
        "level": int,
        "title": str,
        "edges": list[list[str, str]],
        "nodes": list[str],
        "chunk_ids": list[str],
        "occurrence": float,
        "sub_communities": list[str],
    },
)

# 带报告的社区结构定义
class CommunitySchema(SingleCommunitySchema):
    report_string: str
    report_json: dict

# 泛型类型变量
T = TypeVar("T")

# ------------------- 存储命名空间基类 -------------------

@dataclass
class StorageNameSpace:
    namespace: str  # 存储空间名（如 "full_docs"、"text_chunks" 等）
    global_config: dict  # 全局配置参数

    async def index_start_callback(self):
        """插入流程前的回调（如加锁、索引准备等），可被子类重写"""
        pass

    async def index_done_callback(self):
        """插入流程后的回调（如解锁、持久化等），可被子类重写"""
        pass

    async def query_done_callback(self):
        """检索流程后的回调（如缓存持久化等），可被子类重写"""
        pass

# ------------------- 向量存储基类 -------------------

@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc  # 嵌入函数
    meta_fields: set = field(default_factory=set)  # 元信息字段

    async def query(self, query: str, top_k: int) -> list[dict]:
        """向量检索接口，需子类实现"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """
        插入或更新数据，使用 'content' 字段做 embedding，key 作为 id。
        如果 embedding_func 为 None，则用 value 的 'embedding' 字段。
        """
        raise NotImplementedError

# ------------------- KV 存储基类 -------------------

@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    async def all_keys(self) -> list[str]:
        """返回所有 key，需子类实现"""
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        """根据 id 获取数据，需子类实现"""
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        """批量获取数据，支持字段过滤，需子类实现"""
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """返回 data 中未存储的 key，需子类实现"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        """批量插入或更新，需子类实现"""
        raise NotImplementedError

    async def drop(self):
        """清空存储，需子类实现"""
        raise NotImplementedError

# ------------------- 图存储基类 -------------------

@dataclass
class BaseGraphStorage(StorageNameSpace):
    async def has_node(self, node_id: str) -> bool:
        """判断节点是否存在"""
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """判断边是否存在"""
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        """获取节点度数"""
        raise NotImplementedError
    
    async def node_degrees_batch(self, node_ids: List[str]) -> List[str]:
        """批量获取节点度数"""
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """获取边的度数"""
        raise NotImplementedError

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        """批量获取边的度数"""
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        """获取节点数据"""
        raise NotImplementedError

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, Union[dict, None]]:
        """批量获取节点数据"""
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """获取边数据"""
        raise NotImplementedError

    async def get_edges_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> list[Union[dict, None]]:
        """批量获取边数据"""
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        """获取节点的所有边"""
        raise NotImplementedError

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> list[list[tuple[str, str]]]:
        """批量获取节点的所有边"""
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        """插入或更新节点"""
        raise NotImplementedError

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, str]]]):
        """批量插入或更新节点"""
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        """插入或更新边"""
        raise NotImplementedError

    async def upsert_edges_batch(
        self, edges_data: list[tuple[str, str, dict[str, str]]]
    ):
        """批量插入或更新边"""
        raise NotImplementedError

    async def clustering(self, algorithm: str):
        """对图进行聚类分析"""
        raise NotImplementedError

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """返回社区结构及报告"""
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        """对节点做嵌入（nano-graphrag 默认未用）"""
        raise NotImplementedError("Node embedding is not used in nano-graphrag.")
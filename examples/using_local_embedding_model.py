import sys
sys.path.append("..")  # 将上级目录加入模块搜索路径，方便本地开发和导入 nano_graphrag 包
import logging
import numpy as np
from nano_graphrag import GraphRAG, QueryParam  # 导入主类和查询参数
from nano_graphrag._utils import wrap_embedding_func_with_attrs  # 导入包装 embedding 函数的工具
from sentence_transformers import SentenceTransformer  # 导入本地 embedding 模型

logging.basicConfig(level=logging.WARNING)  # 设置日志输出级别为 WARNING
logging.getLogger("nano-graphrag").setLevel(logging.INFO)  # nano-graphrag 的日志单独设置为 INFO，便于调试

# 加载本地的 SentenceTransformer 模型（MiniLM），用于生成文本向量
EMBED_MODEL = SentenceTransformer(
    "/Users/huawuque/Desktop/RAG实验/nano-graphrag/model/all-MiniLM-L6-v2",  # 选择的模型名称
    device="cpu"                               # 使用 CPU 推理
)

# 用装饰器包装 embedding 函数，声明向量维度和最大 token 长度，便于 nano-graphrag 识别
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),  # 获取模型的输出向量维度
    max_token_size=EMBED_MODEL.max_seq_length,                    # 获取模型支持的最大序列长度
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    """
    使用本地 SentenceTransformer 模型对输入文本列表进行编码，返回归一化后的向量。
    这是一个异步函数，符合 nano-graphrag 的接口要求。
    """
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)
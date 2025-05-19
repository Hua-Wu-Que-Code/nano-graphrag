import os
from dataclasses import dataclass

from .._utils import load_json, logger, write_json
from ..base import (
    BaseKVStorage, # KV 存储基类
)

@dataclass
class JsonKVStorage(BaseKVStorage):
    """
    基于本地 JSON 文件的简单 KV 存储实现，继承自 BaseKVStorage。
    每个 namespace 对应一个独立的 JSON 文件，所有数据均保存在本地磁盘。
    """

    def __post_init__(self):
        # 获取工作目录路径
        working_dir = self.global_config["working_dir"]
        # 构造当前 namespace 的 JSON 文件路径
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        # 加载已有数据，如果文件不存在则初始化为空字典
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        """
        返回所有已存储的 key 列表。
        """
        return list(self._data.keys())

    async def index_done_callback(self):
        """
        持久化当前内存中的数据到 JSON 文件（通常在插入流程结束后调用）。
        """
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        """
        根据 id 获取对应的数据（若不存在则返回 None）。
        """
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        """
        批量获取多个 id 对应的数据。
        如果 fields 不为 None，则只返回指定字段的子集。
        """
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        """
        过滤出 data 中尚未存储的 key（即新 key）。
        """
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        """
        批量插入或更新数据（以字典形式合并到现有数据）。
        """
        self._data.update(data)

    async def drop(self):
        """
        清空当前 namespace 下的所有数据（仅内存，未立即写盘）。
        """
        self._data = {}
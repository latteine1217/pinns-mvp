"""
DDP 工具函數

提供資料分割與損失同步的輔助方法。
"""

import logging
from typing import Dict, Any

import torch
import torch.distributed as dist


def _dist_ready() -> bool:
    """判斷 torch.distributed 是否可用且已初始化"""
    return dist.is_available() and dist.is_initialized()


def _resolve_rank_world_size(rank: int | None, world_size: int | None) -> tuple[int, int]:
    """取得 rank 與 world_size（優先使用傳入值）"""
    if rank is None or world_size is None:
        if not _dist_ready():
            return 0, 1
        rank = dist.get_rank() if rank is None else rank
        world_size = dist.get_world_size() if world_size is None else world_size
    return int(rank), int(world_size)


def _split_tensor_by_rank(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """沿 batch 維度分割張量"""
    if tensor.dim() == 0:
        return tensor
    if tensor.shape[0] == 0:
        return tensor
    if tensor.shape[0] < world_size:
        return tensor
    chunks = torch.chunk(tensor, world_size, dim=0)
    if rank < len(chunks):
        return chunks[rank]
    return tensor[:0]


def split_data_by_rank(
    data_dict: Dict[str, Any],
    rank: int | None = None,
    world_size: int | None = None
) -> Dict[str, Any]:
    """
    按 rank 分割訓練數據（支援巢狀字典）。

    策略：
    - 所有 Tensor 沿 dim=0 均分
    - 樣本數 < world_size 時保持完整數據
    - 非 Tensor 或配置資訊保留原樣
    """
    rank, world_size = _resolve_rank_world_size(rank, world_size)

    def split_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return _split_tensor_by_rank(value, rank, world_size)
        if isinstance(value, dict):
            return {key: split_value(sub_value) for key, sub_value in value.items()}
        return value

    return {key: split_value(value) for key, value in data_dict.items()}


def reduce_loss_dict(
    loss_dict: Dict[str, Any],
    average: bool = True
) -> Dict[str, Any]:
    """
    在所有 rank 間同步損失（All-Reduce）。

    Args:
        loss_dict: 損失字典
        average: 是否取平均

    Returns:
        同步後的損失字典
    """
    if not _dist_ready():
        return loss_dict

    world_size = dist.get_world_size()
    reduced: Dict[str, Any] = {}

    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            reduced_value = value.detach().clone()
            dist.all_reduce(reduced_value, op=dist.ReduceOp.SUM)
            if average:
                reduced_value = reduced_value / world_size
            reduced[key] = reduced_value
        else:
            reduced[key] = value

    return reduced


def verify_data_split(data_dict: Dict[str, Any], rank: int | None = None, world_size: int | None = None) -> None:
    """輸出分割後的資料形狀（診斷用）"""
    rank, world_size = _resolve_rank_world_size(rank, world_size)

    def log_value(prefix: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            logging.info(f"[DDP] Rank {rank}/{world_size} - {prefix}: shape={tuple(value.shape)}")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                log_value(f"{prefix}.{sub_key}", sub_value)

    for key, value in data_dict.items():
        log_value(key, value)

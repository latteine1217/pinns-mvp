"""
檢查點管理模組

提供模型訓練過程中的檢查點保存與載入功能。
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    loss: float,
    config: Dict[str, Any],
    checkpoint_dir: str = "checkpoints"
) -> str:
    """
    保存模型檢查點。
    
    自動保存兩份檢查點：
    1. 帶 epoch 編號的檢查點（用於歷史記錄）
    2. 最新檢查點（方便快速恢復訓練）
    
    Parameters
    ----------
    model : nn.Module
        要保存的模型。
    optimizer : Optimizer
        優化器（會保存其狀態以便恢復訓練）。
    epoch : int
        當前訓練 epoch 數。
    loss : float
        當前損失值。
    config : Dict[str, Any]
        完整的訓練配置字典（需包含 'experiment.name' 欄位）。
    checkpoint_dir : str, optional
        檢查點保存目錄，預設為 "checkpoints"。
    
    Returns
    -------
    str
        保存的檢查點檔案路徑（帶 epoch 編號的版本）。
    
    Raises
    ------
    KeyError
        若 config 中缺少 'experiment.name' 欄位。
    
    Examples
    --------
    >>> model = FourierMLP(...)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> config = {'experiment': {'name': 'test_run'}}
    >>> path = save_checkpoint(model, optimizer, epoch=100, loss=0.01, config=config)
    >>> print(path)
    'checkpoints/test_run_epoch_100.pth'
    
    Notes
    -----
    檢查點包含以下信息：
    - epoch: 訓練輪數
    - model_state_dict: 模型參數
    - optimizer_state_dict: 優化器狀態
    - loss: 損失值
    - config: 完整配置
    """
    # 確保目錄存在
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # 構建檢查點字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    # 獲取實驗名稱
    try:
        experiment_name = config['experiment']['name']
    except KeyError as e:
        raise KeyError("config 必須包含 'experiment.name' 欄位") from e
    
    # 保存帶 epoch 編號的檢查點
    checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"檢查點已保存: {checkpoint_path}")
    
    # 保存最新檢查點（覆蓋式）
    latest_path = os.path.join(checkpoint_dir, f"{experiment_name}_latest.pth")
    torch.save(checkpoint, latest_path)
    logger.debug(f"最新檢查點已更新: {latest_path}")
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: str = "cpu"
) -> Tuple[int, float, Optional[Dict[str, Any]]]:
    """
    載入模型檢查點。
    
    Parameters
    ----------
    checkpoint_path : str
        檢查點檔案路徑。
    model : nn.Module
        要載入參數的模型（會 in-place 修改）。
    optimizer : Optimizer, optional
        要載入狀態的優化器（若為 None 則不載入優化器狀態）。
    device : str, optional
        載入到的設備，預設為 "cpu"。
        可選值: "cpu", "cuda", "cuda:0", "mps" 等。
    
    Returns
    -------
    epoch : int
        檢查點保存時的 epoch 數。
    loss : float
        檢查點保存時的損失值。
    config : Dict[str, Any] or None
        訓練配置（若檢查點中不包含則返回 None）。
    
    Raises
    ------
    FileNotFoundError
        若檢查點檔案不存在。
    RuntimeError
        若模型參數載入失敗（例如架構不匹配）。
    
    Examples
    --------
    >>> model = FourierMLP(...)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> epoch, loss, config = load_checkpoint(
    ...     "checkpoints/test_run_latest.pth", 
    ...     model, 
    ...     optimizer,
    ...     device="cuda"
    ... )
    >>> print(f"從 epoch {epoch} 恢復訓練，損失: {loss:.6f}")
    
    Notes
    -----
    - 模型和優化器會被 in-place 修改
    - 若只需推理，可不傳入 optimizer
    - 使用 map_location 確保跨設備載入安全
    """
    # 檢查檔案是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"檢查點檔案不存在: {checkpoint_path}")
    
    # 載入檢查點
    logger.info(f"正在載入檢查點: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 載入模型參數
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("模型參數載入成功")
    except RuntimeError as e:
        logger.error(f"模型參數載入失敗: {e}")
        raise
    
    # 載入優化器狀態（可選）
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("優化器狀態載入成功")
        except Exception as e:
            logger.warning(f"優化器狀態載入失敗（將使用初始狀態）: {e}")
    
    # 提取元數據
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    config = checkpoint.get('config', None)
    
    logger.info(f"檢查點載入完成 - Epoch: {epoch}, Loss: {loss:.6f}")
    
    return epoch, loss, config

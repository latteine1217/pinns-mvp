#!/usr/bin/env python3
"""
訓練環境設置工具模組
提供日誌、隨機種子、設備選擇等基礎功能
"""

import logging
import random
import sys
from typing import Optional

import numpy as np
import torch


def setup_logging(
    level: str = "info",
    log_file: Optional[str] = "pinnx.log"
) -> logging.Logger:
    """
    設置日誌系統
    
    Args:
        level: 日誌級別 ('debug', 'info', 'warning', 'error', 'critical')
        log_file: 日誌檔案路徑，None 則不寫入檔案
        
    Returns:
        logging.Logger: 配置完成的 logger 實例
        
    Raises:
        ValueError: 當日誌級別無效時
        
    Example:
        >>> logger = setup_logging(level="debug", log_file="train.log")
        >>> logger.info("Training started")
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """
    設置隨機種子確保重現性
    
    Args:
        seed: 隨機種子值
        deterministic: 是否啟用完全確定性模式（可能降低效能）
        
    Note:
        - 設置 Python、NumPy、PyTorch 的隨機種子
        - deterministic=True 時會禁用 CUDA 的非確定性優化
        - 完全重現性需要單線程 + 固定硬體環境
        
    Example:
        >>> set_random_seed(42, deterministic=True)
        >>> # 現在所有隨機操作都是可重現的
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_name: str) -> torch.device:
    """
    獲取運算設備（自動選擇或手動指定）
    
    Args:
        device_name: 設備名稱
            - "auto": 自動選擇最佳可用設備（CUDA > MPS > CPU）
            - "cuda": 使用 NVIDIA GPU
            - "mps": 使用 Apple Metal Performance Shaders
            - "cpu": 使用 CPU
            
    Returns:
        torch.device: PyTorch 設備物件
        
    Example:
        >>> device = get_device("auto")
        >>> model = model.to(device)
    """
    if device_name == "auto":
        # 自動選擇最佳可用設備
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"Auto-selected CUDA: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Auto-selected Apple Metal Performance Shaders")
        else:
            device = torch.device("cpu")
            logging.info("Auto-selected CPU (no GPU available)")
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple Metal Performance Shaders")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    
    return device


__all__ = ['setup_logging', 'set_random_seed', 'get_device']

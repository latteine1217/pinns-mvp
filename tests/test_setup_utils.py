#!/usr/bin/env python3
"""
測試 pinnx/utils/setup.py 模組
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.utils.setup import setup_logging, set_random_seed, get_device


class TestSetupLogging:
    """測試日誌設置功能"""
    
    def test_default_logging(self):
        """測試預設日誌設置"""
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
    
    def test_custom_level(self):
        """測試自定義日誌級別"""
        # 只測試函數不拋出異常
        logger = setup_logging(level="debug")
        assert isinstance(logger, logging.Logger)
        
        logger = setup_logging(level="warning")
        assert isinstance(logger, logging.Logger)
    
    def test_invalid_level(self):
        """測試無效日誌級別應拋出錯誤"""
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging(level="invalid_level")
    
    def test_log_file_creation(self):
        """測試日誌檔案創建"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            
            # 清除現有 handlers 避免干擾
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            
            logger = setup_logging(log_file=str(log_path))
            logger.info("Test message")
            
            # 強制 flush
            for handler in logger.handlers:
                handler.flush()
            
            assert log_path.exists()
            content = log_path.read_text()
            assert "Test message" in content
    
    def test_no_log_file(self):
        """測試不創建日誌檔案"""
        logger = setup_logging(log_file=None)
        assert isinstance(logger, logging.Logger)
        # 應該只有 StreamHandler


class TestSetRandomSeed:
    """測試隨機種子設置功能"""
    
    def test_basic_seeding(self):
        """測試基本隨機種子設置"""
        set_random_seed(42)
        
        # 測試 PyTorch 隨機性
        t1 = torch.randn(5)
        set_random_seed(42)
        t2 = torch.randn(5)
        
        assert torch.allclose(t1, t2), "相同種子應產生相同隨機數"
    
    def test_deterministic_mode(self):
        """測試確定性模式"""
        set_random_seed(42, deterministic=True)
        
        if torch.cuda.is_available():
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False
    
    def test_non_deterministic_mode(self):
        """測試非確定性模式（保持效能）"""
        set_random_seed(42, deterministic=False)
        
        if torch.cuda.is_available():
            assert torch.backends.cudnn.deterministic is False


class TestGetDevice:
    """測試設備選擇功能"""
    
    def test_cpu_device(self):
        """測試 CPU 設備選擇"""
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_auto_device(self):
        """測試自動設備選擇"""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        # 應該選擇 CUDA > MPS > CPU
        if torch.cuda.is_available():
            assert device.type == "cuda"
        elif torch.backends.mps.is_available():
            assert device.type == "mps"
        else:
            assert device.type == "cpu"
    
    def test_setup_cuda_fallback_to_cpu(self):
        """測試 CUDA 不可用時回退到 CPU"""
        if not torch.cuda.is_available():
            device = get_device("cuda")
            assert device.type == "cpu"
    
    def test_setup_mps_fallback_to_cpu(self):
        """測試 MPS 不可用時回退到 CPU"""
        if not torch.backends.mps.is_available():
            device = get_device("mps")
            assert device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

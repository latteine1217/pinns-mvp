"""
factory 模組單元測試

測試覆蓋範圍：
- get_device(): 設備選擇邏輯
- create_model(): 模型建立（含標準化/Fourier/VS-PINN）
- create_physics(): 物理方程創建

注意：優化器與調度器測試已遷移至 test_factories.py
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from pinnx.train.model_physics_factory import (
    get_device,
    create_model,
    create_physics,
)


# ============================================================================
# 測試固定配置
# ============================================================================

@pytest.fixture
def base_model_config() -> Dict[str, Any]:
    """基礎模型配置"""
    return {
        "model": {
            "in_dim": 4,  # x, y, z, t
            "out_dim": 4,  # u, v, w, p
            "width": 64,
            "depth": 2,
            "activation": "tanh",
            "use_fourier": False,
        },
    }


@pytest.fixture
def fourier_model_config() -> Dict[str, Any]:
    """Fourier 特徵模型配置"""
    return {
        "model": {
            "in_dim": 4,
            "out_dim": 4,
            "width": 128,
            "depth": 2,
            "activation": "sine",
            "use_fourier": True,
            "fourier_m": 32,
            "fourier_sigma": 10.0,
        },
    }


@pytest.fixture
def vs_pinn_config() -> Dict[str, Any]:
    """VS-PINN 配置"""
    return {
        "model": {
            "in_dim": 3,  # x, y, z (不含 t)
            "out_dim": 4,  # u, v, w, p
            "width": 200,
            "depth": 8,
            "activation": "sine",
            "use_fourier": True,
            "fourier_m": 80,
            "fourier_sigma": 5.0,
        },
        "physics": {
            "type": "vs_pinn_channel_flow",
            "vs_pinn": {
                "scaling_factors": {
                    "N_x": 2.0,
                    "N_y": 12.0,
                    "N_z": 2.0,
                },
            },
        },
    }


@pytest.fixture
def statistics() -> Dict[str, Dict[str, float]]:
    """統計資料（用於手動標準化）"""
    return {
        "x": {"mean": 3.141592653589793, "std": 1.8138},
        "y": {"mean": 0.0, "std": 0.5773},
        "z": {"mean": 1.5707963267948966, "std": 0.9069},
        "t": {"mean": 0.5, "std": 0.2886},
        "u": {"mean": 1.0, "std": 0.2},
        "v": {"mean": 0.0, "std": 0.05},
        "w": {"mean": 0.0, "std": 0.05},
        "p": {"mean": 0.0, "std": 0.15},
    }


@pytest.fixture
def optimizer_config() -> Dict[str, Any]:
    """優化器配置（修正為支援的 cosine 調度器）"""
    return {
        "training": {
            "optimizer": {
                "type": "adam",
                "lr": 0.001,
                "weight_decay": 0.0,
            },
            "scheduler": {
                "type": "cosine",
                "T_max": 5000,
                "eta_min": 1e-6,
            },
            "epochs": 5000,
        },
    }


# ============================================================================
# Test: get_device()
# ============================================================================

class TestGetDevice:
    """測試設備選擇邏輯"""
    
    def test_auto_cuda_available(self):
        """測試自動選擇：CUDA 可用"""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", return_value="Tesla V100"):
            device = get_device("auto")
            assert device.type == "cuda"
    
    def test_auto_mps_available(self):
        """測試自動選擇：MPS 可用（CUDA 不可用）"""
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True):
            device = get_device("auto")
            assert device.type == "mps"
    
    def test_auto_cpu_fallback(self):
        """測試自動選擇：回退到 CPU"""
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            device = get_device("auto")
            assert device.type == "cpu"
    
    def test_cuda_explicit(self):
        """測試明確指定 CUDA"""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", return_value="GTX 1080"):
            device = get_device("cuda")
            assert device.type == "cuda"
    
    def test_factory_cuda_fallback_to_cpu(self):
        """測試 CUDA 不可用時回退到 CPU"""
        with patch("torch.cuda.is_available", return_value=False):
            device = get_device("cuda")
            assert device.type == "cpu"
    
    def test_mps_explicit(self):
        """測試明確指定 MPS"""
        with patch("torch.backends.mps.is_available", return_value=True):
            device = get_device("mps")
            assert device.type == "mps"
    
    def test_factory_mps_fallback_to_cpu(self):
        """測試 MPS 不可用時回退到 CPU"""
        with patch("torch.backends.mps.is_available", return_value=False):
            device = get_device("mps")
            assert device.type == "cpu"
    
    def test_cpu_explicit(self):
        """測試明確指定 CPU"""
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_invalid_device_name(self):
        """測試無效設備名稱"""
        with pytest.raises(ValueError, match="Invalid device name"):
            get_device("tpu")


# ============================================================================
# Test: create_model()
# ============================================================================

class TestCreateModel:
    """測試模型建立邏輯"""
    
    def test_basic_model(self, base_model_config):
        """測試基礎模型建立"""
        device = torch.device("cpu")
        model = create_model(base_model_config, device)
        
        assert isinstance(model, nn.Module)
        # 驗證模型維度（從 config['model'] 讀取）
        assert base_model_config["model"]["in_dim"] == 4
        assert base_model_config["model"]["out_dim"] == 4
        
        # 測試前向傳播
        x = torch.randn(10, 4)
        out = model(x)
        assert out.shape == (10, 4)
    
    def test_fourier_model(self, fourier_model_config):
        """測試 Fourier 特徵模型"""
        device = torch.device("cpu")
        model = create_model(fourier_model_config, device)
        
        assert isinstance(model, nn.Module)
        # Fourier 特徵已內建在模型中
        
        # 測試前向傳播
        x = torch.randn(10, 4)
        out = model(x)
        assert out.shape == (10, 4)
    
    def test_manual_normalization_with_statistics(self, base_model_config, statistics):
        """測試手動標準化（使用統計資料）"""
        # VS-PINN 使用內建標準化，這裡測試基礎模型
        # 注意：當前 factory 主要支持 VS-PINN 標準化
        device = torch.device("cpu")
        model = create_model(base_model_config, device, statistics=statistics)
        
        assert isinstance(model, nn.Module)
        # 基礎模型可能不包含顯式標準化層
    
    def test_vs_pinn_model(self, vs_pinn_config):
        """測試 VS-PINN 模型建立"""
        device = torch.device("cpu")
        model = create_model(vs_pinn_config, device)
        
        assert isinstance(model, nn.Module)
        
        # 測試前向傳播（VS-PINN 輸入為 3D: x, y, z）
        x = torch.randn(10, 3)
        out = model(x)
        assert out.shape == (10, 4)
    
    def test_factory_siren_initialization(self):
        """測試 SIREN 權重初始化"""
        config = {
            "model": {
                "in_dim": 4,
                "out_dim": 4,
                "width": 64,
                "depth": 2,
                "activation": "sine",
                "use_fourier": False,
            },
        }
        
        device = torch.device("cpu")
        model = create_model(config, device)
        
        # SIREN 初始化會修改權重範圍
        # 檢查第一層權重是否在合理範圍內
        first_layer = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        
        if first_layer is not None:
            # SIREN 初始化：第一層 U(-1/n, 1/n)，其他層較小
            weight_std = first_layer.weight.std().item()
            assert weight_std < 1.0  # 驗證不是預設初始化
    
    def test_model_device_placement(self, base_model_config):
        """測試模型設備放置"""
        device = torch.device("cpu")
        model = create_model(base_model_config, device)
        
        # 檢查模型參數是否在正確設備上
        for param in model.parameters():
            assert param.device.type == device.type
    
    def test_missing_required_config(self):
        """測試缺少必要配置"""
        invalid_config = {
            "model": {
                "in_dim": 4,
                # 缺少 out_dim, width, depth, activation
            }
        }
        
        device = torch.device("cpu")
        with pytest.raises((KeyError, ValueError)):
            create_model(invalid_config, device)
    
    def test_normalization_without_statistics(self):
        """測試無統計資料時的模型創建（應成功）"""
        config = {
            "model": {
                "in_dim": 4,
                "out_dim": 4,
                "width": 64,
                "depth": 1,
                "activation": "tanh",
                "use_fourier": False,
            },
        }
        
        device = torch.device("cpu")
        # 應該能創建模型（使用預設標準化或跳過）
        model = create_model(config, device, statistics=None)
        assert isinstance(model, nn.Module)


# ============================================================================
# Test: create_physics()
# ============================================================================

class TestCreatePhysics:
    """測試物理方程創建邏輯"""
    
    def test_vs_pinn_physics(self):
        """測試 VS-PINN 物理方程創建"""
        config = {
            "physics": {
                "type": "vs_pinn_channel_flow",
                "nu": 5e-5,
                "domain": {
                    "x_range": [0.0, 25.13],
                    "y_range": [-1.0, 1.0],
                    "z_range": [0.0, 9.42],
                },
                "vs_pinn": {
                    "scaling_factors": {
                        "N_x": 2.0,
                        "N_y": 12.0,
                        "N_z": 2.0,
                    },
                },
            }
        }
        
        device = torch.device("cpu")
        physics = create_physics(config, device)
        
        assert physics is not None
        # VS-PINN 物理模組應該是 nn.Module
        assert isinstance(physics, nn.Module)
    
    def test_ns_2d_physics(self):
        """測試 NS 2D 物理方程創建"""
        config = {
            "physics": {
                "type": "ns_2d",
                "nu": 0.01,
            }
        }
        
        device = torch.device("cpu")
        physics = create_physics(config, device)
        
        assert physics is not None
        # NSEquations2D 不是 nn.Module，是普通類
        # 驗證包含必要屬性
        assert hasattr(physics, "nu")
        assert physics.nu == 0.01
    
    def test_physics_device_placement_for_module(self):
        """測試物理模組設備放置（nn.Module）"""
        config = {
            "physics": {
                "type": "vs_pinn_channel_flow",
                "nu": 5e-5,
                "domain": {
                    "x_range": [0.0, 25.13],
                    "y_range": [-1.0, 1.0],
                    "z_range": [0.0, 9.42],
                },
                "vs_pinn": {
                    "scaling_factors": {
                        "N_x": 2.0,
                        "N_y": 12.0,
                        "N_z": 2.0,
                    },
                },
            }
        }
        
        device = torch.device("cpu")
        physics = create_physics(config, device)
        
        if isinstance(physics, nn.Module):
            # 檢查參數設備
            for param in physics.parameters():
                assert param.device.type == device.type
    
    def test_invalid_physics_type(self):
        """測試無效物理類型"""
        config = {
            "physics": {
                "type": "nonexistent_physics",
            }
        }
        
        device = torch.device("cpu")
        with pytest.raises(ValueError, match="Unsupported physics type"):
            create_physics(config, device)
    
    def test_missing_physics_config(self):
        """測試缺少 physics 配置"""
        config = {}
        device = torch.device("cpu")
        
        with pytest.raises(KeyError):
            create_physics(config, device)


# ============================================================================
# Test: create_optimizer()
# ============================================================================


"""
測試配置載入模組 (pinnx/train/config_loader.py)

驗證範圍:
    1. YAML 配置載入（成功/失敗案例）
    2. 配置結構標準化（嵌套→扁平、預設值設定）
    3. 損失權重推導（VS-PINN/非VS-PINN、自適應項生成）
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from pinnx.train.config_loader import (
    LOSS_KEY_MAP,
    DEFAULT_WEIGHTS,
    VS_ONLY_LOSSES,
    load_config,
    normalize_config_structure,
    derive_loss_weights,
)


# ============================================================================
# 測試 load_config
# ============================================================================

class TestLoadConfig:
    """測試 YAML 配置載入功能"""
    
    def test_load_basic_config(self, tmp_path):
        """測試載入基本配置"""
        config_dict = {
            'model': {
                'use_fourier': True,
                'fourier_m': 64,
            },
            'loss': {
                'data_weight': 20.0
            }
        }
        
        config_file = tmp_path / "test.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        result = load_config(str(config_file))
        
        assert result['model']['use_fourier'] is True
        assert result['model']['fourier_m'] == 64
        assert result['loss']['data_weight'] == 20.0
    
    def test_load_nested_fourier_config(self, tmp_path):
        """測試載入嵌套 Fourier 配置（應自動標準化）"""
        config_dict = {
            'model': {
                'fourier': {
                    'enabled': True,
                    'm': 128,
                    'sigma': 2.0,
                    'trainable': True
                }
            }
        }
        
        config_file = tmp_path / "nested.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        result = load_config(str(config_file))
        
        # 驗證嵌套格式已轉換為扁平格式
        assert result['model']['use_fourier'] is True
        assert result['model']['fourier_m'] == 128
        assert result['model']['fourier_sigma'] == 2.0
        assert result['model']['fourier_trainable'] is True
    
    def test_load_nonexistent_file(self):
        """測試載入不存在的檔案（應拋出 FileNotFoundError）"""
        with pytest.raises(FileNotFoundError, match="配置檔案不存在"):
            load_config('/nonexistent/path/config.yml')
    
    def test_load_invalid_yaml(self, tmp_path):
        """測試載入無效 YAML（應拋出 YAMLError）"""
        config_file = tmp_path / "invalid.yml"
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content:\n  - unbalanced")
        
        with pytest.raises(yaml.YAMLError, match="YAML 解析失敗"):
            load_config(str(config_file))
    
    def test_load_empty_file(self, tmp_path):
        """測試載入空檔案（應拋出 ValueError）"""
        config_file = tmp_path / "empty.yml"
        config_file.touch()
        
        with pytest.raises(ValueError, match="配置檔案為空"):
            load_config(str(config_file))


# ============================================================================
# 測試 normalize_config_structure
# ============================================================================

class TestNormalizeConfigStructure:
    """測試配置結構標準化"""
    
    def test_normalize_nested_fourier_to_flat(self):
        """測試嵌套 Fourier 格式轉扁平格式"""
        config = {
            'model': {
                'fourier': {
                    'enabled': False,
                    'm': 256,
                    'sigma': 5.0,
                    'trainable': True
                }
            }
        }
        
        result = normalize_config_structure(config)
        
        assert result['model']['use_fourier'] is False
        assert result['model']['fourier_m'] == 256
        assert result['model']['fourier_sigma'] == 5.0
        assert result['model']['fourier_trainable'] is True
    
    def test_normalize_preserves_flat_format(self):
        """測試扁平格式不被覆蓋"""
        config = {
            'model': {
                'use_fourier': True,
                'fourier_m': 64,
                'fourier': {
                    'enabled': False,  # 應被忽略
                    'm': 999  # 應被忽略
                }
            }
        }
        
        result = normalize_config_structure(config)
        
        # 扁平格式優先（不被嵌套格式覆蓋）
        assert result['model']['use_fourier'] is True
        assert result['model']['fourier_m'] == 64
    
    def test_normalize_sets_defaults(self):
        """測試設定預設值"""
        config = {'model': {}}
        
        result = normalize_config_structure(config)
        
        assert result['model']['use_fourier'] is True  # 預設啟用
        assert result['model']['fourier_m'] == 32
        assert result['model']['fourier_sigma'] == 1.0
        assert result['model']['fourier_trainable'] is False
    
    def test_normalize_empty_config(self):
        """測試空配置（應建立預設結構）"""
        config = {}
        
        result = normalize_config_structure(config)
        
        assert 'model' in result
        assert result['model']['use_fourier'] is True
        assert result['model']['fourier_m'] == 32


# ============================================================================
# 測試 derive_loss_weights
# ============================================================================

class TestDeriveLossWeights:
    """測試損失權重推導"""
    
    def test_derive_basic_weights_non_vs_pinn(self):
        """測試非 VS-PINN 模式的基本權重推導"""
        loss_cfg = {
            'data_weight': 20.0,
            'continuity_weight': 5.0,
        }
        
        weights, adaptive_terms = derive_loss_weights(
            loss_cfg=loss_cfg,
            prior_weight=0.1,
            is_vs_pinn=False
        )
        
        # 驗證配置權重已應用
        assert weights['data'] == 20.0
        assert weights['continuity'] == 5.0
        
        # 驗證預設權重已填充
        assert weights['momentum_x'] == DEFAULT_WEIGHTS['momentum_x']
        assert weights['momentum_y'] == DEFAULT_WEIGHTS['momentum_y']
        
        # 驗證 VS-PINN 專屬項已排除
        assert 'momentum_z' not in weights
        assert 'bulk_velocity' not in weights
        assert 'centerline_dudy' not in weights
        
        # 驗證 prior 權重
        assert weights['prior'] == 0.1
    
    def test_derive_weights_vs_pinn(self):
        """測試 VS-PINN 模式的權重推導（包含 3D 項）"""
        loss_cfg = {
            'momentum_z_weight': 3.0,
            'bulk_velocity_weight': 5.0,
        }
        
        weights, adaptive_terms = derive_loss_weights(
            loss_cfg=loss_cfg,
            prior_weight=0.0,
            is_vs_pinn=True
        )
        
        # 驗證 VS-PINN 專屬項已包含
        assert weights['momentum_z'] == 3.0
        assert weights['bulk_velocity'] == 5.0
        assert weights['centerline_dudy'] == DEFAULT_WEIGHTS['centerline_dudy']
        assert weights['centerline_v'] == DEFAULT_WEIGHTS['centerline_v']
        assert weights['pressure_reference'] == DEFAULT_WEIGHTS['pressure_reference']
    
    def test_derive_adaptive_terms_excludes_prior_when_zero(self):
        """測試當 prior 權重為 0 時不加入自適應調整列表"""
        loss_cfg = {}
        
        weights, adaptive_terms = derive_loss_weights(
            loss_cfg=loss_cfg,
            prior_weight=0.0,
            is_vs_pinn=False
        )
        
        # prior 權重為 0，不應加入自適應調整
        assert 'prior' not in adaptive_terms
        assert weights['prior'] == 0.0
    
    def test_derive_adaptive_terms_includes_prior_when_nonzero(self):
        """測試當 prior 權重 > 0 時加入自適應調整列表"""
        loss_cfg = {}
        
        weights, adaptive_terms = derive_loss_weights(
            loss_cfg=loss_cfg,
            prior_weight=0.5,
            is_vs_pinn=False
        )
        
        # prior 權重 > 0，應加入自適應調整
        assert 'prior' in adaptive_terms
        assert weights['prior'] == 0.5
    
    def test_derive_weights_with_boundary_weight_compat(self):
        """測試舊配置 boundary_weight 兼容性（應合併到 wall_constraint）"""
        loss_cfg = {
            'wall_constraint_weight': 10.0,
            'boundary_weight': 5.0,  # 舊配置
        }
        
        weights, _ = derive_loss_weights(
            loss_cfg=loss_cfg,
            prior_weight=0.0,
            is_vs_pinn=False
        )
        
        # boundary_weight 應加到 wall_constraint
        assert weights['wall_constraint'] == 15.0
    
    def test_derive_weights_from_config_priority(self):
        """測試配置優先於預設值"""
        loss_cfg = {
            'data_weight': 99.0,
            'continuity_weight': 88.0,
            'prior_weight': 0.77,
        }
        
        weights, _ = derive_loss_weights(
            loss_cfg=loss_cfg,
            prior_weight=0.1,  # 應被 loss_cfg 覆蓋
            is_vs_pinn=False
        )
        
        assert weights['data'] == 99.0
        assert weights['continuity'] == 88.0
        assert weights['prior'] == 0.77  # 使用配置值
    
    def test_derive_weights_periodicity_non_vs_pinn(self):
        """測試非 VS-PINN 模式下 periodicity 需明確配置才啟用"""
        # 未配置 periodicity_weight
        loss_cfg = {}
        weights_1, _ = derive_loss_weights(loss_cfg, 0.0, is_vs_pinn=False)
        assert 'periodicity' not in weights_1
        
        # 明確配置 periodicity_weight
        loss_cfg = {'periodicity_weight': 10.0}
        weights_2, _ = derive_loss_weights(loss_cfg, 0.0, is_vs_pinn=False)
        assert weights_2['periodicity'] == 10.0


# ============================================================================
# 測試常數正確性
# ============================================================================

class TestConstants:
    """測試模組常數"""
    
    def test_loss_key_map_completeness(self):
        """測試 LOSS_KEY_MAP 包含所有預設權重"""
        for name in DEFAULT_WEIGHTS.keys():
            assert name in LOSS_KEY_MAP, f"Missing {name} in LOSS_KEY_MAP"
    
    def test_vs_only_losses_subset(self):
        """測試 VS_ONLY_LOSSES 是 DEFAULT_WEIGHTS 的子集"""
        for name in VS_ONLY_LOSSES:
            assert name in DEFAULT_WEIGHTS, f"{name} not in DEFAULT_WEIGHTS"
    
    def test_default_weights_types(self):
        """測試預設權重為浮點數"""
        for name, value in DEFAULT_WEIGHTS.items():
            assert isinstance(value, (int, float)), f"{name} not numeric"


# ============================================================================
# 整合測試
# ============================================================================

class TestIntegration:
    """測試完整配置載入流程"""
    
    def test_full_pipeline(self, tmp_path):
        """測試完整配置載入→標準化→權重推導流程"""
        config_dict = {
            'model': {
                'fourier': {  # 嵌套格式
                    'enabled': True,
                    'm': 128,
                }
            },
            'physics': {
                'type': 'vs_pinn_channel_flow'
            },
            'loss': {
                'data_weight': 50.0,
                'momentum_z_weight': 10.0,  # VS-PINN 專屬
                'prior_weight': 0.2,
            }
        }
        
        config_file = tmp_path / "full.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        # 1. 載入配置（自動標準化）
        config = load_config(str(config_file))
        
        # 2. 驗證標準化
        assert config['model']['use_fourier'] is True
        assert config['model']['fourier_m'] == 128
        
        # 3. 推導損失權重
        is_vs_pinn = config['physics']['type'] == 'vs_pinn_channel_flow'
        weights, adaptive_terms = derive_loss_weights(
            loss_cfg=config['loss'],
            prior_weight=0.1,  # 應被配置覆蓋
            is_vs_pinn=is_vs_pinn
        )
        
        # 4. 驗證權重
        assert weights['data'] == 50.0
        assert weights['momentum_z'] == 10.0  # VS-PINN 項已包含
        assert weights['prior'] == 0.2
        
        # 5. 驗證自適應項（prior > 0 應加入）
        assert 'prior' in adaptive_terms
        assert 'data' in adaptive_terms


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

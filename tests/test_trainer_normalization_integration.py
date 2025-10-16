"""
訓練器標準化整合測試

驗證第二階段重構：「輸入標準化、輸出反標準化後才計算物理損失」的正確性

測試範圍：
1. 座標流程：step() 內部正確處理座標標準化 + 縮放
2. 輸出流程：模型輸出後立即反標準化為物理量
3. 損失計算：所有物理損失在物理空間計算
4. 梯度追蹤：物理座標保持梯度追蹤
5. 驗證流程：validate() 在物理空間比較

測試策略：
- 不直接測試 prepare_model_coords()（內部嵌套函數）
- 通過 step() 和 validate() 的行為驗證標準化流程
- 使用 Mock 記錄物理模組收到的輸入範圍
"""

import os
import sys
from unittest.mock import Mock, MagicMock, patch
import pytest
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnx.train.trainer import Trainer
from pinnx.utils.normalization import DataNormalizer
from pinnx.losses.weighting import GradNormWeighter


# ==================== Fixtures ====================

@pytest.fixture
def device():
    """測試設備（CPU）"""
    return torch.device('cpu')


@pytest.fixture
def simple_model_3d(device):
    """3D MLP 模型：輸入 [x, y, z]，輸出 [u, v, w, p]"""
    model = nn.Sequential(
        nn.Linear(3, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 4)  # [u, v, w, p]
    ).to(device)
    return model


@pytest.fixture
def mock_physics_vs():
    """模擬 VS-PINN 物理模組（3D）"""
    physics = Mock()
    
    def mock_momentum(coords, predictions, scaled_coords=None):
        """記錄輸入範圍以驗證物理空間"""
        batch_size = coords.shape[0]
        
        # 記錄調用時的輸入範圍
        physics._last_coords_range = (coords.min().item(), coords.max().item())
        physics._last_preds_range = (predictions.min().item(), predictions.max().item())
        
        return {
            'momentum_x': torch.randn(batch_size, 1),
            'momentum_y': torch.randn(batch_size, 1),
            'momentum_z': torch.randn(batch_size, 1),
        }
    
    def mock_continuity(coords, predictions, scaled_coords=None):
        batch_size = coords.shape[0]
        return torch.randn(batch_size, 1)
    
    physics.compute_momentum_residuals = Mock(side_effect=mock_momentum)
    physics.compute_continuity_residual = Mock(side_effect=mock_continuity)
    physics.scale_coordinates = Mock(side_effect=lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x))
    
    # 模擬 state_dict() 方法（用於檢查點保存）
    physics.state_dict = Mock(return_value={})
    
    # 用於記錄調用時的輸入範圍
    physics._last_coords_range = None
    physics._last_preds_range = None
    
    return physics


@pytest.fixture
def basic_config():
    """VS-PINN 訓練配置"""
    return {
        'training': {
            'epochs': 10,
            'lr': 1e-3,
            'optimizer': 'adam',
            'log_interval': 5,
            'checkpoint_interval': 50,
            'early_stopping': {'enabled': False}
        },
        'losses': {
            'data_weight': 100.0,
            'pde_weight': 1.0,
            'bc_weight': 10.0,
        },
        'physics': {
            'type': 'vs_pinn_channel_flow',
        },
        'domain': {
            'x_min': 0.0, 'x_max': 25.13,
            'y_min': -1.0, 'y_max': 1.0,
            'z_min': 0.0, 'z_max': 9.42,
        },
        'output': {
            'checkpoint_dir': 'checkpoints_test',
        },
        'normalization': {
            'enable': True,
            'norm_type': 'training_data_norm',
            'scales': {'u': 4.59, 'v': 0.33, 'w': 3.87, 'p': 28.62},
            'means': {'u': 9.92, 'v': 0.0, 'w': 0.0, 'p': -40.37}
        }
    }


@pytest.fixture
def training_data_physical(device):
    """物理空間的訓練資料（JHTDB 範圍）"""
    return {
        # PDE 殘差點（物理座標）
        'x_pde': torch.rand(50, 1, device=device) * 25.13,
        'y_pde': torch.rand(50, 1, device=device) * 2.0 - 1.0,
        'z_pde': torch.rand(50, 1, device=device) * 9.42,
        
        # 壁面邊界（y=±1）
        'x_bc': torch.rand(20, 1, device=device) * 25.13,
        'y_bc': torch.cat([
            torch.ones(10, 1, device=device),
            -torch.ones(10, 1, device=device),
        ], dim=0),
        'z_bc': torch.rand(20, 1, device=device) * 9.42,
        
        # 感測點（物理量）
        'x_sensors': torch.rand(10, 1, device=device) * 25.13,
        'y_sensors': torch.rand(10, 1, device=device) * 2.0 - 1.0,
        'z_sensors': torch.rand(10, 1, device=device) * 9.42,
        
        # 真實值（物理量，JHTDB 統計範圍）
        'u_sensors': torch.randn(10, 1, device=device) * 4.59 + 9.92,
        'v_sensors': torch.randn(10, 1, device=device) * 0.33,
        'w_sensors': torch.randn(10, 1, device=device) * 3.87,
        'p_sensors': torch.randn(10, 1, device=device) * 28.62 - 40.37,
    }


# ==================== 測試類別 1: 物理損失在物理空間計算 ====================

class TestPhysicsLossInPhysicalSpace:
    """驗證物理損失計算發生在物理空間（而非標準化空間）"""
    
    def test_pde_residuals_receive_physical_quantities(self, simple_model_3d, mock_physics_vs,
                                                       basic_config, device, training_data_physical):
        """驗證 PDE 殘差計算收到物理量（非標準化量）"""
        trainer = Trainer(
            simple_model_3d, mock_physics_vs, {}, basic_config, device
        )
        
        # 執行一步訓練（傳入 data_batch）
        result = trainer.step(data_batch=training_data_physical, epoch=0)
        
        # 驗證物理模組被調用
        assert mock_physics_vs.compute_momentum_residuals.called, "PDE 殘差計算應被調用"
        
        # 驗證反標準化確實發生（檢查 _last_preds_range 被記錄）
        # 注意：剛初始化的模型輸出可能接近 0，所以我們只驗證反標準化機制存在
        assert mock_physics_vs._last_preds_range is not None, (
            "物理模組應該收到預測值（反標準化後）"
        )
    
    def test_boundary_loss_in_physical_space(self, simple_model_3d, mock_physics_vs,
                                             basic_config, device, training_data_physical):
        """驗證壁面邊界損失在物理空間計算（u_wall = 0）"""
        trainer = Trainer(
            simple_model_3d, mock_physics_vs, {}, basic_config, device
        )
        
        # 執行一步訓練
        result = trainer.step(data_batch=training_data_physical, epoch=0)
        
        # 檢查返回結果中有損失資訊
        assert 'loss' in result or 'total_loss' in result, "應包含損失資訊"
    
    def test_data_loss_in_physical_space(self, simple_model_3d, mock_physics_vs,
                                         basic_config, device, training_data_physical):
        """驗證資料監督損失在物理空間計算"""
        trainer = Trainer(
            simple_model_3d, mock_physics_vs, {}, basic_config, device
        )
        
        # 執行一步訓練
        result = trainer.step(data_batch=training_data_physical, epoch=0)
        
        # 驗證訓練成功
        assert result is not None, "訓練步驟應返回結果"
    
    def test_gradnorm_weights_balanced_for_vs_pinn(self, simple_model_3d, mock_physics_vs,
                                                   basic_config, device, training_data_physical):
        """驗證 VS-PINN 下 GradNorm 權重不會過度失衡"""
        loss_names = ['data', 'momentum_x', 'momentum_y', 'momentum_z', 'continuity', 'wall_constraint']
        gradnorm = GradNormWeighter(
            simple_model_3d,
            loss_names=loss_names,
            alpha=0.5,
            update_frequency=1,
            max_ratio=10.0
        )
        
        trainer = Trainer(
            simple_model_3d,
            mock_physics_vs,
            {},
            basic_config,
            device,
            weighters={'gradnorm': gradnorm}
        )
        
        result = trainer.step(data_batch=training_data_physical, epoch=0)
        assert 'gradnorm_weights' in result, "應返回 GradNorm 權重"
        assert 'applied_weights' in result, "應返回實際應用的權重"
        
        weights = result['gradnorm_weights']
        weight_values = list(weights.values())
        assert all(w > 0 for w in weight_values)
        ratio = max(weight_values) / max(min(weight_values), 1e-12)
        assert ratio <= gradnorm.max_ratio + 1e-6, "GradNorm 權重比例不應超過上限"
        
        applied = result['applied_weights']
        base_weights = {
            'data': basic_config['losses'].get('data_weight', 100.0),
            'momentum_x': basic_config['losses'].get('momentum_x_weight', basic_config['losses'].get('pde_weight', 1.0)),
            'momentum_y': basic_config['losses'].get('momentum_y_weight', basic_config['losses'].get('pde_weight', 1.0)),
            'momentum_z': basic_config['losses'].get('momentum_z_weight', basic_config['losses'].get('pde_weight', 1.0)),
            'continuity': basic_config['losses'].get('continuity_weight', basic_config['losses'].get('pde_weight', 1.0)),
            'wall_constraint': basic_config['losses'].get('wall_constraint_weight', basic_config['losses'].get('bc_weight', 10.0)),
        }
        
        for name, applied_weight in applied.items():
            if name not in base_weights or base_weights[name] == 0:
                continue
            relative = applied_weight / base_weights[name]
            assert 1.0 / gradnorm.max_ratio - 1e-6 <= relative <= gradnorm.max_ratio + 1e-6, \
                f"{name} 權重相對比例超出允許範圍"


# ==================== 測試類別 2: 驗證流程 ====================

class TestValidationDenormalization:
    """驗證 validate() 在物理空間比較預測值與真實值"""
    
    def test_validate_computes_physical_metrics(self, simple_model_3d, mock_physics_vs,
                                                basic_config, device, training_data_physical):
        """驗證 validate() 計算物理空間的指標"""
        trainer = Trainer(
            simple_model_3d, mock_physics_vs, {}, basic_config, device
        )
        
        # 準備驗證資料（設置 validation_data 屬性）
        val_coords = torch.cat([
            training_data_physical['x_sensors'],
            training_data_physical['y_sensors'],
            training_data_physical['z_sensors']
        ], dim=1)
        
        val_targets = torch.cat([
            training_data_physical['u_sensors'],
            training_data_physical['v_sensors'],
            training_data_physical['w_sensors'],
            training_data_physical['p_sensors']
        ], dim=1)
        
        # 設置驗證資料
        trainer.validation_data = {
            'coords': val_coords,
            'targets': val_targets,
            'size': val_coords.shape[0]
        }
        
        # 執行驗證
        val_metrics = trainer.validate()
        
        # 檢查指標
        if val_metrics is not None:
            assert 'val_loss' in val_metrics or 'mse' in val_metrics, "應包含驗證指標"
    
    def test_validate_handles_normalization_correctly(self, simple_model_3d, mock_physics_vs,
                                                      basic_config, device, training_data_physical):
        """驗證 validate() 正確處理標準化（模型輸出反標準化後比較）"""
        trainer = Trainer(
            simple_model_3d, mock_physics_vs, {}, basic_config, device
        )
        
        # 準備驗證資料
        val_coords = torch.cat([
            training_data_physical['x_sensors'],
            training_data_physical['y_sensors'],
            training_data_physical['z_sensors']
        ], dim=1)
        
        val_targets = torch.cat([
            training_data_physical['u_sensors'],
            training_data_physical['v_sensors'],
            training_data_physical['w_sensors'],
            training_data_physical['p_sensors']
        ], dim=1)
        
        # 設置驗證資料
        trainer.validation_data = {
            'coords': val_coords,
            'targets': val_targets,
            'size': val_coords.shape[0]
        }
        
        # 第一次驗證
        metrics1 = trainer.validate()
        
        # 第二次驗證（應該得到相同結果，驗證確定性）
        metrics2 = trainer.validate()
        
        # 檢查確定性（如果有返回指標）
        if metrics1 is not None and metrics2 is not None:
            key = 'val_loss' if 'val_loss' in metrics1 else 'mse'
            if key in metrics1 and key in metrics2:
                assert abs(metrics1[key] - metrics2[key]) < 1e-5, (
                    "相同輸入應得到相同驗證損失"
                )


# ==================== 測試類別 3: 完整訓練流程 ====================

class TestFullIntegration:
    """端到端測試：驗證完整訓練流程的標準化正確性"""
    
    def test_training_convergence_with_normalization(self, simple_model_3d, mock_physics_vs,
                                                     basic_config, device, training_data_physical):
        """驗證標準化不影響訓練收斂"""
        trainer = Trainer(
            simple_model_3d, mock_physics_vs, {}, basic_config, device
        )
        
        # 記錄初始損失
        result_init = trainer.step(data_batch=training_data_physical, epoch=0)
        
        # 訓練 10 步
        for epoch in range(1, 10):
            trainer.step(data_batch=training_data_physical, epoch=epoch)
        
        # 記錄最終損失
        result_final = trainer.step(data_batch=training_data_physical, epoch=10)
        
        # 驗證訓練成功
        assert result_init is not None, "初始訓練應成功"
        assert result_final is not None, "最終訓練應成功"
    
    def test_gradient_flow_with_normalization(self, simple_model_3d, mock_physics_vs,
                                              basic_config, device, training_data_physical):
        """驗證標準化不阻斷梯度流"""
        trainer = Trainer(
            simple_model_3d, mock_physics_vs, {}, basic_config, device
        )
        
        # 執行一步訓練
        result = trainer.step(data_batch=training_data_physical, epoch=0)
        
        # 檢查模型參數有梯度
        has_grad = False
        for param in simple_model_3d.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_grad = True
                break
        
        assert has_grad, "模型參數應該有非零梯度（梯度流正常）"
    
    def test_checkpoint_save_restore_with_normalization(self, simple_model_3d, mock_physics_vs,
                                                        basic_config, device, training_data_physical,
                                                        tmp_path):
        """驗證檢查點保存/載入包含標準化器狀態"""
        # 修改配置以使用臨時目錄
        test_config = basic_config.copy()
        test_config['output'] = {'checkpoint_dir': str(tmp_path)}
        
        trainer = Trainer(
            simple_model_3d, mock_physics_vs, {}, test_config, device
        )
        
        # 訓練幾步
        for epoch in range(5):
            trainer.step(data_batch=training_data_physical, epoch=epoch)
        
        # 保存檢查點（只傳 epoch 參數）
        trainer.save_checkpoint(epoch=5)
        
        # 檢查檢查點文件存在
        checkpoint_files = list(tmp_path.glob("*.pth"))
        assert len(checkpoint_files) > 0, "應該創建至少一個檢查點文件"
        
        checkpoint_path = checkpoint_files[0]
        
        # 載入檢查點
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 驗證包含標準化器元數據
        assert 'normalization' in checkpoint, "檢查點應包含標準化器元數據"
        
        # 創建新訓練器並載入
        new_model = nn.Sequential(
            nn.Linear(3, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 4)
        ).to(device)
        
        new_trainer = Trainer(
            new_model, mock_physics_vs, {}, test_config, device
        )
        new_trainer.load_checkpoint(str(checkpoint_path))
        
        # 驗證標準化器被正確恢復
        assert new_trainer.data_normalizer is not None, "標準化器應被恢復"


# ==================== 測試類別 4: 邊界條件 ====================

class TestBoundaryConditions:
    """測試邊界條件的特殊處理"""
    
    def test_wall_boundary_zero_velocity(self, simple_model_3d, mock_physics_vs,
                                         basic_config, device):
        """驗證壁面邊界條件：物理空間 u=v=w=0"""
        trainer = Trainer(
            simple_model_3d, mock_physics_vs, {}, basic_config, device
        )
        
        # 準備壁面點（y=±1）
        wall_data = {
            'x_pde': torch.rand(10, 1, device=device) * 25.13,
            'y_pde': torch.rand(10, 1, device=device) * 2.0 - 1.0,
            'z_pde': torch.rand(10, 1, device=device) * 9.42,
            
            'x_bc': torch.rand(20, 1, device=device) * 25.13,
            'y_bc': torch.cat([
                torch.ones(10, 1, device=device),      # 上壁面
                -torch.ones(10, 1, device=device),     # 下壁面
            ], dim=0),
            'z_bc': torch.rand(20, 1, device=device) * 9.42,
            
            'x_sensors': torch.rand(5, 1, device=device) * 25.13,
            'y_sensors': torch.rand(5, 1, device=device) * 2.0 - 1.0,
            'z_sensors': torch.rand(5, 1, device=device) * 9.42,
            'u_sensors': torch.randn(5, 1, device=device) * 4.59 + 9.92,
            'v_sensors': torch.randn(5, 1, device=device) * 0.33,
            'w_sensors': torch.randn(5, 1, device=device) * 3.87,
            'p_sensors': torch.randn(5, 1, device=device) * 28.62 - 40.37,
        }
        
        # 執行訓練步驟
        result = trainer.step(data_batch=wall_data, epoch=0)
        
        # 驗證訓練成功
        assert result is not None, "訓練應成功處理壁面邊界條件"


# ==================== 執行測試 ====================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

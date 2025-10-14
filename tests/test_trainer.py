"""
Trainer 類單元測試

測試 pinnx/train/trainer.py 的 Trainer 類核心功能：
- 初始化邏輯
- 單步訓練 (step)
- 驗證指標 (validate)
- 完整訓練循環 (train)
- 檢查點管理 (save/load checkpoint)
- 早停機制 (early stopping)
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnx.train.trainer import Trainer


# ==================== Fixtures ====================

@pytest.fixture
def device():
    """測試設備（優先使用 CPU 以確保測試一致性）"""
    return torch.device('cpu')


@pytest.fixture
def simple_model(device):
    """簡單的 MLP 模型用於測試（2D 輸入）"""
    model = nn.Sequential(
        nn.Linear(2, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 3)  # 輸出 [u, v, p]
    ).to(device)
    return model


@pytest.fixture
def simple_model_3d(device):
    """簡單的 3D MLP 模型用於測試（3D 輸入）"""
    model = nn.Sequential(
        nn.Linear(3, 32),  # 3D 輸入 [x, y, z]
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 4)  # 輸出 [u, v, w, p]
    ).to(device)
    return model


@pytest.fixture
def mock_physics():
    """模擬物理模組（2D NS 方程）"""
    physics = Mock()
    
    # 模擬 residual() 方法（2D PINN）
    def mock_residual(coords, velocity, pressure):
        """返回固定的殘差（用於測試）"""
        batch_size = coords.shape[0]
        return {
            'momentum_x': torch.randn(batch_size, 1),
            'momentum_y': torch.randn(batch_size, 1),
            'momentum_z': torch.zeros(batch_size, 1),  # 2D 沒有 z 分量
            'continuity': torch.randn(batch_size, 1),
        }
    
    physics.residual = Mock(side_effect=mock_residual)
    physics.compute_momentum_residuals = None  # 標記為 2D PINN
    physics.scale_coordinates = Mock(side_effect=lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x))  # 恆等映射
    
    return physics


@pytest.fixture
def mock_physics_vs():
    """模擬 VS-PINN 物理模組（3D）"""
    physics = Mock()
    
    # 模擬 compute_momentum_residuals() 方法（VS-PINN）
    def mock_momentum(coords, predictions, scaled_coords=None):
        batch_size = coords.shape[0]
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
    physics.scale_coordinates = Mock(side_effect=lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x))  # 恆等映射
    
    return physics


@pytest.fixture
def mock_losses():
    """模擬損失函數字典"""
    return {}


@pytest.fixture
def basic_config():
    """基礎訓練配置"""
    return {
        'training': {
            'epochs': 100,
            'lr': 1e-3,
            'optimizer': 'adam',
            'log_interval': 10,
            'checkpoint_interval': 50,
            'early_stopping': {
                'enabled': False,
                'patience': 20,
                'min_delta': 1e-6,
            }
        },
        'losses': {
            'data_weight': 100.0,
            'pde_weight': 1.0,
            'bc_weight': 10.0,
        },
        'logging': {
            'log_freq': 10,
        },
        'output': {
            'checkpoint_dir': 'checkpoints_test',
        },
        'physics': {
            'type': 'navier_stokes_2d',
        },
        'domain': {
            'x_min': 0.0, 'x_max': 8.0,
            'y_min': -1.0, 'y_max': 1.0,
        }
    }


@pytest.fixture
def training_data_2d(device):
    """2D PINN 訓練資料"""
    return {
        'x_pde': torch.randn(50, 1, device=device),
        'y_pde': torch.randn(50, 1, device=device),
        'x_bc': torch.cat([
            torch.ones(10, 1, device=device),    # 上壁面 y=1
            -torch.ones(10, 1, device=device),   # 下壁面 y=-1
        ], dim=0),
        'y_bc': torch.cat([
            torch.ones(10, 1, device=device),
            -torch.ones(10, 1, device=device),
        ], dim=0),
        'x_sensors': torch.randn(10, 1, device=device),
        'y_sensors': torch.randn(10, 1, device=device),
        'u_sensors': torch.randn(10, 1, device=device),
        'v_sensors': torch.randn(10, 1, device=device),
        'p_sensors': torch.randn(10, 1, device=device),
    }


@pytest.fixture
def training_data_3d(device):
    """3D VS-PINN 訓練資料"""
    return {
        'x_pde': torch.randn(50, 1, device=device),
        'y_pde': torch.randn(50, 1, device=device),
        'z_pde': torch.randn(50, 1, device=device),
        'x_bc': torch.cat([
            torch.ones(10, 1, device=device),
            -torch.ones(10, 1, device=device),
        ], dim=0),
        'y_bc': torch.cat([
            torch.ones(10, 1, device=device),
            -torch.ones(10, 1, device=device),
        ], dim=0),
        'z_bc': torch.randn(20, 1, device=device),
        'x_sensors': torch.randn(10, 1, device=device),
        'y_sensors': torch.randn(10, 1, device=device),
        'z_sensors': torch.randn(10, 1, device=device),
        'u_sensors': torch.randn(10, 1, device=device),
        'v_sensors': torch.randn(10, 1, device=device),
        'p_sensors': torch.randn(10, 1, device=device),
    }


@pytest.fixture
def validation_data(device):
    """驗證資料"""
    n_val = 30
    return {
        'coords': torch.randn(n_val, 2, device=device),
        'targets': torch.randn(n_val, 3, device=device),  # [u, v, p]
        'size': n_val,
    }


# ==================== 測試類別 1: 初始化 ====================

class TestTrainerInitialization:
    """測試 Trainer 初始化邏輯"""
    
    def test_init_basic(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試基本初始化"""
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        
        # 檢查基本屬性
        assert trainer.model is simple_model
        assert trainer.physics is mock_physics
        assert trainer.device == device
        assert trainer.epoch == 0
        assert trainer.global_step == 0
        
        # 檢查優化器
        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)
        
        # 檢查訓練配置
        assert trainer.train_cfg['epochs'] == 100
        assert trainer.train_cfg['lr'] == 1e-3
    
    def test_init_optimizer_adam(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試 Adam 優化器初始化"""
        basic_config['training']['optimizer'] = 'adam'
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        
        assert isinstance(trainer.optimizer, torch.optim.Adam)
    
    def test_init_optimizer_adamw(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試 AdamW 優化器初始化"""
        basic_config['training']['optimizer'] = 'adamw'
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
    
    def test_init_optimizer_lbfgs(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試 L-BFGS 優化器初始化"""
        basic_config['training']['optimizer'] = 'lbfgs'
        basic_config['training']['optimizer'] = {
            'type': 'lbfgs',
            'lr': 1.0,
        }
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        
        assert isinstance(trainer.optimizer, torch.optim.LBFGS)
    
    def test_init_with_normalizer(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試帶輸入標準化器的初始化"""
        mock_normalizer = Mock()
        mock_normalizer.transform = Mock(side_effect=lambda x: x)  # 恆等映射
        
        trainer = Trainer(
            simple_model, mock_physics, mock_losses, basic_config, device,
            input_normalizer=mock_normalizer
        )
        
        assert trainer.input_normalizer is mock_normalizer
    
    def test_init_early_stopping_disabled(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試早停機制禁用"""
        basic_config['training']['early_stopping']['enabled'] = False
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        
        assert trainer.early_stopping_enabled is False
    
    def test_init_early_stopping_enabled(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試早停機制啟用"""
        basic_config['training']['early_stopping']['enabled'] = True
        basic_config['training']['early_stopping']['patience'] = 30
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        
        assert trainer.early_stopping_enabled is True
        assert trainer.patience == 30


# ==================== 測試類別 2: 單步訓練 ====================

class TestTrainerStep:
    """測試 Trainer.step() 方法"""
    
    def test_step_2d_basic(self, simple_model, mock_physics, mock_losses, basic_config, device, training_data_2d):
        """測試 2D PINN 基本單步訓練"""
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        trainer.training_data = training_data_2d
        
        # 執行訓練步驟
        result = trainer.step(training_data_2d, epoch=0)
        
        # 檢查返回值結構
        assert 'total_loss' in result
        assert 'data_loss' in result
        assert 'pde_loss' in result
        assert 'wall_loss' in result
        
        # 檢查損失值有效性（無 NaN/Inf）
        assert not np.isnan(result['total_loss'])
        assert not np.isinf(result['total_loss'])
        assert result['total_loss'] >= 0
        
        # 檢查物理模組被調用
        assert mock_physics.residual.called
    
    def test_step_3d_vs_pinn(self, simple_model_3d, mock_physics_vs, mock_losses, basic_config, device, training_data_3d):
        """測試 3D VS-PINN 單步訓練"""
        # 修改配置為 VS-PINN
        basic_config['physics']['type'] = 'vs_pinn_channel_flow'
        basic_config['domain']['z_min'] = 0.0
        basic_config['domain']['z_max'] = 6.0
        
        trainer = Trainer(simple_model_3d, mock_physics_vs, mock_losses, basic_config, device)
        trainer.training_data = training_data_3d
        
        # 執行訓練步驟
        result = trainer.step(training_data_3d, epoch=0)
        
        # 檢查返回值
        assert 'total_loss' in result
        assert 'momentum_z_loss' in result  # 3D 獨有
        assert not np.isnan(result['total_loss'])
        
        # 檢查 VS-PINN 方法被調用
        assert mock_physics_vs.compute_momentum_residuals.called
        assert mock_physics_vs.compute_continuity_residual.called
    
    def test_step_gradient_computation(self, simple_model, mock_physics, mock_losses, basic_config, device, training_data_2d):
        """測試梯度計算正確性"""
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        trainer.training_data = training_data_2d
        
        # 執行前檢查梯度為空
        for param in simple_model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)
        
        # 執行訓練步驟
        result = trainer.step(training_data_2d, epoch=0)
        
        # 執行後檢查梯度已計算
        has_gradient = False
        for param in simple_model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradient = True
                break
        
        assert has_gradient, "模型參數未計算梯度"
    
    def test_step_with_point_weights(self, simple_model, mock_physics, mock_losses, basic_config, device, training_data_2d):
        """測試帶點權重的訓練步驟"""
        # 添加點權重
        training_data_2d['pde_point_weights'] = torch.rand(50, 1, device=device) + 0.5
        
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        trainer.training_data = training_data_2d
        
        result = trainer.step(training_data_2d, epoch=0)
        
        # 檢查損失值有效
        assert not np.isnan(result['total_loss'])
        assert result['total_loss'] >= 0
    
    def test_step_with_gradnorm_weighting(self, simple_model, mock_physics, mock_losses,
                                          basic_config, device, training_data_2d):
        """啟用 GradNorm 時應輸出動態權重資訊"""
        basic_config['losses'].update({
            'adaptive_weighting': True,
            'adaptive_loss_terms': ['data', 'momentum_x', 'momentum_y', 'continuity', 'wall_constraint'],
            'weight_update_freq': 1,
            'grad_norm_alpha': 0.2,
        })
        
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        trainer.training_data = training_data_2d
        
        result = trainer.step(training_data_2d, epoch=0)
        
        # 應該回傳 GradNorm 權重與實際套用的權重
        assert 'gradnorm_weights' in result
        assert 'applied_weights' in result
        gradnorm_weights = result['gradnorm_weights']
        applied_weights = result['applied_weights']
        
        # 取出訓練器內保存的初始權重以驗證縮放邏輯
        gradnorm_module = trainer.weighters.get('gradnorm')
        assert gradnorm_module is not None
        initial_weights = gradnorm_module.initial_weight_values
        
        # 驗證資料項權重符合縮放比例
        expected_data_weight = (
            basic_config['losses']['data_weight']
            * gradnorm_weights['data']
            / initial_weights['data']
        )
        assert applied_weights['data'] == pytest.approx(expected_data_weight, rel=1e-4)
        
        # 驗證帶權重的 loss 數值一致
        assert result['weighted_data_loss'] == pytest.approx(
            applied_weights['data'] * result['data_loss'], rel=1e-4
        )
    
    def test_step_gradient_clip(self, simple_model, mock_physics, mock_losses, basic_config, device, training_data_2d):
        """測試梯度裁剪"""
        basic_config['training']['gradient_clip'] = 1.0
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        trainer.training_data = training_data_2d
        
        result = trainer.step(training_data_2d, epoch=0)
        
        # 檢查梯度範數是否被限制
        total_norm = 0.0
        for param in simple_model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # 梯度裁剪後應該 ≤ 閾值（考慮數值誤差）
        assert total_norm <= 1.1, f"梯度範數 {total_norm} 超過裁剪閾值 1.0"


# ==================== 測試類別 3: 驗證 ====================

class TestTrainerValidation:
    """測試 Trainer.validate() 方法"""
    
    def test_validate_basic(self, simple_model, mock_physics, mock_losses, basic_config, device, validation_data):
        """測試基本驗證功能"""
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        trainer.validation_data = validation_data
        
        metrics = trainer.validate()
        
        # 檢查返回值結構
        assert metrics is not None
        assert 'mse' in metrics
        assert 'relative_l2' in metrics
        
        # 檢查指標值有效性
        assert not np.isnan(metrics['mse'])
        assert not np.isnan(metrics['relative_l2'])
        assert metrics['mse'] >= 0
        assert metrics['relative_l2'] >= 0
    
    def test_validate_no_data(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試無驗證資料時返回 None"""
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        trainer.validation_data = None
        
        metrics = trainer.validate()
        
        assert metrics is None
    
    def test_validate_empty_data(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試空驗證資料時返回 None"""
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        trainer.validation_data = {'size': 0}
        
        metrics = trainer.validate()
        
        assert metrics is None
    
    def test_validate_dimension_mismatch(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試維度不匹配時的處理（模型輸出 3，目標 4）"""
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        
        # 驗證資料目標維度為 4（模型輸出為 3）
        trainer.validation_data = {
            'coords': torch.randn(20, 2, device=device),
            'targets': torch.randn(20, 4, device=device),  # 4 維目標
            'size': 20,
        }
        
        # 應該只比較前 3 個分量
        metrics = trainer.validate()
        
        assert metrics is not None
        assert not np.isnan(metrics['mse'])
    
    def test_validate_preserves_training_mode(self, simple_model, mock_physics, mock_losses, basic_config, device, validation_data):
        """測試驗證後恢復訓練模式"""
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        trainer.validation_data = validation_data
        
        # 設置為訓練模式
        simple_model.train()
        assert simple_model.training is True
        
        # 執行驗證
        metrics = trainer.validate()
        
        # 檢查訓練模式恢復
        assert simple_model.training is True


# ==================== 測試類別 4: 完整訓練循環 ====================

class TestTrainerTraining:
    """測試 Trainer.train() 方法"""
    
    def test_train_basic(self, simple_model, mock_physics, mock_losses, basic_config, device, training_data_2d):
        """測試基本訓練循環"""
        basic_config['training']['epochs'] = 5  # 減少 epoch 以加速測試
        basic_config['training']['log_interval'] = 2
        
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        trainer.training_data = training_data_2d
        
        result = trainer.train()
        
        # 檢查返回值結構
        assert 'final_loss' in result
        assert 'training_time' in result
        assert 'epochs_completed' in result
        assert 'history' in result
        
        # 檢查訓練完成
        assert result['epochs_completed'] == 5
        assert result['training_time'] > 0
        assert len(result['history']['total_loss']) == 5
    
    def test_train_with_early_stopping(self, simple_model, mock_physics, mock_losses, basic_config, device, training_data_2d):
        """測試帶早停的訓練循環"""
        basic_config['training']['epochs'] = 100
        basic_config['training']['early_stopping']['enabled'] = True
        basic_config['training']['early_stopping']['patience'] = 3
        basic_config['training']['early_stopping']['monitor'] = 'val_loss'
        
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        trainer.training_data = training_data_2d
        
        # Mock 驗證指標（模擬不改善情況：指標遞增）
        # 第一次驗證返回最好值 (0.5)，後續逐漸變差以觸發早停
        # 前 3 次: 0.5, 0.51, 0.52, 0.53 -> 在第 4 次時觸發早停（patience=3）
        def mock_validate_side_effect():
            """生成器：模擬指標逐漸變差"""
            yield {'mse': 1.0, 'relative_l2': 0.50}  # epoch 0: 最佳
            yield {'mse': 1.0, 'relative_l2': 0.51}  # epoch 1: 變差
            yield {'mse': 1.0, 'relative_l2': 0.52}  # epoch 2: 變差
            yield {'mse': 1.0, 'relative_l2': 0.53}  # epoch 3: 變差 -> 觸發早停
            # 不應該到達這裡
            for i in range(4, 100):
                yield {'mse': 1.0, 'relative_l2': 0.5 + i * 0.01}
        
        trainer.validate = Mock(side_effect=mock_validate_side_effect())
        trainer.validation_data = {'size': 10}
        
        result = trainer.train()
        
        # 檢查早停觸發（應該在 epoch 4 停止，完成 4 個 epoch）
        # 注意：epochs_completed 是實際完成的 epoch 數（0-3 共 4 個）
        assert result['epochs_completed'] <= 4, f"Expected ≤4 epochs, got {result['epochs_completed']}"
        assert result['epochs_completed'] < 100, "Early stopping should trigger before 100 epochs"
        assert trainer.best_epoch == 0, f"Best epoch should be 0, got {trainer.best_epoch}"
    
    def test_train_with_lr_scheduler(self, simple_model, mock_physics, mock_losses, basic_config, device, training_data_2d):
        """測試帶學習率調度器的訓練"""
        basic_config['training']['epochs'] = 10
        basic_config['training']['log_interval'] = 1  # 每個 epoch 都記錄
        basic_config['training']['lr_scheduler'] = {
            'type': 'step',
            'step_size': 3,
            'gamma': 0.5
        }
        
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        trainer.training_data = training_data_2d
        
        initial_lr = trainer.get_current_lr()
        
        result = trainer.train()
        
        # 檢查學習率歷史（應該有 10 個記錄，對應 epoch 0, 1, ..., 9）
        assert len(trainer.history['lr']) == 10
        # 學習率應該在訓練過程中逐步下降
        # StepLR(step_size=3, gamma=0.5) 的行為：
        # - epoch 0-1: lr=1e-3 (初始值)
        # - epoch 2: lr 降為 5e-4 (第 3 次 step 後)
        # - epoch 3-4: lr=5e-4
        # - epoch 5: lr 降為 2.5e-4 (第 6 次 step 後)
        # - epoch 6-7: lr=2.5e-4
        # - epoch 8: lr 降為 1.25e-4 (第 9 次 step 後)
        # - epoch 9: lr=1.25e-4
        assert trainer.history['lr'][0] == initial_lr
        assert trainer.history['lr'][2] < trainer.history['lr'][1]  # epoch 2 時第一次下降
        assert trainer.history['lr'][5] < trainer.history['lr'][4]  # epoch 5 時第二次下降
        assert trainer.history['lr'][8] < trainer.history['lr'][7]  # epoch 8 時第三次下降
    
    def test_train_with_checkpointing(self, simple_model, mock_physics, mock_losses, basic_config, device, training_data_2d):
        """測試檢查點保存"""
        with tempfile.TemporaryDirectory() as tmpdir:
            basic_config['training']['epochs'] = 10
            basic_config['training']['checkpoint_freq'] = 5
            basic_config['output']['checkpoint_dir'] = tmpdir
            
            trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
            trainer.training_data = training_data_2d
            
            result = trainer.train()
            
            # 檢查檢查點文件是否存在
            checkpoint_dir = Path(tmpdir)
            checkpoint_files = list(checkpoint_dir.glob('*.pth'))
            
            # 應該有 epoch_5 和最終檢查點
            assert len(checkpoint_files) >= 1
    
    def test_train_fast_convergence(self, simple_model, mock_physics, mock_losses, basic_config, device, training_data_2d):
        """測試快速收斂時提前停止"""
        basic_config['training']['epochs'] = 100
        
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        trainer.training_data = training_data_2d
        
        # Mock step() 返回極小損失
        original_step = trainer.step
        def mock_step(data, epoch):
            return {'total_loss': 1e-7, 'data_loss': 1e-8, 'pde_loss': 1e-8, 
                    'wall_loss': 0.0, 'momentum_x_loss': 0.0, 'momentum_y_loss': 0.0,
                    'momentum_z_loss': 0.0, 'continuity_loss': 0.0,
                    'u_loss': 0.0, 'v_loss': 0.0, 'pressure_loss': 0.0}
        
        trainer.step = Mock(side_effect=mock_step)
        
        result = trainer.train()
        
        # 檢查提前停止（loss < 1e-6）
        assert result['epochs_completed'] < 100


# ==================== 測試類別 5: 檢查點管理 ====================

class TestTrainerCheckpointing:
    """測試檢查點保存與載入"""
    
    def test_save_checkpoint(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試檢查點保存"""
        with tempfile.TemporaryDirectory() as tmpdir:
            basic_config['output']['checkpoint_dir'] = tmpdir
            trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
            
            # 保存檢查點
            trainer.save_checkpoint(epoch=10, metrics={'loss': 0.5})
            
            # 檢查文件存在
            checkpoint_path = Path(tmpdir) / 'epoch_10.pth'
            assert checkpoint_path.exists()
            
            # 檢查檢查點內容
            checkpoint = torch.load(checkpoint_path, map_location=device)
            assert 'epoch' in checkpoint
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert checkpoint['epoch'] == 10
    
    def test_load_checkpoint(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試檢查點載入"""
        with tempfile.TemporaryDirectory() as tmpdir:
            basic_config['output']['checkpoint_dir'] = tmpdir
            
            # 創建並保存檢查點
            trainer1 = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
            trainer1.epoch = 15
            checkpoint_path = Path(tmpdir) / 'test_checkpoint.pth'
            torch.save({
                'epoch': 15,
                'model_state_dict': trainer1.model.state_dict(),
                'optimizer_state_dict': trainer1.optimizer.state_dict(),
                'history': {'loss': [1.0, 0.8, 0.6]},
            }, checkpoint_path)
            
            # 創建新訓練器並載入
            trainer2 = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
            trainer2.load_checkpoint(str(checkpoint_path))
            
            # 檢查狀態恢復
            assert trainer2.epoch == 15
    
    def test_save_best_model(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試最佳模型保存"""
        with tempfile.TemporaryDirectory() as tmpdir:
            basic_config['output']['checkpoint_dir'] = tmpdir
            trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
            
            # 保存最佳模型
            trainer.save_checkpoint(epoch=20, metrics={'loss': 0.1}, is_best=True)
            
            # 檢查最佳模型文件存在
            best_path = Path(tmpdir) / 'best_model.pth'
            assert best_path.exists()


# ==================== 測試類別 6: 早停機制 ====================

class TestTrainerEarlyStopping:
    """測試早停邏輯"""
    
    def test_early_stopping_improvement(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試指標改善時不觸發早停"""
        basic_config['training']['early_stopping']['enabled'] = True
        basic_config['training']['early_stopping']['patience'] = 5
        
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        
        # 模擬指標改善
        assert trainer.check_early_stopping(1.0) is False
        assert trainer.check_early_stopping(0.8) is False
        assert trainer.check_early_stopping(0.6) is False
        
        # 檢查最佳指標更新
        assert trainer.best_val_loss == 0.6
        assert trainer.patience_counter == 0
    
    def test_early_stopping_trigger(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試達到 patience 時觸發早停"""
        basic_config['training']['early_stopping']['enabled'] = True
        basic_config['training']['early_stopping']['patience'] = 3
        
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        
        # 模擬指標不改善
        assert trainer.check_early_stopping(1.0) is False  # 新最佳
        assert trainer.check_early_stopping(1.1) is False  # +1
        assert trainer.check_early_stopping(1.2) is False  # +2
        assert trainer.check_early_stopping(1.3) is True   # +3，觸發早停
        
        # 檢查計數器
        assert trainer.patience_counter == 3
    
    def test_early_stopping_disabled(self, simple_model, mock_physics, mock_losses, basic_config, device):
        """測試早停禁用時不觸發"""
        basic_config['training']['early_stopping']['enabled'] = False
        trainer = Trainer(simple_model, mock_physics, mock_losses, basic_config, device)
        
        # 無論指標如何都不應觸發
        assert trainer.check_early_stopping(1.0) is False
        assert trainer.check_early_stopping(2.0) is False
        assert trainer.check_early_stopping(3.0) is False


# ==================== 測試執行入口 ====================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

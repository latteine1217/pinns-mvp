"""
檢查點管理模組的單元測試

測試 pinnx/train/checkpointing.py 中的保存與載入功能。
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
import tempfile
import shutil
from pinnx.train.checkpointing import save_checkpoint, load_checkpoint


class SimpleModel(nn.Module):
    """測試用的簡單模型"""
    def __init__(self, input_dim: int = 3, hidden_dim: int = 10, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def temp_checkpoint_dir():
    """創建臨時檢查點目錄"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # 清理
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def model_and_optimizer():
    """創建測試用模型與優化器"""
    model = SimpleModel(input_dim=3, hidden_dim=10, output_dim=1)
    optimizer = Adam(model.parameters(), lr=0.001)
    return model, optimizer


@pytest.fixture
def valid_config():
    """有效的配置字典"""
    return {
        'experiment': {'name': 'test_experiment'},
        'model': {'hidden_dim': 10},
        'training': {'lr': 0.001}
    }


class TestSaveCheckpoint:
    """測試 save_checkpoint 函數"""
    
    def test_basic_save(self, model_and_optimizer, valid_config, temp_checkpoint_dir):
        """測試基本保存功能"""
        model, optimizer = model_and_optimizer
        
        # 保存檢查點
        saved_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=100,
            loss=0.123,
            config=valid_config,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        # 驗證返回路徑
        expected_path = Path(temp_checkpoint_dir) / "test_experiment_epoch_100.pth"
        assert saved_path == str(expected_path)
        
        # 驗證檔案存在
        assert Path(saved_path).exists()
        
        # 驗證檢查點內容
        checkpoint = torch.load(saved_path)
        assert checkpoint['epoch'] == 100
        assert checkpoint['loss'] == 0.123
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert checkpoint['config'] == valid_config
    
    def test_auto_create_directory(self, model_and_optimizer, valid_config):
        """測試自動創建目錄功能"""
        model, optimizer = model_and_optimizer
        
        # 使用不存在的目錄
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_dir = Path(temp_dir) / "nested" / "checkpoint_dir"
            assert not non_existent_dir.exists()
            
            # 保存應自動創建目錄
            saved_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=50,
                loss=0.5,
                config=valid_config,
                checkpoint_dir=str(non_existent_dir)
            )
            
            # 驗證目錄已創建
            assert non_existent_dir.exists()
            assert Path(saved_path).exists()
    
    def test_save_both_epoch_and_latest(self, model_and_optimizer, valid_config, temp_checkpoint_dir):
        """測試同時保存 epoch 和 latest 版本"""
        model, optimizer = model_and_optimizer
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=200,
            loss=0.01,
            config=valid_config,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        # 驗證兩個檔案都存在
        epoch_path = Path(temp_checkpoint_dir) / "test_experiment_epoch_200.pth"
        latest_path = Path(temp_checkpoint_dir) / "test_experiment_latest.pth"
        
        assert epoch_path.exists()
        assert latest_path.exists()
        
        # 驗證內容一致
        epoch_ckpt = torch.load(epoch_path)
        latest_ckpt = torch.load(latest_path)
        
        assert epoch_ckpt['epoch'] == latest_ckpt['epoch'] == 200
        assert epoch_ckpt['loss'] == latest_ckpt['loss'] == 0.01
    
    def test_missing_experiment_name_raises_error(self, model_and_optimizer, temp_checkpoint_dir):
        """測試缺少 experiment.name 時拋出 KeyError"""
        model, optimizer = model_and_optimizer
        
        # 配置缺少 experiment.name
        invalid_config = {'model': {'hidden_dim': 10}}
        
        with pytest.raises(KeyError, match="experiment.name"):
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=1,
                loss=1.0,
                config=invalid_config,
                checkpoint_dir=temp_checkpoint_dir
            )


class TestLoadCheckpoint:
    """測試 load_checkpoint 函數"""
    
    def test_basic_load(self, model_and_optimizer, valid_config, temp_checkpoint_dir):
        """測試基本載入功能"""
        model, optimizer = model_and_optimizer
        
        # 先保存
        saved_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=150,
            loss=0.234,
            config=valid_config,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        # 創建新模型與優化器
        new_model = SimpleModel(input_dim=3, hidden_dim=10, output_dim=1)
        new_optimizer = Adam(new_model.parameters(), lr=0.001)
        
        # 載入檢查點
        epoch, loss, loaded_config = load_checkpoint(
            checkpoint_path=saved_path,
            model=new_model,
            optimizer=new_optimizer,
            device="cpu"
        )
        
        # 驗證元數據
        assert epoch == 150
        assert loss == 0.234
        assert loaded_config == valid_config
        
        # 驗證模型參數已載入（比較狀態字典）
        for (key1, param1), (key2, param2) in zip(
            model.state_dict().items(), 
            new_model.state_dict().items()
        ):
            assert key1 == key2
            assert torch.allclose(param1, param2)
    
    def test_load_to_different_device(self, model_and_optimizer, valid_config, temp_checkpoint_dir):
        """測試載入到不同設備"""
        model, optimizer = model_and_optimizer
        
        # 保存檢查點
        saved_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=100,
            loss=0.1,
            config=valid_config,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        # 測試載入到 CPU
        new_model_cpu = SimpleModel()
        epoch, loss, _ = load_checkpoint(saved_path, new_model_cpu, device="cpu")
        assert epoch == 100
        
        # 測試載入到 CUDA（若可用）
        if torch.cuda.is_available():
            new_model_cuda = SimpleModel()
            epoch, loss, _ = load_checkpoint(saved_path, new_model_cuda, device="cuda")
            assert epoch == 100
        
        # 測試載入到 MPS（若可用，macOS）
        if torch.backends.mps.is_available():
            new_model_mps = SimpleModel()
            epoch, loss, _ = load_checkpoint(saved_path, new_model_mps, device="mps")
            assert epoch == 100
    
    def test_load_without_optimizer(self, model_and_optimizer, valid_config, temp_checkpoint_dir):
        """測試不載入優化器狀態（推理模式）"""
        model, optimizer = model_and_optimizer
        
        # 保存檢查點
        saved_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=100,
            loss=0.1,
            config=valid_config,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        # 只載入模型（不傳入 optimizer）
        new_model = SimpleModel()
        epoch, loss, config = load_checkpoint(
            checkpoint_path=saved_path,
            model=new_model,
            optimizer=None,  # 不載入優化器
            device="cpu"
        )
        
        # 應成功載入
        assert epoch == 100
        assert loss == 0.1
        assert config == valid_config
    
    def test_file_not_found_raises_error(self):
        """測試檔案不存在時拋出 FileNotFoundError"""
        model = SimpleModel()
        non_existent_path = "/tmp/non_existent_checkpoint_xyz123.pth"
        
        with pytest.raises(FileNotFoundError, match="檢查點檔案不存在"):
            load_checkpoint(non_existent_path, model)
    
    def test_model_architecture_mismatch_raises_error(self, model_and_optimizer, valid_config, temp_checkpoint_dir):
        """測試模型架構不匹配時拋出 RuntimeError"""
        model, optimizer = model_and_optimizer
        
        # 保存檢查點
        saved_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=100,
            loss=0.1,
            config=valid_config,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        # 創建不同架構的模型
        class DifferentModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)  # 不同的架構
            
            def forward(self, x):
                return self.fc(x)
        
        wrong_model = DifferentModel()
        
        # 載入應失敗
        with pytest.raises(RuntimeError):
            load_checkpoint(saved_path, wrong_model)
    
    def test_load_checkpoint_missing_metadata_raises(self, model_and_optimizer, temp_checkpoint_dir):
        """測試缺少必要欄位的檢查點會被拒絕"""
        model, optimizer = model_and_optimizer
        
        legacy_checkpoint = {
            'epoch': 50,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0.5
        }
        
        legacy_path = Path(temp_checkpoint_dir) / "legacy_checkpoint.pth"
        torch.save(legacy_checkpoint, legacy_path)
        
        new_model = SimpleModel()
        with pytest.raises(KeyError):
            load_checkpoint(str(legacy_path), new_model)


class TestCheckpointIntegration:
    """測試保存與載入的整合流程"""
    
    def test_save_and_resume_training(self, valid_config, temp_checkpoint_dir):
        """測試保存後恢復訓練的完整流程"""
        # === 第一階段：初始訓練 ===
        model1 = SimpleModel()
        optimizer1 = Adam(model1.parameters(), lr=0.001)
        
        # 模擬訓練
        x = torch.randn(10, 3)
        y = torch.randn(10, 1)
        
        for _ in range(10):
            optimizer1.zero_grad()
            loss = nn.MSELoss()(model1(x), y)
            loss.backward()
            optimizer1.step()
        
        # 保存檢查點
        final_loss = loss.item()
        saved_path = save_checkpoint(
            model=model1,
            optimizer=optimizer1,
            epoch=10,
            loss=final_loss,
            config=valid_config,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        # === 第二階段：恢復訓練 ===
        model2 = SimpleModel()
        optimizer2 = Adam(model2.parameters(), lr=0.001)
        
        # 載入檢查點
        epoch, loaded_loss, _ = load_checkpoint(saved_path, model2, optimizer2)
        
        # 驗證恢復狀態
        assert epoch == 10
        assert loaded_loss == final_loss
        
        # 驗證模型權重完全一致
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
        
        # 驗證優化器狀態一致（檢查學習率）
        assert optimizer1.param_groups[0]['lr'] == optimizer2.param_groups[0]['lr']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

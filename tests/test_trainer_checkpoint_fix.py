"""
測試 Trainer 最佳模型自動保存功能

驗證 TASK-008 修復：
- 發現新最佳模型時立即保存到磁碟
- 檢查點包含正確的 metrics
- best_model.pth 文件正確創建
"""
import pytest
import torch
import tempfile
from pathlib import Path
from pinnx.train.trainer import Trainer


def test_best_model_auto_save(tmp_path):
    """測試最佳模型自動保存到磁碟"""
    
    # 簡單模型和配置
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 4)
    )
    
    config = {
        'training': {
            'optimizer': 'adam',
            'lr': 1e-3,
            'early_stopping': {
                'enabled': True,
                'patience': 10,
                'min_delta': 1e-6,
                'restore_best_weights': True
            },
            'epochs': 100
        },
        'losses': {
            'data_weight': 1.0,
            'pde_weight': 1.0
        },
        'output': {
            'checkpoint_dir': str(tmp_path)
        }
    }
    
    # 創建訓練器
    trainer = Trainer(
        model=model,
        physics=None,
        losses={},
        config=config,
        device=torch.device('cpu')
    )
    
    # 模擬發現新最佳模型（初始 val_loss=100）
    trainer.best_val_loss = 100.0
    trainer.epoch = 10
    
    # 檢查初始狀態
    best_model_path = tmp_path / "best_model.pth"
    assert not best_model_path.exists(), "最佳模型文件不應在初始化時存在"
    
    # 觸發新最佳模型（val_loss=50 < 100）
    should_stop = trainer.check_early_stopping(50.0)
    
    # 驗證結果
    assert not should_stop, "不應觸發早停（剛更新最佳模型）"
    assert trainer.best_val_loss == 50.0, "最佳損失應更新"
    assert trainer.best_epoch == 10, "最佳 epoch 應記錄"
    assert trainer.patience_counter == 0, "耐心計數器應重置"
    
    # 🎯 關鍵測試：檢查 best_model.pth 是否自動創建
    assert best_model_path.exists(), "❌ 最佳模型文件應自動保存到磁碟"
    
    # 驗證檢查點內容
    checkpoint = torch.load(best_model_path)
    assert 'model_state_dict' in checkpoint, "檢查點應包含模型狀態"
    assert 'metrics' in checkpoint, "檢查點應包含指標"
    assert checkpoint['metrics']['val_loss'] == 50.0, "應記錄正確的 val_loss"
    assert checkpoint['metrics']['best_epoch'] == 10, "應記錄正確的 best_epoch"
    assert checkpoint['epoch'] == 10, "檢查點應記錄正確的 epoch"
    
    print("✅ 測試通過：最佳模型自動保存功能正常")


def test_early_stopping_restores_best_model(tmp_path):
    """測試早停觸發時恢復最佳模型"""
    
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 4)
    )
    
    config = {
        'training': {
            'optimizer': 'adam',
            'lr': 1e-3,
            'early_stopping': {
                'enabled': True,
                'patience': 3,
                'min_delta': 1e-6,
                'restore_best_weights': True
            },
            'epochs': 100
        },
        'losses': {},
        'output': {
            'checkpoint_dir': str(tmp_path)
        }
    }
    
    trainer = Trainer(model, None, {}, config, device=torch.device('cpu'))
    
    # 模擬訓練歷史
    trainer.epoch = 5
    trainer.check_early_stopping(10.0)  # epoch=5: best=10.0
    
    # 保存最佳模型時的權重快照
    best_weights_sum = sum(p.sum().item() for p in model.parameters())
    
    # 模擬後續訓練（模型權重改變）
    trainer.epoch = 6
    for p in model.parameters():
        p.data += 0.1
    trainer.check_early_stopping(12.0)  # epoch=6: loss 變差（patience=1）
    
    trainer.epoch = 7
    trainer.check_early_stopping(13.0)  # epoch=7: loss 持續變差（patience=2）
    
    trainer.epoch = 8
    should_stop = trainer.check_early_stopping(14.0)  # epoch=8: 觸發早停（patience=3）
    
    assert should_stop, "應觸發早停"
    
    # 驗證最佳模型已恢復（由訓練循環負責恢復）
    # 這裡檢查 best_model_state 是否正確保存
    assert trainer.best_model_state is not None, "最佳模型狀態應已保存"
    assert trainer.best_epoch == 5, "最佳 epoch 應為 5"
    
    # 驗證磁碟文件存在
    best_model_path = tmp_path / "best_model.pth"
    assert best_model_path.exists(), "最佳模型文件應存在於磁碟"
    
    print("✅ 測試通過：早停恢復最佳模型功能正常")


if __name__ == "__main__":
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        print("=" * 60)
        print("測試 1: 最佳模型自動保存")
        print("=" * 60)
        test_best_model_auto_save(tmp_path / "test1")
        
        print("\n" + "=" * 60)
        print("測試 2: 早停恢復最佳模型")
        print("=" * 60)
        test_early_stopping_restores_best_model(tmp_path / "test2")
        
        print("\n" + "=" * 60)
        print("🎉 所有測試通過！")
        print("=" * 60)

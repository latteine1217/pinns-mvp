"""
PirateNet 整合測試

驗證項目:
1. SOAP 優化器初始化與參數
2. Swish (SiLU) 激活函數輸出正確性
3. Steps-based Warmup Scheduler 調度行為
4. RWF 參數配置（μ=1.0, σ=0.1）
5. 完整配置載入無錯誤
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import yaml
import sys

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

# 導入核心模組
from pinnx.optim.soap import SOAP
from pinnx.models.fourier_mlp import PINNNet, RWFLinear
from pinnx.train.schedulers import StepsBasedWarmupScheduler


# ============================================
# 1. SOAP 優化器測試
# ============================================

def test_soap_optimizer_initialization():
    """測試 SOAP 優化器初始化與參數設置"""
    # 創建簡單模型
    model = nn.Linear(10, 5)
    
    # 初始化 SOAP 優化器
    optimizer = SOAP(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        precondition_frequency=2
    )
    
    # 驗證參數
    assert optimizer.defaults['lr'] == 1e-3
    assert optimizer.defaults['betas'] == (0.9, 0.999)
    assert optimizer.defaults['weight_decay'] == 0.0
    assert optimizer.defaults['precondition_frequency'] == 2
    
    print("✅ SOAP optimizer 初始化測試通過")


def test_soap_optimizer_step():
    """測試 SOAP 優化器執行步驟"""
    # 創建簡單模型與資料
    model = nn.Linear(10, 1)
    optimizer = SOAP(model.parameters(), lr=1e-2)  # 增加學習率以便觀察變化
    
    # 儲存初始權重
    initial_weight = model.weight.data.clone()
    
    # 執行多次訓練步驟（SOAP 需要幾步來初始化 preconditioner）
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    for _ in range(5):  # 執行 5 步確保優化器完全啟動
        loss = nn.MSELoss()(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 驗證權重已更新
    assert not torch.allclose(model.weight.data, initial_weight, atol=1e-6), \
        "權重應在 5 步訓練後有顯著變化"
    
    print("✅ SOAP optimizer step 執行測試通過")


# ============================================
# 2. Swish 激活函數測試
# ============================================

def test_swish_activation():
    """測試 Swish (SiLU) 激活函數輸出正確性"""
    # PyTorch 內建 SiLU
    swish = nn.SiLU()
    
    # 測試輸入
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # 計算輸出
    output = swish(x)
    
    # 手動計算 Swish: x * sigmoid(x)
    expected = x * torch.sigmoid(x)
    
    # 驗證輸出
    assert torch.allclose(output, expected, atol=1e-6)
    
    # 驗證零點附近行為（應該接近線性）
    assert torch.abs(swish(torch.tensor(0.0)) - 0.0) < 1e-6
    
    print("✅ Swish 激活函數輸出測試通過")


def test_swish_in_model():
    """測試 Swish 在模型中正確使用"""
    # 創建使用 Swish 的模型
    model = PINNNet(
        in_dim=3,
        out_dim=4,
        depth=2,
        width=32,
        activation='swish',
        use_fourier=False,
        use_rwf=False
    )
    
    # 檢查激活函數類型
    # 在 DenseLayer 中，activation 應該是 SiLU
    for layer in model.hidden_layers:
        assert isinstance(layer.activation, nn.SiLU), \
            f"Expected SiLU, got {type(layer.activation)}"
    
    # 測試前向傳播
    x = torch.randn(16, 3)
    output = model(x)
    
    # 驗證輸出形狀
    assert output.shape == (16, 4)
    assert not torch.isnan(output).any()
    
    print("✅ Swish 在模型中使用測試通過")


# ============================================
# 3. Steps-based Scheduler 測試
# ============================================

def test_steps_based_scheduler_warmup():
    """測試 Steps-based Warmup Scheduler 的 warmup 階段"""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    scheduler = StepsBasedWarmupScheduler(
        optimizer,
        base_lr=1e-3,
        warmup_steps=100,
        total_steps=1000,
        decay_steps=200,
        gamma=0.9
    )
    
    # 記錄學習率變化（在 step() 之後記錄）
    lrs = []
    for step in range(100):
        scheduler.step()  # 先更新學習率
        lrs.append(optimizer.param_groups[0]['lr'])
    
    # 驗證 warmup 階段學習率遞增
    assert lrs[0] < lrs[50] < lrs[99], \
        f"Warmup 階段學習率應遞增: lr[0]={lrs[0]:.6f}, lr[50]={lrs[50]:.6f}, lr[99]={lrs[99]:.6f}"
    assert abs(lrs[99] - 1e-3) < 1e-6, \
        f"Warmup 結束時應達到 base_lr=1e-3, 實際為 {lrs[99]:.6f}"
    
    print("✅ Steps-based scheduler warmup 測試通過")


def test_steps_based_scheduler_decay():
    """測試 Steps-based Scheduler 的 decay 階段"""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    scheduler = StepsBasedWarmupScheduler(
        optimizer,
        base_lr=1e-3,
        warmup_steps=100,
        total_steps=10000,
        decay_steps=200,
        gamma=0.9
    )
    
    # 跳過 warmup 階段（執行 100 步到達 base_lr）
    for _ in range(100):
        scheduler.step()
    
    lr_before_decay = optimizer.param_groups[0]['lr']
    
    # 執行 201 步（觸發一次 decay）
    # 註: scheduler 在設定 LR 時使用 current_step，然後才遞增
    # 所以需要 201 步才能讓 current_step=300 時設定 LR
    for _ in range(201):
        scheduler.step()
    
    lr_after_decay = optimizer.param_groups[0]['lr']
    
    # 驗證學習率衰減
    expected_lr = lr_before_decay * 0.9
    assert abs(lr_after_decay - expected_lr) < 1e-7, \
        f"Expected LR={expected_lr:.8f}, got {lr_after_decay:.8f}"
    
    print("✅ Steps-based scheduler decay 測試通過")


# ============================================
# 4. RWF 參數配置測試
# ============================================

def test_rwf_scale_mean_parameter():
    """測試 RWF scale_mean 參數可配置性"""
    # 測試 μ=1.0 配置
    rwf_layer = RWFLinear(
        in_features=32,
        out_features=64,
        scale_mean=1.0,
        scale_std=0.1
    )
    
    # 驗證參數設置
    assert rwf_layer.scale_mean == 1.0
    assert rwf_layer.scale_std == 0.1
    
    # 檢查縮放參數的統計特性（大致符合 N(1.0, 0.1)）
    scale_mean = rwf_layer.s.mean().item()
    scale_std = rwf_layer.s.std().item()
    
    # 允許一定誤差（統計估計）
    assert 0.8 < scale_mean < 1.2, f"Mean={scale_mean} 應接近 1.0"
    assert 0.05 < scale_std < 0.15, f"Std={scale_std} 應接近 0.1"
    
    print("✅ RWF scale_mean 參數測試通過")


def test_rwf_in_enhanced_fourier_mlp():
    """測試 RWF 在 PINNNet 中正確配置"""
    model = PINNNet(
        in_dim=3,
        out_dim=4,
        depth=4,
        width=128,
        activation='swish',
        use_rwf=True,
        rwf_scale_mean=1.0,
        rwf_scale_std=0.1
    )
    
    # 檢查所有隱藏層是否使用 RWF
    for layer in model.hidden_layers:
        assert hasattr(layer.linear, 's'), "應包含 RWF 縮放參數 's'"
        # 驗證縮放參數形狀正確
        assert layer.linear.s.shape == (layer.linear.out_features,)
    
    # 測試前向傳播
    x = torch.randn(16, 3)
    output = model(x)
    assert output.shape == (16, 4)
    assert not torch.isnan(output).any()
    
    print("✅ RWF 在 PINNNet 中配置測試通過")


# ============================================
# 5. 完整配置載入測試
# ============================================

def test_piratenet_config_loading():
    """測試 PirateNet 配置檔案載入無錯誤"""
    config_path = Path(__file__).parent.parent / "configs/templates/piratenet_baseline.yml"
    
    # 檢查配置檔案是否存在
    assert config_path.exists(), f"配置檔案不存在: {config_path}"
    
    # 載入配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 驗證關鍵參數
    assert config['model']['activation'] == 'swish'
    assert config['model']['depth'] == 6
    assert config['model']['width'] == 768
    assert config['model']['rwf_scale_mean'] == 1.0
    assert config['model']['rwf_scale_std'] == 0.1
    assert config['model']['fourier_sigma'] == 2.0
    
    assert config['training']['optimizer']['type'] == 'soap'
    assert config['training']['optimizer']['betas'] == [0.9, 0.999]
    assert config['training']['optimizer']['precondition_frequency'] == 2
    
    assert config['training']['scheduler']['type'] == 'warmup_exponential_steps'
    assert config['training']['scheduler']['warmup_steps'] == 2000
    assert config['training']['scheduler']['decay_steps'] == 2000
    assert config['training']['scheduler']['decay_gamma'] == 0.9
    
    print("✅ PirateNet 配置載入測試通過")


def test_piratenet_model_creation_from_config():
    """測試從配置創建 PirateNet 模型"""
    from pinnx.models.fourier_mlp import create_pinn_model
    
    config_path = Path(__file__).parent.parent / "configs/templates/piratenet_baseline.yml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 創建模型 (使用工廠函數)
    model = create_pinn_model(config['model'])
    
    # 驗證模型結構
    assert len(model.hidden_layers) == config['model']['depth']
    
    # 驗證激活函數
    for layer in model.hidden_layers:
        assert isinstance(layer.activation, nn.SiLU)
    
    # 驗證 RWF
    for layer in model.hidden_layers:
        assert hasattr(layer.linear, 's')
    
    # 測試前向傳播
    x = torch.randn(8, 3)
    output = model(x)
    assert output.shape == (8, 4)
    assert not torch.isnan(output).any()
    
    print("✅ PirateNet 模型創建測試通過")


# ============================================
# 6. 整合測試
# ============================================

def test_piratenet_full_integration():
    """完整整合測試: 模型 + SOAP + Scheduler"""
    # 創建模型
    model = PINNNet(
        in_dim=3,
        out_dim=4,
        depth=6,
        width=768,
        activation='swish',
        use_fourier=True,
        fourier_m=64,
        fourier_sigma=2.0,
        use_rwf=True,
        rwf_scale_mean=1.0,
        rwf_scale_std=0.1
    )
    
    # 創建 SOAP 優化器
    optimizer = SOAP(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        precondition_frequency=2
    )
    
    # 創建 Steps-based Scheduler
    scheduler = StepsBasedWarmupScheduler(
        optimizer,
        base_lr=1e-3,
        warmup_steps=2000,
        total_steps=100000,
        decay_steps=2000,
        gamma=0.9
    )
    
    # 模擬訓練步驟
    x = torch.randn(32, 3)
    y = torch.randn(32, 4)
    
    for step in range(10):
        # 前向傳播
        output = model(x)
        loss = nn.MSELoss()(output, y)
        
        # 後向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 驗證無 NaN
        assert not torch.isnan(loss).item()
        assert not torch.isnan(output).any()
    
    print("✅ PirateNet 完整整合測試通過")


# ============================================
# 執行所有測試
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("開始 PirateNet 整合測試")
    print("="*60 + "\n")
    
    # 1. SOAP 優化器
    print("1️⃣  測試 SOAP 優化器...")
    test_soap_optimizer_initialization()
    test_soap_optimizer_step()
    
    # 2. Swish 激活函數
    print("\n2️⃣  測試 Swish 激活函數...")
    test_swish_activation()
    test_swish_in_model()
    
    # 3. Steps-based Scheduler
    print("\n3️⃣  測試 Steps-based Scheduler...")
    test_steps_based_scheduler_warmup()
    test_steps_based_scheduler_decay()
    
    # 4. RWF 參數
    print("\n4️⃣  測試 RWF 參數配置...")
    test_rwf_scale_mean_parameter()
    test_rwf_in_enhanced_fourier_mlp()
    
    # 5. 配置載入
    print("\n5️⃣  測試配置載入...")
    test_piratenet_config_loading()
    test_piratenet_model_creation_from_config()
    
    # 6. 完整整合
    print("\n6️⃣  測試完整整合...")
    test_piratenet_full_integration()
    
    print("\n" + "="*60)
    print("🎉 所有測試通過！PirateNet 整合成功")
    print("="*60 + "\n")

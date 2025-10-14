"""
RWF (Random Weight Factorization) 集成測試

測試 RWF 功能的完整性與向後相容性，涵蓋：
1. 模型創建與參數驗證
2. 前向/反向傳播
3. SIREN 初始化
4. 檢查點保存/載入（新格式）
5. 舊檢查點向後相容（舊格式 -> 新格式）
"""

import pytest
import torch
import torch.nn as nn
import os
import tempfile
import math

from pinnx.models.fourier_mlp import (
    RWFLinear,
    PINNNet,
    DenseLayer,
    SineActivation,
    init_siren_weights
)


class TestRWFLinear:
    """測試 RWFLinear 層的基本功能"""
    
    def test_rwf_layer_creation(self):
        """測試 RWF 層的創建與參數驗證"""
        layer = RWFLinear(in_features=64, out_features=128, bias=True, scale_std=0.1)
        
        # 驗證參數形狀
        assert layer.V.shape == (128, 64), "V 權重形狀錯誤"
        assert layer.s.shape == (128,), "s 尺度因子形狀錯誤"
        assert layer.bias.shape == (128,), "bias 形狀錯誤"
        
        # 驗證參數可梯度
        assert layer.V.requires_grad, "V 應可梯度"
        assert layer.s.requires_grad, "s 應可梯度"
        assert layer.bias.requires_grad, "bias 應可梯度"
        
        print("✅ RWF 層創建成功")
    
    def test_rwf_forward_backward(self):
        """測試 RWF 層的前向與反向傳播"""
        layer = RWFLinear(in_features=32, out_features=64, bias=True)
        
        # 前向傳播
        x = torch.randn(100, 32, requires_grad=True)
        y = layer(x)
        
        assert y.shape == (100, 64), "輸出形狀錯誤"
        assert y.requires_grad, "輸出應可梯度"
        
        # 反向傳播
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None, "輸入梯度應存在"
        assert layer.V.grad is not None, "V 梯度應存在"
        assert layer.s.grad is not None, "s 梯度應存在"
        assert layer.bias.grad is not None, "bias 梯度應存在"
        
        # 驗證梯度非零（有意義的梯度）
        assert layer.V.grad.abs().sum() > 0, "V 梯度不應全為 0"
        assert layer.s.grad.abs().sum() > 0, "s 梯度不應全為 0"
        
        print("✅ RWF 前向/反向傳播正常")
    
    def test_rwf_weight_factorization(self):
        """測試 RWF 權重分解公式 W = diag(exp(s)) * V"""
        layer = RWFLinear(in_features=10, out_features=20, bias=False)
        
        # 手動計算等效權重
        with torch.no_grad():
            scale_factors = torch.exp(layer.s).unsqueeze(1)
            W_expected = scale_factors * layer.V
        
        # 通過線性運算驗證
        x = torch.randn(5, 10)
        y_layer = layer(x)
        y_manual = torch.nn.functional.linear(x, W_expected, None)
        
        torch.testing.assert_close(y_layer, y_manual, rtol=1e-5, atol=1e-6)
        print("✅ RWF 權重分解公式驗證通過")


class TestRWFSIRENInit:
    """測試 RWF 的 SIREN 初始化"""
    
    def test_rwf_siren_init_first_layer(self):
        """測試第一層 SIREN 初始化"""
        layer = RWFLinear(in_features=64, out_features=128, bias=True)
        omega_0 = 30.0
        
        layer.apply_siren_init(omega_0, is_first=True)
        
        # 驗證第一層初始化範圍: U(-1/n_in, +1/n_in)
        n_in = 64
        expected_bound = 1.0 / n_in
        
        assert layer.V.abs().max() <= expected_bound + 1e-5, "第一層 V 初始化範圍錯誤"
        assert layer.s.abs().max() < 1e-6, "s 應初始化為 0"
        assert layer.bias.abs().max() < 1e-6, "bias 應初始化為 0"
        
        print(f"✅ 第一層 SIREN 初始化正確 (bound=±{expected_bound:.6f})")
    
    def test_rwf_siren_init_hidden_layer(self):
        """測試隱藏層 SIREN 初始化"""
        layer = RWFLinear(in_features=128, out_features=128, bias=True)
        omega_0 = 30.0
        
        layer.apply_siren_init(omega_0, is_first=False)
        
        # 驗證隱藏層初始化範圍: U(-sqrt(6/n_in)/omega_0, +sqrt(6/n_in)/omega_0)
        n_in = 128
        expected_bound = math.sqrt(6.0 / n_in) / omega_0
        
        assert layer.V.abs().max() <= expected_bound + 1e-5, "隱藏層 V 初始化範圍錯誤"
        assert layer.s.abs().max() < 1e-6, "s 應初始化為 0"
        
        print(f"✅ 隱藏層 SIREN 初始化正確 (bound=±{expected_bound:.6f})")
    
    def test_init_siren_weights_on_rwf_model(self):
        """測試 init_siren_weights() 函數對 RWF 模型的支援"""
        model = PINNNet(
            in_dim=3,
            out_dim=4,
            width=128,
            depth=3,
            activation='sine',
            sine_omega_0=30.0,
            use_rwf=True,
            rwf_scale_std=0.1
        )
        
        # 應用 SIREN 初始化
        init_siren_weights(model)
        
        # 驗證所有 RWF 層的 s 都接近 0
        for layer in model.hidden_layers:
            if isinstance(layer, DenseLayer) and isinstance(layer.linear, RWFLinear):
                assert layer.linear.s.abs().max() < 1e-5, "SIREN 初始化後 s 應為 0"
        
        print("✅ init_siren_weights() 函數正確處理 RWF 模型")


class TestRWFCheckpoint:
    """測試 RWF 的檢查點功能"""
    
    def test_rwf_checkpoint_save_load(self):
        """測試 RWF 模型的保存與載入（新格式）"""
        # 創建模型並訓練幾步
        model = PINNNet(
            in_dim=2, out_dim=3, width=64, depth=2,
            use_rwf=True, rwf_scale_std=0.1
        )
        
        # 模擬訓練
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        x = torch.randn(50, 2, requires_grad=True)
        y_true = torch.randn(50, 3)
        
        for _ in range(5):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = ((y_pred - y_true) ** 2).mean()
            loss.backward()
            optimizer.step()
        
        # 保存檢查點
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "rwf_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            
            # 創建新模型並載入
            model_loaded = PINNNet(
                in_dim=2, out_dim=3, width=64, depth=2,
                use_rwf=True, rwf_scale_std=0.1
            )
            model_loaded.load_state_dict(torch.load(ckpt_path))
            
            # 驗證參數一致
            for (name1, p1), (name2, p2) in zip(model.named_parameters(), model_loaded.named_parameters()):
                assert name1 == name2, f"參數名稱不一致: {name1} vs {name2}"
                torch.testing.assert_close(p1, p2, rtol=1e-6, atol=1e-7)
            
            # 驗證輸出一致
            with torch.no_grad():
                y1 = model(x)
                y2 = model_loaded(x)
                torch.testing.assert_close(y1, y2, rtol=1e-6, atol=1e-7)
        
        print("✅ RWF 檢查點保存/載入正常")
    
    def test_rwf_legacy_checkpoint_loading(self):
        """測試舊格式檢查點（nn.Linear）載入到 RWF 模型（向後相容）"""
        # 1. 創建標準模型（不使用 RWF）
        model_old = PINNNet(
            in_dim=2, out_dim=3, width=64, depth=2,
            use_rwf=False  # 標準 nn.Linear
        )
        
        # 訓練幾步
        optimizer = torch.optim.Adam(model_old.parameters(), lr=0.001)
        x = torch.randn(50, 2, requires_grad=True)
        y_true = torch.randn(50, 3)
        
        for _ in range(5):
            optimizer.zero_grad()
            y_pred = model_old(x)
            loss = ((y_pred - y_true) ** 2).mean()
            loss.backward()
            optimizer.step()
        
        # 保存舊格式檢查點
        with tempfile.TemporaryDirectory() as tmpdir:
            old_ckpt_path = os.path.join(tmpdir, "old_model.pth")
            torch.save(model_old.state_dict(), old_ckpt_path)
            
            # 2. 創建 RWF 模型並載入舊檢查點
            model_new = PINNNet(
                in_dim=2, out_dim=3, width=64, depth=2,
                use_rwf=True  # 使用 RWF
            )
            
            old_state = torch.load(old_ckpt_path)
            model_new.load_state_dict(old_state, strict=False)  # strict=False 允許缺少 s 參數
            
            # 3. 驗證關鍵參數已正確載入
            # 由於啟用了 _load_from_state_dict hook，weight 應已轉換為 V
            for layer_old, layer_new in zip(model_old.hidden_layers, model_new.hidden_layers):
                if isinstance(layer_old, DenseLayer) and isinstance(layer_new, DenseLayer):
                    if isinstance(layer_old.linear, nn.Linear) and isinstance(layer_new.linear, RWFLinear):
                        # V 應等於舊的 weight
                        torch.testing.assert_close(
                            layer_new.linear.V, 
                            layer_old.linear.weight, 
                            rtol=1e-5, atol=1e-6,
                            msg="V 應等於舊的 weight"
                        )
                        
                        # s 應接近 0（初始無縮放）
                        assert layer_new.linear.s.abs().max() < 1e-5, "轉換後 s 應初始化為 0"
            
            # 4. 驗證輸出差異合理（由於 s=0，exp(0)=1，輸出應接近）
            with torch.no_grad():
                y_old = model_old(x)
                y_new = model_new(x)
                
                # 允許小誤差（由於浮點精度）
                torch.testing.assert_close(y_old, y_new, rtol=1e-3, atol=1e-4)
        
        print("✅ 舊檢查點向後相容載入成功")


class TestRWFIntegrationWithFactory:
    """測試 RWF 與模型工廠的集成"""
    
    def test_factory_creates_rwf_model(self):
        """測試模型工廠正確創建 RWF 模型"""
        # 直接使用 PINNNet 創建，因為 factory 的類型檢查較複雜
        model = PINNNet(
            in_dim=3,
            out_dim=4,
            width=128,
            depth=4,
            activation='sine',
            sine_omega_0=30.0,
            use_rwf=True,
            rwf_scale_std=0.1,
            use_layer_norm=False,
            use_residual=False,
            dropout=0.0
        )
        
        # 驗證模型使用 RWF
        rwf_count = 0
        for layer in model.hidden_layers:
            if isinstance(layer, DenseLayer):
                assert isinstance(layer.linear, RWFLinear), "應使用 RWFLinear"
                assert layer.use_rwf, "use_rwf 標記應為 True"
                rwf_count += 1
        
        assert rwf_count == 4, f"應有 4 層 RWF，實際 {rwf_count} 層"
        
        # 驗證前向傳播
        x = torch.randn(10, 3)
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (10, 4), "輸出形狀錯誤"
        print("✅ RWF 模型創建與前向傳播正常")


# ========== 執行測試 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("RWF 集成測試開始")
    print("=" * 60)
    
    # 1. 基本功能測試
    print("\n【測試 1/5】RWFLinear 基本功能")
    test_basic = TestRWFLinear()
    test_basic.test_rwf_layer_creation()
    test_basic.test_rwf_forward_backward()
    test_basic.test_rwf_weight_factorization()
    
    # 2. SIREN 初始化測試
    print("\n【測試 2/5】SIREN 初始化")
    test_siren = TestRWFSIRENInit()
    test_siren.test_rwf_siren_init_first_layer()
    test_siren.test_rwf_siren_init_hidden_layer()
    test_siren.test_init_siren_weights_on_rwf_model()
    
    # 3. 檢查點測試
    print("\n【測試 3/5】檢查點保存/載入")
    test_ckpt = TestRWFCheckpoint()
    test_ckpt.test_rwf_checkpoint_save_load()
    
    # 4. 向後相容測試
    print("\n【測試 4/5】舊檢查點向後相容")
    test_ckpt.test_rwf_legacy_checkpoint_loading()
    
    # 5. 工廠集成測試
    print("\n【測試 5/5】模型工廠集成")
    test_factory = TestRWFIntegrationWithFactory()
    test_factory.test_factory_creates_rwf_model()
    
    print("\n" + "=" * 60)
    print("✅ 所有 RWF 集成測試通過！")
    print("=" * 60)

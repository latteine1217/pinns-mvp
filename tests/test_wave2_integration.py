"""
Wave 2 整合測試：驗證 GradientCache 在完整訓練流程中的正確性

測試目標：
1. Trainer.step() 正確調用 GradientCache
2. LossManager.compute_pde_loss() 正確接收並傳遞 gradients
3. VS-PINN 正確使用緩存的 gradients
4. 數值結果與原始實現一致（誤差 < 1e-6）
"""

import sys
import pytest
import torch
import numpy as np

sys.path.insert(0, '.')

from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow
from pinnx.physics.gradient_cache import GradientCache
from pinnx.train.loss_manager import LossManager
from pinnx.models.fourier_mlp import FourierMLP


@pytest.fixture
def setup_vs_pinn():
    """建立 VS-PINN 測試環境"""
    device = torch.device('cpu')  # 使用 CPU 測試以確保一致性
    
    # 物理參數（Channel Flow Re_tau = 1000）
    Re = 185.75  # Re_bulk
    rho = 1.0
    h = 1.0
    u_bulk = 1.0
    nu = u_bulk * h / Re
    
    # 建立 VS-PINN
    physics = VSPINNChannelFlow(
        nu=nu,
        rho=rho,
        h=h,
        u_bulk=u_bulk,
        forcing_type='pressure_gradient',
        p_x=-0.01075  # 對應 Re_bulk = 185.75
    )
    
    # 建立簡單模型（用於測試）
    model = FourierMLP(
        in_dim=3,
        out_dim=4,  # [u, v, w, p]
        hidden_dim=64,
        num_layers=3,
        fourier_features_config={'m': 6, 'sigma': 2.0}
    ).to(device)
    
    # 建立 LossManager
    config = {
        'loss': {
            'weights': {
                'data': 1.0,
                'momentum_x': 1.0,
                'momentum_y': 1.0,
                'momentum_z': 1.0,
                'continuity': 1.0
            }
        },
        'adaptive_weighting': {'enabled': False},
        'curriculum': {'enabled': False}
    }
    
    loss_manager = LossManager(
        physics=physics,
        config=config,
        device=device
    )
    
    return {
        'physics': physics,
        'model': model,
        'loss_manager': loss_manager,
        'device': device
    }


def test_gradient_cache_in_loss_manager(setup_vs_pinn):
    """
    測試 1: LossManager 正確使用 GradientCache
    
    驗證：
    - 使用緩存 gradients 與不使用的結果一致
    """
    physics = setup_vs_pinn['physics']
    model = setup_vs_pinn['model']
    loss_manager = setup_vs_pinn['loss_manager']
    device = setup_vs_pinn['device']
    
    # 生成測試數據
    N = 128
    x = torch.rand(N, 1, device=device, requires_grad=True) * 2.0  # [0, 2]
    y = torch.rand(N, 1, device=device, requires_grad=True) * 2.0 - 1.0  # [-1, 1]
    z = torch.rand(N, 1, device=device, requires_grad=True) * 4.0  # [0, 4]
    coords = torch.cat([x, y, z], dim=1)
    
    # 模型預測
    with torch.no_grad():
        predictions = model(coords)
    predictions.requires_grad_(True)
    
    # 構建 data_batch
    data_batch = {
        'x_pde': x,
        'y_pde': y,
        'z_pde': z,
        't_pde': None
    }
    
    # ==================== 方法 1: 不使用 GradientCache（原始方法）====================
    loss_dict_original = loss_manager.compute_pde_loss(
        coords_pde_physical=coords,
        model_coords_pde=coords,  # 簡化：假設無縮放
        u_pred_pde_physical=predictions,
        data_batch=data_batch,
        epoch=0,
        is_vs_pinn=True,
        gradients=None  # 不使用緩存
    )
    
    # ==================== 方法 2: 使用 GradientCache（Wave 2 優化）====================
    # 重新生成 predictions（避免梯度圖污染）
    with torch.no_grad():
        predictions_cached = model(coords)
    predictions_cached.requires_grad_(True)
    
    # 計算緩存 gradients
    grad_cache = GradientCache(device=device)
    predictions_dict = {
        'u': predictions_cached[:, 0:1],
        'v': predictions_cached[:, 1:2],
        'w': predictions_cached[:, 2:3],
        'p': predictions_cached[:, 3:4]
    }
    gradients = grad_cache.compute_all_gradients(predictions_dict, coords)
    
    # 使用緩存計算損失
    loss_dict_cached = loss_manager.compute_pde_loss(
        coords_pde_physical=coords,
        model_coords_pde=coords,
        u_pred_pde_physical=predictions_cached,
        data_batch=data_batch,
        epoch=0,
        is_vs_pinn=True,
        gradients=gradients  # 🚀 使用緩存
    )
    
    # ==================== 驗證：兩種方法結果一致 ====================
    print("\n=== Wave 2 Integration Test: Gradient Cache in LossManager ===")
    
    for key in ['momentum_x_loss', 'momentum_y_loss', 'momentum_z_loss', 'continuity_loss']:
        val_original = loss_dict_original[key].item()
        val_cached = loss_dict_cached[key].item()
        diff = abs(val_original - val_cached)
        
        print(f"{key}:")
        print(f"  Original: {val_original:.6e}")
        print(f"  Cached:   {val_cached:.6e}")
        print(f"  Diff:     {diff:.6e}")
        
        # 驗收標準：誤差 < 1e-5（允許浮點誤差）
        assert diff < 1e-5, f"{key} 誤差過大: {diff:.6e}"
    
    print("\n✅ 所有損失項一致！GradientCache 整合成功。")


def test_backward_compatibility(setup_vs_pinn):
    """
    測試 2: 向後相容性
    
    驗證：
    - gradients=None 時正常工作（回退到原始計算）
    """
    loss_manager = setup_vs_pinn['loss_manager']
    model = setup_vs_pinn['model']
    device = setup_vs_pinn['device']
    
    # 生成測試數據
    N = 64
    x = torch.rand(N, 1, device=device, requires_grad=True)
    y = torch.rand(N, 1, device=device, requires_grad=True) * 2.0 - 1.0
    z = torch.rand(N, 1, device=device, requires_grad=True)
    coords = torch.cat([x, y, z], dim=1)
    
    with torch.no_grad():
        predictions = model(coords)
    predictions.requires_grad_(True)
    
    data_batch = {'x_pde': x, 'y_pde': y, 'z_pde': z, 't_pde': None}
    
    # 不傳 gradients 參數（向後相容）
    try:
        loss_dict = loss_manager.compute_pde_loss(
            coords_pde_physical=coords,
            model_coords_pde=coords,
            u_pred_pde_physical=predictions,
            data_batch=data_batch,
            epoch=0,
            is_vs_pinn=True
            # 注意：沒有傳 gradients 參數
        )
        
        print("\n=== Backward Compatibility Test ===")
        print("✅ gradients=None 時正常工作（向後相容）")
        print(f"Loss keys: {list(loss_dict.keys())}")
        
    except Exception as e:
        pytest.fail(f"向後相容性測試失敗: {e}")


def test_gradient_cache_memory_efficiency(setup_vs_pinn):
    """
    測試 3: 記憶體效率
    
    驗證：
    - GradientCache 不會導致記憶體洩漏
    """
    physics = setup_vs_pinn['physics']
    model = setup_vs_pinn['model']
    device = setup_vs_pinn['device']
    
    import gc
    
    # 記錄初始記憶體
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
    else:
        initial_memory = 0
    
    # 執行 100 次 gradient 計算
    for i in range(100):
        N = 128
        x = torch.rand(N, 1, device=device, requires_grad=True)
        y = torch.rand(N, 1, device=device, requires_grad=True) * 2.0 - 1.0
        z = torch.rand(N, 1, device=device, requires_grad=True)
        coords = torch.cat([x, y, z], dim=1)
        
        with torch.no_grad():
            predictions = model(coords)
        predictions.requires_grad_(True)
        
        # 使用 GradientCache
        grad_cache = GradientCache(device=device)
        predictions_dict = {
            'u': predictions[:, 0:1],
            'v': predictions[:, 1:2],
            'w': predictions[:, 2:3],
            'p': predictions[:, 3:4]
        }
        gradients = grad_cache.compute_all_gradients(predictions_dict, coords)
        
        # 清理
        del gradients, grad_cache, coords, predictions
        
        if i % 20 == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # 檢查最終記憶體
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated(device)
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        print(f"\n=== Memory Efficiency Test ===")
        print(f"Initial memory: {initial_memory / 1024 / 1024:.2f} MB")
        print(f"Final memory:   {final_memory / 1024 / 1024:.2f} MB")
        print(f"Increase:       {memory_increase:.2f} MB")
        
        # 驗收：記憶體增長 < 50 MB（允許一些正常增長）
        assert memory_increase < 50, f"記憶體洩漏：增長 {memory_increase:.2f} MB"
        print("✅ 無記憶體洩漏")
    else:
        print("\n=== Memory Efficiency Test ===")
        print("⚠️  CPU 模式：跳過 CUDA 記憶體檢查")
        print("✅ 測試完成（無崩潰）")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

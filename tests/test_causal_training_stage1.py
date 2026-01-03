"""
Stage 1 集成測試：驗證優化後的 CausalWeighter 在訓練流程中正常工作

測試內容：
1. 配置文件正確傳遞參數（epsilon, n_bins, time_range, device）
2. 因果矩陣成功預計算並移動到正確設備
3. 訓練初期權重計算正常（無性能退化）
4. 設備切換正常（CPU ↔ CUDA/MPS）
"""

import pytest
import torch
import yaml
from pathlib import Path

# 從實際訓練腳本導入
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from pinnx.losses.weighting import CausalWeighter


def test_causal_weighter_config_integration():
    """測試從配置文件正確初始化 CausalWeighter"""
    
    # 模擬配置（與 kolmogorov_re50_kf4_K100.yml 一致）
    config = {
        'data': {
            'kolmogorov_config': {
                'time_range': [15.0, 35.0]
            }
        },
        'losses': {
            'causal_weighting': True,
            'causal_eps': 1.5,
            'causal_n_bins': 20
        }
    }
    
    loss_cfg = config['losses']
    kol_cfg = config['data']['kolmogorov_config']
    time_range = kol_cfg['time_range']
    t_min, t_max = time_range
    device = 'cpu'
    
    # 初始化（模擬 train.py Line 1072-1078）
    weighter = CausalWeighter(
        epsilon=loss_cfg['causal_eps'],
        n_time_bins=loss_cfg['causal_n_bins'],
        t_min=t_min,
        t_max=t_max,
        device=device
    )
    
    # 驗證
    assert weighter.epsilon == 1.5
    assert weighter.num_chunks == 20
    assert weighter.t_min == 15.0
    assert weighter.t_max == 35.0
    assert weighter.causal_matrix.shape == (20, 20)
    assert weighter.causal_matrix.device.type == 'cpu'
    
    print("✅ 配置集成測試通過")


def test_causal_weighter_precomputed_performance():
    """測試預計算因果矩陣的性能優勢"""
    
    import time
    
    # 大量採樣點（真實訓練場景）
    N = 20000
    device = 'cpu'
    
    pde_losses = torch.rand(N, 1).to(device)
    time_coords = torch.linspace(15, 35, N).unsqueeze(1).to(device)
    
    weighter = CausalWeighter(
        epsilon=1.5,
        num_chunks=20,
        t_min=15.0,
        t_max=35.0,
        device=device
    )
    
    # 測試 10 次計算（模擬訓練多個 iteration）
    times = []
    for _ in range(10):
        start = time.time()
        weights = weighter.compute_weights(pde_losses, time_coords)
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # ms
    
    avg_time = sum(times) / len(times)
    
    # 驗證
    assert weights.shape == (N, 1)
    assert avg_time < 50.0  # 應該 < 50ms（優化前可能 > 100ms）
    
    print(f"✅ 性能測試通過：平均 {avg_time:.2f} ms/iteration")


def test_causal_weighter_device_switching():
    """測試設備切換功能"""
    
    weighter = CausalWeighter(
        epsilon=1.5,
        num_chunks=20,
        t_min=15.0,
        t_max=35.0,
        device='cpu'
    )
    
    assert weighter.causal_matrix.device.type == 'cpu'
    
    # 測試 to(device) 方法
    if torch.cuda.is_available():
        weighter.to('cuda')
        assert weighter.causal_matrix.device.type == 'cuda'
        
        weighter.to('cpu')
        assert weighter.causal_matrix.device.type == 'cpu'
        
        print("✅ CUDA 設備切換測試通過")
    
    elif torch.backends.mps.is_available():
        weighter.to('mps')
        assert weighter.causal_matrix.device.type == 'mps'
        
        weighter.to('cpu')
        assert weighter.causal_matrix.device.type == 'cpu'
        
        print("✅ MPS 設備切換測試通過")
    
    else:
        print("⚠️  僅 CPU 可用，跳過設備切換測試")


def test_causal_weighter_auto_device_matching():
    """測試自動設備匹配（compute_weights 內部）"""
    
    # 在 CPU 初始化
    weighter = CausalWeighter(
        epsilon=1.5,
        num_chunks=20,
        device='cpu'
    )
    
    # 測試 CPU 輸入
    pde_losses = torch.rand(1000, 1)
    time_coords = torch.linspace(0, 10, 1000).unsqueeze(1)
    weights = weighter.compute_weights(pde_losses, time_coords)
    
    assert weights.device.type == 'cpu'
    
    # 測試設備不匹配情況（如果 CUDA/MPS 可用）
    if torch.cuda.is_available():
        pde_losses_cuda = pde_losses.to('cuda')
        time_coords_cuda = time_coords.to('cuda')
        
        # compute_weights 應該自動移動 causal_matrix 到 CUDA
        weights_cuda = weighter.compute_weights(pde_losses_cuda, time_coords_cuda)
        assert weights_cuda.device.type == 'cuda'
        assert weighter.causal_matrix.device.type == 'cuda'  # 已自動移動
        
        print("✅ CUDA 自動設備匹配測試通過")
    
    elif torch.backends.mps.is_available():
        pde_losses_mps = pde_losses.to('mps')
        time_coords_mps = time_coords.to('mps')
        
        weights_mps = weighter.compute_weights(pde_losses_mps, time_coords_mps)
        assert weights_mps.device.type == 'mps'
        assert weighter.causal_matrix.device.type == 'mps'
        
        print("✅ MPS 自動設備匹配測試通過")
    
    else:
        print("⚠️  僅 CPU 可用，跳過自動設備匹配測試")


def test_causal_weights_causality_property():
    """驗證權重符合因果性（早期 > 後期）"""
    
    N = 10000
    weighter = CausalWeighter(epsilon=1.5, num_chunks=20, device='cpu')
    
    # 模擬均勻損失分佈
    pde_losses = torch.ones(N, 1) * 0.5
    time_coords = torch.linspace(0, 10, N).unsqueeze(1)
    
    # 取 chunk-level 權重
    chunk_weights, _ = weighter.compute_weights(
        pde_losses, time_coords, return_pointwise=False
    )
    
    # 驗證遞減性
    assert chunk_weights[0] > chunk_weights[-1], "權重應該遞減（因果性）"
    
    # 驗證第一個 chunk 權重為 1（無先前損失）
    assert torch.isclose(chunk_weights[0], torch.tensor(1.0), atol=1e-5)
    
    print(f"✅ 因果性驗證通過：w[0]={chunk_weights[0]:.4f}, w[-1]={chunk_weights[-1]:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 1 集成測試：CausalWeighter v2.0")
    print("=" * 60)
    
    test_causal_weighter_config_integration()
    test_causal_weighter_precomputed_performance()
    test_causal_weighter_device_switching()
    test_causal_weighter_auto_device_matching()
    test_causal_weights_causality_property()
    
    print("\n" + "=" * 60)
    print("✅ 所有 Stage 1 測試通過！")
    print("=" * 60)
    print("\n下一步：運行完整訓練測試以驗證端到端集成")

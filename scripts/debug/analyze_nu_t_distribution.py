"""分析檢查點中 ν_t/ν 的實際分布"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def analyze_checkpoint(checkpoint_path):
    """分析檢查點中的 ν_t/ν 分布"""
    print("=" * 80)
    print(f"📊 分析檢查點：{checkpoint_path}")
    print("=" * 80)
    
    # 載入檢查點
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # 創建測試網格
    n_points = 2048
    x = torch.linspace(0, 8*np.pi, n_points)
    y = torch.linspace(-1, 1, n_points)
    z = torch.linspace(0, 3*np.pi, n_points)
    t = torch.zeros(n_points)
    
    # 隨機採樣（模擬訓練時的採樣）
    indices = torch.randperm(n_points)[:n_points]
    coords = torch.stack([
        x[indices],
        y[indices],
        z[indices],
        t[indices]
    ], dim=1)
    
    # 假設模型在 state_dict 中
    from pinnx.models.fourier_mlp import FourierMLP
    
    # 嘗試從配置重建模型
    model = FourierMLP(
        input_dim=4,
        output_dim=4,
        hidden_layers=[200] * 8,
        activation='sine',
        fourier_features={'enabled': True, 'num_frequencies': 500, 'trainable': False}
    )
    
    # 載入權重
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # 前向傳播
    with torch.no_grad():
        outputs = model(coords)
        u, v, w, p = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
    
    # 計算渦量和應變率（簡化版本，完整計算需要梯度）
    # 這裡我們檢查檢查點中是否有保存的 ν_t 數據
    if 'turbulent_viscosity_stats' in ckpt:
        nu_t_stats = ckpt['turbulent_viscosity_stats']
        print("\n✅ 檢查點包含 ν_t 統計數據：")
        for key, value in nu_t_stats.items():
            print(f"   {key}: {value}")
    else:
        print("\n⚠️ 檢查點未包含 ν_t 統計數據")
    
    # 檢查損失記錄
    if 'loss_history' in ckpt:
        history = ckpt['loss_history']
        if 'turbulent_viscosity_loss' in history:
            tv_loss = history['turbulent_viscosity_loss']
            print(f"\n📈 turbulent_viscosity_loss 歷史（最後 10 個）：")
            for i, loss in enumerate(tv_loss[-10:], start=len(tv_loss)-10):
                print(f"   Epoch {i}: {loss:.2f}")
    
    return ckpt

if __name__ == "__main__":
    checkpoint_path = "checkpoints/test_rans_phase6c_v2/epoch_50.pth"
    analyze_checkpoint(checkpoint_path)

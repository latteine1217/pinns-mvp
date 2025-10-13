"""
診斷壓力場預測失敗問題
- 檢查模型輸出層權重分布
- 檢查梯度流
- 檢查損失項權重配置
- 檢查訓練數據統計
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pinnx.models.fourier_mlp import FourierMLP


def load_checkpoint_and_config(checkpoint_path: str):
    """載入檢查點與配置"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config')
    
    print("=" * 60)
    print("檢查點資訊")
    print("=" * 60)
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Loss: {checkpoint['loss']:.6f}")
    print(f"Config keys: {list(config.keys()) if config else 'None'}")
    print()
    
    return checkpoint, config


def analyze_model_weights(checkpoint, config):
    """分析模型權重，特別關注輸出層"""
    
    # 重建模型結構
    model_config = config['model']
    model = FourierMLP(
        in_dim=model_config['in_dim'],
        out_dim=model_config['out_dim'],
        width=model_config['width'],
        depth=model_config['depth'],
        activation=model_config.get('activation', 'tanh'),
        fourier_m=model_config.get('fourier_m', 48),
        fourier_sigma=model_config.get('fourier_sigma', 3.0),
        scaling_config=model_config.get('scaling', {})
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("=" * 60)
    print("模型權重分析")
    print("=" * 60)
    
    # 檢查輸出層權重
    output_layer = model.output_layer
    weight = output_layer.weight.data  # [out_dim=3, hidden_dim]
    bias = output_layer.bias.data      # [out_dim=3]
    
    print(f"輸出層權重形狀: {weight.shape}")
    print(f"輸出層偏置形狀: {bias.shape}")
    print()
    
    # 分析每個輸出通道（u, v, p）
    output_names = ['u', 'v', 'p']
    for i, name in enumerate(output_names):
        w = weight[i]  # [hidden_dim]
        b = bias[i].item()
        
        print(f"{name.upper()} 通道:")
        print(f"  權重統計:")
        print(f"    均值: {w.mean():.6f}")
        print(f"    標準差: {w.std():.6f}")
        print(f"    最小值: {w.min():.6f}")
        print(f"    最大值: {w.max():.6f}")
        print(f"    L2 範數: {w.norm():.6f}")
        print(f"  偏置: {b:.6f}")
        
        # 檢查權重是否過小（可能導致梯度消失）
        if w.std() < 0.01:
            print(f"  ⚠️ 警告：{name} 權重標準差過小 ({w.std():.6f})")
        if abs(w.mean()) < 0.001:
            print(f"  ⚠️ 警告：{name} 權重均值接近零 ({w.mean():.6f})")
        
        print()
    
    # 檢查隱藏層權重範數（判斷是否訓練充分）
    print("隱藏層權重範數:")
    for name, param in model.named_parameters():
        if 'weight' in name and 'hidden' in name:
            print(f"  {name}: {param.data.norm():.4f}")
    print()
    
    return model


def analyze_training_data(config):
    """分析訓練數據統計"""
    
    print("=" * 60)
    print("訓練數據配置")
    print("=" * 60)
    
    # 輸出標準化範圍
    output_norm = config['model']['scaling'].get('output_norm', {})
    print("輸出標準化範圍:")
    for var, range_val in output_norm.items():
        if isinstance(range_val, list):
            print(f"  {var}: [{range_val[0]:.2f}, {range_val[1]:.2f}]  (範圍: {range_val[1] - range_val[0]:.2f})")
    print()
    
    # 感測點配置
    sensors = config.get('sensors', {})
    print(f"感測點數量 K: {sensors.get('K', 'N/A')}")
    print(f"選擇方法: {sensors.get('selection_method', 'N/A')}")
    print()


def analyze_loss_configuration(config):
    """分析損失函數配置"""
    
    print("=" * 60)
    print("損失函數配置分析")
    print("=" * 60)
    
    # 檢查各階段權重
    curriculum = config.get('curriculum', {})
    if curriculum.get('enable', False):
        stages = curriculum.get('stages', [])
        
        print("課程訓練各階段權重:")
        for stage in stages:
            name = stage.get('name', 'Unknown')
            weights = stage.get('weights', {})
            
            print(f"\n  {name}:")
            print(f"    data: {weights.get('data', 'N/A')}")
            print(f"    wall_constraint: {weights.get('wall_constraint', 'N/A')}")
            print(f"    momentum_x: {weights.get('momentum_x', 'N/A')}")
            print(f"    momentum_y: {weights.get('momentum_y', 'N/A')}")
            print(f"    continuity: {weights.get('continuity', 'N/A')}")
            
            # 計算總權重
            total_weight = sum([
                weights.get('data', 0),
                weights.get('wall_constraint', 0),
                weights.get('periodicity', 0),
                weights.get('momentum_x', 0),
                weights.get('momentum_y', 0),
                weights.get('continuity', 0),
            ])
            print(f"    總權重: {total_weight}")
            
            # ⚠️ 檢查是否有壓力相關的約束
            print(f"    ⚠️ 是否有壓力專屬約束: {'pressure_constraint' in weights or 'pressure_bc' in weights}")
    
    print()


def test_model_output_range(model, config):
    """測試模型輸出範圍"""
    
    print("=" * 60)
    print("模型輸出範圍測試")
    print("=" * 60)
    
    model.eval()
    
    # 生成測試點
    domain = config['physics']['domain']
    x_range = domain['x_range']
    y_range = domain['y_range']
    
    nx, ny = 128, 64
    x = torch.linspace(x_range[0], x_range[1], nx)
    y = torch.linspace(y_range[0], y_range[1], ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    coords = torch.stack([X.flatten(), Y.flatten()], dim=-1)  # [N, 2]
    
    with torch.no_grad():
        output = model(coords)  # [N, 3]
    
    u_pred = output[:, 0].numpy()
    v_pred = output[:, 1].numpy()
    p_pred = output[:, 2].numpy()
    
    print(f"U 預測範圍: [{u_pred.min():.4f}, {u_pred.max():.4f}]  (範圍: {u_pred.max() - u_pred.min():.4f})")
    print(f"V 預測範圍: [{v_pred.min():.4f}, {v_pred.max():.4f}]  (範圍: {v_pred.max() - v_pred.min():.4f})")
    print(f"P 預測範圍: [{p_pred.min():.4f}, {p_pred.max():.4f}]  (範圍: {p_pred.max() - p_pred.min():.4f})")
    print()
    
    # 檢查輸出標準化範圍
    output_norm = config['model']['scaling'].get('output_norm', {})
    print("期望輸出範圍 (配置):")
    print(f"  U: {output_norm.get('u', 'N/A')}")
    print(f"  V: {output_norm.get('v', 'N/A')}")
    print(f"  P: {output_norm.get('p', 'N/A')}")
    print()
    
    # 計算空間變異係數
    print("空間變異係數 (std/mean):")
    print(f"  U: {u_pred.std() / (abs(u_pred.mean()) + 1e-8):.4f}")
    print(f"  V: {v_pred.std() / (abs(v_pred.mean()) + 1e-8):.4f}")
    print(f"  P: {p_pred.std() / (abs(p_pred.mean()) + 1e-8):.4f}")
    
    # ⚠️ 壓力場幾乎無變化是主要問題
    if p_pred.std() < 1.0:
        print(f"  🚨 嚴重問題：壓力場標準差僅 {p_pred.std():.4f}，幾乎無空間變化！")
    
    print()
    
    return u_pred.reshape(nx, ny), v_pred.reshape(nx, ny), p_pred.reshape(nx, ny)


def visualize_output_distribution(u, v, p, output_dir: Path):
    """視覺化輸出分布"""
    
    print("=" * 60)
    print("生成視覺化圖表")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 場分布
    im0 = axes[0, 0].imshow(u.T, origin='lower', aspect='auto', cmap='viridis')
    axes[0, 0].set_title('U Field')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(v.T, origin='lower', aspect='auto', cmap='viridis')
    axes[0, 1].set_title('V Field')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(p.T, origin='lower', aspect='auto', cmap='viridis')
    axes[0, 2].set_title('P Field (⚠️ 問題區域)')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # 直方圖
    axes[1, 0].hist(u.flatten(), bins=50, alpha=0.7, color='blue')
    axes[1, 0].set_title('U Distribution')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(v.flatten(), bins=50, alpha=0.7, color='green')
    axes[1, 1].set_title('V Distribution')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].hist(p.flatten(), bins=50, alpha=0.7, color='red')
    axes[1, 2].set_title('P Distribution (⚠️ 範圍極窄)')
    axes[1, 2].set_xlabel('Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / "model_output_diagnosis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 圖表已儲存: {save_path}")
    
    plt.close()


def main():
    checkpoint_path = "checkpoints/pinnx_channel_flow_curriculum_latest.pth"
    output_dir = Path("evaluation_results/pressure_diagnosis")
    
    print("\n" + "=" * 60)
    print("壓力場預測失敗診斷工具")
    print("=" * 60 + "\n")
    
    # 1. 載入檢查點
    checkpoint, config = load_checkpoint_and_config(checkpoint_path)
    
    # 2. 分析訓練數據配置
    analyze_training_data(config)
    
    # 3. 分析損失函數配置
    analyze_loss_configuration(config)
    
    # 4. 分析模型權重
    model = analyze_model_weights(checkpoint, config)
    
    # 5. 測試模型輸出範圍
    u, v, p = test_model_output_range(model, config)
    
    # 6. 視覺化
    visualize_output_distribution(u, v, p, output_dir)
    
    print("\n" + "=" * 60)
    print("診斷完成")
    print("=" * 60)
    print("\n建議下一步:")
    print("1. 檢查是否缺少壓力邊界條件約束")
    print("2. 檢查 momentum 方程中壓力梯度項的計算")
    print("3. 檢查感測點是否包含足夠的壓力資訊")
    print("4. 考慮添加壓力正則化項或壓力範圍約束")
    print()


if __name__ == "__main__":
    main()

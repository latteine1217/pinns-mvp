#!/usr/bin/env python3
"""
Fourier Features 對比實驗評估腳本

比較 Baseline (無 Fourier) 與 Fourier (有 Fourier) 的訓練結果：
1. 載入兩個模型的最終檢查點
2. 在相同測試集上評估
3. 生成對比分析報告

使用方式:
    python scripts/compare_fourier_experiments.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
import re
import logging

# 反標準化工具
from pinnx.utils.denormalization import denormalize_output

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ============================================================
# 配置
# ============================================================

BASELINE_CONFIG = "configs/vs_pinn_baseline_1k.yml"
FOURIER_CONFIG = "configs/vs_pinn_fourier_1k.yml"

BASELINE_CHECKPOINT = "checkpoints/vs_pinn_baseline_1k_latest.pth"
FOURIER_CHECKPOINT = "checkpoints/vs_pinn_fourier_1k_latest.pth"

TEST_DATA = "data/jhtdb/channel_flow_re1000/cutout3d_128x128x32.npz"

BASELINE_LOG = "log/baseline_1k_training.log"
FOURIER_LOG = "log/fourier_1k_training.log"

OUTPUT_DIR = Path("results/fourier_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "comparison_report.md"

# ============================================================
# 工具函數
# ============================================================

def load_config(config_path):
    """載入 YAML 配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_test_data(data_path):
    """載入測試資料"""
    print(f"\n📂 載入測試資料: {data_path}")
    data = np.load(data_path)
    
    # 檢查資料格式
    if 'coords' in data and 'u' in data:
        # 格式 1: coords, u, v, w, p
        coords = data['coords']
        u_true = data['u']
        v_true = data['v']
        w_true = data['w'] if 'w' in data else None
        p_true = data['p']
    elif 'x' in data and 'y' in data:
        # 格式 2: x, y, z (網格)
        x = data['x']
        y = data['y']
        z = data['z'] if 'z' in data else None
        
        if z is not None:
            # 3D 網格
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        else:
            # 2D 網格
            X, Y = np.meshgrid(x, y, indexing='ij')
            coords = np.stack([X.ravel(), Y.ravel()], axis=-1)
        
        u_true = data['u'].ravel()
        v_true = data['v'].ravel()
        w_true = data['w'].ravel() if 'w' in data else None
        p_true = data['p'].ravel()
    else:
        raise ValueError(f"無法識別資料格式: {list(data.keys())}")
    
    n_points = coords.shape[0]
    n_vars = 4 if w_true is not None else 3
    
    print(f"  ✅ 座標形狀: {coords.shape}")
    print(f"  ✅ 測試點數: {n_points}")
    print(f"  ✅ 變數數量: {n_vars} {'(u,v,w,p)' if w_true is not None else '(u,v,p)'}")
    
    # 計算真實值範圍（用於後處理縮放）
    true_ranges = {
        'u': (float(u_true.min()), float(u_true.max())),
        'v': (float(v_true.min()), float(v_true.max())),
    }
    if w_true is not None:
        true_ranges['w'] = (float(w_true.min()), float(w_true.max()))
    true_ranges['p'] = (float(p_true.min()), float(p_true.max()))
    
    return {
        'coords': coords,
        'u': u_true,
        'v': v_true,
        'w': w_true,
        'p': p_true,
        'n_points': n_points,
        'is_3d': w_true is not None,
        'true_ranges': true_ranges  # 新增
    }

def load_model(config_path, checkpoint_path, device):
    """載入模型（簡化版，適用於評估）"""
    from pinnx.models.fourier_mlp import PINNNet
    
    print(f"\n🔧 載入模型: {checkpoint_path}")
    
    # 載入配置
    config = load_config(config_path)
    model_cfg = config['model']
    
    # 創建模型
    model = PINNNet(
        in_dim=model_cfg['in_dim'],
        out_dim=model_cfg['out_dim'],
        width=model_cfg['width'],
        depth=model_cfg['depth'],
        activation=model_cfg.get('activation', 'sine'),
        use_fourier=model_cfg.get('use_fourier', False),
        fourier_m=model_cfg.get('fourier_m', 0),
        fourier_sigma=model_cfg.get('fourier_sigma', 1.0),
    ).to(device)
    
    # 載入權重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        # 使用 strict=False 允許額外的 VS-PINN 參數（如 input_scale_factors）
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if unexpected_keys:
            print(f"  ℹ️  跳過的額外參數: {unexpected_keys}")
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ 模型參數: {n_params:,}")
    print(f"  ✅ Fourier Features: {'啟用' if model_cfg.get('use_fourier', False) else '禁用'}")
    
    return model, config

def predict(model, coords, device, batch_size=4096):
    """批次預測"""
    n_points = coords.shape[0]
    n_batches = (n_points + batch_size - 1) // batch_size
    
    predictions = []
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_points)
            
            batch_coords = torch.tensor(coords[start_idx:end_idx], dtype=torch.float32, device=device)
            batch_pred = model(batch_coords)
            predictions.append(batch_pred.cpu().numpy())
    
    return np.vstack(predictions)

def compute_metrics(pred, true, var_name):
    """計算評估指標"""
    # 相對 L2 誤差
    rel_l2 = np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-10)
    
    # RMSE
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    
    # 最大誤差
    max_error = np.max(np.abs(pred - true))
    
    # 平均絕對誤差
    mae = np.mean(np.abs(pred - true))
    
    return {
        'rel_l2': rel_l2 * 100,  # 轉為百分比
        'rmse': rmse,
        'max_error': max_error,
        'mae': mae
    }

def parse_training_log(log_path):
    """解析訓練日誌，提取損失曲線"""
    epochs = []
    total_losses = []
    residual_losses = []
    data_losses = []
    conservation_errors = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # 解析 epoch 損失
            epoch_match = re.search(r'Epoch\s+(\d+)\s+\|\s+Total:\s+([\d.]+)\s+\|\s+Residual:\s+([\d.]+)\s+.*Data:\s+([\d.]+)', line)
            if epoch_match:
                epochs.append(int(epoch_match.group(1)))
                total_losses.append(float(epoch_match.group(2)))
                residual_losses.append(float(epoch_match.group(3)))
                data_losses.append(float(epoch_match.group(4)))
            
            # 解析最佳守恆誤差
            cons_match = re.search(r'New best conservation_error:\s+([\d.]+)\s+at epoch\s+(\d+)', line)
            if cons_match:
                error = float(cons_match.group(1))
                epoch = int(cons_match.group(2))
                conservation_errors.append((epoch, error))
    
    return {
        'epochs': np.array(epochs),
        'total_loss': np.array(total_losses),
        'residual_loss': np.array(residual_losses),
        'data_loss': np.array(data_losses),
        'conservation_errors': conservation_errors
    }

# ============================================================
# 主評估函數
# ============================================================

def main():
    print("=" * 60)
    print("📊 Fourier Features 對比實驗評估")
    print("=" * 60)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n🖥️  使用設備: {device}")
    
    # ----------------------------------------
    # 1. 載入測試資料
    # ----------------------------------------
    test_data = load_test_data(TEST_DATA)
    
    # ----------------------------------------
    # 2. 載入模型
    # ----------------------------------------
    baseline_model, baseline_config = load_model(BASELINE_CONFIG, BASELINE_CHECKPOINT, device)
    fourier_model, fourier_config = load_model(FOURIER_CONFIG, FOURIER_CHECKPOINT, device)
    
    # ----------------------------------------
    # 3. 預測
    # ----------------------------------------
    print("\n" + "=" * 60)
    print("🔮 開始預測...")
    print("=" * 60)
    
    print("\n📊 Baseline 預測中...")
    baseline_pred = predict(baseline_model, test_data['coords'], device)
    
    # 🔧 後處理縮放 Baseline 預測（自動範圍映射）
    print("  🔄 執行後處理縮放 (post_scaling)...")
    baseline_pred = denormalize_output(
        baseline_pred, 
        baseline_config, 
        output_norm_type='post_scaling',
        true_ranges=test_data['true_ranges'],
        verbose=False
    )
    print(f"  📊 Baseline 縮放後範圍: u=[{baseline_pred[:, 0].min():.2f}, {baseline_pred[:, 0].max():.2f}], "
          f"p=[{baseline_pred[:, 3].min():.2f}, {baseline_pred[:, 3].max():.2f}]")
    
    print("\n📊 Fourier 預測中...")
    fourier_pred = predict(fourier_model, test_data['coords'], device)
    
    # 🔧 後處理縮放 Fourier 預測（自動範圍映射）
    print("  🔄 執行後處理縮放 (post_scaling)...")
    fourier_pred = denormalize_output(
        fourier_pred, 
        fourier_config, 
        output_norm_type='post_scaling',
        true_ranges=test_data['true_ranges'],
        verbose=False
    )
    print(f"  📊 Fourier 縮放後範圍: u=[{fourier_pred[:, 0].min():.2f}, {fourier_pred[:, 0].max():.2f}], "
          f"p=[{fourier_pred[:, 3].min():.2f}, {fourier_pred[:, 3].max():.2f}]")
    
    # 📊 真實值範圍（驗證）
    print(f"\n  📊 真實值範圍: u=[{test_data['u'].min():.2f}, {test_data['u'].max():.2f}], "
          f"p=[{test_data['p'].min():.2f}, {test_data['p'].max():.2f}]")
    
    # ----------------------------------------
    # 4. 計算指標
    # ----------------------------------------
    print("\n" + "=" * 60)
    print("📈 計算評估指標...")
    print("=" * 60)
    
    results = {
        'baseline': {},
        'fourier': {}
    }
    
    var_names = ['u', 'v', 'w', 'p'] if test_data['is_3d'] else ['u', 'v', 'p']
    
    for i, var in enumerate(var_names):
        if var == 'w' and test_data['w'] is None:
            continue
        
        true_data = test_data[var]
        
        baseline_metrics = compute_metrics(baseline_pred[:, i], true_data, var)
        fourier_metrics = compute_metrics(fourier_pred[:, i], true_data, var)
        
        results['baseline'][var] = baseline_metrics
        results['fourier'][var] = fourier_metrics
        
        print(f"\n{var.upper()} 速度:" if var != 'p' else f"\n壓力:")
        print(f"  Baseline - 相對 L2: {baseline_metrics['rel_l2']:.2f}%")
        print(f"  Fourier  - 相對 L2: {fourier_metrics['rel_l2']:.2f}%")
    
    # ----------------------------------------
    # 5. 解析訓練日誌
    # ----------------------------------------
    print("\n" + "=" * 60)
    print("📜 解析訓練日誌...")
    print("=" * 60)
    
    baseline_log = parse_training_log(BASELINE_LOG)
    fourier_log = parse_training_log(FOURIER_LOG)
    
    print(f"\n  Baseline 訓練 epochs: {len(baseline_log['epochs'])}")
    print(f"  Fourier 訓練 epochs: {len(fourier_log['epochs'])}")
    
    # ----------------------------------------
    # 6. 生成對比圖表
    # ----------------------------------------
    print("\n" + "=" * 60)
    print("📊 生成對比圖表...")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 6.1 Total Loss
    ax = axes[0, 0]
    ax.plot(baseline_log['epochs'], baseline_log['total_loss'], label='Baseline', linewidth=2)
    ax.plot(fourier_log['epochs'], fourier_log['total_loss'], label='Fourier', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 6.2 Residual Loss
    ax = axes[0, 1]
    ax.plot(baseline_log['epochs'], baseline_log['residual_loss'], label='Baseline', linewidth=2)
    ax.plot(fourier_log['epochs'], fourier_log['residual_loss'], label='Fourier', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Residual Loss')
    ax.set_title('Physics Residual Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 6.3 Data Loss
    ax = axes[1, 0]
    ax.plot(baseline_log['epochs'], baseline_log['data_loss'], label='Baseline', linewidth=2)
    ax.plot(fourier_log['epochs'], fourier_log['data_loss'], label='Fourier', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Data Loss')
    ax.set_title('Data Fitting Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6.4 相對 L2 誤差對比（bar chart）
    ax = axes[1, 1]
    x_pos = np.arange(len(var_names))
    baseline_errors = [results['baseline'][v]['rel_l2'] for v in var_names]
    fourier_errors = [results['fourier'][v]['rel_l2'] for v in var_names]
    
    width = 0.35
    ax.bar(x_pos - width/2, baseline_errors, width, label='Baseline', alpha=0.8)
    ax.bar(x_pos + width/2, fourier_errors, width, label='Fourier', alpha=0.8)
    ax.set_xlabel('Variable')
    ax.set_ylabel('Relative L2 Error (%)')
    ax.set_title('Final Prediction Error')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(var_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = OUTPUT_DIR / "training_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 圖表已保存: {plot_path}")
    
    # ----------------------------------------
    # 7. 生成 Markdown 報告
    # ----------------------------------------
    print("\n" + "=" * 60)
    print("📝 生成評估報告...")
    print("=" * 60)
    
    with open(REPORT_FILE, 'w') as f:
        f.write("# Fourier Features 對比實驗報告\n\n")
        f.write(f"**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 📋 實驗配置\n\n")
        f.write("| 項目 | Baseline | Fourier |\n")
        f.write("|------|----------|----------|\n")
        f.write(f"| **Fourier Features** | 禁用 | 啟用 |\n")
        f.write(f"| **Fourier M** | 0 | 64 |\n")
        f.write(f"| **Fourier Sigma** | - | 5.0 |\n")
        f.write(f"| **模型架構** | {baseline_config['model']['depth']}×{baseline_config['model']['width']} | {fourier_config['model']['depth']}×{fourier_config['model']['width']} |\n")
        f.write(f"| **激活函數** | {baseline_config['model']['activation']} | {fourier_config['model']['activation']} |\n")
        f.write(f"| **訓練 Epochs** | {len(baseline_log['epochs'])} | {len(fourier_log['epochs'])} |\n")
        f.write(f"| **感測點數 (K)** | 30 | 30 |\n\n")
        
        f.write("## 📊 預測誤差對比\n\n")
        f.write("**測試集**: 3D Cutout (128×128×32 = 524,288 點)\n\n")
        f.write("| 變數 | Baseline L2 (%) | Fourier L2 (%) | 改善 |\n")
        f.write("|------|----------------|---------------|------|\n")
        
        for var in var_names:
            baseline_l2 = results['baseline'][var]['rel_l2']
            fourier_l2 = results['fourier'][var]['rel_l2']
            improvement = ((baseline_l2 - fourier_l2) / baseline_l2) * 100 if baseline_l2 > 0 else 0
            improvement_str = f"{'↓' if improvement > 0 else '↑'} {abs(improvement):.1f}%"
            
            f.write(f"| **{var.upper()}** | {baseline_l2:.2f} | {fourier_l2:.2f} | {improvement_str} |\n")
        
        f.write("\n## 📈 訓練效率對比\n\n")
        
        baseline_final_loss = baseline_log['total_loss'][-1] if len(baseline_log['total_loss']) > 0 else 0
        fourier_final_loss = fourier_log['total_loss'][-1] if len(fourier_log['total_loss']) > 0 else 0
        
        baseline_best_cons = min([e[1] for e in baseline_log['conservation_errors']]) if baseline_log['conservation_errors'] else 0
        fourier_best_cons = min([e[1] for e in fourier_log['conservation_errors']]) if fourier_log['conservation_errors'] else 0
        
        f.write("| 指標 | Baseline | Fourier |\n")
        f.write("|------|----------|----------|\n")
        f.write(f"| **最終 Total Loss** | {baseline_final_loss:.4f} | {fourier_final_loss:.4f} |\n")
        f.write(f"| **最佳 Conservation Error** | {baseline_best_cons:.6f} | {fourier_best_cons:.6f} |\n")
        f.write(f"| **訓練 Epochs** | {len(baseline_log['epochs'])} | {len(fourier_log['epochs'])} |\n")
        
        f.write("\n## 📊 可視化\n\n")
        f.write(f"![訓練對比]({plot_path.name})\n\n")
        
        f.write("## 🎯 結論\n\n")
        
        # 計算平均改善
        avg_improvement = np.mean([
            ((results['baseline'][v]['rel_l2'] - results['fourier'][v]['rel_l2']) / results['baseline'][v]['rel_l2']) * 100
            for v in var_names
        ])
        
        if avg_improvement > 0:
            f.write(f"✅ **Fourier Features 帶來顯著改善**：平均相對 L2 誤差下降 **{avg_improvement:.1f}%**\n\n")
        elif avg_improvement < -5:
            f.write(f"❌ **Fourier Features 效果不佳**：平均相對 L2 誤差上升 **{abs(avg_improvement):.1f}%**\n\n")
        else:
            f.write(f"⚖️ **Fourier Features 影響不大**：平均相對 L2 誤差變化 **{avg_improvement:.1f}%**\n\n")
        
        f.write("---\n\n")
        f.write("**報告結束**\n")
    
    print(f"  ✅ 報告已保存: {REPORT_FILE}")
    
    print("\n" + "=" * 60)
    print("✅ 評估完成！")
    print("=" * 60)
    print(f"\n📁 結果位置: {OUTPUT_DIR}")
    print(f"   - 報告: {REPORT_FILE.name}")
    print(f"   - 圖表: training_comparison.png")

if __name__ == "__main__":
    main()

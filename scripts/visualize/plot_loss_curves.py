#!/usr/bin/env python3
"""
Loss Curves Visualization from pinnx.log
提取訓練日誌中的所有 loss 並以 log scale 繪圖
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_log_line(line):
    """解析一行訓練日誌，提取 epoch 和各項 loss"""
    # Epoch 格式: Epoch 0/10000 | total_loss: 8.197990 | ...
    epoch_match = re.search(r'Epoch (\d+)/\d+', line)
    if not epoch_match:
        return None
    
    epoch = int(epoch_match.group(1))
    data: dict = {'epoch': epoch}
    
    # 提取所有 loss 項目（格式：key: value）
    loss_pattern = r'(\w+_loss|\w+_weighted_loss|lr): ([\d.e+-]+)'
    for match in re.finditer(loss_pattern, line):
        key = match.group(1)
        value = float(match.group(2))
        data[key] = value
    
    return data

def load_training_log(log_path):
    """從日誌文件載入所有訓練數據"""
    data_list = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Epoch' in line and '/10000' in line:
                parsed = parse_log_line(line)
                if parsed:
                    data_list.append(parsed)
    
    return data_list

def extract_arrays(data_list):
    """將數據列表轉換為 numpy arrays"""
    if not data_list:
        return {}
    
    # 獲取所有可能的 keys
    all_keys = set()
    for d in data_list:
        all_keys.update(d.keys())
    
    arrays = {}
    for key in all_keys:
        values = [d.get(key, np.nan) for d in data_list]
        arrays[key] = np.array(values)
    
    return arrays

def plot_loss_curves(arrays, output_dir):
    """繪製所有 loss curves (log scale)"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = arrays['epoch']
    
    # ====================================================================
    # 圖 1: 總體損失 (Total, Data, PDE, Prior)
    # ====================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'total_loss' in arrays:
        ax.semilogy(epochs, arrays['total_loss'], label='Total Loss', linewidth=2, alpha=0.9)
    if 'data_loss' in arrays:
        ax.semilogy(epochs, arrays['data_loss'], label='Data Loss', linewidth=1.5, alpha=0.8)
    if 'pde_loss' in arrays:
        ax.semilogy(epochs, arrays['pde_loss'], label='PDE Loss', linewidth=1.5, alpha=0.8)
    if 'prior_consistency_loss' in arrays:
        ax.semilogy(epochs, arrays['prior_consistency_loss'], label='RANS Prior Loss', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Training Loss Overview (Total, Data, PDE, RANS Prior)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_overview.png', dpi=150)
    print(f"✅ 已儲存: {output_dir / 'loss_overview.png'}")
    plt.close()
    
    # ====================================================================
    # 圖 2: PDE 子項 (Momentum X/Y or Merged, Continuity)
    # ====================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 🔥 支援合併動量模式 (merge_momentum=True)
    if 'momentum_loss' in arrays:
        ax.semilogy(epochs, arrays['momentum_loss'], label='Momentum (Merged)', linewidth=2, alpha=0.9, color='blue')
    else:
        # 標準模式：分別顯示 X/Y
        if 'momentum_x_loss' in arrays:
            ax.semilogy(epochs, arrays['momentum_x_loss'], label='Momentum X', linewidth=1.5, alpha=0.8)
        if 'momentum_y_loss' in arrays:
            ax.semilogy(epochs, arrays['momentum_y_loss'], label='Momentum Y', linewidth=1.5, alpha=0.8)
    
    if 'continuity_loss' in arrays:
        ax.semilogy(epochs, arrays['continuity_loss'], label='Continuity (∇·u)', linewidth=2, alpha=0.9, color='red')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Physics Constraints Loss (Momentum & Continuity)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_physics.png', dpi=150)
    print(f"✅ 已儲存: {output_dir / 'loss_physics.png'}")
    plt.close()
    
    # ====================================================================
    # 圖 3: 數據擬合損失 (u, v, p)
    # ====================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'u_loss' in arrays:
        ax.semilogy(epochs, arrays['u_loss'], label='u (velocity x)', linewidth=1.5, alpha=0.8)
    if 'v_loss' in arrays:
        ax.semilogy(epochs, arrays['v_loss'], label='v (velocity y)', linewidth=1.5, alpha=0.8)
    if 'pressure_loss' in arrays:
        ax.semilogy(epochs, arrays['pressure_loss'], label='p (pressure)', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Data Fitting Loss per Variable (u, v, p)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_data.png', dpi=150)
    print(f"✅ 已儲存: {output_dir / 'loss_data.png'}")
    plt.close()
    
    # ====================================================================
    # 圖 4: RANS Prior 子項 (u, v, p)
    # ====================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    has_prior = False
    if 'prior_loss_u' in arrays:
        ax.semilogy(epochs, arrays['prior_loss_u'], label='Prior u', linewidth=1.5, alpha=0.8)
        has_prior = True
    if 'prior_loss_v' in arrays:
        ax.semilogy(epochs, arrays['prior_loss_v'], label='Prior v', linewidth=1.5, alpha=0.8)
        has_prior = True
    if 'prior_loss_p' in arrays:
        # prior_loss_p 可能非常小，需要處理
        prior_p = arrays['prior_loss_p']
        prior_p_clean = np.where(prior_p > 1e-10, prior_p, np.nan)
        ax.semilogy(epochs, prior_p_clean, label='Prior p', linewidth=1.5, alpha=0.8)
        has_prior = True
    
    if has_prior:
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (log scale)', fontsize=12)
        ax.set_title('RANS Prior Consistency Loss per Variable', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_dir / 'loss_prior.png', dpi=150)
        print(f"✅ 已儲存: {output_dir / 'loss_prior.png'}")
    plt.close()
    
    # ====================================================================
    # 圖 5: Weighted Loss (用於分析權重平衡)
    # ====================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'weighted_data_loss' in arrays:
        ax.semilogy(epochs, arrays['weighted_data_loss'], label='Weighted Data Loss', linewidth=1.5, alpha=0.8)
    if 'weighted_pde_loss' in arrays:
        ax.semilogy(epochs, arrays['weighted_pde_loss'], label='Weighted PDE Loss', linewidth=1.5, alpha=0.8)
    if 'weighted_div_loss' in arrays:
        ax.semilogy(epochs, arrays['weighted_div_loss'], label='Weighted Continuity Loss', linewidth=2, alpha=0.9, color='red')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Weighted Loss Components (After Weight Multiplication)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_weighted.png', dpi=150)
    print(f"✅ 已儲存: {output_dir / 'loss_weighted.png'}")
    plt.close()
    
    # ====================================================================
    # 圖 6: Learning Rate
    # ====================================================================
    if 'lr' in arrays:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.semilogy(epochs, arrays['lr'], label='Learning Rate', linewidth=2, alpha=0.9, color='purple')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate (log scale)', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_dir / 'learning_rate.png', dpi=150)
        print(f"✅ 已儲存: {output_dir / 'learning_rate.png'}")
        plt.close()
    
    # ====================================================================
    # 圖 7: 全景圖 (4×2 subplot)
    # ====================================================================
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    
    # 1. Total Loss
    ax = axes[0, 0]
    if 'total_loss' in arrays:
        ax.semilogy(epochs, arrays['total_loss'], linewidth=2, color='black')
    ax.set_title('Total Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log)')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # 2. Data Loss
    ax = axes[0, 1]
    if 'data_loss' in arrays:
        ax.semilogy(epochs, arrays['data_loss'], linewidth=2, color='blue')
    ax.set_title('Data Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log)')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # 3. PDE Loss
    ax = axes[1, 0]
    if 'pde_loss' in arrays:
        ax.semilogy(epochs, arrays['pde_loss'], linewidth=2, color='green')
    ax.set_title('PDE Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log)')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # 4. Continuity Loss
    ax = axes[1, 1]
    if 'continuity_loss' in arrays:
        ax.semilogy(epochs, arrays['continuity_loss'], linewidth=2, color='red')
    ax.set_title('Continuity Loss (∇·u)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log)')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # 5. Momentum X
    ax = axes[2, 0]
    if 'momentum_x_loss' in arrays:
        ax.semilogy(epochs, arrays['momentum_x_loss'], linewidth=2, color='orange')
    ax.set_title('Momentum X Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log)')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # 6. Momentum Y
    ax = axes[2, 1]
    if 'momentum_y_loss' in arrays:
        ax.semilogy(epochs, arrays['momentum_y_loss'], linewidth=2, color='brown')
    ax.set_title('Momentum Y Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log)')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # 7. RANS Prior
    ax = axes[3, 0]
    if 'prior_consistency_loss' in arrays:
        ax.semilogy(epochs, arrays['prior_consistency_loss'], linewidth=2, color='purple')
    ax.set_title('RANS Prior Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log)')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # 8. Learning Rate
    ax = axes[3, 1]
    if 'lr' in arrays:
        ax.semilogy(epochs, arrays['lr'], linewidth=2, color='magenta')
    ax.set_title('Learning Rate', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR (log)')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    plt.suptitle('Training Dynamics Overview (Epoch 0-6500)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_all_in_one.png', dpi=150)
    print(f"✅ 已儲存: {output_dir / 'loss_all_in_one.png'}")
    plt.close()
    
    # ====================================================================
    # 輸出數值統計
    # ====================================================================
    print("\n" + "="*70)
    print("訓練統計摘要 (Epoch 0 → 最後)")
    print("="*70)
    
    for key in ['total_loss', 'data_loss', 'pde_loss', 'continuity_loss', 
                'momentum_x_loss', 'momentum_y_loss', 'prior_consistency_loss']:
        if key in arrays:
            arr = arrays[key]
            arr_clean = arr[~np.isnan(arr)]
            if len(arr_clean) > 0:
                initial = arr_clean[0]
                final = arr_clean[-1]
                change = ((final - initial) / initial * 100) if initial != 0 else 0
                print(f"{key:30s}: {initial:10.6f} → {final:10.6f} ({change:+6.1f}%)")
    
    print("="*70)

def main():
    # 讀取日誌
    log_path = Path(__file__).parents[2] / 'pinnx.log'
    
    if not log_path.exists():
        print(f"❌ 找不到日誌文件: {log_path}")
        return
    
    print(f"📂 讀取日誌: {log_path}")
    data_list = load_training_log(log_path)
    
    if not data_list:
        print("❌ 未找到任何訓練記錄")
        return
    
    print(f"✅ 成功解析 {len(data_list)} 條訓練記錄")
    
    # 轉換為 arrays
    arrays = extract_arrays(data_list)
    print(f"✅ 提取 {len(arrays)} 個指標")
    
    # 繪圖
    output_dir = Path(__file__).parents[2] / 'results' / 'loss_curves'
    print(f"\n📊 開始繪圖，輸出目錄: {output_dir}")
    plot_loss_curves(arrays, output_dir)
    
    print("\n✅ 所有圖表生成完成！")

if __name__ == '__main__':
    main()

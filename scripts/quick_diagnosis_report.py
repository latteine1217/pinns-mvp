#!/usr/bin/env python3
"""
快速診斷報告：檢查完整訓練後的模型狀況
生成時間：2025-10-11
"""

import torch
import numpy as np
from pathlib import Path

def load_checkpoint_info(ckpt_path):
    """載入檢查點並提取關鍵資訊"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    info = {
        'epoch': ckpt.get('epoch', 'N/A'),
        'total_loss': ckpt.get('loss', 'N/A'),
        'conservation_error': ckpt.get('conservation_error', 'N/A'),
        'model_state_keys': len(ckpt.get('model_state_dict', {})),
    }
    
    # 提取配置資訊
    if 'config' in ckpt:
        config = ckpt['config']
        info['fourier_enabled'] = config.get('fourier_features', {}).get('enabled', False)
        info['fourier_m'] = config.get('fourier_features', {}).get('m', 0)
        info['sensor_k'] = config.get('sensors', {}).get('n_sensors', 'N/A')
    
    return info, ckpt

def print_diagnosis():
    """生成診斷報告"""
    print("=" * 70)
    print("🔍 VS-PINN 完整訓練診斷報告")
    print("=" * 70)
    
    # 檢查檢查點
    baseline_path = Path('checkpoints/vs_pinn_baseline_1k_latest.pth')
    fourier_path = Path('checkpoints/vs_pinn_fourier_1k_latest.pth')
    
    if not baseline_path.exists():
        print(f"❌ Baseline 檢查點不存在: {baseline_path}")
        return
    
    if not fourier_path.exists():
        print(f"❌ Fourier 檢查點不存在: {fourier_path}")
        return
    
    # 載入檢查點資訊
    print("\n📦 檢查點資訊：\n")
    
    baseline_info, baseline_ckpt = load_checkpoint_info(baseline_path)
    print("Baseline 模型:")
    print(f"  ✅ Epoch: {baseline_info['epoch']}")
    if isinstance(baseline_info['total_loss'], (int, float)):
        print(f"  ✅ Total Loss: {baseline_info['total_loss']:.6f}")
    else:
        print(f"  ✅ Total Loss: {baseline_info['total_loss']}")
    if isinstance(baseline_info['conservation_error'], (int, float)):
        print(f"  ✅ Conservation Error: {baseline_info['conservation_error']:.6f}")
    else:
        print(f"  ✅ Conservation Error: {baseline_info['conservation_error']}")
    print(f"  ✅ Fourier Features: {'❌ 禁用' if not baseline_info.get('fourier_enabled', False) else '✅ 啟用'}")
    print(f"  ✅ 感測點數 K: {baseline_info.get('sensor_k', 'N/A')}")
    
    print()
    
    fourier_info, fourier_ckpt = load_checkpoint_info(fourier_path)
    print("Fourier 模型:")
    print(f"  ✅ Epoch: {fourier_info['epoch']}")
    if isinstance(fourier_info['total_loss'], (int, float)):
        print(f"  ✅ Total Loss: {fourier_info['total_loss']:.6f}")
    else:
        print(f"  ✅ Total Loss: {fourier_info['total_loss']}")
    if isinstance(fourier_info['conservation_error'], (int, float)):
        print(f"  ✅ Conservation Error: {fourier_info['conservation_error']:.6f}")
    else:
        print(f"  ✅ Conservation Error: {fourier_info['conservation_error']}")
    print(f"  ✅ Fourier Features: {'✅ 啟用' if fourier_info.get('fourier_enabled', False) else '❌ 禁用'}")
    print(f"  ✅ Fourier M: {fourier_info.get('fourier_m', 'N/A')}")
    print(f"  ✅ 感測點數 K: {fourier_info.get('sensor_k', 'N/A')}")
    
    # 載入測試資料
    print("\n" + "=" * 70)
    print("📊 測試資料資訊：")
    print("=" * 70)
    
    data_path = Path('data/jhtdb/channel_flow_re1000/cutout3d_128x128x32.npz')
    if not data_path.exists():
        print(f"❌ 測試資料不存在: {data_path}")
        return
    
    data = np.load(data_path)
    
    # 檢查資料格式
    if 'coords' in data:
        coords = data['coords']
        u, v, w, p = data['u'], data['v'], data['w'], data['p']
    else:
        # 分離座標格式（網格資料）
        x_1d = data['x']  # (128,)
        y_1d = data['y']  # (128,)
        z_1d = data['z']  # (32,)
        
        # 創建網格並展平
        X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
        coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        u = data['u'].flatten()
        v = data['v'].flatten()
        w = data['w'].flatten()
        p = data['p'].flatten()
    
    print(f"  ✅ 測試點數: {coords.shape[0]:,}")
    print(f"  ✅ 真實 U 範圍: [{u.min():.2f}, {u.max():.2f}]")
    print(f"  ✅ 真實 V 範圍: [{v.min():.2f}, {v.max():.2f}]")
    print(f"  ✅ 真實 W 範圍: [{w.min():.2f}, {w.max():.2f}]")
    print(f"  ✅ 真實 P 範圍: [{p.min():.2f}, {p.max():.2f}]")
    
    # 訓練效率對比
    print("\n" + "=" * 70)
    print("📈 訓練效率對比：")
    print("=" * 70)
    
    print(f"\n{'指標':<25} {'Baseline':<20} {'Fourier':<20} {'比值':<10}")
    print("-" * 75)
    
    loss_ratio = None
    cons_ratio = None
    
    if isinstance(baseline_info['epoch'], (int, float)) and isinstance(fourier_info['epoch'], (int, float)):
        epochs_ratio = fourier_info['epoch'] / baseline_info['epoch']
        print(f"{'訓練 Epochs':<25} {baseline_info['epoch']:<20} {fourier_info['epoch']:<20} {epochs_ratio:.2f}x")
    
    if isinstance(baseline_info['total_loss'], (int, float)) and isinstance(fourier_info['total_loss'], (int, float)):
        loss_ratio = fourier_info['total_loss'] / baseline_info['total_loss']
        print(f"{'最終 Total Loss':<25} {baseline_info['total_loss']:<20.6f} {fourier_info['total_loss']:<20.6f} {loss_ratio:.2f}x")
    
    if isinstance(baseline_info['conservation_error'], (int, float)) and isinstance(fourier_info['conservation_error'], (int, float)):
        cons_ratio = fourier_info['conservation_error'] / baseline_info['conservation_error']
        print(f"{'Conservation Error':<25} {baseline_info['conservation_error']:<20.6f} {fourier_info['conservation_error']:<20.6f} {cons_ratio:.2f}x")
    
    # 關鍵觀察
    print("\n" + "=" * 70)
    print("🔑 關鍵觀察：")
    print("=" * 70)
    
    if baseline_info['epoch'] < 1000:
        print(f"\n⚠️  Baseline 早停於 epoch {baseline_info['epoch']} (目標: 1000)")
        print(f"   原因：Conservation error 連續 200 epochs 無改善")
    
    if loss_ratio and loss_ratio > 5:
        print(f"\n⚠️  Fourier 總損失是 Baseline 的 {loss_ratio:.1f}x")
        print(f"   可能原因：Fourier features 增加表達能力，但物理損失權重可能需調整")
    
    if cons_ratio and cons_ratio > 5:
        print(f"\n⚠️  Fourier 守恆誤差是 Baseline 的 {cons_ratio:.1f}x")
        print(f"   需要檢查：連續方程權重是否足夠 (當前配置未知)")
    
    print("\n" + "=" * 70)
    print("📝 診斷完成")
    print("=" * 70)
    print("\n💡 建議後續動作：")
    print("   1. 檢查可視化圖表: results/field_comparison/*.png")
    print("   2. 查看訓練日誌: log/{baseline,fourier}_1k_training.log")
    print("   3. 如流場結構合理，考慮調整損失權重繼續訓練")
    print("   4. 如流場結構錯誤，需要重新審視網路配置或邊界條件")

if __name__ == '__main__':
    print_diagnosis()

#!/usr/bin/env python3
"""
簡單的訓練進度監控腳本
監控 training_log_stable.txt 並提供進度摘要
"""
import re
import os
import time
from datetime import datetime, timedelta

def parse_epoch_line(line):
    """解析包含epoch資訊的日誌行"""
    pattern = r'Epoch\s+(\d+)\s+\|\s+Total:\s+([\d.]+)\s+\|\s+Residual:\s+([\d.]+)\s+\|\s+BC:\s+([\d.]+)\s+\|\s+Data:\s+([\d.]+)\s+\|\s+Time:\s+([\d.]+)s\s+\|\s+W_data:\s+([\d.]+)\s+\|\s+W_residual:\s+([\d.]+)\s+\|\s+W_boundary:\s+([\d.]+)'
    match = re.search(pattern, line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'total_loss': float(match.group(2)),
            'residual': float(match.group(3)),
            'bc': float(match.group(4)),
            'data': float(match.group(5)),
            'time': float(match.group(6)),
            'w_data': float(match.group(7)),
            'w_residual': float(match.group(8)),
            'w_boundary': float(match.group(9))
        }
    return None

def monitor_training(log_file="training_log_stable.txt", target_epochs=15000):
    """監控訓練進度"""
    print("=" * 60)
    print("🔍 JHTDB 穩定版訓練監控")
    print("=" * 60)
    print(f"監控文件: {log_file}")
    print(f"目標 epochs: {target_epochs}")
    print()
    
    if not os.path.exists(log_file):
        print(f"❌ 日誌文件不存在: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # 解析所有epoch資訊
    epochs_data = []
    for line in lines:
        if "Epoch" in line and "Total:" in line:
            data = parse_epoch_line(line)
            if data:
                epochs_data.append(data)
    
    if not epochs_data:
        print("❌ 未找到epoch資訊")
        return
    
    # 顯示進度摘要
    latest = epochs_data[-1]
    first = epochs_data[0]
    
    print(f"📊 **當前狀態** (最新: Epoch {latest['epoch']})")
    print(f"   Total Loss: {latest['total_loss']:.6f}")
    print(f"   Residual: {latest['residual']:.6f}")
    print(f"   BC: {latest['bc']:.6f}") 
    print(f"   Data: {latest['data']:.6f}")
    print()
    
    print(f"⚖️ **權重穩定性檢查**")
    print(f"   W_data: {latest['w_data']:.1f}")
    print(f"   W_residual: {latest['w_residual']:.1f}")
    print(f"   W_boundary: {latest['w_boundary']:.1f}")
    
    # 檢查權重是否穩定
    weights_stable = (latest['w_data'] == 10.0 and 
                     latest['w_residual'] == 1.0 and 
                     latest['w_boundary'] == 10.0)
    print(f"   狀態: {'✅ 穩定' if weights_stable else '❌ 不穩定'}")
    print()
    
    # 計算進度
    progress_pct = (latest['epoch'] / target_epochs) * 100
    print(f"📈 **訓練進度**")
    print(f"   進度: {latest['epoch']:,}/{target_epochs:,} ({progress_pct:.1f}%)")
    
    # 估算剩餘時間
    if len(epochs_data) >= 2:
        time_per_epoch = latest['time'] / max(1, latest['epoch'])
        remaining_epochs = target_epochs - latest['epoch']
        remaining_time = remaining_epochs * time_per_epoch
        remaining_hours = remaining_time / 3600
        print(f"   每epoch時間: {time_per_epoch:.1f}s")
        print(f"   預估剩餘時間: {remaining_hours:.1f} 小時")
    print()
    
    # 損失變化
    if len(epochs_data) >= 2:
        loss_reduction = ((first['total_loss'] - latest['total_loss']) / first['total_loss']) * 100
        print(f"📉 **損失收斂**")
        print(f"   初始損失: {first['total_loss']:.6f}")
        print(f"   當前損失: {latest['total_loss']:.6f}")
        print(f"   下降幅度: {loss_reduction:.1f}%")
        print()
    
    # 顯示最近幾個epoch
    print(f"📝 **最近進度** (最近 {min(3, len(epochs_data))} 個記錄)")
    for data in epochs_data[-3:]:
        print(f"   Epoch {data['epoch']:>6}: Loss={data['total_loss']:.6f}, "
              f"Res={data['residual']:.6f}, Data={data['data']:.6f}")

if __name__ == "__main__":
    monitor_training()
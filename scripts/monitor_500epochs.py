#!/usr/bin/env python3
"""
500 Epochs穩定性測試監控工具
監控RANS湍流系統的長期穩定性訓練進度
"""

import time
import re
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

def extract_epoch_info(log_file):
    """從日誌提取epoch信息"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        epochs = []
        for line in lines:
            if "Epoch" in line and "Total:" in line:
                # 解析 epoch 信息
                match = re.search(r'Epoch\s+(\d+).*Total:\s+([\d.]+).*Time:\s+([\d.]+)s', line)
                if match:
                    epoch = int(match.group(1))
                    total_loss = float(match.group(2))
                    time_s = float(match.group(3))
                    epochs.append({
                        'epoch': epoch,
                        'total_loss': total_loss,
                        'time': time_s
                    })
        
        return epochs
    except FileNotFoundError:
        return []

def check_training_status():
    """檢查訓練進程狀態"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'train_rans_optimized.py' in line and 'python' in line:
                return True
        return False
    except:
        return False

def plot_progress(epochs):
    """繪製進度圖表"""
    if len(epochs) < 2:
        return
    
    df = pd.DataFrame(epochs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 損失演化
    ax1.plot(df['epoch'], df['total_loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Loss Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 訓練速度
    ax2.plot(df['epoch'], df['time'], 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cumulative Time (s)')
    ax2.set_title('Training Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tasks/task-006/500epochs_progress.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    log_file = 'rans_stability_500epochs_v3.log'
    
    print(f"=== Task-006: 500 Epochs穩定性測試監控 ===")
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日誌文件: {log_file}")
    print("-" * 50)
    
    while True:
        # 檢查訓練狀態
        is_running = check_training_status()
        epochs = extract_epoch_info(log_file)
        
        if not epochs:
            print("⏳ 等待訓練日誌...")
            time.sleep(30)
            continue
        
        current = epochs[-1]
        epoch = current['epoch']
        loss = current['total_loss']
        time_elapsed = current['time']
        
        # 計算進度
        progress = (epoch / 500) * 100
        avg_time_per_epoch = time_elapsed / epoch if epoch > 0 else 0
        remaining_epochs = 500 - epoch
        eta_seconds = remaining_epochs * avg_time_per_epoch
        eta = datetime.now() + timedelta(seconds=eta_seconds)
        
        print(f"\r🔄 Epoch {epoch:3d}/500 ({progress:5.1f}%) | "
              f"Loss: {loss:.6f} | "
              f"ETA: {eta.strftime('%H:%M:%S')} | "
              f"{'✅ Running' if is_running else '❌ Stopped'}", end='')
        
        # 每50個epoch繪製一次圖表
        if epoch % 50 == 0:
            plot_progress(epochs)
            print(f"\n📊 Progress plot updated at epoch {epoch}")
        
        # 檢查是否完成
        if epoch >= 500 or not is_running:
            print(f"\n\n🎉 訓練完成! 最終epoch: {epoch}")
            plot_progress(epochs)
            
            # 生成最終報告
            print("\n=== 最終統計 ===")
            print(f"總epochs: {epoch}")
            print(f"最終損失: {loss:.6f}")
            print(f"總訓練時間: {time_elapsed:.1f}秒 ({time_elapsed/60:.1f}分鐘)")
            print(f"平均速度: {avg_time_per_epoch:.2f}秒/epoch")
            
            if len(epochs) > 1:
                initial_loss = epochs[0]['total_loss']
                improvement = (initial_loss - loss) / initial_loss * 100
                print(f"損失改善: {improvement:.1f}% (從{initial_loss:.6f}→{loss:.6f})")
            
            break
        
        time.sleep(30)  # 每30秒更新一次

if __name__ == "__main__":
    main()
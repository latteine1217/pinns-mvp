#!/usr/bin/env python3
"""
訓練速度監測腳本
監控訓練進度，計算每 epoch 時間、估計總訓練時間
"""

import time
import argparse
import re
from pathlib import Path
from datetime import timedelta


def parse_log_file(log_path: Path):
    """解析訓練日誌，提取速度指標"""

    if not log_path.exists():
        print(f"❌ 日誌文件不存在: {log_path}")
        return None

    epoch_times = []
    epoch_numbers = []
    losses = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # 正則表達式匹配 epoch 信息
    # 例如: Epoch 10/10000 | Loss: 1.234 | Time: 2.34s
    epoch_pattern = re.compile(r'Epoch\s+(\d+)/(\d+).*?Time[:\s]+([\d.]+)s')
    loss_pattern = re.compile(r'Loss[:\s]+([\d.e+-]+)')

    for line in lines:
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
            epoch_time = float(epoch_match.group(3))

            epoch_numbers.append(epoch_num)
            epoch_times.append(epoch_time)

        loss_match = loss_pattern.search(line)
        if loss_match:
            loss_val = float(loss_match.group(1))
            losses.append(loss_val)

    if not epoch_times:
        print("⚠️  未能從日誌中解析出 epoch 時間")
        return None

    return {
        'epoch_numbers': epoch_numbers,
        'epoch_times': epoch_times,
        'losses': losses,
        'total_epochs': total_epochs
    }


def analyze_training_speed(data: dict):
    """分析訓練速度"""

    epoch_times = data['epoch_times']
    epoch_numbers = data['epoch_numbers']
    total_epochs = data['total_epochs']

    # 計算統計量
    avg_time_per_epoch = sum(epoch_times) / len(epoch_times)
    min_time = min(epoch_times)
    max_time = max(epoch_times)

    # 計算最近 10 個 epoch 的平均時間（更準確）
    recent_avg = sum(epoch_times[-10:]) / min(10, len(epoch_times))

    # 估計剩餘時間
    current_epoch = epoch_numbers[-1] if epoch_numbers else 0
    remaining_epochs = total_epochs - current_epoch
    estimated_remaining_time = remaining_epochs * recent_avg

    # 估計總訓練時間
    estimated_total_time = total_epochs * recent_avg

    print("\n" + "="*60)
    print("📊 訓練速度分析")
    print("="*60)
    print(f"已完成 Epochs: {current_epoch}/{total_epochs} ({current_epoch/total_epochs*100:.1f}%)")
    print(f"\n時間統計:")
    print(f"  • 平均每 epoch: {avg_time_per_epoch:.2f}s")
    print(f"  • 最近 10 epochs 平均: {recent_avg:.2f}s")
    print(f"  • 最快: {min_time:.2f}s")
    print(f"  • 最慢: {max_time:.2f}s")

    print(f"\n⏱️  時間估計:")
    print(f"  • 剩餘時間: {timedelta(seconds=int(estimated_remaining_time))}")
    print(f"  • 預估總訓練時間: {timedelta(seconds=int(estimated_total_time))}")

    # 速度評估
    print(f"\n🚦 速度評估:")
    if recent_avg < 1.0:
        status = "✅ 極快 (< 1s/epoch)"
    elif recent_avg < 2.0:
        status = "✅ 良好 (1-2s/epoch)"
    elif recent_avg < 5.0:
        status = "⚠️  中等 (2-5s/epoch)"
    elif recent_avg < 10.0:
        status = "⚠️  偏慢 (5-10s/epoch)"
    else:
        status = "❌ 過慢 (>10s/epoch)"

    print(f"  {status}")

    # 給出建議
    print(f"\n💡 建議:")
    if recent_avg > 5.0:
        print("  ⚠️  訓練速度較慢，建議:")
        print("     1. 檢查是否在使用 GPU (--device cuda)")
        print("     2. 減少 batch_size 或 N_pde")
        print("     3. 減少網絡深度 (depth)")
        print("     4. 關閉不必要的特徵 (如 Fourier)")
    elif recent_avg < 2.0:
        print("  ✅ 訓練速度良好，可以考慮:")
        print("     1. 增加網絡容量以提升精度")
        print("     2. 啟用更多先進特徵")

    if data.get('losses'):
        print(f"\n📉 Loss 趨勢:")
        losses = data['losses']
        if len(losses) >= 2:
            initial_loss = losses[0]
            recent_loss = losses[-1]
            reduction = (initial_loss - recent_loss) / initial_loss * 100
            print(f"  • 初始 Loss: {initial_loss:.6f}")
            print(f"  • 當前 Loss: {recent_loss:.6f}")
            print(f"  • 降低: {reduction:.1f}%")

    print("="*60 + "\n")

    return {
        'avg_time': avg_time_per_epoch,
        'recent_avg': recent_avg,
        'estimated_total_hours': estimated_total_time / 3600,
        'current_epoch': current_epoch,
        'total_epochs': total_epochs
    }


def monitor_live(log_path: Path, interval: int = 10):
    """實時監控訓練日誌"""

    print(f"🔍 實時監控: {log_path}")
    print(f"📊 更新間隔: {interval}秒")
    print("按 Ctrl+C 停止監控\n")

    try:
        while True:
            data = parse_log_file(log_path)
            if data:
                analyze_training_speed(data)
            else:
                print(f"⏳ 等待訓練開始... ({time.strftime('%H:%M:%S')})")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n✋ 監控已停止")


def main():
    parser = argparse.ArgumentParser(description="訓練速度監測工具")
    parser.add_argument('--log', type=str, required=True,
                        help='訓練日誌文件路徑')
    parser.add_argument('--live', action='store_true',
                        help='實時監控模式')
    parser.add_argument('--interval', type=int, default=10,
                        help='實時監控更新間隔（秒）')

    args = parser.parse_args()

    log_path = Path(args.log)

    if args.live:
        monitor_live(log_path, args.interval)
    else:
        data = parse_log_file(log_path)
        if data:
            analyze_training_speed(data)


if __name__ == '__main__':
    main()

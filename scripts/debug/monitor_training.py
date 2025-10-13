#!/usr/bin/env python3
"""
簡易訓練監控腳本 - 即時顯示訓練進度
"""

import sys
import time
import re
from pathlib import Path

def monitor_log(log_file: str, interval: int = 5):
    """監控訓練日誌文件"""
    
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"❌ 日誌文件不存在: {log_file}")
        return
    
    print(f"📊 監控訓練日誌: {log_file}")
    print("=" * 80)
    
    # 記錄已讀取的行數
    last_position = 0
    current_epoch = 0
    best_conservation = float('inf')
    current_phase = "phase1_foundation"
    
    while True:
        try:
            with open(log_path, 'r') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()
            
            for line in new_lines:
                # 檢測階段切換
                phase_match = re.search(r'🔄.*切換到階段:\s*(\w+)', line)
                if phase_match:
                    current_phase = phase_match.group(1)
                    print(f"\n{'='*80}")
                    print(f"🔄 階段切換: {current_phase}")
                    print(f"{'='*80}\n")
                
                # 檢測新的最佳 conservation_error
                best_match = re.search(r'🎯 New best conservation_error:\s*([\d.]+)\s+at epoch\s+(\d+)', line)
                if best_match:
                    conservation = float(best_match.group(1))
                    epoch = int(best_match.group(2))
                    
                    if conservation < best_conservation:
                        best_conservation = conservation
                        current_epoch = epoch
                        
                        # 顏色編碼（基於目標值）
                        if conservation < 0.95:
                            status = "🎉 已達標"
                        elif conservation < 1.0:
                            status = "🔥 接近目標"
                        elif conservation < 1.08:
                            status = "📈 改善中"
                        else:
                            status = "⏳ 訓練中"
                        
                        print(f"Epoch {epoch:4d} | Phase: {current_phase:20s} | "
                              f"Best Conservation Error: {conservation:.6f} | {status}")
                
                # 檢測訓練完成
                if "Training completed" in line or "Early stopping triggered" in line:
                    print(f"\n{'='*80}")
                    print(f"✅ 訓練結束！")
                    print(f"   最終 Epoch: {current_epoch}")
                    print(f"   最佳 Conservation Error: {best_conservation:.6f}")
                    print(f"{'='*80}\n")
                    return
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n\n⏸️  監控已停止")
            print(f"   當前 Epoch: {current_epoch}")
            print(f"   當前最佳 Conservation Error: {best_conservation:.6f}")
            break
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            time.sleep(interval)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 自動尋找最新日誌
        log_dir = Path("logs")
        if log_dir.exists():
            log_files = sorted(log_dir.glob("phase4b_staged_*.log"), key=lambda p: p.stat().st_mtime)
            if log_files:
                log_file = str(log_files[-1])
                print(f"📁 自動選擇最新日誌: {log_file}\n")
            else:
                print("❌ 未找到日誌文件")
                sys.exit(1)
        else:
            print("❌ logs/ 目錄不存在")
            sys.exit(1)
    else:
        log_file = sys.argv[1]
    
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    monitor_log(log_file, interval)

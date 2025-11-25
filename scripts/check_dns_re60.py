#!/usr/bin/env python3
"""Re=60 DNS 模擬快速狀態檢查"""

import re
import sys
from datetime import datetime, timedelta

LOG_FILE = 'log/dns_re60_20251121_125441.log'

try:
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
    
    step_lines = [l for l in lines if 'Step' in l and 'KE=' in l]
    
    if not step_lines:
        print("⚠️  模擬尚未開始或日誌為空")
        sys.exit(1)
    
    latest = step_lines[-1]
    match = re.search(
        r'Step\s+(\d+)/(\d+).*t=\s*([\d.]+).*KE=([\de.+-]+).*Balance=([\d.]+)',
        latest
    )
    
    if match:
        step, total, t, ke, balance = match.groups()
        step, total = int(step), int(total)
        
        print(f"進度: {step:,} / {total:,} ({100*step/total:.1f}%)")
        print(f"時間: t={float(t):.2f}/200.0")
        print(f"動能: KE={float(ke):.4f}")
        print(f"平衡: {float(balance):.4f}", end="")
        
        bal = float(balance)
        if 0.95 <= bal <= 1.05:
            print(" ✅ 穩態")
        elif bal < 1.5:
            print(" ⏳ 接近")
        else:
            print(" 🔄 收斂中")
        
        # 預估完成時間
        start = datetime.strptime("2025-11-21 12:54:41", "%Y-%m-%d %H:%M:%S")
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed > 0:
            rate = step / elapsed
            remaining = (total - step) / rate if rate > 0 else 0
            eta = datetime.now() + timedelta(seconds=remaining)
            print(f"預計: {eta.strftime('%H:%M')} ({remaining/60:.0f}分鐘)")
        
except FileNotFoundError:
    print(f"❌ 找不到日誌: {LOG_FILE}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 錯誤: {e}")
    sys.exit(1)

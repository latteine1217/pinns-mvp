#!/usr/bin/env python3
"""
快速驗證標準化修復是否生效
檢查：
1. DNS cutout 文件存在且統計合理
2. validate_sensor_data_quality 函數可正常調用
3. 模擬標準化計算流程
"""

import numpy as np
from pathlib import Path
import sys

print("="*80)
print("🔍 Normalization Fix Verification")
print("="*80)

# 1. 檢查 DNS cutout 文件
dns_cutout_path = "data/jhtdb/channel_flow_re1000/cutout_128x64x128.npz"
print(f"\n1️⃣  Checking DNS cutout: {dns_cutout_path}")

if not Path(dns_cutout_path).exists():
    print(f"❌ DNS cutout NOT found!")
    sys.exit(1)

print(f"✅ DNS cutout exists")

# 2. 載入並檢查統計
print(f"\n2️⃣  Loading DNS data...")
dns_data = np.load(dns_cutout_path)

print(f"   Available fields: {list(dns_data.keys())}")

normalization_data = {}
for var in ['u', 'v', 'w', 'p']:
    if var in dns_data:
        data = dns_data[var].flatten()
        normalization_data[var] = data
        mean = data.mean()
        std = data.std()
        print(f"   {var}: mean={mean:.6f}, std={std:.6f}, shape={dns_data[var].shape}")
    else:
        print(f"   {var}: NOT FOUND")

# 3. 驗證統計是否合理
print(f"\n3️⃣  Validating statistics...")

checks_passed = True

# v 和 w 應該有合理的標準差（~0.04）
for var in ['v', 'w']:
    if var in normalization_data:
        std = normalization_data[var].std()
        if std < 1e-3:
            print(f"   ❌ {var}: std={std:.2e} TOO SMALL (< 1e-3) - NUMERICAL NOISE!")
            checks_passed = False
        else:
            print(f"   ✅ {var}: std={std:.6f} OK (>= 1e-3)")

# p 應該有合理的標準差（~0.003）
if 'p' in normalization_data:
    std = normalization_data['p'].std()
    if std < 1e-4:
        print(f"   ❌ p: std={std:.2e} TOO SMALL (< 1e-4) - GAUGE RESIDUALS!")
        checks_passed = False
    else:
        print(f"   ✅ p: std={std:.6f} OK (>= 1e-4)")

# 4. 測試 validate_sensor_data_quality 函數
print(f"\n4️⃣  Testing validate_sensor_data_quality function...")

sys.path.insert(0, str(Path(__file__).parent))

try:
    # 直接導入我們剛加入的函數（需要 mock logger）
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # 從 train.py 導入（這會觸發模組載入）
    from scripts.train.train import validate_sensor_data_quality
    
    validate_sensor_data_quality(normalization_data, logger)
    print("   ✅ validate_sensor_data_quality PASSED")
    
except Exception as e:
    print(f"   ❌ validate_sensor_data_quality FAILED: {e}")
    checks_passed = False

# 5. 總結
print(f"\n" + "="*80)
if checks_passed:
    print("✅ ALL CHECKS PASSED - Fix is ready!")
    print("\n📋 Expected normalization statistics:")
    print(f"   u: std ≈ 0.083  (actual: {normalization_data['u'].std():.6f})")
    print(f"   v: std ≈ 0.041  (actual: {normalization_data['v'].std():.6f})")
    print(f"   w: std ≈ 0.044  (actual: {normalization_data['w'].std():.6f})")
    print(f"   p: std ≈ 0.003  (actual: {normalization_data['p'].std():.6f})")
    print("\n🚀 Ready to retrain with:")
    print("   python scripts/train/train.py --config configs/channel_flow_re1000.yml")
else:
    print("❌ SOME CHECKS FAILED - Please review above errors")
    sys.exit(1)

print("="*80)

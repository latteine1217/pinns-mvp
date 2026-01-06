"""
Google Colab 環境設置腳本

在執行任何訓練腳本前，先在 Colab cell 中運行此腳本
"""

import os
import sys

# 設置專案路徑
PROJECT_ROOT = "/content/drive/MyDrive/pinns-sparse-flow"

# 切換到專案目錄
if os.path.exists(PROJECT_ROOT):
    os.chdir(PROJECT_ROOT)
    print(f"✓ 已切換到專案目錄: {PROJECT_ROOT}")
else:
    print(f"❌ 專案目錄不存在: {PROJECT_ROOT}")
    print("請確認：")
    print("  1. 已掛載 Google Drive")
    print("  2. 專案位於正確路徑")
    sys.exit(1)

# 添加專案路徑到 PYTHONPATH
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"✓ 已添加到 sys.path: {PROJECT_ROOT}")

# 驗證 pinnx 模組是否可導入
try:
    import pinnx
    print("✓ pinnx 模組可正常導入")
    print(f"  模組位置: {pinnx.__file__}")
except ImportError as e:
    print("❌ 無法導入 pinnx 模組")
    print(f"  錯誤: {e}")
    print("\n嘗試安裝...")
    import subprocess
    result = subprocess.run(
        ["pip", "install", "-e", "."],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✓ pinnx 安裝成功")
        import pinnx
        print(f"  模組位置: {pinnx.__file__}")
    else:
        print("❌ 安裝失敗")
        print(result.stderr)
        sys.exit(1)

# 設置環境變數
os.environ['PYTHONPATH'] = PROJECT_ROOT
print(f"✓ PYTHONPATH 已設置: {os.environ.get('PYTHONPATH')}")

# 檢查重要目錄
important_dirs = [
    "configs/experiments/S2_k_scan",
    "data/sensors/kolmogorov",
    "data/kolmogorov_dns",
    "scripts/train"
]

print("\n📁 檢查重要目錄：")
for dir_path in important_dirs:
    full_path = os.path.join(PROJECT_ROOT, dir_path)
    if os.path.exists(full_path):
        print(f"  ✓ {dir_path}")
    else:
        print(f"  ✗ {dir_path} (不存在)")

# 檢查 GPU
print("\n🖥️  GPU 狀態：")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA 可用")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠ CUDA 不可用，請檢查 Runtime 設置")
except ImportError:
    print("  ⚠ PyTorch 未安裝")

print("\n✅ 環境設置完成！")
print("\n下一步：")
print("  執行訓練: !python scripts/train/train.py --cfg configs/experiments/S2_k_scan/s2_qr_K50_2d_re50.yml")
print("  或使用腳本: !bash scripts/experiments/run_s2_k_scan_colab.sh 30 50")

#!/usr/bin/env python
"""
數據完整性驗證腳本

用途：驗證 data 目錄中的數據文件是否完整且可讀取
使用方式：python scripts/tools/verify_data_integrity.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import json

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_file_exists(filepath, required=True):
    """檢查文件是否存在"""
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  ✓ {filepath.name} ({size_mb:.2f} MB)")
        return True
    else:
        if required:
            print(f"  ✗ {filepath.name} (缺失，必要)")
            return False
        else:
            print(f"  - {filepath.name} (缺失，可選)")
            return True


def verify_npz_file(filepath):
    """驗證 .npz 文件可讀取"""
    try:
        data = np.load(filepath, allow_pickle=True)
        keys = list(data.keys())
        print(f"    包含 {len(keys)} 個變數: {keys[:5]}{'...' if len(keys) > 5 else ''}")
        data.close()
        return True
    except Exception as e:
        print(f"    ✗ 讀取失敗: {e}")
        return False


def verify_json_file(filepath):
    """驗證 JSON 文件可讀取"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"    包含 {len(data)} 個欄位")
        return True
    except Exception as e:
        print(f"    ✗ 讀取失敗: {e}")
        return False


def main():
    print("=" * 70)
    print("數據完整性驗證")
    print("=" * 70)
    print()
    
    data_dir = project_root / "data"
    
    if not data_dir.exists():
        print(f"✗ 錯誤：data 目錄不存在 ({data_dir})")
        return False
    
    print(f"數據目錄: {data_dir}")
    print()
    
    all_ok = True
    
    # ========================================================================
    # 1. 檢查 Kolmogorov DNS 數據
    # ========================================================================
    print("【1】Kolmogorov DNS 數據")
    print("-" * 70)
    
    kolmogorov_dns_dir = data_dir / "kolmogorov_dns"
    if kolmogorov_dns_dir.exists():
        # 必要文件
        snapshot_file = kolmogorov_dns_dir / "snapshot_re50_for_eval.npz"
        if check_file_exists(snapshot_file, required=True):
            all_ok &= verify_npz_file(snapshot_file)
        else:
            all_ok = False
        
        # 感測器文件
        sensor_file = kolmogorov_dns_dir / "qr_sensors_K100_v7_standard.npz"
        if check_file_exists(sensor_file, required=True):
            all_ok &= verify_npz_file(sensor_file)
        else:
            all_ok = False
    else:
        print(f"  ✗ kolmogorov_dns 目錄不存在（必要）")
        all_ok = False
    
    print()
    
    # ========================================================================
    # 2. 檢查 JHTDB 數據
    # ========================================================================
    print("【2】JHTDB Channel Flow 數據")
    print("-" * 70)
    
    jhtdb_dir = data_dir / "jhtdb"
    if jhtdb_dir.exists():
        # HDF5 文件
        h5_files = list(jhtdb_dir.glob("*.h5"))
        if h5_files:
            for h5_file in h5_files:
                check_file_exists(h5_file, required=False)
        else:
            print("  - 無 HDF5 文件（可選）")
        
        # 感測器文件
        sensor_files = [
            "sensors_kf8_qr_K100.npz",
            "sensors_kf8_physical_K100.npz",
        ]
        for sensor_name in sensor_files:
            sensor_file = jhtdb_dir / sensor_name
            if check_file_exists(sensor_file, required=False):
                verify_npz_file(sensor_file)
        
        # Metadata
        metadata_file = jhtdb_dir / "cache_metadata.json"
        if check_file_exists(metadata_file, required=False):
            verify_json_file(metadata_file)
    else:
        print(f"  - jhtdb 目錄不存在（可選）")
    
    print()
    
    # ========================================================================
    # 3. 檢查 Sensors 數據
    # ========================================================================
    print("【3】Sensors 配置")
    print("-" * 70)
    
    sensors_dir = data_dir / "sensors"
    if sensors_dir.exists():
        sensor_configs = list(sensors_dir.glob("*.json"))
        if sensor_configs:
            for config_file in sensor_configs:
                if check_file_exists(config_file, required=False):
                    verify_json_file(config_file)
        else:
            print("  - 無感測器配置文件（可選）")
    else:
        print("  - sensors 目錄不存在（可選）")
    
    print()
    
    # ========================================================================
    # 4. 檢查 Low-Fidelity Prior 數據
    # ========================================================================
    print("【4】Low-Fidelity Prior 數據")
    print("-" * 70)
    
    lowfi_npy_dir = data_dir / "lowfi_npy"
    if lowfi_npy_dir.exists():
        required_vars = ['u', 'v', 'p']
        for var_name in required_vars:
            var_file = lowfi_npy_dir / f"{var_name}.npy"
            if check_file_exists(var_file, required=False):
                try:
                    arr = np.load(var_file)
                    print(f"    形狀: {arr.shape}, 範圍: [{arr.min():.3f}, {arr.max():.3f}]")
                except Exception as e:
                    print(f"    ✗ 讀取失敗: {e}")
                    all_ok = False
    else:
        print("  - lowfi_npy 目錄不存在（可選）")
    
    print()
    
    # ========================================================================
    # 總結
    # ========================================================================
    print("=" * 70)
    if all_ok:
        print("✅ 數據完整性驗證通過")
        print()
        print("後續步驟：")
        print("  1. 執行訓練測試：python scripts/train/train.py --cfg configs/quick_test.yml --epochs 2")
        print("  2. 執行 DDP 訓練：torchrun --nproc_per_node=2 scripts/train/train.py --cfg configs/main.yml")
    else:
        print("✗ 數據完整性驗證失敗")
        print()
        print("缺失必要文件，請檢查：")
        print("  - data/kolmogorov_dns/snapshot_re50_for_eval.npz")
        print("  - data/kolmogorov_dns/qr_sensors_K100_v7_standard.npz")
    print("=" * 70)
    
    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

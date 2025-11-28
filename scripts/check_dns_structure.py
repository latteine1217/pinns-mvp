#!/usr/bin/env python
"""檢查 Kolmogorov DNS HDF5 文件結構"""

import h5py
import numpy as np
import sys
from pathlib import Path

def inspect_hdf5(file_path):
    """檢查 HDF5 文件結構"""
    print(f"\n{'='*70}")
    print(f"  檢查 DNS 文件: {file_path}")
    print(f"{'='*70}\n")
    
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        return
    
    try:
        with h5py.File(file_path, 'r') as f:
            print("📂 頂層數據集/群組:")
            print("-"*70)
            
            for key in f.keys():
                obj = f[key]
                if isinstance(obj, h5py.Dataset):
                    print(f"  📄 {key}: shape={obj.shape}, dtype={obj.dtype}")
                    # 顯示數值範圍
                    if obj.ndim > 0 and obj.size > 0:
                        data = obj[...]
                        print(f"      範圍: [{np.min(data):.4e}, {np.max(data):.4e}]")
                elif isinstance(obj, h5py.Group):
                    print(f"  📁 {key}/ (群組)")
            
            print("\n📋 屬性 (Attributes):")
            print("-"*70)
            if len(f.attrs) > 0:
                for attr_name in f.attrs:
                    attr_value = f.attrs[attr_name]
                    print(f"  {attr_name}: {attr_value}")
            else:
                print("  (無屬性)")
            
            print("\n✅ 文件結構分析:")
            print("-"*70)
            
            # 檢查必要數據集
            required = ['u', 'v', 'p']
            missing = [k for k in required if k not in f]
            if missing:
                print(f"  ❌ 缺少數據集: {missing}")
            else:
                print(f"  ✅ 流場數據完整: {required}")
            
            # 檢查時間數據
            if 'time' in f:
                time_key = 'time'
                print(f"  ✅ 時間數據: '{time_key}'")
            elif 't' in f:
                time_key = 't'
                print(f"  ✅ 時間數據: '{time_key}'")
            else:
                print(f"  ❌ 缺少時間數據")
                time_key = None
            
            # 檢查座標
            if 'x' in f and 'y' in f:
                print(f"  ✅ 空間座標: 'x', 'y'")
            else:
                print(f"  ℹ️  空間座標需自動生成（DNS 文件不包含）")
            
            # 顯示數據形狀
            if all(k in f for k in ['u', 'v', 'p']):
                print(f"\n📐 數據維度:")
                print("-"*70)
                u_shape = f['u'].shape
                print(f"  流場形狀: {u_shape} (Nt × Ny × Nx)")
                
                if time_key and time_key in f:
                    t_shape = f[time_key].shape
                    print(f"  時間形狀: {t_shape}")
                    print(f"  總快照數: {u_shape[0]} (應與時間點數一致: {t_shape[0]})")
                
                print(f"\n💾 預計記憶體需求:")
                element_size = 4  # float32
                total_bytes = np.prod(u_shape) * element_size * 3  # u, v, p
                print(f"  單精度 (float32): {total_bytes/1e9:.2f} GB")
                
        print(f"\n{'='*70}\n")
                
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # 默認路徑
        file_path = 'data/kolmogorov_dns_re100_512x512_kf4.h5'
        print(f"使用默認路徑: {file_path}")
        print(f"或指定文件: python {sys.argv[0]} <path_to_file.h5>\n")
    
    inspect_hdf5(file_path)

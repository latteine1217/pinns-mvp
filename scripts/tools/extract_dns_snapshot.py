#!/usr/bin/env python3
"""
從 HDF5 格式的 DNS 資料中提取單一快照並儲存為 NPZ 格式

用途：為圖表生成工具提供背景場資料
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def extract_snapshot(h5_path: str, output_path: str, time_index: int = 0):
    """
    從 HDF5 檔案提取單一時間快照
    
    Args:
        h5_path: HDF5 輸入檔案路徑
        output_path: NPZ 輸出檔案路徑
        time_index: 時間索引（預設為 0，即第一個快照）
    """
    print(f"讀取 DNS 資料: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # 顯示可用資料集
        print(f"可用資料集: {list(f.keys())}")
        
        # 從 config 重建座標網格
        N = f['config'].attrs['N']
        L = f['config'].attrs['L']
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        
        # 提取速度與壓力場（取指定時間快照）
        u = f['u'][time_index, :, :]
        v = f['v'][time_index, :, :]
        p = f['p'][time_index, :, :] if 'p' in f else None
        
        # 計算渦度 (ω_z = ∂v/∂x - ∂u/∂y)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        dv_dx = np.gradient(v, dx, axis=1)
        du_dy = np.gradient(u, dy, axis=0)
        vorticity = dv_dx - du_dy
        
        # 計算速度梯度張量的 Frobenius norm
        du_dx = np.gradient(u, dx, axis=1)
        dv_dy = np.gradient(v, dy, axis=0)
        grad_u_norm = np.sqrt(du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2)
        
        print(f"資料形狀:")
        print(f"  x: {x.shape}, 範圍 [{x.min():.3f}, {x.max():.3f}]")
        print(f"  y: {y.shape}, 範圍 [{y.min():.3f}, {y.max():.3f}]")
        print(f"  u: {u.shape}, 範圍 [{u.min():.3f}, {u.max():.3f}]")
        print(f"  v: {v.shape}, 範圍 [{v.min():.3f}, {v.max():.3f}]")
        print(f"  vorticity: {vorticity.shape}, 範圍 [{vorticity.min():.3f}, {vorticity.max():.3f}]")
        
        # 儲存為 NPZ
        save_dict = {
            'x': x,
            'y': y,
            'u': u,
            'v': v,
            'vorticity': vorticity,
            'grad_u_norm': grad_u_norm,
            'time_index': time_index,
        }
        
        if p is not None:
            save_dict['p'] = p
            print(f"  p: {p.shape}, 範圍 [{p.min():.3f}, {p.max():.3f}]")
        
        np.savez_compressed(output_path, **save_dict)
        print(f"\n✓ 快照已儲存至: {output_path}")
        print(f"  檔案大小: {Path(output_path).stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="從 HDF5 DNS 資料提取單一快照為 NPZ 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    # 提取 Re=50 Kolmogorov flow 的第一個快照
    python extract_dns_snapshot.py \\
        --input data/kolmogorov_dns/dns_re50_t100.h5 \\
        --output data/kolmogorov_dns/snapshot_re50_t0.npz \\
        --time-index 0
    
    # 提取中間時刻快照（用於統計穩定的湍流場）
    python extract_dns_snapshot.py \\
        --input data/kolmogorov_dns/dns_re50_t100.h5 \\
        --output data/kolmogorov_dns/snapshot_re50_mid.npz \\
        --time-index 50
        """
    )
    parser.add_argument('--input', '-i', required=True, help='輸入 HDF5 檔案路徑')
    parser.add_argument('--output', '-o', required=True, help='輸出 NPZ 檔案路徑')
    parser.add_argument('--time-index', '-t', type=int, default=0, 
                        help='時間快照索引（預設: 0）')
    
    args = parser.parse_args()
    
    # 確保輸出目錄存在
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    extract_snapshot(args.input, args.output, args.time_index)


if __name__ == '__main__':
    main()

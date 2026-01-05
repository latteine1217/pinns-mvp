#!/usr/bin/env python
"""
生成測試用的 Kolmogorov Flow 資料（小規模）

用於快速驗證時間窗口訓練系統，不追求物理精度。
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_mock_kolmogorov_data(
    output_path: str,
    N: int = 64,
    T: int = 100,
    dt: float = 0.5,
    L: float = 2 * np.pi,
    Re: float = 10000,
):
    """
    生成模擬的 Kolmogorov Flow 資料（用於測試）
    
    注意：這是簡化的模擬資料，僅用於測試系統功能，
    不代表真實的 Kolmogorov Flow 物理行為。
    
    Args:
        output_path: 輸出 HDF5 檔案路徑
        N: 空間解析度（N×N 網格）
        T: 時間步數
        dt: 時間步長
        L: 域長度（正方形域）
        Re: 雷諾數
    """
    logger.info(f"{'='*70}")
    logger.info(f"🔧 生成測試用 Kolmogorov Flow 資料")
    logger.info(f"{'='*70}")
    logger.info(f"   解析度: {N}×{N}")
    logger.info(f"   時間步數: {T}")
    logger.info(f"   時間步長: {dt}s")
    logger.info(f"   總時間: {T*dt}s")
    logger.info(f"   域長度: {L:.4f}")
    logger.info(f"   雷諾數: {Re}")
    logger.info(f"   輸出路徑: {output_path}")
    logger.info(f"{'='*70}\n")
    
    # 創建輸出目錄
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 時間軸
    time = np.arange(0, T * dt, dt)[:T]  # 確保長度為 T
    
    # 空間網格
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # 初始化流場陣列
    u_field = np.zeros((T, N, N), dtype=np.float32)
    v_field = np.zeros((T, N, N), dtype=np.float32)
    p_field = np.zeros((T, N, N), dtype=np.float32)
    
    logger.info("🌊 生成模擬流場...")
    
    # 簡化的 Kolmogorov Flow 近似（正弦波 + 時間演化）
    # 僅用於測試，不代表真實物理
    k_forcing = 4  # 強迫波數
    
    for t_idx, t in enumerate(time):
        # 基礎流場（時間演化的正弦波）
        phase = 0.1 * t  # 時間相位
        
        # u 分量：主要沿 x 方向
        u_field[t_idx] = np.sin(k_forcing * Y) * np.cos(phase)
        u_field[t_idx] += 0.2 * np.cos(2 * X) * np.sin(2 * Y + phase)
        
        # v 分量：主要沿 y 方向（較小振幅）
        v_field[t_idx] = 0.3 * np.cos(k_forcing * X) * np.sin(phase)
        v_field[t_idx] += 0.1 * np.sin(2 * X + phase) * np.cos(2 * Y)
        
        # 壓力場（簡化）
        p_field[t_idx] = 0.5 * (u_field[t_idx]**2 + v_field[t_idx]**2)
        
        if (t_idx + 1) % 20 == 0:
            logger.info(f"   進度: {t_idx+1}/{T} ({(t_idx+1)/T*100:.1f}%)")
    
    logger.info("✅ 流場生成完成\n")
    
    # 保存到 HDF5
    logger.info(f"💾 保存到 HDF5: {output_path}")
    with h5py.File(output_path, 'w') as f:
        # 時間軸
        f.create_dataset('time', data=time, dtype=np.float32)
        
        # 流場
        f.create_dataset('u', data=u_field, dtype=np.float32, compression='gzip')
        f.create_dataset('v', data=v_field, dtype=np.float32, compression='gzip')
        f.create_dataset('p', data=p_field, dtype=np.float32, compression='gzip')
        
        # 配置資訊
        config_grp = f.create_group('config')
        config_grp.attrs['N'] = N
        config_grp.attrs['L'] = L
        config_grp.attrs['Re'] = Re
        config_grp.attrs['dt'] = dt
        config_grp.attrs['T'] = T
        config_grp.attrs['description'] = 'Mock Kolmogorov Flow data for testing'
    
    logger.info("✅ 檔案保存完成\n")
    
    # 統計資訊
    logger.info("📊 資料統計:")
    logger.info(f"   u 範圍: [{u_field.min():.3f}, {u_field.max():.3f}]")
    logger.info(f"   v 範圍: [{v_field.min():.3f}, {v_field.max():.3f}]")
    logger.info(f"   p 範圍: [{p_field.min():.3f}, {p_field.max():.3f}]")
    logger.info(f"   檔案大小: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    logger.info(f"{'='*70}\n")


def generate_random_sensors(
    output_path: str,
    N: int = 64,
    K: int = 100,
    seed: int = 42
):
    """
    生成隨機感測點位置（用於測試）
    
    注意：真實應用應使用 QR-Pivot 算法選擇最優感測點。
    這裡僅為測試目的生成隨機位置。
    
    Args:
        output_path: 輸出 JSON 檔案路徑
        N: 空間解析度（N×N 網格）
        K: 感測點數量
        seed: 隨機種子
    """
    logger.info(f"{'='*70}")
    logger.info(f"📍 生成隨機感測點位置")
    logger.info(f"{'='*70}")
    logger.info(f"   網格大小: {N}×{N} = {N*N} 點")
    logger.info(f"   感測點數: {K}")
    logger.info(f"   隨機種子: {seed}")
    logger.info(f"   輸出路徑: {output_path}")
    logger.info(f"{'='*70}\n")
    
    # 創建輸出目錄
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 隨機選擇感測點（不重複）
    np.random.seed(seed)
    total_points = N * N
    
    if K > total_points:
        raise ValueError(f"感測點數 K={K} 超過總點數 {total_points}")
    
    indices = np.random.choice(total_points, size=K, replace=False).tolist()
    indices.sort()  # 排序便於檢查
    
    # 保存到 JSON
    sensor_data = {
        'indices': indices,
        'K': K,
        'N': N,
        'total_points': total_points,
        'method': 'random',
        'seed': seed,
        'coverage': K / total_points * 100,
        'description': 'Random sensor placement for testing (not optimal)'
    }
    
    with open(output_path, 'w') as f:
        json.dump(sensor_data, f, indent=2)
    
    logger.info("✅ 感測點位置保存完成")
    logger.info(f"   覆蓋率: {sensor_data['coverage']:.2f}%")
    logger.info(f"   索引範圍: [{min(indices)}, {max(indices)}]")
    logger.info(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='生成測試用 Kolmogorov Flow 資料'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data',
        help='輸出目錄（預設: ./data）'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=64,
        help='空間解析度 N×N（預設: 64）'
    )
    parser.add_argument(
        '--time_steps',
        type=int,
        default=100,
        help='時間步數（預設: 100）'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.5,
        help='時間步長（預設: 0.5s）'
    )
    parser.add_argument(
        '--num_sensors',
        type=int,
        default=100,
        help='感測點數量（預設: 100）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='隨機種子（預設: 42）'
    )
    
    args = parser.parse_args()
    
    # 輸出路徑
    output_dir = Path(args.output_dir)
    data_path = output_dir / 'kolmogorov' / 'dns_Re10000_test.h5'
    sensor_path = output_dir / 'sensors' / 'random_K100_test.json'
    
    # 生成資料
    print("\n" + "="*70)
    print("🚀 開始生成測試資料")
    print("="*70 + "\n")
    
    # 1. 生成 DNS 資料
    generate_mock_kolmogorov_data(
        output_path=str(data_path),
        N=args.resolution,
        T=args.time_steps,
        dt=args.dt,
    )
    
    # 2. 生成感測點位置
    generate_random_sensors(
        output_path=str(sensor_path),
        N=args.resolution,
        K=args.num_sensors,
        seed=args.seed
    )
    
    # 總結
    print("="*70)
    print("🎉 測試資料生成完成！")
    print("="*70)
    print("\n📁 生成的檔案:")
    print(f"   1. DNS 資料: {data_path}")
    print(f"   2. 感測點:   {sensor_path}")
    print("\n📝 下一步:")
    print("   更新配置文件以使用這些測試資料:")
    print(f"   - data.kolmogorov_config.data_path: {data_path}")
    print(f"   - sensors.sensor_file: {sensor_path}")
    print("\n   然後運行:")
    print("   python scripts/train_time_window.py \\")
    print("       --config configs/experiments/time_window_kolmogorov.yml \\")
    print("       --dry_run")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

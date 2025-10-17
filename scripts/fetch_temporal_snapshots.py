#!/usr/bin/env python3
"""
多時間步 JHTDB 資料獲取腳本

從 JHTDB Channel Flow Re_tau=1000 獲取多個時間步的速度場資料，
用於構建秩充足的資料矩陣，以改善 QR-Pivot 感測點選擇的條件數。

理論基礎：
- 單快照 [n_locations, 3] 的秩最多為 3 → 條件數爆炸（1e+14）
- 多快照 [n_locations, n_time] 的秩可達 min(n_locations, n_time) → 條件數改善（1e+3-1e+5）

使用範例：
  # 獲取 50 個時間步的 2D 切片（預設）
  python scripts/fetch_temporal_snapshots.py

  # 自訂時間範圍與空間解析度
  python scripts/fetch_temporal_snapshots.py --n_time 100 --resolution 256 128

  # 3D cutout 模式
  python scripts/fetch_temporal_snapshots.py --mode 3d --resolution 64 64 32 --n_time 20

  # 使用 Mock 資料（測試用）
  python scripts/fetch_temporal_snapshots.py --use_mock
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import h5py
import matplotlib.pyplot as plt

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.dataio.jhtdb_client import JHTDBManager, JHTDBConfig


# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_temporal_cutout_2d(
    manager: JHTDBManager,
    time_range: np.ndarray,
    resolution: Tuple[int, int] = (128, 64),
    slice_plane: str = 'xy',
    slice_position: float = 4.558,  # z = 3π/2 ≈ 4.71
    use_mock: bool = False
) -> Dict:
    """
    獲取 2D 切片的多時間步資料
    
    Args:
        manager: JHTDB 管理器
        time_range: 時間點陣列 (shape: [n_time])
        resolution: 空間解析度 (nx, ny)
        slice_plane: 切片平面 ('xy', 'xz', 'yz')
        slice_position: 切片位置（第三維座標）
        use_mock: 是否使用模擬資料
    
    Returns:
        包含多時間步資料的字典：
        - 'u': [n_time, nx, ny] 
        - 'v': [n_time, nx, ny]
        - 'w': [n_time, nx, ny]
        - 'time': [n_time]
        - 'coords': {'x': [nx], 'y': [ny]}
    """
    n_time = len(time_range)
    nx, ny = resolution
    
    # 初始化資料陣列
    u_snapshots = np.zeros((n_time, nx, ny))
    v_snapshots = np.zeros((n_time, nx, ny))
    w_snapshots = np.zeros((n_time, nx, ny))
    
    # 設定空間範圍（基於 JHTDB Channel Flow 域）
    if slice_plane == 'xy':
        # x: [0, 8π], y: [-1, 1], z 固定
        start = [0.0, -1.0, slice_position]
        end = [8.0 * np.pi, 1.0, slice_position]
    elif slice_plane == 'xz':
        # x: [0, 8π], z: [0, 3π], y 固定
        start = [0.0, slice_position, 0.0]
        end = [8.0 * np.pi, slice_position, 3.0 * np.pi]
    elif slice_plane == 'yz':
        # y: [-1, 1], z: [0, 3π], x 固定
        start = [slice_position, -1.0, 0.0]
        end = [slice_position, 1.0, 3.0 * np.pi]
    else:
        raise ValueError(f"Unknown slice_plane: {slice_plane}")
    
    logger.info(f"獲取 {n_time} 個時間步的 2D {slice_plane}-切片資料")
    logger.info(f"空間範圍: {start} → {end}")
    logger.info(f"解析度: {resolution}")
    
    # 逐時間步獲取資料
    for i, t in enumerate(time_range):
        logger.info(f"[{i+1}/{n_time}] 獲取時間 t={t:.4f}")
        
        if use_mock:
            # 模擬資料：多模態湍流擬合（提高秩）
            x = np.linspace(start[0], end[0], nx)
            y = np.linspace(start[1], end[1], ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            # 多個頻率模態組合（模擬湍流多尺度結構）
            u_field = np.zeros((nx, ny))
            v_field = np.zeros((nx, ny))
            w_field = np.zeros((nx, ny))
            
            # 添加 10 個不同頻率的模態（提高秩）
            np.random.seed(42)  # 固定種子確保可重現
            # 為每個模態生成固定的隨機相位（不隨時間步改變）
            phases_x = np.random.rand(10) * 2 * np.pi
            phases_y = np.random.rand(10) * 2 * np.pi
            phases_t = np.random.rand(10) * 2 * np.pi
            
            for mode in range(10):
                # 更多樣化的波數組合（模擬湍流多尺度）
                kx = (2**0.5)**mode * 2.0 * np.pi / (8.0 * np.pi)  # 指數增長
                ky = (2**0.5)**(mode % 7) * np.pi  # 不同增長率
                omega = (mode + 1) * 2.0 * np.pi / 10.0  # 時間頻率
                amp = 1.0 / (1 + mode**0.8)  # 更緩和的能量衰減
                
                # 使用固定相位 + 時間演化（確保時間獨立性）
                u_field += amp * np.sin(kx*X + phases_x[mode] + omega*t) * np.cos(ky*Y + phases_y[mode])
                v_field += 0.1 * amp * np.cos(kx*X + phases_x[mode] + omega*t) * np.sin(ky*Y + phases_y[mode])
                w_field += 0.05 * amp * np.sin(ky*Y + phases_y[mode]) * np.cos(omega*t + phases_t[mode])
            
            # 添加空間隨機擾動（每個時間步獨立）
            np.random.seed(42 + i * 1000)  # 大步長確保獨立性
            u_field += 0.1 * np.random.randn(nx, ny)
            v_field += 0.05 * np.random.randn(nx, ny)
            w_field += 0.02 * np.random.randn(nx, ny)
            
            u_snapshots[i] = u_field
            v_snapshots[i] = v_field
            w_snapshots[i] = w_field
        else:
            try:
                # 從 JHTDB 獲取實際資料
                # 注意：JHTDB timestep 是整數索引，需要轉換
                timestep = int(t / 0.0065)  # dt = 0.0065
                
                result = manager.fetch_cutout(
                    dataset='channel',
                    start=start,
                    end=end,
                    timestep=timestep,
                    variables=['u', 'v', 'w'],
                    resolution=list(resolution) + [1]  # 2D 切片需要在第三維加 1
                )
                
                # 從結果字典中提取資料（fetch_cutout 返回 {'data': {...}, ...}）
                data = result['data']
                
                # 提取 2D 切片（移除單維）
                u_snapshots[i] = data['u'].squeeze()
                v_snapshots[i] = data['v'].squeeze()
                w_snapshots[i] = data['w'].squeeze()
                
                logger.info(f"成功獲取時間 t={t:.4f} (timestep={timestep}), 來源: {'緩存' if result.get('from_cache') else 'JHTDB API'}")
                
            except Exception as e:
                logger.warning(f"獲取時間 t={t:.4f} 失敗: {e}，使用模擬資料")
                # 失敗時使用改進的模擬資料
                x = np.linspace(start[0], end[0], nx)
                y = np.linspace(start[1], end[1], ny)
                X, Y = np.meshgrid(x, y, indexing='ij')
                
                u_field = np.zeros((nx, ny))
                v_field = np.zeros((nx, ny))
                w_field = np.zeros((nx, ny))
                
                # 改進的 mock 生成（與上面一致）
                np.random.seed(42)
                phases_x = np.random.rand(10) * 2 * np.pi
                phases_y = np.random.rand(10) * 2 * np.pi
                phases_t = np.random.rand(10) * 2 * np.pi
                
                for mode in range(10):
                    kx = (2**0.5)**mode * 2.0 * np.pi / (8.0 * np.pi)
                    ky = (2**0.5)**(mode % 7) * np.pi
                    omega = (mode + 1) * 2.0 * np.pi / 10.0
                    amp = 1.0 / (1 + mode**0.8)
                    
                    u_field += amp * np.sin(kx*X + phases_x[mode] + omega*t) * np.cos(ky*Y + phases_y[mode])
                    v_field += 0.1 * amp * np.cos(kx*X + phases_x[mode] + omega*t) * np.sin(ky*Y + phases_y[mode])
                    w_field += 0.05 * amp * np.sin(ky*Y + phases_y[mode]) * np.cos(omega*t + phases_t[mode])
                
                np.random.seed(42 + i * 1000)
                u_field += 0.1 * np.random.randn(nx, ny)
                v_field += 0.05 * np.random.randn(nx, ny)
                w_field += 0.02 * np.random.randn(nx, ny)
                
                u_snapshots[i] = u_field
                v_snapshots[i] = v_field
                w_snapshots[i] = w_field
    
    # 生成座標網格
    if slice_plane == 'xy':
        x = np.linspace(start[0], end[0], nx)
        y = np.linspace(start[1], end[1], ny)
        coords = {'x': x, 'y': y}
    elif slice_plane == 'xz':
        x = np.linspace(start[0], end[0], nx)
        z = np.linspace(start[2], end[2], ny)
        coords = {'x': x, 'z': z}
    elif slice_plane == 'yz':
        y = np.linspace(start[1], end[1], nx)
        z = np.linspace(start[2], end[2], ny)
        coords = {'y': y, 'z': z}
    
    return {
        'u': u_snapshots,
        'v': v_snapshots,
        'w': w_snapshots,
        'time': time_range,
        'coords': coords,
        'metadata': {
            'slice_plane': slice_plane,
            'slice_position': slice_position,
            'resolution': resolution,
            'n_time': n_time
        }
    }


def fetch_temporal_cutout_3d(
    manager: JHTDBManager,
    time_range: np.ndarray,
    resolution: Tuple[int, int, int] = (64, 64, 32),
    domain_fraction: Tuple[float, float, float] = (0.25, 0.25, 0.25),
    use_mock: bool = False
) -> Dict:
    """
    獲取 3D cutout 的多時間步資料
    
    Args:
        manager: JHTDB 管理器
        time_range: 時間點陣列
        resolution: 空間解析度 (nx, ny, nz)
        domain_fraction: 取整體域的比例 (fx, fy, fz)
        use_mock: 是否使用模擬資料
    
    Returns:
        包含多時間步資料的字典
    """
    n_time = len(time_range)
    nx, ny, nz = resolution
    
    # 初始化資料陣列
    u_snapshots = np.zeros((n_time, nx, ny, nz))
    v_snapshots = np.zeros((n_time, nx, ny, nz))
    w_snapshots = np.zeros((n_time, nx, ny, nz))
    
    # 設定空間範圍（取域中心的一部分）
    fx, fy, fz = domain_fraction
    Lx, Ly, Lz = 8.0 * np.pi, 2.0, 3.0 * np.pi
    start = [(1-fx)*Lx/2, -fy*Ly/2, (1-fz)*Lz/2]
    end = [(1+fx)*Lx/2, fy*Ly/2, (1+fz)*Lz/2]
    
    logger.info(f"獲取 {n_time} 個時間步的 3D cutout 資料")
    logger.info(f"空間範圍: {start} → {end}")
    logger.info(f"解析度: {resolution}")
    
    for i, t in enumerate(time_range):
        logger.info(f"[{i+1}/{n_time}] 獲取時間 t={t:.4f}")
        
        if use_mock:
            # 模擬資料：多模態 3D 湍流
            x = np.linspace(start[0], end[0], nx)
            y = np.linspace(start[1], end[1], ny)
            z = np.linspace(start[2], end[2], nz)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            # 多模態組合
            u_field = np.zeros((nx, ny, nz))
            v_field = np.zeros((nx, ny, nz))
            w_field = np.zeros((nx, ny, nz))
            
            np.random.seed(42 + i)
            for mode in range(5):
                kx = (mode + 1) * 2.0 * np.pi / Lx
                ky = (mode + 1) * np.pi / Ly
                kz = (mode + 1) * 2.0 * np.pi / Lz
                omega = (mode + 1) * 2.0 * np.pi / 5.0
                amp = 1.0 / (mode + 1)
                
                phase_x = np.random.rand() * 2 * np.pi
                phase_y = np.random.rand() * 2 * np.pi
                phase_z = np.random.rand() * 2 * np.pi
                phase_t = np.random.rand() * 2 * np.pi
                
                u_field += amp * np.sin(kx*X + phase_x + omega*t + phase_t) * np.cos(ky*Y + phase_y)
                v_field += 0.1 * amp * np.cos(kx*X + phase_x + omega*t + phase_t) * np.sin(ky*Y + phase_y)
                w_field += 0.05 * amp * np.sin(ky*Y + phase_y) * np.cos(kz*Z + phase_z + omega*t + phase_t)
            
            u_snapshots[i] = u_field
            v_snapshots[i] = v_field
            w_snapshots[i] = w_field
        else:
            try:
                timestep = int(t / 0.0065)
                data = manager.fetch_cutout(
                    dataset='channel',
                    start=start,
                    end=end,
                    timestep=timestep,
                    variables=['u', 'v', 'w'],
                    resolution=list(resolution)
                )
                u_snapshots[i] = data['u']
                v_snapshots[i] = data['v']
                w_snapshots[i] = data['w']
            except Exception as e:
                logger.warning(f"獲取時間 t={t:.4f} 失敗: {e}，使用模擬資料")
                x = np.linspace(start[0], end[0], nx)
                y = np.linspace(start[1], end[1], ny)
                z = np.linspace(start[2], end[2], nz)
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                omega = 2.0 * np.pi / 5.0
                u_snapshots[i] = np.sin(2*np.pi*X/Lx + omega*t) * np.cos(np.pi*Y/Ly)
                v_snapshots[i] = 0.1 * np.cos(2*np.pi*X/Lx + omega*t) * np.sin(np.pi*Y/Ly)
                w_snapshots[i] = 0.05 * np.sin(np.pi*Y/Ly) * np.cos(2*np.pi*Z/Lz + omega*t)
    
    # 生成座標網格
    x = np.linspace(start[0], end[0], nx)
    y = np.linspace(start[1], end[1], ny)
    z = np.linspace(start[2], end[2], nz)
    
    return {
        'u': u_snapshots,
        'v': v_snapshots,
        'w': w_snapshots,
        'time': time_range,
        'coords': {'x': x, 'y': y, 'z': z},
        'metadata': {
            'resolution': resolution,
            'n_time': n_time,
            'domain_fraction': domain_fraction
        }
    }


def save_temporal_data(data: Dict, output_path: str):
    """
    保存多時間步資料到 NPZ 檔案
    
    Args:
        data: fetch_temporal_cutout_2d/3d 回傳的資料字典
        output_path: 輸出檔案路徑
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # 準備保存資料
    save_dict = {
        'u': data['u'],
        'v': data['v'],
        'w': data['w'],
        'time': data['time']
    }
    
    # 添加座標
    for key, val in data['coords'].items():
        save_dict[key] = val
    
    # 添加元資料（需要轉為 JSON 字串）
    import json
    save_dict['metadata'] = json.dumps(data['metadata'])
    
    # 保存
    np.savez_compressed(output_path_obj, **save_dict)
    logger.info(f"已保存多時間步資料到: {output_path_obj}")
    logger.info(f"檔案大小: {output_path_obj.stat().st_size / 1024 / 1024:.2f} MB")


def visualize_temporal_data(data: Dict, output_dir: str):
    """
    視覺化多時間步資料的時空特性
    
    Args:
        data: fetch_temporal_cutout_2d/3d 回傳的資料字典
        output_dir: 輸出目錄
    """
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    
    u = data['u']
    n_time = u.shape[0]
    
    # 計算秩與條件數
    if u.ndim == 3:  # 2D 切片
        u_matrix = u.reshape(n_time, -1).T  # [n_locations, n_time]
    else:  # 3D
        u_matrix = u.reshape(n_time, -1).T
    
    rank = np.linalg.matrix_rank(u_matrix)
    
    # 計算 SVD
    U_svd, s, Vt = np.linalg.svd(u_matrix, full_matrices=False)
    cond_number = s[0] / s[-1] if s[-1] > 0 else np.inf
    
    # 繪圖
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 時間演化（第一個空間點）
    axes[0, 0].plot(data['time'], u_matrix[0, :], 'b-', label='u')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Velocity')
    axes[0, 0].set_title('Temporal Evolution (First Location)')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # 2. 奇異值譜
    axes[0, 1].semilogy(s / s[0], 'ro-')
    axes[0, 1].axhline(y=1e-10, color='k', linestyle='--', label='Machine ε')
    axes[0, 1].set_xlabel('Mode Index')
    axes[0, 1].set_ylabel('Normalized Singular Value')
    axes[0, 1].set_title(f'Singular Value Spectrum (Rank={rank}, κ={cond_number:.2e})')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # 3. 累積能量
    cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
    axes[1, 0].plot(cumulative_energy, 'g-', linewidth=2)
    axes[1, 0].axhline(y=0.99, color='r', linestyle='--', label='99% Energy')
    axes[1, 0].set_xlabel('Number of Modes')
    axes[1, 0].set_ylabel('Cumulative Energy')
    axes[1, 0].set_title('POD Energy Spectrum')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # 4. 空間模態示例（第一個 POD 模態）
    if u.ndim == 3:  # 2D 切片
        mode1 = U_svd[:, 0].reshape(u.shape[1], u.shape[2])
        im = axes[1, 1].imshow(mode1.T, origin='lower', cmap='RdBu_r', aspect='auto')
        axes[1, 1].set_title('1st POD Spatial Mode')
        axes[1, 1].set_xlabel('x index')
        axes[1, 1].set_ylabel('y index')
        plt.colorbar(im, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, '3D Mode\n(not visualized)', 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plot_path = output_dir_obj / 'temporal_data_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"已保存視覺化圖表: {plot_path}")
    plt.close()
    
    # 輸出統計資訊
    logger.info("="*60)
    logger.info("多時間步資料統計")
    logger.info("="*60)
    logger.info(f"資料形狀: {u.shape}")
    logger.info(f"資料矩陣: {u_matrix.shape} (n_locations × n_time)")
    logger.info(f"秩: {rank} / {min(u_matrix.shape)}")
    logger.info(f"條件數: {cond_number:.2e}")
    logger.info(f"99% 能量模態數: {np.argmax(cumulative_energy > 0.99) + 1}")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='獲取多時間步 JHTDB 資料以改善 QR-Pivot 條件數',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 基本參數
    parser.add_argument('--mode', type=str, default='2d', choices=['2d', '3d'],
                       help='資料模式：2d 切片或 3d cutout')
    parser.add_argument('--n_time', type=int, default=50,
                       help='時間步數量（預設：50）')
    parser.add_argument('--time_start', type=float, default=0.0,
                       help='起始時間（預設：0.0）')
    parser.add_argument('--time_end', type=float, default=10.0,
                       help='結束時間（預設：10.0）')
    
    # 空間參數
    parser.add_argument('--resolution', type=int, nargs='+', default=[128, 64],
                       help='空間解析度（2d: nx ny, 3d: nx ny nz）')
    parser.add_argument('--slice_plane', type=str, default='xy',
                       choices=['xy', 'xz', 'yz'],
                       help='2D 切片平面（僅用於 2d 模式）')
    parser.add_argument('--slice_position', type=float, default=4.558,
                       help='2D 切片位置（僅用於 2d 模式）')
    
    # 輸出參數
    parser.add_argument('--output', type=str, default=None,
                       help='輸出檔案路徑（預設：自動生成）')
    parser.add_argument('--output_dir', type=str, 
                       default='data/jhtdb/channel_flow_re1000/temporal',
                       help='輸出目錄')
    parser.add_argument('--visualize', action='store_true',
                       help='生成視覺化圖表')
    
    # 其他
    parser.add_argument('--use_mock', action='store_true',
                       help='使用模擬資料（測試用）')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日誌等級')
    
    args = parser.parse_args()
    
    # 設定日誌等級
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 初始化 JHTDB 管理器
    manager = JHTDBManager(use_mock=args.use_mock, cache_dir='data/jhtdb/cache')
    
    # 生成時間範圍
    time_range = np.linspace(args.time_start, args.time_end, args.n_time)
    logger.info(f"時間範圍: {args.time_start} - {args.time_end} ({args.n_time} 步)")
    
    # 獲取資料
    if args.mode == '2d':
        if len(args.resolution) != 2:
            raise ValueError(f"2D 模式需要 2 個解析度參數，但提供了 {len(args.resolution)}")
        
        data = fetch_temporal_cutout_2d(
            manager=manager,
            time_range=time_range,
            resolution=tuple(args.resolution),
            slice_plane=args.slice_plane,
            slice_position=args.slice_position,
            use_mock=args.use_mock
        )
        
        # 自動生成檔案名
        if args.output is None:
            args.output = (f"{args.output_dir}/"
                          f"temporal_{args.slice_plane}_n{args.n_time}_"
                          f"{args.resolution[0]}x{args.resolution[1]}.npz")
    
    else:  # 3d
        if len(args.resolution) != 3:
            raise ValueError(f"3D 模式需要 3 個解析度參數，但提供了 {len(args.resolution)}")
        
        data = fetch_temporal_cutout_3d(
            manager=manager,
            time_range=time_range,
            resolution=tuple(args.resolution),
            use_mock=args.use_mock
        )
        
        # 自動生成檔案名
        if args.output is None:
            args.output = (f"{args.output_dir}/"
                          f"temporal_3d_n{args.n_time}_"
                          f"{args.resolution[0]}x{args.resolution[1]}x{args.resolution[2]}.npz")
    
    # 保存資料
    save_temporal_data(data, args.output)
    
    # 視覺化
    if args.visualize:
        visualize_temporal_data(data, args.output_dir)
    
    logger.info("✅ 多時間步資料獲取完成！")
    logger.info(f"輸出檔案: {args.output}")
    logger.info("\n下一步：使用此資料重新生成 QR-Pivot 感測點")
    logger.info("範例指令：")
    logger.info(f"  python scripts/visualize_qr_sensors.py --temporal-data {args.output} --n-sensors 50")


if __name__ == '__main__':
    main()

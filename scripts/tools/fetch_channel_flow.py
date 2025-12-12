#!/usr/bin/env python3
"""
Channel Flow Re1000 JHTDB 資料擷取腳本

從約翰霍普金斯湍流資料庫 (JHTDB) 擷取 Channel Flow Re_tau=1000 資料，
並進行預處理以支援 PINNs 逆重建實驗。

主要功能：
1. JHTDB Channel Flow cutout 批量資料下載
2. 2D 切片提取與時間平均處理  
3. QR-pivot 感測點選擇與噪聲模擬
4. 低保真 RANS 資料生成作為軟先驗
5. 資料格式標準化與快取管理
6. 3D cutout 快取與切片 CLI（支援 --use_mock）

使用範例：
  python fetch_channel_flow.py --config configs/channel_flow_re1000.yml
  python fetch_channel_flow.py --cutout_mode --resolution 128 64
  python fetch_channel_flow.py --sensor_only --K 8 --method qr_pivot
  python fetch_channel_flow.py --cutout3d_mode --resolution3d 128 128 32 --use_mock --log_level info
  python fetch_channel_flow.py --extract_slice --plane xy --position 4.71 --slice_res 128 64 --log_level info
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import yaml
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
import hashlib

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.dataio.jhtdb_client import JHTDBManager, JHTDBConfig
from pinnx.sensors import FieldSensorSelector
from pinnx.dataio.lowfi_loader import LowFiData


@dataclass
class ChannelFlowConfig:
    """Channel Flow Re1000 專用配置"""
    # JHTDB 資料集參數
    dataset: str = "channel"
    description: str = "通道流 (Re_tau=1000)"
    
    # 域範圍 (基於摩擦速度 u_tau 標準化)
    domain_x: Optional[List[float]] = None  # [0, 25.13] = [0, 8π]
    domain_y: Optional[List[float]] = None  # [-1.0, 1.0]
    domain_z: Optional[List[float]] = None  # [0, 9.42] = [0, 3π]
    
    # 解析度
    resolution_x: int = 2048
    resolution_y: int = 512
    resolution_z: int = 1536
    
    # 物理參數
    Re_tau: float = 1000.0
    nu: float = 1.0e-3  # ν = 1/Re_tau
    u_tau: float = 1.0  # 摩擦速度
    
    # 時間參數
    time_range: Optional[List[float]] = None  # [0.0, 26.0]
    dt: float = 0.0065
    time_average_window: Optional[List[float]] = None  # [20.0, 26.0]
    
    # 2D 切片設定
    slice_plane: str = "xy"  # xy, xz, yz
    slice_position: float = 4.71  # z = 3π/2
    
    def __post_init__(self):
        """設定預設值"""
        if self.domain_x is None:
            self.domain_x = [0.0, 8 * np.pi]
        if self.domain_y is None:
            self.domain_y = [-1.0, 1.0]
        if self.domain_z is None:
            self.domain_z = [0.0, 3 * np.pi]
        if self.time_range is None:
            self.time_range = [0.0, 26.0]
        if self.time_average_window is None:
            self.time_average_window = [20.0, 26.0]


class ChannelFlowDataFetcher:
    """Channel Flow Re1000 資料擷取器"""
    
    def __init__(self, config: ChannelFlowConfig, cache_dir: str = "./data/jhtdb/channel_flow_re1000", use_mock: bool = False):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "slices").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "raw").mkdir(parents=True, exist_ok=True)
        
        # 初始化 JHTDB 管理器
        self.jhtdb_manager = JHTDBManager(
            cache_dir=str(self.cache_dir / "raw"),
            use_mock=use_mock
        )
        
        # 日誌設定
        self.logger = logging.getLogger(__name__)
    
    # ---------------- 3D cutout 與檢查 ----------------
    def fetch_cutout3d_data(self,
                            resolution3d: Tuple[int, int, int] = (128, 128, 32),
                            variables: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        擷取 3D cutout 並保存為 npz（含 metadata 與 checksum），同時輸出物理檢查報告
        """
        if variables is None:
            variables = ['u', 'v', 'w', 'p']
        self.logger.info(f"擷取 3D cutout，解析度: {resolution3d}, 變數: {variables}")
        
        # 範圍覆蓋全域（或配置域），此處使用配置域
        # 安全取得域範圍（允許 None，回退到預設區間）
        domain_x = self.config.domain_x or [0.0, 8 * np.pi]
        domain_y = self.config.domain_y or [-1.0, 1.0]
        domain_z = self.config.domain_z or [0.0, 3 * np.pi]
        start = [float(domain_x[0]), float(domain_y[0]), float(domain_z[0])]
        end = [float(domain_x[1]), float(domain_y[1]), float(domain_z[1])]
        
        result = self.jhtdb_manager.fetch_cutout(
            dataset=self.config.dataset,
            start=start,
            end=end,
            resolution=list(resolution3d),  # 僅 mock 使用
            variables=variables,
            timestep=0
        )
        data3d = result.get('data', {})
        if not data3d:
            raise RuntimeError("JHTDB cutout 回傳空資料或格式不符（缺少 'data' 欄位）")
        
        # 合成等距座標
        nx, ny, nz = resolution3d
        x = np.linspace(start[0], end[0], nx, dtype=np.float32)
        y = np.linspace(start[1], end[1], ny, dtype=np.float32)
        z = np.linspace(start[2], end[2], nz, dtype=np.float32)
        
        # 轉為 float32 並組裝保存字典
        save_dict: Dict[str, Any] = {
            'x': x, 'y': y, 'z': z
        }
        for var in variables:
            if var in data3d:
                save_dict[var] = np.asarray(data3d[var], dtype=np.float32)
        
        # metadata 與 checksum
        metadata = {
            'dataset': self.config.dataset,
            'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'resolution3d': list(resolution3d),
            'domain': {'x': self.config.domain_x, 'y': self.config.domain_y, 'z': self.config.domain_z},
            'from_cache': bool(result.get('from_cache', False))
        }
        checksum = self._compute_checksum(save_dict)
        metadata['checksum'] = checksum
        
        # 保存 npz
        npz_path = self.cache_dir / f"cutout3d_{nx}x{ny}x{nz}.npz"
        np.savez_compressed(npz_path, **save_dict)
        self.logger.info(f"3D cutout 已保存: {npz_path}")
        
        # 保存 metadata.json
        meta_path = self.cache_dir / f"cutout3d_{nx}x{ny}x{nz}.metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 生成檢查報告
        report = self._generate_physics_report_3d(save_dict)
        report['metadata'] = metadata
        report_path = self.cache_dir / "reports" / f"cutout3d_{nx}x{ny}x{nz}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        self.logger.info(f"檢查報告已保存: {report_path}")
        
        return save_dict
    
    def _compute_checksum(self, arrays: Dict[str, np.ndarray]) -> str:
        """計算 sha256 校驗和（基於所有陣列按鍵排序串接）"""
        h = hashlib.sha256()
        for k in sorted(arrays.keys()):
            arr = arrays[k]
            if isinstance(arr, np.ndarray):
                h.update(k.encode('utf-8'))
                h.update(arr.tobytes(order='C'))
        return h.hexdigest()
    
    def _generate_physics_report_3d(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """生成基礎物理檢查報告（NaN/Inf、散度）"""
        report: Dict[str, Any] = {
            'nan_inf': {},
            'divergence': {}
        }
        keys = [k for k in ['u', 'v', 'w', 'p'] if k in data]
        for k in keys:
            arr = data[k]
            report['nan_inf'][k] = {
                'shape': list(arr.shape),
                'nan_count': int(np.isnan(arr).sum()),
                'inf_count': int(np.isinf(arr).sum()),
                'min': float(np.nanmin(arr)),
                'max': float(np.nanmax(arr)),
                'mean': float(np.nanmean(arr)),
                'std': float(np.nanstd(arr)),
            }
        # 散度檢查（中心差分）
        if all(k in data for k in ['u', 'v', 'w']):
            u, v, w = data['u'], data['v'], data['w']
            x, y, z = data['x'], data['y'], data['z']
            dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
            dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
            dz = float(z[1] - z[0]) if len(z) > 1 else 1.0
            div = self._compute_divergence(u, v, w, dx, dy, dz)
            abs_div = np.abs(div)
            report['divergence'] = {
                'mean_abs': float(abs_div.mean()),
                'p99_abs': float(np.quantile(abs_div, 0.99)),
                'max_abs': float(abs_div.max()),
                'thresholds': {
                    'mean_abs_max': float(1e-2),
                    'p99_abs_max': float(5e-2)
                },
                'pass': bool((abs_div.mean() <= 1e-2) and (np.quantile(abs_div, 0.99) <= 5e-2))
            }
        else:
            report['divergence'] = {'error': '缺少速度分量以計算散度'}
        return report
    
    @staticmethod
    def _compute_divergence(u: np.ndarray, v: np.ndarray, w: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """以中心差分計算散度，邊界採用一階差分"""
        dudx = np.zeros_like(u, dtype=np.float32)
        dvdy = np.zeros_like(v, dtype=np.float32)
        dwdz = np.zeros_like(w, dtype=np.float32)
        
        # x 方向
        dudx[1:-1, :, :] = (u[2:, :, :] - u[:-2, :, :]) / (2.0 * dx)
        dudx[0, :, :] = (u[1, :, :] - u[0, :, :]) / dx
        dudx[-1, :, :] = (u[-1, :, :] - u[-2, :, :]) / dx
        # y 方向
        dvdy[:, 1:-1, :] = (v[:, 2:, :] - v[:, :-2, :]) / (2.0 * dy)
        dvdy[:, 0, :] = (v[:, 1, :] - v[:, 0, :]) / dy
        dvdy[:, -1, :] = (v[:, -1, :] - v[:, -2, :]) / dy
        # z 方向
        dwdz[:, :, 1:-1] = (w[:, :, 2:] - w[:, :, :-2]) / (2.0 * dz)
        dwdz[:, :, 0] = (w[:, :, 1] - w[:, :, 0]) / dz
        dwdz[:, :, -1] = (w[:, :, -1] - w[:, :, -2]) / dz
        return dudx + dvdy + dwdz

    # ---------------- 既有 2D 切片流程 ----------------
    def fetch_cutout_data(self, 
                         resolution: Tuple[int, int] = (128, 64),
                         variables: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        擷取 JHTDB cutout 資料並提取 2D 切片
        
        Args:
            resolution: (nx, ny) 2D 切片解析度
            variables: 變數列表，預設 ['u', 'v', 'p']
            
        Returns:
            處理後的 2D 場資料
        """
        if variables is None:
            variables = ['u', 'v', 'p']
            
        self.logger.info(f"擷取 Channel Flow cutout 資料，解析度: {resolution}")
        
        # 計算 3D cutout 範圍 (確保包含目標切片位置)
        z_margin = 0.5  # z 方向邊界
        cutout_start = [
            self.config.domain_x[0] if self.config.domain_x else 0.0,
            self.config.domain_y[0] if self.config.domain_y else -1.0,
            max(self.config.domain_z[0] if self.config.domain_z else 0.0, self.config.slice_position - z_margin)
        ]
        cutout_end = [
            self.config.domain_x[1] if self.config.domain_x else 25.13,
            self.config.domain_y[1] if self.config.domain_y else 1.0,
            min(self.config.domain_z[1] if self.config.domain_z else 9.42, self.config.slice_position + z_margin)
        ]
        
        # 3D cutout 解析度 (z 方向使用最小解析度)
        cutout_resolution = [resolution[0], resolution[1], 16]
        
        # 從 JHTDB 獲取資料
        cutout_result = self.jhtdb_manager.fetch_cutout(
            dataset=self.config.dataset,
            start=cutout_start,
            end=cutout_end,
            resolution=cutout_resolution,
            variables=variables,
            timestep=0  # 使用瞬時資料，後續進行時間平均
        )
        
        # 取出實際 3D 資料陣列
        data_3d = cutout_result.get('data', {})
        if not data_3d:
            raise RuntimeError("JHTDB cutout 回傳空資料或格式不符（缺少 'data' 欄位）")
        
        # 提取 2D 切片（若無座標，合成 x,y,z）
        slice_data = self._extract_2d_slice(
            data_3d,
            resolution,
            cutout_start=cutout_start,
            cutout_end=cutout_end
        )
        
        # 快取結果
        cache_file = self.cache_dir / f"cutout_{resolution[0]}x{resolution[1]}.npz"
        np.savez_compressed(cache_file, **slice_data)
        self.logger.info(f"資料已快取至: {cache_file}")
        
        return slice_data
    
    def _extract_2d_slice(self,
                         cutout_data: Dict[str, np.ndarray],
                         target_resolution: Tuple[int, int],
                         cutout_start: Optional[List[float]] = None,
                         cutout_end: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """從 3D cutout 資料提取 2D 切片
        
        - 若 cutout_data 中未提供座標，則根據 cutout_start/cutout_end 與資料形狀合成 x, y, z 座標。
        - 於 z 方向選擇最接近 config.slice_position 的切片，並視情況插值到 target_resolution。
        """
        
        # 取得一個示例變數以判斷 3D 形狀
        sample_var = None
        for key in ['u', 'v', 'w', 'p']:
            if key in cutout_data:
                sample_var = key
                break
        if sample_var is None:
            raise ValueError("cutout_data 中沒有可用變數（u/v/w/p）")
        nx3d, ny3d, nz3d = cutout_data[sample_var].shape
        
        # 構造/取得座標
        if 'coordinates' in cutout_data and isinstance(cutout_data['coordinates'], dict):
            coords = cutout_data['coordinates']
            x = np.asarray(coords['x'])
            y = np.asarray(coords['y'])
            z = np.asarray(coords['z'])
        else:
            # 若未提供，合成等距座標
            if cutout_start is None or cutout_end is None:
                # 回退使用全域 domain 區間
                x0, x1 = (self.config.domain_x or [0.0, 8*np.pi])
                y0, y1 = (self.config.domain_y or [-1.0, 1.0])
                z0, z1 = (self.config.domain_z or [0.0, 3*np.pi])
            else:
                x0, y0, z0 = cutout_start
                x1, y1, z1 = cutout_end
            x = np.linspace(x0, x1, nx3d)
            y = np.linspace(y0, y1, ny3d)
            z = np.linspace(z0, z1, nz3d)
        
        # 找到最接近目標切片位置的 z 索引
        z_idx = int(np.argmin(np.abs(z - self.config.slice_position)))
        self.logger.info(f"選擇 z 切片位置: {z[z_idx]:.3f} (目標: {self.config.slice_position:.3f})")
        
        # 提取 2D 場
        slice_data: Dict[str, Any] = {}
        # 內存結構保留 coordinates dict 以便下游使用；同時平鋪 x,y 便於 npz 讀回
        slice_data['coordinates'] = {'x': x, 'y': y}
        slice_data['x'] = x
        slice_data['y'] = y
        
        for var in ['u', 'v', 'w', 'p']:
            if var in cutout_data:
                field_3d = cutout_data[var]
                if field_3d.ndim != 3:
                    raise ValueError(f"期望 3D 場，但 {var} 維度為 {field_3d.ndim}")
                field_2d = field_3d[:, :, z_idx]
                
                # 調整到目標解析度 (如果需要)
                if field_2d.shape != tuple(target_resolution):
                    field_2d = self._interpolate_to_resolution(field_2d, x, y, target_resolution)
                slice_data[var] = field_2d
        
        # 附上實際切片位置與誤差
        slice_data['slice_plane'] = 'xy'
        slice_data['slice_position'] = float(z[z_idx])
        slice_data['slice_position_target'] = float(self.config.slice_position)
        slice_data['slice_position_error'] = float(abs(z[z_idx] - self.config.slice_position))
        
        return slice_data
    
    def _interpolate_to_resolution(self, field: np.ndarray, 
                                  x: np.ndarray, y: np.ndarray,
                                  target_resolution: Tuple[int, int]) -> np.ndarray:
        """將場插值到目標解析度"""
        from scipy.interpolate import RegularGridInterpolator
        
        # 建立插值器
        interpolator = RegularGridInterpolator(
            (x, y), field, 
            method='linear', 
            bounds_error=False, 
            fill_value=0.0
        )
        
        # 新網格
        x_new = np.linspace(x[0], x[-1], target_resolution[0])
        y_new = np.linspace(y[0], y[-1], target_resolution[1])
        X_new, Y_new = np.meshgrid(x_new, y_new, indexing='ij')
        
        # 插值
        points_new = np.stack([X_new.ravel(), Y_new.ravel()], axis=-1)
        field_new = interpolator(points_new).reshape(target_resolution)
        
        return field_new

    def extract_and_save_slice(self,
                               resolution3d: Tuple[int, int, int],
                               plane: str = 'xy',
                               position: float = 0.0,
                               slice_res: Tuple[int, int] = (128, 64)) -> Path:
        """從 3D 快取提取特定平面的 2D 切片並保存"""
        nx, ny, nz = resolution3d
        npz_path = self.cache_dir / f"cutout3d_{nx}x{ny}x{nz}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"未找到 3D 快取: {npz_path}")
        data = dict(np.load(npz_path, allow_pickle=True))
        x = data['x']; y = data['y']; z = data['z']
        
        # 選擇平面與索引
        if plane == 'xy':
            axis_coords = z
            idx = int(np.argmin(np.abs(axis_coords - position)))
            fields_2d = {}
            for var in ['u', 'v', 'w', 'p']:
                if var in data:
                    fields_2d[var] = data[var][:, :, idx]
            coord_x, coord_y = x, y
        elif plane == 'xz':
            axis_coords = y
            idx = int(np.argmin(np.abs(axis_coords - position)))
            fields_2d = {}
            for var in ['u', 'v', 'w', 'p']:
                if var in data:
                    fields_2d[var] = data[var][:, idx, :]
            coord_x, coord_y = x, z
        elif plane == 'yz':
            axis_coords = x
            idx = int(np.argmin(np.abs(axis_coords - position)))
            fields_2d = {}
            for var in ['u', 'v', 'w', 'p']:
                if var in data:
                    fields_2d[var] = data[var][idx, :, :]
            coord_x, coord_y = y, z
        else:
            raise ValueError("plane 必須為 {xy|xz|yz}")
        
        # 插值到指定解析度
        slice_save: Dict[str, Any] = {
            'x': np.linspace(coord_x[0], coord_x[-1], slice_res[0]).astype(np.float32),
            'y': np.linspace(coord_y[0], coord_y[-1], slice_res[1]).astype(np.float32),
            'slice_plane': plane,
            'slice_position': float(axis_coords[idx]),
            'slice_position_target': float(position),
            'slice_position_error': float(abs(axis_coords[idx] - position))
        }
        for var, f2d in fields_2d.items():
            slice_save[var] = self._interpolate_to_resolution(
                f2d, coord_x, coord_y, slice_res
            ).astype(np.float32)
        
        # 保存
        out_path = self.cache_dir / "slices" / f"{plane}_pos{axis_coords[idx]:.3f}_{slice_res[0]}x{slice_res[1]}.npz"
        np.savez_compressed(out_path, **slice_save)
        self.logger.info(f"2D 切片已保存: {out_path}")
        
        # 報告
        report = {
            'plane': plane,
            'target_position': float(position),
            'actual_position': float(axis_coords[idx]),
            'position_error': float(abs(axis_coords[idx] - position)),
            'shape': {k: (slice_save[k].shape if isinstance(slice_save[k], np.ndarray) else None)
                      for k in slice_save.keys() if k in ['u','v','w','p']}
        }
        rep_path = self.cache_dir / "reports" / f"slice_{plane}_pos{axis_coords[idx]:.3f}_{slice_res[0]}x{slice_res[1]}.json"
        with open(rep_path, 'w') as f:
            json.dump(report, f, indent=2)
        return out_path

    # ---------------- 感測點與先驗 ----------------
    def _compute_periodic_distance_squared(self, coords1, coords2, domain_lengths):
        """計算週期性距離的平方 (支援多維)"""
        dists_sq = np.zeros(len(coords2))
        
        for dim, length in domain_lengths.items():
            dx = np.abs(coords1[dim] - coords2[:, dim])
            dx = np.minimum(dx, length - dx)
            dists_sq += dx**2
            
        # 對於非週期軸，使用標準歐式距離
        for dim in range(len(coords1)):
            if dim not in domain_lengths:
                d = coords1[dim] - coords2[:, dim]
                dists_sq += d**2
                
        return dists_sq

    def generate_sensor_points_with_periodicity(self,
                                               field_data: Dict[str, np.ndarray],
                                               K: int = 500,
                                               periodic_axes: Optional[List[int]] = None,
                                               use_periodic: bool = True,
                                               n_wrap_layers: int = 2,
                                               seam_weight: float = 1.0,
                                               noise_sigma: float = 0.02,
                                               dropout_prob: float = 0.10,
                                               min_dist_factor: float = 0.5) -> Dict[str, np.ndarray]:
        """
        生成考慮週期性邊界條件的感測點（使用底層 QR-Pivot API）
        
        改進：
        1. 空間過濾：使用最小距離約束防止點聚集
        2. 無偏邊界：支援 seam_weight=1.0
        3. 過採樣：從更多候選點中篩選

        Args:
            field_data: 完整場資料（必須包含 'x', 'y' 座標與 'u', 'v', 'w', 'p' 場）
            K: 感測點數量
            periodic_axes: 週期性軸索引列表（0=x, 1=y, 2=z）
                          預設 [0] 對應 2D 切片的 x 方向週期性
            use_periodic: 是否啟用週期性處理
            n_wrap_layers: 接縫處環繞層數（推薦 2-3）
            seam_weight: 環繞層權重係數（1.0 為無偏）
            noise_sigma: 高斯噪聲標準差
            dropout_prob: 隨機遺失比例
            min_dist_factor: 最小距離因子 (相對於平均間距)

        Returns:
            感測點資料字典
        """
        from pinnx.sensors.qr_pivot import QRPivotSelector

        self.logger.info(f"生成 {K} 個感測點（週期性處理: {use_periodic}, 空間過濾: {min_dist_factor > 0}）")

        # ============ 1. 提取座標與網格形狀 ============
        if 'x' in field_data and 'y' in field_data:
            x = np.asarray(field_data['x'])
            y = np.asarray(field_data['y'])
        else:
            coordinates = field_data.get('coordinates')
            if coordinates is None:
                raise KeyError("field_data 缺少 x/y 座標資訊")
            if isinstance(coordinates, dict):
                coord_dict = coordinates
            elif isinstance(coordinates, np.ndarray) and coordinates.ndim == 0:
                coord_dict = coordinates.item()
            else:
                raise TypeError(f"不支援的 coordinates 型別: {type(coordinates)}")
            x = np.asarray(coord_dict['x'])
            y = np.asarray(coord_dict['y'])

        nx, ny = len(x), len(y)
        self.logger.info(f"網格形狀: {nx} × {ny} = {nx*ny} 點")

        # ============ 2. 建立座標網格 ============
        X, Y = np.meshgrid(x, y, indexing='ij')
        coords = np.stack([X.ravel(), Y.ravel()], axis=-1)  # [nx*ny, 2]

        # ============ 3. 堆疊場數據為快照矩陣 ============
        snapshots = []
        available_vars = []
        for var in ['u', 'v', 'w', 'p']:
            if var in field_data:
                field_arr = np.asarray(field_data[var])
                # 確保形狀為 (nx, ny) 或可 reshape
                if field_arr.shape != (nx, ny):
                    try:
                        field_arr = field_arr.reshape(nx, ny)
                    except ValueError:
                        self.logger.warning(f"跳過變量 {var}：形狀 {field_arr.shape} 無法 reshape 至 ({nx}, {ny})")
                        continue
                snapshots.append(field_arr.ravel()[:, np.newaxis])
                available_vars.append(var)

        if not snapshots:
            raise ValueError("未找到有效的場變量（u, v, w, p）")

        data_matrix = np.hstack(snapshots)  # [nx*ny, n_vars]
        
        # Z-score 標準化 (對特徵維度)
        mean_val = data_matrix.mean(axis=0)
        std_val = data_matrix.std(axis=0) + 1e-8
        data_matrix_norm = (data_matrix - mean_val) / std_val
        
        self.logger.info(f"快照矩陣: {data_matrix.shape}，變量: {available_vars} (已標準化)")

        # ============ 4. 創建 QR-Pivot 選擇器 ============
        if periodic_axes is None:
            # 預設 2D 切片僅 x 方向週期
            periodic_axes = [0] if use_periodic else None
        
        indices = []
        metrics = {}

        if use_periodic and periodic_axes:
            # 週期性模式 (含空間過濾)
            
            # 計算域長度
            domain_lengths = {}
            domain_area = 1.0
            if 0 in periodic_axes: domain_lengths[0] = float(x[-1] - x[0])
            if 1 in periodic_axes: domain_lengths[1] = float(y[-1] - y[0])
            
            # 計算總面積 (用於最小距離估算)
            Lx = float(x[-1] - x[0])
            Ly = float(y[-1] - y[0])
            domain_area = Lx * Ly

            # 過採樣設置
            oversample_factor = 3.0
            K_oversample = int(K * oversample_factor)
            K_oversample = min(K_oversample, len(coords))
            
            self.logger.info(
                f"週期性配置: axes={periodic_axes}, "
                f"wrap_layers={n_wrap_layers}, "
                f"oversample={K}->{K_oversample}"
            )

            selector = QRPivotSelector(
                use_circular_indexing=True,
                n_wrap_layers=n_wrap_layers,
                seam_weight=seam_weight,
                mode='column',
                pivoting=True,
                regularization=1e-12
            )

            # 執行 QR-Pivot (獲取候選點)
            candidate_indices, metrics = selector.select_sensors(
                data_matrix_norm,
                n_sensors=K_oversample,
                coords=coords,
                grid_shape=(nx, ny),
                periodic_axes=periodic_axes,
                domain_lengths=domain_lengths
            )
            
            # 執行空間過濾 (Minimum Distance Constraint)
            characteristic_dist = np.sqrt(domain_area / K)
            min_dist = min_dist_factor * characteristic_dist
            min_dist_sq = min_dist ** 2
            
            self.logger.info(f"空間過濾: min_dist={min_dist:.4f} (avg_spacing={characteristic_dist:.4f})")
            
            final_indices = []
            final_coords = []
            skipped_count = 0
            
            for idx in candidate_indices:
                if len(final_indices) >= K:
                    break
                
                current_coord = coords[idx]
                
                if not final_indices:
                    final_indices.append(idx)
                    final_coords.append(current_coord)
                    continue
                
                # 計算與已選點的距離
                dists_sq = self._compute_periodic_distance_squared(
                    current_coord, 
                    np.array(final_coords), 
                    domain_lengths
                )
                
                if np.all(dists_sq >= min_dist_sq):
                    final_indices.append(idx)
                    final_coords.append(current_coord)
                else:
                    skipped_count += 1
            
            # 回補點數 (如果過濾太嚴格)
            if len(final_indices) < K:
                self.logger.warning(f"過濾後點數不足 ({len(final_indices)}/{K})，正在回補...")
                existing_set = set(final_indices)
                for idx in candidate_indices:
                    if len(final_indices) >= K:
                        break
                    if idx not in existing_set:
                        final_indices.append(idx)
            
            indices = np.array(final_indices)
            metrics['skipped_count'] = skipped_count
            metrics['min_dist_factor'] = min_dist_factor

        else:
            # 標準模式（不考慮週期性）
            selector = QRPivotSelector(
                use_circular_indexing=False,
                mode='column',
                pivoting=True,
                regularization=1e-12
            )

            self.logger.info("標準 QR-Pivot（無週期性處理）")
            indices, metrics = selector.select_sensors(data_matrix_norm, n_sensors=K)

        # ============ 5. 提取感測點座標與數據 ============
        sensor_points = coords[indices]  # [K, 2]
        sensor_data = {}

        for i, var in enumerate(available_vars):
            field_values = data_matrix[indices, i]  # [K]

            # 添加噪聲
            if noise_sigma > 0:
                noise_scale = np.std(field_values)
                noise = np.random.normal(0, noise_sigma * noise_scale, len(field_values))
                field_values = field_values + noise

            # 隨機遺失
            if dropout_prob > 0:
                n_dropout = int(dropout_prob * len(field_values))
                if n_dropout > 0:
                    dropout_indices = np.random.choice(len(field_values), n_dropout, replace=False)
                    field_values[dropout_indices] = np.nan

            sensor_data[var] = field_values

        # ============ 6. 組裝結果 ============
        selection_info = {
            'strategy': 'qr_pivot_periodic' if use_periodic else 'qr_pivot_standard',
            'periodic_axes': periodic_axes if use_periodic else None,
            'n_wrap_layers': n_wrap_layers if use_periodic else 0,
            'use_periodic': use_periodic,
            'K_requested': K,
            'K_actual': len(indices),
            'available_vars': available_vars,
            **metrics
        }

        result = {
            'sensor_points': sensor_points,
            'sensor_data': sensor_data,
            'sensor_indices': indices,
            'selection_info': selection_info,
            'noise_sigma': noise_sigma,
            'dropout_prob': dropout_prob
        }

        # ============ 7. 快取結果 ============
        method_suffix = 'periodic' if use_periodic else 'standard'
        cache_file = self.cache_dir / f"sensors_K{K}_qr_pivot_{method_suffix}.npz"
        np.savez_compressed(cache_file, **result)
        self.logger.info(f"感測點已快取: {cache_file}")

        # 打印週期性覆蓋率統計
        if use_periodic and periodic_axes and 0 in periodic_axes:
            x_sensors = sensor_points[:, 0]
            x_min, x_max = x[0], x[-1]
            seam_threshold = 0.05 * (x_max - x_min)  # 5% 邊界區域
            near_x0 = np.sum(x_sensors < (x_min + seam_threshold))
            near_xL = np.sum(x_sensors > (x_max - seam_threshold))
            self.logger.info(
                f"週期接縫覆蓋: x≈0 ({near_x0} 點), "
                f"x≈L ({near_xL} 點), 總計 {near_x0+near_xL}/{K}"
            )

        return result

    def generate_sensor_points(self,
                              field_data: Dict[str, np.ndarray],
                              K: int = 8,
                              method: str = "qr_pivot",
                              noise_sigma: float = 0.02,
                              dropout_prob: float = 0.10,
                              use_periodic: bool = False,
                              min_dist_factor: float = 0.5) -> Dict[str, np.ndarray]:
        """
        生成稀疏感測點資料（向後相容包裝器）

        Args:
            field_data: 完整場資料
            K: 感測點數量
            method: 感測點選擇方法 ('qr_pivot', 'random')
            noise_sigma: 高斯噪聲標準差
            dropout_prob: 隨機遺失比例
            use_periodic: 是否啟用週期性處理（預設 False 保持向後相容）

        Returns:
            感測點資料與位置
        """
        if method == "qr_pivot" and use_periodic:
            # 使用週期性處理的新函數
            return self.generate_sensor_points_with_periodicity(
                field_data=field_data,
                K=K,
                periodic_axes=[0],  # 2D 切片 x 方向週期
                use_periodic=True,
                n_wrap_layers=2,
                noise_sigma=noise_sigma,
                dropout_prob=dropout_prob,
                min_dist_factor=min_dist_factor
            )

        # ============ 舊版邏輯（向後相容）============
        self.logger.info(f"生成 {K} 個感測點，方法: {method}（標準模式）")

        # 兼容性處理：優先使用平鋪的 x/y，否則從 coordinates 還原
        if 'x' in field_data and 'y' in field_data:
            x = np.asarray(field_data['x'])
            y = np.asarray(field_data['y'])
        else:
            coordinates = field_data.get('coordinates')
            if coordinates is None:
                raise KeyError("field_data 缺少 x/y 與 coordinates")
            if isinstance(coordinates, dict):
                coord_dict = coordinates
            elif isinstance(coordinates, np.ndarray):
                try:
                    coord_dict = coordinates.item()
                except Exception as e:
                    raise TypeError(f"無法從 coordinates 還原字典，型別: {type(coordinates)}") from e
            else:
                raise TypeError(f"不支援的 coordinates 型別: {type(coordinates)}")
            x = np.asarray(coord_dict['x'])
            y = np.asarray(coord_dict['y'])

        # 建立完整網格
        X, Y = np.meshgrid(x, y, indexing='ij')
        points_full = np.stack([X.ravel(), Y.ravel()], axis=-1)

        # 感測點選擇
        if method == "qr_pivot":
            selector = FieldSensorSelector(
                strategy='qr_pivot',
                feature_scaling='standard',
                nan_policy='raise'
            )
            selection = selector.select(field_data, n_sensors=K)
            sensor_indices = selection.indices
            selection_info = dict(selection.metrics)
            selection_info['strategy'] = 'qr_pivot'
            sensor_points = selection.coordinates()
            if sensor_points is None:
                sensor_points = points_full[sensor_indices]
        elif method == "random":
            np.random.seed(42)
            sensor_indices = np.random.choice(len(points_full), K, replace=False)
            selection_info = {"method": "random"}
            sensor_points = points_full[sensor_indices]
        else:
            raise ValueError(f"未支援的感測點選擇方法: {method}")

        # 提取感測點座標與數值
        sensor_data = {}

        for var in ['u', 'v', 'w', 'p']:
            if var in field_data:
                if method == "qr_pivot":
                    component_values = selection.component_values.get(var)
                    if component_values is None:
                        continue
                    field_values = component_values
                else:
                    full_values = field_data[var].reshape(-1)
                    field_values = full_values[sensor_indices]

                if field_values.ndim == 2 and field_values.shape[1] == 1:
                    field_values = field_values[:, 0]

                # 添加噪聲
                if noise_sigma > 0:
                    if field_values.ndim == 1:
                        noise_scale = np.std(field_values)
                        noise = np.random.normal(0, noise_sigma * noise_scale, len(field_values))
                        field_values = field_values + noise
                    else:
                        noise_scale = np.std(field_values, axis=0, keepdims=True)
                        noise = np.random.normal(0, noise_sigma, size=field_values.shape) * noise_scale
                        field_values = field_values + noise

                # 隨機遺失
                if dropout_prob > 0:
                    n_dropout = int(dropout_prob * len(field_values))
                    dropout_indices = np.random.choice(len(field_values), n_dropout, replace=False)
                    if field_values.ndim == 1:
                        field_values[dropout_indices] = np.nan
                    else:
                        field_values[dropout_indices, :] = np.nan

                sensor_data[var] = field_values

        result = {
            'sensor_points': sensor_points,
            'sensor_data': sensor_data,
            'sensor_indices': sensor_indices,
            'selection_info': selection_info,
            'noise_sigma': noise_sigma,
            'dropout_prob': dropout_prob
        }

        # 快取感測點資料
        cache_file = self.cache_dir / f"sensors_K{K}_{method}.npz"
        np.savez_compressed(cache_file, **result)
        self.logger.info(f"感測點資料已快取至: {cache_file}")

        return result
    
    def generate_lowfi_prior(self, 
                           resolution: Tuple[int, int] = (128, 64),
                           prior_type: str = "simplified_rans") -> Dict[str, np.ndarray]:
        """
        生成低保真先驗資料 (RANS/簡化模型)
        
        Args:
            resolution: 先驗資料解析度
            prior_type: 先驗類型
            
        Returns:
            低保真先驗場
        """
        self.logger.info(f"生成低保真先驗資料，類型: {prior_type}")
        
        # 建立網格
        domain_x = self.config.domain_x if self.config.domain_x else [0.0, 25.13]
        domain_y = self.config.domain_y if self.config.domain_y else [-1.0, 1.0]
        x = np.linspace(domain_x[0], domain_x[1], resolution[0])
        y = np.linspace(domain_y[0], domain_y[1], resolution[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        if prior_type == "simplified_rans":
            # 簡化的 Channel Flow 速度剖面
            y_plus = np.abs(Y) * self.config.Re_tau  # y+ = y * Re_tau
            
            # 使用簡化的對數律 + 線性律
            u_mean = np.zeros_like(X)
            for i in range(len(x)):
                for j in range(len(y)):
                    yp = y_plus[i, j]
                    if yp < 11:  # 粘性底層
                        u_mean[i, j] = yp
                    else:  # 對數律層
                        u_mean[i, j] = 1/0.41 * np.log(yp) + 5.0
            
            # 添加流向波動 (簡化湍流結構)
            domain_x_len = domain_x[1] - domain_x[0]
            u_fluctuation = 0.1 * np.sin(2 * np.pi * X / domain_x_len) * \
                           np.exp(-10 * np.abs(Y))
            
            prior_data = {
                'coordinates': {'x': x, 'y': y},
                'u': u_mean + u_fluctuation,
                'v': 0.05 * np.sin(4 * np.pi * X / domain_x_len) * \
                     np.sin(np.pi * Y / 2),  # 小幅壁法向速度
                'p': -1.0 * X + 0.01 * np.sin(2 * np.pi * X / domain_x_len),  # 壓力梯度
            }
            
        else:
            raise ValueError(f"未支援的先驗類型: {prior_type}")
        
        # 快取先驗資料
        cache_file = self.cache_dir / f"lowfi_prior_{prior_type}_{resolution[0]}x{resolution[1]}.npz"
        np.savez_compressed(cache_file, **prior_data)
        self.logger.info(f"低保真先驗已快取至: {cache_file}")
        
        return prior_data
    
    def create_complete_dataset(self,
                              resolution: Tuple[int, int] = (128, 64),
                              K: int = 8,
                              sensor_method: str = "qr_pivot") -> Dict[str, Any]:
        """
        建立完整的 Channel Flow Re1000 資料集
        
        包含：高保真場、稀疏感測點、低保真先驗
        """
        self.logger.info("建立完整 Channel Flow Re1000 資料集")
        
        # 1. 獲取高保真場資料
        hifi_data = self.fetch_cutout_data(resolution)
        
        # 2. 生成稀疏感測點
        sensor_data = self.generate_sensor_points(hifi_data, K, sensor_method)
        
        # 3. 生成低保真先驗
        lowfi_data = self.generate_lowfi_prior(resolution)
        
        # 4. 組合完整資料集
        dataset = {
            'hifi_data': hifi_data,
            'sensor_data': sensor_data,
            'lowfi_data': lowfi_data,
            'config': self.config.__dict__,
            'metadata': {
                'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'resolution': resolution,
                'K_sensors': K,
                'sensor_method': sensor_method,
                'Re_tau': self.config.Re_tau
            }
        }
        
        # 快取完整資料集
        cache_file = self.cache_dir / f"complete_dataset_{resolution[0]}x{resolution[1]}_K{K}.npz"
        np.savez_compressed(cache_file, **dataset)
        self.logger.info(f"完整資料集已快取至: {cache_file}")
        
        return dataset
    
    def visualize_data(self, dataset: Dict[str, Any], output_dir: Optional[str] = None):
        """資料視覺化"""
        if output_dir is None:
            output_dir_path = self.cache_dir / "visualization"
        else:
            output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        hifi_data = dataset['hifi_data']
        sensor_data = dataset['sensor_data']
        
        # 建立圖表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Channel Flow Re1000 Dataset Visualization', fontsize=16)
        
        # 座標
        x = hifi_data['coordinates']['x']
        y = hifi_data['coordinates']['y']
        
        # 感測點座標
        sensor_points = sensor_data['sensor_points']
        
        # 繪製各變數場
        for i, var in enumerate(['u', 'v', 'p']):
            if var in hifi_data:
                # 完整場
                ax1 = axes[0, i]
                im1 = ax1.contourf(x, y, hifi_data[var].T, levels=50, cmap='RdBu_r')
                ax1.scatter(sensor_points[:, 0], sensor_points[:, 1], 
                           c='black', s=30, marker='o', label=f'Sensors (K={len(sensor_points)})')
                ax1.set_title(f'High-Fi {var.upper()} Field')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.legend()
                plt.colorbar(im1, ax=ax1)
                
                # 感測點數值
                ax2 = axes[1, i]
                if var in sensor_data['sensor_data']:
                    sensor_values = sensor_data['sensor_data'][var]
                    valid_mask = ~np.isnan(sensor_values)
                    ax2.scatter(sensor_points[valid_mask, 0], sensor_points[valid_mask, 1], 
                               c=sensor_values[valid_mask], s=50, cmap='RdBu_r')
                    ax2.set_title(f'Sensor {var.upper()} Values')
                    ax2.set_xlabel('x')
                    ax2.set_ylabel('y')
        
        plt.tight_layout()
        
        # 儲存圖表
        plot_file = output_dir_path / "dataset_overview.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"資料視覺化已儲存至: {plot_file}")


def load_config(config_file: str) -> ChannelFlowConfig:
    """載入配置檔案"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 提取 Channel Flow 相關配置
    jhtdb_config = config_dict.get('data', {}).get('jhtdb_config', {})
    
    return ChannelFlowConfig(
        dataset=jhtdb_config.get('dataset_name', 'channel'),
        domain_x=jhtdb_config.get('domain', {}).get('x', [0, 25.13]),
        domain_y=jhtdb_config.get('domain', {}).get('y', [-1, 1]),
        domain_z=jhtdb_config.get('domain', {}).get('z', [0, 9.42]),
        Re_tau=config_dict.get('physics', {}).get('channel_flow', {}).get('Re_tau', 1000.0),
        slice_position=config_dict.get('data', {}).get('slice_config', {}).get('z_position', 4.71)
    )


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='Channel Flow Re1000 JHTDB 資料擷取')
    
    # 基本參數
    parser.add_argument('--config', type=str, default='configs/channel_flow_re1000.yml',
                       help='配置檔案路徑')
    parser.add_argument('--cache_dir', type=str, default='./data/jhtdb/channel_flow_re1000',
                       help='快取目錄')
    parser.add_argument('--log_level', type=str, default='info',
                       choices=['debug', 'info', 'warning', 'error'],
                       help='日誌等級')
    parser.add_argument('--use_mock', action='store_true', help='使用模擬 JHTDB 資料來源')
    
    # 運行模式
    parser.add_argument('--cutout_mode', action='store_true',
                       help='只下載 2D cutout（舊流程）')
    parser.add_argument('--sensor_only', action='store_true', 
                       help='只生成感測點資料')
    parser.add_argument('--complete_dataset', action='store_true',
                       help='建立完整資料集 (預設)')
    parser.add_argument('--cutout3d_mode', action='store_true', help='下載並快取 3D cutout')
    parser.add_argument('--extract_slice', action='store_true', help='從 3D 快取提取 2D 切片')
    
    # 2D 參數設定
    parser.add_argument('--resolution', type=int, nargs=2, default=[128, 64],
                       help='2D 切片解析度 [nx, ny]')
    # 3D 參數設定
    parser.add_argument('--resolution3d', type=int, nargs=3, default=[128, 128, 32],
                       help='3D cutout 解析度 [nx, ny, nz]（mock 有效）')
    parser.add_argument('--plane', type=str, default='xy', choices=['xy','xz','yz'],
                       help='切片平面')
    parser.add_argument('--position', type=float, default=4.71, help='切片位置（依平面軸）')
    parser.add_argument('--slice_res', type=int, nargs=2, default=[128, 64], help='切片輸出解析度 [nx, ny]')
    
    # 感測點
    parser.add_argument('--K', type=int, default=8,
                       help='感測點數量')
    parser.add_argument('--sensor_method', type=str, default='qr_pivot',
                       choices=['qr_pivot', 'random'],
                       help='感測點選擇方法')
    parser.add_argument('--use_periodic', action='store_true',
                       help='啟用週期性邊界處理（僅對 qr_pivot 有效）')
    parser.add_argument('--noise_sigma', type=float, default=0.02,
                       help='感測點噪聲標準差')
    parser.add_argument('--dropout_prob', type=float, default=0.10,
                       help='感測點隨機遺失比例')
    parser.add_argument('--min_dist_factor', type=float, default=0.5,
                       help='最小距離因子 (用於空間過濾，0.5 為預設)')
    
    # 輸出選項
    parser.add_argument('--visualize', action='store_true',
                       help='生成資料視覺化')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    # 設定日誌
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fetch_channel_flow.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # 載入配置
        config = load_config(args.config)
        logger.info(f"載入配置檔案: {args.config}")
        
        # 建立資料擷取器
        fetcher = ChannelFlowDataFetcher(config, args.cache_dir, use_mock=args.use_mock)
        resolution = tuple(args.resolution)
        res3d = tuple(args.resolution3d)
        
        # 執行對應模式
        if args.cutout3d_mode:
            logger.info("執行 3D cutout 模式")
            fetcher.fetch_cutout3d_data(resolution3d=res3d)
        elif args.extract_slice:
            logger.info("從 3D 快取提取 2D 切片")
            out_path = fetcher.extract_and_save_slice(resolution3d=res3d,
                                                      plane=args.plane,
                                                      position=args.position,
                                                      slice_res=tuple(args.slice_res))
            logger.info(f"切片已輸出: {out_path}")
        elif args.cutout_mode:
            logger.info("執行 2D cutout 模式（舊流程）")
            hifi_data = fetcher.fetch_cutout_data(resolution)
            logger.info(f"完成 cutout 資料下載，形狀: {hifi_data['u'].shape if 'u' in hifi_data else 'N/A'}")
        elif args.sensor_only:
            logger.info("執行感測點生成模式")
            # 需要先載入現有的高保真資料
            cache_file = Path(args.cache_dir) / f"cutout_{resolution[0]}x{resolution[1]}.npz"
            if cache_file.exists():
                hifi_data = dict(np.load(cache_file, allow_pickle=True))
                sensor_data = fetcher.generate_sensor_points(
                    hifi_data, args.K, args.sensor_method,
                    args.noise_sigma, args.dropout_prob,
                    use_periodic=args.use_periodic,  # ✅ 傳入週期性參數
                    min_dist_factor=args.min_dist_factor # ✅ 新增最小距離因子
                )
                logger.info(f"完成 {args.K} 個感測點生成（週期性: {args.use_periodic}）")
            else:
                logger.error(f"未找到高保真資料快取: {cache_file}")
                return 1
        else:
            # 預設：建立完整資料集
            logger.info("執行完整資料集建立模式")
            dataset = fetcher.create_complete_dataset(resolution, args.K, args.sensor_method)
            logger.info("完成完整資料集建立")
            
            # 可視化
            if args.visualize:
                fetcher.visualize_data(dataset, args.output_dir)
        
        logger.info("資料擷取完成")
        return 0
        
    except Exception as e:
        logger.error(f"資料擷取失敗: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

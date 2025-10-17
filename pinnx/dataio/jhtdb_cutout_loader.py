"""
JHTDB Cutout Service HDF5 數據載入器

從 JHTDB Cutout Service (https://turbulence.idies.jhu.edu/cutout/) 
下載的 HDF5 文件載入器

支援：
- Channel Flow Re_τ=1000 (1024×256×768 下採樣)
- 速度場 (u, v, w) 和壓力場 (p)
- 非均勻網格處理（Y 方向）
- 與 QR-Pivot 感測點選擇整合

文件格式：
- Pressure: Pressure_0001 [z, y, x, 1] + xcoor, ycoor, zcoor
- Velocity: Velocity_0001 [z, y, x, 3] + xcoor, ycoor, zcoor
  (注意：Velocity 文件格式可能與 Pressure 不同，需驗證)
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class JHTDBCutoutLoader:
    """JHTDB Cutout Service HDF5 數據載入器"""
    
    def __init__(self, data_dir: str = "data/jhtdb/channel_flow_re1000/raw"):
        """
        Args:
            data_dir: Cutout HDF5 文件所在目錄
        """
        self.data_dir = Path(data_dir)
        
        # 預期的文件結構（處理檔名包含空格的情況）
        self.file_patterns = {
            'pressure': [
                'JHU Turbulence Channel_t1_pressure.h5',
                'channel_t1_pressure.h5',  # 重命名後
                '*pressure*.h5'
            ],
            'velocity': [
                'JHU Turbulence Channel_t1_velocity.h5',
                'channel_t1_velocity.h5',  # 重命名後
                '*velocity*.h5'
            ]
        }
        
        # 標準化後的文件路徑
        self.pressure_file = None
        self.velocity_file = None
        
        # 快取數據
        self._coords_cache = None
        self._pressure_cache = None
        self._velocity_cache = None
        
        self._find_files()
    
    def _find_files(self):
        """查找並驗證 HDF5 文件"""
        logger.info(f"在目錄 {self.data_dir} 中查找 Cutout 文件...")
        
        if not self.data_dir.exists():
            logger.warning(f"目錄不存在: {self.data_dir}")
            return
        
        # 查找壓力場文件
        for pattern in self.file_patterns['pressure']:
            files = list(self.data_dir.glob(pattern))
            if files:
                self.pressure_file = files[0]
                logger.info(f"找到壓力場文件: {self.pressure_file.name}")
                break
        
        # 查找速度場文件
        for pattern in self.file_patterns['velocity']:
            files = list(self.data_dir.glob(pattern))
            if files:
                self.velocity_file = files[0]
                logger.info(f"找到速度場文件: {self.velocity_file.name}")
                break
        
        # 驗證文件
        if self.pressure_file:
            if not self._verify_hdf5_file(self.pressure_file):
                logger.error(f"壓力場文件損壞: {self.pressure_file}")
                self.pressure_file = None
        
        if self.velocity_file:
            if not self._verify_hdf5_file(self.velocity_file):
                logger.error(f"速度場文件損壞: {self.velocity_file}")
                self.velocity_file = None
    
    def _verify_hdf5_file(self, file_path: Path) -> bool:
        """驗證 HDF5 文件完整性"""
        try:
            with h5py.File(file_path, 'r') as f:
                # 基本檢查：確保至少有一個 dataset
                if len(f.keys()) == 0:
                    return False
                return True
        except Exception as e:
            logger.error(f"HDF5 文件驗證失敗: {e}")
            return False
    
    def load_coordinates(self) -> Dict[str, np.ndarray]:
        """
        載入座標網格
        
        Returns:
            {'x': x_coords, 'y': y_coords, 'z': z_coords}
        """
        if self._coords_cache is not None:
            return self._coords_cache
        
        # 優先從壓力場文件讀取（已驗證可用）
        file_to_use = self.pressure_file or self.velocity_file
        
        if file_to_use is None:
            raise FileNotFoundError("未找到有效的 Cutout HDF5 文件")
        
        logger.info(f"從 {file_to_use.name} 載入座標...")
        
        with h5py.File(file_to_use, 'r') as f:
            x = np.array(f['xcoor'][:])
            y = np.array(f['ycoor'][:])
            z = np.array(f['zcoor'][:])
        
        coords = {'x': x, 'y': y, 'z': z}
        
        logger.info(f"座標範圍:")
        logger.info(f"  X: [{x.min():.4f}, {x.max():.4f}], shape={x.shape}")
        logger.info(f"  Y: [{y.min():.4f}, {y.max():.4f}], shape={y.shape} (非均勻)")
        logger.info(f"  Z: [{z.min():.4f}, {z.max():.4f}], shape={z.shape}")
        
        self._coords_cache = coords
        return coords
    
    def load_pressure(self) -> np.ndarray:
        """
        載入壓力場
        
        Returns:
            pressure: [z, y, x] 或 [x, y, z]（自動轉置）
        """
        if self._pressure_cache is not None:
            return self._pressure_cache
        
        if self.pressure_file is None:
            raise FileNotFoundError("未找到壓力場文件")
        
        logger.info(f"載入壓力場: {self.pressure_file.name}")
        
        with h5py.File(self.pressure_file, 'r') as f:
            # 查找壓力場數據集
            pressure_key = None
            for key in f.keys():
                if 'pressure' in key.lower():
                    pressure_key = key
                    break
            
            if pressure_key is None:
                raise KeyError(f"未找到壓力場 dataset，可用 keys: {list(f.keys())}")
            
            logger.info(f"使用 dataset: {pressure_key}")
            
            p_data = f[pressure_key][:]
            
            logger.info(f"原始形狀: {p_data.shape}")
            
            # 移除單一維度 (如果是 [z, y, x, 1])
            if p_data.ndim == 4 and p_data.shape[-1] == 1:
                p_data = p_data[..., 0]
                logger.info(f"移除單一維度後: {p_data.shape}")
            
            # 轉置為標準順序 [x, y, z]（如果是 [z, y, x]）
            if p_data.shape[0] < p_data.shape[2]:  # 簡單啟發式判斷
                p_data = np.transpose(p_data, (2, 1, 0))
                logger.info(f"轉置為 [x, y, z]: {p_data.shape}")
        
        logger.info(f"壓力場統計: min={p_data.min():.6f}, max={p_data.max():.6f}, "
                   f"mean={p_data.mean():.6f}, std={p_data.std():.6f}")
        
        self._pressure_cache = p_data
        return p_data
    
    def load_velocity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        載入速度場
        
        Returns:
            (u, v, w): 每個分量形狀為 [x, y, z]
        """
        if self._velocity_cache is not None:
            return self._velocity_cache
        
        if self.velocity_file is None:
            raise FileNotFoundError(
                "未找到速度場文件或文件已損壞。\n"
                "請從 https://turbulence.idies.jhu.edu/cutout/ 重新下載"
            )
        
        logger.info(f"載入速度場: {self.velocity_file.name}")
        
        with h5py.File(self.velocity_file, 'r') as f:
            # 查找速度場數據集
            velocity_key = None
            for key in f.keys():
                if 'velocity' in key.lower():
                    velocity_key = key
                    break
            
            if velocity_key is None:
                raise KeyError(f"未找到速度場 dataset，可用 keys: {list(f.keys())}")
            
            logger.info(f"使用 dataset: {velocity_key}")
            
            vel_data = f[velocity_key][:]
            
            logger.info(f"原始形狀: {vel_data.shape}")
            
            # 預期格式: [z, y, x, 3] 或 [x, y, z, 3]
            if vel_data.ndim != 4 or vel_data.shape[-1] != 3:
                raise ValueError(f"速度場格式錯誤: {vel_data.shape}，預期 [..., 3]")
            
            # 轉置為 [x, y, z, 3]（如果是 [z, y, x, 3]）
            if vel_data.shape[0] < vel_data.shape[2]:
                vel_data = np.transpose(vel_data, (2, 1, 0, 3))
                logger.info(f"轉置為 [x, y, z, 3]: {vel_data.shape}")
            
            # 分離速度分量
            u = vel_data[..., 0]
            v = vel_data[..., 1]
            w = vel_data[..., 2]
        
        logger.info(f"速度場統計:")
        logger.info(f"  U: min={u.min():.4f}, max={u.max():.4f}, mean={u.mean():.4f}")
        logger.info(f"  V: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")
        logger.info(f"  W: min={w.min():.4f}, max={w.max():.4f}, mean={w.mean():.4f}")
        
        self._velocity_cache = (u, v, w)
        return u, v, w
    
    def load_full_state(self) -> Dict[str, np.ndarray]:
        """
        載入完整流場狀態
        
        Returns:
            {
                'coords': {'x': ..., 'y': ..., 'z': ...},
                'u': [x, y, z],
                'v': [x, y, z],
                'w': [x, y, z],
                'p': [x, y, z]
            }
        """
        logger.info("載入完整流場狀態...")
        
        state = {}
        
        # 載入座標
        state['coords'] = self.load_coordinates()
        
        # 載入壓力場
        try:
            state['p'] = self.load_pressure()
        except Exception as e:
            logger.warning(f"壓力場載入失敗: {e}")
            state['p'] = None
        
        # 載入速度場
        try:
            u, v, w = self.load_velocity()
            state['u'] = u
            state['v'] = v
            state['w'] = w
        except Exception as e:
            logger.warning(f"速度場載入失敗: {e}")
            state['u'] = None
            state['v'] = None
            state['w'] = None
        
        # 報告載入狀態
        available_vars = [k for k, v in state.items() if v is not None and k != 'coords']
        logger.info(f"可用變數: {available_vars}")
        
        return state
    
    def get_grid_shape(self) -> Tuple[int, int, int]:
        """獲取網格形狀"""
        coords = self.load_coordinates()
        return (len(coords['x']), len(coords['y']), len(coords['z']))
    
    def get_physical_domain(self) -> Dict[str, Tuple[float, float]]:
        """獲取物理域範圍"""
        coords = self.load_coordinates()
        return {
            'x': (coords['x'].min(), coords['x'].max()),
            'y': (coords['y'].min(), coords['y'].max()),
            'z': (coords['z'].min(), coords['z'].max())
        }
    
    def extract_points(self, 
                      points: np.ndarray,
                      variables: List[str] = ['u', 'v', 'w', 'p']) -> Dict[str, np.ndarray]:
        """
        從場數據中提取指定點的值（使用線性插值）
        
        Args:
            points: [N, 3] 物理座標點 (x, y, z)
            variables: 需要提取的變數列表
            
        Returns:
            {var_name: [N] 插值結果}
        """
        from scipy.interpolate import RegularGridInterpolator
        
        logger.info(f"從 {len(points)} 個點提取 {variables}...")
        
        # 載入數據
        state = self.load_full_state()
        coords = state['coords']
        
        results = {}
        
        for var in variables:
            if var not in state or state[var] is None:
                logger.warning(f"變數 {var} 不可用，跳過")
                continue
            
            # 創建插值器（注意：RegularGridInterpolator 需要升序座標）
            # 我們的座標已經是 [x, y, z]，數據也是 [x, y, z]
            interpolator = RegularGridInterpolator(
                (coords['x'], coords['y'], coords['z']),
                state[var],
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            
            # 插值
            results[var] = interpolator(points)
            
            logger.info(f"  {var}: {results[var].shape}, "
                       f"range=[{np.nanmin(results[var]):.4f}, {np.nanmax(results[var]):.4f}]")
        
        return results
    
    def rename_files(self):
        """重命名文件以移除空格（方便後續處理）"""
        renames = []
        
        if self.pressure_file and ' ' in self.pressure_file.name:
            new_name = self.data_dir / 'channel_t1_pressure.h5'
            renames.append((self.pressure_file, new_name))
        
        if self.velocity_file and ' ' in self.velocity_file.name:
            new_name = self.data_dir / 'channel_t1_velocity.h5'
            renames.append((self.velocity_file, new_name))
        
        if renames:
            logger.info("準備重命名文件以移除空格:")
            for old, new in renames:
                logger.info(f"  {old.name} -> {new.name}")
            
            confirm = input("執行重命名？(y/n): ")
            if confirm.lower() == 'y':
                for old, new in renames:
                    old.rename(new)
                    logger.info(f"✅ 已重命名: {new.name}")
                
                # 重新查找文件
                self._find_files()
            else:
                logger.info("取消重命名")
    
    def check_file_integrity(self):
        """檢查並報告文件完整性"""
        logger.info("=== 文件完整性檢查 ===")
        
        # 檢查壓力場
        if self.pressure_file:
            logger.info(f"✅ 壓力場文件: {self.pressure_file.name}")
            try:
                with h5py.File(self.pressure_file, 'r') as f:
                    logger.info(f"   Keys: {list(f.keys())}")
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            logger.info(f"   {key}: {f[key].shape}, {f[key].dtype}")
            except Exception as e:
                logger.error(f"   錯誤: {e}")
        else:
            logger.warning("❌ 壓力場文件未找到或已損壞")
        
        # 檢查速度場
        if self.velocity_file:
            logger.info(f"✅ 速度場文件: {self.velocity_file.name}")
            try:
                with h5py.File(self.velocity_file, 'r') as f:
                    logger.info(f"   Keys: {list(f.keys())}")
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            logger.info(f"   {key}: {f[key].shape}, {f[key].dtype}")
            except Exception as e:
                logger.error(f"   錯誤: {e}")
        else:
            logger.warning("❌ 速度場文件未找到或已損壞")
            logger.info("   請從 https://turbulence.idies.jhu.edu/cutout/ 重新下載")


if __name__ == "__main__":
    # 測試載入器
    logging.basicConfig(level=logging.INFO)
    
    loader = JHTDBCutoutLoader()
    
    # 檢查文件完整性
    loader.check_file_integrity()
    
    # 嘗試載入數據
    try:
        state = loader.load_full_state()
        
        if state['p'] is not None:
            print(f"\n=== 壓力場成功載入 ===")
            print(f"形狀: {state['p'].shape}")
            print(f"範圍: [{state['p'].min():.6f}, {state['p'].max():.6f}]")
        
        if state['u'] is not None:
            print(f"\n=== 速度場成功載入 ===")
            print(f"U 形狀: {state['u'].shape}")
            print(f"V 形狀: {state['v'].shape}")
            print(f"W 形狀: {state['w'].shape}")
        
        # 測試點提取（如果數據可用）
        if state['p'] is not None:
            coords = state['coords']
            test_points = np.array([
                [coords['x'][0], coords['y'][len(coords['y'])//2], coords['z'][0]],
                [coords['x'][-1], coords['y'][len(coords['y'])//2], coords['z'][-1]]
            ])
            
            extracted = loader.extract_points(test_points, variables=['p'])
            print(f"\n=== 測試點提取 ===")
            print(f"測試點: {test_points}")
            print(f"壓力值: {extracted['p']}")
    
    except Exception as e:
        print(f"\n❌ 載入失敗: {e}")
        import traceback
        traceback.print_exc()

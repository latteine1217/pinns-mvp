"""
低保真資料載入器

提供統一介面載入 RANS、粗LES、下採樣DNS 等低保真資料，
並支援插值到 PINN 訓練網格，作為軟先驗使用。

主要功能：
1. 多格式讀取 (NetCDF, HDF5, NPZ)
2. 空間插值與時間對齊
3. 物理量驗證與單位轉換
4. 軟先驗資料預處理
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# 數據處理
from scipy.interpolate import RegularGridInterpolator, RBFInterpolator
from scipy.spatial.distance import cdist

# 檔案格式支援
try:
    import netCDF4 as nc
    HAS_NETCDF = True
except ImportError:
    HAS_NETCDF = False
    logging.warning("NetCDF4 not available, NetCDF files cannot be loaded")

try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False
    logging.warning("h5py not available, HDF5 files cannot be loaded")


@dataclass
class LowFiData:
    """低保真資料容器"""
    coordinates: Dict[str, np.ndarray]  # {'x': x_coords, 'y': y_coords, 't': t_coords}
    fields: Dict[str, np.ndarray]      # {'u': u_field, 'v': v_field, 'p': p_field}
    metadata: Dict[str, Any]           # 原始資料元信息
    
    def get_spatial_dims(self) -> Tuple[int, ...]:
        """獲取空間維度"""
        spatial_keys = [k for k in self.coordinates.keys() if k != 't']
        return tuple(len(self.coordinates[k]) for k in sorted(spatial_keys))
    
    def get_time_range(self) -> Optional[Tuple[float, float]]:
        """獲取時間範圍"""
        if 't' in self.coordinates:
            t = self.coordinates['t']
            return float(t.min()), float(t.max())
        return None
    
    def validate_physics(self) -> Dict[str, bool]:
        """驗證物理合理性"""
        checks = {}
        
        # 速度場範圍檢查
        for vel_comp in ['u', 'v', 'w']:
            if vel_comp in self.fields:
                field = self.fields[vel_comp]
                checks[f'{vel_comp}_finite'] = np.all(np.isfinite(field))
                checks[f'{vel_comp}_reasonable'] = np.abs(field).max() < 100.0  # 合理速度上限
        
        # 壓力場檢查
        if 'p' in self.fields:
            p = self.fields['p']
            checks['pressure_finite'] = np.all(np.isfinite(p))
            
        return checks


class DataReader(ABC):
    """抽象資料讀取器基類"""
    
    @abstractmethod
    def read(self, filepath: Union[str, Path]) -> LowFiData:
        """讀取檔案並返回 LowFiData"""
        pass
    
    @abstractmethod
    def supports_format(self, filepath: Union[str, Path]) -> bool:
        """檢查是否支援此檔案格式"""
        pass


class NetCDFReader(DataReader):
    """NetCDF 檔案讀取器"""
    
    def __init__(self, coord_mapping: Optional[Dict[str, str]] = None,
                 field_mapping: Optional[Dict[str, str]] = None):
        """
        初始化 NetCDF 讀取器
        
        Args:
            coord_mapping: 座標變數名稱映射 {'x': 'X', 'y': 'Y', 't': 'time'}
            field_mapping: 物理場變數名稱映射 {'u': 'velocity_x', 'v': 'velocity_y'}
        """
        if not HAS_NETCDF:
            logging.warning("NetCDF4 not available. NetCDF file support will be disabled.")
            # 創建一個佔位符，但不會被使用
            return
            
        self.coord_mapping = coord_mapping or {
            'x': ['x', 'X', 'lon', 'longitude'],
            'y': ['y', 'Y', 'lat', 'latitude'],
            't': ['t', 'time', 'Time']
        }
        self.field_mapping = field_mapping or {
            'u': ['u', 'U', 'velocity_x', 'vel_x'],
            'v': ['v', 'V', 'velocity_y', 'vel_y'],
            'w': ['w', 'W', 'velocity_z', 'vel_z'],
            'p': ['p', 'P', 'pressure']
        }
    
    def supports_format(self, filepath: Union[str, Path]) -> bool:
        """檢查是否為 NetCDF 檔案"""
        return str(filepath).lower().endswith(('.nc', '.nc4', '.netcdf'))
    
    def read(self, filepath: Union[str, Path]) -> LowFiData:
        """讀取 NetCDF 檔案"""
        with nc.Dataset(filepath, 'r') as ds:
            # 提取座標
            coordinates = {}
            for std_name, possible_names in self.coord_mapping.items():
                coord_var = self._find_variable(ds, possible_names)
                if coord_var:
                    coordinates[std_name] = np.array(ds[coord_var][:])
            
            # 提取物理場
            fields = {}
            for std_name, possible_names in self.field_mapping.items():
                field_var = self._find_variable(ds, possible_names)
                if field_var:
                    fields[std_name] = np.array(ds[field_var][:])
            
            # 提取元數據
            metadata = {
                'source_file': str(filepath),
                'format': 'NetCDF',
                'global_attrs': {attr: ds.getncattr(attr) for attr in ds.ncattrs()},
                'variables': list(ds.variables.keys())
            }
            
        return LowFiData(coordinates=coordinates, fields=fields, metadata=metadata)
    
    def _find_variable(self, dataset, possible_names: List[str]) -> Optional[str]:
        """在資料集中尋找變數"""
        for name in possible_names:
            if name in dataset.variables:
                return name
        return None


class HDF5Reader(DataReader):
    """HDF5 檔案讀取器"""
    
    def __init__(self, group_path: str = '/', 
                 coord_mapping: Optional[Dict[str, str]] = None,
                 field_mapping: Optional[Dict[str, str]] = None):
        if not HAS_HDF5:
            raise ImportError("h5py is required for HDF5 files")
            
        self.group_path = group_path
        self.coord_mapping = coord_mapping or {
            'x': ['coordinates/x', 'x', 'X'],
            'y': ['coordinates/y', 'y', 'Y'],
            't': ['coordinates/t', 'time', 'Time']
        }
        self.field_mapping = field_mapping or {
            'u': ['fields/u', 'u', 'U'],
            'v': ['fields/v', 'v', 'V'],
            'w': ['fields/w', 'w', 'W'],
            'p': ['fields/p', 'p', 'P']
        }
    
    def supports_format(self, filepath: Union[str, Path]) -> bool:
        """檢查是否為 HDF5 檔案"""
        return str(filepath).lower().endswith(('.h5', '.hdf5', '.hdf'))
    
    def read(self, filepath: Union[str, Path]) -> LowFiData:
        """讀取 HDF5 檔案"""
        with h5py.File(filepath, 'r') as f:
            group = f[self.group_path]
            
            # 提取座標
            coordinates = {}
            for std_name, possible_paths in self.coord_mapping.items():
                coord_data = self._find_dataset(group, possible_paths)
                if coord_data is not None:
                    coordinates[std_name] = coord_data
            
            # 提取物理場
            fields = {}
            for std_name, possible_paths in self.field_mapping.items():
                field_data = self._find_dataset(group, possible_paths)
                if field_data is not None:
                    fields[std_name] = field_data
            
            # 提取元數據
            metadata = {
                'source_file': str(filepath),
                'format': 'HDF5',
                'group_path': self.group_path,
                'datasets': self._list_datasets(group)
            }
            
        return LowFiData(coordinates=coordinates, fields=fields, metadata=metadata)
    
    def _find_dataset(self, group, possible_paths: List[str]) -> Optional[np.ndarray]:
        """在 HDF5 群組中尋找資料集"""
        for path in possible_paths:
            try:
                return np.array(group[path])
            except KeyError:
                continue
        return None
    
    def _list_datasets(self, group) -> List[str]:
        """列出群組中的所有資料集"""
        datasets = []
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(name)
        group.visititems(visitor)
        return datasets


class NPZReader(DataReader):
    """NumPy .npz 檔案讀取器"""
    
    def supports_format(self, filepath: Union[str, Path]) -> bool:
        """檢查是否為 NPZ 檔案"""
        return str(filepath).lower().endswith('.npz')
    
    def read(self, filepath: Union[str, Path]) -> LowFiData:
        """讀取 NPZ 檔案"""
        data = np.load(filepath)
        
        # 自動識別座標和場
        coordinates = {}
        fields = {}
        
        for key in data.files:
            arr = data[key]
            if key.lower() in ['x', 'y', 'z', 't', 'time']:
                coordinates[key.lower()] = arr
            elif key.lower() in ['u', 'v', 'w', 'p', 'pressure']:
                fields[key.lower()] = arr
            else:
                # 嘗試從形狀推斷
                if arr.ndim == 1:
                    coordinates[key] = arr
                else:
                    fields[key] = arr
        
        metadata = {
            'source_file': str(filepath),
            'format': 'NPZ',
            'arrays': list(data.files)
        }
        
        return LowFiData(coordinates=coordinates, fields=fields, metadata=metadata)


class DownsampledDNSProcessor:
    """下採樣 DNS 處理器 - 優化版本"""
    
    def __init__(self, downsample_factor: Union[int, Tuple[int, ...]] = 4,
                 filter_type: str = 'box', filter_width: Optional[float] = None,
                 preserve_energy: bool = True, adaptive_filter: bool = False):
        """
        初始化下採樣處理器
        
        Args:
            downsample_factor: 下採樣倍數
            filter_type: 濾波器類型 ('box', 'gaussian', 'spectral', 'none')
            filter_width: 濾波器寬度
            preserve_energy: 是否保持能量守恆
            adaptive_filter: 是否使用自適應濾波
        """
        self.downsample_factor = downsample_factor
        self.filter_type = filter_type
        self.filter_width = filter_width
        self.preserve_energy = preserve_energy
        self.adaptive_filter = adaptive_filter
    
    def process(self, hifi_data: LowFiData) -> LowFiData:
        """將高保真 DNS 資料下採樣為低保真"""
        if isinstance(self.downsample_factor, int):
            factor = (self.downsample_factor,) * len(hifi_data.get_spatial_dims())
        else:
            factor = self.downsample_factor
        
        # 下採樣座標
        new_coords = {}
        for coord_name, coord_vals in hifi_data.coordinates.items():
            if coord_name == 't':
                new_coords[coord_name] = coord_vals  # 時間不下採樣
            else:
                axis_idx = ['x', 'y', 'z'].index(coord_name)
                if axis_idx < len(factor):
                    step = factor[axis_idx]
                    new_coords[coord_name] = coord_vals[::step]
                else:
                    new_coords[coord_name] = coord_vals
        
        # 下採樣場
        new_fields = {}
        original_energy = {}
        
        for field_name, field_data in hifi_data.fields.items():
            # 計算原始能量（如果需要）
            if self.preserve_energy and field_name in ['u', 'v', 'w']:
                original_energy[field_name] = np.mean(field_data**2)
            
            # 先應用濾波器
            filtered_data = self._apply_filter(field_data, factor)
            
            # 下採樣
            slices = []
            for i, dim_size in enumerate(field_data.shape):
                if i < len(factor):
                    step = factor[i]
                    slices.append(slice(None, None, step))
                else:
                    slices.append(slice(None))
            
            downsampled = filtered_data[tuple(slices)]
            
            # 能量校正（如果需要）
            if (self.preserve_energy and field_name in ['u', 'v', 'w'] 
                and field_name in original_energy):
                new_energy = np.mean(downsampled**2)
                if new_energy > 0:
                    correction_factor = np.sqrt(original_energy[field_name] / new_energy)
                    downsampled *= correction_factor
            
            new_fields[field_name] = downsampled
        
        # 更新元數據
        new_metadata = hifi_data.metadata.copy()
        new_metadata.update({
            'downsample_factor': factor,
            'filter_type': self.filter_type,
            'preserve_energy': self.preserve_energy,
            'adaptive_filter': self.adaptive_filter,
            'original_resolution': hifi_data.get_spatial_dims(),
            'downsampled_resolution': tuple(new_fields[list(new_fields.keys())[0]].shape[:len(factor)]),
            'energy_ratio': {k: np.mean(v**2)/original_energy.get(k, 1.0) 
                           for k, v in new_fields.items() 
                           if k in original_energy}
        })
        
        return LowFiData(coordinates=new_coords, fields=new_fields, metadata=new_metadata)
    
    def _apply_filter(self, data: np.ndarray, factor: Tuple[int, ...]) -> np.ndarray:
        """應用濾波器 - 優化版本"""
        if self.filter_type == 'none':
            return data
        elif self.filter_type == 'box':
            return self._box_filter(data, factor)
        elif self.filter_type == 'gaussian':
            return self._gaussian_filter(data, factor)
        elif self.filter_type == 'spectral':
            return self._spectral_filter(data, factor)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
    
    def _box_filter(self, data: np.ndarray, factor: Tuple[int, ...]) -> np.ndarray:
        """優化的箱型濾波器"""
        from scipy import ndimage
        
        # 自適應核大小
        if self.adaptive_filter:
            kernel_sizes = [min(f, data.shape[i]) for i, f in enumerate(factor)]
        else:
            if isinstance(self.downsample_factor, int):
                kernel_sizes = [self.downsample_factor] * data.ndim
            else:
                kernel_sizes = list(factor) + [1] * (data.ndim - len(factor))
        
        # 分離式卷積提高效率
        filtered = data.copy()
        for axis, ksize in enumerate(kernel_sizes[:data.ndim]):
            if ksize > 1:
                kernel = np.ones(ksize) / ksize
                filtered = ndimage.convolve1d(filtered, kernel, axis=axis, mode='reflect')
        
        return filtered
    
    def _gaussian_filter(self, data: np.ndarray, factor: Tuple[int, ...]) -> np.ndarray:
        """優化的高斯濾波器"""
        from scipy import ndimage
        
        # 計算各軸的 sigma
        if self.filter_width:
            sigmas = [self.filter_width] * data.ndim
        else:
            sigmas = [max(factor) / 3.0] * len(factor)
            sigmas += [0.0] * (data.ndim - len(factor))  # 其他維度不濾波
        
        return ndimage.gaussian_filter(data, sigma=sigmas, mode='reflect')
    
    def _spectral_filter(self, data: np.ndarray, factor: Tuple[int, ...]) -> np.ndarray:
        """頻譜域銳截止濾波器"""
        # 使用 FFT 進行頻譜濾波
        fft_data = np.fft.fftn(data)
        
        # 計算截止頻率
        cutoff_freqs = []
        for i, (size, f) in enumerate(zip(data.shape, factor)):
            if i < len(factor):
                cutoff = size // (2 * f)  # Nyquist criterion
                cutoff_freqs.append(cutoff)
            else:
                cutoff_freqs.append(size // 2)
        
        # 創建濾波器
        filter_mask = np.ones_like(fft_data)
        for axis, cutoff in enumerate(cutoff_freqs):
            if cutoff < data.shape[axis] // 2:
                # 創建低通濾波器
                freqs = np.fft.fftfreq(data.shape[axis])
                mask = np.abs(freqs) <= cutoff / data.shape[axis]
                
                # 廣播到對應軸
                shape = [1] * data.ndim
                shape[axis] = len(mask)
                mask = mask.reshape(shape)
                filter_mask *= mask
        
        # 應用濾波器並逆變換
        filtered_fft = fft_data * filter_mask
        return np.real(np.fft.ifftn(filtered_fft))


class SpatialInterpolator:
    """空間插值器，將低保真資料插值到 PINN 網格 - 增強版本"""
    
    def __init__(self, method: str = 'linear', 
                 rbf_function: str = 'thin_plate_spline',
                 extrapolation_mode: str = 'nearest',
                 boundary_treatment: str = 'reflect',
                 quality_threshold: float = 0.01):
        """
        初始化插值器
        
        Args:
            method: 插值方法 ('linear', 'cubic', 'rbf', 'kriging', 'idw')
            rbf_function: RBF 函數類型
            extrapolation_mode: 外插模式 ('nearest', 'linear', 'constant', 'boundary')
            boundary_treatment: 邊界處理 ('reflect', 'wrap', 'constant')
            quality_threshold: 插值品質閾值
        """
        self.method = method
        self.rbf_function = rbf_function
        self.extrapolation_mode = extrapolation_mode
        self.boundary_treatment = boundary_treatment
        self.quality_threshold = quality_threshold
    
    def interpolate_to_points(self, lowfi_data: LowFiData, 
                            target_points: np.ndarray,
                            quality_check: bool = True) -> Dict[str, np.ndarray]:
        """
        將低保真資料插值到目標點
        
        Args:
            lowfi_data: 低保真資料
            target_points: 目標點座標 [N, ndim]
            quality_check: 是否進行插值品質檢查
            
        Returns:
            插值後的場數據和品質指標
        """
        # 選擇插值方法
        if self.method in ['linear', 'cubic']:
            result = self._regular_grid_interpolation(lowfi_data, target_points)
        elif self.method == 'rbf':
            result = self._rbf_interpolation(lowfi_data, target_points)
        elif self.method == 'idw':
            result = self._idw_interpolation(lowfi_data, target_points)
        elif self.method == 'kriging':
            result = self._kriging_interpolation(lowfi_data, target_points)
        else:
            raise ValueError(f"Unknown interpolation method: {self.method}")
        
        # 品質檢查
        if quality_check:
            quality_metrics = self._assess_interpolation_quality(
                lowfi_data, target_points, result
            )
            result['_quality_metrics'] = quality_metrics
        
        return result
    
    def interpolate_to_grid(self, lowfi_data: LowFiData,
                           target_grid: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """插值到規則網格"""
        # 構建目標點陣列
        coords = [target_grid[k] for k in ['x', 'y'] if k in target_grid]
        if len(coords) == 2:
            xx, yy = np.meshgrid(coords[0], coords[1], indexing='ij')
            target_points = np.column_stack([xx.ravel(), yy.ravel()])
        else:
            raise NotImplementedError("Only 2D grids supported currently")
        
        # 插值
        result = self.interpolate_to_points(lowfi_data, target_points)
        
        # 重新整形為網格
        grid_shape = (len(coords[0]), len(coords[1]))
        for field_name in result:
            if field_name != '_quality_metrics':
                result[field_name] = result[field_name].reshape(grid_shape)
        
        return result
    
    def _regular_grid_interpolation(self, lowfi_data: LowFiData, 
                                  target_points: np.ndarray) -> Dict[str, np.ndarray]:
        """規則網格插值 - 增強版本"""
        # 構建座標網格
        spatial_coords = [lowfi_data.coordinates[k] for k in ['x', 'y'] 
                         if k in lowfi_data.coordinates]
        
        if len(spatial_coords) != target_points.shape[1]:
            raise ValueError("Coordinate dimensions mismatch")
        
        interpolated_fields = {}
        
        for field_name, field_data in lowfi_data.fields.items():
            # 處理 NaN 值
            if np.any(np.isnan(field_data)):
                field_data = self._handle_nans(field_data)
            
            # 建立插值器
            try:
                interp = RegularGridInterpolator(
                    spatial_coords, field_data,
                    method=self.method,
                    bounds_error=False,
                    fill_value=None
                )
                
                # 插值
                interpolated = interp(target_points)
                
                # 處理外插值
                interpolated = self._handle_extrapolation(
                    interpolated, lowfi_data, target_points, field_name
                )
                
                interpolated_fields[field_name] = interpolated
                
            except Exception as e:
                logging.warning(f"Interpolation failed for field {field_name}: {e}")
                interpolated_fields[field_name] = np.full(len(target_points), np.nan)
        
        return interpolated_fields
    
    def _idw_interpolation(self, lowfi_data: LowFiData, 
                          target_points: np.ndarray,
                          power: float = 2.0, max_distance: Optional[float] = None) -> Dict[str, np.ndarray]:
        """反距離權重插值"""
        # 構建源點網格
        coords = [lowfi_data.coordinates[k] for k in ['x', 'y'] 
                 if k in lowfi_data.coordinates]
        
        if len(coords) == 2:
            xx, yy = np.meshgrid(coords[0], coords[1], indexing='ij')
            source_points = np.column_stack([xx.ravel(), yy.ravel()])
        else:
            raise NotImplementedError("IDW interpolation only supports 2D currently")
        
        interpolated_fields = {}
        
        for field_name, field_data in lowfi_data.fields.items():
            field_values = field_data.ravel()
            
            # 移除無效值
            valid_mask = np.isfinite(field_values)
            if not np.any(valid_mask):
                interpolated_fields[field_name] = np.full(len(target_points), np.nan)
                continue
            
            valid_points = source_points[valid_mask]
            valid_values = field_values[valid_mask]
            
            # 計算距離
            distances = cdist(target_points, valid_points)
            
            # 處理距離為零的情況
            zero_dist_mask = distances < 1e-12
            if np.any(zero_dist_mask):
                result = np.zeros(len(target_points))
                for i, target_idx in enumerate(np.where(np.any(zero_dist_mask, axis=1))[0]):
                    source_idx = np.where(zero_dist_mask[target_idx])[0][0]
                    result[target_idx] = valid_values[source_idx]
            else:
                result = np.zeros(len(target_points))
            
            # IDW 插值
            non_zero_mask = ~np.any(zero_dist_mask, axis=1)
            if np.any(non_zero_mask):
                weights = 1.0 / (distances[non_zero_mask] ** power)
                
                # 限制最大距離
                if max_distance:
                    far_mask = distances[non_zero_mask] > max_distance
                    weights[far_mask] = 0.0
                
                weight_sum = np.sum(weights, axis=1)
                valid_weight_mask = weight_sum > 0
                
                if np.any(valid_weight_mask):
                    weighted_values = np.sum(weights[valid_weight_mask] * valid_values, axis=1)
                    result[non_zero_mask][valid_weight_mask] = (
                        weighted_values / weight_sum[valid_weight_mask]
                    )
            
            interpolated_fields[field_name] = result
        
        return interpolated_fields
    
    def _rbf_interpolation(self, lowfi_data: LowFiData, 
                          target_points: np.ndarray,
                          epsilon: Optional[float] = None) -> Dict[str, np.ndarray]:
        """徑向基函數插值"""
        try:
            from scipy.interpolate import RBFInterpolator
        except ImportError:
            logging.warning("RBF interpolation requires SciPy >= 1.7. Falling back to IDW.")
            return self._idw_interpolation(lowfi_data, target_points)
        
        # 構建源點網格
        coords = [lowfi_data.coordinates[k] for k in ['x', 'y'] 
                 if k in lowfi_data.coordinates]
        
        if len(coords) == 2:
            xx, yy = np.meshgrid(coords[0], coords[1], indexing='ij')
            source_points = np.column_stack([xx.ravel(), yy.ravel()])
        else:
            raise NotImplementedError("RBF interpolation only supports 2D currently")
        
        interpolated_fields = {}
        
        for field_name, field_data in lowfi_data.fields.items():
            field_values = field_data.ravel()
            
            # 移除無效值
            valid_mask = np.isfinite(field_values)
            if not np.any(valid_mask):
                interpolated_fields[field_name] = np.full(len(target_points), np.nan)
                continue
            
            valid_points = source_points[valid_mask]
            valid_values = field_values[valid_mask]
            
            try:
                # 選擇 RBF 核函數
                if self.rbf_function == 'thin_plate_spline':
                    kernel = 'thin_plate_spline'
                elif self.rbf_function == 'multiquadric':
                    kernel = 'multiquadric'  
                elif self.rbf_function == 'inverse_multiquadric':
                    kernel = 'inverse_multiquadric'
                elif self.rbf_function == 'linear':
                    kernel = 'linear'
                elif self.rbf_function == 'cubic':
                    kernel = 'cubic'
                elif self.rbf_function == 'quintic':
                    kernel = 'quintic'
                else:
                    kernel = 'thin_plate_spline'  # 預設值
                
                # 設定epsilon參數（形狀參數）
                if epsilon is None:
                    # 自動選擇epsilon基於點之間的平均距離
                    if len(valid_points) > 1:
                        distances = cdist(valid_points, valid_points)
                        np.fill_diagonal(distances, np.inf)  # 忽略自身距離
                        epsilon = np.mean(np.min(distances, axis=1))
                    else:
                        epsilon = 1.0
                
                # 建立RBF插值器
                rbf = RBFInterpolator(
                    valid_points, 
                    valid_values,
                    kernel=kernel,
                    epsilon=epsilon
                )
                
                # 進行插值
                interpolated_values = rbf(target_points)
                interpolated_fields[field_name] = interpolated_values
                
            except Exception as e:
                logging.warning(f"RBF interpolation failed for field {field_name}: {e}")
                # 回退到IDW插值
                idw_result = self._idw_interpolation(lowfi_data, target_points)
                interpolated_fields[field_name] = idw_result[field_name]
        
        return interpolated_fields
    
    def _kriging_interpolation(self, lowfi_data: LowFiData, 
                              target_points: np.ndarray,
                              variogram_model: str = 'spherical') -> Dict[str, np.ndarray]:
        """簡化的克立金插值 (需要 scikit-gstat 或類似庫)"""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        except ImportError:
            logging.warning("Kriging requires scikit-learn. Falling back to RBF.")
            return self._rbf_interpolation(lowfi_data, target_points)
        
        # 構建源點網格
        coords = [lowfi_data.coordinates[k] for k in ['x', 'y'] 
                 if k in lowfi_data.coordinates]
        
        if len(coords) == 2:
            xx, yy = np.meshgrid(coords[0], coords[1], indexing='ij')
            source_points = np.column_stack([xx.ravel(), yy.ravel()])
        else:
            raise NotImplementedError("Kriging only supports 2D currently")
        
        interpolated_fields = {}
        
        for field_name, field_data in lowfi_data.fields.items():
            field_values = field_data.ravel()
            
            # 移除無效值
            valid_mask = np.isfinite(field_values)
            if not np.any(valid_mask) or np.sum(valid_mask) < 10:
                interpolated_fields[field_name] = np.full(len(target_points), np.nan)
                continue
            
            valid_points = source_points[valid_mask]
            valid_values = field_values[valid_mask]
            
            # 高斯過程回歸
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)
            
            try:
                gp.fit(valid_points, valid_values)
                mean, std = gp.predict(target_points, return_std=True)
                interpolated_fields[field_name] = mean
                interpolated_fields[f'{field_name}_uncertainty'] = std
            except Exception as e:
                logging.warning(f"Kriging failed for field {field_name}: {e}")
                interpolated_fields[field_name] = np.full(len(target_points), np.nan)
        
        return interpolated_fields
    
    def _handle_nans(self, field_data: np.ndarray) -> np.ndarray:
        """處理 NaN 值"""
        if not np.any(np.isnan(field_data)):
            return field_data
        
        from scipy import ndimage
        
        # 簡單的 NaN 填補
        mask = np.isnan(field_data)
        field_data = field_data.copy()
        
        if np.all(mask):
            return np.zeros_like(field_data)
        
        # 用最近鄰填補
        indices = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
        field_data[mask] = field_data[tuple(indices[:, mask])]
        
        return field_data
    
    def _handle_extrapolation(self, interpolated: np.ndarray, lowfi_data: LowFiData,
                             target_points: np.ndarray, field_name: str) -> np.ndarray:
        """處理外插值 - 增強版本"""
        nan_mask = np.isnan(interpolated)
        if not np.any(nan_mask):
            return interpolated
        
        if self.extrapolation_mode == 'nearest':
            interpolated = self._nearest_extrapolation(
                lowfi_data, target_points, field_name, nan_mask
            )
        elif self.extrapolation_mode == 'boundary':
            interpolated = self._boundary_extrapolation(
                lowfi_data, target_points, field_name, nan_mask
            )
        elif self.extrapolation_mode == 'constant':
            interpolated[nan_mask] = 0.0
        elif self.extrapolation_mode == 'linear':
            # 線性外插 (簡化實現)
            if np.any(~nan_mask):
                mean_val = np.mean(interpolated[~nan_mask])
                interpolated[nan_mask] = mean_val
        
        return interpolated
    
    def _boundary_extrapolation(self, lowfi_data: LowFiData, target_points: np.ndarray,
                               field_name: str, nan_mask: np.ndarray) -> np.ndarray:
        """邊界值外插"""
        coords = [lowfi_data.coordinates[k] for k in ['x', 'y'] 
                 if k in lowfi_data.coordinates]
        field_data = lowfi_data.fields[field_name]
        
        result = np.full(len(target_points), np.nan)
        
        # 找到域邊界上的值
        if len(coords) == 2:
            x_coords, y_coords = coords
            x_min, x_max = x_coords[0], x_coords[-1]
            y_min, y_max = y_coords[0], y_coords[-1]
            
            for i, (x, y) in enumerate(target_points):
                if nan_mask[i]:
                    # 找最近的邊界點
                    if x < x_min:
                        xi = 0
                    elif x > x_max:
                        xi = len(x_coords) - 1
                    else:
                        xi = np.argmin(np.abs(x_coords - x))
                    
                    if y < y_min:
                        yi = 0
                    elif y > y_max:
                        yi = len(y_coords) - 1
                    else:
                        yi = np.argmin(np.abs(y_coords - y))
                    
                    result[i] = field_data[xi, yi]
        
        # 填補仍然是 NaN 的值
        remaining_nan = np.isnan(result)
        if np.any(remaining_nan):
            result[remaining_nan] = np.nanmean(field_data)
        
        return result
    
    def _assess_interpolation_quality(self, lowfi_data: LowFiData, 
                                    target_points: np.ndarray,
                                    interpolated: Dict[str, np.ndarray]) -> Dict[str, float]:
        """評估插值品質"""
        quality_metrics = {}
        
        # 計算覆蓋率
        total_points = len(target_points)
        for field_name, values in interpolated.items():
            if field_name.startswith('_'):
                continue
            
            finite_points = np.sum(np.isfinite(values))
            coverage = finite_points / total_points
            quality_metrics[f'{field_name}_coverage'] = coverage
            
            # 檢查物理合理性
            if np.any(np.isfinite(values)):
                finite_values = values[np.isfinite(values)]
                quality_metrics[f'{field_name}_range'] = np.ptp(finite_values)
                quality_metrics[f'{field_name}_std'] = np.std(finite_values)
        
        # 整體品質分數
        avg_coverage = np.mean([v for k, v in quality_metrics.items() if k.endswith('_coverage')])
        quality_metrics['overall_quality'] = avg_coverage
        
        return quality_metrics


class RANSReader(DataReader):
    """RANS 特定讀取器，處理雷諾平均場和湍流統計量"""
    
    def __init__(self, base_reader: DataReader):
        """
        基於已有讀取器創建 RANS 讀取器
        
        Args:
            base_reader: 基礎讀取器 (NetCDF, HDF5, NPZ)
        """
        self.base_reader = base_reader
        self.rans_field_mapping = {
            # 平均場
            'u_mean': ['u_mean', 'U_mean', 'umean', 'vel_x_mean'],
            'v_mean': ['v_mean', 'V_mean', 'vmean', 'vel_y_mean'],
            'p_mean': ['p_mean', 'P_mean', 'pmean', 'pressure_mean'],
            # 雷諾應力分量
            'uu': ['uu', 'R11', 'reynolds_stress_11', 'u_u'],
            'uv': ['uv', 'R12', 'reynolds_stress_12', 'u_v'],
            'vv': ['vv', 'R22', 'reynolds_stress_22', 'v_v'],
            # 湍流動能
            'k': ['k', 'tke', 'turbulent_kinetic_energy'],
            # 渦流黏度
            'nut': ['nut', 'nu_t', 'turbulent_viscosity', 'eddy_viscosity']
        }
    
    def supports_format(self, filepath: Union[str, Path]) -> bool:
        """委託給基礎讀取器"""
        return self.base_reader.supports_format(filepath)
    
    def read(self, filepath: Union[str, Path]) -> LowFiData:
        """讀取 RANS 資料並添加特定處理"""
        # 先用基礎讀取器讀取
        data = self.base_reader.read(filepath)
        
        # 重新映射 RANS 特定場
        rans_fields = {}
        for std_name, possible_names in self.rans_field_mapping.items():
            for name in possible_names:
                if name in data.fields:
                    rans_fields[std_name] = data.fields[name]
                    break
        
        # 合併原始場和 RANS 場
        combined_fields = {**data.fields, **rans_fields}
        
        # 計算缺失的 RANS 量
        combined_fields = self._compute_derived_quantities(combined_fields)
        
        # 添加 RANS 特定元數據
        rans_metadata = data.metadata.copy()
        rans_metadata.update({
            'data_type': 'RANS',
            'has_reynolds_stress': any(k in rans_fields for k in ['uu', 'uv', 'vv']),
            'has_turbulent_viscosity': 'nut' in rans_fields,
            'available_rans_fields': list(rans_fields.keys())
        })
        
        return LowFiData(
            coordinates=data.coordinates,
            fields=combined_fields,
            metadata=rans_metadata
        )
    
    def _compute_derived_quantities(self, fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """計算衍生的 RANS 量"""
        # 如果有雷諾應力分量但沒有湍流動能，計算 k
        if all(k in fields for k in ['uu', 'vv']) and 'k' not in fields:
            if 'ww' in fields:  # 3D 情況
                fields['k'] = 0.5 * (fields['uu'] + fields['vv'] + fields['ww'])
            else:  # 2D 情況
                fields['k'] = 0.5 * (fields['uu'] + fields['vv'])
        
        # 如果沒有平均場但有總場，估算（簡化）
        if 'u_mean' not in fields and 'u' in fields:
            fields['u_mean'] = fields['u']  # 假設已經是時間平均
        if 'v_mean' not in fields and 'v' in fields:
            fields['v_mean'] = fields['v']
        if 'p_mean' not in fields and 'p' in fields:
            fields['p_mean'] = fields['p']
        
        return fields


class LESReader(DataReader):
    """LES 特定讀取器，處理過濾場和亞格度模型"""
    
    def __init__(self, base_reader: DataReader, 
                 filter_width: Optional[float] = None,
                 sgs_model: str = 'smagorinsky'):
        """
        基於已有讀取器創建 LES 讀取器
        
        Args:
            base_reader: 基礎讀取器
            filter_width: LES 濾波寬度
            sgs_model: 亞格度模型類型
        """
        self.base_reader = base_reader
        self.filter_width = filter_width
        self.sgs_model = sgs_model
        self.les_field_mapping = {
            # 過濾場
            'u_filtered': ['u_filt', 'u_filtered', 'u_f', 'vel_x_filtered'],
            'v_filtered': ['v_filt', 'v_filtered', 'v_f', 'vel_y_filtered'],
            'p_filtered': ['p_filt', 'p_filtered', 'p_f', 'pressure_filtered'],
            # SGS 應力
            'tau11': ['tau11', 'sgs_stress_11', 'tau_xx'],
            'tau12': ['tau12', 'sgs_stress_12', 'tau_xy'],
            'tau22': ['tau22', 'sgs_stress_22', 'tau_yy'],
            # SGS 動能和黏度
            'k_sgs': ['k_sgs', 'sgs_ke', 'subgrid_ke'],
            'nu_sgs': ['nu_sgs', 'nut_sgs', 'sgs_viscosity']
        }
    
    def supports_format(self, filepath: Union[str, Path]) -> bool:
        """委託給基礎讀取器"""
        return self.base_reader.supports_format(filepath)
    
    def read(self, filepath: Union[str, Path]) -> LowFiData:
        """讀取 LES 資料並添加特定處理"""
        # 先用基礎讀取器讀取
        data = self.base_reader.read(filepath)
        
        # 重新映射 LES 特定場
        les_fields = {}
        for std_name, possible_names in self.les_field_mapping.items():
            for name in possible_names:
                if name in data.fields:
                    les_fields[std_name] = data.fields[name]
                    break
        
        # 合併原始場和 LES 場
        combined_fields = {**data.fields, **les_fields}
        
        # 計算缺失的 SGS 模型量
        combined_fields = self._compute_sgs_quantities(combined_fields, data.coordinates)
        
        # 添加 LES 特定元數據
        les_metadata = data.metadata.copy()
        les_metadata.update({
            'data_type': 'LES',
            'filter_width': self.filter_width,
            'sgs_model': self.sgs_model,
            'has_sgs_stress': any(k in les_fields for k in ['tau11', 'tau12', 'tau22']),
            'available_les_fields': list(les_fields.keys())
        })
        
        return LowFiData(
            coordinates=data.coordinates,
            fields=combined_fields,
            metadata=les_metadata
        )
    
    def _compute_sgs_quantities(self, fields: Dict[str, np.ndarray], 
                               coordinates: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """計算 SGS 模型量"""
        # 如果有速度場但沒有 SGS 黏度，用 Smagorinsky 模型估算
        if (all(k in fields for k in ['u', 'v']) and 'nu_sgs' not in fields 
            and self.sgs_model == 'smagorinsky'):
            
            # 簡化的 Smagorinsky 模型實現
            if 'x' in coordinates and 'y' in coordinates:
                dx = coordinates['x'][1] - coordinates['x'][0] if len(coordinates['x']) > 1 else 1.0
                dy = coordinates['y'][1] - coordinates['y'][0] if len(coordinates['y']) > 1 else 1.0
                filter_width = self.filter_width or (dx * dy) ** 0.5
                
                # 計算應變率張量模 (簡化為2D)
                u, v = fields['u'], fields['v']
                # 這裡需要實際的梯度計算，暫時用簡化版本
                S_mag = np.sqrt(2.0) * np.sqrt(
                    np.gradient(u, dx, axis=0)**2 + 
                    np.gradient(v, dy, axis=1)**2
                )
                
                # Smagorinsky 常數
                Cs = 0.1
                fields['nu_sgs'] = (Cs * filter_width)**2 * S_mag
        
        return fields


class LowFiLoader:
    """低保真資料載入器主類"""
    
    def __init__(self):
        """初始化載入器"""
        self.base_readers = []
        
        # 只有在NetCDF4可用時才添加NetCDFReader
        if HAS_NETCDF:
            self.base_readers.append(NetCDFReader())
        
        # 總是添加HDF5和NPZ讀取器
        self.base_readers.extend([
            HDF5Reader(),
            NPZReader()
        ])
        self.interpolator = SpatialInterpolator()
        self.dns_processor = DownsampledDNSProcessor()
    
    def load(self, filepath: Union[str, Path], 
             data_type: str = 'auto') -> LowFiData:
        """
        載入低保真資料
        
        Args:
            filepath: 檔案路徑
            data_type: 資料類型 ('auto', 'rans', 'les', 'dns_downsampled')
            
        Returns:
            載入的低保真資料
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # 根據資料類型選擇讀取器
        if data_type == 'rans':
            reader = self._get_rans_reader(filepath)
        elif data_type == 'les':
            reader = self._get_les_reader(filepath)
        else:
            # 自動選擇基礎讀取器
            reader = self._get_base_reader(filepath)
        
        if reader is None:
            raise ValueError(f"No suitable reader found for file: {filepath}")
        
        # 讀取資料
        data = reader.read(filepath)
        
        # 驗證物理合理性
        physics_checks = data.validate_physics()
        failed_checks = [k for k, v in physics_checks.items() if not v]
        if failed_checks:
            logging.warning(f"Physics validation failed for: {failed_checks}")
        
        # 添加資料類型到元數據
        if data_type == 'auto':
            data.metadata['data_type'] = self._infer_data_type(data)
        else:
            data.metadata['data_type'] = data_type
        
        return data
    
    def _get_base_reader(self, filepath: Path) -> Optional[DataReader]:
        """獲取基礎讀取器"""
        for reader in self.base_readers:
            if reader.supports_format(filepath):
                return reader
        return None
    
    def _get_rans_reader(self, filepath: Path) -> Optional[DataReader]:
        """獲取 RANS 讀取器"""
        base_reader = self._get_base_reader(filepath)
        if base_reader:
            return RANSReader(base_reader)
        return None
    
    def _get_les_reader(self, filepath: Path) -> Optional[DataReader]:
        """獲取 LES 讀取器"""
        base_reader = self._get_base_reader(filepath)
        if base_reader:
            return LESReader(base_reader)
        return None
    
    def _infer_data_type(self, data: LowFiData) -> str:
        """推斷資料類型"""
        fields = set(data.fields.keys())
        
        # 檢查 RANS 特徵
        rans_indicators = {'u_mean', 'v_mean', 'uu', 'uv', 'vv', 'k', 'nut'}
        if len(fields & rans_indicators) >= 2:
            return 'rans'
        
        # 檢查 LES 特徵
        les_indicators = {'u_filtered', 'v_filtered', 'tau11', 'tau12', 'k_sgs', 'nu_sgs'}
        if len(fields & les_indicators) >= 2:
            return 'les'
        
        # 檢查是否為下採樣 DNS
        if 'downsampled_resolution' in data.metadata:
            return 'dns_downsampled'
        
        return 'unknown'
    
    def create_prior_data(self, lowfi_data: LowFiData, 
                         target_points: np.ndarray,
                         fields: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        創建軟先驗資料（插值到 PINN 訓練點）
        
        Args:
            lowfi_data: 低保真資料
            target_points: PINN 訓練點座標
            fields: 需要的場列表，None 表示全部
            
        Returns:
            插值後的先驗場資料
        """
        if fields is None:
            fields = list(lowfi_data.fields.keys())
        
        # 插值到目標點
        interpolated = self.interpolator.interpolate_to_points(lowfi_data, target_points)
        
        # 只返回需要的場
        return {k: v for k, v in interpolated.items() if k in fields}
    
    def downsample_dns(self, hifi_data: LowFiData, 
                      factor: Union[int, Tuple[int, ...]] = 4,
                      filter_type: str = 'box') -> LowFiData:
        """
        將高保真 DNS 下採樣為低保真
        
        Args:
            hifi_data: 高保真 DNS 資料
            factor: 下採樣倍數
            filter_type: 濾波器類型
            
        Returns:
            下採樣後的低保真資料
        """
        processor = DownsampledDNSProcessor(factor, filter_type)
        return processor.process(hifi_data)
    
    def get_statistics(self, lowfi_data: LowFiData) -> Dict[str, Dict[str, float]]:
        """
        計算低保真資料統計量（用於 VS-PINN 尺度化）
        
        Args:
            lowfi_data: 低保真資料
            
        Returns:
            統計量字典
        """
        stats = {}
        
        # 座標統計
        for coord_name, coord_vals in lowfi_data.coordinates.items():
            stats[f'coord_{coord_name}'] = {
                'mean': float(np.mean(coord_vals)),
                'std': float(np.std(coord_vals)),
                'min': float(np.min(coord_vals)),
                'max': float(np.max(coord_vals))
            }
        
        # 場統計
        for field_name, field_data in lowfi_data.fields.items():
            valid_mask = np.isfinite(field_data)
            if np.any(valid_mask):
                valid_data = field_data[valid_mask]
                stats[f'field_{field_name}'] = {
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data))
                }
            else:
                stats[f'field_{field_name}'] = {
                    'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 0.0
                }
        
        return stats


# 便利函數
def load_lowfi_data(filepath: Union[str, Path], 
                   data_type: str = 'auto') -> LowFiData:
    """便利函數：載入低保真資料"""
    loader = LowFiLoader()
    return loader.load(filepath, data_type)


def create_mock_rans_data(nx: int = 64, ny: int = 64, 
                         case: str = 'channel') -> LowFiData:
    """創建模擬 RANS 資料用於測試"""
    x = np.linspace(0, 4*np.pi, nx)
    y = np.linspace(-1, 1, ny)
    
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    if case == 'channel':
        # 通道流 RANS 近似
        u_mean = 1 - yy**2  # 拋物線分佈
        v_mean = np.zeros_like(u_mean)
        p_mean = -2 * xx  # 線性壓力梯度
    else:
        # 簡單剪切流
        u_mean = yy
        v_mean = 0.1 * np.sin(xx) * np.cos(yy)
        p_mean = np.sin(xx) * np.sin(yy)
    
    coordinates = {'x': x, 'y': y}
    fields = {'u': u_mean, 'v': v_mean, 'p': p_mean}
    metadata = {
        'case': case,
        'type': 'mock_rans',
        'nx': nx, 'ny': ny
    }
    
    return LowFiData(coordinates=coordinates, fields=fields, metadata=metadata)

"""
低保真資料載入器 (Simplified)

提供統一介面載入 RANS、粗LES、下採樣DNS 等低保真資料,
並支援插值到 PINN 訓練網格,作為軟先驗使用。

=== 支援的功能 (SUPPORTED) ===
✅ HDF5Reader: 讀取 .h5/.hdf5 檔案
✅ RANSReader: 3D Channel Flow RANS k-ε 模型專用
✅ SpatialInterpolator: 空間插值到 PINN 網格

=== 已棄用 (DEPRECATED - Retained for backward compatibility) ===
⚠️ NetCDFReader: 專案僅支援 HDF5 格式
⚠️ LESReader: LES 模型不在專案範圍內
⚠️ DownsampledDNSProcessor: DNS 下採樣不使用

注意: 實際訓練管道使用 scripts/train/train.py::load_rans_prior_data()
      直接用 h5py 載入,而非使用此模組的 LowFiLoader 類別。

參考文檔: docs/LOWFI_PRIOR_GUIDE.md
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
    """
    抽象資料讀取器基類
    
    提供共用的資料查找、驗證與錯誤處理邏輯，
    子類只需實現格式特定的讀取邏輯。
    """
    
    @abstractmethod
    def read(self, filepath: Union[str, Path]) -> LowFiData:
        """讀取檔案並返回 LowFiData"""
        pass
    
    @abstractmethod
    def supports_format(self, filepath: Union[str, Path]) -> bool:
        """檢查是否支援此檔案格式"""
        pass
    
    # === 共用工具方法 (Phase 5-1 新增) ===
    
    @staticmethod
    def find_by_names(container, possible_names: List[str], 
                      accessor: callable) -> Optional[Any]:
        """
        統一的名稱查找邏輯
        
        Args:
            container: 資料容器 (Dataset, HDF5 Group, dict 等)
            possible_names: 可能的變數名列表
            accessor: 存取函數，接收 (container, name) 返回資料
        
        Returns:
            找到的資料，或 None
        
        Example:
            >>> # NetCDF
            >>> DataReader.find_by_names(
            ...     ds, ['u', 'U', 'velocity_x'],
            ...     lambda c, n: c.variables[n][:]
            ... )
            >>> # HDF5
            >>> DataReader.find_by_names(
            ...     group, ['fields/u', 'u'],
            ...     lambda c, n: np.array(c[n])
            ... )
        """
        for name in possible_names:
            try:
                result = accessor(container, name)
                if result is not None:
                    return result
            except (KeyError, AttributeError, ValueError):
                continue
        return None
    
    def validate_data(self, data: np.ndarray, field_name: str,
                      max_value: float = 1e6) -> bool:
        """
        統一的資料驗證邏輯
        
        Args:
            data: 待驗證的資料陣列
            field_name: 欄位名稱 (用於日誌)
            max_value: 合理值上限
        
        Returns:
            True 如果資料有效
        """
        if not np.all(np.isfinite(data)):
            logging.warning(f"{field_name}: contains NaN/Inf values")
            return False
        
        if np.abs(data).max() > max_value:
            logging.warning(
                f"{field_name}: max value {np.abs(data).max():.2e} "
                f"exceeds threshold {max_value:.2e}"
            )
            return False
        
        return True
    
    def build_metadata(self, filepath: Union[str, Path], 
                       format_name: str, **extra) -> Dict[str, Any]:
        """
        構建標準元數據結構
        
        Args:
            filepath: 來源檔案路徑
            format_name: 格式名稱 (NetCDF, HDF5, NPZ)
            **extra: 額外的元數據欄位
        
        Returns:
            標準化的元數據字典
        """
        metadata = {
            'source_file': str(filepath),
            'format': format_name,
            'read_timestamp': pd.Timestamp.now().isoformat(),
        }
        metadata.update(extra)
        return metadata


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
            
            # 提取座標 (使用基類的 find_by_names)
            coordinates = {}
            for std_name, possible_paths in self.coord_mapping.items():
                coord_data = self.find_by_names(
                    group,
                    possible_paths,
                    lambda g, path: np.array(g[path])
                )
                if coord_data is not None:
                    coordinates[std_name] = coord_data
            
            # 提取物理場 (使用基類的 find_by_names)
            fields = {}
            for std_name, possible_paths in self.field_mapping.items():
                field_data = self.find_by_names(
                    group,
                    possible_paths,
                    lambda g, path: np.array(g[path])
                )
                if field_data is not None:
                    fields[std_name] = field_data
            
            # 提取元數據 (使用基類的 build_metadata)
            metadata = self.build_metadata(
                filepath, 'HDF5',
                group_path=self.group_path,
                datasets=self._list_datasets(group)
            )
            
        return LowFiData(coordinates=coordinates, fields=fields, metadata=metadata)
    
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
        
        # ✅ Metadata keys (should NOT be treated as coordinates)
        metadata_keys = {'shape', 'domain_size', 'domain_lengths', 'grid_shape',
                        'source_cas', 'source_dat', 'model_type', 
                        'Re_tau_estimate', 'nu', 'rho', 'timestamp'}
        
        for key in data.files:
            arr = data[key]
            # ✅ Skip metadata fields
            if key in metadata_keys:
                continue
            # Known coordinate axes
            if key.lower() in ['x', 'y', 'z', 't', 'time']:
                coordinates[key.lower()] = arr
            # Known flow fields
            elif key.lower() in ['u', 'v', 'w', 'p', 'pressure', 'k', 'mu_t', 'nu_t']:
                fields[key.lower()] = arr
            # Infer from shape (only for unknowns)
            else:
                # 嘗試從形狀推斷
                if arr.ndim == 1 and len(arr) > 10:
                    # Large 1D array → likely a coordinate axis
                    coordinates[key] = arr
                else:
                    # Multi-dim or small 1D array → field data
                    fields[key] = arr
        
        # 使用基類的 build_metadata
        metadata = self.build_metadata(
            filepath, 'NPZ',
            arrays=list(data.files)
        )
        
        return LowFiData(coordinates=coordinates, fields=fields, metadata=metadata)



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
        # 構建座標網格（支援 2D/3D）
        axis_priority = ['x', 'y', 'z']
        spatial_axes = [axis for axis in axis_priority if axis in lowfi_data.coordinates]
        # 附加其他空間軸（排除時間維度）
        spatial_axes.extend([
            axis for axis in lowfi_data.coordinates.keys()
            if axis not in spatial_axes and axis != 't'
        ])
        
        if not spatial_axes:
            raise ValueError("Low-fidelity data does not contain spatial coordinates suitable for interpolation")
        
        spatial_coords = [np.asarray(lowfi_data.coordinates[axis]) for axis in spatial_axes]
        
        if len(spatial_coords) != target_points.shape[1]:
            raise ValueError(
                f"Coordinate dimensions mismatch: low-fi axes {spatial_axes} "
                f"expect {len(spatial_coords)}-D points but received shape {target_points.shape}"
            )
        
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


class LowFiLoader:
    """低保真資料載入器主類"""
    
    def __init__(self):
        """初始化載入器"""
        self.base_readers = []
        
        # 條件式添加 HDF5 讀取器
        if HAS_HDF5:
            self.base_readers.append(HDF5Reader())
        else:
            logging.warning("h5py not available. HDF5 file support will be disabled.")
        
        # NPZ 讀取器總是可用
        self.base_readers.append(NPZReader())
        self.interpolator = SpatialInterpolator()
    
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
        interpolated = self.interpolator.interpolate_to_points(
            lowfi_data,
            target_points,
            quality_check=False
        )
        
        # 只返回需要的場
        return {
            k: v for k, v in interpolated.items()
            if k in fields and not k.startswith('_')
        }
    
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

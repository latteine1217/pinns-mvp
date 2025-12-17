"""
QR-pivot 感測點選擇算法

實現基於 QR 分解的最適感測點選擇，這是稀疏感測與重建問題的經典方法。
特別適用於 PINNs 逆問題中的少量觀測點最適化配置。

核心算法：
1. QR-pivot: 基於 QR 分解選主元的貪心最適化
2. POD-based: 結合 POD 模態的感測點配置
3. Greedy: 貪心最適化策略
4. Multi-objective: 多目標最適化 (精度 vs. 穩健性 vs. K)

參考文獻：
- Sensor Selection via Convex Optimization (IEEE 2009)
- Sparsity-promoting optimal control for a class of distributed systems (SIAM 2012)
- Sparse sensor placement optimization for classification (SIAM 2016)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from scipy.linalg import qr, svd
from scipy.optimize import differential_evolution, minimize
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================================
# 週期邊界處理與特徵工程工具
# ============================================================================

class PeriodicBoundaryHandler:
    """
    週期邊界處理工具

    解決JHTDB通道流等週期邊界條件下的「入口聚集」問題。
    核心策略：循環平移 + 週期距離計算

    適用場景：
    - 通道流（x, z方向週期）
    - 均勻各向同性湍流（x, y, z全週期）
    """

    def __init__(self, periodic_axes: List[int] = [0, 2]):
        """
        Args:
            periodic_axes: 週期軸索引列表（0=x, 1=y, 2=z）
                         預設 [0, 2] 對應通道流的 x, z 方向
        """
        self.periodic_axes = periodic_axes

    def circular_shift_augmentation(self,
                                   data_matrix: np.ndarray,
                                   coords: np.ndarray,
                                   n_shifts: int = 5,
                                   random_seed: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        對資料在週期方向做循環平移（資料增強）

        Args:
            data_matrix: [n_locations, n_features]
            coords: [n_locations, n_dims]
            n_shifts: 循環平移次數
            random_seed: 隨機種子

        Returns:
            (shifted_matrices, shifted_coords): 多個平移版本的列表
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        n_locations = data_matrix.shape[0]
        shifted_matrices = []
        shifted_coords = []

        # 計算每個週期軸的網格大小（假設等間距網格）
        grid_sizes = {}
        for ax in self.periodic_axes:
            unique_vals = np.unique(coords[:, ax])
            grid_sizes[ax] = len(unique_vals)

        for i in range(n_shifts):
            # 隨機選擇平移量
            shift_indices = {}
            for ax in self.periodic_axes:
                if grid_sizes[ax] > 1:
                    shift_indices[ax] = np.random.randint(0, grid_sizes[ax])
                else:
                    shift_indices[ax] = 0

            # 執行循環平移
            shifted_data, shifted_coord = self._apply_circular_shift(
                data_matrix, coords, shift_indices, grid_sizes
            )

            shifted_matrices.append(shifted_data)
            shifted_coords.append(shifted_coord)

        return shifted_matrices, shifted_coords

    def _apply_circular_shift(self,
                            data_matrix: np.ndarray,
                            coords: np.ndarray,
                            shift_indices: Dict[int, int],
                            grid_sizes: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        執行單次循環平移（結構化網格版本）

        對於結構化網格，根據推斷的晶格維度執行真正的循環滾動。

        Args:
            data_matrix: [n_locations, n_features]
            coords: [n_locations, n_dims]
            shift_indices: {軸索引: 平移量}
            grid_sizes: {軸索引: 網格大小}

        Returns:
            (shifted_data, shifted_coords): 循環平移後的資料
        """
        n_locations = data_matrix.shape[0]
        n_dims = coords.shape[1]

        # 推斷網格形狀（假設資料按 Fortran/C 順序排列）
        grid_shape = self._infer_grid_shape(coords, grid_sizes)

        if grid_shape is None:
            # 回退到簡化版本（無法推斷結構）
            logger.warning("無法推斷網格結構，使用簡化循環平移")
            perm = np.roll(np.arange(n_locations), shift_indices.get(0, 0))
            return data_matrix[perm, :], coords[perm, :]

        # 將線性索引轉換為多重索引
        linear_indices = np.arange(n_locations)

        # 使用 NumPy 多重索引處理循環滾動
        # 假設數據按 C-order (row-major) 排列：最後一維變化最快
        if n_dims == 2:
            # 2D 網格：(nx, ny)
            multi_indices = np.unravel_index(linear_indices, grid_shape, order='C')

            # 對週期軸進行循環滾動
            shifted_multi = list(multi_indices)
            for ax in self.periodic_axes:
                if ax < len(shifted_multi) and ax in shift_indices:
                    shifted_multi[ax] = (multi_indices[ax] + shift_indices[ax]) % grid_shape[ax]

            # 轉換回線性索引
            shifted_linear = np.ravel_multi_index(tuple(shifted_multi), grid_shape, order='C')

        elif n_dims == 3:
            # 3D 網格：(nx, ny, nz)
            multi_indices = np.unravel_index(linear_indices, grid_shape, order='C')

            # 對週期軸進行循環滾動
            shifted_multi = list(multi_indices)
            for ax in self.periodic_axes:
                if ax < len(shifted_multi) and ax in shift_indices:
                    shifted_multi[ax] = (multi_indices[ax] + shift_indices[ax]) % grid_shape[ax]

            # 轉換回線性索引
            shifted_linear = np.ravel_multi_index(tuple(shifted_multi), grid_shape, order='C')

        else:
            # 不支援的維度，回退
            logger.warning(f"不支援 {n_dims}D 循環平移，使用簡化版本")
            perm = np.roll(np.arange(n_locations), shift_indices.get(0, 0))
            return data_matrix[perm, :], coords[perm, :]

        # 應用置換
        shifted_data = data_matrix[shifted_linear, :]
        shifted_coords = coords[shifted_linear, :]

        return shifted_data, shifted_coords

    def _infer_grid_shape(self, coords: np.ndarray, grid_sizes: Dict[int, int]) -> Optional[Tuple[int, ...]]:
        """
        從座標推斷網格形狀

        Args:
            coords: [n_locations, n_dims]
            grid_sizes: {軸索引: 網格大小}（從 circular_shift_augmentation 計算）

        Returns:
            grid_shape: (nx, ny, nz) 或 (nx, ny) 或 None（無法推斷）
        """
        n_dims = coords.shape[1]

        # 方法 1: 使用提供的 grid_sizes（優先）
        if grid_sizes:
            # 假設軸順序為 0=x, 1=y, 2=z
            shape_list = []
            for dim in range(n_dims):
                if dim in grid_sizes:
                    shape_list.append(grid_sizes[dim])
                else:
                    # 回退：計算該維度的唯一值數量
                    unique_vals = len(np.unique(coords[:, dim]))
                    shape_list.append(unique_vals)

            grid_shape = tuple(shape_list)

            # 驗證：總點數應匹配
            expected_total = int(np.prod(grid_shape))
            if expected_total == coords.shape[0]:
                return grid_shape
            else:
                logger.warning(
                    f"網格形狀驗證失敗：推斷 {grid_shape} (總數 {expected_total}) "
                    f"!= 實際 {coords.shape[0]}"
                )

        # 方法 2: 從座標計算唯一值數量（回退）
        try:
            shape_list = []
            for dim in range(n_dims):
                unique_vals = len(np.unique(np.round(coords[:, dim], decimals=8)))
                shape_list.append(unique_vals)

            grid_shape = tuple(shape_list)
            expected_total = int(np.prod(grid_shape))

            if expected_total == coords.shape[0]:
                return grid_shape
        except Exception as e:
            logger.warning(f"網格形狀推斷失敗: {e}")

        return None

    def compute_periodic_distance(self,
                                 coords1: np.ndarray,
                                 coords2: np.ndarray,
                                 domain_bounds: Dict[int, Tuple[float, float]]) -> np.ndarray:
        """
        計算週期座標的最小距離

        Args:
            coords1, coords2: [n_points, n_dims]
            domain_bounds: {軸索引: (min, max)} 域邊界

        Returns:
            distances: [n_points] 最小週期距離
        """
        distances = np.zeros(len(coords1))

        for dim in range(coords1.shape[1]):
            if dim in self.periodic_axes and dim in domain_bounds:
                # 週期軸：計算最小週期距離
                L = domain_bounds[dim][1] - domain_bounds[dim][0]
                diff = np.abs(coords1[:, dim] - coords2[:, dim])
                periodic_diff = np.minimum(diff, L - diff)
                distances += periodic_diff**2
            else:
                # 非週期軸：標準歐式距離
                distances += (coords1[:, dim] - coords2[:, dim])**2

        return np.sqrt(distances)


def build_circular_snapshot_matrix(snapshots: np.ndarray,
                                    coords: np.ndarray,
                                    grid_shape: Tuple[int, ...],
                                    periodic_axes: List[int],
                                    n_wrap_layers: int = 1,
                                    domain_lengths: Optional[Dict[int, float]] = None,
                                    seam_weight: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    建構循環索引快照矩陣（用於週期邊界條件）

    對於週期邊界條件（如通道流的 x/z 方向），在接縫處添加環繞層，
    使 QR-Pivot 能夠在連續的環狀域中選擇感測點，避免接縫聚集。

    實現策略：
    1. 將快照矩陣重塑為結構化網格 [nx, ny, nz, n_snapshots]
    2. 在週期軸上堆疊相鄰切片（例如：在 x=0 前堆疊 x=L 附近的切片）
    3. 展平回 [n_locations_augmented, n_snapshots]
    4. 同步更新座標陣列（可選：添加角度嵌入）

    Args:
        snapshots: 原始快照矩陣 [n_locations, n_snapshots]
        coords: 空間座標 [n_locations, n_dims]
        grid_shape: 網格形狀 (nx, ny, nz) 或 (nx, ny)
        periodic_axes: 週期軸索引列表（0=x, 1=y, 2=z）
        n_wrap_layers: 環繞層數（預設 1 層）
        domain_lengths: {軸索引: 域長度}（用於角度嵌入，可選）
        seam_weight: 環繞層權重係數（預設 1.0），<1 時降低接縫被選中的概率

    Returns:
        (augmented_snapshots, augmented_coords):
            - augmented_snapshots: [n_augmented, n_snapshots]
            - augmented_coords: [n_augmented, n_dims] 或 [n_augmented, n_dims+2*len(periodic_axes)]
              （如果添加角度嵌入）

    Example:
        >>> # 通道流：x/z 週期，y 非週期
        >>> snapshots = ...  # [16384, 10]
        >>> coords = ...     # [16384, 3]
        >>> grid_shape = (128, 64, 128)
        >>> periodic_axes = [0, 2]
        >>> aug_snap, aug_coords = build_circular_snapshot_matrix(
        ...     snapshots, coords, grid_shape, periodic_axes, n_wrap_layers=2
        ... )
        >>> # 結果：接縫處添加 2 層鄰域，QR-Pivot 可選擇跨越接縫的點
    """
    n_dims = coords.shape[1]
    n_locations, n_snapshots = snapshots.shape

    # 驗證網格形狀
    expected_total = int(np.prod(grid_shape))
    if expected_total != n_locations:
        raise ValueError(
            f"網格形狀 {grid_shape} 的總點數 {expected_total} "
            f"與快照矩陣行數 {n_locations} 不符"
        )

    # 重塑為結構化網格
    # 假設資料按 C-order (row-major) 排列
    if n_dims == 2:
        # 2D: [nx, ny, n_snapshots]
        snap_grid = snapshots.reshape(grid_shape[0], grid_shape[1], n_snapshots)
        coord_grid = coords.reshape(grid_shape[0], grid_shape[1], n_dims)
    elif n_dims == 3:
        # 3D: [nx, ny, nz, n_snapshots]
        snap_grid = snapshots.reshape(grid_shape[0], grid_shape[1], grid_shape[2], n_snapshots)
        coord_grid = coords.reshape(grid_shape[0], grid_shape[1], grid_shape[2], n_dims)
    else:
        raise ValueError(f"不支援 {n_dims}D 網格（僅支援 2D/3D）")

    # 對每個週期軸進行環繞堆疊
    for ax in periodic_axes:
        if ax >= len(grid_shape):
            logger.warning(f"週期軸 {ax} 超出網格維度 {len(grid_shape)}，跳過")
            continue

        # 在軸 ax 的兩端堆疊鄰域
        # 例如：x 方向週期，堆疊 x[-n:] 到 x[0] 前，x[:n] 到 x[-1] 後
        snap_grid = _wrap_grid_along_axis(snap_grid, ax, n_wrap_layers, seam_weight)
        coord_grid = _wrap_grid_along_axis(coord_grid, ax, n_wrap_layers, seam_weight=1.0)  # 座標不降權

    # 展平回 [n_augmented, n_snapshots/n_dims]
    augmented_snapshots = snap_grid.reshape(-1, n_snapshots)
    augmented_coords_flat = coord_grid.reshape(-1, n_dims)

    # 可選：添加角度嵌入（週期座標 → sin/cos）
    if domain_lengths is not None:
        augmented_coords = _add_angular_embedding(
            augmented_coords_flat, periodic_axes, domain_lengths
        )
    else:
        augmented_coords = augmented_coords_flat

    logger.info(
        f"循環快照矩陣建構完成：原始 {n_locations} → 增強 {augmented_snapshots.shape[0]} "
        f"(+{augmented_snapshots.shape[0] - n_locations} 環繞點)"
    )

    return augmented_snapshots, augmented_coords


def _wrap_grid_along_axis(grid: np.ndarray, axis: int, n_layers: int, seam_weight: float = 1.0) -> np.ndarray:
    """
    沿指定軸進行週期環繞堆疊

    Args:
        grid: [nx, ny, (nz), n_features]
        axis: 環繞軸索引
        n_layers: 環繞層數
        seam_weight: 環繞層權重係數（<1 降低接縫被選中的概率，預設 1.0 不調整）

    Returns:
        wrapped_grid: 在軸 axis 上擴展後的網格

    Note:
        當 seam_weight < 1.0 時，環繞層的數值會被縮放，
        這會降低 QR-Pivot 選擇這些重複區域的概率，
        從而避免感測點過度集中在週期邊界附近。
    """
    # 提取軸兩端的切片
    # 前端：grid[:n_layers, ...] (在 axis 維度上)
    # 後端：grid[-n_layers:, ...] (在 axis 維度上)

    # 使用 np.take 提取切片（支援任意軸）
    indices_front = np.arange(-n_layers, 0) % grid.shape[axis]  # 環繞取最後 n_layers
    indices_back = np.arange(0, n_layers) % grid.shape[axis]    # 環繞取前 n_layers

    front_slice = np.take(grid, indices_front, axis=axis)
    back_slice = np.take(grid, indices_back, axis=axis)

    # ✅ 對環繞層應用權重降低（避免被過度選擇）
    if seam_weight < 1.0:
        front_slice = front_slice * seam_weight
        back_slice = back_slice * seam_weight
        logger.debug(
            f"環繞層降權：seam_weight={seam_weight:.2f}，"
            f"軸 {axis} 的前後各 {n_layers} 層數值縮放至原始的 {100*seam_weight:.0f}%"
        )

    # 堆疊：[front | original | back]
    wrapped_grid = np.concatenate([front_slice, grid, back_slice], axis=axis)

    return wrapped_grid


def _add_angular_embedding(coords: np.ndarray,
                           periodic_axes: List[int],
                           domain_lengths: Dict[int, float]) -> np.ndarray:
    """
    為週期座標添加角度嵌入（sin/cos）

    Args:
        coords: [n_points, n_dims]
        periodic_axes: 週期軸索引
        domain_lengths: {軸索引: 域長度}

    Returns:
        coords_with_angles: [n_points, n_dims + 2*len(periodic_axes)]
            原始座標 + [sin(θ_i), cos(θ_i)] for each periodic axis i
    """
    angular_features = []

    for ax in periodic_axes:
        if ax not in domain_lengths:
            logger.warning(f"週期軸 {ax} 缺少域長度資訊，跳過角度嵌入")
            continue

        L = domain_lengths[ax]
        x_periodic = coords[:, ax]

        # 角度：θ = 2π * x / L
        theta = 2 * np.pi * x_periodic / L

        # 添加 sin/cos 特徵
        angular_features.append(np.sin(theta)[:, np.newaxis])
        angular_features.append(np.cos(theta)[:, np.newaxis])

    if angular_features:
        coords_with_angles = np.concatenate([coords] + angular_features, axis=1)
    else:
        coords_with_angles = coords

    return coords_with_angles


def prepare_turbulence_features(snapshots: Union[np.ndarray, Dict[str, np.ndarray]],
                                method: str = 'fluctuation',
                                n_time_lags: int = 3) -> np.ndarray:
    """
    準備湍流特徵（脈動量、時間lag）

    Args:
        snapshots: 時間快照資料
                  - np.ndarray: [n_time, n_locations] 單變量
                  - Dict: {'u': [n_time, n_locations], 'v': ..., 'w': ...}
        method: 特徵提取方法
               - 'fluctuation': 脈動量 u' = u - <u>
               - 'time_lag': 時間延遲特徵 [u(t), u(t-1), ...]
               - 'combined': 脈動量 + time-lag
               - 'raw': 原始快照（無處理）
        n_time_lags: 時間延遲步數（method='time_lag' 時使用）

    Returns:
        feature_matrix: [n_locations, n_features]
    """

    if isinstance(snapshots, dict):
        # 多變量情況：處理u, v, w
        variables = ['u', 'v', 'w']
        processed_vars = []

        for var in variables:
            if var in snapshots:
                var_data = snapshots[var]

                if method == 'fluctuation':
                    # 脈動量：減去時間平均
                    mean_field = var_data.mean(axis=0, keepdims=True)
                    fluctuation = var_data - mean_field
                    processed_vars.append(fluctuation)

                elif method == 'time_lag':
                    # 時間延遲特徵
                    lags = []
                    for lag in range(n_time_lags):
                        if lag < var_data.shape[0]:
                            lags.append(var_data[lag, :])
                    processed_vars.append(np.array(lags))

                elif method == 'combined':
                    # 組合：脈動量 + 第一個時間步
                    mean_field = var_data.mean(axis=0, keepdims=True)
                    fluctuation = var_data - mean_field
                    processed_vars.append(fluctuation)

                elif method == 'raw':
                    # 原始資料
                    processed_vars.append(var_data)

                else:
                    raise ValueError(f"未知的特徵提取方法: {method}")

        # 組合所有變量：[n_time, n_locations, n_vars] -> [n_locations, n_time*n_vars]
        if method == 'time_lag':
            # 時間lag：[n_lags, n_locations, n_vars] -> [n_locations, n_lags*n_vars]
            feature_matrix = np.concatenate([v.T for v in processed_vars], axis=1)
        else:
            # 其他方法：[n_time, n_locations, n_vars] -> [n_locations, n_time*n_vars]
            feature_matrix = np.concatenate([v.T for v in processed_vars], axis=1)

    else:
        # 單變量情況
        if method == 'fluctuation':
            mean_field = snapshots.mean(axis=0, keepdims=True)
            fluctuation = snapshots - mean_field
            feature_matrix = fluctuation.T  # [n_locations, n_time]

        elif method == 'time_lag':
            lags = []
            for lag in range(n_time_lags):
                if lag < snapshots.shape[0]:
                    lags.append(snapshots[lag, :])
            feature_matrix = np.array(lags).T  # [n_locations, n_lags]

        elif method == 'combined':
            mean_field = snapshots.mean(axis=0, keepdims=True)
            fluctuation = snapshots - mean_field
            feature_matrix = fluctuation.T

        elif method == 'raw':
            feature_matrix = snapshots.T  # [n_locations, n_time]

        else:
            raise ValueError(f"未知的特徵提取方法: {method}")

    return feature_matrix


def apply_min_distance_constraint(selected_indices: np.ndarray,
                                  coords: np.ndarray,
                                  min_distance: float,
                                  replacement_pool: Optional[np.ndarray] = None) -> np.ndarray:
    """
    應用最小距離約束（k-center型後處理）

    Args:
        selected_indices: 初始選擇的索引
        coords: 空間座標 [n_locations, n_dims]
        min_distance: 最小允許距離
        replacement_pool: 候選替換點索引（None = 使用全部點）

    Returns:
        refined_indices: 滿足最小距離約束的索引
    """
    refined_indices = []
    selected_coords = []

    if replacement_pool is None:
        replacement_pool = np.arange(len(coords))

    for idx in selected_indices:
        coord = coords[idx]

        # 檢查是否與已選點太近
        is_far_enough = True
        for sel_coord in selected_coords:
            dist = np.linalg.norm(coord - sel_coord)
            if dist < min_distance:
                is_far_enough = False
                break

        if is_far_enough:
            refined_indices.append(idx)
            selected_coords.append(coord)
        else:
            # 尋找替換點
            for candidate_idx in replacement_pool:
                if candidate_idx in refined_indices:
                    continue

                cand_coord = coords[candidate_idx]

                # 檢查候選點是否遠離所有已選點
                is_valid = True
                for sel_coord in selected_coords:
                    dist = np.linalg.norm(cand_coord - sel_coord)
                    if dist < min_distance:
                        is_valid = False
                        break

                if is_valid:
                    refined_indices.append(candidate_idx)
                    selected_coords.append(cand_coord)
                    break

    return np.array(refined_indices)


# ============================================================================
# 基礎選擇器類別（前置定義）
# ============================================================================

class BaseSensorSelector(ABC):
    """感測點選擇器基類"""

    @abstractmethod
    def select_sensors(self,
                      data_matrix: np.ndarray,
                      n_sensors: int,
                      **kwargs) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        選擇感測點

        Args:
            data_matrix: 資料矩陣 [n_samples, n_features]
            n_sensors: 感測點數量 K

        Returns:
            (selected_indices, metrics)
        """
        pass


# ============================================================================
# POD-DEIM 組合選擇器
# ============================================================================

class PODQREIMSelector(BaseSensorSelector):
    """
    POD + Q-DEIM 組合選擇器

    先進行POD分解獲取主要模態Φ，然後在模態空間Φᵀ上執行QR-pivot（Q-DEIM）。
    這是理論上更嚴謹的方法，避免直接在高維場值空間選點。

    參考文獻：
    - Chaturantabut & Sorensen (2010): Nonlinear Model Reduction via DEIM
    - Saibaba et al. (2016): Randomized algorithms for generalized DEIM
    """

    def __init__(self,
                 n_modes: Optional[int] = None,
                 energy_threshold: float = 0.99,
                 use_qr_pivot: bool = True,
                 periodic_axes: Optional[List[int]] = None,
                 n_circular_shifts: int = 0):
        """
        Args:
            n_modes: POD模態數量（None=自動選擇）
            energy_threshold: 能量保留閾值
            use_qr_pivot: 是否使用QR-pivot（否則用POD模態峰值）
            periodic_axes: 週期軸索引（用於循環平移）
            n_circular_shifts: 循環平移次數（0=不使用）
        """
        self.n_modes = n_modes
        self.energy_threshold = energy_threshold
        self.use_qr_pivot = use_qr_pivot
        self.periodic_handler = PeriodicBoundaryHandler(periodic_axes) if periodic_axes else None
        self.n_circular_shifts = n_circular_shifts

    def select_sensors(self,
                      data_matrix: np.ndarray,
                      n_sensors: int,
                      coords: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        POD-DEIM感測點選擇

        Args:
            data_matrix: 快照矩陣 [n_locations, n_snapshots]
            n_sensors: 感測點數量
            coords: 空間座標（用於循環平移）

        Returns:
            (selected_indices, metrics)
        """
        if isinstance(data_matrix, torch.Tensor):
            data_matrix = data_matrix.detach().cpu().numpy()
        if coords is not None and isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()

        coords_original = coords.copy() if coords is not None else None

        X = data_matrix.copy()

        # 循環平移ensemble（如果啟用）
        if self.n_circular_shifts > 0 and coords is not None and self.periodic_handler is not None:
            shifted_matrices, shifted_coords = self.periodic_handler.circular_shift_augmentation(
                X, coords, self.n_circular_shifts
            )

            # 對每個平移版本執行POD-DEIM，取出現次數最多的點
            all_selected = []
            for shifted_X in shifted_matrices:
                indices, _ = self._single_pod_deim(shifted_X, n_sensors)
                all_selected.extend(indices)

            # 統計出現頻率，選擇top-K
            unique, counts = np.unique(all_selected, return_counts=True)
            top_k_idx = np.argsort(counts)[-n_sensors:][::-1]
            selected_indices = unique[top_k_idx]

            logger.info(f"POD-DEIM ensemble: {self.n_circular_shifts} 次循環平移")

        else:
            # 標準POD-DEIM
            selected_indices, pod_metrics = self._single_pod_deim(X, n_sensors)

        # 計算品質指標
        metrics = self._compute_metrics(X, selected_indices)

        # 合併POD metrics（如果有）
        if 'pod_metrics' in locals():
            metrics.update(pod_metrics)

        return selected_indices, metrics

    def _single_pod_deim(self,
                        data_matrix: np.ndarray,
                        n_sensors: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """單次POD-DEIM選點"""

        # 1. POD分解
        U, s, Vt = svd(data_matrix, full_matrices=False)

        # 2. 確定模態數量
        if self.n_modes is None:
            cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
            n_modes = np.argmax(cumulative_energy >= self.energy_threshold) + 1
            n_modes = min(n_modes, len(s), n_sensors)
        else:
            n_modes = min(self.n_modes, len(s), n_sensors)

        # 3. 提取POD模態 Φ = U[:, :n_modes]
        pod_modes = U[:, :n_modes]  # [n_locations, n_modes]

        # 4. 在模態空間執行QR-pivot (Q-DEIM)
        if self.use_qr_pivot:
            # 對 Φᵀ 做QR分解
            Q, R, piv = qr(pod_modes.T, mode='economic', pivoting=True)
            selected_indices = piv[:n_sensors]
        else:
            # 回退方法：選擇每個模態的峰值點
            selected_indices = []
            for i in range(min(n_modes, n_sensors)):
                mode = pod_modes[:, i]
                max_idx = np.argmax(np.abs(mode))
                if max_idx not in selected_indices:
                    selected_indices.append(max_idx)
            selected_indices = np.array(selected_indices)

        # 限制數量
        selected_indices = selected_indices[:n_sensors]

        metrics = {
            'n_pod_modes': n_modes,
            'pod_energy_ratio': float(np.sum(s[:n_modes]**2) / np.sum(s**2)),
        }

        return selected_indices, metrics

    def _compute_metrics(self,
                        data_matrix: np.ndarray,
                        selected_indices: np.ndarray) -> Dict[str, float]:
        """計算品質指標"""
        qr_selector = QRPivotSelector()
        return qr_selector._compute_metrics(data_matrix, selected_indices)


# ============================================================================
# QR Pivot 選擇器
# ============================================================================

class QRPivotSelector(BaseSensorSelector):
    """
    QR-pivot 感測點選擇器

    使用 QR 分解的選主元策略選擇最具代表性的感測點。
    這是經典的貪心算法，計算高效且理論保證良好。

    **新增功能**：循環索引支援（v2024.10）
    - 透過 `use_circular_indexing=True` 啟用週期邊界處理
    - 在接縫處添加環繞層，使 QR-Pivot 能在連續環狀域中選點
    - 避免週期邊界處的感測點聚集問題
    """

    def __init__(self,
                 mode: str = 'column',
                 pivoting: bool = True,
                 regularization: float = 1e-12,
                 use_circular_indexing: bool = False,
                 n_wrap_layers: int = 1,
                 seam_weight: float = 1.0,
                 seam_width_fraction: float = 0.05,
                 max_seam_fraction: float = 0.1):
        """
        Args:
            mode: 選擇模式 ('column' 選列, 'row' 選行)
            pivoting: 是否使用選主元
            regularization: 正則化項避免數值不穩定
            use_circular_indexing: 是否使用循環索引處理週期邊界（預設 False）
            n_wrap_layers: 環繞層數（僅在 use_circular_indexing=True 時生效）
            seam_weight: 週期接縫區域的權重係數（<1 抑制接縫，=1 不調整）
            seam_width_fraction: 接縫寬度（相對於域長度的比例）
            max_seam_fraction: 接縫感測點的最大比例（超過則回補內部點）
        """
        self.mode = mode
        self.pivoting = pivoting
        self.regularization = regularization
        self.use_circular_indexing = use_circular_indexing
        self.n_wrap_layers = n_wrap_layers
        self.seam_weight = seam_weight
        self.seam_width_fraction = seam_width_fraction
        self.max_seam_fraction = max_seam_fraction
    
    def select_sensors(self,
                      data_matrix: np.ndarray,
                      n_sensors: int,
                      coords: Optional[np.ndarray] = None,
                      grid_shape: Optional[Tuple[int, ...]] = None,
                      periodic_axes: Optional[List[int]] = None,
                      domain_lengths: Optional[Dict[int, float]] = None,
                      return_qr: bool = False) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        使用 QR-pivot 選擇感測點

        Args:
            data_matrix: 資料矩陣 [n_locations, n_features] (快照法或 POD 模態)
            n_sensors: 感測點數量 K
            coords: 空間座標 [n_locations, n_dims]（循環索引時必需）
            grid_shape: 網格形狀 (nx, ny, nz) 或 (nx, ny)（循環索引時必需）
            periodic_axes: 週期軸索引列表（例如 [0, 2] 對應 x/z）
            domain_lengths: {軸索引: 域長度}（用於角度嵌入，可選）
            return_qr: 是否返回 QR 分解結果

        Returns:
            (selected_indices, metrics)
            - selected_indices: 原始網格中的感測點索引 [n_sensors]
            - metrics: 品質指標字典

        Note:
            當 `use_circular_indexing=True` 時，需要提供 coords, grid_shape, periodic_axes。
            返回的索引會自動映射回原始（未增強）網格的索引範圍。
        """
        # 確保數據為 numpy 數組
        if isinstance(data_matrix, torch.Tensor):
            data_matrix = data_matrix.detach().cpu().numpy()
        if coords is not None and isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()

        coords_original = coords.copy() if coords is not None else None

        X = data_matrix.copy()
        n_locations, n_features = X.shape

        # === 循環索引預處理 ===
        original_n_locations = n_locations  # 記錄原始點數
        index_mapping = None  # 增強索引 → 原始索引映射
        original_seam_mask = None
        if coords_original is not None and periodic_axes is not None:
            original_seam_mask = self._compute_seam_mask(
                coords_original, periodic_axes, domain_lengths, grid_shape
            )

        if self.use_circular_indexing:
            # 驗證必需參數
            if coords is None or grid_shape is None or periodic_axes is None:
                raise ValueError(
                    "循環索引模式需要提供 coords, grid_shape, periodic_axes 參數"
                )

            # 建構循環快照矩陣
            X_augmented, coords_augmented = build_circular_snapshot_matrix(
                snapshots=X,
                coords=coords,
                grid_shape=grid_shape,
                periodic_axes=periodic_axes,
                n_wrap_layers=self.n_wrap_layers,
                domain_lengths=domain_lengths,
                seam_weight=self.seam_weight  # ✅ 傳遞接縫權重
            )

            # 建立增強索引 → 原始索引的映射
            # 策略：環繞層的點映射回原始域中的對應點
            index_mapping = self._build_index_mapping(
                n_original=original_n_locations,
                n_augmented=X_augmented.shape[0],
                grid_shape=grid_shape,
                periodic_axes=periodic_axes,
                n_wrap_layers=self.n_wrap_layers
            )

            # 使用增強矩陣進行後續處理
            X = X_augmented
            coords = coords_augmented
            n_locations = X.shape[0]

            logger.info(
                f"循環索引啟用：原始 {original_n_locations} 點 → "
                f"增強 {n_locations} 點 (環繞層 {self.n_wrap_layers})"
            )
            augmented_seam_mask = self._compute_seam_mask(
                coords, periodic_axes, domain_lengths, grid_shape
            )
        else:
            augmented_seam_mask = None

        # 標準化資料（Z-Score）以改善數值穩定性
        # 避免不同特徵的數值尺度差異導致條件數過高
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-8  # 避免除以零
        X = (X - X_mean) / X_std

        if (self.use_circular_indexing and
                self.seam_weight < 1.0 and
                augmented_seam_mask is not None):
            weights = np.ones(n_locations, dtype=X.dtype)
            weights[augmented_seam_mask] = self.seam_weight
            X = (weights[:, None] ** 0.5) * X

        # 限制感測點數量（只受空間點數限制，不受特徵數限制）
        n_sensors = min(n_sensors, n_locations)
        
        Q = None
        R = None
        try:
            if self.pivoting:
                # 使用選主元 QR 分解
                # X 形狀：[n_locations, n_features]
                # 目標：選擇空間點（行），而非特徵（列）
                # QR 分解的 pivot 選擇的是「列」（對應轉置後的行）
                # 因此統一對 X.T 做 QR 分解，pivot 對應空間點索引
                if self.mode == 'column':
                    # 對 X^T 做 QR 分解選擇列 (對應原矩陣的行/空間點)
                    Q, R, piv = qr(X.T, mode='economic', pivoting=True)
                    selected_indices = piv[:n_sensors]
                else:
                    # mode='row': 同樣對 X.T 做 QR 分解選擇空間點
                    # 註：mode 參數已棄用，建議統一使用 'column' 行為
                    Q, R, piv = qr(X.T, mode='economic', pivoting=True)
                    selected_indices = piv[:n_sensors]
            else:
                # 標準 QR 分解
                Q, R = qr(X.T if self.mode == 'column' else X, mode='economic')
                # 使用對角元素大小選擇
                diag_importance = np.abs(np.diag(R))
                selected_indices = np.argsort(diag_importance)[-n_sensors:][::-1]
        
        except np.linalg.LinAlgError as e:
            logger.warning(f"QR 分解失敗，使用 SVD 回退: {e}")
            # 回退到 SVD 方法
            U, s, Vt = svd(X, full_matrices=False)
            # 使用奇異值權重選擇
            importance = np.sum(np.abs(Vt.T) * s, axis=1)
            selected_indices = np.argsort(importance)[-n_sensors:][::-1]
        
        # 確保索引在有效範圍內
        selected_indices = selected_indices[selected_indices < n_locations]
        selected_indices = selected_indices[:n_sensors]

        # === 循環索引後處理：映射回原始網格索引 ===
        if self.use_circular_indexing and index_mapping is not None:
            # 將增強網格的索引映射回原始網格
            selected_indices_original = index_mapping[selected_indices]

            # 去重（環繞層可能選到相同的原始點）
            unique_indices, unique_counts = np.unique(selected_indices_original, return_counts=True)

            fallback_added = 0

            if len(unique_indices) < n_sensors:
                logger.warning(
                    f"循環索引去重後點數不足：{len(unique_indices)} < {n_sensors}，"
                    f"保留所有唯一點"
                )
                selected_indices_final = unique_indices

                remaining = n_sensors - len(selected_indices_final)
                candidate_mask = np.ones(original_n_locations, dtype=bool)
                candidate_mask[selected_indices_final] = False
                if original_seam_mask is not None:
                    candidate_mask &= ~original_seam_mask

                candidate_indices = np.where(candidate_mask)[0]

                if candidate_indices.size > 0 and remaining > 0:
                    extra_indices = self._select_far_from_seam(
                        candidate_indices,
                        remaining,
                        coords_original,
                        periodic_axes
                    )

                    selected_indices_final = np.concatenate([selected_indices_final, extra_indices])
                    fallback_added = len(extra_indices)
                else:
                    logger.warning("沒有可用的非接縫候選點可供回補")
            else:
                # 優先保留出現次數多的點（在多個環繞位置都被選中 = 更重要）
                sort_by_importance = np.argsort(unique_counts)[::-1]
                selected_indices_final = unique_indices[sort_by_importance[:n_sensors]]

            # 限制接縫點比例
            seam_cap_applied = 0
            if (original_seam_mask is not None and
                    self.max_seam_fraction < 1.0 and
                    len(selected_indices_final) > 0):
                seam_flags = original_seam_mask[selected_indices_final]
                seam_count = int(seam_flags.sum())
                max_allowed = int(np.floor(self.max_seam_fraction * len(selected_indices_final)))
                max_allowed = max(max_allowed, 0)
                if seam_count > max_allowed:
                    excess = seam_count - max_allowed
                    seam_indices = selected_indices_final[seam_flags]
                    to_remove = seam_indices[:excess]

                    keep_mask = np.ones(len(selected_indices_final), dtype=bool)
                    remove_set = set(to_remove.tolist())
                    for i, idx in enumerate(selected_indices_final):
                        if idx in remove_set:
                            keep_mask[i] = False
                            remove_set.remove(idx)
                            if not remove_set:
                                break

                    selected_indices_pruned = selected_indices_final[keep_mask]

                    candidate_mask = np.ones(original_n_locations, dtype=bool)
                    candidate_mask[selected_indices_pruned] = False
                    candidate_mask &= ~original_seam_mask
                    candidate_indices = np.where(candidate_mask)[0]

                    replacements = self._select_far_from_seam(
                        candidate_indices,
                        excess,
                        coords_original,
                        periodic_axes
                    )

                    selected_indices_final = np.concatenate([selected_indices_pruned, replacements])
                    seam_cap_applied = excess - len(replacements)
                    fallback_added += len(replacements)
                    if len(replacements) < excess:
                        logger.warning("接縫比例限制：替換候選不足，已縮減接縫點數但未達目標比例")

            logger.info(
                f"循環索引映射：增強網格 {len(selected_indices)} 點 → "
                f"原始網格 {len(selected_indices_final)} 點（去重 {len(selected_indices) - len(unique_indices)}）"
            )

            # 更新為原始索引
            selected_indices = selected_indices_final

            # 在原始矩陣上計算指標
            metrics = self._compute_metrics(data_matrix, selected_indices)
            metrics['circular_indexing_enabled'] = True
            metrics['n_wrap_layers'] = self.n_wrap_layers
            metrics['n_duplicates_removed'] = int(len(selected_indices_original) - len(unique_indices))
            if len(unique_indices) < n_sensors:
                metrics['fallback_interior_added'] = int(fallback_added)
            if seam_cap_applied:
                metrics['seam_cap_residual'] = int(seam_cap_applied)
            if original_seam_mask is not None and len(selected_indices) > 0:
                seam_selected = int(original_seam_mask[selected_indices].sum())
                metrics['seam_selected_count'] = seam_selected
                metrics['seam_selected_ratio'] = seam_selected / len(selected_indices)
                metrics['seam_weight'] = self.seam_weight
                metrics['seam_width_fraction'] = self.seam_width_fraction
        else:
            # 標準模式：直接計算指標
            metrics = self._compute_metrics(data_matrix, selected_indices)
            metrics['circular_indexing_enabled'] = False
            if original_seam_mask is not None and len(selected_indices) > 0:
                seam_selected = int(original_seam_mask[selected_indices].sum())
                metrics['seam_selected_count'] = seam_selected
                metrics['seam_selected_ratio'] = seam_selected / len(selected_indices)

        result = (selected_indices, metrics)
        if return_qr:
            result = (*result, Q, R)

        return result

    def select_sensors_per_feature(self,
                                   data_matrix: np.ndarray,
                                   n_sensors_per_feature: int,
                                   coords: Optional[np.ndarray] = None,
                                   feature_names: Optional[List[str]] = None,
                                   return_details: bool = False) -> Union[Tuple[np.ndarray, Dict], Tuple[np.ndarray, Dict, Dict]]:
        """
        Per-feature QR-Pivot 選點策略：每個特徵獨立選擇感測點
        
        這種策略確保每個物理量（u, v, w, p, k, ...）都有專屬的代表性感測點，
        避免某些特徵被主導特徵淹沒。
        
        策略：
        1. 對每個特徵列 data_matrix[:, i] 單獨執行 QR-Pivot
        2. 為每個特徵選擇 n_sensors_per_feature 個最重要的空間點
        3. 合併所有特徵的感測點（去重）
        4. 總感測點數 ≤ n_features * n_sensors_per_feature
        
        Args:
            data_matrix: [n_locations, n_features]
            n_sensors_per_feature: 每個特徵選擇的感測點數（例如 5）
            coords: [n_locations, n_dims]（可選，用於空間分析）
            feature_names: 特徵名稱列表（可選，用於診斷）
            return_details: 是否返回詳細的 per-feature 選點資訊
        
        Returns:
            (selected_indices, metrics, [details])
            - selected_indices: 合併後的感測點索引 [≤ n_features * n_sensors_per_feature]
            - metrics: 整體品質指標
            - details (可選): 每個特徵的詳細選點資訊
        
        Example:
            >>> # 18 個特徵，每個特徵選 5 個點
            >>> indices, metrics, details = selector.select_sensors_per_feature(
            ...     data_matrix, n_sensors_per_feature=5, return_details=True
            ... )
            >>> print(f"Total sensors: {len(indices)}")  # 最多 18*5=90 個（去重後可能更少）
            >>> print(f"Feature 'u' sensors: {details['u']['indices']}")
        """
        # 確保數據為 numpy 數組
        if isinstance(data_matrix, torch.Tensor):
            data_matrix = data_matrix.detach().cpu().numpy()
        if coords is not None and isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()
        
        n_locations, n_features = data_matrix.shape
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        logger.info(f"\n🔍 Per-Feature QR-Pivot Selection:")
        logger.info(f"   Features: {n_features}")
        logger.info(f"   Sensors per feature: {n_sensors_per_feature}")
        logger.info(f"   Max total sensors: {n_features * n_sensors_per_feature}")
        
        # 標準化整個資料矩陣（跨特徵一致性）
        X_mean = data_matrix.mean(axis=0, keepdims=True)
        X_std = data_matrix.std(axis=0, keepdims=True) + 1e-8
        data_normalized = (data_matrix - X_mean) / X_std
        
        # 儲存每個特徵的選點結果
        per_feature_indices = {}
        per_feature_details = {}
        all_indices_list = []
        
        # 對每個特徵單獨執行 QR-Pivot
        for i, fname in enumerate(feature_names):
            # 提取單一特徵列作為 [n_locations, 1] 矩陣
            feature_col = data_normalized[:, i:i+1]
            
            try:
                # 對單一特徵執行 QR 分解
                # feature_col.T: [1, n_locations]
                Q, R, piv = qr(feature_col.T, mode='economic', pivoting=True)
                
                # 選擇前 n_sensors_per_feature 個主元
                n_select = min(n_sensors_per_feature, n_locations)
                feature_indices = piv[:n_select]
                
                # 計算該特徵的 R 對角線（重要性指標）
                r_diag = np.abs(np.diag(R)[:n_select])
                
                per_feature_indices[fname] = feature_indices
                per_feature_details[fname] = {
                    'indices': feature_indices,
                    'importance': r_diag,
                    'feature_index': i,
                    'n_selected': len(feature_indices)
                }
                
                all_indices_list.append(feature_indices)
                
                logger.info(f"   ✓ {fname:12s}: selected {len(feature_indices):2d} points, "
                          f"importance range [{r_diag.min():.2e}, {r_diag.max():.2e}]")
            
            except Exception as e:
                logger.warning(f"   ✗ {fname:12s}: QR failed ({e}), skipping")
                per_feature_indices[fname] = np.array([], dtype=int)
                per_feature_details[fname] = {
                    'indices': np.array([], dtype=int),
                    'importance': np.array([]),
                    'feature_index': i,
                    'n_selected': 0,
                    'error': str(e)
                }
        
        # 合併所有索引（去重）
        all_indices = np.concatenate(all_indices_list) if all_indices_list else np.array([], dtype=int)
        unique_indices, unique_counts = np.unique(all_indices, return_counts=True)
        
        # 按照出現次數排序（多個特徵都選中的點更重要）
        sort_by_importance = np.argsort(unique_counts)[::-1]
        selected_indices_final = unique_indices[sort_by_importance]
        
        logger.info(f"\n📊 Merging Results:")
        logger.info(f"   Total indices collected: {len(all_indices)}")
        logger.info(f"   Unique sensors after deduplication: {len(unique_indices)}")
        logger.info(f"   Reduction: {len(all_indices) - len(unique_indices)} duplicates removed "
                   f"({(1 - len(unique_indices)/max(len(all_indices), 1))*100:.1f}%)")
        
        # 統計哪些點被多個特徵選中
        multi_feature_sensors = unique_counts > 1
        if multi_feature_sensors.any():
            logger.info(f"   Multi-feature sensors: {multi_feature_sensors.sum()} points selected by ≥2 features")
            max_count = unique_counts.max()
            logger.info(f"   Most important sensor: selected by {max_count} features")
        
        # 計算整體指標
        metrics = self._compute_metrics(data_matrix, selected_indices_final)
        metrics['n_features'] = n_features
        metrics['n_sensors_per_feature'] = n_sensors_per_feature
        metrics['n_total_selected'] = len(selected_indices_final)
        metrics['deduplication_rate'] = float(1 - len(unique_indices) / max(len(all_indices), 1))
        metrics['multi_feature_sensors'] = int(multi_feature_sensors.sum())
        metrics['max_feature_count'] = int(unique_counts.max())
        
        # 空間分佈分析（如果提供了座標）
        if coords is not None:
            selected_coords = coords[selected_indices_final]
            for dim in range(coords.shape[1]):
                coord_name = ['x', 'y', 'z'][dim] if dim < 3 else f'dim{dim}'
                metrics[f'{coord_name}_mean'] = float(selected_coords[:, dim].mean())
                metrics[f'{coord_name}_std'] = float(selected_coords[:, dim].std())
                metrics[f'{coord_name}_range'] = float(selected_coords[:, dim].ptp())
        
        if return_details:
            return selected_indices_final, metrics, per_feature_details
        else:
            return selected_indices_final, metrics
    
    def _select_far_from_seam(self,
                              candidate_indices: np.ndarray,
                              count: int,
                              coords_original: Optional[np.ndarray],
                              periodic_axes: Optional[List[int]]) -> np.ndarray:
        """選擇距離接縫最遠的候選點"""
        if count <= 0 or candidate_indices.size == 0:
            return np.array([], dtype=int)

        if coords_original is None or periodic_axes is None:
            return candidate_indices[:count]

        candidate_coords = coords_original[candidate_indices]

        seam_dist = None
        for axis in periodic_axes:
            if axis >= candidate_coords.shape[1]:
                continue
            axis_vals = candidate_coords[:, axis]
            axis_min = coords_original[:, axis].min()
            axis_max = coords_original[:, axis].max()
            axis_dist = np.minimum(axis_vals - axis_min, axis_max - axis_vals)
            seam_dist = axis_dist if seam_dist is None else np.minimum(seam_dist, axis_dist)

        if seam_dist is None:
            seam_dist = np.ones(candidate_indices.size)

        order = np.argsort(seam_dist)
        return candidate_indices[order[-count:]]

    def _compute_seam_mask(self,
                           coords: np.ndarray,
                           periodic_axes: Optional[List[int]],
                           domain_lengths: Optional[Dict[int, float]] = None,
                           grid_shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
        """根據週期座標找出接縫區域"""
        if periodic_axes is None:
            return None

        n_points = coords.shape[0]
        seam_mask = np.zeros(n_points, dtype=bool)

        for axis in periodic_axes:
            if axis >= coords.shape[1]:
                continue

            axis_vals = coords[:, axis]
            coord_min = axis_vals.min()
            coord_max = axis_vals.max()

            if domain_lengths and axis in domain_lengths:
                domain_length = float(domain_lengths[axis])
            else:
                domain_length = coord_max - coord_min

            if domain_length <= 0:
                continue

            seam_width = self.seam_width_fraction * domain_length
            if grid_shape is not None and axis < len(grid_shape):
                cell_width = domain_length / grid_shape[axis]
                seam_width = max(seam_width, (self.n_wrap_layers + 1) * cell_width)
            seam_width = max(seam_width, 1e-12)
            lower_bound = coord_min + seam_width
            upper_bound = coord_max - seam_width

            seam_mask |= (axis_vals <= lower_bound) | (axis_vals >= upper_bound)

        return seam_mask

    def _build_index_mapping(self,
                            n_original: int,
                            n_augmented: int,
                            grid_shape: Tuple[int, ...],
                            periodic_axes: List[int],
                            n_wrap_layers: int) -> np.ndarray:
        """
        建立增強網格索引 → 原始網格索引的映射

        Args:
            n_original: 原始網格點數
            n_augmented: 增強網格點數
            grid_shape: 原始網格形狀 (nx, ny, nz) 或 (nx, ny)
            periodic_axes: 週期軸索引
            n_wrap_layers: 環繞層數

        Returns:
            mapping: [n_augmented] 陣列，mapping[i] 是增強索引 i 對應的原始索引
        """
        n_dims = len(grid_shape)

        # 計算增強網格形狀
        augmented_shape = list(grid_shape)
        for ax in periodic_axes:
            if ax < len(augmented_shape):
                augmented_shape[ax] += 2 * n_wrap_layers

        augmented_shape = tuple(augmented_shape)

        # 生成增強網格的所有線性索引
        augmented_linear = np.arange(n_augmented)

        # 轉換為多重索引
        augmented_multi = np.unravel_index(augmented_linear, augmented_shape, order='C')

        # 對每個週期軸進行模運算，映射回原始範圍
        original_multi = list(augmented_multi)
        for ax in periodic_axes:
            if ax < len(original_multi):
                # 環繞層的點：映射回原始域
                # 例如：增強網格 x ∈ [0, nx+2n_wrap)，原始網格 x ∈ [0, nx)
                # 映射：x_aug → x_orig = (x_aug - n_wrap) % nx
                original_multi[ax] = (augmented_multi[ax] - n_wrap_layers) % grid_shape[ax]

        # 轉換回原始網格的線性索引
        original_linear = np.ravel_multi_index(tuple(original_multi), grid_shape, order='C')

        return original_linear
    
    def _compute_metrics(self, 
                        data_matrix: np.ndarray, 
                        selected_indices: np.ndarray) -> Dict[str, float]:
        """計算感測點配置的品質指標"""
        
        selected_data = data_matrix[selected_indices, :]
        
        # 條件數：使用速度場條件數 κ(V)，而非 Gram 矩陣 κ(V @ V^T)
        # 原因：對於 K >> d 的低秩矩陣，Gram 矩陣有 (K-d) 個零特徵值，
        #       數值誤差會導致條件數計算出現誤導性天文數字
        try:
            _, s, _ = svd(selected_data, full_matrices=False)
            cond_number = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
        except:
            cond_number = np.inf
        
        # 行列式 (體積)
        try:
            det_value = np.linalg.det(selected_data @ selected_data.T + self.regularization * np.eye(len(selected_indices)))
            log_det = np.log(max(det_value, 1e-16))
        except:
            log_det = -np.inf
        
        # 覆蓋率 (子空間角度) 與 能量比例
        # 正確計算：比較選中點的左奇異向量能否重建全數據的主要模態
        coverage = 0.0
        energy_ratio = 0.0
        
        try:
            # 全數據的 SVD：data_matrix = U_full @ diag(s_full) @ Vt_full
            # U_full: [n_locations, n_features], 空間模態
            # Vt_full: [n_features, n_features], 特徵模態
            U_full, s_full, Vt_full = svd(data_matrix, full_matrices=False)
            
            # 選中點的 SVD：selected_data = U_selected @ diag(s_selected) @ Vt_selected
            # U_selected: [n_sensors, n_features]
            # Vt_selected: [n_features, n_features]
            U_selected, s_selected, Vt_selected = svd(selected_data, full_matrices=False)
            
            # 比較特徵模態的一致性（在特徵空間中比較）
            # Vt_full 和 Vt_selected 都是 [n_features, ...], 可以直接比較
            if len(s_selected) > 0 and len(s_full) > 0:
                n_compare = min(len(s_selected), len(s_full), min(Vt_full.shape[1], Vt_selected.shape[1]))
                
                # 子空間覆蓋率：測量選中點的特徵模態與全數據特徵模態的一致性
                # 使用 Frobenius norm 的投影比例
                # Vt_full[:n_compare, :]: (n_compare, n_features)
                # Vt_selected[:n_compare, :].conj().T: (n_features, n_compare)
                # overlap: (n_compare, n_compare) - 投影矩陣
                overlap = Vt_full[:n_compare, :] @ Vt_selected[:n_compare, :].conj().T
                # 計算正交投影的 Frobenius norm（歸一化到 [0, 1]）
                coverage = float(np.linalg.norm(overlap, 'fro')**2 / n_compare)
                
                # 能量比例：使用兩種方法計算並取平均
                #
                # 方法 1: 子空間覆蓋率（理論估計）
                # - 子空間覆蓋率 (coverage) 衡量「選中點的模態能多大程度對齊全場主模態」
                # - 這直接反映重建能力：高覆蓋率 → 選中點能有效重建全場 → 高能量捕捉
                #
                # 方法 2: 重建誤差（實際驗證）
                # - 使用選中點重建全場資料，計算實際能量比例
                # - energy_ratio = 1 - ||X - X_reconstructed||^2 / ||X||^2
                #
                # 為何不直接比較奇異值能量：
                # - s_selected 來自 [n_sensors, n_features] 矩陣（50 個空間點）
                # - s_full 來自 [n_locations, n_features] 矩陣（16384 個空間點）
                # - 兩者的奇異值尺度不可比（空間維度差異 300+ 倍）
                # - 直接比較會得到 ~0.05 的誤導性低值（僅反映採樣比例，而非重建能力）

                # 方法 1: 子空間覆蓋率
                coverage_energy = float(coverage)

                # 方法 2: 重建誤差法（實際能量比例）
                reconstruction_energy = 0.0
                try:
                    # 使用 Ridge 回歸重建全場（避免最小二乘過擬合）
                    from sklearn.linear_model import Ridge

                    # selected_data: [n_sensors, n_features]
                    # data_matrix: [n_locations, n_features]
                    # 目標：從選中的 n_sensors 個點重建全部 n_locations 個點

                    # 訓練重建模型：X_full ≈ A @ X_selected
                    # A: [n_locations, n_sensors] 重建係數矩陣
                    ridge = Ridge(alpha=1e-6, fit_intercept=False)
                    ridge.fit(selected_data.T, data_matrix.T)  # 轉置以符合 sklearn API

                    # 重建全場
                    reconstructed = ridge.predict(selected_data.T).T  # [n_locations, n_features]

                    # 計算能量比例
                    total_energy = np.linalg.norm(data_matrix, 'fro')**2
                    residual_energy = np.linalg.norm(data_matrix - reconstructed, 'fro')**2
                    reconstruction_energy = 1.0 - residual_energy / (total_energy + 1e-16)
                    reconstruction_energy = max(0.0, min(1.0, reconstruction_energy))  # 截斷到 [0, 1]

                except ImportError:
                    # sklearn 未安裝，回退到覆蓋率方法
                    reconstruction_energy = coverage_energy
                except Exception as e:
                    # 其他錯誤，回退到覆蓋率方法
                    logger.debug(f"重建能量計算失敗: {e}")
                    reconstruction_energy = coverage_energy

                # 綜合兩種方法（取平均或取重建誤差法優先）
                # 優先使用重建誤差法（更準確），但如果計算失敗則用覆蓋率
                if reconstruction_energy > 0.0:
                    energy_ratio = float(reconstruction_energy)
                else:
                    energy_ratio = float(coverage_energy)
            
        except Exception as e:
            # 靜默失敗，避免中斷流程
            pass
        
        return {
            'condition_number': float(cond_number),
            'log_determinant': float(log_det),
            'subspace_coverage': float(coverage),
            'energy_ratio': float(energy_ratio),
            'n_sensors': len(selected_indices)
        }


class PODBasedSelector(BaseSensorSelector):
    """
    基於 POD 的感測點選擇器
    
    先進行 POD 分解，然後在 POD 模態空間中進行感測點選擇。
    適用於具有明確低維結構的流場資料。
    """
    
    def __init__(self,
                 n_modes: Optional[int] = None,
                 energy_threshold: float = 0.99,
                 mode_weighting: str = 'energy'):
        """
        Args:
            n_modes: POD 模態數量 (None 為自動選擇)
            energy_threshold: 能量保留閾值
            mode_weighting: 模態權重策略 ('energy', 'uniform', 'decay')
        """
        self.n_modes = n_modes
        self.energy_threshold = energy_threshold
        self.mode_weighting = mode_weighting
        
    def select_sensors(self, 
                      data_matrix: np.ndarray,
                      n_sensors: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        基於 POD 的感測點選擇
        
        Args:
            data_matrix: 快照矩陣 [n_locations, n_snapshots]
            n_sensors: 感測點數量
            
        Returns:
            (selected_indices, metrics)
        """
        # 確保數據為 numpy 數組
        if isinstance(data_matrix, torch.Tensor):
            data_matrix = data_matrix.detach().cpu().numpy()
        
        # POD 分解
        U, s, Vt = svd(data_matrix, full_matrices=False)
        
        # 確定 POD 模態數量
        if self.n_modes is None:
            cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
            n_modes = np.argmax(cumulative_energy >= self.energy_threshold) + 1
            n_modes = min(n_modes, len(s))
        else:
            n_modes = min(self.n_modes, len(s))
        
        # 提取 POD 模態
        pod_modes = U[:, :n_modes]  # [n_locations, n_modes]
        
        # 根據模態權重策略調整
        if self.mode_weighting == 'energy':
            # 使用奇異值作為權重
            weights = s[:n_modes]
            weighted_modes = pod_modes * weights[np.newaxis, :]
        elif self.mode_weighting == 'uniform':
            # 統一權重
            weighted_modes = pod_modes
        elif self.mode_weighting == 'decay':
            # 指數衰減權重
            weights = np.exp(-np.arange(n_modes) / max(1, n_modes / 3))
            weighted_modes = pod_modes * weights[np.newaxis, :]
        else:
            weighted_modes = pod_modes
        
        # 在 POD 模態空間中使用 QR-pivot 選擇
        qr_selector = QRPivotSelector(mode='row', pivoting=True)
        selected_indices, qr_metrics = qr_selector.select_sensors(weighted_modes, n_sensors)
        
        # 計算 POD 相關指標
        pod_metrics = {
            'n_pod_modes': n_modes,
            'pod_energy_ratio': float(np.sum(s[:n_modes]**2) / np.sum(s**2)),
            'effective_rank': float(np.sum(s**2)**2 / np.sum(s**4)),  # 有效秩
        }
        
        # 合併指標
        metrics = {**qr_metrics, **pod_metrics}
        
        return selected_indices, metrics


class GreedySelector(BaseSensorSelector):
    """
    貪心感測點選擇器
    
    使用貪心算法逐步選擇最大化某個目標函數的感測點。
    支援多種目標函數：資訊增益、條件數最適化、能量最大化等。
    """
    
    def __init__(self,
                 objective: str = 'info_gain',
                 regularization: float = 1e-8):
        """
        Args:
            objective: 目標函數 ('info_gain', 'condition', 'energy', 'determinant')
            regularization: 正則化參數
        """
        self.objective = objective
        self.regularization = regularization
        
    def select_sensors(self, 
                      data_matrix: np.ndarray,
                      n_sensors: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        貪心感測點選擇
        
        Args:
            data_matrix: 資料矩陣 [n_locations, n_features]
            n_sensors: 感測點數量
            
        Returns:
            (selected_indices, metrics)
        """
        if isinstance(data_matrix, torch.Tensor):
            data_matrix = data_matrix.detach().cpu().numpy()
        
        n_locations, n_features = data_matrix.shape
        n_sensors = min(n_sensors, n_locations)
        
        selected_indices = []
        remaining_indices = list(range(n_locations))
        objective_values = []
        
        for step in range(n_sensors):
            best_idx = None
            best_objective = -np.inf
            
            for candidate_idx in remaining_indices:
                # 暫時添加候選點
                test_indices = selected_indices + [candidate_idx]
                test_data = data_matrix[test_indices, :]
                
                # 計算目標函數值
                objective_val = self._compute_objective(test_data)
                
                if objective_val > best_objective:
                    best_objective = objective_val
                    best_idx = candidate_idx
            
            # 添加最佳候選點
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                objective_values.append(best_objective)
            else:
                logger.warning(f"無法在第 {step+1} 步找到有效的感測點")
                break
        
        selected_indices = np.array(selected_indices)
        
        # 計算最終指標
        final_data = data_matrix[selected_indices, :]
        metrics = {
            'final_objective': float(best_objective),
            'objective_progression': objective_values,
            'greedy_efficiency': float(len(selected_indices) / n_sensors),
        }
        
        # 添加基本指標
        qr_selector = QRPivotSelector()
        basic_metrics = qr_selector._compute_metrics(data_matrix, selected_indices)
        metrics.update(basic_metrics)
        
        return selected_indices, metrics
    
    def _compute_objective(self, data_subset: np.ndarray) -> float:
        """計算目標函數值"""
        
        if data_subset.shape[0] == 0:
            return -np.inf
        
        try:
            gram_matrix = data_subset @ data_subset.T + self.regularization * np.eye(data_subset.shape[0])
            
            if self.objective == 'info_gain':
                # 資訊增益 = log det(Gram)
                sign, logdet = np.linalg.slogdet(gram_matrix)
                return logdet if sign > 0 else -np.inf
                
            elif self.objective == 'condition':
                # 條件數的倒數 (越大越好)
                # 使用速度場條件數而非 Gram 矩陣條件數
                _, s, _ = svd(data_subset, full_matrices=False)
                cond = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
                return -np.log(cond + 1e-16)
                
            elif self.objective == 'energy':
                # 能量 = trace(Gram)
                return np.trace(gram_matrix)
                
            elif self.objective == 'determinant':
                # 行列式
                det = np.linalg.det(gram_matrix)
                return det if det > 0 else -np.inf
                
            else:
                raise ValueError(f"未知的目標函數: {self.objective}")
                
        except np.linalg.LinAlgError:
            return -np.inf


class MultiObjectiveSelector(BaseSensorSelector):
    """
    多目標感測點選擇器
    
    同時最適化多個目標：精度、穩健性、感測點數量等。
    使用進化算法或梯度為基礎的多目標最適化。
    """
    
    def __init__(self,
                 objectives: List[str] = ['accuracy', 'robustness', 'efficiency'],
                 weights: Optional[List[float]] = None,
                 method: str = 'weighted_sum',
                 max_iterations: int = 100):
        """
        Args:
            objectives: 目標函數列表
            weights: 目標權重 (None 為等權重)
            method: 多目標方法 ('weighted_sum', 'pareto', 'lexicographic')
            max_iterations: 最大迭代次數
        """
        self.objectives = objectives
        self.weights = weights or [1.0/len(objectives)] * len(objectives)
        self.method = method
        self.max_iterations = max_iterations
        
    def select_sensors(self, 
                      data_matrix: np.ndarray,
                      n_sensors: int,
                      noise_level: float = 0.01) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        多目標感測點選擇
        
        Args:
            data_matrix: 資料矩陣
            n_sensors: 感測點數量
            noise_level: 雜訊水準 (用於穩健性評估)
            
        Returns:
            (selected_indices, metrics)
        """
        if isinstance(data_matrix, torch.Tensor):
            data_matrix = data_matrix.detach().cpu().numpy()
        
        n_locations = data_matrix.shape[0]
        
        if self.method == 'weighted_sum':
            return self._weighted_sum_optimization(data_matrix, n_sensors, noise_level)
        elif self.method == 'pareto':
            return self._pareto_optimization(data_matrix, n_sensors, noise_level)
        else:
            # 回退到 QR-pivot
            logger.warning(f"未實現的多目標方法 {self.method}，使用 QR-pivot")
            qr_selector = QRPivotSelector()
            return qr_selector.select_sensors(data_matrix, n_sensors)
    
    def _weighted_sum_optimization(self, 
                                 data_matrix: np.ndarray, 
                                 n_sensors: int, 
                                 noise_level: float) -> Tuple[np.ndarray, Dict[str, float]]:
        """加權和多目標最適化"""
        
        n_locations = data_matrix.shape[0]
        
        def objective_function(binary_selection):
            """目標函數：二進制選擇向量 -> 標量目標值"""
            indices = np.where(binary_selection > 0.5)[0]
            if len(indices) == 0:
                return 1e10  # 懲罰空選擇
            
            # 調整選擇的感測點數量
            if len(indices) > n_sensors:
                # 如果選擇太多，保留最重要的
                importance = np.sum(np.abs(data_matrix[indices, :]), axis=1)
                top_indices = np.argsort(importance)[-n_sensors:]
                indices = indices[top_indices]
            
            objectives_values = self._compute_multi_objectives(data_matrix, indices, noise_level)
            
            # 加權組合
            weighted_objective = sum(w * obj for w, obj in zip(self.weights, objectives_values))
            
            # 懲罰項：感測點數量偏差
            count_penalty = abs(len(indices) - n_sensors) * 0.1
            
            return -weighted_objective + count_penalty  # 負號因為要最大化
        
        # 使用差分進化算法
        bounds = [(0, 1)] * n_locations
        
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=self.max_iterations,
            popsize=min(15, max(10, n_locations // 10)),
            seed=42,
            atol=1e-6,
            tol=1e-6
        )
        
        # 提取選擇的感測點
        binary_solution = result.x
        selected_indices = np.where(binary_solution > 0.5)[0]
        
        # 如果數量不對，使用貪心調整
        if len(selected_indices) != n_sensors:
            if len(selected_indices) > n_sensors:
                # 移除重要性較低的點
                importance = np.sum(np.abs(data_matrix[selected_indices, :]), axis=1)
                top_k = np.argsort(importance)[-n_sensors:]
                selected_indices = selected_indices[top_k]
            else:
                # 添加重要性較高的點
                remaining = np.setdiff1d(np.arange(n_locations), selected_indices)
                importance = np.sum(np.abs(data_matrix[remaining, :]), axis=1)
                n_add = n_sensors - len(selected_indices)
                top_add = np.argsort(importance)[-n_add:]
                selected_indices = np.concatenate([selected_indices, remaining[top_add]])
        
        # 計算最終指標
        final_objectives = self._compute_multi_objectives(data_matrix, selected_indices, noise_level)
        
        metrics = {
            'multi_objective_score': float(-result.fun),
            'optimization_success': bool(result.success),
            'n_iterations': int(result.nit),
        }
        
        # 添加各個目標的值
        for i, obj_name in enumerate(self.objectives):
            metrics[f'objective_{obj_name}'] = float(final_objectives[i])
        
        return selected_indices, metrics
    
    def _compute_multi_objectives(self, 
                                data_matrix: np.ndarray, 
                                indices: np.ndarray, 
                                noise_level: float) -> List[float]:
        """計算多個目標函數值"""
        
        if len(indices) == 0:
            return [0.0] * len(self.objectives)
        
        selected_data = data_matrix[indices, :]
        objectives_values = []
        
        for obj_name in self.objectives:
            if obj_name == 'accuracy':
                # 精度：使用速度場條件數的倒數（避免 Gram 矩陣低秩問題）
                try:
                    s = np.linalg.svd(selected_data, compute_uv=False)
                    if s[-1] > 1e-15:
                        cond = s[0] / s[-1]
                    else:
                        cond = np.inf
                    accuracy = 1.0 / (1.0 + np.log(cond + 1e-16))
                except:
                    accuracy = 0.0
                objectives_values.append(accuracy)
                
            elif obj_name == 'robustness':
                # 穩健性：對雜訊的敏感度
                try:
                    # 添加雜訊並計算重建誤差
                    noisy_data = selected_data + noise_level * np.random.randn(*selected_data.shape)
                    reconstruction_error = np.linalg.norm(noisy_data - selected_data, 'fro')
                    robustness = 1.0 / (1.0 + reconstruction_error)
                except:
                    robustness = 0.0
                objectives_values.append(robustness)
                
            elif obj_name == 'efficiency':
                # 效率：單位感測點的資訊量
                try:
                    info_content = np.linalg.slogdet(selected_data @ selected_data.T + 1e-12 * np.eye(len(indices)))[1]
                    efficiency = info_content / max(1, len(indices))
                except:
                    efficiency = 0.0
                objectives_values.append(efficiency)
                
            elif obj_name == 'coverage':
                # 覆蓋率：空間分佈的均勻性
                if len(indices) > 1:
                    # 計算感測點之間的最小距離
                    min_dist = np.min([np.linalg.norm(data_matrix[i] - data_matrix[j]) 
                                     for i in indices for j in indices if i != j])
                    coverage = min_dist / (np.linalg.norm(data_matrix.max(axis=0) - data_matrix.min(axis=0)) + 1e-16)
                else:
                    coverage = 0.0
                objectives_values.append(coverage)
                
            else:
                objectives_values.append(0.0)
        
        return objectives_values
    
    def _pareto_optimization(self, 
                           data_matrix: np.ndarray, 
                           n_sensors: int, 
                           noise_level: float) -> Tuple[np.ndarray, Dict[str, float]]:
        """Pareto 前沿多目標最適化 (簡化版)"""
        
        # 簡化實現：生成多個候選解，選擇 Pareto 最適
        n_candidates = min(50, data_matrix.shape[0])
        candidates = []
        
        # 使用不同策略生成候選解
        selectors = [
            QRPivotSelector(mode='column'),
            PODBasedSelector(n_modes=min(10, data_matrix.shape[1] // 2)),
            GreedySelector(objective='info_gain'),
            GreedySelector(objective='condition')
        ]
        
        for selector in selectors:
            try:
                indices, _ = selector.select_sensors(data_matrix, n_sensors)
                objectives = self._compute_multi_objectives(data_matrix, indices, noise_level)
                candidates.append((indices, objectives))
            except:
                continue
        
        # 添加隨機候選
        for _ in range(n_candidates - len(candidates)):
            random_indices = np.random.choice(data_matrix.shape[0], n_sensors, replace=False)
            objectives = self._compute_multi_objectives(data_matrix, random_indices, noise_level)
            candidates.append((random_indices, objectives))
        
        # 找到 Pareto 前沿
        pareto_candidates = self._find_pareto_front(candidates)
        
        if pareto_candidates:
            # 從 Pareto 前沿中選擇加權最佳解
            best_score = -np.inf
            best_solution = None
            
            for indices, objectives in pareto_candidates:
                weighted_score = sum(w * obj for w, obj in zip(self.weights, objectives))
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_solution = (indices, objectives)
            
            selected_indices, final_objectives = best_solution
        else:
            # 回退到第一個候選
            selected_indices, final_objectives = candidates[0]
        
        metrics = {
            'pareto_front_size': len(pareto_candidates),
            'n_candidates_evaluated': len(candidates),
            'pareto_score': float(best_score),
        }
        
        for i, obj_name in enumerate(self.objectives):
            metrics[f'objective_{obj_name}'] = float(final_objectives[i])
        
        return selected_indices, metrics
    
    def _find_pareto_front(self, candidates: List[Tuple]) -> List[Tuple]:
        """找到 Pareto 前沿"""
        pareto_front = []
        
        for candidate in candidates:
            is_dominated = False
            
            for other in candidates:
                if candidate == other:
                    continue
                
                # 檢查是否被支配（所有目標都不優於其他解）
                candidate_objectives = candidate[1]
                other_objectives = other[1]
                
                if all(c <= o for c, o in zip(candidate_objectives, other_objectives)) and \
                   any(c < o for c, o in zip(candidate_objectives, other_objectives)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        return pareto_front


class SensorOptimizer:
    """
    感測點最適化器
    
    提供高層級的感測點選擇接口，整合多種算法並支援自動超參數調優。
    """
    
    def __init__(self,
                 strategy: str = 'auto',
                 config: Optional[Dict] = None):
        """
        Args:
            strategy: 選擇策略 ('qr_pivot', 'pod_based', 'greedy', 'multi_objective', 'auto')
            config: 策略配置字典
        """
        self.strategy = strategy
        self.config = config or {}
        
    def optimize_sensor_placement(self,
                                 data_matrix: np.ndarray,
                                 n_sensors: int,
                                 validation_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        最適化感測點配置
        
        Args:
            data_matrix: 訓練資料矩陣
            n_sensors: 感測點數量
            validation_data: 驗證資料 (用於評估)
            
        Returns:
            (optimal_indices, comprehensive_metrics)
        """
        if self.strategy == 'auto':
            return self._auto_strategy_selection(data_matrix, n_sensors, validation_data)
        else:
            selector = self._create_selector(self.strategy)
            selected_indices, metrics = selector.select_sensors(data_matrix, n_sensors)
            
            # 如果有驗證資料，計算驗證指標
            if validation_data is not None:
                validation_metrics = self._evaluate_on_validation(
                    data_matrix, validation_data, selected_indices)
                metrics.update(validation_metrics)
            
            return selected_indices, metrics
    
    def _create_selector(self, strategy: str) -> BaseSensorSelector:
        """創建特定策略的選擇器"""
        
        if strategy == 'qr_pivot':
            return QRPivotSelector(**self.config.get('qr_pivot', {}))
        elif strategy == 'pod_based':
            return PODBasedSelector(**self.config.get('pod_based', {}))
        elif strategy == 'greedy':
            return GreedySelector(**self.config.get('greedy', {}))
        elif strategy == 'multi_objective':
            return MultiObjectiveSelector(**self.config.get('multi_objective', {}))
        else:
            raise ValueError(f"未知的感測點選擇策略: {strategy}")
    
    def _auto_strategy_selection(self,
                               data_matrix: np.ndarray,
                               n_sensors: int,
                               validation_data: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """自動策略選擇"""
        
        # 分析資料特性
        n_locations, n_features = data_matrix.shape
        data_rank = np.linalg.matrix_rank(data_matrix)
        aspect_ratio = n_features / n_locations
        
        # 根據資料特性選擇策略
        if data_rank < min(n_locations, n_features) * 0.8:
            # 低秩資料：使用 POD
            strategy = 'pod_based'
            logger.info("檢測到低秩結構，使用 POD-based 策略")
        elif aspect_ratio > 2.0:
            # 寬矩陣：使用 QR-pivot
            strategy = 'qr_pivot'
            logger.info("檢測到寬矩陣結構，使用 QR-pivot 策略")
        elif n_sensors / n_locations < 0.1:
            # 極稀疏感測：使用多目標最適化
            strategy = 'multi_objective'
            logger.info("檢測到極稀疏感測需求，使用多目標最適化")
        else:
            # 預設：貪心算法
            strategy = 'greedy'
            logger.info("使用預設貪心策略")
        
        # 執行選擇
        selector = self._create_selector(strategy)
        selected_indices, metrics = selector.select_sensors(data_matrix, n_sensors)
        
        # 添加自動選擇信息
        metrics['auto_selected_strategy'] = strategy
        metrics['data_analysis'] = {
            'rank': int(data_rank),
            'aspect_ratio': float(aspect_ratio),
            'sparsity_ratio': float(n_sensors / n_locations)
        }
        
        # 驗證評估
        if validation_data is not None:
            validation_metrics = self._evaluate_on_validation(
                data_matrix, validation_data, selected_indices)
            metrics.update(validation_metrics)
        
        return selected_indices, metrics
    
    def _evaluate_on_validation(self,
                              train_data: np.ndarray,
                              validation_data: np.ndarray,
                              selected_indices: np.ndarray) -> Dict[str, float]:
        """在驗證資料上評估感測點配置"""
        
        try:
            # 使用選擇的感測點進行重建
            sensor_data_train = train_data[selected_indices, :]
            sensor_data_val = validation_data[selected_indices, :]
            
            # 計算重建誤差（簡單線性重建）
            if sensor_data_train.shape[0] >= sensor_data_train.shape[1]:
                # 超定系統
                reconstruction_matrix = np.linalg.pinv(sensor_data_train)
                coefficients = reconstruction_matrix @ validation_data
                reconstructed = sensor_data_train @ coefficients
            else:
                # 欠定系統
                regularization = 1e-6
                gram = sensor_data_train @ sensor_data_train.T + regularization * np.eye(sensor_data_train.shape[0])
                reconstruction_matrix = sensor_data_train.T @ np.linalg.pinv(gram)
                coefficients = reconstruction_matrix @ sensor_data_val
                reconstructed = train_data @ coefficients
            
            # 計算誤差指標
            mse = np.mean((validation_data - reconstructed)**2)
            relative_error = np.linalg.norm(validation_data - reconstructed, 'fro') / \
                           (np.linalg.norm(validation_data, 'fro') + 1e-16)
            
            return {
                'validation_mse': float(mse),
                'validation_relative_error': float(relative_error),
                'reconstruction_rank': int(np.linalg.matrix_rank(reconstruction_matrix))
            }
            
        except Exception as e:
            logger.warning(f"驗證評估失敗: {e}")
            return {
                'validation_mse': np.inf,
                'validation_relative_error': np.inf,
                'reconstruction_rank': 0
            }


def evaluate_sensor_placement(data_matrix: np.ndarray,
                            selected_indices: np.ndarray,
                            test_data: Optional[np.ndarray] = None,
                            noise_levels: List[float] = [0.01, 0.05, 0.1]) -> Dict[str, Any]:
    """
    評估感測點配置的品質
    
    Args:
        data_matrix: 原始資料矩陣
        selected_indices: 選擇的感測點索引
        test_data: 測試資料 (可選)
        noise_levels: 雜訊水準列表
        
    Returns:
        綜合評估指標字典
    """
    metrics = {}
    
    # 基本指標
    qr_selector = QRPivotSelector()
    basic_metrics = qr_selector._compute_metrics(data_matrix, selected_indices)
    metrics.update(basic_metrics)
    
    # 雜訊穩健性測試
    if test_data is not None:
        robustness_metrics = {}
        
        for noise_level in noise_levels:
            try:
                # 添加雜訊
                noisy_test = test_data + noise_level * np.random.randn(*test_data.shape)
                
                # 重建測試
                sensor_train = data_matrix[selected_indices, :]
                sensor_test = noisy_test[selected_indices, :]
                
                # 簡單線性重建
                reconstruction_matrix = np.linalg.pinv(sensor_train)
                reconstructed = sensor_train @ (reconstruction_matrix @ test_data)
                
                # 計算誤差
                reconstruction_error = np.linalg.norm(reconstructed - test_data, 'fro') / \
                                     (np.linalg.norm(test_data, 'fro') + 1e-16)
                
                robustness_metrics[f'noise_{noise_level}_error'] = float(reconstruction_error)
                
            except Exception as e:
                robustness_metrics[f'noise_{noise_level}_error'] = np.inf
        
        metrics['robustness'] = robustness_metrics
    
    # 幾何分佈分析
    if len(selected_indices) > 1:
        coordinates = data_matrix[selected_indices, :2] if data_matrix.shape[1] >= 2 else data_matrix[selected_indices, :]
        
        # 計算最小距離
        min_distance = np.inf
        max_distance = 0.0
        
        for i in range(len(selected_indices)):
            for j in range(i+1, len(selected_indices)):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                min_distance = min(min_distance, dist)
                max_distance = max(max_distance, dist)
        
        metrics['geometry'] = {
            'min_sensor_distance': float(min_distance),
            'max_sensor_distance': float(max_distance),
            'distance_ratio': float(max_distance / (min_distance + 1e-16))
        }
    
    return metrics


def create_sensor_selector(strategy: str = 'qr_pivot', 
                         **kwargs) -> BaseSensorSelector:
    """
    創建感測點選擇器的便捷函數
    
    Args:
        strategy: 選擇策略
        **kwargs: 策略特定參數
        
    Returns:
        感測點選擇器實例
    """
    if strategy == 'qr_pivot':
        return QRPivotSelector(**kwargs)
    elif strategy == 'pod_based':
        return PODBasedSelector(**kwargs)
    elif strategy == 'greedy':
        return GreedySelector(**kwargs)
    elif strategy == 'multi_objective':
        return MultiObjectiveSelector(**kwargs)
    else:
        raise ValueError(f"未知的感測點選擇策略: {strategy}")


if __name__ == "__main__":
    # 測試程式碼
    print("🧪 測試感測點選擇模組...")
    
    # 創建測試資料
    np.random.seed(42)
    n_locations = 100
    n_snapshots = 50
    
    # 模擬低維流場資料
    t = np.linspace(0, 2*np.pi, n_snapshots)
    x = np.linspace(0, 1, n_locations)
    
    # 創建含有幾個主要模態的資料
    data_matrix = np.zeros((n_locations, n_snapshots))
    for i in range(3):  # 3個主要模態
        mode = np.sin((i+1) * np.pi * x[:, np.newaxis])
        coeff = np.cos((i+1) * t) * np.exp(-0.1 * i)
        data_matrix += mode @ coeff[np.newaxis, :]
    
    # 添加雜訊
    data_matrix += 0.01 * np.random.randn(n_locations, n_snapshots)
    
    n_sensors = 8
    
    # 測試不同的選擇策略
    strategies = {
        'QR-Pivot': QRPivotSelector(),
        'POD-based': PODBasedSelector(n_modes=5),
        'Greedy': GreedySelector(objective='info_gain'),
        'Multi-objective': MultiObjectiveSelector(objectives=['accuracy', 'robustness'])
    }
    
    results = {}
    
    for name, selector in strategies.items():
        print(f"\n測試 {name} 策略...")
        try:
            indices, metrics = selector.select_sensors(data_matrix, n_sensors)
            results[name] = {
                'indices': indices,
                'condition_number': metrics.get('condition_number', np.inf),
                'energy_ratio': metrics.get('energy_ratio', 0.0),
                'n_selected': len(indices)
            }
            print(f"  選擇感測點: {len(indices)} 個")
            print(f"  條件數: {metrics.get('condition_number', 'N/A'):.2f}")
            print(f"  能量比例: {metrics.get('energy_ratio', 0.0):.3f}")
        except Exception as e:
            print(f"  ❌ 失敗: {e}")
            results[name] = {'error': str(e)}
    
    # 測試自動策略選擇
    print(f"\n測試自動策略選擇...")
    optimizer = SensorOptimizer(strategy='auto')
    auto_indices, auto_metrics = optimizer.optimize_sensor_placement(data_matrix, n_sensors)
    print(f"  自動選擇策略: {auto_metrics.get('auto_selected_strategy', 'unknown')}")
    print(f"  選擇感測點: {len(auto_indices)} 個")
    
    # 評估所有策略
    print(f"\n綜合評估...")
    for name, result in results.items():
        if 'error' not in result:
            eval_metrics = evaluate_sensor_placement(data_matrix, result['indices'])
            print(f"  {name}: 條件數={eval_metrics.get('condition_number', 'N/A'):.2f}, "
                  f"覆蓋率={eval_metrics.get('subspace_coverage', 0.0):.3f}")
    
    print("✅ 感測點選擇模組測試完成！")


class PhysicsGuidedQRPivotSelector(QRPivotSelector):
    """
    物理引導 QR-Pivot 感測點選擇器
    
    在標準 QR-Pivot 基礎上引入物理先驗（壁面邊界條件），
    通過對 POD 模態矩陣進行物理加權，優先選擇壁面高梯度區域的感測點。
    
    核心改進：
    1. 壁面區域識別（基於 y+ 或 y/h）
    2. 物理權重矩陣（壁面權重放大）
    3. 加權 QR-Pivot（在加權模態空間中選點）
    4. 壁面覆蓋率統計（驗證策略有效性）
    
    適用場景：
    - 湍流通道流（壁面剪應力重要）
    - 邊界層流動（壁面梯度敏感）
    - 任何需要優先捕捉邊界條件的流場
    
    參考文獻：
    - Manohar et al. (2018): Data-driven sparse sensor placement
    - 本專案 PDE 約束消融實驗：Exp3 (Wall No-Center) 證實壁面密集採樣的優勢
    """
    
    def __init__(self, 
                 mode: str = 'column',
                 pivoting: bool = True,
                 regularization: float = 1e-12,
                 wall_weight: float = 5.0,
                 wall_threshold: float = 0.1,
                 threshold_type: str = 'y_over_h'):
        """
        Args:
            mode: 選擇模式 ('column' 選列)
            pivoting: 是否使用選主元
            regularization: 正則化項避免數值不穩定
            wall_weight: 壁面區域權重倍數（預設 5.0，基於 Exp3 最優配置）
            wall_threshold: 壁面區域閾值
                - threshold_type='y_over_h': y/h < 0.1 (對應 y+ ≈ 100 at Re_τ=1000)
                - threshold_type='y_plus': y+ < 100 (黏性底層 + 緩衝層)
            threshold_type: 壁面識別類型 ('y_over_h' 或 'y_plus')
        """
        super().__init__(mode=mode, pivoting=pivoting, regularization=regularization)
        self.wall_weight = wall_weight
        self.wall_threshold = wall_threshold
        self.threshold_type = threshold_type
        
        # 記錄壁面權重應用狀態
        self._wall_mask = None
        self._wall_coverage = 0.0
    
    def select_sensors(self, 
                      data_matrix: np.ndarray,
                      n_sensors: int,
                      coords: Optional[np.ndarray] = None,
                      re_tau: float = 1000.0,
                      return_qr: bool = False) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        使用物理引導 QR-pivot 選擇感測點
        
        Args:
            data_matrix: POD 模態矩陣 [n_locations, n_modes] 或快照矩陣 [n_locations, n_snapshots]
            n_sensors: 感測點數量 K
            coords: 空間座標 [n_locations, 3] (x, y, z)，必須提供用於計算壁面距離
            re_tau: 摩擦雷諾數（用於 y+ 計算，預設 1000.0 對應 JHTDB Channel Flow）
            return_qr: 是否返回 QR 分解結果
            
        Returns:
            (selected_indices, metrics)
            
        Raises:
            ValueError: 如果未提供 coords 且需要計算壁面距離
        """
        # 確保數據為 numpy 數組
        if isinstance(data_matrix, torch.Tensor):
            data_matrix = data_matrix.detach().cpu().numpy()
        if coords is not None and isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()
        
        # 驗證座標輸入
        if coords is None:
            raise ValueError(
                "PhysicsGuidedQRPivotSelector 需要提供空間座標 'coords' 用於計算壁面距離。"
                "座標格式：[n_locations, 3] (x, y, z)"
            )
        
        if coords.shape[0] != data_matrix.shape[0]:
            raise ValueError(
                f"座標數量 ({coords.shape[0]}) 與資料點數量 ({data_matrix.shape[0]}) 不匹配"
            )
        
        X = data_matrix.copy()
        n_locations, n_features = X.shape
        
        # 標準化資料（Z-Score）
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - X_mean) / X_std
        
        # 限制感測點數量
        n_sensors = min(n_sensors, n_locations)
        
        # === 核心改進：物理引導加權 ===
        
        # 1. 識別壁面區域
        wall_mask = self._identify_wall_region(coords, re_tau)
        self._wall_mask = wall_mask  # 記錄用於後續統計
        
        # 2. 建立物理權重矩陣（對角矩陣）
        weights = np.ones(n_locations, dtype=np.float64)
        weights[wall_mask] = self.wall_weight  # 壁面區域權重放大
        W = np.diag(weights)
        
        # 3. 對 POD 模態矩陣進行物理加權
        # weighted_modes: [n_locations, n_features]
        # 壁面點的模態係數被放大，在 QR-Pivot 中優先選擇
        X_weighted = W @ X
        
        logger.info(
            f"物理引導 QR-Pivot: 壁面點 {wall_mask.sum()}/{n_locations} "
            f"({100*wall_mask.sum()/n_locations:.1f}%), 權重 {self.wall_weight:.1f}x"
        )
        
        # 4. 對加權矩陣執行 QR-Pivot
        Q = None
        R = None
        try:
            if self.pivoting:
                # 對 X_weighted^T 做 QR 分解
                Q, R, piv = qr(X_weighted.T, mode='economic', pivoting=True)
                selected_indices = piv[:n_sensors]
            else:
                # 標準 QR 分解（不推薦，加權後仍應使用 pivoting）
                Q, R = qr(X_weighted.T if self.mode == 'column' else X_weighted, mode='economic')
                diag_importance = np.abs(np.diag(R))
                selected_indices = np.argsort(diag_importance)[-n_sensors:][::-1]
        
        except np.linalg.LinAlgError as e:
            logger.warning(f"QR 分解失敗，使用 SVD 回退: {e}")
            # 回退到 SVD 方法
            U, s, Vt = svd(X_weighted, full_matrices=False)
            importance = np.sum(np.abs(Vt.T) * s, axis=1)
            selected_indices = np.argsort(importance)[-n_sensors:][::-1]
        
        # 確保索引在有效範圍內
        selected_indices = selected_indices[selected_indices < n_locations]
        selected_indices = selected_indices[:n_sensors]
        
        # 5. 計算品質指標（使用原始未加權矩陣）
        metrics = self._compute_metrics(X, selected_indices)
        
        # 6. 添加物理引導特定指標
        wall_coverage = wall_mask[selected_indices].sum() / len(selected_indices)
        self._wall_coverage = wall_coverage
        
        physics_metrics = {
            'wall_coverage': float(wall_coverage),  # 壁面覆蓋率（選中點中壁面點的比例）
            'wall_weight': float(self.wall_weight),
            'wall_threshold': float(self.wall_threshold),
            'threshold_type': self.threshold_type,
            'total_wall_points': int(wall_mask.sum()),
            'selected_wall_points': int(wall_mask[selected_indices].sum()),
        }
        metrics.update(physics_metrics)
        
        result = (selected_indices, metrics)
        if return_qr:
            result = (*result, Q, R)
        
        return result
    
    def _identify_wall_region(self, coords: np.ndarray, re_tau: float) -> np.ndarray:
        """
        識別壁面區域
        
        Args:
            coords: 空間座標 [n_locations, 3] (x, y, z)
            re_tau: 摩擦雷諾數
            
        Returns:
            wall_mask: 布林陣列 [n_locations]，True 表示壁面區域
        """
        # 假設通道流幾何：y ∈ [-h, h]，h=1
        # 壁面位於 y=-1 和 y=1
        y_coords = coords[:, 1]  # 提取 y 座標
        
        if self.threshold_type == 'y_over_h':
            # 使用無因次距離 y/h
            # 計算到最近壁面的距離（歸一化）
            h = 1.0  # 通道半高
            y_min, y_max = -h, h
            
            # 到上下壁面的距離
            dist_to_lower_wall = np.abs(y_coords - y_min)
            dist_to_upper_wall = np.abs(y_coords - y_max)
            dist_to_wall = np.minimum(dist_to_lower_wall, dist_to_upper_wall)
            
            # 歸一化距離 (0 在壁面, 1 在中心)
            y_over_h = dist_to_wall / h
            
            # 壁面區域：y/h < threshold（例如 0.1 對應 y+ ≈ 100）
            wall_mask = y_over_h < self.wall_threshold
            
        elif self.threshold_type == 'y_plus':
            # 使用壁面座標 y+（需要摩擦速度 u_τ）
            # JHTDB Channel Flow Re_τ=1000:
            #   u_τ = 0.04997
            #   ν = 5e-5
            #   δ_ν = ν/u_τ ≈ 1.0e-3
            
            u_tau = 0.04997  # JHTDB 統計量
            nu = 5.0e-5      # JHTDB 黏滯係數
            delta_nu = nu / u_tau  # 黏性長度尺度
            
            # 計算到最近壁面的物理距離
            h = 1.0
            y_min, y_max = -h, h
            dist_to_lower_wall = np.abs(y_coords - y_min)
            dist_to_upper_wall = np.abs(y_coords - y_max)
            dist_to_wall = np.minimum(dist_to_lower_wall, dist_to_upper_wall)
            
            # 壁面座標 y+ = y_physical / δ_ν
            y_plus = dist_to_wall / delta_nu
            
            # 壁面區域：y+ < threshold（例如 100 對應黏性底層 + 緩衝層）
            wall_mask = y_plus < self.wall_threshold
            
        else:
            raise ValueError(f"未知的壁面識別類型: {self.threshold_type}")
        
        return wall_mask
    
    def get_wall_statistics(self) -> Dict[str, Any]:
        """
        獲取壁面統計信息（需在 select_sensors 後調用）
        
        Returns:
            統計字典
        """
        if self._wall_mask is None:
            raise RuntimeError("請先調用 select_sensors() 方法")
        
        return {
            'wall_coverage': float(self._wall_coverage),
            'total_wall_points': int(self._wall_mask.sum()),
            'wall_ratio': float(self._wall_mask.sum() / len(self._wall_mask)),
            'wall_weight': float(self.wall_weight),
            'threshold': float(self.wall_threshold),
            'threshold_type': self.threshold_type,
        }

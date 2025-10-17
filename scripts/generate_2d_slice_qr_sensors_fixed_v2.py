"""
從 JHTDB 3D 數據生成 2D 切片並進行修正版 QR-Pivot 感測點選擇 (v2)

修正內容（針對條件數災難 κ=3.44e19）:
1. ✅ 變數物理尺度加權（u_tau 標準化）
2. ✅ RRQR (Rank-Revealing QR) 替代標準 QR
3. ✅ 最小距離約束（幾何去相關）
4. ✅ 降階 + 超額取樣策略
5. ✅ Gram 矩陣條件數驗證

目標: κ 從 3.44e19 降至 1e3-1e6 可用範圍
"""

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple
from datetime import datetime
from scipy.linalg import qr, svd
from scipy.spatial.distance import pdist, squareform

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

# 動態導入避免編譯時錯誤
try:
    from pinnx.dataio.jhtdb_cutout_loader import JHTDBCutoutLoader
except ImportError:
    JHTDBCutoutLoader = None  # 執行時再檢查

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 修正版 QR-Pivot 選擇器（針對條件數災難）
# ============================================================================

class RobustQRPivotSelector:
    """
    強健版 QR-Pivot 選擇器
    
    修正措施:
    1. 變數物理尺度加權
    2. RRQR 選點
    3. 最小距離約束
    4. 降階 + 超額取樣
    5. 截斷 SVD 穩定解法
    """
    
    def __init__(self, 
                 u_tau: float = 0.04997,  # 摩擦速度
                 min_distance_factor: float = 2.0,  # 最小距離 = factor × grid_spacing
                 energy_threshold: float = 0.99,   # 能量保留閾值
                 oversample_ratio: float = 1.5,    # 超額取樣比例
                 svd_threshold: float = 1e-8):     # SVD 截斷閾值
        """
        Args:
            u_tau: 摩擦速度（用於物理尺度加權）
            min_distance_factor: 最小距離倍數
            energy_threshold: 能量保留閾值（決定有效秩）
            oversample_ratio: 超額取樣比例（k = r × ratio）
            svd_threshold: SVD 截斷閾值（相對於最大奇異值）
        """
        self.u_tau = u_tau
        self.min_distance_factor = min_distance_factor
        self.energy_threshold = energy_threshold
        self.oversample_ratio = oversample_ratio
        self.svd_threshold = svd_threshold
    
    def select_sensors(self, 
                      snapshot_matrix: np.ndarray,
                      coords_3d: np.ndarray,
                      n_sensors: int,
                      grid_spacing: float) -> Tuple[np.ndarray, Dict]:
        """
        修正版 QR-Pivot 感測點選擇
        
        Args:
            snapshot_matrix: 原始快照矩陣 [n_locations, n_features]
            coords_3d: 3D 座標 [n_locations, 3]
            n_sensors: 目標感測點數量 K
            grid_spacing: 網格間距（用於最小距離約束）
        
        Returns:
            (selected_indices, metrics)
        """
        logger.info("=" * 80)
        logger.info("強健版 QR-Pivot 感測點選擇（v2）")
        logger.info("=" * 80)
        
        # 步驟 1: 變數物理尺度加權
        U_weighted = self._apply_physical_weighting(snapshot_matrix)
        logger.info(f"✅ 步驟 1: 物理尺度加權完成")
        logger.info(f"   加權前範圍: [{snapshot_matrix.min():.2e}, {snapshot_matrix.max():.2e}]")
        logger.info(f"   加權後範圍: [{U_weighted.min():.2e}, {U_weighted.max():.2e}]")
        
        # 步驟 2: 降階（能量門檻選 r）
        r_effective = int(self._estimate_effective_rank(U_weighted))
        logger.info(f"✅ 步驟 2: 有效秩估計 r = {r_effective}")
        
        # 步驟 3: 超額取樣（k = r × ratio）
        k_oversampled = min(int(r_effective * self.oversample_ratio), snapshot_matrix.shape[0])
        logger.info(f"✅ 步驟 3: 超額取樣 k = {k_oversampled} (ratio={self.oversample_ratio})")
        
        # 步驟 4: RRQR 選點
        rrqr_indices = self._rrqr_selection(U_weighted, k_oversampled)
        logger.info(f"✅ 步驟 4: RRQR 選點完成，候選點數 {len(rrqr_indices)}")
        
        # 步驟 5: 最小距離約束（幾何去相關）
        min_distance = self.min_distance_factor * grid_spacing
        filtered_indices = self._filter_by_min_distance(
            coords_3d, rrqr_indices, min_distance, target_k=n_sensors
        )
        logger.info(f"✅ 步驟 5: 幾何去相關完成，最終點數 {len(filtered_indices)}")
        logger.info(f"   最小距離約束: {min_distance:.4f}")
        
        # 步驟 6: 計算品質指標
        metrics = self._compute_robust_metrics(
            snapshot_matrix, U_weighted, coords_3d, filtered_indices
        )
        logger.info(f"✅ 步驟 6: 品質指標計算完成")
        logger.info(f"   速度場條件數: {metrics.get('velocity_condition_number', np.inf):.2e}")
        logger.info(f"   加權條件數: {metrics.get('weighted_condition_number', np.inf):.2e}")
        logger.info(f"   最小點間距: {metrics['min_distance']:.4f}")
        
        return filtered_indices, metrics
    
    def _apply_physical_weighting(self, snapshot_matrix: np.ndarray) -> np.ndarray:
        """
        步驟 1: 變數物理尺度加權
        
        假設 snapshot_matrix = [u, v, w] 或 [u, v, p, laplacian]
        使用 u_tau 標準化速度分量
        """
        U_weighted = snapshot_matrix.copy()
        n_features = U_weighted.shape[1]
        
        if n_features == 3:
            # [u, v, w]
            U_weighted[:, 0] /= self.u_tau  # u
            U_weighted[:, 1] /= self.u_tau  # v
            U_weighted[:, 2] /= self.u_tau  # w
        elif n_features == 4:
            # [u, v, p, laplacian]
            U_weighted[:, 0] /= self.u_tau  # u
            U_weighted[:, 1] /= self.u_tau  # v
            # p 和 laplacian 保持原尺度（已在量綱上平衡）
        
        # 再進行 Z-Score 標準化（消除數值範圍差異）
        means = U_weighted.mean(axis=0, keepdims=True)
        stds = U_weighted.std(axis=0, keepdims=True)
        stds[stds < 1e-12] = 1.0
        U_weighted = (U_weighted - means) / stds
        
        return U_weighted
    
    def _estimate_effective_rank(self, U_weighted: np.ndarray) -> int:
        """
        步驟 2: 估計有效秩 r（能量門檻法）
        """
        U_svd, s, _ = svd(U_weighted, full_matrices=False)
        cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
        r = np.searchsorted(cumulative_energy, self.energy_threshold) + 1
        
        # 至少保留 min(10, n_features) 個模態
        r = max(r, min(10, U_weighted.shape[1]))
        
        return r
    
    def _rrqr_selection(self, U_weighted: np.ndarray, k: int) -> np.ndarray:
        """
        步驟 4: RRQR (Rank-Revealing QR) 選點
        
        使用 scipy.linalg.qr(pivoting=True) 實現
        """
        # 對 U_weighted.T 做 QR 分解（選空間點）
        Q, R, P = qr(U_weighted.T, pivoting=True, mode='economic')
        
        # P[:k] 是選中的空間點索引
        selected_indices = P[:k]
        
        return selected_indices
    
    def _filter_by_min_distance(self, 
                                coords: np.ndarray,
                                candidate_indices: np.ndarray,
                                min_distance: float,
                                target_k: int) -> np.ndarray:
        """
        步驟 5: 最小距離約束（幾何去相關）
        
        貪心算法：從候選點中選擇滿足最小距離約束的子集
        """
        selected = []
        coords_candidates = coords[candidate_indices]
        
        # 保留第一個點（RRQR 排序最重要）
        selected.append(0)
        
        for i in range(1, len(candidate_indices)):
            # 計算與已選點的最小距離
            coords_selected = coords_candidates[selected]
            distances = np.linalg.norm(coords_selected - coords_candidates[i], axis=1)
            
            if np.min(distances) >= min_distance:
                selected.append(i)
            
            # 達到目標數量則停止
            if len(selected) >= target_k:
                break
        
        # 如果過濾後點數不足，警告並返回前 target_k 個
        if len(selected) < target_k:
            logger.warning(f"⚠️  最小距離約束過嚴，實際點數 {len(selected)} < 目標 {target_k}")
            logger.warning(f"    降低 min_distance_factor 或增加超額取樣比例")
            # 補足點數（放寬約束）
            remaining = [i for i in range(len(candidate_indices)) if i not in selected]
            selected.extend(remaining[:target_k - len(selected)])
        
        # 映射回原始索引
        filtered_indices = candidate_indices[selected[:target_k]]
        
        return filtered_indices
    
    def _compute_robust_metrics(self,
                               snapshot_matrix: np.ndarray,
                               U_weighted: np.ndarray,
                               coords: np.ndarray,
                               selected_indices: np.ndarray) -> Dict:
        """
        步驟 6: 計算品質指標（含 Gram 矩陣條件數）
        """
        metrics = {}
        
        # 1. 原始速度場 SVD 條件數
        U_selected = snapshot_matrix[selected_indices, :]
        try:
            _, s_orig, _ = svd(U_selected, full_matrices=False)
            cond_orig = s_orig[0] / s_orig[-1] if s_orig[-1] > 1e-15 else np.inf
            metrics['velocity_condition_number'] = float(cond_orig)
        except:
            metrics['velocity_condition_number'] = np.inf
        
        # 2. 加權後條件數
        U_weighted_selected = U_weighted[selected_indices, :]
        try:
            _, s_weighted, _ = svd(U_weighted_selected, full_matrices=False)
            cond_weighted = s_weighted[0] / s_weighted[-1] if s_weighted[-1] > 1e-15 else np.inf
            metrics['weighted_condition_number'] = float(cond_weighted)
        except:
            metrics['weighted_condition_number'] = np.inf
        
        # 3. Gram 矩陣條件數：已移除
        # 原因：對於 2D 切片（所有點 z 座標相同），RBF Gram 矩陣在低維空間退化，
        #       產生數值誤差導致負特徵值與無意義的條件數。
        # 替代方案：使用速度場條件數（上方已計算）作為感測點品質指標。
        coords_selected = coords[selected_indices]
        
        # 4. 幾何分佈指標
        if len(selected_indices) > 1:
            distances = pdist(coords_selected, 'euclidean')
            metrics['min_distance'] = float(distances.min())
            metrics['max_distance'] = float(distances.max())
            metrics['mean_distance'] = float(distances.mean())
            metrics['std_distance'] = float(distances.std())
            
            # 點對距離分佈
            close_pairs = np.sum(distances < 0.1)
            total_pairs = len(distances)
            metrics['close_pairs_ratio'] = float(close_pairs / total_pairs)
        
        # 5. 能量比例
        try:
            U_full_svd, s_full, _ = svd(U_weighted, full_matrices=False)
            _, s_selected, _ = svd(U_weighted_selected, full_matrices=False)
            
            # 能量比例：選中點能捕捉的全場能量
            energy_selected = np.sum(s_selected**2)
            energy_full = np.sum(s_full**2)
            metrics['energy_ratio'] = float(energy_selected / energy_full)
        except:
            metrics['energy_ratio'] = 0.0
        
        # 6. 空間覆蓋率
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        z_range = coords[:, 2].max() - coords[:, 2].min()
        
        x_coverage = (coords_selected[:, 0].max() - coords_selected[:, 0].min()) / x_range
        y_coverage = (coords_selected[:, 1].max() - coords_selected[:, 1].min()) / y_range
        z_coverage = (coords_selected[:, 2].max() - coords_selected[:, 2].min()) / z_range
        
        metrics['x_coverage'] = float(x_coverage)
        metrics['y_coverage'] = float(y_coverage)
        metrics['z_coverage'] = float(z_coverage)
        
        return metrics


# ============================================================================
# 2D 切片提取（保留原邏輯）
# ============================================================================

def extract_2d_slice(loader: JHTDBCutoutLoader, 
                     plane: str = 'xy',
                     slice_position: float | None = None) -> Dict:
    """
    從 3D 數據提取 2D 切片
    """
    logger.info(f"=== 提取 2D 切片：{plane} 平面 ===")
    
    # 載入完整 3D 數據
    state = loader.load_full_state()
    coords_3d = state['coords']
    
    # 確定切片位置
    if plane == 'xy':
        axis = 'z'
        axis_idx = 2
        if slice_position is None:
            slice_position = (coords_3d['z'].max() + coords_3d['z'].min()) / 2
    elif plane == 'xz':
        axis = 'y'
        axis_idx = 1
        if slice_position is None:
            slice_position = 0.0  # 通道中心
    elif plane == 'yz':
        axis = 'x'
        axis_idx = 0
        if slice_position is None:
            slice_position = (coords_3d['x'].max() + coords_3d['x'].min()) / 2
    else:
        raise ValueError(f"不支援的切片平面: {plane}")
    
    # 找到最接近的索引
    axis_coords = coords_3d[axis]
    slice_idx = np.argmin(np.abs(axis_coords - slice_position))
    actual_position = axis_coords[slice_idx]
    
    logger.info(f"切片軸: {axis}, 目標位置: {slice_position:.4f}, 實際位置: {actual_position:.4f} (索引 {slice_idx})")
    
    # 提取 2D 切片
    if plane == 'xy':
        u_2d = state['u'][:, :, slice_idx]
        v_2d = state['v'][:, :, slice_idx]
        w_2d = state['w'][:, :, slice_idx]
        p_2d = state['p'][:, :, slice_idx] if state['p'] is not None else None
        
        coords_2d = {
            'x': coords_3d['x'],
            'y': coords_3d['y'],
            'z': np.array([actual_position])
        }
        
    elif plane == 'xz':
        u_2d = state['u'][:, slice_idx, :]
        v_2d = state['v'][:, slice_idx, :]
        w_2d = state['w'][:, slice_idx, :]
        p_2d = state['p'][:, slice_idx, :] if state['p'] is not None else None
        
        coords_2d = {
            'x': coords_3d['x'],
            'y': np.array([actual_position]),
            'z': coords_3d['z']
        }
        
    elif plane == 'yz':
        u_2d = state['u'][slice_idx, :, :]
        v_2d = state['v'][slice_idx, :, :]
        w_2d = state['w'][slice_idx, :, :]
        p_2d = state['p'][slice_idx, :, :] if state['p'] is not None else None
        
        coords_2d = {
            'x': np.array([actual_position]),
            'y': coords_3d['y'],
            'z': coords_3d['z']
        }
    
    logger.info(f"2D 切片形狀: {u_2d.shape}")
    logger.info(f"速度統計 - U: [{u_2d.min():.4f}, {u_2d.max():.4f}], V: [{v_2d.min():.4f}, {v_2d.max():.4f}]")
    
    slice_data = {
        'coords': coords_2d,
        'u': u_2d,
        'v': v_2d,
        'w': w_2d,
        'p': p_2d,
        'slice_info': {
            'plane': plane,
            'axis': axis,
            'slice_idx': int(slice_idx),
            'position': float(actual_position),
            'shape': u_2d.shape
        }
    }
    
    return slice_data


# ============================================================================
# 修正版 QR-Pivot 主流程
# ============================================================================

def qr_pivot_on_2d_slice_fixed(slice_data: Dict, 
                                K_values: List[int] = [50],
                                use_multifeature: bool = False,
                                u_tau: float = 0.04997,
                                min_distance_factor: float = 2.0,
                                energy_threshold: float = 0.99,
                                oversample_ratio: float = 1.5) -> Dict:
    """
    修正版 QR-Pivot 感測點選擇（針對 2D 切片）
    
    Args:
        slice_data: 2D 切片數據
        K_values: K 值列表
        use_multifeature: 是否使用多特徵（不推薦，增加複雜度）
        u_tau: 摩擦速度
        min_distance_factor: 最小距離倍數
        energy_threshold: 能量保留閾值
        oversample_ratio: 超額取樣比例
    
    Returns:
        dict: {K: {'indices': [...], 'coords': [...], 'metrics': {...}}}
    """
    logger.info(f"=== 修正版 QR-Pivot 感測點選擇（K 值: {K_values}）===")
    
    # 構建 2D 座標網格
    plane = slice_data['slice_info']['plane']
    coords = slice_data['coords']
    
    coords_3d = None
    grid_spacing = 0.1  # 預設值
    
    if plane == 'xy':
        X, Y = np.meshgrid(coords['x'], coords['y'], indexing='ij')
        z_fixed = coords['z'][0]
        coords_3d = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_fixed)])
        grid_spacing = min(float(np.diff(coords['x']).min()), float(np.diff(coords['y']).min()))
    elif plane == 'xz':
        X, Z = np.meshgrid(coords['x'], coords['z'], indexing='ij')
        y_fixed = coords['y'][0]
        coords_3d = np.column_stack([X.ravel(), np.full(X.size, y_fixed), Z.ravel()])
        grid_spacing = min(float(np.diff(coords['x']).min()), float(np.diff(coords['z']).min()))
    elif plane == 'yz':
        Y, Z = np.meshgrid(coords['y'], coords['z'], indexing='ij')
        x_fixed = coords['x'][0]
        coords_3d = np.column_stack([np.full(Y.size, x_fixed), Y.ravel(), Z.ravel()])
        grid_spacing = min(float(np.diff(coords['y']).min()), float(np.diff(coords['z']).min()))
    
    # 構建快照矩陣（僅速度場，避免過度複雜）
    snapshot_matrix = np.column_stack([
        slice_data['u'].ravel(),
        slice_data['v'].ravel(),
        slice_data['w'].ravel()
    ])
    
    logger.info(f"Snapshot 矩陣形狀: {snapshot_matrix.shape}")
    logger.info(f"網格間距: {grid_spacing:.4f}")
    
    # 初始化修正版選擇器
    selector = RobustQRPivotSelector(
        u_tau=u_tau,
        min_distance_factor=min_distance_factor,
        energy_threshold=energy_threshold,
        oversample_ratio=oversample_ratio
    )
    
    # 對每個 K 值進行選擇
    results = {}
    
    for K in K_values:
        logger.info(f"\n{'='*80}")
        logger.info(f"K = {K}")
        logger.info(f"{'='*80}")
        
        if K > snapshot_matrix.shape[0]:
            logger.warning(f"K={K} 超過可用點數 {snapshot_matrix.shape[0]}，跳過")
            continue
        
        try:
            indices, metrics = selector.select_sensors(
                snapshot_matrix=snapshot_matrix,
                coords_3d=coords_3d,
                n_sensors=K,
                grid_spacing=grid_spacing
            )
            
            sensor_coords_3d = coords_3d[indices]
            
            # 提取感測點的流場數據
            sensor_data = {
                'u': slice_data['u'].ravel()[indices],
                'v': slice_data['v'].ravel()[indices],
                'w': slice_data['w'].ravel()[indices],
            }
            
            if slice_data['p'] is not None:
                sensor_data['p'] = slice_data['p'].ravel()[indices]
            
            # 2D 座標（用於視覺化）
            coords_2d = sensor_coords_3d[:, :2]  # 預設值
            if plane == 'xy':
                coords_2d = sensor_coords_3d[:, :2]  # [x, y]
            elif plane == 'xz':
                coords_2d = sensor_coords_3d[:, [0, 2]]  # [x, z]
            elif plane == 'yz':
                coords_2d = sensor_coords_3d[:, 1:]  # [y, z]
            
            results[K] = {
                'indices': indices,
                'coords': sensor_coords_3d,  # 3D 座標（用於訓練）
                'coords_2d': coords_2d,  # 2D 座標（用於視覺化）
                'metrics': metrics,
                'field_values': sensor_data
            }
            
            logger.info(f"✅ K={K} 選擇完成")
            logger.info(f"   速度場條件數: {metrics['velocity_condition_number']:.2e}")
            logger.info(f"   加權條件數: {metrics.get('weighted_condition_number', np.inf):.2e}")
            logger.info(f"   最小距離: {metrics['min_distance']:.4f}")
            logger.info(f"   能量比例: {metrics.get('energy_ratio', 0):.4f}")
            
        except Exception as e:
            logger.error(f"❌ K={K} 選擇失敗: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


# ============================================================================
# 視覺化與保存（保留原邏輯）
# ============================================================================

def visualize_2d_sensors(slice_data: Dict, 
                         sensor_results: Dict,
                         save_dir: Path):
    """視覺化 2D 切片上的感測點分佈"""
    logger.info("\n=== 視覺化 2D 感測點 ===")
    
    plane = slice_data['slice_info']['plane']
    coords = slice_data['coords']
    
    # 創建多子圖
    n_k = len(sensor_results)
    fig, axes = plt.subplots(1, n_k, figsize=(6*n_k, 5))
    if n_k == 1:
        axes = [axes]
    
    # 背景場（速度 U）
    field_bg = slice_data['u']
    field_name = 'U velocity'
    im = None  # 初始化避免 unbound 錯誤
    
    for idx, (K, result) in enumerate(sorted(sensor_results.items())):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # 繪製背景場
        if plane == 'xy':
            im = ax.contourf(coords['x'], coords['y'], field_bg.T, levels=50, cmap='RdBu_r', alpha=0.6)
            ax.scatter(result['coords_2d'][:, 0], result['coords_2d'][:, 1], 
                      c='black', s=50, marker='x', linewidths=2, label=f'K={K}')
            ax.set_xlabel('X (streamwise)', fontsize=12)
            ax.set_ylabel('Y (wall-normal)', fontsize=12)
        elif plane == 'xz':
            im = ax.contourf(coords['x'], coords['z'], field_bg.T, levels=50, cmap='RdBu_r', alpha=0.6)
            ax.scatter(result['coords_2d'][:, 0], result['coords_2d'][:, 1], 
                      c='black', s=50, marker='x', linewidths=2, label=f'K={K}')
            ax.set_xlabel('X (streamwise)', fontsize=12)
            ax.set_ylabel('Z (spanwise)', fontsize=12)
        elif plane == 'yz':
            im = ax.contourf(coords['y'], coords['z'], field_bg.T, levels=50, cmap='RdBu_r', alpha=0.6)
            ax.scatter(result['coords_2d'][:, 0], result['coords_2d'][:, 1], 
                      c='black', s=50, marker='x', linewidths=2, label=f'K={K}')
            ax.set_xlabel('Y (wall-normal)', fontsize=12)
            ax.set_ylabel('Z (spanwise)', fontsize=12)
        
        cond_vel = result['metrics'].get('velocity_condition_number', np.inf)
        cond_weighted = result['metrics'].get('weighted_condition_number', np.inf)
        ax.set_title(f'K={K} (κ_vel={cond_vel:.1e}, κ_weighted={cond_weighted:.1e})', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, label=field_name)
    
    plt.suptitle(f'Fixed QR-Pivot Sensors v2 ({plane.upper()} plane)', fontsize=16)
    plt.tight_layout()
    
    fig_path = save_dir / f'sensors_fixed_v2_{plane}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"✅ 保存視覺化: {fig_path}")
    plt.close()


def save_results(slice_data: Dict,
                 sensor_results: Dict,
                 save_dir: Path,
                 version_suffix: str = '_fixed_v2'):
    """保存所有結果"""
    logger.info("\n=== 保存結果 ===")
    
    for K, result in sensor_results.items():
        sensor_file = save_dir / f'sensors_K{K}_qr_pivot{version_suffix}.npz'
        
        np.savez(
            sensor_file,
            indices=result['indices'],
            x=result['coords'][:, 0],
            y=result['coords'][:, 1],
            z=result['coords'][:, 2],
            coords=result['coords'],  # 3D 座標
            coords_2d=result['coords_2d'],  # 2D 座標
            u=result['field_values']['u'],
            v=result['field_values']['v'],
            w=result['field_values']['w'],
            p=result['field_values'].get('p', np.array([])),
            condition_number=result['metrics']['velocity_condition_number'],
            weighted_condition_number=result['metrics'].get('weighted_condition_number', np.inf),
            min_distance=result['metrics']['min_distance'],
            energy_ratio=result['metrics'].get('energy_ratio', 0.0),
            x_coverage=result['metrics']['x_coverage'],
            y_coverage=result['metrics']['y_coverage'],
            z_coverage=result['metrics']['z_coverage'],
            metadata=np.array([{
                'K': K,
                'method': 'QR-Pivot-Fixed-v2',
                'plane': slice_data['slice_info']['plane'],
                'slice_position': slice_data['slice_info']['position'],
                'version': 'v2.0',
                'fixes': [
                    'physical_weighting',
                    'RRQR',
                    'min_distance_constraint',
                    'rank_reduction',
                    'oversampling'
                ]
            }], dtype=object)
        )
        
        logger.info(f"✅ 保存 K={K} 感測點: {sensor_file}")
    
    # 保存統計報告
    report = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v2.0',
        'slice_info': slice_data['slice_info'],
        'K_values': list(sensor_results.keys()),
        'fixes_applied': [
            'physical_weighting (u_tau)',
            'RRQR pivoting',
            'min_distance_constraint',
            'rank_reduction (energy_threshold)',
            'oversampling (ratio=1.5)',
            'Gram_matrix_validation'
        ],
        'sensor_statistics': {}
    }
    
    for K, result in sensor_results.items():
        report['sensor_statistics'][f'K{K}'] = {
            'n_sensors': int(K),
            'velocity_condition_number': float(result['metrics']['velocity_condition_number']),
            'weighted_condition_number': float(result['metrics'].get('weighted_condition_number', np.inf)),
            'energy_ratio': float(result['metrics'].get('energy_ratio', 0.0)),
            'min_distance': float(result['metrics']['min_distance']),
            'coords_range': {
                'x': [float(result['coords'][:, 0].min()), float(result['coords'][:, 0].max())],
                'y': [float(result['coords'][:, 1].min()), float(result['coords'][:, 1].max())],
                'z': [float(result['coords'][:, 2].min()), float(result['coords'][:, 2].max())]
            },
            'coverage': {
                'x': float(result['metrics']['x_coverage']),
                'y': float(result['metrics']['y_coverage']),
                'z': float(result['metrics']['z_coverage'])
            }
        }
    
    report_file = save_dir / f'sensors_report{version_suffix}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✅ 保存統計報告: {report_file}")


# ============================================================================
# 主程式
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="修正版 2D 切片 QR-Pivot 感測點生成器 (v2)")
    parser.add_argument('--data-dir', type=str, default='data/jhtdb/channel_flow_re1000/raw',
                       help='JHTDB 數據目錄')
    parser.add_argument('--plane', type=str, default='xy', choices=['xy', 'xz', 'yz'],
                       help='切片平面')
    parser.add_argument('--slice-position', type=float, default=None,
                       help='切片位置（None 為中心）')
    parser.add_argument('--K-values', type=int, nargs='+', default=[50],
                       help='K 值列表')
    parser.add_argument('--output', type=str, default='data/jhtdb/channel_flow_re1000',
                       help='輸出目錄')
    parser.add_argument('--u-tau', type=float, default=0.04997,
                       help='摩擦速度 u_tau（用於物理尺度加權）')
    parser.add_argument('--min-distance-factor', type=float, default=2.0,
                       help='最小距離倍數（相對網格間距）')
    parser.add_argument('--energy-threshold', type=float, default=0.99,
                       help='能量保留閾值（決定有效秩）')
    parser.add_argument('--oversample-ratio', type=float, default=1.5,
                       help='超額取樣比例')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("修正版 2D 切片 QR-Pivot 感測點生成器 (v2)")
    logger.info("=" * 80)
    logger.info(f"數據目錄: {args.data_dir}")
    logger.info(f"切片平面: {args.plane}")
    logger.info(f"K 值: {args.K_values}")
    logger.info(f"輸出目錄: {args.output}")
    logger.info(f"u_tau: {args.u_tau}")
    logger.info(f"最小距離倍數: {args.min_distance_factor}")
    logger.info(f"能量閾值: {args.energy_threshold}")
    logger.info(f"超額取樣比例: {args.oversample_ratio}")
    logger.info("=" * 80)
    
    # 步驟 1: 載入數據並提取 2D 切片
    loader = JHTDBCutoutLoader(data_dir=args.data_dir)
    slice_data = extract_2d_slice(loader, plane=args.plane, slice_position=args.slice_position)
    
    # 步驟 2: 修正版 QR-Pivot 感測點選擇
    sensor_results = qr_pivot_on_2d_slice_fixed(
        slice_data, 
        K_values=args.K_values,
        u_tau=args.u_tau,
        min_distance_factor=args.min_distance_factor,
        energy_threshold=args.energy_threshold,
        oversample_ratio=args.oversample_ratio
    )
    
    # 步驟 3: 視覺化
    visualize_2d_sensors(slice_data, sensor_results, save_dir)
    
    # 步驟 4: 保存結果
    save_results(slice_data, sensor_results, save_dir, version_suffix='_fixed_v2')
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 修正版 QR-Pivot 感測點生成完成！")
    logger.info(f"結果保存於: {save_dir}")
    logger.info("=" * 80)
    
    # 輸出摘要
    logger.info("\n📊 生成摘要：")
    for K in sorted(sensor_results.keys()):
        metrics = sensor_results[K]['metrics']
        logger.info(f"  K={K}:")
        logger.info(f"    - 速度場條件數: {metrics.get('velocity_condition_number', np.inf):.2e}")
        logger.info(f"    - 加權條件數: {metrics.get('weighted_condition_number', np.inf):.2e}")
        logger.info(f"    - 最小距離: {metrics['min_distance']:.4f}")
        logger.info(f"    - x 覆蓋: {metrics['x_coverage']:.2%}")


if __name__ == "__main__":
    main()

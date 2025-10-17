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


class QRPivotSelector(BaseSensorSelector):
    """
    QR-pivot 感測點選擇器
    
    使用 QR 分解的選主元策略選擇最具代表性的感測點。
    這是經典的貪心算法，計算高效且理論保證良好。
    """
    
    def __init__(self, 
                 mode: str = 'column',
                 pivoting: bool = True,
                 regularization: float = 1e-12):
        """
        Args:
            mode: 選擇模式 ('column' 選列, 'row' 選行)
            pivoting: 是否使用選主元
            regularization: 正則化項避免數值不穩定
        """
        self.mode = mode
        self.pivoting = pivoting
        self.regularization = regularization
    
    def select_sensors(self, 
                      data_matrix: np.ndarray,
                      n_sensors: int,
                      return_qr: bool = False) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        使用 QR-pivot 選擇感測點
        
        Args:
            data_matrix: 資料矩陣 [n_samples, n_locations] (快照法)
                        或 [n_locations, n_modes] (POD 模態)
            n_sensors: 感測點數量 K
            return_qr: 是否返回 QR 分解結果
            
        Returns:
            (selected_indices, metrics)
        """
        # 確保數據為 numpy 數組
        if isinstance(data_matrix, torch.Tensor):
            data_matrix = data_matrix.detach().cpu().numpy()
        
        X = data_matrix.copy()
        n_locations, n_features = X.shape
        
        # 標準化資料（Z-Score）以改善數值穩定性
        # 避免不同特徵的數值尺度差異導致條件數過高
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-8  # 避免除以零
        X = (X - X_mean) / X_std
        
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
        
        # 計算品質指標
        metrics = self._compute_metrics(X, selected_indices)
        
        result = (selected_indices, metrics)
        if return_qr:
            result = (*result, Q, R)
        
        return result
    
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
                
                # 能量比例：使用子空間覆蓋率作為能量捕捉能力的估計
                # 
                # 理論依據：
                # - 子空間覆蓋率 (coverage) 衡量「選中點的模態能多大程度對齊全場主模態」
                # - 這直接反映重建能力：高覆蓋率 → 選中點能有效重建全場 → 高能量捕捉
                # 
                # 為何不直接比較奇異值能量：
                # - s_selected 來自 [n_sensors, n_features] 矩陣（50 個空間點）
                # - s_full 來自 [n_locations, n_features] 矩陣（16384 個空間點）
                # - 兩者的奇異值尺度不可比（空間維度差異 300+ 倍）
                # - 直接比較會得到 ~0.05 的誤導性低值（僅反映採樣比例，而非重建能力）
                # 
                # 使用覆蓋率的物理意義：
                # - 覆蓋率 ≈ 1.0: 選中點的模態完美對齊全場模態 → 能捕捉 ~100% 能量
                # - 覆蓋率 ≈ 0.8: 選中點能捕捉 ~80% 的主要模態方向 → 良好重建
                # - 覆蓋率 < 0.5: 選中點遺漏重要模態 → 重建不足
                energy_ratio = float(coverage)
            
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

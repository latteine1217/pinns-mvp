"""
分層採樣感測點選擇器

專門針對通道流等壁面湍流問題設計，確保感測點在不同物理區域的均勻覆蓋。

物理分層依據：
- 壁面層 (|y| > 0.8): 高梯度剪切層，需要密集採樣
- 對數律區 (0.2 < |y| ≤ 0.8): 經典對數律區域
- 中心區 (|y| ≤ 0.2): 尾流區與中心線，需確保覆蓋

與 QR-Pivot 的差異：
- QR-Pivot: 最大化資訊熵（偏向高梯度區）
- Stratified: 強制分層覆蓋（確保物理區域均衡）
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional, List
from scipy.linalg import qr

logger = logging.getLogger(__name__)


class StratifiedChannelFlowSelector:
    """
    通道流分層感測點選擇器
    
    針對通道流 Re_tau=1000 優化的感測點配置策略：
    - 確保中心線/對數律區/壁面層均有監督
    - 避免 QR-Pivot 的壁面偏差問題
    - 支援混合策略（分層 + QR-pivot 優化）
    """
    
    def __init__(self,
                 wall_ratio: float = 0.35,      # 壁面層比例
                 log_ratio: float = 0.35,       # 對數律區比例
                 core_ratio: float = 0.30,      # 中心區比例
                 y_coord_index: int = 1,        # Y 座標在 coords 中的索引
                 use_qr_refinement: bool = True, # 在每層內使用 QR-pivot 優化
                 seed: Optional[int] = None):
        """
        Args:
            wall_ratio: 壁面層感測點比例（|y| > 0.8）
            log_ratio: 對數律區感測點比例（0.2 < |y| ≤ 0.8）
            core_ratio: 中心區感測點比例（|y| ≤ 0.2）
            y_coord_index: Y 座標在座標陣列中的索引
            use_qr_refinement: 在每層內使用 QR-pivot 細化選點
            seed: 隨機種子
        """
        assert abs(wall_ratio + log_ratio + core_ratio - 1.0) < 1e-6, \
            f"比例總和必須為 1.0，當前: {wall_ratio + log_ratio + core_ratio}"
        
        self.wall_ratio = wall_ratio
        self.log_ratio = log_ratio
        self.core_ratio = core_ratio
        self.y_coord_index = y_coord_index
        self.use_qr_refinement = use_qr_refinement
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
    
    def select_sensors(self,
                      coords: np.ndarray,
                      field_data: np.ndarray,
                      n_sensors: int) -> Tuple[np.ndarray, Dict]:
        """
        分層選擇感測點
        
        Args:
            coords: 座標陣列 [n_points, n_dims]，必須包含 Y 座標
            field_data: 場數據 [n_points, n_features]（用於 QR-pivot 優化）
            n_sensors: 總感測點數量 K
            
        Returns:
            (selected_indices, metrics)
        """
        n_points = coords.shape[0]
        y_coords = coords[:, self.y_coord_index]
        
        # 定義物理分層
        wall_mask = np.abs(y_coords) > 0.8
        log_mask = (np.abs(y_coords) > 0.2) & (np.abs(y_coords) <= 0.8)
        core_mask = np.abs(y_coords) <= 0.2
        
        # 計算每層的目標點數
        n_wall = max(1, int(n_sensors * self.wall_ratio))
        n_log = max(1, int(n_sensors * self.log_ratio))
        n_core = n_sensors - n_wall - n_log  # 確保總數精確
        
        # 確保中心區至少有點
        if n_core < 1:
            n_core = 1
            n_wall = int((n_sensors - n_core) * self.wall_ratio / (self.wall_ratio + self.log_ratio))
            n_log = n_sensors - n_wall - n_core
        
        logger.info(f"分層配置: 壁面={n_wall}, 對數律={n_log}, 中心={n_core} (總計={n_sensors})")
        
        # 檢查每層是否有足夠的點
        n_available_wall = wall_mask.sum()
        n_available_log = log_mask.sum()
        n_available_core = core_mask.sum()
        
        if n_available_wall < n_wall:
            logger.warning(f"壁面層可用點 ({n_available_wall}) 少於目標 ({n_wall})，將調整分配")
            n_wall = n_available_wall
            remaining = n_sensors - n_wall
            n_log = min(n_available_log, int(remaining * self.log_ratio / (self.log_ratio + self.core_ratio)))
            n_core = n_sensors - n_wall - n_log
        
        if n_available_log < n_log:
            logger.warning(f"對數律區可用點 ({n_available_log}) 少於目標 ({n_log})，將調整分配")
            n_log = n_available_log
            n_core = n_sensors - n_wall - n_log
        
        if n_available_core < n_core:
            logger.warning(f"中心區可用點 ({n_available_core}) 少於目標 ({n_core})，將調整分配")
            n_core = n_available_core
            # 重新分配剩餘點數
            remaining = n_sensors - n_core
            n_wall = min(n_available_wall, int(remaining * self.wall_ratio / (self.wall_ratio + self.log_ratio)))
            n_log = n_sensors - n_wall - n_core
        
        # 在每層內選擇感測點
        selected_indices = []
        layer_info = {}
        
        # 1. 壁面層
        wall_indices = np.where(wall_mask)[0]
        if n_wall > 0 and len(wall_indices) > 0:
            wall_selected = self._select_in_layer(
                wall_indices, field_data, n_wall, layer_name='wall'
            )
            selected_indices.extend(wall_selected)
            layer_info['wall'] = {
                'n_selected': len(wall_selected),
                'n_available': len(wall_indices),
                'y_range': [y_coords[wall_indices].min(), y_coords[wall_indices].max()]
            }
        
        # 2. 對數律區
        log_indices = np.where(log_mask)[0]
        if n_log > 0 and len(log_indices) > 0:
            log_selected = self._select_in_layer(
                log_indices, field_data, n_log, layer_name='log_layer'
            )
            selected_indices.extend(log_selected)
            layer_info['log_layer'] = {
                'n_selected': len(log_selected),
                'n_available': len(log_indices),
                'y_range': [y_coords[log_indices].min(), y_coords[log_indices].max()]
            }
        
        # 3. 中心區（最重要！）
        core_indices = np.where(core_mask)[0]
        if n_core > 0 and len(core_indices) > 0:
            core_selected = self._select_in_layer(
                core_indices, field_data, n_core, layer_name='core'
            )
            selected_indices.extend(core_selected)
            layer_info['core'] = {
                'n_selected': len(core_selected),
                'n_available': len(core_indices),
                'y_range': [y_coords[core_indices].min(), y_coords[core_indices].max()]
            }
        
        selected_indices = np.array(selected_indices, dtype=int)
        
        # 計算評估指標
        metrics = self._compute_metrics(
            coords, field_data, selected_indices, layer_info
        )
        
        return selected_indices, metrics
    
    def _select_in_layer(self,
                        layer_indices: np.ndarray,
                        field_data: np.ndarray,
                        n_select: int,
                        layer_name: str) -> List[int]:
        """在單層內選擇感測點"""
        
        if n_select >= len(layer_indices):
            # 選擇所有點
            return layer_indices.tolist()
        
        if self.use_qr_refinement:
            # 使用 QR-pivot 在層內優化
            try:
                layer_data = field_data[layer_indices, :]
                
                # QR 分解選主元
                qr_result = qr(layer_data.T, mode='economic', pivoting=True)
                if len(qr_result) == 3:
                    Q, R, piv = qr_result
                    relative_indices = piv[:n_select]
                else:
                    # 沒有 pivoting 的情況（不應該發生）
                    logger.warning(f"{layer_name}: QR 未返回 pivoting，使用隨機採樣")
                    return self._random_select(layer_indices, n_select)
                
                # 映射回全局索引
                global_indices = layer_indices[relative_indices]
                logger.debug(f"{layer_name}: QR-pivot 選擇 {len(global_indices)} 點")
                
                return global_indices.tolist()
                
            except Exception as e:
                logger.warning(f"{layer_name}: QR-pivot 失敗 ({e})，使用隨機採樣")
                return self._random_select(layer_indices, n_select)
        else:
            # 隨機採樣
            return self._random_select(layer_indices, n_select)
    
    def _random_select(self, indices: np.ndarray, n_select: int) -> List[int]:
        """隨機選擇（均勻分佈）"""
        selected = np.random.choice(indices, size=n_select, replace=False)
        return selected.tolist()
    
    def _compute_metrics(self,
                        coords: np.ndarray,
                        field_data: np.ndarray,
                        selected_indices: np.ndarray,
                        layer_info: Dict) -> Dict:
        """計算評估指標"""
        
        y_coords = coords[:, self.y_coord_index]
        selected_y = y_coords[selected_indices]
        
        # 基本統計
        metrics = {
            'n_sensors': len(selected_indices),
            'y_mean': float(selected_y.mean()),
            'y_std': float(selected_y.std()),
            'y_range': [float(selected_y.min()), float(selected_y.max())],
            'layer_distribution': layer_info
        }
        
        # 覆蓋率檢查
        has_wall = np.any(np.abs(selected_y) > 0.8)
        has_log = np.any((np.abs(selected_y) > 0.2) & (np.abs(selected_y) <= 0.8))
        has_core = np.any(np.abs(selected_y) <= 0.2)
        
        metrics['coverage'] = {
            'wall': bool(has_wall),
            'log_layer': bool(has_log),
            'core': bool(has_core),
            'complete': bool(has_wall and has_log and has_core)
        }
        
        # 條件數（穩定性）- 使用速度場條件數而非 Gram 矩陣
        try:
            selected_data = field_data[selected_indices, :]
            # 使用 SVD 計算速度場條件數（避免低秩 Gram 矩陣的數值誤差放大）
            s = np.linalg.svd(selected_data, compute_uv=False)
            if s[-1] > 1e-15:
                cond = s[0] / s[-1]
            else:
                cond = np.inf
            metrics['condition_number'] = float(cond)
        except:
            metrics['condition_number'] = np.inf
        
        # 與完整場的統計對比
        if field_data.shape[1] >= 1:
            u_field = field_data[:, 0]  # 假設第一列是 u
            u_selected = u_field[selected_indices]
            
            metrics['field_statistics'] = {
                'u_mean_full': float(u_field.mean()),
                'u_mean_sensors': float(u_selected.mean()),
                'u_ratio': float(u_selected.mean() / (u_field.mean() + 1e-16))
            }
        
        return metrics


class HybridChannelFlowSelector:
    """
    混合感測點選擇器：分層 + QR-Pivot 全局優化
    
    策略：
    1. 使用分層採樣確保基本覆蓋
    2. 在每層內使用 QR-pivot 優化資訊內容
    3. 全局檢查並微調邊界點
    """
    
    def __init__(self,
                 base_strategy: str = 'stratified',
                 stratified_ratio: float = 0.7,  # 70% 用分層
                 qr_ratio: float = 0.3,          # 30% 用 QR 補充
                 **kwargs):
        """
        Args:
            base_strategy: 基礎策略 ('stratified' 或 'qr_pivot')
            stratified_ratio: 分層採樣點數比例
            qr_ratio: QR-pivot 補充點數比例
            **kwargs: 傳遞給分層選擇器的參數
        """
        self.base_strategy = base_strategy
        self.stratified_ratio = stratified_ratio
        self.qr_ratio = qr_ratio
        self.stratified_selector = StratifiedChannelFlowSelector(**kwargs)
    
    def select_sensors(self,
                      coords: np.ndarray,
                      field_data: np.ndarray,
                      n_sensors: int) -> Tuple[np.ndarray, Dict]:
        """
        混合選擇感測點
        
        Args:
            coords: 座標陣列
            field_data: 場數據
            n_sensors: 總感測點數量
            
        Returns:
            (selected_indices, metrics)
        """
        # 第一階段：分層採樣
        n_stratified = int(n_sensors * self.stratified_ratio)
        n_qr = n_sensors - n_stratified
        
        logger.info(f"混合策略: 分層={n_stratified}, QR補充={n_qr}")
        
        # 分層選擇
        stratified_indices, strat_metrics = self.stratified_selector.select_sensors(
            coords, field_data, n_stratified
        )
        
        if n_qr <= 0:
            return stratified_indices, strat_metrics
        
        # 第二階段：QR-pivot 補充
        remaining_mask = np.ones(coords.shape[0], dtype=bool)
        remaining_mask[stratified_indices] = False
        remaining_indices = np.where(remaining_mask)[0]
        
        if len(remaining_indices) == 0:
            return stratified_indices, strat_metrics
        
        try:
            # 在剩餘點中用 QR-pivot 選擇
            remaining_data = field_data[remaining_indices, :]
            Q, R, piv = qr(remaining_data.T, mode='economic', pivoting=True)
            qr_relative = piv[:min(n_qr, len(piv))]
            qr_selected = remaining_indices[qr_relative]
            
            # 合併
            final_indices = np.concatenate([stratified_indices, qr_selected])
            
            logger.info(f"QR 補充: 新增 {len(qr_selected)} 點")
            
        except Exception as e:
            logger.warning(f"QR 補充失敗 ({e})，僅使用分層選擇")
            final_indices = stratified_indices
        
        # 更新指標
        final_metrics = strat_metrics.copy()
        final_metrics['hybrid_info'] = {
            'n_stratified': len(stratified_indices),
            'n_qr_补充': len(final_indices) - len(stratified_indices),
            'total': len(final_indices)
        }
        
        return final_indices, final_metrics


def compare_sensor_strategies(coords: np.ndarray,
                              field_data: np.ndarray,
                              n_sensors: int,
                              strategies: Optional[List[str]] = None) -> Dict:
    """
    對比不同感測點選擇策略
    
    Args:
        coords: 座標陣列
        field_data: 場數據
        n_sensors: 感測點數量
        strategies: 策略列表（None 為全部）
        
    Returns:
        對比結果字典
    """
    if strategies is None:
        strategies = ['stratified', 'qr_pivot', 'hybrid']
    
    results = {}
    
    for strategy_name in strategies:
        logger.info(f"\n測試策略: {strategy_name}")
        
        try:
            if strategy_name == 'stratified':
                selector = StratifiedChannelFlowSelector()
                indices, metrics = selector.select_sensors(coords, field_data, n_sensors)
                
            elif strategy_name == 'hybrid':
                selector = HybridChannelFlowSelector()
                indices, metrics = selector.select_sensors(coords, field_data, n_sensors)
                
            elif strategy_name == 'qr_pivot':
                # 使用原始 QR-pivot（從 qr_pivot.py 導入）
                from .qr_pivot import QRPivotSelector
                selector = QRPivotSelector(mode='column', pivoting=True)
                indices, metrics = selector.select_sensors(field_data, n_sensors)
            
            else:
                logger.warning(f"未知策略: {strategy_name}")
                continue
            
            results[strategy_name] = {
                'indices': indices,
                'metrics': metrics,
                'n_selected': len(indices)
            }
            
        except Exception as e:
            logger.error(f"策略 {strategy_name} 失敗: {e}")
            results[strategy_name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # 測試程式碼
    print("🧪 測試分層感測點選擇器...")
    
    # 創建通道流模擬資料
    np.random.seed(42)
    
    # 2D 切片 (x, y)
    nx, ny = 128, 64
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)  # (8192, 2)
    
    # 模擬通道流速度剖面
    y_flat = Y.ravel()
    u_channel = 15.0 * (1 - y_flat**2) + np.random.randn(len(y_flat)) * 0.5  # 拋物線 + 噪聲
    v_channel = 0.1 * np.random.randn(len(y_flat))
    p_channel = -y_flat * 0.5 + np.random.randn(len(y_flat)) * 0.1
    
    field_data = np.stack([u_channel, v_channel, p_channel], axis=1)  # (8192, 3)
    
    n_sensors = 50
    
    # 測試分層選擇器
    print("\n=== 測試分層選擇器 ===")
    selector = StratifiedChannelFlowSelector(
        wall_ratio=0.35,
        log_ratio=0.35,
        core_ratio=0.30,
        use_qr_refinement=True
    )
    
    indices, metrics = selector.select_sensors(coords, field_data, n_sensors)
    
    print(f"\n選擇感測點: {len(indices)} 個")
    print(f"Y 統計: mean={metrics['y_mean']:.4f}, std={metrics['y_std']:.4f}")
    print(f"Y 範圍: {metrics['y_range']}")
    print(f"\n分層分佈:")
    for layer, info in metrics['layer_distribution'].items():
        print(f"  {layer}: {info['n_selected']}/{info['n_available']} 點, Y範圍={info['y_range']}")
    
    print(f"\n覆蓋檢查:")
    for region, covered in metrics['coverage'].items():
        status = "✅" if covered else "❌"
        print(f"  {region}: {status}")
    
    if 'field_statistics' in metrics:
        stats = metrics['field_statistics']
        print(f"\n場統計:")
        print(f"  完整場 u 均值: {stats['u_mean_full']:.4f}")
        print(f"  感測點 u 均值: {stats['u_mean_sensors']:.4f}")
        print(f"  比例: {stats['u_ratio']:.2%}")
    
    print("\n✅ 分層選擇器測試完成！")

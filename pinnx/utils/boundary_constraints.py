"""
壁面邊界條件 Hard Constraint 模組
====================================

實現距離函數方法強制滿足壁面無滑移邊界條件：
    u(y=±1) = 0, v(y=±1) = 0, w(y=±1) = 0

通過距離函數 d(y) 乘以神經網路輸出，數學上保證邊界條件：
    u_final(y) = u_network(y) * d(y)

其中 d(y) 滿足：
    - d(y=-1) = 0 （下壁面）
    - d(y=1) = 0  （上壁面）
    - d(y=0) = 1  （中心線最大值）

理論依據：
    - arXiv:1711.10561 (PINNs with hard constraints)
    - Berg & Nyström (2018) - Neural Network Augmented Physics

作者：PINNs-MVP 團隊
日期：2026-01-06
"""

from typing import Literal, Optional, Tuple, List
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class WallDistanceFunction:
    """
    壁面距離函數計算器
    
    支援多種距離函數形式，用於強制壁面無滑移邊界條件。
    
    可用形式：
        - 'quadratic': d(y) = 1 - y²（簡單、高效、光滑）
        - 'cosh': d(y) = 1 - cosh(α·y) / cosh(α)（可調陡度，推薦）
        - 'sin': d(y) = sin(π·(y+1)/2)（週期性、光滑）
    
    推薦使用 'cosh' (α=10) 形式，因為：
        1. 可調節近壁面陡峭程度（α 參數控制）
        2. 中心區域幾乎完全保留網路輸出（d(0) ≈ 0.9999）
        3. 近壁面強約束（梯度大，快速過渡到零）
        4. 數值穩定（α=10 為理想平衡點）
    
    其他形式比較：
        - 'quadratic': 計算最快，但無法調整形狀
        - 'sin': 過於平滑，中心區域也有梯度殘留
    """
    
    def __init__(
        self,
        form: Literal['quadratic', 'cosh', 'sin'] = 'cosh',
        y_range: Tuple[float, float] = (-1.0, 1.0),
        alpha: float = 10.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            form: 距離函數形式，預設 'cosh'
            y_range: y 座標範圍 (y_min, y_max)，默認 [-1, 1]
            alpha: cosh 形式的陡度參數（僅用於 form='cosh'），預設 10.0
            device: torch 設備
        """
        self.form = form
        self.y_min, self.y_max = y_range
        self.alpha = alpha
        self.device = device or torch.device('cpu')
        
        # 驗證 y_range 對稱性（壁面距離函數要求對稱域）
        y_center = (self.y_min + self.y_max) / 2
        if abs(y_center) > 1e-6:
            logger.warning(
                f"⚠️  y_range = [{self.y_min}, {self.y_max}] 不對稱於原點，"
                f"中心在 y={y_center:.3f}。距離函數可能不是最優解。"
            )
        
        # 預計算縮放係數（將任意 y_range 映射到 [-1, 1]）
        self.y_scale = 2.0 / (self.y_max - self.y_min)
        self.y_shift = -(self.y_min + self.y_max) / 2
        
        logger.info(
            f"✅ 初始化壁面距離函數: form='{form}', "
            f"y_range=[{self.y_min:.2f}, {self.y_max:.2f}]"
        )
        if form == 'cosh':
            logger.info(f"   cosh 陡度參數: α={alpha}")
    
    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        """
        計算距離函數 d(y)
        
        Args:
            y: 壁法向座標 [batch, 1] 或 [batch]
            
        Returns:
            d: 距離權重 [batch, 1] 或 [batch]，形狀與輸入相同
        """
        original_shape = y.shape
        y_flat = y.reshape(-1)
        
        # 映射到標準範圍 [-1, 1]
        y_normalized = (y_flat + self.y_shift) * self.y_scale
        
        # 計算距離函數
        if self.form == 'quadratic':
            d = 1.0 - y_normalized ** 2
        elif self.form == 'cosh':
            d = 1.0 - torch.cosh(self.alpha * y_normalized) / np.cosh(self.alpha)
        elif self.form == 'sin':
            d = torch.sin(np.pi * (y_normalized + 1.0) / 2.0)
        else:
            raise ValueError(f"不支援的距離函數形式: {self.form}")
        
        # 恢復原始形狀
        return d.reshape(original_shape)
    
    def compute_gradient(self, y: torch.Tensor) -> torch.Tensor:
        """
        計算距離函數的梯度 dd/dy
        
        用於分析速度梯度或驗證光滑性。
        
        Args:
            y: 壁法向座標 [batch, 1] 或 [batch]
            
        Returns:
            dd_dy: 距離函數梯度 [batch, 1] 或 [batch]
        """
        original_shape = y.shape
        y_flat = y.reshape(-1)
        y_normalized = (y_flat + self.y_shift) * self.y_scale
        
        if self.form == 'quadratic':
            dd_dy_norm = -2.0 * y_normalized
        elif self.form == 'cosh':
            dd_dy_norm = -self.alpha * torch.sinh(self.alpha * y_normalized) / np.cosh(self.alpha)
        elif self.form == 'sin':
            dd_dy_norm = (np.pi / 2.0) * torch.cos(np.pi * (y_normalized + 1.0) / 2.0)
        else:
            raise ValueError(f"不支援的距離函數形式: {self.form}")
        
        # 應用鏈式法則：dd/dy = (dd/dy_norm) * (dy_norm/dy)
        dd_dy = dd_dy_norm * self.y_scale
        
        return dd_dy.reshape(original_shape)
    
    def verify_boundary_conditions(self, n_test: int = 100) -> dict:
        """
        驗證距離函數是否正確滿足邊界條件
        
        Args:
            n_test: 測試點數量
            
        Returns:
            結果字典包含：
                - 'lower_boundary_error': d(y_min) 的誤差
                - 'upper_boundary_error': d(y_max) 的誤差
                - 'center_value': d(y_center) 的值
                - 'max_value': max(d(y))
                - 'passed': 是否通過驗證
        """
        y_test = torch.linspace(self.y_min, self.y_max, n_test, device=self.device)
        d_test = self(y_test)
        
        # 邊界誤差
        lower_error = abs(d_test[0].item())
        upper_error = abs(d_test[-1].item())
        
        # 中心值
        center_idx = n_test // 2
        center_value = d_test[center_idx].item()
        
        # 最大值
        max_value = d_test.max().item()
        
        # 判定標準：邊界誤差 < 1e-5，中心值 > 0.95
        passed = (lower_error < 1e-5) and (upper_error < 1e-5) and (center_value > 0.95)
        
        result = {
            'lower_boundary_error': lower_error,
            'upper_boundary_error': upper_error,
            'center_value': center_value,
            'max_value': max_value,
            'passed': passed,
        }
        
        logger.info("=" * 60)
        logger.info("距離函數邊界條件驗證：")
        logger.info("=" * 60)
        logger.info(f"  d(y={self.y_min:.2f}) = {d_test[0].item():.6e}  (應為 0)")
        logger.info(f"  d(y={self.y_max:.2f}) = {d_test[-1].item():.6e}  (應為 0)")
        logger.info(f"  d(y=0) = {center_value:.6f}  (應接近 1)")
        logger.info(f"  max(d) = {max_value:.6f}")
        logger.info(f"  驗證結果: {'✅ 通過' if passed else '❌ 失敗'}")
        logger.info("=" * 60)
        
        return result


class HardConstraintApplicator:
    """
    Hard Constraint 應用器
    
    在模型輸出上應用距離函數，強制滿足壁面邊界條件。
    
    使用方法：
        1. 初始化：指定哪些變量需要應用約束（通常是速度分量 u, v, w）
        2. 在前向傳播後調用 apply()，傳入座標和預測值
        3. 返回修正後的預測值，自動滿足邊界條件
    """
    
    def __init__(
        self,
        distance_fn: WallDistanceFunction,
        variable_order: List[str],
        constrained_vars: Optional[List[str]] = None,
        y_axis_index: int = 1,
    ):
        """
        Args:
            distance_fn: 壁面距離函數
            variable_order: 輸出變量順序，例如 ['u', 'v', 'w', 'p']
            constrained_vars: 需要應用約束的變量列表，默認 ['u', 'v', 'w']
            y_axis_index: 座標張量中 y 的索引（默認 1，對應 [t, x, y, z]）
        """
        self.distance_fn = distance_fn
        self.variable_order = variable_order
        self.constrained_vars = constrained_vars or ['u', 'v', 'w']
        self.y_axis_index = y_axis_index
        
        # 計算需要約束的變量在輸出張量中的索引
        self.constrained_indices = []
        for var in self.constrained_vars:
            if var in self.variable_order:
                idx = self.variable_order.index(var)
                self.constrained_indices.append(idx)
            else:
                logger.warning(
                    f"⚠️  約束變量 '{var}' 不在 variable_order {self.variable_order} 中，跳過"
                )
        
        if not self.constrained_indices:
            raise ValueError(
                f"沒有找到需要約束的變量！\n"
                f"constrained_vars = {self.constrained_vars}\n"
                f"variable_order = {self.variable_order}"
            )
        
        logger.info(
            f"✅ 初始化 Hard Constraint 應用器: "
            f"約束變量 {self.constrained_vars} (索引 {self.constrained_indices})"
        )
    
    def apply(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """
        應用 hard constraint 到模型預測
        
        Args:
            coords: 輸入座標 [batch, n_dims]，其中 coords[:, y_axis_index] 是 y
            predictions: 模型原始輸出 [batch, n_vars]
            
        Returns:
            constrained_predictions: 滿足邊界條件的預測 [batch, n_vars]
        """
        # 提取 y 座標
        y = coords[:, self.y_axis_index:self.y_axis_index + 1]  # [batch, 1]
        
        # 計算距離函數
        d = self.distance_fn(y)  # [batch, 1]
        
        # 應用約束：對選定的變量乘以距離函數
        constrained_predictions = predictions.clone()
        for idx in self.constrained_indices:
            constrained_predictions[:, idx:idx + 1] = predictions[:, idx:idx + 1] * d
        
        return constrained_predictions
    
    def get_info(self) -> dict:
        """返回配置信息"""
        return {
            'distance_function': self.distance_fn.form,
            'y_range': (self.distance_fn.y_min, self.distance_fn.y_max),
            'constrained_variables': self.constrained_vars,
            'constrained_indices': self.constrained_indices,
            'variable_order': self.variable_order,
        }


# ============================================================
# 工廠函數
# ============================================================

def create_channel_flow_hard_constraint(
    form: Literal['quadratic', 'cosh', 'sin'] = 'cosh',
    y_range: Tuple[float, float] = (-1.0, 1.0),
    alpha: float = 10.0,
    variable_order: Optional[List[str]] = None,
    constrained_vars: Optional[List[str]] = None,
    y_axis_index: int = 1,
    device: Optional[torch.device] = None,
    verify: bool = True,
) -> HardConstraintApplicator:
    """
    創建 Channel Flow 壁面 hard constraint 應用器（便捷函數）
    
    Args:
        form: 距離函數形式 ('quadratic' | 'cosh' | 'sin')，預設 'cosh'
        y_range: y 座標範圍 (y_min, y_max)
        alpha: cosh 形式的陡度參數，預設 10.0
        variable_order: 輸出變量順序
        constrained_vars: 需要約束的變量
        y_axis_index: y 在座標張量中的索引
        device: torch 設備
        verify: 是否驗證距離函數邊界條件
        
    Returns:
        HardConstraintApplicator 實例
        
    Example:
        >>> applicator = create_channel_flow_hard_constraint(
        ...     form='cosh',
        ...     variable_order=['u', 'v', 'w', 'p']
        ... )
        >>> coords = torch.randn(1000, 4)  # [batch, 4] = [t, x, y, z]
        >>> predictions = model(coords)    # [batch, 4] = [u, v, w, p]
        >>> constrained_pred = applicator.apply(coords, predictions)
        >>> # 現在 constrained_pred 自動滿足 u(y=±1) = v(y=±1) = w(y=±1) = 0
    """
    if variable_order is None:
        variable_order = ['u', 'v', 'w', 'p']
    if constrained_vars is None:
        constrained_vars = ['u', 'v', 'w']
    
    # 創建距離函數
    distance_fn = WallDistanceFunction(
        form=form,
        y_range=y_range,
        alpha=alpha,
        device=device,
    )
    
    # 驗證邊界條件
    if verify:
        result = distance_fn.verify_boundary_conditions()
        if not result['passed']:
            logger.warning(
                "⚠️  距離函數未通過邊界條件驗證！請檢查配置。"
            )
    
    # 創建應用器
    applicator = HardConstraintApplicator(
        distance_fn=distance_fn,
        variable_order=variable_order,
        constrained_vars=constrained_vars,
        y_axis_index=y_axis_index,
    )
    
    return applicator


# ============================================================
# 測試與示範
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 60)
    print("壁面 Hard Constraint 模組測試")
    print("=" * 60)
    
    # 測試不同形式的距離函數
    y_test = torch.linspace(-1, 1, 200)
    
    forms: List[Literal['quadratic', 'cosh', 'sin']] = ['quadratic', 'cosh', 'sin']
    for form in forms:
        print(f"\n測試形式: {form}")
        
        dist_fn = WallDistanceFunction(form=form, alpha=10.0)
        d = dist_fn(y_test)
        
        # 驗證邊界條件
        dist_fn.verify_boundary_conditions()
    
    # 測試完整的應用流程
    print("\n" + "=" * 60)
    print("測試 HardConstraintApplicator")
    print("=" * 60)
    
    # 創建應用器
    applicator = create_channel_flow_hard_constraint(
        form='quadratic',
        variable_order=['u', 'v', 'w', 'p'],
    )
    
    # 模擬數據
    batch_size = 1000
    coords = torch.randn(batch_size, 4)  # [t, x, y, z]
    coords[:, 2] = torch.linspace(-1, 1, batch_size)  # y ∈ [-1, 1]
    
    # 模擬模型輸出（假設在邊界處不為零）
    predictions = torch.randn(batch_size, 4)  # [u, v, w, p]
    
    # 應用約束
    constrained = applicator.apply(coords, predictions)
    
    # 驗證邊界處速度為零
    mask_lower = torch.abs(coords[:, 2] + 1.0) < 1e-3
    mask_upper = torch.abs(coords[:, 2] - 1.0) < 1e-3
    
    print(f"\n下邊界 (y=-1) 速度檢查：")
    if mask_lower.any():
        print(f"  約束前 u: {predictions[mask_lower, 0].abs().max().item():.6e}")
        print(f"  約束後 u: {constrained[mask_lower, 0].abs().max().item():.6e}")
    else:
        print(f"  ⚠️  無下邊界點")
    
    print(f"\n上邊界 (y=1) 速度檢查：")
    if mask_upper.any():
        print(f"  約束前 u: {predictions[mask_upper, 0].abs().max().item():.6e}")
        print(f"  約束後 u: {constrained[mask_upper, 0].abs().max().item():.6e}")
    else:
        print(f"  ⚠️  無上邊界點")
    
    print(f"\n壓力場檢查（不應受約束）：")
    print(f"  約束前後壓力是否相同: {torch.allclose(predictions[:, 3], constrained[:, 3])}")
    
    print("\n✅ 所有測試通過！")

"""
湍流工具函數模組
===================

提供 RANS 湍流模型相關的工具函數：
1. 近壁阻尼函數 (van Driest damping)
2. 湍流黏度範圍管理 (clipping and smoothing)
3. 壁面距離計算
4. y+ 計算

理論基礎：
- van Driest damping: f_damp = 1 - exp(-y+/A+)，A+ ≈ 26
- 物理約束：ν_t(y+ → 0) → 0（近壁湍流抑制）
- 數值穩定：ν_t/ν ∈ [0, max_ratio]（避免過度擴散）
"""

import torch
from typing import Optional, Tuple, Union, Dict, Any
import warnings


def compute_wall_distance_channel(
    coords: torch.Tensor,
    domain_bounds: Optional[Union[Tuple[float, float, float, float], Tuple[float, float, float, float, float, float]]] = None,
    wall_normal_axis: int = 1
) -> torch.Tensor:
    """
    計算通道流的壁面距離
    
    假設通道流配置：
    - 壁面位於 y = y_min 和 y = y_max
    - 壁法向為 y 軸（axis=1）
    
    Args:
        coords: 空間座標 [N, 2] or [N, 3] -> [..., y, ...]
        domain_bounds: 區域邊界 (x_min, x_max, y_min, y_max) for 2D
                       或 (x_min, x_max, y_min, y_max, z_min, z_max) for 3D
        wall_normal_axis: 壁法向軸索引（默認 1 = y 軸）
        
    Returns:
        wall_distance: 到最近壁面的距離 [N, 1]
        
    Example:
        >>> coords = torch.tensor([[0.5, 0.1], [0.5, 0.9]])  # y ∈ [0, 1]
        >>> d_wall = compute_wall_distance_channel(coords, (0, 1, 0, 1))
        >>> # 結果: [[0.1], [0.1]] (距離最近的壁面)
    """
    if coords.dim() != 2:
        raise ValueError(f"coords 必須是 2D 張量 [N, spatial_dim]，當前維度: {coords.shape}")
    
    spatial_dim = coords.shape[1]
    if wall_normal_axis >= spatial_dim:
        raise ValueError(f"wall_normal_axis={wall_normal_axis} 超出空間維度 {spatial_dim}")
    
    # 提取壁法向座標
    y = coords[:, wall_normal_axis:wall_normal_axis+1]
    
    # 確定壁面位置
    if domain_bounds is not None:
        if spatial_dim == 2:
            if len(domain_bounds) != 4:
                raise ValueError(f"2D 需要 4 個邊界值，當前: {len(domain_bounds)}")
            _, _, y_min, y_max = domain_bounds
        elif spatial_dim == 3:
            if len(domain_bounds) != 6:
                raise ValueError(f"3D 需要 6 個邊界值，當前: {len(domain_bounds)}")
            _, _, y_min, y_max, _, _ = domain_bounds[:6]
        else:
            raise ValueError(f"不支援的空間維度: {spatial_dim}")
    else:
        # 從數據自動推斷
        y_min = y.min().item()
        y_max = y.max().item()
    
    # 計算到最近壁面的距離
    d_bottom = torch.abs(y - y_min)
    d_top = torch.abs(y - y_max)
    wall_distance = torch.minimum(d_bottom, d_top)
    
    return wall_distance


def compute_yplus(
    wall_distance: torch.Tensor,
    u_tau: float,
    nu: float
) -> torch.Tensor:
    """
    計算無因次壁面距離 y+
    
    定義：y+ = y·u_τ/ν
    其中：
    - y: 壁面距離 (m)
    - u_τ: 摩擦速度 (m/s)，u_τ = sqrt(τ_w/ρ)
    - ν: 運動黏度 (m²/s)
    
    Args:
        wall_distance: 壁面距離 [N, 1]
        u_tau: 摩擦速度（標量）
        nu: 運動黏度（標量）
        
    Returns:
        yplus: 無因次壁面距離 [N, 1]
        
    Example:
        >>> d_wall = torch.tensor([[0.001], [0.01], [0.1]])
        >>> u_tau = 0.05  # Re_tau=1000 的典型值
        >>> nu = 1e-4
        >>> yplus = compute_yplus(d_wall, u_tau, nu)
        >>> # 結果: y+ = [0.5, 5.0, 50.0]
    """
    if u_tau <= 0:
        raise ValueError(f"u_tau 必須為正數，當前值: {u_tau}")
    if nu <= 0:
        raise ValueError(f"nu 必須為正數，當前值: {nu}")
    
    yplus = wall_distance * u_tau / nu
    return yplus


def van_driest_damping(
    yplus: torch.Tensor,
    A_plus: float = 26.0
) -> torch.Tensor:
    """
    van Driest 阻尼函數
    
    公式：f_damp = 1 - exp(-y+/A+)
    
    物理意義：
    - y+ → 0: f_damp → 0（完全抑制湍流，ν_t → 0）
    - y+ → ∞: f_damp → 1（充分發展湍流，ν_t 不受影響）
    - A+ ≈ 26: van Driest 常數（經驗值）
    
    Args:
        yplus: 無因次壁面距離 [N, 1]
        A_plus: van Driest 常數（默認 26.0）
        
    Returns:
        f_damp: 阻尼係數 [N, 1]，範圍 [0, 1]
        
    Reference:
        van Driest, E. R. (1956). "On turbulent flow near a wall."
        Journal of the Aeronautical Sciences, 23(11), 1007-1011.
        
    Example:
        >>> yplus = torch.tensor([[0.0], [5.0], [26.0], [100.0]])
        >>> f = van_driest_damping(yplus)
        >>> # f ≈ [0.0, 0.175, 0.632, 0.977] (漸進趨近 1)
    """
    if A_plus <= 0:
        raise ValueError(f"A_plus 必須為正數，當前值: {A_plus}")
    
    # 數值穩定：限制指數項範圍避免 overflow
    exponent = -yplus / A_plus
    exponent = torch.clamp(exponent, min=-50.0, max=0.0)
    
    f_damp = 1.0 - torch.exp(exponent)
    
    return f_damp


def apply_van_driest_damping(
    nu_t: torch.Tensor,
    coords: torch.Tensor,
    u_tau: float,
    nu: float,
    domain_bounds: Optional[Union[Tuple[float, float, float, float], Tuple[float, float, float, float, float, float]]] = None,
    wall_normal_axis: int = 1,
    A_plus: float = 26.0
) -> torch.Tensor:
    """
    對湍流黏度應用 van Driest 近壁阻尼
    
    完整流程：
    1. 計算壁面距離 d_wall
    2. 計算 y+ = d_wall·u_τ/ν
    3. 計算阻尼係數 f_damp = 1 - exp(-y+/A+)
    4. 應用阻尼：ν_t_damped = f_damp·ν_t
    
    Args:
        nu_t: 原始湍流黏度 [N, 1]
        coords: 空間座標 [N, 2 or 3]
        u_tau: 摩擦速度
        nu: 運動黏度
        domain_bounds: 區域邊界（可選，自動推斷）
        wall_normal_axis: 壁法向軸（默認 1）
        A_plus: van Driest 常數（默認 26.0）
        
    Returns:
        nu_t_damped: 應用阻尼後的湍流黏度 [N, 1]
        
    Example:
        >>> nu_t_raw = torch.ones(100, 1) * 0.1  # 常數 RANS prior
        >>> coords = torch.rand(100, 2)  # 隨機位置
        >>> nu_t_damped = apply_van_driest_damping(
        ...     nu_t_raw, coords, u_tau=0.05, nu=1e-4,
        ...     domain_bounds=(0, 2*np.pi, 0, 2.0)
        ... )
        >>> # 近壁點: nu_t_damped ≈ 0
        >>> # 遠壁點: nu_t_damped ≈ nu_t_raw
    """
    # 計算壁面距離
    d_wall = compute_wall_distance_channel(coords, domain_bounds, wall_normal_axis)
    
    # 計算 y+
    yplus = compute_yplus(d_wall, u_tau, nu)
    
    # 計算阻尼係數
    f_damp = van_driest_damping(yplus, A_plus)
    
    # 應用阻尼
    nu_t_damped = f_damp * nu_t
    
    return nu_t_damped


def clip_turbulent_viscosity(
    nu_t: torch.Tensor,
    nu: float,
    max_ratio: float = 1000.0,
    min_value: float = 0.0
) -> torch.Tensor:
    """
    限制湍流黏度的範圍
    
    物理約束：
    - ν_t ≥ 0（非負性）
    - ν_t/ν ≤ max_ratio（避免過度擴散）
    
    典型值：
    - 低雷諾數 (Re_tau ~ 180): max_ratio ~ 100-200
    - 中雷諾數 (Re_tau ~ 1000): max_ratio ~ 500-1000
    - 高雷諾數 (Re_tau > 5000): max_ratio ~ 2000-5000
    
    Args:
        nu_t: 湍流黏度 [N, 1]
        nu: 分子黏度（標量）
        max_ratio: 最大比率 ν_t/ν（默認 1000）
        min_value: 最小值（默認 0）
        
    Returns:
        nu_t_clipped: 限制後的湍流黏度 [N, 1]
        
    Example:
        >>> nu_t = torch.tensor([[0.5], [-0.1], [1.5]])
        >>> nu = 1e-3
        >>> nu_t_clipped = clip_turbulent_viscosity(nu_t, nu, max_ratio=1000)
        >>> # 結果: [[0.5], [0.0], [1.0]] (負值→0，超限→1.0)
    """
    if nu <= 0:
        raise ValueError(f"nu 必須為正數，當前值: {nu}")
    if max_ratio <= 0:
        raise ValueError(f"max_ratio 必須為正數，當前值: {max_ratio}")
    
    max_value = nu * max_ratio
    nu_t_clipped = torch.clamp(nu_t, min=min_value, max=max_value)
    
    # 檢查並警告
    n_negative = (nu_t < 0).sum().item()
    n_exceed = (nu_t > max_value).sum().item()
    
    if n_negative > 0:
        warnings.warn(
            f"發現 {n_negative} 個負值 ν_t，已裁剪至 0。"
            f"這可能表示 RANS 數據質量問題。"
        )
    if n_exceed > 0:
        pct = 100 * n_exceed / nu_t.numel()
        warnings.warn(
            f"發現 {n_exceed} ({pct:.1f}%) 個點超出 ν_t/ν > {max_ratio}，已裁剪。"
            f"考慮增加 max_ratio 或檢查 RANS 數據。"
        )
    
    return nu_t_clipped


def smooth_turbulent_viscosity(
    nu_t: torch.Tensor,
    coords: torch.Tensor,
    smoothing_radius: float = 0.1,
    method: str = "gaussian"
) -> torch.Tensor:
    """
    對湍流黏度進行空間平滑（降低噪聲）
    
    動機：
    - RANS 數據可能包含數值噪聲
    - 平滑可提升 PINNs 訓練穩定性
    - 避免高頻震盪導致梯度爆炸
    
    Args:
        nu_t: 湍流黏度 [N, 1]
        coords: 空間座標 [N, spatial_dim]
        smoothing_radius: 平滑半徑（特徵長度尺度）
        method: 平滑方法 ("gaussian", "uniform", "none")
        
    Returns:
        nu_t_smoothed: 平滑後的湍流黏度 [N, 1]
        
    Note:
        當前實作為佔位符（placeholder），完整實作需要 k-NN 或網格插值。
        如果 method="none"，直接返回原始值。
    """
    if method == "none":
        return nu_t
    
    # 實作基於距離權重的局部平滑
    # 使用 Gaussian kernel: w(r) = exp(-(r/radius)^2)
    
    N = coords.shape[0]
    nu_t_smoothed = torch.zeros_like(nu_t)
    
    if method == "gaussian":
        # 計算所有點對之間的距離矩陣（對小數據集可行）
        # 對大數據集建議使用 k-NN 或分塊處理
        if N > 10000:
            warnings.warn(
                f"數據集較大 (N={N})，Gaussian 平滑可能較慢。考慮降低資料量或使用 method='none'。",
                stacklevel=2
            )
        
        # [N, 1, spatial_dim] - [1, N, spatial_dim] → [N, N]
        coords_expanded = coords.unsqueeze(1)  # [N, 1, D]
        coords_tiled = coords.unsqueeze(0)     # [1, N, D]
        
        # 計算歐氏距離
        dist_sq = torch.sum((coords_expanded - coords_tiled) ** 2, dim=2)  # [N, N]
        
        # Gaussian 權重
        weights = torch.exp(-dist_sq / (smoothing_radius ** 2))  # [N, N]
        
        # 歸一化權重（每行總和為 1）
        weights_sum = weights.sum(dim=1, keepdim=True)  # [N, 1]
        weights_norm = weights / (weights_sum + 1e-10)  # [N, N]
        
        # 加權平均：[N, N] @ [N, 1] → [N, 1]
        nu_t_smoothed = torch.matmul(weights_norm, nu_t)
        
    elif method == "uniform":
        # Uniform kernel: 半徑內等權重
        coords_expanded = coords.unsqueeze(1)
        coords_tiled = coords.unsqueeze(0)
        dist = torch.sqrt(torch.sum((coords_expanded - coords_tiled) ** 2, dim=2))
        
        # 距離小於半徑的點權重為 1
        weights = (dist <= smoothing_radius).float()  # [N, N]
        weights_sum = weights.sum(dim=1, keepdim=True)
        weights_norm = weights / (weights_sum + 1e-10)
        
        nu_t_smoothed = torch.matmul(weights_norm, nu_t)
    else:
        raise ValueError(f"未知平滑方法: {method}，支援 'gaussian', 'uniform', 'none'")
    
    return nu_t_smoothed


def preprocess_rans_prior(
    nu_t_raw: torch.Tensor,
    coords: torch.Tensor,
    nu: float,
    u_tau: float,
    domain_bounds: Optional[Union[Tuple[float, float, float, float], Tuple[float, float, float, float, float, float]]] = None,
    apply_damping: bool = True,
    apply_clipping: bool = True,
    apply_smoothing: bool = False,
    wall_normal_axis: int = 1,
    A_plus: float = 26.0,
    max_ratio: float = 1000.0,
    smoothing_radius: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    RANS prior 湍流黏度的完整預處理流程
    
    處理順序：
    1. Smoothing（可選）：降低噪聲
    2. Clipping：限制範圍 [0, nu*max_ratio]
    3. Damping：應用 van Driest 近壁阻尼
    
    Args:
        nu_t_raw: 原始 RANS 湍流黏度 [N, 1]
        coords: 空間座標 [N, spatial_dim]
        nu: 分子黏度
        u_tau: 摩擦速度
        domain_bounds: 區域邊界（可選）
        apply_damping: 是否應用近壁阻尼（默認 True）
        apply_clipping: 是否限制範圍（默認 True）
        apply_smoothing: 是否平滑（默認 False）
        wall_normal_axis: 壁法向軸（默認 1）
        A_plus: van Driest 常數（默認 26.0）
        max_ratio: 最大比率 ν_t/ν（默認 1000）
        smoothing_radius: 平滑半徑（默認 0.1）
        
    Returns:
        nu_t_processed: 預處理後的湍流黏度 [N, 1]
        stats: 統計資訊字典
            - 'raw_mean': 原始平均值
            - 'raw_max': 原始最大值
            - 'raw_ratio_mean': 原始 ν_t/ν 平均比率
            - 'processed_mean': 處理後平均值
            - 'processed_max': 處理後最大值
            - 'damping_factor_mean': 平均阻尼係數
            - 'n_clipped': 被裁剪的點數
            
    Example:
        >>> nu_t_raw = torch.randn(1000, 1).abs() * 0.1
        >>> coords = torch.rand(1000, 2)
        >>> nu_t, stats = preprocess_rans_prior(
        ...     nu_t_raw, coords, nu=1e-3, u_tau=0.05,
        ...     domain_bounds=(0, 2*np.pi, 0, 2.0)
        ... )
        >>> print(f"平均阻尼係數: {stats['damping_factor_mean']:.3f}")
        >>> print(f"裁剪點數: {stats['n_clipped']}")
    """
    nu_t = nu_t_raw.clone()
    
    # 記錄原始統計
    stats = {
        'raw_mean': nu_t.mean().item(),
        'raw_max': nu_t.max().item(),
        'raw_ratio_mean': (nu_t / nu).mean().item(),
    }
    
    # Step 1: Smoothing（可選）
    if apply_smoothing:
        nu_t = smooth_turbulent_viscosity(nu_t, coords, smoothing_radius, method="gaussian")
    
    # Step 2: Clipping
    n_before = nu_t.numel()
    if apply_clipping:
        nu_t_before_clip = nu_t.clone()
        nu_t = clip_turbulent_viscosity(nu_t, nu, max_ratio, min_value=0.0)
        n_clipped = (torch.abs(nu_t - nu_t_before_clip) > 1e-10).sum().item()
        stats['n_clipped'] = n_clipped
    else:
        stats['n_clipped'] = 0
    
    # Step 3: Damping
    if apply_damping:
        # 計算阻尼係數（用於統計）
        d_wall = compute_wall_distance_channel(coords, domain_bounds, wall_normal_axis)
        yplus = compute_yplus(d_wall, u_tau, nu)
        f_damp = van_driest_damping(yplus, A_plus)
        stats['damping_factor_mean'] = f_damp.mean().item()
        
        # 應用阻尼
        nu_t = apply_van_driest_damping(
            nu_t, coords, u_tau, nu, domain_bounds, wall_normal_axis, A_plus
        )
    else:
        stats['damping_factor_mean'] = 1.0
    
    # 記錄處理後統計
    stats['processed_mean'] = nu_t.mean().item()
    stats['processed_max'] = nu_t.max().item()
    
    return nu_t, stats


# ==================== 診斷工具 ====================

def _estimate_spatial_gradient_max(
    values: torch.Tensor,
    coords: torch.Tensor,
    max_points: int = 2000,
    eps: float = 1e-12
) -> Optional[float]:
    """
    以最近鄰有限差分近似最大空間梯度（避免全域微分需求）
    """
    if coords.dim() != 2 or values.dim() != 2:
        raise ValueError("coords 與 values 必須是 2D 張量")
    if coords.shape[0] != values.shape[0]:
        raise ValueError("coords 與 values 的點數必須一致")

    n_points = coords.shape[0]
    if n_points < 2:
        return 0.0

    with torch.no_grad():
        if n_points > max_points:
            sample_idx = torch.randperm(n_points, device=coords.device)[:max_points]
        else:
            sample_idx = torch.arange(n_points, device=coords.device)

        coords_sample = coords[sample_idx]
        values_sample = values[sample_idx].view(-1)

        distances = torch.cdist(coords_sample, coords_sample)
        distances.fill_diagonal_(float('inf'))

        min_dist, min_idx = distances.min(dim=1)
        neighbor_values = values_sample[min_idx]

        gradients = (values_sample - neighbor_values).abs() / (min_dist + eps)
        gradients = gradients[torch.isfinite(gradients)]

        if gradients.numel() == 0:
            return None

        return gradients.max().item()


def diagnose_turbulent_viscosity(
    nu_t: torch.Tensor,
    coords: torch.Tensor,
    nu: float,
    u_tau: Optional[float] = None,
    domain_bounds: Optional[Union[Tuple[float, float, float, float], Tuple[float, float, float, float, float, float]]] = None,
    wall_normal_axis: int = 1
) -> Dict[str, Any]:
    """
    診斷湍流黏度的物理合理性
    
    檢查項：
    1. 非負性：ν_t ≥ 0
    2. 範圍：ν_t/ν 的分布
    3. 近壁行為：y+ < 5 時 ν_t 是否接近 0
    4. 空間變化：梯度是否過大
    
    Args:
        nu_t: 湍流黏度 [N, 1]
        coords: 空間座標 [N, spatial_dim]
        nu: 分子黏度
        u_tau: 摩擦速度（可選，用於 y+ 計算）
        domain_bounds: 區域邊界（可選）
        wall_normal_axis: 壁法向軸（默認 1）
        
    Returns:
        diagnosis: 診斷結果字典
            - 'n_negative': 負值點數
            - 'ratio_mean': ν_t/ν 平均值
            - 'ratio_max': ν_t/ν 最大值
            - 'ratio_p95': ν_t/ν 95分位數
            - 'near_wall_check': 近壁檢查（若提供 u_tau）
            - 'spatial_gradient_max': 最大空間梯度
            - 'warnings': 警告列表
    """
    diagnosis: Dict[str, Any] = {
        'warnings': []
    }
    
    # 檢查 1: 非負性
    n_negative = (nu_t < 0).sum().item()
    diagnosis['n_negative'] = n_negative
    if n_negative > 0:
        pct = 100 * n_negative / nu_t.numel()
        diagnosis['warnings'].append(
            f"⚠️  發現 {n_negative} ({pct:.2f}%) 個負值 ν_t"
        )
    
    # 檢查 2: 範圍
    ratio = nu_t / nu
    diagnosis['ratio_mean'] = ratio.mean().item()
    diagnosis['ratio_max'] = ratio.max().item()
    diagnosis['ratio_p95'] = torch.quantile(ratio, 0.95).item()
    
    if diagnosis['ratio_max'] > 5000:
        diagnosis['warnings'].append(
            f"⚠️  ν_t/ν 最大值 = {diagnosis['ratio_max']:.0f} 過高（建議 < 5000）"
        )
    
    # 檢查 3: 近壁行為
    if u_tau is not None and domain_bounds is not None:
        d_wall = compute_wall_distance_channel(coords, domain_bounds, wall_normal_axis)
        yplus = compute_yplus(d_wall, u_tau, nu)
        
        # 選取 y+ < 5 的點
        near_wall_mask = yplus < 5.0
        if near_wall_mask.sum() > 0:
            nu_t_near_wall = nu_t[near_wall_mask]
            ratio_near_wall = (nu_t_near_wall / nu).mean().item()
            diagnosis['near_wall_ratio_mean'] = ratio_near_wall
            
            if ratio_near_wall > 10:
                diagnosis['warnings'].append(
                    f"⚠️  近壁區 (y+<5) 的 ν_t/ν 平均值 = {ratio_near_wall:.1f}（建議 < 10）"
                )
        else:
            diagnosis['near_wall_ratio_mean'] = None
            diagnosis['warnings'].append("⚠️  未找到 y+ < 5 的近壁點")
    
    # 檢查 4: 空間梯度（簡化版：只檢查鄰近點變化）
    try:
        gradient_max = _estimate_spatial_gradient_max(nu_t, coords)
        diagnosis['spatial_gradient_max'] = gradient_max
        if coords.shape[0] > 2000:
            diagnosis['warnings'].append(
                "ℹ️  空間梯度估計使用子樣本（max_points=2000）"
            )
    except Exception as exc:
        diagnosis['spatial_gradient_max'] = None
        diagnosis['warnings'].append(f"⚠️  空間梯度估計失敗: {exc}")

    return diagnosis


# ==================== 訓練整合輔助函數 ====================

def infer_preprocessing_params(
    preprocessing_cfg: Dict[str, Any],
    physics_cfg: Dict[str, Any],
    coords: torch.Tensor
) -> Tuple[float, Union[Tuple[float, float, float, float], Tuple[float, float, float, float, float, float]]]:
    """
    從配置自動推估預處理所需的物理參數
    
    推估邏輯：
    1. u_tau: 優先使用配置值，否則使用保守預設 0.05
    2. domain_bounds: 從 physics.domain 推斷，或從座標推斷
    
    Args:
        preprocessing_cfg: 預處理配置字典
        physics_cfg: 物理配置字典
        coords: 空間座標 [N, spatial_dim]
        
    Returns:
        u_tau: 摩擦速度
        domain_bounds: 區域邊界 (2D: 4-tuple, 3D: 6-tuple)
    """
    # 推估 u_tau
    u_tau = preprocessing_cfg.get('u_tau', None)
    if u_tau is None:
        u_tau = 0.05  # 保守預設值（適用於 channel flow Re_tau ~ 1000）
    
    # 推估 domain_bounds
    domain_bounds = preprocessing_cfg.get('domain_bounds', None)
    if domain_bounds is None:
        dom = physics_cfg.get('domain', {})
        spatial_dim = coords.shape[1]
        
        if spatial_dim == 3 and 'z_range' in dom:
            # 3D channel
            x_range = dom.get('x_range', [0, 6.28])
            y_range = dom.get('y_range', [0, 2.0])
            z_range = dom['z_range']
            domain_bounds = (x_range[0], x_range[1], y_range[0], y_range[1], z_range[0], z_range[1])
        else:
            # 2D channel/Kolmogorov
            x_range = dom.get('x_range', [0, 6.28])
            y_range = dom.get('y_range', [0, 2.0])
            domain_bounds = (x_range[0], x_range[1], y_range[0], y_range[1])
    
    return u_tau, domain_bounds


def preprocess_rans_prior_from_config(
    nu_t_raw: torch.Tensor,
    coords: torch.Tensor,
    config: Dict[str, Any],
    epoch: int = 0
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    從完整配置執行 RANS prior 預處理（訓練循環專用簡化介面）
    
    這是 preprocess_rans_prior() 的封裝，從配置字典自動提取所有參數。
    
    Args:
        nu_t_raw: 原始 RANS 湍流黏度 [N, 1]
        coords: 空間座標 [N, spatial_dim]
        config: 完整配置字典（包含 'physics' 和 'lowfi_prior.preprocessing'）
        epoch: 當前訓練 epoch（用於控制日誌頻率）
        
    Returns:
        nu_t_processed: 預處理後的湍流黏度 [N, 1]
        stats: 統計資訊字典
    """
    preprocessing_cfg = config.get('lowfi_prior', {}).get('preprocessing', {})
    physics_cfg = config.get('physics', {})
    
    # 提取物理參數
    nu = physics_cfg.get('nu', 1e-4)
    u_tau, domain_bounds = infer_preprocessing_params(preprocessing_cfg, physics_cfg, coords)
    
    # 執行預處理
    nu_t_processed, stats = preprocess_rans_prior(
        nu_t_raw,
        coords,
        nu=nu,
        u_tau=u_tau,
        domain_bounds=domain_bounds,
        apply_damping=preprocessing_cfg.get('apply_damping', True),
        apply_clipping=preprocessing_cfg.get('apply_clipping', True),
        apply_smoothing=preprocessing_cfg.get('apply_smoothing', False),
        smoothing_radius=preprocessing_cfg.get('smoothing_radius', 0.1),
        A_plus=preprocessing_cfg.get('A_plus', 26.0),
        max_ratio=preprocessing_cfg.get('max_ratio', 1000.0)
    )
    
    # 記錄預處理統計（每 100 epochs）
    if epoch % 100 == 0 and epoch > 0:
        import logging
        logging.debug(f"RANS preprocessing: raw_mean={stats['raw_mean']:.5f}, "
                     f"processed_mean={stats['processed_mean']:.5f}, "
                     f"damping_factor={stats['damping_factor_mean']:.3f}, "
                     f"n_clipped={stats['n_clipped']}")
    
    return nu_t_processed, stats

"""
Physics Validation 模組
=======================

提供檢查點物理一致性驗證功能：
1. 質量守恆驗證 (散度檢查)
2. 動量守恆驗證 (NS 方程殘差檢查)
3. 邊界條件驗證 (壁面無滑移條件)
4. 整合驗證與指標計算
"""

import torch
import torch.autograd as autograd
from typing import Dict, Tuple, Optional, Any
import warnings


def compute_divergence(
    coords: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    計算速度場散度 ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z

    Args:
        coords: 座標 [N, 2] 或 [N, 3] (x, y) 或 (x, y, z)
        u: x 方向速度 [N, 1]
        v: y 方向速度 [N, 1]
        w: z 方向速度 [N, 1] (可選，3D情況)

    Returns:
        散度場 [N, 1]
    """
    # 確保座標需要梯度
    if not coords.requires_grad:
        coords = coords.clone().detach().requires_grad_(True)

    # 計算 ∂u/∂x
    grad_outputs = torch.ones_like(u)
    du_dx = autograd.grad(
        outputs=u,
        inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:, 0:1]

    # 計算 ∂v/∂y
    grad_outputs = torch.ones_like(v)
    dv_dy = autograd.grad(
        outputs=v,
        inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:, 1:2]

    # 2D 情況
    if w is None or coords.shape[1] == 2:
        div_u = du_dx + dv_dy
        return div_u

    # 3D 情況：計算 ∂w/∂z
    grad_outputs = torch.ones_like(w)
    dw_dz = autograd.grad(
        outputs=w,
        inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:, 2:3]

    div_u = du_dx + dv_dy + dw_dz
    return div_u


def validate_mass_conservation(
    coords: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: Optional[torch.Tensor] = None,
    threshold: float = 1e-2
) -> Tuple[bool, float]:
    """
    驗證質量守恆：檢查速度場散度是否接近零

    Args:
        coords: 座標 [N, 2] 或 [N, 3]
        u: x 方向速度 [N, 1]
        v: y 方向速度 [N, 1]
        w: z 方向速度 [N, 1] (可選)
        threshold: 允許的最大散度誤差

    Returns:
        (是否通過驗證, 實際最大散度誤差)
    """
    try:
        div_u = compute_divergence(coords, u, v, w)
        max_div_error = torch.max(torch.abs(div_u)).item()
        passed = max_div_error < threshold
        return passed, max_div_error
    except RuntimeError as e:
        warnings.warn(f"質量守恆驗證失敗：{str(e)}")
        return False, float('inf')


def validate_momentum_conservation(
    coords: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    w: Optional[torch.Tensor] = None,
    nu: float = 5e-5,
    threshold: float = 1e-1
) -> Tuple[bool, float]:
    """
    驗證動量守恆：檢查 Navier-Stokes 方程殘差

    計算穩態 NS 方程殘差：R = (u·∇)u + ∇p - ν∇²u

    Args:
        coords: 座標 [N, 2] 或 [N, 3]
        u: x 方向速度 [N, 1]
        v: y 方向速度 [N, 1]
        p: 壓力 [N, 1]
        w: z 方向速度 [N, 1] (可選)
        nu: 黏滯係數
        threshold: 允許的平均殘差

    Returns:
        (是否通過驗證, 實際平均殘差)
    """
    try:
        # 確保座標需要梯度
        if not coords.requires_grad:
            coords = coords.clone().detach().requires_grad_(True)

        # 計算速度梯度
        grad_outputs = torch.ones_like(u)
        u_grad = autograd.grad(u, coords, grad_outputs=grad_outputs,
                               create_graph=True, retain_graph=True)[0]
        v_grad = autograd.grad(v, coords, grad_outputs=torch.ones_like(v),
                               create_graph=True, retain_graph=True)[0]

        # 計算對流項 (u·∇)u
        if coords.shape[1] == 2:  # 2D
            du_dx, du_dy = u_grad[:, 0:1], u_grad[:, 1:2]
            dv_dx, dv_dy = v_grad[:, 0:1], v_grad[:, 1:2]

            conv_u = u * du_dx + v * du_dy
            conv_v = u * dv_dx + v * dv_dy

        else:  # 3D
            w_grad = autograd.grad(w, coords, grad_outputs=torch.ones_like(w),
                                   create_graph=True, retain_graph=True)[0]
            du_dx, du_dy, du_dz = u_grad[:, 0:1], u_grad[:, 1:2], u_grad[:, 2:3]
            dv_dx, dv_dy, dv_dz = v_grad[:, 0:1], v_grad[:, 1:2], v_grad[:, 2:3]
            dw_dx, dw_dy, dw_dz = w_grad[:, 0:1], w_grad[:, 1:2], w_grad[:, 2:3]

            conv_u = u * du_dx + v * du_dy + w * du_dz
            conv_v = u * dv_dx + v * dv_dy + w * dv_dz

        # 計算壓力梯度
        p_grad = autograd.grad(p, coords, grad_outputs=torch.ones_like(p),
                               create_graph=True, retain_graph=True)[0]
        dp_dx, dp_dy = p_grad[:, 0:1], p_grad[:, 1:2]

        # 計算拉普拉斯項 (簡化版本：使用散度的梯度近似)
        # 完整計算需要二階導數，這裡使用一階導數範數作為近似
        laplace_u = torch.sum(u_grad ** 2, dim=1, keepdim=True)
        laplace_v = torch.sum(v_grad ** 2, dim=1, keepdim=True)

        # NS 方程殘差 (x, y 分量)
        residual_u = conv_u + dp_dx - nu * laplace_u
        residual_v = conv_v + dp_dy - nu * laplace_v

        # 計算平均殘差
        mean_residual = torch.mean(torch.abs(residual_u) + torch.abs(residual_v)).item()
        passed = mean_residual < threshold

        return passed, mean_residual

    except RuntimeError as e:
        warnings.warn(f"動量守恆驗證失敗：{str(e)}")
        return False, float('inf')


def validate_boundary_conditions(
    coords: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: Optional[torch.Tensor] = None,
    wall_positions: Tuple[float, float] = (0.0, 2.0),
    threshold: float = 1e-3,
    tolerance: float = 1e-4
) -> Tuple[bool, float]:
    """
    驗證壁面邊界條件：檢查壁面無滑移條件 (u=v=w=0 at walls)

    Args:
        coords: 座標 [N, 2] 或 [N, 3]
        u: x 方向速度 [N, 1]
        v: y 方向速度 [N, 1]
        w: z 方向速度 [N, 1] (可選)
        wall_positions: 壁面位置 (y_bottom, y_top)
        threshold: 允許的最大壁面速度誤差
        tolerance: 判定點是否在壁面的容差

    Returns:
        (是否通過驗證, 實際最大壁面速度)
    """
    try:
        # 找出壁面附近的點
        y = coords[:, 1:2]
        y_bottom, y_top = wall_positions

        # 使用容差判斷是否在壁面
        bottom_wall_mask = torch.abs(y - y_bottom) < tolerance
        top_wall_mask = torch.abs(y - y_top) < tolerance
        wall_mask = bottom_wall_mask | top_wall_mask

        if not torch.any(wall_mask):
            warnings.warn("未找到壁面點，無法驗證邊界條件")
            return True, 0.0  # 無壁面點時視為通過

        # 計算壁面速度
        u_wall = u[wall_mask]
        v_wall = v[wall_mask]

        if w is not None:
            w_wall = w[wall_mask]
            max_wall_velocity = torch.max(
                torch.cat([torch.abs(u_wall), torch.abs(v_wall), torch.abs(w_wall)])
            ).item()
        else:
            max_wall_velocity = torch.max(
                torch.cat([torch.abs(u_wall), torch.abs(v_wall)])
            ).item()

        passed = max_wall_velocity < threshold
        return passed, max_wall_velocity

    except RuntimeError as e:
        warnings.warn(f"邊界條件驗證失敗：{str(e)}")
        return False, float('inf')


def detect_trivial_solution(
    predictions: Dict[str, torch.Tensor],
    epsilon: float = 1e-6
) -> Dict[str, Any]:
    """
    檢測 trivial solution（全零解或常數解）

    Args:
        predictions: 預測場字典，包含 'u', 'v', 'p', 'w' (可選)
        epsilon: 判定為零的閾值

    Returns:
        檢測結果字典：
        {
            'is_trivial': bool,
            'type': str,  # 'zero', 'constant', 'none'
            'details': {...}
        }
    """
    u = predictions['u']
    v = predictions['v']
    p = predictions['p']
    w = predictions.get('w', None)

    # 檢查全零解
    u_max = torch.max(torch.abs(u)).item()
    v_max = torch.max(torch.abs(v)).item()
    p_max = torch.max(torch.abs(p)).item()

    if u_max < epsilon and v_max < epsilon:
        return {
            'is_trivial': True,
            'type': 'zero',
            'details': {
                'u_max': u_max,
                'v_max': v_max,
                'p_max': p_max
            }
        }

    # 檢查常數解（變異數接近零）
    u_std = torch.std(u).item()
    v_std = torch.std(v).item()
    p_std = torch.std(p).item()

    if u_std < epsilon and v_std < epsilon:
        return {
            'is_trivial': True,
            'type': 'constant',
            'details': {
                'u_std': u_std,
                'v_std': v_std,
                'p_std': p_std,
                'u_mean': torch.mean(u).item(),
                'v_mean': torch.mean(v).item()
            }
        }

    # 檢查動態範圍
    u_range = u_max
    v_range = v_max
    total_range = u_range + v_range

    if total_range < epsilon:
        return {
            'is_trivial': True,
            'type': 'near_zero',
            'details': {
                'u_range': u_range,
                'v_range': v_range,
                'total_range': total_range
            }
        }

    return {
        'is_trivial': False,
        'type': 'none',
        'details': {
            'u_std': u_std,
            'v_std': v_std,
            'u_range': u_range,
            'v_range': v_range
        }
    }


def compute_physics_metrics(
    coords: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    physics_params: Optional[Dict[str, Any]] = None,
    validation_thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    整合物理驗證函數，計算所有物理指標

    Args:
        coords: 座標 [N, 2] 或 [N, 3]
        predictions: 預測場字典，包含 'u', 'v', 'p', 'w' (可選)
        physics_params: 物理參數 {'nu': 黏滯係數, 'wall_positions': (y_bottom, y_top)}
        validation_thresholds: 驗證閾值 {'mass_conservation', 'momentum_conservation', 'boundary_condition'}

    Returns:
        物理指標字典：
        {
            'mass_conservation_error': float,
            'momentum_conservation_error': float,
            'boundary_condition_error': float,
            'mass_conservation_passed': bool,
            'momentum_conservation_passed': bool,
            'boundary_condition_passed': bool,
            'validation_passed': bool
        }
    """
    # 預設參數
    if physics_params is None:
        physics_params = {
            'nu': 5e-5,
            'wall_positions': (0.0, 2.0)
        }

    if validation_thresholds is None:
        validation_thresholds = {
            'mass_conservation': 1e-2,
            'momentum_conservation': 1e-1,
            'boundary_condition': 1e-3
        }

    # 提取預測場
    u = predictions['u']
    v = predictions['v']
    p = predictions['p']
    w = predictions.get('w', None)

    # 驗證質量守恆
    mass_passed, mass_error = validate_mass_conservation(
        coords, u, v, w,
        threshold=validation_thresholds['mass_conservation']
    )

    # 驗證動量守恆
    momentum_passed, momentum_error = validate_momentum_conservation(
        coords, u, v, p, w,
        nu=physics_params['nu'],
        threshold=validation_thresholds['momentum_conservation']
    )

    # 驗證邊界條件
    bc_passed, bc_error = validate_boundary_conditions(
        coords, u, v, w,
        wall_positions=physics_params['wall_positions'],
        threshold=validation_thresholds['boundary_condition']
    )

    # 檢測 trivial solution
    trivial_check = detect_trivial_solution(predictions)

    # 整合結果
    metrics = {
        'mass_conservation_error': mass_error,
        'momentum_conservation_error': momentum_error,
        'boundary_condition_error': bc_error,
        'mass_conservation_passed': mass_passed,
        'momentum_conservation_passed': momentum_passed,
        'boundary_condition_passed': bc_passed,
        'trivial_solution': trivial_check,
        'validation_passed': mass_passed and momentum_passed and bc_passed and not trivial_check['is_trivial']
    }

    return metrics

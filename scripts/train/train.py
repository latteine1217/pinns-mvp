#!/usr/bin/env python3
"""
PINNs 逆重建主訓練腳本
負責協調資料載入、模型建立、訓練迴圈與評估輸出
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.losses.residuals import NSResidualLoss, BoundaryConditionLoss
from pinnx.losses.priors import PriorLossManager
from pinnx.losses.weighting import GradNormWeighter, CausalWeighter, AdaptiveWeightScheduler
from pinnx.losses import MeanConstraintLoss  # ⭐ Phase 6C: 均值約束損失
from pinnx.train.trainer import Trainer  # 新的訓練器類
from pinnx.utils.normalization import InputNormalizer, NormalizationConfig

# 從重構模組導入配置、檢查點與工廠函數
from pinnx.train.config_loader import (
    load_config,
    normalize_config_structure,
    derive_loss_weights,
)
from pinnx.train.checkpointing import (
    save_checkpoint,
    load_checkpoint,
)
from pinnx.train.factory import (
    get_device,
    create_model,
    create_physics,
    create_optimizer,
)
from pinnx.utils.setup import (
    setup_logging,
    set_random_seed,
)

# Kolmogorov flow data preparation (Fixed Sensors x Time Series)
def prepare_kolmogorov_training_data(config: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """準備 Kolmogorov Flow 訓練資料 (固定感測器 x 時間序列)
    
    Args:
        config: 配置字典
        device: PyTorch 設備
        
    Returns:
        訓練資料字典
    """
    import h5py
    import json
    
    # 載入配置
    kol_cfg = config['data']['kolmogorov_config']
    data_path = kol_cfg['data_path']
    time_range = kol_cfg['time_range']
    
    # 1. 讀取感測器位置檔案 (QR-Pivot 結果)
    sensor_file = config.get('sensors', {}).get('sensor_file')
    if not sensor_file:
        raise ValueError("必須在 config['sensors']['sensor_file'] 指定感測器位置檔案 (.json)")
        
    logging.info(f"📂 載入感測器位置: {sensor_file}")
    with open(sensor_file, 'r') as f:
        sensor_data = json.load(f)
        
    # sensor_indices 是展平後的空間索引 (0 ~ N*N-1)
    spatial_indices = np.array(sensor_data['indices'])
    K = len(spatial_indices)
    logging.info(f"   已選定 {K} 個固定空間感測點")

    # 2. 載入 DNS 全場數據
    logging.info(f"📂 載入 DNS 數據: {data_path}")
    with h5py.File(data_path, 'r') as f:
        # 讀取時間軸
        time_all = np.array(f['time'])
        
        # 選擇時間範圍
        t_start, t_end = time_range
        time_mask = (time_all >= t_start) & (time_all <= t_end)
        time_selected = time_all[time_mask]
        T_selected = len(time_selected)
        
        logging.info(f"   時間範圍: [{t_start:.1f}, {t_end:.1f}], 共 {T_selected} 個時間步")
        
        # 讀取空間網格資訊
        N = int(f['config'].attrs['N'])
        L = float(f['config'].attrs['L'])
        
        # 建立空間座標網格
        x_1d = np.linspace(0, L, N, endpoint=False)
        y_1d = np.linspace(0, L, N, endpoint=False)
        X_mesh, Y_mesh = np.meshgrid(x_1d, y_1d, indexing='ij')
        
        # 提取感測點的 (x, y) 座標
        X_flat = X_mesh.flatten()
        Y_flat = Y_mesh.flatten()
        
        x_sensor_locs = X_flat[spatial_indices]  # [K]
        y_sensor_locs = Y_flat[spatial_indices]  # [K]
        
        # 3. 提取感測點的時間序列數據
        # 策略：讀取需要的時間步，然後只取選定的空間點
        # 為了效率，我們先讀取所需的時間切片 [T_selected, N, N]
        # 注意：如果是大檔案，可能需要更精細的讀取策略
        u_slice = f['u'][time_mask]  # [T, N, N]
        v_slice = f['v'][time_mask]
        if 'p' in f:
            p_slice = f['p'][time_mask]
        else:
            p_slice = None
            
        # 展平空間維度 [T, N*N]
        u_flat = u_slice.reshape(T_selected, -1)
        v_flat = v_slice.reshape(T_selected, -1)
        p_flat = p_slice.reshape(T_selected, -1) if p_slice is not None else None
        
        # 提取感測點的值 [T, K]
        u_sensors_vals = u_flat[:, spatial_indices]
        v_sensors_vals = v_flat[:, spatial_indices]
        p_sensors_vals = p_flat[:, spatial_indices] if p_flat is not None else None
        
    # 4. 構建訓練張量 (T * K 樣本)
    # 我們需要將 [T, K] 展平成 [T*K, 1]
    
    # 時間座標: 每個感測點重複 T 次
    # [t0, t1, ..., t0, t1, ...]
    # 為了方便，我們使用 meshgrid 構造 (t, k)
    T_grid, K_grid = np.meshgrid(time_selected, np.arange(K), indexing='ij')
    
    # 展平
    t_train = T_grid.flatten()  # [T*K]
    k_indices = K_grid.flatten() # [T*K] 用於索引空間座標
    
    x_train = x_sensor_locs[k_indices]
    y_train = y_sensor_locs[k_indices]
    
    u_train = u_sensors_vals.flatten()
    v_train = v_sensors_vals.flatten()
    if p_sensors_vals is not None:
        p_train = p_sensors_vals.flatten()
    else:
        p_train = np.zeros_like(u_train)
        
    # 轉換為 Tensor
    x_sensors = torch.tensor(x_train, dtype=torch.float32, device=device).unsqueeze(1)
    y_sensors = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    t_sensors = torch.tensor(t_train, dtype=torch.float32, device=device).unsqueeze(1)
    u_sensors = torch.tensor(u_train, dtype=torch.float32, device=device).unsqueeze(1)
    v_sensors = torch.tensor(v_train, dtype=torch.float32, device=device).unsqueeze(1)
    p_sensors = torch.tensor(p_train, dtype=torch.float32, device=device).unsqueeze(1)
    
    # 5. PDE 配點採樣 (隨機時空採樣)
    # 從全域 (x, y, t) 中隨機採樣
    N_pde = config.get('sampling', {}).get('N_pde', 10000)
    
    x_pde = torch.rand(N_pde, 1, device=device) * L
    y_pde = torch.rand(N_pde, 1, device=device) * L
    t_pde = torch.rand(N_pde, 1, device=device) * (t_end - t_start) + t_start
    
    # 排序 t_pde 以優化因果訓練效率 (雖然 CausalWeighter 會自己處理，但排序好是個好習慣)
    t_pde, sort_idx = torch.sort(t_pde, dim=0)
    x_pde = x_pde[sort_idx].reshape(-1, 1)
    y_pde = y_pde[sort_idx].reshape(-1, 1)
    
    # 構建訓練資料字典
    training_data = {
        # 感測點 (固定位置 x 時間序列)
        'x_sensors': x_sensors,
        'y_sensors': y_sensors,
        't_sensors': t_sensors,
        'u_sensors': u_sensors,
        'v_sensors': v_sensors,
        'p_sensors': p_sensors,
        
        # PDE 配點 (隨機時空)
        'x_pde': x_pde,
        'y_pde': y_pde,
        't_pde': t_pde,
        
        # 邊界條件 (Kolmogorov 是週期性的，這裡留空或設為空集合)
        'x_bc': torch.empty(0, 1, device=device),
        'y_bc': torch.empty(0, 1, device=device),
        't_bc': torch.empty(0, 1, device=device),
        
        # 初始條件 (t=0 全場快照，可選)
        # 若需要強 IC 約束，可在此添加 t=t_start 的全場數據
        'x_ic': torch.empty(0, 1, device=device),
        'y_ic': torch.empty(0, 1, device=device),
        't_ic': torch.empty(0, 1, device=device),
    }
    
    logging.info(f"✅ Kolmogorov 訓練數據準備完成 (Fixed Sensors):")
    logging.info(f"   感測點數 K: {K}")
    logging.info(f"   時間步數 T: {T_selected}")
    logging.info(f"   總監督樣本 (K*T): {len(x_sensors)}")
    logging.info(f"   PDE 配點: {N_pde}")
    logging.info(f"   時間範圍: [{t_start:.1f}, {t_end:.1f}]")
    
    return training_data


def load_rans_prior_data(
    config: Dict[str, Any], 
    training_data: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """載入 RANS 低保真先驗資料並插值到訓練點
    
    Args:
        config: 配置字典
        training_data: 訓練資料字典（包含座標）
        device: PyTorch 設備
        
    Returns:
        包含 RANS 先驗場的字典 {'u': ..., 'v': ..., 'p': ...}
    """
    import h5py
    from scipy.interpolate import RegularGridInterpolator
    
    lowfi_cfg = config.get('lowfi_prior', {})
    if not lowfi_cfg.get('enabled', False):
        logging.info("⏭️  RANS 先驗未啟用")
        return {}
    
    rans_path = lowfi_cfg.get('data_path')
    if not rans_path:
        logging.warning("⚠️  lowfi_prior.enabled=True 但未指定 data_path")
        return {}
    
    logging.info(f"📂 載入 RANS 先驗資料: {rans_path}")
    
    # 讀取 RANS 資料
    with h5py.File(rans_path, 'r') as f:
        rans_structure = lowfi_cfg.get('rans_structure', {})
        group_path = rans_structure.get('group_path', '/mean_field')
        
        # 讀取座標網格
        group = f[group_path]
        X_rans = np.array(group['X'])  # [N_rans, N_rans]
        Y_rans = np.array(group['Y'])
        
        # 提取 1D 座標（假設規則網格）
        x_rans_1d = X_rans[:, 0]  # 第一列
        y_rans_1d = Y_rans[0, :]  # 第一行
        
        # 讀取場數據
        u_rans = np.array(group['u'])  # [N_rans, N_rans]
        v_rans = np.array(group['v'])
        
        # RANS 通常沒有壓力，設為零或從 DNS 估計
        if 'p' in group:
            p_rans = np.array(group['p'])
        else:
            p_rans = np.zeros_like(u_rans)
            logging.info("   RANS 資料無壓力場，設為零")
        
        # ⭐ 讀取 RANS 湍流黏度 (nu_t / eddy viscosity)
        nu_t_rans = None
        for nu_t_key in ['nu_t', 'nut', 'eddy_viscosity', 'turbulent_viscosity']:
            if nu_t_key in group:
                nu_t_rans = np.array(group[nu_t_key])
                logging.info(f"   ✅ 讀取 RANS 湍流黏度: {nu_t_key}")
                break
        
        # 如果找不到 nu_t，嘗試從 k-epsilon 模型計算
        if nu_t_rans is None:
            if 'k' in group and 'epsilon' in group:
                k_rans = np.array(group['k'])
                eps_rans = np.array(group['epsilon'])
                C_mu = 0.09  # k-epsilon 模型常數
                nu_t_rans = C_mu * k_rans**2 / (eps_rans + 1e-10)
                logging.info(f"   ✅ 從 k-epsilon 計算 nu_t: C_μ={C_mu}")
            else:
                # Fallback: 設為零（僅使用分子黏度）
                nu_t_rans = np.zeros_like(u_rans)
                logging.warning("   ⚠️  RANS 資料無湍流黏度 (nu_t/k/epsilon)，設為零")
    
    logging.info(f"   RANS 解析度: {u_rans.shape}")
    logging.info(f"   RANS 座標範圍: x=[{x_rans_1d.min():.3f}, {x_rans_1d.max():.3f}], "
                 f"y=[{y_rans_1d.min():.3f}, {y_rans_1d.max():.3f}]")
    
    # 建立插值器
    interp_method = lowfi_cfg.get('interpolation', {}).get('method', 'linear')
    u_interp = RegularGridInterpolator((x_rans_1d, y_rans_1d), u_rans, method=interp_method, bounds_error=False, fill_value=None)
    v_interp = RegularGridInterpolator((x_rans_1d, y_rans_1d), v_rans, method=interp_method, bounds_error=False, fill_value=None)
    p_interp = RegularGridInterpolator((x_rans_1d, y_rans_1d), p_rans, method=interp_method, bounds_error=False, fill_value=None)
    
    # ⭐ 建立 nu_t 插值器（如果存在）
    if nu_t_rans is not None:
        nu_t_interp = RegularGridInterpolator(
            (x_rans_1d, y_rans_1d), nu_t_rans, 
            method=interp_method, bounds_error=False, fill_value=None
        )
    else:
        nu_t_interp = None
    
    # 提取訓練點座標（只需空間座標，忽略時間）
    # 假設使用 PDE 配點作為插值目標
    x_pde_np = training_data['x_pde'].cpu().numpy().flatten()
    y_pde_np = training_data['y_pde'].cpu().numpy().flatten()
    
    coords_pde = np.column_stack([x_pde_np, y_pde_np])
    
    # 插值到 PDE 配點
    u_prior_pde = u_interp(coords_pde)
    v_prior_pde = v_interp(coords_pde)
    p_prior_pde = p_interp(coords_pde)
    
    # ⭐ 插值 nu_t 到 PDE 配點
    if nu_t_interp is not None:
        nu_t_prior_pde = nu_t_interp(coords_pde)
    else:
        nu_t_prior_pde = None
    
    # 同樣插值到感測點（用於驗證）
    x_sensors_np = training_data['x_sensors'].cpu().numpy().flatten()
    y_sensors_np = training_data['y_sensors'].cpu().numpy().flatten()
    coords_sensors = np.column_stack([x_sensors_np, y_sensors_np])
    
    u_prior_sensors = u_interp(coords_sensors)
    v_prior_sensors = v_interp(coords_sensors)
    p_prior_sensors = p_interp(coords_sensors)
    
    # ⭐ 插值 nu_t 到感測點（目前不用於訓練，僅用於診斷）
    if nu_t_interp is not None:
        nu_t_prior_sensors = nu_t_interp(coords_sensors)
    else:
        nu_t_prior_sensors = None
    
    # 轉換為 Tensor
    rans_prior = {
        'u_pde': torch.tensor(u_prior_pde, dtype=torch.float32, device=device).unsqueeze(1),
        'v_pde': torch.tensor(v_prior_pde, dtype=torch.float32, device=device).unsqueeze(1),
        'p_pde': torch.tensor(p_prior_pde, dtype=torch.float32, device=device).unsqueeze(1),
        'u_sensors': torch.tensor(u_prior_sensors, dtype=torch.float32, device=device).unsqueeze(1),
        'v_sensors': torch.tensor(v_prior_sensors, dtype=torch.float32, device=device).unsqueeze(1),
        'p_sensors': torch.tensor(p_prior_sensors, dtype=torch.float32, device=device).unsqueeze(1),
    }
    
    # ⭐ 添加 nu_t 到返回字典（如果存在）
    if nu_t_prior_pde is not None:
        rans_prior['nu_t_pde'] = torch.tensor(nu_t_prior_pde, dtype=torch.float32, device=device).unsqueeze(1)
    if nu_t_prior_sensors is not None:
        rans_prior['nu_t_sensors'] = torch.tensor(nu_t_prior_sensors, dtype=torch.float32, device=device).unsqueeze(1)
    
    logging.info(f"✅ RANS 先驗插值完成:")
    logging.info(f"   PDE 配點: {len(u_prior_pde)} 個")
    logging.info(f"   感測點: {len(u_prior_sensors)} 個")
    logging.info(f"   u 統計: min={u_prior_pde.min():.4f}, max={u_prior_pde.max():.4f}, mean={u_prior_pde.mean():.4f}")
    logging.info(f"   v 統計: min={v_prior_pde.min():.4f}, max={v_prior_pde.max():.4f}, mean={v_prior_pde.mean():.4f}")
    
    return rans_prior


# ============================================================================
# 全局變數（保留訓練專用快取）
# ============================================================================


def _collect_coordinate_tensors(training_data: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    prefixes = ['sensors', 'pde', 'bc', 'ic']
    coords: List[torch.Tensor] = []
    axes = ['x', 'y', 'z']
    for prefix in prefixes:
        components = []
        for axis in axes:
            key = f'{axis}_{prefix}'
            if key in training_data and training_data[key].numel() > 0:
                components.append(training_data[key])
        if components:
            coords.append(torch.cat(components, dim=1))
    return coords


def create_input_normalizer(
    config: Dict[str, Any],
    training_data: Dict[str, torch.Tensor],
    is_vs_pinn: bool,
    device: torch.device
) -> Optional[InputNormalizer]:
    scaling_cfg = config.get('model', {}).get('scaling', {})
    norm_type = scaling_cfg.get('input_norm', 'none')
    if norm_type is None:
        norm_type = 'none'

    norm_type = norm_type.lower()

    if norm_type in ('none', 'identity'):
        return None

    if is_vs_pinn and norm_type in ('vs_pinn', 'channel_flow'):
        # VS-PINN already applies dedicated scaling; avoid double normalization.
        return None

    bounds_tensor: Optional[torch.Tensor] = None
    if norm_type in ('channel_flow',):
        domain = config.get('physics', {}).get('domain', {})
        bounds: List[Tuple[float, float]] = []
        for axis in ['x', 'y', 'z']:
            rng = domain.get(f'{axis}_range')
            if rng is not None:
                bounds.append((float(rng[0]), float(rng[1])))
        if bounds:
            bounds_tensor = torch.tensor(bounds, dtype=torch.float32, device=device)

    feature_range = tuple(scaling_cfg.get('input_norm_range', [-1.0, 1.0]))
    config_obj = NormalizationConfig(
        norm_type=norm_type,
        feature_range=(float(feature_range[0]), float(feature_range[1])),
        bounds=bounds_tensor
    )
    normalizer = InputNormalizer(config_obj)

    coord_tensors = _collect_coordinate_tensors(training_data)
    if coord_tensors:
        samples = torch.cat(coord_tensors, dim=0)
        if normalizer.bounds is not None and normalizer.bounds.shape[0] > samples.shape[1]:
            normalizer.bounds = normalizer.bounds[:samples.shape[1], :]
        normalizer.fit(samples)
    else:
        return None

    normalizer.to(device)
    return normalizer


# ============================================================================
# Warmup + CosineAnnealing 學習率調度器
# ============================================================================
class WarmupCosineScheduler:
    """
    Warmup + CosineAnnealing 學習率調度器
    
    前 warmup_epochs 個 epoch 線性增加學習率從 0 到 base_lr
    之後使用 CosineAnnealing 衰減到 min_lr
    """
    
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, 
                 base_lr: float, min_lr: float = 0.0):
        """
        Args:
            optimizer: PyTorch 優化器
            warmup_epochs: Warmup 階段的 epoch 數量
            max_epochs: 總訓練 epoch 數量
            base_lr: 基礎學習率（Warmup 後的峰值）
            min_lr: 最小學習率（CosineAnnealing 的下限）
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
        # 建立內部的 CosineAnnealing 調度器（用於 Warmup 後階段）
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=max_epochs - warmup_epochs,
            eta_min=min_lr
        )
        
        logging.info(f"✅ WarmupCosineScheduler initialized:")
        logging.info(f"   Warmup epochs: {warmup_epochs}")
        logging.info(f"   Max epochs: {max_epochs}")
        logging.info(f"   Base LR: {base_lr:.6f}")
        logging.info(f"   Min LR: {min_lr:.6f}")
    
    def step(self):
        """更新學習率"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup 階段：線性增加
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # CosineAnnealing 階段
            self.cosine_scheduler.step()
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        """返回當前學習率（兼容性接口）"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


# ============================================================================
# 階段式權重調度器
# ============================================================================
class StagedWeightScheduler:
    """階段式權重調度器 - 根據 epoch 切換不同訓練階段的權重"""
    
    def __init__(self, phases: list):
        """
        Args:
            phases: 階段配置列表，每個元素包含:
                - name: 階段名稱
                - epoch_range: [start, end] epoch 範圍
                - weights: 該階段的權重字典
        """
        self.phases = phases
        self.current_phase_idx = 0
        self.current_phase_name = phases[0]['name'] if phases else "default"
        
        # 按 epoch_range 排序
        self.phases.sort(key=lambda p: p['epoch_range'][0])
        
        logging.info(f"✅ StagedWeightScheduler initialized with {len(phases)} phases:")
        for p in self.phases:
            logging.info(f"   {p['name']}: Epoch {p['epoch_range'][0]}-{p['epoch_range'][1]}")
    
    def get_phase_weights(self, epoch: int) -> tuple:
        """
        獲取當前 epoch 對應的權重
        
        Returns:
            (weights_dict, phase_name, is_transition)
        """
        # 找到當前 epoch 所屬階段
        for idx, phase in enumerate(self.phases):
            start, end = phase['epoch_range']
            if start <= epoch < end:
                # 檢測是否為階段切換點
                is_transition = (idx != self.current_phase_idx)
                self.current_phase_idx = idx
                self.current_phase_name = phase['name']
                
                return phase['weights'], phase['name'], is_transition
        
        # 如果超出所有階段，返回最後階段
        last_phase = self.phases[-1]
        return last_phase['weights'], last_phase['name'], False


# 課程訓練調度器 - 支援雷諾數動態變化
class CurriculumScheduler:
    """
    課程訓練調度器 - 逐步提升雷諾數，從層流到湍流
    
    特性：
    - 動態調整 Re_tau, nu, pressure_gradient
    - 階段式權重切換
    - 階段式學習率調整
    - 階段式採樣策略調整
    """
    
    def __init__(self, stages: list, physics_module):
        """
        Args:
            stages: 課程階段列表，每個元素包含:
                - name: 階段名稱
                - epoch_range: [start, end]
                - Re_tau: 雷諾數
                - nu: 黏度
                - pressure_gradient: 壓力梯度
                - weights: 損失權重字典
                - sampling: 採樣配置
                - lr: 學習率
            physics_module: 物理方程模組（用於更新參數）
        """
        self.stages = stages
        self.physics = physics_module
        self.current_stage_idx = 0
        self.current_stage = stages[0] if stages else None
        
        # 按 epoch_range 排序
        self.stages.sort(key=lambda s: s['epoch_range'][0])
        
        logging.info("="*80)
        logging.info("🚀 CurriculumScheduler initialized - Progressive Reynolds Number Training")
        logging.info("="*80)
        for i, s in enumerate(self.stages, 1):
            logging.info(f"Stage {i}: {s['name']}")
            logging.info(f"  Epochs: {s['epoch_range'][0]}-{s['epoch_range'][1]}")
            
            # 根據場景類型顯示不同的物理參數
            if 'Re_tau' in s and s['Re_tau'] is not None:  # Channel Flow
                if s.get('pressure_gradient') is not None:
                    logging.info(f"  Re_tau: {s['Re_tau']:.1f}, nu: {s['nu']:.6f}, dP/dx: {s['pressure_gradient']:.3f}")
                else:
                    logging.info(f"  Re_tau: {s['Re_tau']:.1f}, nu: {s['nu']:.6f}")
            elif 'Re' in s and s['Re'] is not None:  # Kolmogorov Flow
                logging.info(f"  Re: {s['Re']:.1f}, nu: {s['nu']:.6f}")
            else:
                logging.info(f"  nu: {s['nu']:.6f}")
            
            logging.info(f"  PDE points: {s['sampling']['pde_points']}, BC points: {s['sampling']['boundary_points']}")
            # 處理 lr（可選參數）
            if 'lr' in s:
                lr_value = float(s['lr']) if isinstance(s['lr'], (str, int)) else s['lr']
                logging.info(f"  Learning rate: {lr_value:.6f} (explicit)")
            else:
                logging.info(f"  Learning rate: inherited + global scheduler")
        logging.info("="*80)
    
    def get_stage_config(self, epoch: int) -> Dict[str, Any]:
        """
        獲取當前 epoch 對應的階段配置
        
        Returns:
            {
                'stage_name': str,
                'is_transition': bool,
                'weights': dict,
                'Re_tau': float,
                'nu': float,
                'pressure_gradient': float,
                'sampling': dict,
                'lr': float
            }
        """
        # 找到當前階段
        for idx, stage in enumerate(self.stages):
            start, end = stage['epoch_range']
            if start <= epoch < end:
                # 檢測階段切換
                is_transition = (idx != self.current_stage_idx)
                
                if is_transition:
                    self.current_stage_idx = idx
                    self.current_stage = stage
                    
                    # 更新物理參數
                    self._update_physics_parameters(stage)
                    
                    logging.info("="*80)
                    logging.info(f"🎯 CURRICULUM STAGE TRANSITION at Epoch {epoch}")
                    logging.info(f"📚 New Stage: {stage['name']}")
                    
                    # 根據場景類型顯示不同的物理參數
                    if 'Re_tau' in stage:  # Channel Flow
                        logging.info(f"🔬 Re_tau: {stage['Re_tau']:.1f}, nu: {stage['nu']:.6f}")
                    elif 'Re' in stage:  # Kolmogorov Flow
                        logging.info(f"🔬 Re: {stage['Re']:.1f}, nu: {stage['nu']:.6f}")
                    else:
                        logging.info(f"🔬 nu: {stage['nu']:.6f}")
                    
                    logging.info(f"⚙️  PDE/BC points: {stage['sampling']['pde_points']}/{stage['sampling']['boundary_points']}")
                    logging.info(f"📊 Weights: {stage['weights']}")
                    
                    # 顯示學習率（如果指定）
                    if 'lr' in stage:
                        lr_value = float(stage['lr']) if isinstance(stage['lr'], (str, int)) else stage['lr']
                        logging.info(f"📉 Learning rate: {lr_value:.6f} (explicit reset)")
                    else:
                        logging.info(f"📉 Learning rate: inherited from previous stage + global scheduler")
                    logging.info("="*80)
                
                # 構建返回配置（只包含實際存在的參數）
                config_dict = {
                    'stage_name': stage['name'],
                    'is_transition': is_transition,
                    'weights': stage['weights'],
                    'nu': stage['nu'],
                    'sampling': stage['sampling'],
                }
                
                # lr 是可選參數（如果未指定，則繼續使用全域 scheduler）
                if 'lr' in stage:
                    config_dict['lr'] = stage['lr']
                
                # 根據場景添加特定參數
                if 'Re_tau' in stage:
                    config_dict['Re_tau'] = stage['Re_tau']
                if 'pressure_gradient' in stage:
                    config_dict['pressure_gradient'] = stage['pressure_gradient']
                if 'Re' in stage:
                    config_dict['Re'] = stage['Re']
                
                return config_dict
        
        # 超出範圍，返回最後階段
        last_stage = self.stages[-1]
        config_dict = {
            'stage_name': last_stage['name'],
            'is_transition': False,
            'weights': last_stage['weights'],
            'nu': last_stage['nu'],
            'sampling': last_stage['sampling'],
        }
        
        # lr 是可選參數
        if 'lr' in last_stage:
            config_dict['lr'] = last_stage['lr']
        
        # 根據場景添加特定參數
        if 'Re_tau' in last_stage:
            config_dict['Re_tau'] = last_stage['Re_tau']
        if 'pressure_gradient' in last_stage:
            config_dict['pressure_gradient'] = last_stage['pressure_gradient']
        if 'Re' in last_stage:
            config_dict['Re'] = last_stage['Re']
        
        return config_dict
    
    def _update_physics_parameters(self, stage: Dict[str, Any]):
        """更新物理方程模組的參數（避免梯度計算問題）"""
        import torch
        
        # nu 是 torch.nn.Buffer，使用 .data 來避免梯度追蹤
        if hasattr(self.physics, 'nu'):
            if isinstance(self.physics.nu, torch.Tensor):
                # 使用 .data.fill_() 來避免影響計算圖
                self.physics.nu.data = torch.tensor(stage['nu'], dtype=self.physics.nu.dtype, device=self.physics.nu.device)
            else:
                self.physics.nu = stage['nu']
        
        # Channel Flow 參數
        if hasattr(self.physics, 'Re_tau') and 'Re_tau' in stage:
            if isinstance(self.physics.Re_tau, torch.Tensor):
                self.physics.Re_tau.data = torch.tensor(stage['Re_tau'], dtype=self.physics.Re_tau.dtype, device=self.physics.Re_tau.device)
            else:
                self.physics.Re_tau = stage['Re_tau']
        if hasattr(self.physics, 'pressure_gradient') and 'pressure_gradient' in stage:
            if isinstance(self.physics.pressure_gradient, torch.Tensor):
                self.physics.pressure_gradient.data = torch.tensor(stage['pressure_gradient'], dtype=self.physics.pressure_gradient.dtype, device=self.physics.pressure_gradient.device)
            else:
                self.physics.pressure_gradient = stage['pressure_gradient']
        
        # Kolmogorov Flow 參數
        if hasattr(self.physics, 'Re') and 'Re' in stage:
            if isinstance(self.physics.Re, torch.Tensor):
                self.physics.Re.data = torch.tensor(stage['Re'], dtype=self.physics.Re.dtype, device=self.physics.Re.device)
            else:
                self.physics.Re = stage['Re']
        
        # 日誌輸出
        if 'Re_tau' in stage:
            logging.debug(f"✅ Physics parameters updated: Re_tau={stage['Re_tau']}, nu={stage['nu']}")
        elif 'Re' in stage:
            logging.debug(f"✅ Physics parameters updated: Re={stage['Re']}, nu={stage['nu']}")
        else:
            logging.debug(f"✅ Physics parameters updated: nu={stage['nu']}")

# 全域快取，用於存儲 Channel Flow 資料和統計資訊
_channel_data_cache: Optional[Dict[str, Any]] = None

# ============================================================================
# 訓練專用輔助函數（保留，未在模組中實現）
# ============================================================================


def create_loss_functions(config: Dict[str, Any], device: torch.device) -> Dict[str, nn.Module]:
    """建立損失函數"""
    loss_cfg = config.get('losses', {})
    physics_type = config.get('physics', {}).get('type', '')
    is_vs_cfg = physics_type == 'vs_pinn_channel_flow'
    base_weight_template, default_adaptive_terms = derive_loss_weights(
        loss_cfg,
        loss_cfg.get('prior_weight', 0.3),
        is_vs_cfg
    )
    
    losses = {
        'residual': NSResidualLoss(
            nu=loss_cfg.get('nu', 1e-3),
            density=loss_cfg.get('rho', 1.0),
            merge_momentum=loss_cfg.get('merge_momentum', False)  # 🔥 合併動量項（Kolmogorov Flow）
        ),
        'boundary': BoundaryConditionLoss(),  # 🆕 邊界條件損失（含 inlet）
        'prior': PriorLossManager(
            consistency_weight=loss_cfg['prior_weight']
        )
    }
    
    # ⭐ Phase 6C: 均值約束損失（可選）
    mean_constraint_cfg = loss_cfg.get('mean_constraint', {})
    if mean_constraint_cfg.get('enabled', False):
        losses['mean_constraint'] = MeanConstraintLoss()
        logging.info(f"✅ MeanConstraintLoss enabled with weight={mean_constraint_cfg.get('weight', 10.0)}")
        logging.info(f"   Target means: {mean_constraint_cfg.get('target_means', {})}")
    
    return losses


def create_weighters(config: Dict[str, Any], model: nn.Module, device: torch.device, physics=None) -> Dict[str, Any]:
    """建立動態權重器 (需要模型實例)"""
    loss_cfg = config.get('losses', {})
    physics_type = config.get('physics', {}).get('type', '')
    is_vs_cfg = physics_type == 'vs_pinn_channel_flow'
    base_weight_template, default_adaptive_terms = derive_loss_weights(
        loss_cfg,
        loss_cfg.get('prior_weight', 0.3),
        is_vs_cfg
    )
    weighters = {}
    
    # 🚀 課程訓練調度器（最高優先級）
    curriculum_cfg = config.get('curriculum', {})
    if curriculum_cfg.get('enable', False):
        stages = curriculum_cfg.get('stages', [])
        if stages and physics is not None:
            weighters['curriculum'] = CurriculumScheduler(stages, physics)
            logging.info(f"✅ Curriculum scheduler enabled with {len(stages)} stages")
            # 課程訓練啟用時，禁用其他「損失權重」調度器（但允許 LR scheduler）
            weighters['staged'] = None
            weighters['gradnorm'] = None
            weighters['causal'] = None
            logging.info("ℹ️  Other loss weight schedulers disabled (curriculum mode active)")
            logging.info("ℹ️  Global LR scheduler is allowed (can coexist with curriculum)")
        else:
            weighters['curriculum'] = None
            if not stages:
                logging.warning("⚠️  curriculum.enable=true but no stages defined")
            if physics is None:
                logging.warning("⚠️  curriculum requires physics module, falling back to staged weights")
    else:
        weighters['curriculum'] = None
    
    # 階段式權重調度器（優先級第二）
    if 'staged_weights' in loss_cfg and loss_cfg['staged_weights'].get('enable', False):
        phases = loss_cfg['staged_weights'].get('phases', [])
        if phases:
            weighters['staged'] = StagedWeightScheduler(phases)
            logging.info(f"✅ Staged weight scheduler enabled with {len(phases)} phases")
        else:
            weighters['staged'] = None
            logging.warning("⚠️  staged_weights.enable=true but no phases defined")
    else:
        weighters['staged'] = None
    
    # GradNorm 權重器（與階段式權重互斥）
    configured_terms = loss_cfg.get('adaptive_loss_terms')
    if configured_terms is not None:
        adaptive_terms = [name for name in configured_terms if name in base_weight_template]
    else:
        adaptive_terms = default_adaptive_terms
    if loss_cfg.get('adaptive_weighting', False) and weighters['staged'] is None and adaptive_terms:
        initial_weights = {name: base_weight_template.get(name, 1.0) for name in adaptive_terms}
        weighters['gradnorm'] = GradNormWeighter(
            model=model,
            loss_names=adaptive_terms,
            alpha=loss_cfg.get('grad_norm_alpha', 0.12),
            update_frequency=loss_cfg.get('weight_update_freq', 100),
            initial_weights=initial_weights,
            device=str(device),
            min_weight=loss_cfg.get('grad_norm_min_weight', 0.1),
            max_weight=loss_cfg.get('grad_norm_max_weight', 10.0)
        )
        logging.info("GradNorm adaptive weighting enabled")
    else:
        weighters['gradnorm'] = None
        if loss_cfg.get('adaptive_weighting', False) and weighters['staged'] is not None:
            logging.info("⚠️  adaptive_weighting disabled (using staged_weights)")
    
    # 因果權重器
    if loss_cfg.get('causal_weighting', False):
        # 🆕 從配置中提取時間範圍（支援 Kolmogorov 和其他數據源）
        time_range = config.get('data', {}).get('kolmogorov_config', {}).get('time_range')
        if time_range is None:
            # 回退：嘗試從 JHTDB 配置提取
            time_range = config.get('data', {}).get('jhtdb_config', {}).get('time_range')
        if time_range is None:
            # 默認範圍
            time_range = [0.0, 1.0]
            logging.warning(
                f"⚠️ Causal weighting enabled but no time_range found in config, "
                f"using default {time_range}"
            )
        
        weighters['causal'] = CausalWeighter(
            epsilon=loss_cfg.get('causal_eps', 1.0),
            n_time_bins=loss_cfg.get('causal_n_bins', 10),
            t_min=float(time_range[0]),
            t_max=float(time_range[1])
        )
        logging.info(
            f"✅ Causal weighting enabled: ε={loss_cfg.get('causal_eps', 1.0):.2f}, "
            f"bins={loss_cfg.get('causal_n_bins', 10)}, "
            f"t_range=[{time_range[0]}, {time_range[1]}]"
        )
    else:
        weighters['causal'] = None
    
    # 自適應權重調度器
    # 🔧 修復：僅在明確要求 phase_scheduling 時啟用（與 GradNorm 衝突）
    if loss_cfg.get('phase_scheduling', False) and weighters['staged'] is None and adaptive_terms:
        weighters['scheduler'] = AdaptiveWeightScheduler(
            loss_names=adaptive_terms
        )
        logging.info("Adaptive weight scheduler created")
    else:
        weighters['scheduler'] = None
        if not loss_cfg.get('phase_scheduling', False) and weighters['staged'] is None:
            logging.info("Adaptive weight scheduler disabled (use 'phase_scheduling: true' to enable)")
    
    return weighters


def prepare_training_data(config: Dict[str, Any], device: torch.device, config_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """準備訓練資料 - 支援 JHTDB Channel Flow 或 Mock 資料
    
    Args:
        config: 配置字典
        device: PyTorch 設備
        config_path: 配置檔案路徑（用於 ChannelFlowLoader）
    """
    
    # 檢查是否使用 JHTDB Channel Flow 載入器
    jhtdb_enabled = config.get('data', {}).get('jhtdb_config', {}).get('enabled', False)
    channel_flow_enabled = 'channel_flow' in config and config['channel_flow'].get('enabled', False)
    
    kolmogorov_enabled = config.get('data', {}).get('kolmogorov_config', {}).get('enabled', False)
    
    if kolmogorov_enabled:
        training_data = prepare_kolmogorov_training_data(config, device)
        # 載入 RANS 先驗（如果啟用）
        rans_prior = load_rans_prior_data(config, training_data, device)
        if rans_prior:
            training_data['lowfi_prior'] = rans_prior
            training_data['has_prior'] = True
        else:
            training_data['has_prior'] = False
        return training_data
    
    if jhtdb_enabled or channel_flow_enabled:
        return prepare_channel_flow_training_data(config, device, config_path)
    else:
        return prepare_mock_training_data(config, device)


def _apply_validation_split(
    training_dict: Dict[str, torch.Tensor],
    validation_split: float,
    is_vs_pinn: bool
) -> Dict[str, torch.Tensor]:
    """
    依據 validation_split 將感測點資料切分成訓練/驗證集合，並在 training_dict 中新增
    'validation' 索引。
    """
    if validation_split is None or validation_split <= 0.0:
        training_dict['validation'] = {'size': 0}
        return training_dict
    
    x_sensors = training_dict.get('x_sensors')
    if x_sensors is None or x_sensors.shape[0] == 0:
        training_dict['validation'] = {'size': 0}
        return training_dict
    
    n_total = x_sensors.shape[0]
    if n_total < 2:
        training_dict['validation'] = {'size': 0}
        return training_dict
    
    n_val = max(1, int(round(n_total * validation_split)))
    if n_val >= n_total:
        # 至少保留一個訓練感測點
        n_val = max(1, n_total - 1)
    if n_val <= 0:
        training_dict['validation'] = {'size': 0}
        return training_dict
    
    device = x_sensors.device
    perm = torch.randperm(n_total, device=device)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    
    if train_idx.numel() == 0:
        # 當 split 幾乎為 1.0 時，確保仍有訓練資料
        train_idx = val_idx[-1:].clone()
        val_idx = val_idx[:-1]
    
    def split_tensor(tensor: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if tensor is None:
            return None, None
        if tensor.shape[0] == 0:
            return tensor, tensor
        return tensor[train_idx], tensor[val_idx]
    
    x_train, x_val = split_tensor(training_dict.get('x_sensors'))
    y_train, y_val = split_tensor(training_dict.get('y_sensors'))
    z_train, z_val = split_tensor(training_dict.get('z_sensors'))
    t_train, t_val = split_tensor(training_dict.get('t_sensors'))
    u_train, u_val = split_tensor(training_dict.get('u_sensors'))
    v_train, v_val = split_tensor(training_dict.get('v_sensors'))
    w_train, w_val = split_tensor(training_dict.get('w_sensors'))
    p_train, p_val = split_tensor(training_dict.get('p_sensors'))
    
    # 更新訓練感測資料
    if x_train is not None:
        training_dict['x_sensors'] = x_train
    if y_train is not None:
        training_dict['y_sensors'] = y_train
    if z_train is not None:
        training_dict['z_sensors'] = z_train
    if t_train is not None:
        training_dict['t_sensors'] = t_train
    if u_train is not None:
        training_dict['u_sensors'] = u_train
    if v_train is not None:
        training_dict['v_sensors'] = v_train
    if w_train is not None:
        training_dict['w_sensors'] = w_train
    if p_train is not None:
        training_dict['p_sensors'] = p_train
    
    # 建立驗證資料
    if x_val is None or x_val.shape[0] == 0:
        training_dict['validation'] = {'size': 0}
        return training_dict
    
    coord_parts = [x_val, y_val]
    if is_vs_pinn and z_val is not None and z_val.shape[0] > 0:
        coord_parts.append(z_val)
    validation_coords = torch.cat(coord_parts, dim=1).detach()
    
    target_parts = [u_val, v_val]
    component_order = ['u', 'v']
    if is_vs_pinn:
        if w_val is not None and w_val.shape[0] == validation_coords.shape[0]:
            target_parts.append(w_val)
        else:
            target_parts.append(torch.zeros_like(u_val))
        component_order.append('w')
    target_parts.append(p_val)
    component_order.append('p')
    validation_targets = torch.cat(target_parts, dim=1).detach()
    
    if t_val is not None:
        validation_time = t_val.detach()
    else:
        validation_time = None
    
    training_dict['validation'] = {
        'coords': validation_coords,
        'targets': validation_targets,
        'time': validation_time,
        'components': component_order,
        'size': int(validation_coords.shape[0])
    }
    
    return training_dict


def prepare_mock_training_data(config: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """建立 Mock 訓練資料用於測試整合"""
    
    # 從配置中讀取參數
    K = config['sensors']['K']
    sampling = config['training']['sampling']
    physics_cfg = config['physics']
    domain = physics_cfg['domain']
    
    # 定義域範圍
    x_range = domain['x_range']
    y_range = domain['y_range']
    
    # 生成感測器點 (均勻分佈)
    x_sensors = torch.rand(K, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    y_sensors = torch.rand(K, 1, device=device) * (y_range[1] - y_range[0]) + y_range[0]
    t_sensors = torch.zeros_like(x_sensors)  # 假設穩態
    
    # 生成 Mock 速度和壓力資料 (基於解析解或簡單模式)
    # 簡單的通道流模式: u = U_max * (1 - (2y/H - 1)^2), v = 0, p = 線性分佈
    y_norm = (y_sensors - y_range[0]) / (y_range[1] - y_range[0])  # 歸一化到 [0,1]
    y_centered = 2 * y_norm - 1  # 歸一化到 [-1,1]
    
    u_max = 1.0  # 最大速度
    u_sensors = u_max * (1 - y_centered**2)  # 拋物線型速度分佈
    v_sensors = torch.zeros_like(u_sensors)   # 垂直速度為零
    p_sensors = torch.ones_like(u_sensors) * 0.1  # 簡單的壓力場
    
    # 生成 PDE 殘差點
    x_pde = torch.rand(sampling['pde_points'], 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    y_pde = torch.rand(sampling['pde_points'], 1, device=device) * (y_range[1] - y_range[0]) + y_range[0]
    t_pde = torch.zeros_like(x_pde)  # 穩態假設
    
    # 生成邊界點 (上下壁面)
    n_bc = sampling['boundary_points']
    x_bc = torch.rand(n_bc, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    y_bc_bottom = torch.full((n_bc//2, 1), y_range[0], device=device)  # 下壁面
    y_bc_top = torch.full((n_bc - n_bc//2, 1), y_range[1], device=device)  # 上壁面
    y_bc = torch.cat([y_bc_bottom, y_bc_top], dim=0)
    x_bc = torch.cat([x_bc[:n_bc//2], x_bc[n_bc//2:]], dim=0)
    t_bc = torch.zeros_like(x_bc)
    
    logging.info(f"Mock training data generated: K={K} sensors, {sampling['pde_points']} PDE points, {n_bc} BC points")
    
    training_dict = {
        'x_pde': x_pde, 'y_pde': y_pde, 't_pde': t_pde,
        'x_bc': x_bc, 'y_bc': y_bc, 't_bc': t_bc,
        'x_sensors': x_sensors, 'y_sensors': y_sensors, 't_sensors': t_sensors,
        'u_sensors': u_sensors, 'v_sensors': v_sensors, 'p_sensors': p_sensors
    }
    
    validation_split = config.get('training', {}).get('validation_split', 0.0)
    training_dict = _apply_validation_split(training_dict, validation_split, is_vs_pinn=False)
    
    return training_dict


def sample_boundary_points(
    n_points: int,
    domain_bounds: Dict[str, Tuple[float, float]],
    device: torch.device,
    distribution: Optional[Dict[str, int]] = None
) -> torch.Tensor:
    """
    在邊界上均勻採樣點
    
    Args:
        n_points: 總邊界點數
        domain_bounds: 域邊界 {'x': (x_min, x_max), 'y': (y_min, y_max), 'z': (z_min, z_max)}
        device: PyTorch device
        distribution: 邊界點分佈 {'wall': int, 'periodic': int, 'inlet': int}
                     如果為 None，預設為 {'wall': 1000, 'periodic': 800, 'inlet': 200}
    
    Returns:
        邊界點座標 [n_points, 3] (x, y, z)
    """
    if distribution is None:
        distribution = {'wall': 1000, 'periodic': 800, 'inlet': 200}
    
    # 驗證總點數
    total_requested = sum(distribution.values())
    if total_requested != n_points:
        logging.warning(f"⚠️ Boundary distribution sum ({total_requested}) != n_points ({n_points}), 自動調整比例")
        # 按比例調整
        scale = n_points / total_requested
        distribution = {k: int(v * scale) for k, v in distribution.items()}
        # 修正舍入誤差
        diff = n_points - sum(distribution.values())
        distribution['wall'] += diff
    
    x_min, x_max = domain_bounds['x']
    y_min, y_max = domain_bounds['y']
    z_min, z_max = domain_bounds['z']
    
    boundary_points = []
    
    # 1. 壁面點 (y = y_min 和 y = y_max)
    n_wall = distribution['wall']
    n_wall_bottom = n_wall // 2
    n_wall_top = n_wall - n_wall_bottom
    
    # 下壁面 (y = y_min)
    x_wall_bottom = torch.rand(n_wall_bottom, 1, device=device) * (x_max - x_min) + x_min
    y_wall_bottom = torch.full((n_wall_bottom, 1), y_min, device=device)
    z_wall_bottom = torch.rand(n_wall_bottom, 1, device=device) * (z_max - z_min) + z_min
    wall_bottom = torch.cat([x_wall_bottom, y_wall_bottom, z_wall_bottom], dim=1)
    
    # 上壁面 (y = y_max)
    x_wall_top = torch.rand(n_wall_top, 1, device=device) * (x_max - x_min) + x_min
    y_wall_top = torch.full((n_wall_top, 1), y_max, device=device)
    z_wall_top = torch.rand(n_wall_top, 1, device=device) * (z_max - z_min) + z_min
    wall_top = torch.cat([x_wall_top, y_wall_top, z_wall_top], dim=1)
    
    boundary_points.extend([wall_bottom, wall_top])
    
    # 2. 週期性邊界點 (x = x_min/x_max, z = z_min/z_max)
    n_periodic = distribution['periodic']
    n_per_face = n_periodic // 4  # 4 個面：x_min, x_max, z_min, z_max
    
    # x = x_min
    x_left = torch.full((n_per_face, 1), x_min, device=device)
    y_left = torch.rand(n_per_face, 1, device=device) * (y_max - y_min) + y_min
    z_left = torch.rand(n_per_face, 1, device=device) * (z_max - z_min) + z_min
    periodic_left = torch.cat([x_left, y_left, z_left], dim=1)
    
    # x = x_max
    x_right = torch.full((n_per_face, 1), x_max, device=device)
    y_right = torch.rand(n_per_face, 1, device=device) * (y_max - y_min) + y_min
    z_right = torch.rand(n_per_face, 1, device=device) * (z_max - z_min) + z_min
    periodic_right = torch.cat([x_right, y_right, z_right], dim=1)
    
    # z = z_min
    x_front = torch.rand(n_per_face, 1, device=device) * (x_max - x_min) + x_min
    y_front = torch.rand(n_per_face, 1, device=device) * (y_max - y_min) + y_min
    z_front = torch.full((n_per_face, 1), z_min, device=device)
    periodic_front = torch.cat([x_front, y_front, z_front], dim=1)
    
    # z = z_max
    x_back = torch.rand(n_per_face, 1, device=device) * (x_max - x_min) + x_min
    y_back = torch.rand(n_per_face, 1, device=device) * (y_max - y_min) + y_min
    z_back = torch.full((n_per_face, 1), z_max, device=device)
    periodic_back = torch.cat([x_back, y_back, z_back], dim=1)
    
    boundary_points.extend([periodic_left, periodic_right, periodic_front, periodic_back])
    
    # 3. Inlet 點 (x = x_min，特別處理)
    n_inlet = distribution['inlet']
    x_inlet = torch.full((n_inlet, 1), x_min, device=device)
    y_inlet = torch.rand(n_inlet, 1, device=device) * (y_max - y_min) + y_min
    z_inlet = torch.rand(n_inlet, 1, device=device) * (z_max - z_min) + z_min
    inlet = torch.cat([x_inlet, y_inlet, z_inlet], dim=1)
    
    boundary_points.append(inlet)
    
    # 合併所有邊界點
    all_boundary_points = torch.cat(boundary_points, dim=0)
    
    return all_boundary_points


def sample_interior_points(
    n_points: int,
    domain_bounds: Dict[str, Tuple[float, float]],
    device: torch.device,
    exclude_boundary_tol: float = 0.01,
    use_sobol: bool = True
) -> torch.Tensor:
    """
    在內部均勻採樣點，排除邊界區域
    
    Args:
        n_points: 內部點數
        domain_bounds: 域邊界 {'x': (x_min, x_max), 'y': (y_min, y_max), 'z': (z_min, z_max)}
        device: PyTorch device
        exclude_boundary_tol: 邊界排除容差（物理座標）
        use_sobol: 是否使用 Sobol 序列（更均勻）
    
    Returns:
        內部點座標 [n_points, 3] (x, y, z)
    """
    x_min, x_max = domain_bounds['x']
    y_min, y_max = domain_bounds['y']
    z_min, z_max = domain_bounds['z']
    
    # 調整內部域範圍（排除邊界容差）
    x_min_inner = x_min + exclude_boundary_tol
    x_max_inner = x_max - exclude_boundary_tol
    y_min_inner = y_min + exclude_boundary_tol
    y_max_inner = y_max - exclude_boundary_tol
    z_min_inner = z_min + exclude_boundary_tol
    z_max_inner = z_max - exclude_boundary_tol
    
    if use_sobol:
        # 使用 Sobol 序列（準均勻分佈）
        sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
        samples = sobol.draw(n_points).to(device)
        
        # 縮放到內部域
        x_interior = samples[:, 0:1] * (x_max_inner - x_min_inner) + x_min_inner
        y_interior = samples[:, 1:2] * (y_max_inner - y_min_inner) + y_min_inner
        z_interior = samples[:, 2:3] * (z_max_inner - z_min_inner) + z_min_inner
    else:
        # 使用均勻隨機採樣
        x_interior = torch.rand(n_points, 1, device=device) * (x_max_inner - x_min_inner) + x_min_inner
        y_interior = torch.rand(n_points, 1, device=device) * (y_max_inner - y_min_inner) + y_min_inner
        z_interior = torch.rand(n_points, 1, device=device) * (z_max_inner - z_min_inner) + z_min_inner
    
    interior_points = torch.cat([x_interior, y_interior, z_interior], dim=1)
    
    return interior_points


def prepare_channel_flow_training_data(config: Dict[str, Any], device: torch.device, config_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """使用 Channel Flow 載入器準備訓練資料
    
    Args:
        config: 配置字典
        device: PyTorch 設備
        config_path: 配置檔案路徑（傳遞給 ChannelFlowLoader）
    """
    from pinnx.dataio.channel_flow_loader import prepare_training_data as load_channel_flow
    
    # 載入 Channel Flow 資料 - 支援兩種配置格式
    if 'channel_flow' in config:
        cf_config = config['channel_flow']
        strategy = cf_config.get('strategy', 'qr_pivot')
    else:
        # 使用 JHTDB 配置格式 - 讀取 sensors.selection_method
        sensors_cfg = config.get('sensors', {})
        strategy = sensors_cfg.get('selection_method', 'qr_pivot')  # 支援從配置讀取策略
    
    K = config['sensors']['K']
    
    # 🆕 讀取自定義感測點文件名（如果有）
    sensor_file = config.get('sensors', {}).get('sensor_file', None)
    
    # 🆕 檢查是否為 3D 案例（決定是否請求 w 分量）
    is_3d = config.get('physics', {}).get('type') == 'vs_pinn_channel_flow'
    
    # 🆕 從配置讀取 data_variables（損失函數中定義的監督變數）
    if 'data_variables' in config.get('losses', {}):
        # 使用配置中明確指定的變數（例如：僅速度場訓練）
        target_fields = config['losses']['data_variables']
    else:
        # 預設行為：包含壓力場
        target_fields = ['u', 'v', 'w', 'p'] if is_3d else ['u', 'v', 'p']
    
    training_bundle = load_channel_flow(
        config_path=config_path,  # ⭐ 傳遞配置路徑給 ChannelFlowLoader
        strategy=strategy,
        K=K,
        target_fields=target_fields,
        sensor_file=sensor_file  # 傳遞自定義文件名
    )
    
    training_data = training_bundle.as_training_dict(
        target_fields=target_fields,
        device=device,
        include_w=is_3d
    )

    coords = training_data['coordinates']  # torch tensor on device
    sensor_data = training_data['sensor_data']
    domain_bounds = training_data['domain_bounds']
    # 計算標準化參數（VS-PINN 風格：映射到 [-1, 1]）
    x_range = domain_bounds['x']
    y_range = domain_bounds['y']
    x_min, x_max = x_range[0], x_range[1]
    y_min, y_max = y_range[0], y_range[1]
    if 'z' in domain_bounds:
        z_min, z_max = domain_bounds['z']
    else:
        z_min = z_max = 0.0
    
    def normalize_coord(coord, c_min, c_max):
        """將座標標準化到 [-1, 1]"""
        return 2.0 * (coord - c_min) / (c_max - c_min) - 1.0
    
    def denormalize_coord(coord_norm, c_min, c_max):
        """從 [-1, 1] 反標準化"""
        return (coord_norm + 1.0) / 2.0 * (c_max - c_min) + c_min
    
    # 🆕 檢查是否為 VS-PINN（需要 3D 坐標）
    is_vs_pinn = config.get('physics', {}).get('type') == 'vs_pinn_channel_flow'
    
    # 轉換為 PyTorch tensor
    x_sensors = coords[:, 0:1]
    y_sensors = coords[:, 1:2]
    
    # 🆕 如果是 VS-PINN 且有真實 z 座標，使用它；否則為 0
    if is_vs_pinn:
        z_sensors = coords[:, 2:3]
    else:
        z_sensors = torch.zeros_like(x_sensors)
    t_sensors = torch.zeros_like(x_sensors)  # 暫時假設 t=0
    
    u_sensors = sensor_data['u']
    v_sensors = sensor_data['v']
    
    # 🆕 壓力場可能不存在（僅速度場訓練）
    p_sensors = sensor_data.get('p')
    if p_sensors is None:
        # 如果沒有壓力數據，初始化為零（由 PINN 從速度場推導）
        p_sensors = torch.zeros_like(u_sensors)
    
    # 🆕 如果是 VS-PINN，添加 w 分量（假設為 0 或從數據中獲取）
    if is_vs_pinn:
        w_sensors = sensor_data.get('w')
        if w_sensors is None:
            w_sensors = torch.zeros_like(u_sensors)
    else:
        w_sensors = None  # 2D 不需要 w
    
    # 生成 PDE 殘差點和邊界點
    sampling = config['training']['sampling']
    
    # 🆕 檢查是否使用分層採樣策略
    use_stratified = sampling.get('strategy', 'uniform') == 'stratified'
    
    if use_stratified:
        # === 分層採樣策略 ===
        logging.info("📊 使用分層採樣策略 (stratified sampling)")
        
        # 構建域邊界字典（物理座標）
        bounds_dict = {
            'x': (x_min, x_max),
            'y': (y_min, y_max),
            'z': (z_min, z_max) if is_vs_pinn else (0.0, 0.0)
        }
        
        # 獲取邊界點分佈配置
        boundary_dist = sampling.get('boundary_distribution', {
            'wall': 1000, 
            'periodic': 800, 
            'inlet': 200
        })
        
        # 生成邊界點（物理座標）
        n_bc = sampling.get('boundary_points', 2000)
        boundary_points_raw = sample_boundary_points(
            n_points=n_bc,
            domain_bounds=bounds_dict,
            device=device,
            distribution=boundary_dist
        )
        
        # 🔧 保持物理座標（由 ManualScalingWrapper 負責標準化）
        x_bc = boundary_points_raw[:, 0:1]
        y_bc = boundary_points_raw[:, 1:2]
        z_bc = boundary_points_raw[:, 2:3] if is_vs_pinn else torch.zeros_like(x_bc)
        t_bc = torch.zeros_like(x_bc)
        
        # 生成內部 PDE 點（物理座標）
        n_pde = sampling.get('interior_points', 10000)
        use_sobol = sampling.get('use_sobol', True)
        exclude_tol = sampling.get('boundary_tolerance', 0.01)
        
        interior_points_raw = sample_interior_points(
            n_points=n_pde,
            domain_bounds=bounds_dict,
            device=device,
            exclude_boundary_tol=exclude_tol,
            use_sobol=use_sobol
        )
        
        # 🔧 保持物理座標（由 ManualScalingWrapper 負責標準化）
        x_pde = interior_points_raw[:, 0:1]
        y_pde = interior_points_raw[:, 1:2]
        z_pde = interior_points_raw[:, 2:3] if is_vs_pinn else torch.zeros_like(x_pde)
        t_pde = torch.zeros_like(x_pde)
        
        logging.info(f"✅ 分層採樣完成: {n_bc} 邊界點 + {n_pde} 內部點")
        logging.info(f"   - 邊界分佈: {boundary_dist}")
        logging.info(f"   - Sobol 採樣: {use_sobol}, 邊界容差: {exclude_tol}")
        
    else:
        # === 原始均勻隨機採樣 ===
        logging.info("📊 使用均勻隨機採樣策略 (uniform sampling)")
        
        # PDE 殘差點（原始座標）
        x_pde_raw = torch.rand(sampling['pde_points'], 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
        y_pde_raw = torch.rand(sampling['pde_points'], 1, device=device) * (y_range[1] - y_range[0]) + y_range[0]
        
        # 🔧 保持物理座標（由 ManualScalingWrapper 負責標準化）
        x_pde = x_pde_raw
        y_pde = y_pde_raw
        t_pde = torch.zeros_like(x_pde)  # 穩態假設
        
        # 🆕 如果是 VS-PINN，添加 z 座標到 PDE 點
        if is_vs_pinn:
            z_pde_raw = torch.rand(sampling['pde_points'], 1, device=device) * (z_max - z_min) + z_min
            z_pde = z_pde_raw
        else:
            z_pde = torch.zeros_like(x_pde)  # 2D 情況下 z=0
        
        # 邊界點（原始座標）
        n_bc = sampling['boundary_points']
        x_bc_raw = torch.rand(n_bc, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
        y_bc_bottom_raw = torch.full((n_bc//2, 1), y_range[0], device=device)  # 下壁面
        y_bc_top_raw = torch.full((n_bc - n_bc//2, 1), y_range[1], device=device)  # 上壁面
        y_bc_raw = torch.cat([y_bc_bottom_raw, y_bc_top_raw], dim=0)
        x_bc_raw = torch.cat([x_bc_raw[:n_bc//2], x_bc_raw[n_bc//2:]], dim=0)
        
        # 🔧 保持物理座標（由 ManualScalingWrapper 負責標準化）
        x_bc = x_bc_raw
        y_bc = y_bc_raw
        t_bc = torch.zeros_like(x_bc)
        
        # 🆕 如果是 VS-PINN，添加 z 座標到邊界點
        if is_vs_pinn:
            z_bc_raw = torch.rand(n_bc, 1, device=device) * (z_max - z_min) + z_min
            z_bc = z_bc_raw
        else:
            z_bc = torch.zeros_like(x_bc)  # 2D 情況下 z=0
    
    # 存儲額外資訊到全局變量（包含標準化參數）
    global _channel_data_cache
    _channel_data_cache = {
        'domain_bounds': domain_bounds,
        'bundle': training_bundle,
        'training_dict': training_data,
        'normalization': {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'z_min': z_min, 'z_max': z_max,  # 🆕 添加 z 範圍
            'normalize_fn': normalize_coord,
            'denormalize_fn': denormalize_coord
        },
        'is_vs_pinn': is_vs_pinn  # 🆕 標記是否為 VS-PINN
    }
    
    # 🆕 生成初始條件點（t=0）- 使用感測器數據作為 IC
    ic_config = config.get('initial_condition', {})
    if ic_config.get('enabled', False):
        n_ic = ic_config.get('n_points', 256)
        # 從感測器數據中隨機採樣作為 IC（或使用完整感測器數據）
        if n_ic >= len(x_sensors):
            # 使用所有感測器數據
            x_ic = x_sensors.clone()
            y_ic = y_sensors.clone()
            z_ic = z_sensors.clone()  # 🆕 添加 z_ic
            t_ic = torch.zeros_like(x_ic)
            u_ic = u_sensors.clone()
            v_ic = v_sensors.clone()
            p_ic = p_sensors.clone()
            if is_vs_pinn:
                w_ic = w_sensors.clone() if w_sensors is not None else torch.zeros_like(u_ic)
            else:
                w_ic = torch.empty(0, 1, device=device)
        else:
            # 隨機採樣
            ic_indices = torch.randperm(len(x_sensors), device=device)[:n_ic]
            x_ic = x_sensors[ic_indices]
            y_ic = y_sensors[ic_indices]
            z_ic = z_sensors[ic_indices]  # 🆕 添加 z_ic
            t_ic = torch.zeros_like(x_ic)
            u_ic = u_sensors[ic_indices]
            v_ic = v_sensors[ic_indices]
            p_ic = p_sensors[ic_indices]
            if is_vs_pinn:
                w_ic = w_sensors[ic_indices] if w_sensors is not None else torch.zeros_like(u_ic)
            else:
                w_ic = torch.empty(0, 1, device=device)
    else:
        # IC 禁用時，使用空張量
        x_ic = torch.empty(0, 1, device=device)
        y_ic = torch.empty(0, 1, device=device)
        z_ic = torch.empty(0, 1, device=device)  # 🆕 添加 z_ic
        t_ic = torch.empty(0, 1, device=device)
        u_ic = torch.empty(0, 1, device=device)
        v_ic = torch.empty(0, 1, device=device)
        w_ic = torch.empty(0, 1, device=device)  # 🆕 添加 w_ic
        p_ic = torch.empty(0, 1, device=device)
    
    # 提取低保真先驗資料 (如果有)
    # ⚠️ 重要：變量順序必須與物理模組輸出順序一致 ['u', 'v', 'w', 'p']
    training_dict = {
        'x_pde': x_pde, 'y_pde': y_pde, 'z_pde': z_pde, 't_pde': t_pde,  # 🆕 添加 z_pde
        'x_bc': x_bc, 'y_bc': y_bc, 'z_bc': z_bc, 't_bc': t_bc,  # 🆕 添加 z_bc
        'x_sensors': x_sensors, 'y_sensors': y_sensors, 'z_sensors': z_sensors, 't_sensors': t_sensors,  # 🆕 添加 z_sensors
        'u_sensors': u_sensors,
        'v_sensors': v_sensors,
        'w_sensors': w_sensors if (is_vs_pinn and w_sensors is not None) else torch.empty(0, 1, device=device),
        'p_sensors': p_sensors,
        'x_ic': x_ic, 'y_ic': y_ic, 'z_ic': z_ic, 't_ic': t_ic,  # 🆕 添加 z_ic
        'u_ic': u_ic,
        'v_ic': v_ic,
        'w_ic': w_ic if is_vs_pinn else torch.empty(0, 1, device=device),
        'p_ic': p_ic,
        'metadata': training_data.get('metadata', {}),
        'statistics': training_data.get('statistics', {})
    }
    
    # 添加低保真先驗資料到批次 (如果可用)
    if training_data['has_prior']:
        lowfi = training_data.get('lowfi_prior', {}) or {}
        if 'u' in lowfi:
            training_dict['u_prior'] = lowfi['u']
        if 'v' in lowfi:
            training_dict['v_prior'] = lowfi['v']
        if 'p' in lowfi:
            training_dict['p_prior'] = lowfi['p']
        training_dict['has_prior'] = True
    else:
        training_dict['has_prior'] = False
    
    validation_split = config.get('training', {}).get('validation_split', 0.0)
    training_dict = _apply_validation_split(training_dict, validation_split, is_vs_pinn=is_vs_pinn)
    
    return training_dict





def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description='PINNs Inverse Training Script')
    parser.add_argument('--cfg', type=str, default='configs/defaults.yml',
                       help='Path to configuration file')
    parser.add_argument('--ensemble', action='store_true',
                       help='Run ensemble training for UQ')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 載入配置
    config = load_config(args.cfg)
    
    # 🆕 根據物理類型自動設置模型輸入輸出維度
    if 'in_dim' not in config['model'] or 'out_dim' not in config['model']:
        physics_type = config.get('physics', {}).get('type', 'ns_2d')
        
        if physics_type == 'vs_pinn_channel_flow':
            # VS-PINN 3D: 輸入 (x, y, z)，輸出 (u, v, w, p)
            config['model']['in_dim'] = 3
            config['model']['out_dim'] = 4
            logger_msg = "VS-PINN 3D: in_dim=3 (x,y,z), out_dim=4 (u,v,w,p)"
        else:
            # 標準 PINN 2D: 輸入 (x, y)，輸出 (u, v, p)
            config['model']['in_dim'] = 2
            config['model']['out_dim'] = 3
            logger_msg = "Standard PINN 2D: in_dim=2 (x,y), out_dim=3 (u,v,p)"
        
        # 暫時用 print（因為 logger 尚未設置）
        print(f"🔧 Auto-configured model dimensions: {logger_msg}")
    
    # 設置日誌
    # 從配置讀取日誌目錄，若未指定則使用預設
    log_dir = config.get('logging', {}).get('log_dir', './log')
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日誌檔案路徑
    exp_name = config['experiment']['name']
    log_file = os.path.join(log_dir, 'training.log')
    
    # 設置日誌系統
    log_level = config['logging'].get('log_level', config['logging'].get('level', 'INFO'))
    logger = setup_logging(level=log_level, log_file=log_file)
    logger.info("=" * 60)
    logger.info("PINNs Inverse Reconstruction Training")
    logger.info("=" * 60)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Log file: {log_file}")
    
    # 設置重現性
    set_random_seed(
        config['experiment']['seed'],
        config['reproducibility']['deterministic']
    )
    
    # 設置設備
    device = get_device(config['experiment']['device'])
    
    # 提前準備資料以提取統計資訊（用於自動輸出範圍）
    logger.info("Preparing training data to extract statistics...")
    training_data_sample = prepare_training_data(config, device, args.cfg)
    
    # 從快取中提取統計資訊（如果可用）
    statistics = None
    if '_channel_data_cache' in globals() and _channel_data_cache is not None:
        cached_bundle = _channel_data_cache.get('bundle')
        if cached_bundle and cached_bundle.statistics:
            statistics = cached_bundle.statistics
            logger.info("✅ Extracted statistics for auto output ranges:")
            logger.info(f"   u: {statistics.get('u', {}).get('range', 'N/A')}")
            logger.info(f"   v: {statistics.get('v', {}).get('range', 'N/A')}")
            logger.info(f"   p: {statistics.get('p', {}).get('range', 'N/A')}")
        else:
            logger.warning("⚠️  No statistics found in cached training bundle")
    else:
        logger.warning("⚠️  Channel data cache not available, will use hardcoded ranges")
    
    # 建立模型和物理模組
    model = create_model(config, device, statistics=statistics)
    physics = create_physics(config, device)
    losses = create_loss_functions(config, device)
    
    physics_type = config.get('physics', {}).get('type', 'unknown')
    is_vs_pinn = physics_type == 'vs_pinn_channel_flow'
    input_normalizer = create_input_normalizer(config, training_data_sample, is_vs_pinn, device)
    
    logger.info(f"Model architecture: {config['model']['type']}")
    logger.info(f"Input dimension: {config['model']['in_dim']}")
    logger.info(f"Output dimension: {config['model']['out_dim']}")
    
    # 安全讀取物理參數
    if physics_type == 'vs_pinn_channel_flow':
        physics_params = config.get('physics', {}).get('physics_params', {})
        logger.info(f"Physics: VS-PINN Channel Flow with nu={physics_params.get('nu', 'N/A')}")
    else:
        nu = config.get('physics', {}).get('nu', config.get('physics', {}).get('physics_params', {}).get('nu', 'N/A'))
        logger.info(f"Physics: NS-2D with nu={nu}")
    
    if args.ensemble:
        logger.info("Running ensemble training...")
        ensemble_cfg = config['ensemble']
        
        models = []
        for i, seed in enumerate(ensemble_cfg['seeds']):
            logger.info(f"Training ensemble member {i+1}/{len(ensemble_cfg['seeds'])} (seed={seed})")
            
            # 重置隨機種子
            set_random_seed(seed, config['reproducibility']['deterministic'])
            
            # 建立新模型（使用相同的統計資訊）
            member_model = create_model(config, device, statistics=statistics)
            
            # 創建動態權重器（GradNorm/Causal/Curriculum）
            member_weighters = create_weighters(config, member_model, device, physics=physics)
            
            # 使用 Trainer 訓練（傳遞 training_data 以支援自動計算標準化統計量）
            trainer = Trainer(member_model, physics, losses, config, device,
                               weighters=member_weighters,
                               input_normalizer=input_normalizer,
                               training_data=training_data_sample)
            trainer.training_data = training_data_sample
            
            # ✅ 從訓練資料計算標準化統計量（若配置要求但 params 為空）
            if config.get('data', {}).get('normalize', False):
                norm_cfg = config.get('normalization', {})
                if norm_cfg.get('type') == 'training_data_norm' and not norm_cfg.get('params'):
                    # 從感測點數據計算 Z-score
                    sensor_data = {
                        'u': training_data_sample['u_sensors'],
                        'v': training_data_sample['v_sensors'],
                        'p': training_data_sample['p_sensors']
                    }
                    if 'w_sensors' in training_data_sample:
                        sensor_data['w'] = training_data_sample['w_sensors']
                    
                    from pinnx.utils.normalization import DataNormalizer
                    trainer.data_normalizer = DataNormalizer.from_data(
                        sensor_data, 
                        norm_type='training_data_norm'
                    )
                    logger.info(f"✅ 已從訓練資料計算標準化統計量（Ensemble member {i+1}）: {trainer.data_normalizer}")
            
            # 載入 checkpoint（若指定）
            if args.resume:
                logger.info(f"⏮️  載入 checkpoint: {args.resume} (Ensemble member {i+1})")
                trainer.load_checkpoint(args.resume)
            
            train_result = trainer.train()
            models.append(member_model)
            
            logger.info(f"Member {i+1} final loss: {train_result['final_loss']:.6f}")
        
        # 儲存模型列表（暫時不使用 EnsemblePINNWrapper）
        logger.info(f"Ensemble training completed with {len(models)} members")
        logger.info("Note: EnsemblePINNWrapper integration pending - models stored as list")
        
    else:
        logger.info("Running single model training...")
        weighters = create_weighters(config, model, device, physics=physics)
        trainer = Trainer(model, physics, losses, config, device,
                          weighters=weighters,
                          input_normalizer=input_normalizer,
                          training_data=training_data_sample)
        trainer.training_data = training_data_sample
        
        # ✅ 從訓練資料計算標準化統計量（若配置要求但 params 為空）
        if config.get('data', {}).get('normalize', False):
            norm_cfg = config.get('normalization', {})
            if norm_cfg.get('type') == 'training_data_norm' and not norm_cfg.get('params'):
                # 從感測點數據計算 Z-score
                sensor_data = {
                    'u': training_data_sample['u_sensors'],
                    'v': training_data_sample['v_sensors'],
                    'p': training_data_sample['p_sensors']
                }
                if 'w_sensors' in training_data_sample:
                    sensor_data['w'] = training_data_sample['w_sensors']
                
                from pinnx.utils.normalization import DataNormalizer
                trainer.data_normalizer = DataNormalizer.from_data(
                    sensor_data, 
                    norm_type='training_data_norm'
                )
                logger.info(f"✅ 已從訓練資料計算標準化統計量: {trainer.data_normalizer}")
        
        # 載入 checkpoint（若指定）
        if args.resume:
            logger.info(f"⏮️  載入 checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        train_result = trainer.train()
        logger.info(f"Training completed. Final loss: {train_result['final_loss']:.6f}")
    
    logger.info("Training script finished successfully!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PINNs 逆重建主訓練腳本
負責協調資料載入、模型建立、訓練迴圈與評估輸出

支援功能：
- 單 GPU 訓練
- 多 GPU DDP 訓練（自動偵測）
- 混合精度訓練 (AMP)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# 導入 pinnx 以觸發 GPU 環境偵測
import pinnx

from pinnx.dataio.loaders.kolmogorov import prepare_kolmogorov_training_data
from pinnx.dataio.loaders.rans_prior import load_rans_prior_data
from pinnx.dataio.sampling import sample_boundary_points, sample_interior_points
from pinnx.train.config_loader import load_config
from pinnx.train.loss_factory import create_loss_functions
from pinnx.train.model_physics_factory import create_model, create_physics, get_device
from pinnx.train.trainer_builder import TrainerBuilder  # ✨ P2-3: TrainerBuilder
from pinnx.train.weighter_factory import create_weighters
from pinnx.utils.normalization_helpers import create_input_normalizer, setup_output_normalization
from pinnx.utils.setup import set_random_seed, setup_logging


# ============================================================================
# 🚀 DDP 初始化與環境設定
# ============================================================================
def init_distributed_mode():
    """
    初始化分散式訓練環境
    
    Returns:
        dict: 分散式訓練配置
            - is_distributed: 是否啟用分散式訓練
            - rank: 全局排名
            - local_rank: 本地排名
            - world_size: 總程序數
            - device: 當前程序使用的裝置
    """
    # 從 pinnx.Config 讀取自動偵測的環境資訊
    use_ddp = pinnx.Config.use_ddp
    
    if not use_ddp:
        return {
            'is_distributed': False,
            'rank': 0,
            'local_rank': 0,
            'world_size': 1,
            'device': torch.device(pinnx.Config.default_device)
        }
    
    # 初始化 DDP
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 由 torchrun 或 torch.distributed.launch 啟動
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # 手動啟動多 GPU 訓練
        rank = 0
        world_size = pinnx.Config.world_size
        local_rank = 0
        
        # 設定環境變數
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
    
    # 設定當前程序的裝置
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    # 初始化程序組
    backend = pinnx.Config.ddp_backend or 'nccl'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    if rank == 0:
        logging.info(f"🚀 DDP 初始化完成:")
        logging.info(f"   Backend: {backend}")
        logging.info(f"   World Size: {world_size}")
        logging.info(f"   Devices: {list(range(world_size))}")
    
    return {
        'is_distributed': True,
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'device': device
    }


def cleanup_distributed():
    """清理分散式訓練環境"""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# 🚨 修復：添加數據質量驗證
# ============================================================================
def _collect_coordinate_tensors(training_data: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    prefixes = ['sensors', 'pde', 'bc', 'ic']
    coords: List[torch.Tensor] = []
    for prefix in prefixes:
        spatial_key = f'coords_{prefix}_spatial'
        if spatial_key not in training_data:
            continue
        spatial = training_data[spatial_key]
        if not isinstance(spatial, torch.Tensor) or spatial.numel() == 0:
            continue
        t_key = f't_{prefix}'
        if t_key in training_data and training_data[t_key] is not None and training_data[t_key].numel() > 0:
            coords.append(torch.cat([spatial, training_data[t_key]], dim=1))
        else:
            coords.append(spatial)
    return coords


# ============================================================================
# 調度器定義已移至 pinnx.train.schedulers
# ============================================================================
# 全域快取，用於存儲 Channel Flow 資料和統計資訊
_channel_data_cache: Optional[Dict[str, Any]] = None

# ============================================================================
# 訓練專用輔助函數（保留，未在模組中實現）
# ============================================================================

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
    training_dict: Dict[str, Any],
    validation_split: float,
    is_vs_pinn: bool
) -> Dict[str, Any]:
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
    
    target_parts: List[torch.Tensor] = []
    component_order = []
    if u_val is not None:
        target_parts.append(u_val)
        component_order.append('u')
    if v_val is not None:
        target_parts.append(v_val)
        component_order.append('v')
    if is_vs_pinn:
        if w_val is not None and w_val.shape[0] == validation_coords.shape[0]:
            target_parts.append(w_val)
        elif u_val is not None:
            target_parts.append(torch.zeros_like(u_val))
        component_order.append('w')
    if p_val is not None:
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
    x_pde = torch.rand(sampling['N_pde'], 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    y_pde = torch.rand(sampling['N_pde'], 1, device=device) * (y_range[1] - y_range[0]) + y_range[0]
    t_pde = torch.zeros_like(x_pde)  # 穩態假設
    
    # 生成邊界點 (上下壁面)
    n_bc = sampling['boundary_points']
    x_bc = torch.rand(n_bc, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    y_bc_bottom = torch.full((n_bc//2, 1), y_range[0], device=device)  # 下壁面
    y_bc_top = torch.full((n_bc - n_bc//2, 1), y_range[1], device=device)  # 上壁面
    y_bc = torch.cat([y_bc_bottom, y_bc_top], dim=0)
    x_bc = torch.cat([x_bc[:n_bc//2], x_bc[n_bc//2:]], dim=0)
    t_bc = torch.zeros_like(x_bc)
    
    logging.info(f"Mock training data generated: K={K} sensors, {sampling['N_pde']} PDE points, {n_bc} BC points")
    
    coords_pde_spatial = torch.cat([x_pde, y_pde], dim=1)
    coords_bc_spatial = torch.cat([x_bc, y_bc], dim=1)
    coords_sensors_spatial = torch.cat([x_sensors, y_sensors], dim=1)

    training_dict = {
        'coords_pde_spatial': coords_pde_spatial,
        'coords_bc_spatial': coords_bc_spatial,
        'coords_sensors_spatial': coords_sensors_spatial,
        'x_pde': x_pde, 'y_pde': y_pde, 't_pde': t_pde,
        'x_bc': x_bc, 'y_bc': y_bc, 't_bc': t_bc,
        'x_sensors': x_sensors, 'y_sensors': y_sensors, 't_sensors': t_sensors,
        'u_sensors': u_sensors, 'v_sensors': v_sensors, 'p_sensors': p_sensors
    }
    
    validation_split = config.get('training', {}).get('validation_split', 0.0)
    training_dict = _apply_validation_split(training_dict, validation_split, is_vs_pinn=False)
    
    return training_dict


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
    
    # ✅ 檢查是否啟用 lowfi_prior (RANS)
    lowfi_cfg = config.get('lowfi_prior', {})
    if lowfi_cfg.get('enabled', False):
        prior_type = 'rans'  # 使用 RANS prior
    else:
        prior_type = 'none'  # 僅使用 sensor data
    
    training_bundle = load_channel_flow(
        config_path=config_path,  # ⭐ 傳遞配置路徑給 ChannelFlowLoader
        strategy=strategy,
        K=K,
        target_fields=target_fields,
        sensor_file=sensor_file,  # 傳遞自定義文件名
        prior_type=prior_type  # ✅ NEW: 傳遞 prior_type
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
    
    # ========== 座標維度一致性檢查（Coordinate Dimension Consistency） ==========
    # 檢查 z 座標是否與物理模式一致
    # 注意：檢查「變化」而非「大小」，使用 std 判斷是否為常數
    coords_z_is_constant = coords.shape[1] >= 3 and coords[:, 2].std().item() < 1e-6
    coords_has_varying_z = coords.shape[1] >= 3 and not coords_z_is_constant
    
    if is_vs_pinn:
        # VS-PINN (3D) 模式：期望 z 座標有變化
        if coords_z_is_constant:
            z_mean = coords[:, 2].mean().item()
            logging.warning(
                f"⚠️ VS-PINN (3D) 模式但 z 座標為常數 (z={z_mean:.4f})。\n"
                f"   這可能表示:\n"
                f"   1. 資料來自 2D 切片且固定 z (驗證 z_default={z_mean:.4f} 是否正確)\n"
                f"   2. 配置錯誤 (這應該是 2D 模式嗎?)\n"
                f"   如果這是刻意的 2D 切片資料，建議設定 physics.type 為非 VS-PINN。"
            )
    else:
        # 2D 模式：期望 z 座標為常數或不重要
        if coords_has_varying_z:
            z_min = coords[:, 2].min().item()
            z_max = coords[:, 2].max().item()
            logging.warning(
                f"⚠️ 2D 物理模式但 z 座標有變化 (範圍: [{z_min:.4f}, {z_max:.4f}])。\n"
                f"   Z 值將被忽略 (強制為零)。\n"
                f"   如果這是 3D 資料，建議:\n"
                f"   1. 設定 physics.type: 'vs_pinn_channel_flow' 以支援 3D\n"
                f"   2. 若確實為 2D 問題，重新生成感測器資料時進行 2D 提取"
            )
    # ====================================================================
    
    u_sensors = sensor_data['u']
    v_sensors = sensor_data['v']
    
    # ========== 壓力場缺失處理（Pressure Field Handling） ==========
    # 檢查壓力場是否存在
    p_sensors = sensor_data.get('p')
    if p_sensors is None:
        # 判斷物理類型是否為壓力驅動流
        physics_config = config.get('physics', {})
        physics_type = physics_config.get('type', '')
        is_pressure_driven = physics_config.get('pressure_driven', False)
        
        # 檢查是否強制要求壓力場（默認：壓力驅動流需要壓力場）
        enforce_pressure_data = config.get('training', {}).get('enforce_pressure_data', is_pressure_driven)
        
        if enforce_pressure_data:
            # 壓力驅動流必須提供壓力場
            raise ValueError(
                f"❌ 壓力場資料缺失錯誤\n"
                f"   物理類型: '{physics_type}' (pressure_driven={is_pressure_driven})\n"
                f"   壓力驅動流必須提供壓力場資料（sensor_data['p']）。\n"
                f"\n"
                f"   可能的解決方案：\n"
                f"   1. 確保感測器 NPZ 檔案包含壓力欄位（'p', 'sensor_p', 或 'pressure'）\n"
                f"   2. 重新生成感測器資料（使用 scripts/generate/sensors/）\n"
                f"   3. 如果這是速度驅動流，請在 config 中設定：\n"
                f"      physics:\n"
                f"        pressure_driven: false\n"
                f"      或\n"
                f"      training:\n"
                f"        enforce_pressure_data: false  # 不推薦，會降低訓練效率\n"
            )
        else:
            # 速度驅動流或用戶明確允許缺失壓力場
            if is_pressure_driven:
                logging.warning(
                    f"⚠️  壓力驅動流缺少壓力場資料！\n"
                    f"    初始化為零向量（將由 PINN 從速度場與 PDE 推導）。\n"
                    f"    這會顯著降低訓練效率與收斂速度。\n"
                    f"    強烈建議提供真實壓力場資料。"
                )
            else:
                logging.info(
                    f"ℹ️  壓力場未提供（速度驅動流或用戶允許），初始化為零。\n"
                    f"    壓力場將由 PINN 從速度場與 PDE 殘差推導。"
                )
            p_sensors = torch.zeros_like(u_sensors)
    else:
        logging.info(f"✅ 壓力場資料已載入：shape={p_sensors.shape}, range=[{p_sensors.min():.4f}, {p_sensors.max():.4f}]")
    # ================================================================
    
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
        x_pde_raw = torch.rand(sampling['N_pde'], 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
        y_pde_raw = torch.rand(sampling['N_pde'], 1, device=device) * (y_range[1] - y_range[0]) + y_range[0]
        
        # 🔧 保持物理座標（由 ManualScalingWrapper 負責標準化）
        x_pde = x_pde_raw
        y_pde = y_pde_raw
        t_pde = torch.zeros_like(x_pde)  # 穩態假設
        
        # 🆕 如果是 VS-PINN，添加 z 座標到 PDE 點
        if is_vs_pinn:
            z_pde_raw = torch.rand(sampling['N_pde'], 1, device=device) * (z_max - z_min) + z_min
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
    
    # ==================== 🚀 Wave 1-2 優化：預拼接座標 ====================
    # 優化目標：消除 trainer.py::step() 中每步重複的 torch.cat 操作
    # 預期效益：減少 10-15% 訓練時間（每步節省 3-12ms）
    
    # 注意：我們只拼接空間座標，不包含時間維度
    # 原因：(1) 大多數配置是穩態（時間維度不需要）
    #      (2) 時間維度是否加入取決於 model_input_dim，應由 trainer 決定
    #      (3) 保持時間維度由 trainer 決定
    
    # 構建 PDE 點預拼接座標（僅空間維度）
    spatial_pde = [x_pde, y_pde]
    if z_pde is not None and z_pde.numel() > 0 and not torch.all(z_pde == 0):
        spatial_pde.append(z_pde)
    coords_pde_spatial = torch.cat(spatial_pde, dim=1)  # [N_pde, 2/3]
    
    # 構建邊界點預拼接座標（僅空間維度）
    spatial_bc = [x_bc, y_bc]
    if z_bc is not None and z_bc.numel() > 0 and not torch.all(z_bc == 0):
        spatial_bc.append(z_bc)
    coords_bc_spatial = torch.cat(spatial_bc, dim=1)  # [N_bc, 2/3]
    
    # 構建感測器點預拼接座標（僅空間維度）
    spatial_sensors = [x_sensors, y_sensors]
    if z_sensors is not None and z_sensors.numel() > 0 and not torch.all(z_sensors == 0):
        spatial_sensors.append(z_sensors)
    coords_sensors_spatial = torch.cat(spatial_sensors, dim=1)  # [N_sensors, 2/3]
    
    # 構建初始條件點預拼接座標（如果有，僅空間維度）
    if x_ic.numel() > 0:
        spatial_ic = [x_ic, y_ic]
        if z_ic is not None and z_ic.numel() > 0 and not torch.all(z_ic == 0):
            spatial_ic.append(z_ic)
        coords_ic_spatial = torch.cat(spatial_ic, dim=1)  # [N_ic, 2/3]
    else:
        coords_ic_spatial = torch.empty(0, 2 if not is_vs_pinn else 3, device=device)
    
    # 提取低保真先驗資料 (如果有)
    # ⚠️ 重要：變量順序必須與物理模組輸出順序一致 ['u', 'v', 'w', 'p']
    training_dict = {
        # 🆕 預拼接座標（優先使用）- 僅空間維度
        'coords_pde_spatial': coords_pde_spatial,
        'coords_bc_spatial': coords_bc_spatial,
        'coords_sensors_spatial': coords_sensors_spatial,
        'coords_ic_spatial': coords_ic_spatial,
        
        # 分量座標（供內部工具使用）
        'x_pde': x_pde, 'y_pde': y_pde, 'z_pde': z_pde,
        'x_bc': x_bc, 'y_bc': y_bc, 'z_bc': z_bc,
        'x_sensors': x_sensors, 'y_sensors': y_sensors, 'z_sensors': z_sensors,
        'x_ic': x_ic, 'y_ic': y_ic, 'z_ic': z_ic,

        # 時間座標（可選）
        't_pde': t_pde,
        't_bc': t_bc,
        't_sensors': t_sensors,
        't_ic': t_ic,
        
        # 感測器數據
        'u_sensors': u_sensors,
        'v_sensors': v_sensors,
        'w_sensors': w_sensors if (is_vs_pinn and w_sensors is not None) else torch.empty(0, 1, device=device),
        'p_sensors': p_sensors,
        
        # 初始條件數據
        'u_ic': u_ic,
        'v_ic': v_ic,
        'w_ic': w_ic if is_vs_pinn else torch.empty(0, 1, device=device),
        'p_ic': p_ic,
        
        # 元數據
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
    
    # 🚀 初始化分散式訓練環境（自動偵測多 GPU）
    ddp_config = init_distributed_mode()
    is_main_process = (ddp_config['rank'] == 0)
    device = ddp_config['device']
    
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
    
    # ========================================
    # ⚡ Fail Fast: 提前驗證 Time Window 配置
    # ========================================
    num_windows = config['training'].get('num_time_windows', 1)
    if num_windows is not None and num_windows > 1:
        kolmogorov_enabled = config.get('data', {}).get('kolmogorov_config', {}).get('enabled', False)
        if not kolmogorov_enabled:
            msg = (f"❌ Time Window 配置錯誤：num_time_windows={num_windows} 但 Kolmogorov Flow 未啟用。"
                   f"請設置 data.kolmogorov_config.enabled = true")
            logger.error(msg)
            raise ValueError("Time Window mode requires Kolmogorov Flow enabled")
        
        time_range = config['data']['kolmogorov_config'].get('time_range')
        if time_range is None:
            msg = "❌ Time Window 配置錯誤：缺少 time_range。請添加 data.kolmogorov_config.time_range: [t_start, t_end]"
            logger.error(msg)
            raise ValueError("Time Window mode requires time_range in kolmogorov_config")
        
        t_start, t_end = time_range
        window_duration = (t_end - t_start) / num_windows
        if window_duration < 0.1:
            logger.warning(f"⚠️  窗口持續時間過短: {window_duration:.4f}s (時間範圍: [{t_start}, {t_end}], 窗口數: {num_windows})")
        
        logger.info(f"✅ Time Window 配置驗證通過")
        logger.info(f"   窗口數: {num_windows}, 時間範圍: [{t_start:.2f}, {t_end:.2f}], 窗口持續時間: {window_duration:.2f}s")
    
    
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
            
            # ✨ P2-3: 使用 TrainerBuilder 創建 Trainer（新路徑）
            logger.info(f"🏗️  使用 TrainerBuilder 創建 Trainer (Ensemble member {i+1})")

            builder = TrainerBuilder(config, device)
            builder.with_model(member_model)
            builder.with_physics(physics)
            builder.with_losses(losses)
            builder.with_training_data(training_data_sample)

            trainer = builder.build()
            logger.info(f"✅ Trainer 創建成功 (Ensemble member {i+1})")
            # ℹ️  所有組件（normalizers, weighters等）已由 TrainerBuilder 自動創建

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
        
        # 🆕 檢查是否啟用時間窗口訓練（改進判斷邏輯）
        training_cfg = config.get('training', {})
        if 'num_time_windows' not in training_cfg or training_cfg.get('num_time_windows') is None:
            training_cfg['num_time_windows'] = 3
            logger.info("ℹ️  num_time_windows 未設定，預設使用 3 個時間窗口")
        num_windows = training_cfg.get('num_time_windows', 3)
        config['training'] = training_cfg
        
        # 驗證 time window 配置完整性
        use_time_window = False
        if num_windows is not None and num_windows > 1:
            # 檢查 Kolmogorov 配置是否啟用（time window 僅支援 Kolmogorov Flow）
            kolmogorov_enabled = config.get('data', {}).get('kolmogorov_config', {}).get('enabled', False)
            
            if not kolmogorov_enabled:
                logger.error(
                    f"❌ Time Window 配置錯誤：\n"
                    f"   num_time_windows={num_windows} (>1) 但 Kolmogorov Flow 未啟用\n"
                    f"   Time Window 模式僅支援 Kolmogorov Flow 資料源\n"
                    f"   請檢查配置：data.kolmogorov_config.enabled 應為 true"
                )
                raise ValueError("Time Window mode requires Kolmogorov Flow enabled")
            
            # 檢查時間範圍是否足夠劃分
            time_range = config['data']['kolmogorov_config'].get('time_range')
            if time_range is None:
                logger.error(
                    f"❌ Time Window 配置錯誤：缺少時間範圍\n"
                    f"   請在 data.kolmogorov_config.time_range 中指定時間範圍"
                )
                raise ValueError("Time Window mode requires time_range in kolmogorov_config")
            
            t_start, t_end = time_range
            window_duration = (t_end - t_start) / num_windows
            
            if window_duration < 0.1:
                logger.warning(
                    f"⚠️  窗口持續時間過短: {window_duration:.4f}s\n"
                    f"   時間範圍: [{t_start}, {t_end}], 窗口數: {num_windows}\n"
                    f"   建議減少窗口數量或增加時間範圍"
                )
            
            use_time_window = True
            logger.info(f"🪟 Time Window Training enabled: {num_windows} windows")
            logger.info(f"   時間範圍: [{t_start:.2f}, {t_end:.2f}]")
            logger.info(f"   窗口持續時間: {window_duration:.2f}s")
        
        if use_time_window:
            from pinnx.train.time_window_trainer import TimeWindowTrainer
            
            # 設置輸出標準化（若配置要求）
            normalizer = setup_output_normalization(config, training_data_sample, logger)
            
            # 創建 Time Window Trainer
            window_trainer = TimeWindowTrainer(
                config=config,
                model=model,
                training_data=training_data_sample,
                device=device,
                physics=physics,
                losses=losses,
                weighters=weighters,
                input_normalizer=input_normalizer,
                data_normalizer=normalizer
            )
            
            # 序列訓練各窗口
            train_result = window_trainer.train_sequential()
        else:
            logger.info("🔄 Standard single-domain training")
            # ✨ P2-3: 使用 TrainerBuilder 創建 Trainer（新路徑）
            logger.info("🏗️  使用 TrainerBuilder 創建 Trainer")

            builder = TrainerBuilder(config, device)
            builder.with_model(model)
            builder.with_physics(physics)
            builder.with_losses(losses)
            builder.with_training_data(training_data_sample)

            trainer = builder.build()
            logger.info("✅ Trainer 創建成功")
            # ℹ️  所有組件（normalizers, weighters等）已由 TrainerBuilder 自動創建

            # 載入 checkpoint（若指定）
            if args.resume:
                logger.info(f"⏮️  載入 checkpoint: {args.resume}")
                trainer.load_checkpoint(args.resume)
            
            train_result = trainer.train()
        
        logger.info(f"Training completed. Final loss: {train_result['final_loss']:.6f}")
    
    logger.info("Training script finished successfully!")


if __name__ == "__main__":
    main()

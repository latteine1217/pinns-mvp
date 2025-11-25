"""
純 PDE 驅動訓練腳本（無 DNS 資料）
===================================

用途：
- 僅使用物理定律（PDE + 邊界條件）訓練 PINNs
- 無需稀疏感測點或 DNS 參考資料
- 適用於 Kolmogorov flow 等理論模型

使用範例：
    python scripts/train_pure_pde.py --cfg configs/kolmogorov_2d_turbulent_pure_pde.yml
"""

import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
import sys
import logging

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.physics.kolmogorov_flow_2d import create_kolmogorov_flow_2d
from pinnx.models.fourier_mlp import PINNNet

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_config(config_path: Path) -> dict:
    """載入 YAML 配置檔案"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_collocation_points(config: dict, device: torch.device) -> torch.Tensor:
    """
    生成配置點（均勻隨機採樣）

    Args:
        config: 配置字典
        device: 計算設備

    Returns:
        coords: [n_collocation, 2] = [x, y]
    """
    n_collocation = config['data']['n_collocation']
    domain = config['physics']['domain']

    x_min, x_max = domain['x_min'], domain['x_max']
    y_min, y_max = domain['y_min'], domain['y_max']

    # 均勻隨機採樣
    x = torch.rand(n_collocation, 1, device=device) * (x_max - x_min) + x_min
    y = torch.rand(n_collocation, 1, device=device) * (y_max - y_min) + y_min

    coords = torch.cat([x, y], dim=1)
    coords.requires_grad_(True)

    logging.info(f"✅ 生成配置點：{n_collocation} 個")

    return coords


def create_boundary_points(config: dict, device: torch.device) -> torch.Tensor:
    """
    生成邊界點（週期性邊界條件）

    策略：
    - x 方向邊界：x=x_min 和 x=x_max（固定 x，隨機 y）
    - y 方向邊界：y=y_min 和 y=y_max（固定 y，隨機 x）

    Args:
        config: 配置字典
        device: 計算設備

    Returns:
        coords: [n_boundary, 2] = [x, y]
    """
    n_boundary = config['data']['n_boundary']
    domain = config['physics']['domain']

    x_min, x_max = domain['x_min'], domain['x_max']
    y_min, y_max = domain['y_min'], domain['y_max']

    # 分配邊界點（四個邊界均分）
    n_per_side = n_boundary // 4

    # x 方向邊界
    y_random_1 = torch.rand(n_per_side, 1, device=device) * (y_max - y_min) + y_min
    x_min_boundary = torch.cat([torch.full_like(y_random_1, x_min), y_random_1], dim=1)

    y_random_2 = torch.rand(n_per_side, 1, device=device) * (y_max - y_min) + y_min
    x_max_boundary = torch.cat([torch.full_like(y_random_2, x_max), y_random_2], dim=1)

    # y 方向邊界
    x_random_1 = torch.rand(n_per_side, 1, device=device) * (x_max - x_min) + x_min
    y_min_boundary = torch.cat([x_random_1, torch.full_like(x_random_1, y_min)], dim=1)

    x_random_2 = torch.rand(n_per_side, 1, device=device) * (x_max - x_min) + x_min
    y_max_boundary = torch.cat([x_random_2, torch.full_like(x_random_2, y_max)], dim=1)

    # 合併所有邊界點
    coords = torch.cat([x_min_boundary, x_max_boundary, y_min_boundary, y_max_boundary], dim=0)
    coords.requires_grad_(True)

    logging.info(f"✅ 生成邊界點：{coords.shape[0]} 個")

    return coords


def create_model(config: dict, device: torch.device) -> torch.nn.Module:
    """
    建立 Fourier MLP 模型

    Args:
        config: 配置字典
        device: 計算設備

    Returns:
        model: PINNNet 實例
    """
    model_config = config['model']

    # 計算網路深度（hidden_layers 列表長度）
    hidden_layers = model_config['hidden_layers']
    depth = len(hidden_layers)
    width = hidden_layers[0] if hidden_layers else 256

    model = PINNNet(
        in_dim=model_config['input_dim'],
        out_dim=model_config['output_dim'],
        width=width,
        depth=depth,
        activation=model_config['activation'],
        use_fourier=model_config['fourier_features']['enabled'],
        fourier_m=model_config['fourier_features']['n_features'],
        fourier_sigma=model_config['fourier_features']['sigma'],
    ).to(device)

    logging.info(f"✅ 模型建立完成：{sum(p.numel() for p in model.parameters())} 參數")

    return model


def create_physics(config: dict) -> torch.nn.Module:
    """
    建立 Kolmogorov flow 物理模型

    Args:
        config: 配置字典

    Returns:
        physics: KolmogorovFlow2D 實例
    """
    physics_config = config['physics']

    # 讀取域範圍配置
    domain_config = physics_config.get('domain', {})
    domain_bounds = {
        'x': (domain_config.get('x_min', 0.0), domain_config.get('x_max', 2.0 * np.pi)),
        'y': (domain_config.get('y_min', 0.0), domain_config.get('y_max', 2.0 * np.pi)),
    }

    physics = create_kolmogorov_flow_2d(
        A=physics_config['forcing']['amplitude'],
        k_f=physics_config['forcing']['wavenumber'],
        nu=physics_config['nu'],
        rho=physics_config['rho'],
        domain_bounds=domain_bounds,
        loss_config=config.get('loss', {}),
    )

    logging.info(f"✅ 物理模型建立完成")
    info = physics.get_physics_info()
    logging.info(f"   雷諾數 Re = {info['physics_parameters']['Reynolds_number']:.1f}")

    return physics


def compute_losses(
    model: torch.nn.Module,
    physics: torch.nn.Module,
    coords_pde: torch.Tensor,
    coords_bc: torch.Tensor,
    config: dict
) -> dict:
    """
    計算所有損失項

    Args:
        model: 神經網路模型
        physics: 物理模型
        coords_pde: PDE 配置點 [n_pde, 2]
        coords_bc: 邊界點 [n_bc, 2]
        config: 配置字典

    Returns:
        loss_dict: 損失字典
    """
    loss_dict = {}
    weights = config['loss']['weights']

    # === 計算 PDE 殘差 ===
    predictions_pde = model(coords_pde)

    # 動量方程殘差
    momentum_residuals = physics.compute_momentum_residuals(coords_pde, predictions_pde)
    loss_dict['momentum_x'] = torch.mean(momentum_residuals['momentum_x'] ** 2)
    loss_dict['momentum_y'] = torch.mean(momentum_residuals['momentum_y'] ** 2)

    # 連續方程殘差
    continuity_residual = physics.compute_continuity_residual(coords_pde, predictions_pde)
    loss_dict['continuity'] = torch.mean(continuity_residual ** 2)

    # === 計算邊界條件損失 ===
    predictions_bc = model(coords_bc)
    periodic_losses = physics.compute_periodic_loss(coords_bc, predictions_bc)
    loss_dict['periodic_x'] = periodic_losses['periodic_x']
    loss_dict['periodic_y'] = periodic_losses['periodic_y']

    # === 應用權重並計算總損失 ===
    total_loss = 0.0
    for key, loss_val in loss_dict.items():
        weight = weights.get(key, 0.0)
        total_loss += weight * loss_val

    loss_dict['total'] = total_loss

    return loss_dict


def train_epoch(
    model: torch.nn.Module,
    physics: torch.nn.Module,
    coords_pde: torch.Tensor,
    coords_bc: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: dict,
    epoch: int
) -> dict:
    """
    訓練一個 epoch

    Args:
        model: 神經網路模型
        physics: 物理模型
        coords_pde: PDE 配置點
        coords_bc: 邊界點
        optimizer: 優化器
        config: 配置字典
        epoch: 當前 epoch

    Returns:
        metrics: 訓練指標字典
    """
    model.train()

    # 計算損失
    loss_dict = compute_losses(model, physics, coords_pde, coords_bc, config)

    # 損失歸一化（可選）
    if config['loss'].get('normalize_losses', False):
        loss_dict_normalized = physics.normalize_loss_dict(
            {k: v for k, v in loss_dict.items() if k != 'total'},
            epoch=epoch
        )
        # 重新計算總損失
        total_loss = sum(
            config['loss']['weights'].get(k, 0.0) * v
            for k, v in loss_dict_normalized.items()
        )
        loss_dict['total'] = total_loss

    # 反向傳播
    optimizer.zero_grad()
    loss_dict['total'].backward()
    optimizer.step()

    # 轉換為標量
    metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    return metrics


def main():
    parser = argparse.ArgumentParser(description='純 PDE 驅動訓練（無 DNS 資料）')
    parser.add_argument('--cfg', type=str, required=True, help='配置檔案路徑')
    args = parser.parse_args()

    # 載入配置
    config_path = Path(args.cfg)
    config = load_config(config_path)

    logging.info(f"📋 載入配置：{config_path}")
    logging.info(f"   實驗名稱：{config['experiment']['name']}")

    # 設定隨機種子
    seed = config['compute']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 設定設備
    device_config = config['compute']['device']
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    logging.info(f"🖥️  計算設備：{device}")

    # 建立輸出目錄
    output_dirs = {
        'checkpoint': Path(config['output']['checkpoint_dir']),
        'result': Path(config['output']['result_dir']),
        'log': Path(config['output']['log_dir']),
    }
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # === 建立模型與物理 ===
    model = create_model(config, device)
    physics = create_physics(config)

    # === 生成訓練資料 ===
    coords_pde = create_collocation_points(config, device)
    coords_bc = create_boundary_points(config, device)

    # === 建立優化器 ===
    optimizer_config = config['training']['optimizer']['phase1']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config['lr']
    )

    logging.info(f"✅ 優化器建立完成：{optimizer_config['type']}, lr={optimizer_config['lr']}")

    # === 訓練循環 ===
    n_epochs = config['training']['n_epochs']
    print_interval = config['output']['logging']['print_interval']
    save_interval = config['training']['checkpoint']['save_interval']

    logging.info(f"▶️  開始訓練：{n_epochs} iterations")
    logging.info("=" * 80)

    for epoch in range(1, n_epochs + 1):
        metrics = train_epoch(model, physics, coords_pde, coords_bc, optimizer, config, epoch)

        # 打印日誌
        if epoch % print_interval == 0 or epoch == 1:
            log_str = f"Epoch {epoch:4d}/{n_epochs} | "
            log_str += f"Total: {metrics['total']:.4e} | "
            log_str += f"Mom_x: {metrics['momentum_x']:.4e} | "
            log_str += f"Mom_y: {metrics['momentum_y']:.4e} | "
            log_str += f"Cont: {metrics['continuity']:.4e} | "
            log_str += f"BC_x: {metrics['periodic_x']:.4e} | "
            log_str += f"BC_y: {metrics['periodic_y']:.4e}"
            logging.info(log_str)

        # 保存檢查點
        if epoch % save_interval == 0:
            checkpoint_path = output_dirs['checkpoint'] / f"epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': config,
            }, checkpoint_path)
            logging.info(f"💾 檢查點已保存：{checkpoint_path}")

    logging.info("=" * 80)
    logging.info(f"✅ 訓練完成！")

    # 保存最終模型
    final_path = output_dirs['checkpoint'] / "final_model.pth"
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
    }, final_path)
    logging.info(f"💾 最終模型已保存：{final_path}")

    # === 評估與視覺化 ===
    logging.info("📊 開始評估...")

    model.eval()

    # 生成驗證網格
    N_eval = 128
    x_eval = torch.linspace(0, 2 * np.pi, N_eval, device=device)
    y_eval = torch.linspace(0, 2 * np.pi, N_eval, device=device)
    X_eval, Y_eval = torch.meshgrid(x_eval, y_eval, indexing='ij')
    coords_eval = torch.stack([X_eval.flatten(), Y_eval.flatten()], dim=1)
    coords_eval.requires_grad_(True)

    # 預測（需要梯度計算）
    predictions_eval = model(coords_eval)

    # 計算渦度和 Enstrophy（需要梯度）
    vorticity = physics.compute_vorticity(coords_eval, predictions_eval)
    enstrophy = physics.compute_enstrophy(coords_eval, predictions_eval).item()

    # 計算動能（不需要梯度）
    with torch.no_grad():
        kinetic_energy = physics.compute_kinetic_energy(predictions_eval).item()

    # 轉換為 numpy（用於保存）
    u_eval = predictions_eval[:, 0].reshape(N_eval, N_eval).detach().cpu().numpy()
    v_eval = predictions_eval[:, 1].reshape(N_eval, N_eval).detach().cpu().numpy()
    p_eval = predictions_eval[:, 2].reshape(N_eval, N_eval).detach().cpu().numpy()
    vorticity_eval = vorticity.reshape(N_eval, N_eval).detach().cpu().numpy()

    logging.info(f"   動能 KE = {kinetic_energy:.6f}")
    logging.info(f"   Enstrophy = {enstrophy:.6f}")

    # 保存結果
    results = {
        'x': x_eval.cpu().numpy(),
        'y': y_eval.cpu().numpy(),
        'u': u_eval,
        'v': v_eval,
        'p': p_eval,
        'vorticity': vorticity_eval,
        'kinetic_energy': kinetic_energy,
        'enstrophy': enstrophy,
    }

    result_path = output_dirs['result'] / "final_results.npz"
    np.savez(result_path, **results)
    logging.info(f"💾 結果已保存：{result_path}")

    logging.info("🎉 完成！")


if __name__ == '__main__':
    main()

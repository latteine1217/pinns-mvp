"""
評估並視覺化 Kolmogorov Flow 2D 快速測試結果
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.models.fourier_mlp import PINNNet
from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D

def compute_physics_diagnostics(model, physics, coords_tensor, device):
    """計算物理診斷指標"""
    model.eval()

    # 計算物理殘差需要啟用梯度
    coords_grad = coords_tensor.clone().requires_grad_(True)
    pred_grad = model(coords_grad)

    # 計算殘差（返回字典）
    residuals_dict = physics.residual_unified(coords_grad, pred_grad)

    # 提取各項殘差
    momentum_x = residuals_dict['momentum_x']
    momentum_y = residuals_dict['momentum_y']
    continuity = residuals_dict['continuity']

    diagnostics = {
        'momentum_x_mean': momentum_x.abs().mean().item(),
        'momentum_x_max': momentum_x.abs().max().item(),
        'momentum_y_mean': momentum_y.abs().mean().item(),
        'momentum_y_max': momentum_y.abs().max().item(),
        'continuity_mean': continuity.abs().mean().item(),
        'continuity_max': continuity.abs().max().item(),
    }

    # 用於可視化的預測（不需要梯度）
    with torch.no_grad():
        predictions = model(coords_tensor)

    return diagnostics, predictions

def main():
    parser = argparse.ArgumentParser(description='評估 Kolmogorov Flow 2D 訓練結果')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/kolmogorov_2d_quick_test/epoch_1000.pth',
                       help='檢查點路徑')
    parser.add_argument('--config', type=str,
                       default='configs/kolmogorov_2d_quick_test.yml',
                       help='配置文件路徑')
    parser.add_argument('--output', type=str,
                       default='results/kolmogorov_2d_quick_test',
                       help='輸出目錄')
    parser.add_argument('--n-points', type=int, default=128,
                       help='評估網格點數')
    args = parser.parse_args()

    print('=' * 70)
    print('📊 Kolmogorov Flow 2D 評估與視覺化')
    print('=' * 70)
    print()

    # 創建輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入檢查點
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    print(f'✅ 載入檢查點: {args.checkpoint}')
    print(f'   Epoch: {checkpoint["epoch"]}')
    if 'loss_history' in checkpoint and checkpoint['loss_history']:
        history = checkpoint['loss_history']
        if history:
            last_loss = history[-1]
            print(f'   最終損失: {last_loss.get("total_loss", "N/A")}')
    print()

    # 優先使用檢查點中保存的配置，否則載入配置文件
    if 'config' in checkpoint:
        config = checkpoint['config']
        print('✅ 使用檢查點中保存的配置')
    else:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f'✅ 載入配置文件: {args.config}')
    print()

    # 重建模型（使用 factory 以確保配置一致）
    from pinnx.train.factory import create_model
    try:
        model = create_model(config, device)
        print(f'✅ 使用 factory 創建模型（類型: {type(model).__name__}）')
    except Exception as e:
        print(f'⚠️  Factory 創建失敗: {e}')
        print('   回退到手動創建...')
        # 回退方案：手動創建
        model_cfg = config['model']
        fourier_cfg = model_cfg.get('fourier_features', {})
        
        # 從配置獲取正確的 width
        width = model_cfg.get('width', 512)
        depth = model_cfg.get('depth', 6)
        
        # 優先使用 fourier_m（實際訓練時的參數），否則使用 n_features
        fourier_m = model_cfg.get('fourier_m', fourier_cfg.get('n_features', 64))
        fourier_sigma = model_cfg.get('fourier_sigma', fourier_cfg.get('sigma', 8.0))

        model = PINNNet(
            in_dim=model_cfg.get('input_dim', 2),
            out_dim=model_cfg.get('output_dim', 3),
            width=width,
            depth=depth,
            fourier_m=fourier_m,
            fourier_sigma=fourier_sigma,
            activation=model_cfg.get('activation', 'sine'),
            use_fourier=model_cfg.get('use_fourier', True)
        ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    model_cfg = config['model']
    width = model_cfg.get('width', 512)
    depth = model_cfg.get('depth', 6)
    print(f'✅ 模型載入完成')
    print(f'   參數量: {n_params:,}')
    print(f'   架構: {depth}層 × {width}神經元')
    print()

    # 初始化物理模組
    physics_cfg = config['physics']
    
    # 支持兩種配置格式
    k_f = physics_cfg['forcing'].get('k_f', physics_cfg['forcing'].get('wavenumber', 8))
    domain = physics_cfg['domain']
    
    # 支持兩種 domain 格式：x_min/x_max 或 x_range
    if 'x_min' in domain:
        x_bounds = (domain['x_min'], domain['x_max'])
        y_bounds = (domain['y_min'], domain['y_max'])
    else:
        x_bounds = tuple(domain['x_range'])
        y_bounds = tuple(domain['y_range'])
    
    physics = KolmogorovFlow2D(
        forcing_params={
            'amplitude': physics_cfg['forcing']['amplitude'],
            'wavenumber': k_f
        },
        physics_params={
            'nu': physics_cfg['nu'],
            'rho': physics_cfg.get('rho', 1.0)
        },
        domain_bounds={
            'x': x_bounds,
            'y': y_bounds
        }
    ).to(device)
    print(f'✅ 物理模組初始化完成')
    print()

    # 生成評估網格
    domain = physics_cfg['domain']
    n_points = args.n_points

    # 支持兩種 domain 格式
    if 'x_min' in domain:
        x = np.linspace(domain['x_min'], domain['x_max'], n_points)
        y = np.linspace(domain['y_min'], domain['y_max'], n_points)
    else:
        x = np.linspace(domain['x_range'][0], domain['x_range'][1], n_points)
        y = np.linspace(domain['y_range'][0], domain['y_range'][1], n_points)
    
    X, Y = np.meshgrid(x, y)

    coords = np.stack([X.flatten(), Y.flatten()], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)

    print(f'✅ 生成評估網格: {n_points}×{n_points} = {len(coords):,} 點')

    # 批次預測
    batch_size = 5000
    predictions_list = []

    print('🔄 開始批次預測...')
    with torch.no_grad():
        for i in range(0, len(coords_tensor), batch_size):
            batch = coords_tensor[i:i+batch_size]
            pred = model(batch)
            predictions_list.append(pred.cpu().numpy())

    predictions = np.vstack(predictions_list)
    print('✅ 預測完成')
    print()

    # 提取場變量
    u = predictions[:, 0].reshape(n_points, n_points)
    v = predictions[:, 1].reshape(n_points, n_points)
    p = predictions[:, 2].reshape(n_points, n_points)

    # 計算物理診斷
    print('🔄 計算物理診斷...')
    diagnostics, _ = compute_physics_diagnostics(model, physics, coords_tensor, device)
    print('✅ 物理診斷完成')
    print()

    # 計算導出量
    # 計算散度 ∇·u = ∂u/∂x + ∂v/∂y
    du_dx = np.gradient(u, x, axis=1)
    dv_dy = np.gradient(v, y, axis=0)
    divergence = du_dx + dv_dy

    # 計算渦度 ω = ∂v/∂x - ∂u/∂y
    dv_dx = np.gradient(v, x, axis=1)
    du_dy = np.gradient(u, y, axis=0)
    vorticity = dv_dx - du_dy

    # 統計量
    print('=' * 70)
    print('📈 物理量統計')
    print('=' * 70)
    print()

    print(f'速度場 u: mean={u.mean():.6e}, std={u.std():.6e}')
    print(f'          range=[{u.min():.6e}, {u.max():.6e}]')
    print()

    print(f'速度場 v: mean={v.mean():.6e}, std={v.std():.6e}')
    print(f'          range=[{v.min():.6e}, {v.max():.6e}]')
    print()

    print(f'壓力場 p: mean={p.mean():.6e}, std={p.std():.6e}')
    print(f'          range=[{p.min():.6e}, {p.max():.6e}]')
    print()

    print(f'散度 (∇·u): mean={divergence.mean():.6e}')
    print(f'            std={divergence.std():.6e}')
    print(f'            max_abs={np.abs(divergence).max():.6e}')
    print()

    print(f'渦度 (ω):   mean={vorticity.mean():.6e}')
    print(f'            std={vorticity.std():.6e}')
    print(f'            range=[{vorticity.min():.6e}, {vorticity.max():.6e}]')
    print()

    print('=' * 70)
    print('🔬 物理殘差診斷')
    print('=' * 70)
    print()

    print(f'動量 X 殘差: mean={diagnostics["momentum_x_mean"]:.6e}, max={diagnostics["momentum_x_max"]:.6e}')
    print(f'動量 Y 殘差: mean={diagnostics["momentum_y_mean"]:.6e}, max={diagnostics["momentum_y_max"]:.6e}')
    print(f'連續性殘差:  mean={diagnostics["continuity_mean"]:.6e}, max={diagnostics["continuity_max"]:.6e}')
    print()

    # 保存結果
    results = {
        'X': X, 'Y': Y,
        'u': u, 'v': v, 'p': p,
        'divergence': divergence,
        'vorticity': vorticity,
        'diagnostics': diagnostics
    }

    np.savez(output_dir / 'evaluation_results.npz', **results)
    print(f'✅ 結果已保存: {output_dir}/evaluation_results.npz')
    print()

    # 視覺化
    print('🎨 生成視覺化圖表...')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # u 速度場
    im0 = axes[0, 0].contourf(X, Y, u, levels=20, cmap='RdBu_r')
    axes[0, 0].set_title('u velocity', fontsize=14)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])

    # v 速度場
    im1 = axes[0, 1].contourf(X, Y, v, levels=20, cmap='RdBu_r')
    axes[0, 1].set_title('v velocity', fontsize=14)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])

    # 壓力場
    im2 = axes[0, 2].contourf(X, Y, p, levels=20, cmap='viridis')
    axes[0, 2].set_title('Pressure', fontsize=14)
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 2])

    # 速度向量場
    skip = 4
    axes[1, 0].quiver(X[::skip, ::skip], Y[::skip, ::skip],
                      u[::skip, ::skip], v[::skip, ::skip])
    axes[1, 0].set_title('Velocity Field', fontsize=14)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_aspect('equal')

    # 散度
    im3 = axes[1, 1].contourf(X, Y, divergence, levels=20, cmap='PuOr')
    axes[1, 1].set_title(f'Divergence (max={np.abs(divergence).max():.2e})', fontsize=14)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 1])

    # 渦度
    im4 = axes[1, 2].contourf(X, Y, vorticity, levels=20, cmap='RdYlBu_r')
    axes[1, 2].set_title('Vorticity', fontsize=14)
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig(output_dir / 'fields_visualization.png', dpi=150, bbox_inches='tight')
    print(f'✅ 場圖已保存: {output_dir}/fields_visualization.png')
    plt.close()

    # 繪製損失歷史
    if 'loss_history' in checkpoint and checkpoint['loss_history']:
        history = checkpoint['loss_history']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        epochs = [h['epoch'] for h in history]
        pde_loss = [h.get('pde_loss', 0) for h in history]
        momentum_x = [h.get('momentum_x_loss', 0) for h in history]
        momentum_y = [h.get('momentum_y_loss', 0) for h in history]
        continuity = [h.get('continuity_loss', 0) for h in history]
        periodic_x = [h.get('periodic_x_loss', 0) for h in history]
        periodic_y = [h.get('periodic_y_loss', 0) for h in history]

        # PDE Loss
        axes[0, 0].semilogy(epochs, pde_loss, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('PDE Loss')
        axes[0, 0].set_title('Total PDE Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # Momentum Losses
        axes[0, 1].semilogy(epochs, momentum_x, 'r-', label='Momentum X', linewidth=2)
        axes[0, 1].semilogy(epochs, momentum_y, 'g-', label='Momentum Y', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Momentum Losses')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Continuity Loss
        axes[1, 0].semilogy(epochs, continuity, 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Continuity Loss')
        axes[1, 0].grid(True, alpha=0.3)

        # Periodic Losses
        axes[1, 1].semilogy(epochs, periodic_x, 'c-', label='Periodic X', linewidth=2)
        axes[1, 1].semilogy(epochs, periodic_y, 'y-', label='Periodic Y', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Periodic BC Losses')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'loss_history.png', dpi=150, bbox_inches='tight')
        print(f'✅ 損失歷史已保存: {output_dir}/loss_history.png')
        plt.close()

    print()
    print('=' * 70)
    print('✅ 評估完成！')
    print('=' * 70)

if __name__ == '__main__':
    main()

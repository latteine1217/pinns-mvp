"""
評估並視覺化 Kolmogorov Flow 完整訓練結果
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.models.fourier_mlp import PINNNet

def main():
    print('=' * 70)
    print('📊 Kolmogorov Flow Re=30 完整訓練評估與視覺化')
    print('=' * 70)
    print()

    # 載入配置
    config_path = 'configs/kolmogorov_2d_chaos_re30_full.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 載入檢查點
    checkpoint_path = 'checkpoints/kolmogorov_2d_chaos_re30_full/epoch_1800.pth'
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f'✅ 載入檢查點: {checkpoint_path}')
    print(f'   Epoch: {checkpoint["epoch"]}')
    if 'metrics' in checkpoint and checkpoint['metrics']:
        metrics = checkpoint['metrics']
        if 'total_loss' in metrics:
            print(f'   損失: {metrics["total_loss"]:.6e}')
        elif isinstance(metrics, dict):
            print(f'   Metrics: {list(metrics.keys())}')
    print()

    # 重建模型
    model_cfg = config['model']
    fourier_cfg = model_cfg['fourier_features']
    hidden_layers = model_cfg['hidden_layers']

    # Fourier特徵: n_features 就是 fourier_m (不是 //2)
    # 因為 FourierFeatures 會生成 2*fourier_m 的維度 (sin + cos)
    model = PINNNet(
        in_dim=model_cfg['input_dim'],
        out_dim=model_cfg['output_dim'],
        width=hidden_layers[0],
        depth=len(hidden_layers),
        fourier_m=fourier_cfg['n_features'],  # 修正：直接使用 n_features
        fourier_sigma=fourier_cfg['sigma'],
        activation=model_cfg['activation'],
        use_fourier=fourier_cfg['enabled']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'✅ 模型載入完成')
    print(f'   參數量: {n_params:,}')
    print(f'   架構: {len(hidden_layers)}層 × {hidden_layers[0]}神經元')
    print()

    # 生成評估網格
    domain = config['physics']['domain']
    n_points = 128

    x = np.linspace(domain['x_min'], domain['x_max'], n_points)
    y = np.linspace(domain['y_min'], domain['y_max'], n_points)
    X, Y = np.meshgrid(x, y)

    coords = np.stack([X.flatten(), Y.flatten()], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)

    print(f'✅ 生成評估網格: {n_points}×{n_points} = {len(coords):,} 點')

    # 批次預測
    batch_size = 5000
    predictions = []

    print(f'🔄 開始批次預測 (batch_size={batch_size})...')
    with torch.no_grad():
        for i in range(0, len(coords_tensor), batch_size):
            batch = coords_tensor[i:i+batch_size]
            pred = model(batch)
            predictions.append(pred.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    u = predictions[:, 0].reshape(n_points, n_points)
    v = predictions[:, 1].reshape(n_points, n_points)
    p = predictions[:, 2].reshape(n_points, n_points)

    print(f'✅ 預測完成')
    print()

    # 計算導數與物理量
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    vorticity = dv_dx - du_dy

    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    divergence = du_dx + dv_dy

    kinetic_energy = 0.5 * (u**2 + v**2).mean()
    enstrophy = (vorticity**2).mean()

    # 打印統計
    print('=' * 70)
    print('📈 物理量統計')
    print('=' * 70)
    print()
    print(f'動能 (KE):         {kinetic_energy:.6e}')
    print(f'擬能 (Enstrophy): {enstrophy:.6e}')
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

    # 保存結果
    output_dir = Path('results/kolmogorov_2d_chaos_re30_full')
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / 'final_results.npz',
        x=x, y=y,
        u=u, v=v, p=p,
        vorticity=vorticity,
        divergence=divergence,
        kinetic_energy=kinetic_energy,
        enstrophy=enstrophy,
        epoch=checkpoint['epoch']
    )

    print(f'✅ 結果已保存: {output_dir / "final_results.npz"}')
    print()

    # 生成視覺化
    print('🎨 生成視覺化圖表...')

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # u 速度場
    im0 = axes[0, 0].contourf(X, Y, u, levels=20, cmap='RdBu_r')
    axes[0, 0].set_title('x-velocity (u)', fontsize=14)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    axes[0, 0].text(0.02, 0.98, f'min={u.min():.4f}, max={u.max():.4f}',
                    transform=axes[0, 0].transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # v 速度場
    im1 = axes[0, 1].contourf(X, Y, v, levels=20, cmap='RdBu_r')
    axes[0, 1].set_title('y-velocity (v)', fontsize=14)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])
    axes[0, 1].text(0.02, 0.98, f'min={v.min():.4f}, max={v.max():.4f}',
                    transform=axes[0, 1].transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 壓力場
    im2 = axes[1, 0].contourf(X, Y, p, levels=20, cmap='viridis')
    axes[1, 0].set_title('Pressure (p)', fontsize=14)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1, 0])
    axes[1, 0].text(0.02, 0.98, f'min={p.min():.4f}, max={p.max():.4f}',
                    transform=axes[1, 0].transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 渦度場
    im3 = axes[1, 1].contourf(X, Y, vorticity, levels=20, cmap='seismic')
    axes[1, 1].set_title('Vorticity (ω)', fontsize=14)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 1])
    axes[1, 1].text(0.02, 0.98, f'min={vorticity.min():.4f}, max={vorticity.max():.4f}',
                    transform=axes[1, 1].transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(f'Kolmogorov Flow Re=30 - Epoch {checkpoint["epoch"]}', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'fields_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✅ 場圖已保存: {output_dir / "fields_comparison.png"}')

    # 流線圖
    fig, ax = plt.subplots(figsize=(12, 6))

    speed = np.sqrt(u**2 + v**2)
    strm = ax.streamplot(X, Y, u, v, color=speed, linewidth=1.5, cmap='viridis', density=1.5, arrowsize=1.2)
    cbar = plt.colorbar(strm.lines, ax=ax)
    cbar.set_label('Velocity Magnitude |V|', fontsize=12)

    ax.set_title(f'Streamlines (colored by velocity magnitude)\\nMax |V| = {speed.max():.4f}', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / 'streamlines.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✅ 流線圖已保存: {output_dir / "streamlines.png"}')

    print()
    print('=' * 70)
    print('✅ 評估與視覺化完成！')
    print('=' * 70)
    print()
    print(f'輸出目錄: {output_dir}')
    print(f'  - final_results.npz      (數值結果)')
    print(f'  - fields_comparison.png  (速度、壓力、渦度場)')
    print(f'  - streamlines.png        (流線圖)')
    print()

if __name__ == '__main__':
    main()

"""
Kolmogorov Flow 2D 課程學習訓練腳本
階段式訓練：Re=50 → Re=100 → Re=240
"""
import sys
from pathlib import Path
import yaml
import torch
import logging

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.models.fourier_mlp import PINNNet
from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D
from pinnx.train.trainer import Trainer
from pinnx.utils.normalization import OutputTransform, OutputNormConfig

def setup_logging():
    """設置日誌"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_model(config, device):
    """創建模型"""
    model_cfg = config['model']
    fourier_m = model_cfg.get('fourier_m', 64)
    fourier_sigma = model_cfg.get('fourier_sigma', 10.0)
    hidden_layers = model_cfg.get('hidden_layers', [384] * 6)

    model = PINNNet(
        in_dim=model_cfg.get('input_dim', 2),
        out_dim=model_cfg.get('output_dim', 3),
        width=hidden_layers[0],
        depth=len(hidden_layers),
        fourier_m=fourier_m,
        fourier_sigma=fourier_sigma,
        activation=model_cfg.get('activation', 'sine'),
        use_fourier=model_cfg.get('use_fourier', True)
    ).to(device)

    return model

def create_physics(physics_cfg, device):
    """創建物理模組"""
    physics = KolmogorovFlow2D(
        forcing_params={
            'amplitude': physics_cfg['forcing']['amplitude'],
            'wavenumber': physics_cfg['forcing']['wavenumber']
        },
        physics_params={
            'nu': physics_cfg['nu'],
            'rho': physics_cfg.get('rho', 1.0)
        },
        domain_bounds={
            'x': (physics_cfg['domain']['x_min'], physics_cfg['domain']['x_max']),
            'y': (physics_cfg['domain']['y_min'], physics_cfg['domain']['y_max'])
        }
    ).to(device)

    return physics

def generate_training_data(config, physics, device):
    """生成訓練資料"""
    import numpy as np

    n_pde = config['data']['n_collocation']
    n_bc = config['data']['n_boundary']

    domain = config['physics']['domain']
    x_range = (domain['x_min'], domain['x_max'])
    y_range = (domain['y_min'], domain['y_max'])

    # PDE 點（域內均勻採樣）
    coords_pde = np.random.uniform(
        low=[x_range[0], y_range[0]],
        high=[x_range[1], y_range[1]],
        size=(n_pde, 2)
    )

    # 週期性邊界點（x 和 y 方向）
    band_width = config['physics']['boundary_conditions']['boundary_band_width']

    # x 方向週期邊界
    n_bc_x = n_bc // 2
    y_bc_x = np.random.uniform(y_range[0], y_range[1], n_bc_x)
    x_left = np.full(n_bc_x, x_range[0])
    x_right = np.full(n_bc_x, x_range[1])

    # y 方向週期邊界
    n_bc_y = n_bc - n_bc_x
    x_bc_y = np.random.uniform(x_range[0], x_range[1], n_bc_y)
    y_bottom = np.full(n_bc_y, y_range[0])
    y_top = np.full(n_bc_y, y_range[1])

    coords_bc_x = np.concatenate([
        np.stack([x_left, y_bc_x], axis=1),
        np.stack([x_right, y_bc_x], axis=1)
    ], axis=0)

    coords_bc_y = np.concatenate([
        np.stack([x_bc_y, y_bottom], axis=1),
        np.stack([x_bc_y, y_top], axis=1)
    ], axis=0)

    coords_bc = np.concatenate([coords_bc_x, coords_bc_y], axis=0)

    # 轉換為張量
    coords_pde_tensor = torch.tensor(coords_pde, dtype=torch.float32, device=device)
    coords_bc_tensor = torch.tensor(coords_bc, dtype=torch.float32, device=device)

    return {
        'coords_pde': coords_pde_tensor,
        'coords_bc': coords_bc_tensor,
        'u_data': None,  # 無監督訓練
        'coords_data': None
    }

def train_phase(phase_name, phase_config, base_config, model, device, logger,
                prev_checkpoint=None):
    """訓練單個階段"""

    logger.info('=' * 70)
    logger.info(f'🚀 開始訓練階段: {phase_name}')
    logger.info('=' * 70)

    # 合併配置
    config = base_config.copy()

    # 更新物理參數
    if 'physics' in phase_config:
        config['physics'].update(phase_config['physics'])

    # 更新損失權重
    if 'loss' in phase_config:
        config['loss']['weights'].update(phase_config['loss']['weights'])

    # 計算雷諾數
    A = config['physics']['forcing']['amplitude']
    k_f = config['physics']['forcing']['wavenumber']
    nu = config['physics']['nu']
    Re = A / (nu * k_f) / nu

    logger.info(f'物理參數:')
    logger.info(f'  強迫振幅 A = {A}')
    logger.info(f'  波數 k_f = {k_f}')
    logger.info(f'  黏滯係數 ν = {nu}')
    logger.info(f'  雷諾數 Re ≈ {Re:.1f}')
    logger.info('')

    # 創建物理模組（使用新的強迫參數）
    physics = create_physics(config['physics'], device)

    # 生成訓練資料
    logger.info('生成訓練資料...')
    training_data = generate_training_data(config, physics, device)
    logger.info(f'  PDE 點: {len(training_data["coords_pde"]):,}')
    logger.info(f'  BC 點: {len(training_data["coords_bc"]):,}')
    logger.info('')

    # 創建訓練器
    norm_config = OutputNormConfig(norm_type='none')
    data_normalizer = OutputTransform(norm_config)

    optimizer_config = phase_config.get('optimizer', {
        'type': 'adam',
        'lr': 0.001
    })

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config['lr'],
        weight_decay=optimizer_config.get('weight_decay', 0.0)
    )

    trainer = Trainer(
        model=model,
        physics=physics,
        losses=config['loss'],
        config=config,
        device=device,
        data_normalizer=data_normalizer,
        optimizer=optimizer
    )

    trainer.training_data = training_data

    # 載入前一階段的檢查點（如果有）
    start_epoch = 0
    if prev_checkpoint:
        logger.info(f'載入前一階段檢查點: {prev_checkpoint}')
        checkpoint = torch.load(prev_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info('✅ 檢查點載入完成')
        logger.info('')

    # 訓練
    n_epochs = phase_config['epochs']
    logger.info(f'開始訓練 {n_epochs} epochs...')
    logger.info('')

    # 修改配置中的 epochs
    config['training']['n_epochs'] = n_epochs

    # 執行訓練
    trainer.train()

    # 保存檢查點
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f'{phase_name}_final.pth'

    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'phase': phase_name,
        'reynolds_number': Re
    }, checkpoint_path)

    logger.info('')
    logger.info(f'✅ 階段 {phase_name} 訓練完成')
    logger.info(f'💾 檢查點已保存: {checkpoint_path}')
    logger.info('=' * 70)
    logger.info('')

    return checkpoint_path

def main():
    logger = setup_logging()

    logger.info('=' * 70)
    logger.info('📚 Kolmogorov Flow 2D 課程學習訓練')
    logger.info('=' * 70)
    logger.info('')

    # 載入配置
    config_path = 'configs/kolmogorov_2d_curriculum.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f'✅ 載入配置: {config_path}')
    logger.info('')

    # 設置設備
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'🖥️  使用設備: {device}')
    logger.info('')

    # 創建模型（只創建一次，在所有階段間共享）
    logger.info('創建模型...')
    model = create_model(config, device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'✅ 模型創建完成')
    logger.info(f'   參數量: {n_params:,}')
    logger.info(f'   架構: {len(config["model"]["hidden_layers"])}層 × {config["model"]["width"]}神經元')
    logger.info('')

    # 課程階段
    phases = [
        ('phase1', config['curriculum']['phase1']),
        ('phase2', config['curriculum']['phase2']),
        ('phase3', config['curriculum']['phase3'])
    ]

    logger.info(f'📋 課程計畫:')
    for i, (phase_name, phase_cfg) in enumerate(phases, 1):
        A = phase_cfg['physics']['forcing']['amplitude']
        nu = config['physics']['nu']
        k_f = phase_cfg['physics']['forcing']['wavenumber']
        Re = A / (nu * k_f) / nu
        logger.info(f'  階段 {i} ({phase_name}): Re≈{Re:.0f}, {phase_cfg["epochs"]} epochs')
    logger.info('')

    # 逐階段訓練
    prev_checkpoint = None

    for phase_name, phase_cfg in phases:
        prev_checkpoint = train_phase(
            phase_name, phase_cfg, config, model, device, logger, prev_checkpoint
        )

    logger.info('=' * 70)
    logger.info('🎉 課程學習訓練完成！')
    logger.info('=' * 70)
    logger.info('')
    logger.info('輸出文件:')
    logger.info(f'  檢查點: {config["output"]["checkpoint_dir"]}/')
    logger.info(f'  結果: {config["output"]["result_dir"]}/')
    logger.info('')

if __name__ == '__main__':
    main()

"""
Time Window Training Module

實現多時間窗口序列訓練，解決長時間範圍（> 2 T_eddy）的誤差累積問題。

設計基於 JAXpi 參考實現：
- 無重疊窗口劃分
- IC 直接轉移（從前窗口預測）
- Transfer Learning（恢復前窗口參數）
- 窗口內獨立使用 Causal Training

參考：
- JAXpi: ~/Documents/coding/jaxpi/examples/kolmogorov_flow/train.py
- Physics Review: context/tasks/stage2_timewindow/physics_review.md
"""

import torch
import logging
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TimeWindowTrainer:
    """
    多時間窗口序列訓練器
    
    核心功能：
    1. 將時間範圍劃分為多個無重疊窗口
    2. 序列訓練各窗口（Window N+1 使用 Window N 的預測作為 IC）
    3. 每個窗口內部使用 Causal Training
    4. Transfer Learning（恢復前窗口模型參數）
    
    Example:
        ```python
        trainer = TimeWindowTrainer(
            config=config,
            model=model,
            training_data=training_data,
            device=device,
            physics=physics,
            losses=losses,
            weighters=weighters
        )
        result = trainer.train_sequential()
        ```
    """
    
    def __init__(
        self,
        config: Dict,
        model: torch.nn.Module,
        training_data: Dict,
        device: torch.device,
        physics,
        losses,
        weighters=None,
        input_normalizer=None,
        data_normalizer=None
    ):
        """
        初始化 Time Window Trainer
        
        Args:
            config: 訓練配置
            model: PINN 模型
            training_data: 完整訓練資料
            device: 訓練設備
            physics: 物理模型
            losses: 損失函數字典
            weighters: 權重器（包含 CausalWeighter）
            input_normalizer: 輸入標準化器（可選）
            data_normalizer: 輸出標準化器（可選）
        """
        self.config = config
        self.model = model
        self.device = device
        self.physics = physics
        self.losses = losses
        self.weighters = weighters
        self.input_normalizer = input_normalizer
        self.data_normalizer = data_normalizer
        
        # 提取時間窗口配置
        training_cfg = config.get('training', {})
        if 'num_time_windows' not in training_cfg or training_cfg.get('num_time_windows') is None:
            training_cfg['num_time_windows'] = 3
            logger.info("ℹ️  num_time_windows 未設定，預設使用 3 個時間窗口")
        self.num_windows = training_cfg.get('num_time_windows', 3)
        self.overlap_ratio = training_cfg.get('time_window_overlap', 0.0)  # Physics Review: 使用 0.0（無重疊）
        self.t_range = self._extract_time_range(config)
        
        # 生成窗口劃分
        self.windows = self._create_time_windows()
        
        # 儲存完整訓練資料
        self.full_training_data = training_data
        
        # 初始化記錄
        self.window_results = []
        
        logger.info(f"🪟 TimeWindowTrainer initialized:")
        logger.info(f"   Number of windows: {self.num_windows}")
        logger.info(f"   Time range: [{self.t_range[0]:.1f}, {self.t_range[1]:.1f}]s")
        logger.info(f"   Overlap ratio: {self.overlap_ratio * 100:.1f}%")
        for idx, (t_start, t_end) in enumerate(self.windows):
            logger.info(f"   Window {idx+1}: [{t_start:.2f}, {t_end:.2f}]s (duration: {t_end - t_start:.2f}s)")
    
    def _extract_time_range(self, config: Dict) -> Tuple[float, float]:
        """
        從配置提取時間範圍
        
        支援：
        - Kolmogorov Flow: config['data']['kolmogorov_config']['time_range']
        - JHTDB: config['data']['jhtdb_config']['time_range']
        """
        if 'kolmogorov_config' in config['data'] and config['data']['kolmogorov_config'].get('enabled', False):
            kol_cfg = config['data']['kolmogorov_config']
            time_range = kol_cfg.get('time_range')
            if time_range:
                return tuple(time_range)

            data_path = kol_cfg.get('data_path')
            if not data_path:
                raise ValueError("Kolmogorov DNS 必須提供 data_path")
            from pinnx.dataio.loaders.kolmogorov import _load_kolmogorov_payload, _backfill_kolmogorov_config
            payload = _load_kolmogorov_payload(Path(data_path))
            if not isinstance(payload, dict):
                raise ValueError("Kolmogorov DNS payload 必須是 dict")
            _backfill_kolmogorov_config(
                kol_cfg,
                payload,
                prefer_config_time_range=True
            )
            if not kol_cfg.get('time_range'):
                raise ValueError("Kolmogorov DNS payload 缺少 time_range")
            return tuple(kol_cfg['time_range'])
        elif 'jhtdb_config' in config['data'] and config['data']['jhtdb_config'].get('enabled', False):
            return tuple(config['data']['jhtdb_config']['time_range'])
        else:
            raise ValueError("Cannot find time_range in config. Check 'kolmogorov_config' or 'jhtdb_config'.")
    
    def _create_time_windows(self) -> List[Tuple[float, float]]:
        """
        劃分時間窗口（基於 JAXpi 無重疊策略）
        
        JAXpi 方式（無重疊）:
        ```
        num_time_steps = len(t_star) // num_windows
        Window 1: t_star[0:num_time_steps]
        Window 2: t_star[num_time_steps:2*num_time_steps]
        ...
        ```
        
        我們的實現（連續時間域）:
        ```
        window_size = (t_max - t_min) / num_windows
        Window 1: [t_min, t_min + window_size]
        Window 2: [t_min + window_size, t_min + 2*window_size]
        ...
        ```
        
        Returns:
            List of (t_start, t_end) tuples
        """
        t_min, t_max = self.t_range
        total_duration = t_max - t_min
        
        if self.num_windows == 1:
            return [(t_min, t_max)]
        
        # Physics Review: 使用無重疊劃分（overlap_ratio = 0.0）
        window_size = total_duration / self.num_windows
        
        windows = []
        for i in range(self.num_windows):
            t_start = t_min + i * window_size
            t_end = t_min + (i + 1) * window_size
            
            # 確保最後一個窗口包含 t_max
            if i == self.num_windows - 1:
                t_end = t_max
            
            windows.append((t_start, t_end))
        
        return windows
    
    def _extract_window_data(self, window_idx: int, t_start: float, t_end: float) -> Dict:
        """
        提取當前窗口的訓練資料
        
        操作：
        1. 過濾 sensor 資料到 [t_start, t_end]
        2. 重新生成 PDE 點（時間在 [t_start, t_end] 範圍內）
        3. 更新 IC（第一個窗口使用配置的 IC，後續窗口從前窗口轉移）
        
        Args:
            window_idx: 當前窗口索引（0-based）
            t_start: 窗口開始時間
            t_end: 窗口結束時間
        
        Returns:
            當前窗口的訓練資料字典
        """
        full_data = self.full_training_data
        window_data = {}
        
        # 1. 過濾 sensor 資料到當前窗口
        if 't_sensors' in full_data and full_data['t_sensors'] is not None:
            t_sensors = full_data['t_sensors']
            
            # 時間mask
            mask = (t_sensors >= t_start) & (t_sensors <= t_end)
            
            # 過濾座標
            if 'x_sensors' in full_data and full_data['x_sensors'] is not None:
                window_data['x_sensors'] = full_data['x_sensors'][mask]
            
            # 過濾時間
            window_data['t_sensors'] = t_sensors[mask]
            
            # 過濾各場變數
            for var in ['u', 'v', 'w', 'p']:
                key = f'{var}_sensors'
                if key in full_data and full_data[key] is not None:
                    window_data[key] = full_data[key][mask]
            
            logger.info(f"   Sensor points in window: {mask.sum().item()}")
        
        # 2. 重新生成 PDE 點（時間範圍在 [t_start, t_end]）
        N_pde = self.config['training']['sampling']['N_pde']
        spatial_bounds = self._get_spatial_bounds()
        
        # 獲取空間維度
        spatial_dim = len(spatial_bounds)
        
        if spatial_dim == 2:
            # 2D (x, y, t)
            x_pde = torch.rand(N_pde, 1, device=self.device) * \
                    (spatial_bounds['x'][1] - spatial_bounds['x'][0]) + spatial_bounds['x'][0]
            y_pde = torch.rand(N_pde, 1, device=self.device) * \
                    (spatial_bounds['y'][1] - spatial_bounds['y'][0]) + spatial_bounds['y'][0]
            t_pde = torch.rand(N_pde, 1, device=self.device) * (t_end - t_start) + t_start
            
            window_data['x_pde'] = torch.cat([x_pde, y_pde, t_pde], dim=1)
        
        elif spatial_dim == 3:
            # 3D (x, y, z, t)
            x_pde = torch.rand(N_pde, 1, device=self.device) * \
                    (spatial_bounds['x'][1] - spatial_bounds['x'][0]) + spatial_bounds['x'][0]
            y_pde = torch.rand(N_pde, 1, device=self.device) * \
                    (spatial_bounds['y'][1] - spatial_bounds['y'][0]) + spatial_bounds['y'][0]
            z_pde = torch.rand(N_pde, 1, device=self.device) * \
                    (spatial_bounds['z'][1] - spatial_bounds['z'][0]) + spatial_bounds['z'][0]
            t_pde = torch.rand(N_pde, 1, device=self.device) * (t_end - t_start) + t_start
            
            window_data['x_pde'] = torch.cat([x_pde, y_pde, z_pde, t_pde], dim=1)
        
        else:
            raise ValueError(f"Unsupported spatial dimension: {spatial_dim}")
        
        logger.info(f"   PDE points generated: {N_pde}")
        
        # 3. IC 處理（第一個窗口使用配置的 IC，後續窗口從前窗口轉移）
        # 注意：IC 轉移在 train_sequential() 中處理，這裡只保留原始 IC（如果是第一個窗口）
        if window_idx == 0:
            # 第一個窗口：使用配置的 IC
            if 'x_ic' in full_data and full_data['x_ic'] is not None:
                window_data['x_ic'] = full_data['x_ic']
                for var in ['u', 'v', 'w', 'p']:
                    key = f'{var}_ic'
                    if key in full_data and full_data[key] is not None:
                        window_data[key] = full_data[key]
        
        # 4. 保留 BC（如有）
        if 'x_bc' in full_data and full_data['x_bc'] is not None:
            window_data['x_bc'] = full_data['x_bc']
            for var in ['u', 'v', 'w', 'p']:
                key = f'{var}_bc'
                if key in full_data and full_data[key] is not None:
                    window_data[key] = full_data[key]
        
        return window_data
    
    def _get_spatial_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        從配置提取空間邊界
        
        Returns:
            空間邊界字典，例如：{'x': [0, 2π], 'y': [0, 2π]}
        """
        if 'kolmogorov_config' in self.config['data'] and self.config['data']['kolmogorov_config'].get('enabled', False):
            kf_cfg = self.config['data']['kolmogorov_config']
            domain_cfg = kf_cfg.get('domain', {})
            bounds = {
                'x': domain_cfg.get('x', [0.0, 2 * np.pi]),
                'y': domain_cfg.get('y', [0.0, 2 * np.pi])
            }
            # 檢查是否有 z（3D）
            if 'z' in domain_cfg:
                bounds['z'] = domain_cfg['z']
            if domain_cfg:
                return bounds

            physics_domain = self.config.get('physics', {}).get('domain', {})
            if physics_domain:
                bounds = {}
                if 'x_range' in physics_domain:
                    bounds['x'] = physics_domain['x_range']
                elif 'x_min' in physics_domain and 'x_max' in physics_domain:
                    bounds['x'] = [physics_domain['x_min'], physics_domain['x_max']]
                if 'y_range' in physics_domain:
                    bounds['y'] = physics_domain['y_range']
                elif 'y_min' in physics_domain and 'y_max' in physics_domain:
                    bounds['y'] = [physics_domain['y_min'], physics_domain['y_max']]
                if 'z_range' in physics_domain:
                    bounds['z'] = physics_domain['z_range']
                elif 'z_min' in physics_domain and 'z_max' in physics_domain:
                    bounds['z'] = [physics_domain['z_min'], physics_domain['z_max']]
                if bounds:
                    return bounds
            return bounds
        
        elif 'jhtdb_config' in self.config['data'] and self.config['data']['jhtdb_config'].get('enabled', False):
            jhtdb_cfg = self.config['data']['jhtdb_config']
            bounds = {
                'x': jhtdb_cfg['domain']['x'],
                'y': jhtdb_cfg['domain']['y'],
                'z': jhtdb_cfg['domain']['z']
            }
            return bounds
        
        else:
            raise ValueError("Cannot find spatial domain in config")
    
    def _transfer_initial_condition(self, window_idx: int, t_start: float) -> Dict:
        """
        從前一窗口的預測值生成當前窗口的 IC（基於 JAXpi 實現）
        
        JAXpi 實現（Line 205-207）:
        ```python
        u0 = model.u_ic_pred_fn(params, t_star[num_time_steps], coords[:, 0], coords[:, 1])
        v0 = model.v_ic_pred_fn(params, t_star[num_time_steps], coords[:, 0], coords[:, 1])
        w0 = model.w_ic_pred_fn(params, t_star[num_time_steps], coords[:, 0], coords[:, 1])
        ```
        
        我們的實現：
        1. 在空間網格上評估前窗口在 t=t_start 的預測
        2. 將這些預測值作為當前窗口的 IC
        
        Args:
            window_idx: 當前窗口索引（必須 > 0）
            t_start: 當前窗口開始時間（= 前窗口結束時間）
        
        Returns:
            IC 資料字典：{'x_ic': coords, 'u_ic': u, 'v_ic': v, ...}
        """
        if window_idx == 0:
            raise ValueError("IC transfer should not be called for the first window")
        
        logger.info(f"🔄 Transferring IC from Window {window_idx} at t={t_start:.2f}s")
        
        # 生成空間網格（用於 IC 評估）
        spatial_bounds = self._get_spatial_bounds()
        N_ic = 256  # IC 網格解析度（可配置）
        
        spatial_dim = len(spatial_bounds)
        
        if spatial_dim == 2:
            # 2D網格
            x_ic = torch.linspace(spatial_bounds['x'][0], spatial_bounds['x'][1], N_ic, device=self.device)
            y_ic = torch.linspace(spatial_bounds['y'][0], spatial_bounds['y'][1], N_ic, device=self.device)
            X, Y = torch.meshgrid(x_ic, y_ic, indexing='ij')
            T = torch.full_like(X, t_start)
            
            # 組合座標 [N_ic^2, 3]
            coords = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
        
        elif spatial_dim == 3:
            # 3D網格（降低解析度避免記憶體問題）
            N_ic_3d = 64
            x_ic = torch.linspace(spatial_bounds['x'][0], spatial_bounds['x'][1], N_ic_3d, device=self.device)
            y_ic = torch.linspace(spatial_bounds['y'][0], spatial_bounds['y'][1], N_ic_3d, device=self.device)
            z_ic = torch.linspace(spatial_bounds['z'][0], spatial_bounds['z'][1], N_ic_3d, device=self.device)
            X, Y, Z = torch.meshgrid(x_ic, y_ic, z_ic, indexing='ij')
            T = torch.full_like(X, t_start)
            
            # 組合座標 [N_ic^3, 4]
            coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1)
        
        else:
            raise ValueError(f"Unsupported spatial dimension: {spatial_dim}")
        
        # 評估前窗口的預測（Physics Review: 直接使用預測值，無校正）
        with torch.no_grad():
            predictions = self.model(coords)
        
        # 構建 IC 資料（修正格式以匹配 prepare_time_window_data 期望）
        # 分離 coords 張量為獨立的 x, y, t 陣列
        if spatial_dim == 2:
            # 2D: coords shape is [N, 3] containing [x, y, t]
            ic_data = {
                'x': coords[:, 0:1],  # Extract x column
                'y': coords[:, 1:2],  # Extract y column
                't': coords[:, 2:3],  # Extract t column
            }
        elif spatial_dim == 3:
            # 3D: coords shape is [N, 4] containing [x, y, z, t]
            ic_data = {
                'x': coords[:, 0:1],
                'y': coords[:, 1:2],
                'z': coords[:, 2:3],
                't': coords[:, 3:4],
            }
        else:
            raise ValueError(f"Unsupported spatial_dim: {spatial_dim}. Must be 2 or 3.")
        
        # 根據模型輸出維度分配各變數（移除 _ic 後綴）
        # 假設模型輸出順序：[u, v, (w), p]
        if predictions.shape[1] >= 2:
            ic_data['u'] = predictions[:, 0:1]  # No '_ic' suffix
            ic_data['v'] = predictions[:, 1:2]
        
        if spatial_dim == 3 and predictions.shape[1] >= 3:
            ic_data['w'] = predictions[:, 2:3]
            if predictions.shape[1] >= 4:
                ic_data['p'] = predictions[:, 3:4]
        elif spatial_dim == 2 and predictions.shape[1] >= 3:
            ic_data['p'] = predictions[:, 2:3]
        
        logger.info(f"   IC transferred: {coords.shape[0]} points")
        
        # 可選：診斷 IC 品質
        if logger.isEnabledFor(logging.DEBUG):
            u_ic = ic_data['u']  # Changed from 'u_ic'
            logger.debug(f"   IC statistics:")
            logger.debug(f"     u: min={u_ic.min():.4f}, max={u_ic.max():.4f}, mean={u_ic.mean():.4f}")
        
        return ic_data
    
    def _init_wandb_once(self):
        """
        在外層初始化 WandB（對齊 JaxPI 實現）
        
        JaxPI 做法（Line 108-135）：
        - 在 train_and_evaluate 最外層調用 wandb.init()
        - 每個 time window 共享同一個 run
        - 使用 step_offset 讓各窗口的 step 連續
        
        Returns:
            wandb_run: WandB run 實例
        """
        import wandb
        from pathlib import Path
        
        # 載入 WandB 配置
        wandb_config_path = Path('.wandb_config')
        wandb_api_key = None
        wandb_project = 'pinns-sparse-turbulence'
        
        if wandb_config_path.exists():
            with open(wandb_config_path, 'r') as f:
                for line in f:
                    if line.startswith('WANDB_API_KEY='):
                        wandb_api_key = line.split('=')[1].strip()
                    elif line.startswith('WANDB_PROJECT='):
                        project_value = line.split('=')[1].strip()
                        if project_value:
                            wandb_project = project_value
        
        if wandb_api_key:
            os.environ['WANDB_API_KEY'] = wandb_api_key
        
        # 從 config 提取實驗配置
        exp_config = self.config.get('experiment', {})
        
        # 初始化 WandB（只調用一次）
        wandb_run = wandb.init(
            project=wandb_project,
            name=exp_config.get('name', 'time_window_training'),
            group=exp_config.get('group', None),
            tags=exp_config.get('tags', []) + ['time_window'],  # 添加 time_window 標籤
            config=self.config,
            reinit=True
        )
        
        logger.info(f"✅ WandB 初始化完成（Time Window 模式）")
        logger.info(f"   Project: {wandb_project}")
        logger.info(f"   Run: {exp_config.get('name')}")
        logger.info(f"   All {self.num_windows} windows will share this run")
        
        return wandb_run
    
    def train_sequential(self) -> Dict:
        """
        主訓練循環：序列訓練各窗口
        
        基於 JAXpi 實現（Line 166-209）：
        ```python
        for idx in range(num_time_windows):
            # 1. 初始化模型（Window 2+ 使用 Transfer Learning）
            model = NavierStokes(config, t, coords, u0, v0, w0, nu)
            
            if config.transfer_learning and idx > 0:
                state = restore_checkpoint(model.state, ckpt_path_prev)
                model.state = _create_train_state(config, tx=model.tx, params=state.params)
            
            # 2. 訓練當前窗口
            model = train_one_window(...)
            
            # 3. 更新 IC
            if idx < num_time_windows - 1:
                u0 = model.u_ic_pred_fn(params, t_next, coords)
        ```
        
        Returns:
            訓練結果字典：{'window_results': [...], 'final_loss': ...}
        """
        import wandb
        from pinnx.train.trainer_builder import TrainerBuilder
        
        # ========== 關鍵修正：在外層初始化 WandB（對齊 JaxPI） ==========
        wandb_run = self._init_wandb_once()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🪟 Starting Time Window Training")
        logger.info(f"   Total windows: {self.num_windows}")
        logger.info(f"   Time range: [{self.t_range[0]:.1f}, {self.t_range[1]:.1f}]s")
        logger.info(f"   Strategy: {'No overlap' if self.overlap_ratio == 0.0 else f'{self.overlap_ratio*100:.0f}% overlap'}")
        logger.info(f"{'='*70}\n")
        
        all_results = []
        cumulative_steps = 0  # ========== 新增：累計步數（對齊 JaxPI step_offset） ==========
        
        for idx, (t_start, t_end) in enumerate(self.windows):
            logger.info(f"\n{'='*70}")
            logger.info(f"🪟 Training Window {idx+1}/{self.num_windows}")
            logger.info(f"   Time interval: [{t_start:.2f}, {t_end:.2f}]s")
            logger.info(f"   Duration: {t_end - t_start:.2f}s")
            logger.info(f"{'='*70}\n")
            
            # 1. 載入當前窗口的資料（使用數據加載器）
            from pinnx.dataio.loaders import prepare_time_window_data
            
            # 準備 IC 轉移（Window 2+）
            prev_window_ic = None
            if idx > 0:
                prev_window_ic = self._transfer_initial_condition(idx, t_start)
                logger.info(f"   IC transferred from Window {idx}")
            
            # 調用數據加載器
            window_data = prepare_time_window_data(
                config=self.config,
                device=self.device,
                window_idx=idx,
                num_windows=self.num_windows,
                prev_window_ic=prev_window_ic
            )
            
            # 2. Transfer Learning：載入前窗口的模型參數（Window 2+）
            if idx > 0 and self.config.get('training', {}).get('transfer_learning', True):
                prev_checkpoint_path = self._get_checkpoint_path(idx - 1)
                if os.path.exists(prev_checkpoint_path):
                    logger.info(f"   Loading previous window checkpoint: {prev_checkpoint_path}")
                    checkpoint = torch.load(prev_checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"   ✅ Transfer Learning applied")
                else:
                    logger.warning(f"   ⚠️ Previous checkpoint not found, starting from current parameters")
            
            # 3. 更新 CausalWeighter 的時間範圍（如果使用）
            if self.weighters and 'causal' in self.weighters and self.weighters['causal'] is not None:
                causal_weighter = self.weighters['causal']
                # 檢查是否有 time_range 屬性（v1 版本）
                if hasattr(causal_weighter, 'time_range'):
                    # 舊版 CausalWeighter (v1)
                    causal_weighter.time_range = (t_start, t_end)
                    logger.info(f"   Causal weighter (v1) updated to [{t_start:.2f}, {t_end:.2f}]s")
                else:
                    # 新版 CausalWeighterPerComponent (v2) 不需要時間範圍
                    logger.info(f"   Causal weighter (v2) active - time range handled dynamically")
            
            # 4. 使用 TrainerBuilder 創建當前窗口的 Trainer
            # ⚠️ 關鍵修正：將總 epochs 平分給各 window
            total_epochs = self.config['training'].get('epochs', 1000)
            epochs_per_window = total_epochs // self.num_windows
            remaining_epochs = total_epochs % self.num_windows
            
            # 最後一個 window 處理剩餘的 epochs
            window_epochs = epochs_per_window + (remaining_epochs if idx == self.num_windows - 1 else 0)
            
            # 創建當前 window 的配置副本（避免修改原始配置）
            import copy
            window_config = copy.deepcopy(self.config)
            window_config['training']['epochs'] = window_epochs
            
            logger.info(f"   Building trainer for Window {idx+1}...")
            logger.info(f"   Total epochs: {total_epochs}, This window: {window_epochs}")
            builder = TrainerBuilder(window_config, self.device, skip_wandb_init=True)  # ✅ 跳過 WandB 初始化
            builder.with_model(self.model)
            builder.with_physics(self.physics)
            builder.with_losses(self.losses)
            builder.with_training_data(window_data)
            
            # 注入標準化器（使用正確的方法名）
            if self.input_normalizer:
                builder.with_normalizer(self.input_normalizer)
            # data_normalizer 會在 build() 中自動創建
            
            # 注入權重器
            if self.weighters:
                builder.with_weighters(self.weighters)
            
            # ========== 新增：注入 WandB run 和 step_offset（對齊 JaxPI） ==========
            builder.with_wandb_run(wandb_run)
            builder.with_step_offset(cumulative_steps)
            
            # 構建 Trainer
            trainer = builder.build()
            
            # 5. 訓練當前窗口
            logger.info(f"   Starting training for Window {idx+1}...")
            result = trainer.train()
            all_results.append(result)
            
            # ========== 更新累計步數（使用實際訓練的 epochs，對齊 JaxPI step_offset） ==========
            # 從訓練結果中獲取實際完成的 epochs，用於下一個 window 的 step_offset
            completed_epochs = result.get('epochs_completed', window_epochs)
            cumulative_steps += completed_epochs
            logger.info(f"   Window {idx+1} completed {completed_epochs} epochs")
            logger.info(f"   Cumulative steps: {cumulative_steps}")
            
            logger.info(f"   ✅ Window {idx+1} training completed")
            logger.info(f"   Final loss: {result.get('final_loss', 'N/A'):.6f}")
            
            # 6. 保存窗口 checkpoint
            checkpoint_path = self._get_checkpoint_path(idx)
            self._save_window_checkpoint(idx, t_start, t_end, checkpoint_path)
            logger.info(f"   💾 Checkpoint saved: {checkpoint_path}")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎉 Time Window Training Completed!")
        logger.info(f"   Total windows trained: {self.num_windows}")
        logger.info(f"   Final loss: {all_results[-1].get('final_loss', 'N/A'):.6f}")
        logger.info(f"{'='*70}\n")
        
        return {
            'window_results': all_results,
            'final_loss': all_results[-1].get('final_loss', float('inf')),
            'num_windows': self.num_windows
        }
    
    def _get_checkpoint_path(self, window_idx: int) -> str:
        """
        生成窗口 checkpoint 路徑
        
        格式：./checkpoints/window_{idx+1}_t{start}_{end}.pth
        例如：./checkpoints/window_1_t15_25.pth
        """
        t_start, t_end = self.windows[window_idx]
        checkpoint_dir = self.config.get('checkpointing', {}).get('checkpoint_dir', './checkpoints')
        filename = f"window_{window_idx+1}_t{int(t_start)}_{int(t_end)}.pth"
        return os.path.join(checkpoint_dir, filename)
    
    def _save_window_checkpoint(self, window_idx: int, t_start: float, t_end: float, path: str):
        """
        保存窗口 checkpoint
        
        包含：
        - 模型參數
        - 窗口資訊（索引、時間範圍）
        - 配置（供恢復使用）
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'window_idx': window_idx,
            'time_range': (t_start, t_end),
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'num_windows': self.num_windows
        }
        
        # 可選：保存 optimizer 狀態（如需要繼續訓練）
        # checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)

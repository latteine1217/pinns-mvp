"""
PINNs 視覺化模組

提供專業級的評估結果視覺化功能：
- 三行對比圖（預測 | 真實 | 誤差）
- 流場總覽圖
- 誤差分析圖
- 物理一致性圖

使用範例:
    >>> from pinnx.evals.visualizer import Visualizer
    >>> viz = Visualizer(output_dir="results")
    >>> viz.plot_comparison(pred_data, ref_data, coords)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .metrics import relative_L2, rmse_metrics, conservation_error

logger = logging.getLogger(__name__)


class Visualizer:
    """PINNs 評估視覺化器"""
    
    def __init__(self, output_dir: str = "results/plots", config: Optional[Dict] = None):
        """
        初始化視覺化器
        
        Args:
            output_dir: 輸出目錄
            config: 可視化配置字典
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 預設配置
        default_config = {
            'dpi': 300,
            'figsize': (15, 10),
            'cmap_field': 'RdBu_r',
            'cmap_error': 'Reds',
            'font_size': 12,
        }
        self.config = {**default_config, **(config or {})}
        
        # 設定 matplotlib
        plt.rcParams.update({
            'figure.figsize': self.config['figsize'],
            'font.size': self.config['font_size'],
            'savefig.dpi': self.config['dpi'],
            'savefig.bbox': 'tight',
        })
        
        logger.info(f"視覺化器初始化完成，輸出至: {self.output_dir}")
    
    def plot_three_panel(self,
                        pred_data: Dict[str, torch.Tensor],
                        ref_data: Dict[str, torch.Tensor],
                        coords: torch.Tensor,
                        field_name: str,
                        grid_shape: Optional[Tuple[int, int]] = None) -> str:
        """
        三行對比圖：預測 | 真實 | 誤差
        
        Args:
            pred_data: 預測數據字典 {"u": tensor, "v": tensor, "p": tensor}
            ref_data: 參考數據字典
            coords: 座標張量 [N, 2] (x, y) 或 [N, 3] (t, x, y)
            field_name: 場名稱 ("u", "v", "p")
            grid_shape: 網格形狀 (Nx, Ny)，用於 reshape
            
        Returns:
            儲存路徑
        """
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)
        
        # 提取數據
        pred = pred_data[field_name].detach().cpu().numpy()
        ref = ref_data[field_name].detach().cpu().numpy()
        error = np.abs(pred - ref)
        
        # 座標處理
        if coords.shape[1] == 3:  # (t, x, y)
            x = coords[:, 1].detach().cpu().numpy()
            y = coords[:, 2].detach().cpu().numpy()
        else:  # (x, y)
                if isinstance(coords, torch.Tensor):
                    x = coords[:, 0].detach().cpu().numpy()
                    y = coords[:, 1].detach().cpu().numpy()
                else:
                    x = np.asarray(coords)[:, 0]
                    y = np.asarray(coords)[:, 1]
        
        # 統一色階
        vmin = min(pred.min(), ref.min())
        vmax = max(pred.max(), ref.max())
        
        # 如果提供網格形狀，使用 imshow
        if grid_shape is not None:
            extent = (x.min(), x.max(), y.min(), y.max())
            pred_2d = pred.reshape(grid_shape)
            ref_2d = ref.reshape(grid_shape)
            error_2d = error.reshape(grid_shape)
            
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(pred_2d.T, origin='lower', extent=extent, 
                           vmin=vmin, vmax=vmax, cmap=self.config['cmap_field'], aspect='auto')
            ax1.set_title(f'PINNs Prediction\n{field_name.upper()}')
            
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(ref_2d.T, origin='lower', extent=extent,
                           vmin=vmin, vmax=vmax, cmap=self.config['cmap_field'], aspect='auto')
            ax2.set_title(f'Reference\n{field_name.upper()}')
            
            ax3 = fig.add_subplot(gs[0, 2])
            im3 = ax3.imshow(error_2d.T, origin='lower', extent=extent,
                           cmap=self.config['cmap_error'], aspect='auto')
            ax3.set_title(f'Absolute Error\n{field_name.upper()}')
            
            for ax in [ax1, ax2, ax3]:
                ax.set_xlabel('x')
                ax.set_ylabel('y')
        else:
            # 使用 scatter plot
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.scatter(x, y, c=pred, vmin=vmin, vmax=vmax, 
                            cmap=self.config['cmap_field'], s=5)
            ax1.set_title(f'PINNs Prediction\n{field_name.upper()}')
            
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.scatter(x, y, c=ref, vmin=vmin, vmax=vmax,
                            cmap=self.config['cmap_field'], s=5)
            ax2.set_title(f'Reference\n{field_name.upper()}')
            
            ax3 = fig.add_subplot(gs[0, 2])
            im3 = ax3.scatter(x, y, c=error, cmap=self.config['cmap_error'], s=5)
            ax3.set_title(f'Absolute Error\n{field_name.upper()}')
            
            for ax in [ax1, ax2, ax3]:
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.grid(True, alpha=0.3)
        
        # 色階條
        cbar_ax = fig.add_subplot(gs[0, 3])
        plt.colorbar(im1, cax=cbar_ax, label=f'{field_name.upper()}')
        
        # 計算指標
        rel_l2 = relative_L2(pred_data[field_name], ref_data[field_name]).item()
        rmse_dict = rmse_metrics(pred_data[field_name].unsqueeze(1) if pred_data[field_name].dim() == 1 else pred_data[field_name],
                                 ref_data[field_name].unsqueeze(1) if ref_data[field_name].dim() == 1 else ref_data[field_name])
        
        fig.suptitle(f'{field_name.upper()} Field Comparison | '
                    f'Rel. L2: {rel_l2:.4f} | Rel. RMSE: {rmse_dict.get("relative_rmse_" + field_name, 0):.4f}',
                    fontsize=14)
        
        # 儲存
        save_path = self.output_dir / f"comparison_3panel_{field_name}.png"
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"三行對比圖已儲存: {save_path}")
        return str(save_path)
    
    def plot_flow_overview(self,
                          pred_data: Dict[str, torch.Tensor],
                          ref_data: Optional[Dict[str, torch.Tensor]] = None,
                          coords: Optional[torch.Tensor] = None,
                          grid_shape: Optional[Tuple[int, int]] = None) -> str:
        """
        流場總覽：u, v, p 綜合展示
        
        Args:
            pred_data: 預測數據
            ref_data: 參考數據（可選）
            coords: 座標（可選）
            grid_shape: 網格形狀（可選）
            
        Returns:
            儲存路徑
        """
        fields = ['u', 'v', 'p']
        n_fields = len(fields)
        n_rows = 2 if ref_data else 1
        
        fig, axes = plt.subplots(n_rows, n_fields, figsize=(18, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, field in enumerate(fields):
            pred = pred_data[field].detach().cpu().numpy()
            
            if grid_shape and coords is not None:
                # 使用 imshow
                x = coords[:, 0].detach().cpu().numpy()
                y = coords[:, 1].detach().cpu().numpy()
                extent = (x.min(), x.max(), y.min(), y.max())
                pred_2d = pred.reshape(grid_shape)
                
                im = axes[0, i].imshow(pred_2d.T, origin='lower', extent=extent,
                                      cmap=self.config['cmap_field'], aspect='auto')
                axes[0, i].set_title(f'PINNs {field.upper()}')
                plt.colorbar(im, ax=axes[0, i])
                
                if ref_data:
                    ref = ref_data[field].detach().cpu().numpy().reshape(grid_shape)
                    im = axes[1, i].imshow(ref.T, origin='lower', extent=extent,
                                          cmap=self.config['cmap_field'], aspect='auto')
                    axes[1, i].set_title(f'Reference {field.upper()}')
                    plt.colorbar(im, ax=axes[1, i])
            else:
                # 使用直方圖
                axes[0, i].hist(pred, bins=50, alpha=0.7, label='PINNs')
                if ref_data:
                    ref = ref_data[field].detach().cpu().numpy()
                    axes[0, i].hist(ref, bins=50, alpha=0.7, label='Reference')
                axes[0, i].set_title(f'{field.upper()} Distribution')
                axes[0, i].legend()
                axes[0, i].set_xlabel(field.upper())
                axes[0, i].set_ylabel('Count')
        
        fig.suptitle('Flow Field Overview', fontsize=16)
        
        save_path = self.output_dir / "flow_field_overview.png"
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"流場總覽圖已儲存: {save_path}")
        return str(save_path)
    
    def plot_error_analysis(self,
                           pred_data: Dict[str, torch.Tensor],
                           ref_data: Dict[str, torch.Tensor],
                           coords: torch.Tensor,
                           grid_shape: Optional[Tuple[int, int]] = None) -> str:
        """
        誤差分析圖：u, v, p 的誤差空間分布
        
        Returns:
            儲存路徑
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fields = ['u', 'v', 'p']
        
        for i, field in enumerate(fields):
            pred = pred_data[field].detach().cpu().numpy()
            ref = ref_data[field].detach().cpu().numpy()
            error = np.abs(pred - ref)
            
            if grid_shape:
                x = coords[:, 0].detach().cpu().numpy()
                y = coords[:, 1].detach().cpu().numpy()
                extent = (x.min(), x.max(), y.min(), y.max())
                error_2d = error.reshape(grid_shape)
                
                im = axes[i].imshow(error_2d.T, origin='lower', extent=extent,
                                   cmap=self.config['cmap_error'], aspect='auto')
                axes[i].set_title(f'{field.upper()} Error')
                axes[i].set_xlabel('x')
                axes[i].set_ylabel('y')
                plt.colorbar(im, ax=axes[i])
            else:
                axes[i].hist(error, bins=50)
                axes[i].set_title(f'{field.upper()} Error Distribution')
                axes[i].set_xlabel('Absolute Error')
                axes[i].set_ylabel('Count')
        
        fig.suptitle('Error Analysis', fontsize=16)
        
        save_path = self.output_dir / "error_analysis.png"
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"誤差分析圖已儲存: {save_path}")
        return str(save_path)
    
    def plot_physics_consistency(self,
                                pred_data: Dict[str, torch.Tensor],
                                coords: torch.Tensor,
                                grid_shape: Optional[Tuple[int, int]] = None) -> str:
        """
        物理一致性圖：質量守恆（散度場）
        
        Returns:
            儲存路徑
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 計算散度
        try:
            div = conservation_error(pred_data['u'], pred_data['v'], coords)
            div_np = div.detach().cpu().numpy()
            
            if grid_shape:
                x = coords[:, 0].detach().cpu().numpy()
                y = coords[:, 1].detach().cpu().numpy()
                extent = (x.min(), x.max(), y.min(), y.max())
                div_2d = div_np.reshape(grid_shape)
                
                im = ax.imshow(div_2d.T, origin='lower', extent=extent,
                              cmap='seismic', aspect='auto')
                ax.set_title('Mass Conservation Error (Divergence)')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                plt.colorbar(im, ax=ax, label='∇·u')
            else:
                ax.hist(div_np, bins=50)
                ax.set_title('Divergence Distribution')
                ax.set_xlabel('∇·u')
                ax.set_ylabel('Count')
            
            # 添加統計
            ax.text(0.02, 0.98, f'Mean: {div_np.mean():.2e}\nStd: {div_np.std():.2e}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except Exception as e:
            logger.warning(f"無法計算散度: {e}")
            ax.text(0.5, 0.5, 'Divergence calculation failed',
                   ha='center', va='center', transform=ax.transAxes)
        
        save_path = self.output_dir / "physics_consistency.png"
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"物理一致性圖已儲存: {save_path}")
        return str(save_path)
    
    def generate_report(self,
                       pred_data: Dict[str, torch.Tensor],
                       ref_data: Dict[str, torch.Tensor],
                       coords: torch.Tensor,
                       grid_shape: Optional[Tuple[int, int]] = None,
                       model_info: Optional[Dict] = None) -> str:
        """
        生成完整評估報告（包含所有圖表）
        
        Args:
            pred_data: 預測數據
            ref_data: 參考數據
            coords: 座標
            grid_shape: 網格形狀
            model_info: 模型資訊
            
        Returns:
            報告路徑
        """
        logger.info("開始生成完整評估報告...")
        
        # 生成所有圖表
        for field in ['u', 'v', 'p']:
            self.plot_three_panel(pred_data, ref_data, coords, field, grid_shape)
        
        self.plot_flow_overview(pred_data, ref_data, coords, grid_shape)
        self.plot_error_analysis(pred_data, ref_data, coords, grid_shape)
        self.plot_physics_consistency(pred_data, coords, grid_shape)
        
        # 生成文字報告
        report_path = self.output_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("PINNs 評估報告\n")
            f.write("="*60 + "\n\n")
            
            if model_info:
                f.write("模型資訊:\n")
                for key, val in model_info.items():
                    f.write(f"  {key}: {val}\n")
                f.write("\n")
            
            f.write("場數據統計:\n")
            f.write("-"*60 + "\n")
            for field in ['u', 'v', 'p']:
                pred = pred_data[field].detach().cpu().numpy()
                ref = ref_data[field].detach().cpu().numpy()
                
                rel_l2 = relative_L2(pred_data[field], ref_data[field]).item()
                
                f.write(f"\n{field.upper()} 場:\n")
                f.write(f"  預測範圍: [{pred.min():.4f}, {pred.max():.4f}]\n")
                f.write(f"  真實範圍: [{ref.min():.4f}, {ref.max():.4f}]\n")
                f.write(f"  相對 L2 誤差: {rel_l2:.6f} {'✅' if rel_l2 <= 0.15 else '❌'}\n")
        
        logger.info(f"✅ 完整報告已生成: {report_path}")
        return str(report_path)

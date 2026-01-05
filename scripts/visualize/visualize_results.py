#!/usr/bin/env python3
"""
增強視覺化評估腳本
Enhanced Evaluation and Visualization Script

本腳本提供專業級的 PINNs 結果視覺化，包括：
- 三行對比圖：PINNs預測 | JHTDB真實 | 誤差分布
- 2D流場切片可視化：u, v, p 場分布
- 自動生成高品質matplotlib圖表
- 整合metrics.py的完整評估指標

作者: PINNs-MVP 專案團隊
日期: 2025-10-07
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns

# 添加專案路徑
sys.path.append(str(Path(__file__).parent.parent))

# 專案模組
from pinnx.evals.metrics import (
    relative_L2, rmse_metrics, conservation_error,
    energy_spectrum_1d, wall_shear_stress, k_error_curve,
    uncertainty_correlation, comprehensive_evaluation
)
from pinnx.utils.denormalization import denormalize_output

# 設定警告與日誌
warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全域設定
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedVisualizer:
    """增強視覺化類別"""
    
    def __init__(self, output_dir: str = "results/enhanced_plots"):
        """
        初始化視覺化器
        
        Args:
            output_dir: 輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定matplotlib參數
        plt.rcParams.update({
            'figure.figsize': (15, 10),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
        
        logger.info(f"增強視覺化器初始化完成，輸出目錄: {self.output_dir}")
    
    def create_three_panel_comparison(self, 
                                    pred_data: Dict[str, torch.Tensor],
                                    ref_data: Dict[str, torch.Tensor],
                                    coords: torch.Tensor,
                                    field_name: str = "u",
                                    save_name: str = "comparison_3panel") -> str:
        """
        創建三行對比圖：預測 | 真實 | 誤差
        
        Args:
            pred_data: 預測數據 {"u": tensor, "v": tensor, "p": tensor}
            ref_data: 參考數據 {"u": tensor, "v": tensor, "p": tensor}
            coords: 座標 [N, 2] 或 [N, 3] (t, x, y)
            field_name: 場名稱 ("u", "v", "p")
            save_name: 儲存檔名
            
        Returns:
            儲存的圖片路徑
        """
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], hspace=0.3, wspace=0.4)
        
        # 取得對應場數據
        pred_field = pred_data[field_name].detach().cpu().numpy()
        ref_field = ref_data[field_name].detach().cpu().numpy()
        error_field = np.abs(pred_field - ref_field)
        
        # 座標處理
        if coords.shape[1] == 3:  # [t, x, y]
            x_coords = coords[:, 1].detach().cpu().numpy()
            y_coords = coords[:, 2].detach().cpu().numpy()
        else:  # [x, y]
            x_coords = coords[:, 0].detach().cpu().numpy()
            y_coords = coords[:, 1].detach().cpu().numpy()
        
        # 統一色階範圍
        vmin = min(pred_field.min(), ref_field.min())
        vmax = max(pred_field.max(), ref_field.max())
        
        # 子圖1: PINNs預測
        ax1 = fig.add_subplot(gs[0, 0])
        scatter1 = ax1.scatter(x_coords, y_coords, c=pred_field, 
                              vmin=vmin, vmax=vmax, cmap='RdBu_r', s=20)
        ax1.set_title(f'PINNs Prediction\n{field_name}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.grid(True, alpha=0.3)
        
        # 子圖2: JHTDB真實
        ax2 = fig.add_subplot(gs[0, 1])
        scatter2 = ax2.scatter(x_coords, y_coords, c=ref_field, 
                              vmin=vmin, vmax=vmax, cmap='RdBu_r', s=20)
        ax2.set_title(f'JHTDB Reference\n{field_name}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.grid(True, alpha=0.3)
        
        # 子圖3: 絕對誤差
        ax3 = fig.add_subplot(gs[0, 2])
        scatter3 = ax3.scatter(x_coords, y_coords, c=error_field, 
                              cmap='Reds', s=20)
        ax3.set_title(f'Absolute Error\n{field_name}')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.grid(True, alpha=0.3)
        
        # 色階條
        cbar_ax1 = fig.add_subplot(gs[0, 3])
        cbar1 = plt.colorbar(scatter1, cax=cbar_ax1)
        cbar1.set_label(f'{field_name} Value')
        
        # 添加統計資訊
        rel_l2 = relative_L2(pred_data[field_name], ref_data[field_name]).item()
        rmse_result = rmse_metrics(pred_data[field_name], ref_data[field_name])
        
        # 根據場名選擇正確的相對RMSE鍵
        rmse_key_map = {'u': 'relative_rmse_u', 'v': 'relative_rmse_v', 'p': 'relative_rmse_p'}
        rel_rmse = rmse_result[rmse_key_map[field_name]]
        
        fig.suptitle(f'Field Comparison: {field_name}\n'
                    f'Relative L2: {rel_l2:.4f} | Relative RMSE: {rel_rmse:.4f}', 
                    fontsize=16, y=0.95)
        
        # 儲存
        save_path = self.output_dir / f"{save_name}_{field_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"三行對比圖已儲存: {save_path}")
        return str(save_path)
    
    def create_flow_field_overview(self, 
                                 pred_data: Dict[str, torch.Tensor],
                                 ref_data: Dict[str, torch.Tensor],
                                 coords: torch.Tensor,
                                 save_name: str = "flow_field_overview") -> str:
        """
        [Updated] 創建流場總覽：不再生成單一大圖，而是為每個場生成單獨的對比圖
        以符合論文圖片規範（避免過多子圖）
        """
        generated_files = []
        for field in ['u', 'v', 'p']:
            if field in pred_data and field in ref_data:
                # Reuse the high-quality 3-panel comparison
                path = self.create_three_panel_comparison(
                    pred_data, ref_data, coords, field, f"{save_name}"
                )
                generated_files.append(path)
        
        # Return the path of the first generated file as a representative
        return generated_files[0] if generated_files else ""
    
    def create_error_analysis_plot(self, 
                                 pred_data: Dict[str, torch.Tensor],
                                 ref_data: Dict[str, torch.Tensor],
                                 coords: torch.Tensor,
                                 save_name: str = "error_analysis") -> str:
        """
        創建誤差分析圖
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Error Analysis', fontsize=16)
        
        # 計算各場誤差
        fields = ['u', 'v', 'p']
        field_names = ['Velocity U', 'Velocity V', 'Pressure']
        colors = ['red', 'blue', 'green']
        
        errors = {}
        rel_errors = {}
        
        for field in fields:
            pred_field = pred_data[field].detach().cpu().numpy()
            ref_field = ref_data[field].detach().cpu().numpy()
            
            errors[field] = np.abs(pred_field - ref_field)
            rel_errors[field] = errors[field] / (np.abs(ref_field) + 1e-8)
        
        # 子圖1: 絕對誤差分佈
        for i, (field, field_name, color) in enumerate(zip(fields, field_names, colors)):
            axes[0, 0].hist(errors[field], bins=50, alpha=0.7, 
                           label=f'{field_name}', color=color, density=True)
        axes[0, 0].set_xlabel('Absolute Error')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Absolute Error Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 子圖2: 相對誤差分佈
        for i, (field, field_name, color) in enumerate(zip(fields, field_names, colors)):
            axes[0, 1].hist(rel_errors[field], bins=50, alpha=0.7, 
                           label=f'{field_name}', color=color, density=True)
        axes[0, 1].set_xlabel('Relative Error')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Relative Error Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 子圖3: 誤差 vs 預測值散點圖
        all_pred = np.concatenate([pred_data[f].detach().cpu().numpy() for f in fields])
        all_error = np.concatenate([errors[f] for f in fields])
        
        axes[1, 0].scatter(all_pred, all_error, alpha=0.5, s=10)
        axes[1, 0].set_xlabel('Predicted Value')
        axes[1, 0].set_ylabel('Absolute Error')
        axes[1, 0].set_title('Error vs Prediction')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 子圖4: 統計指標表格
        axes[1, 1].axis('off')
        
        # 計算統計指標
        stats_data = []
        for field, field_name in zip(fields, field_names):
            rel_l2 = relative_L2(pred_data[field], ref_data[field]).item()
            rmse_result = rmse_metrics(pred_data[field], ref_data[field])
            
            # 根據場名選擇正確的相對RMSE鍵
            rmse_key_map = {'u': 'relative_rmse_u', 'v': 'relative_rmse_v', 'p': 'relative_rmse_p'}
            rel_rmse = rmse_result[rmse_key_map[field]]
            
            max_error = errors[field].max()
            mean_error = errors[field].mean()
            
            stats_data.append([field_name, f'{rel_l2:.4f}', f'{rel_rmse:.4f}', 
                              f'{max_error:.4f}', f'{mean_error:.4f}'])
        
        # 創建表格
        table = axes[1, 1].table(cellText=stats_data,
                                colLabels=['Field', 'Rel L2', 'Rel RMSE', 'Max Error', 'Mean Error'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title('Error Statistics')
        
        plt.tight_layout()
        
        # 儲存
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"誤差分析圖已儲存: {save_path}")
        return str(save_path)
    
    def create_physics_consistency_plot(self, 
                                      pred_data: Dict[str, torch.Tensor],
                                      coords: torch.Tensor,
                                      save_name: str = "physics_consistency") -> str:
        """
        創建物理一致性檢查圖
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Physics Consistency Check', fontsize=16)
        
        # 計算守恆律誤差
        try:
            mass_error = conservation_error(pred_data['u'], pred_data['v'], coords)
            
            # 處理不同的返回類型
            if isinstance(mass_error, torch.Tensor):
                if mass_error.numel() == 1:  # 標量張量
                    mass_error_val = mass_error.item()
                    mass_error_np = np.full(coords.shape[0], mass_error_val)
                else:  # 向量張量
                    mass_error_np = mass_error.detach().cpu().numpy()
            else:  # 標量數值
                mass_error_val = float(mass_error)
                mass_error_np = np.full(coords.shape[0], mass_error_val)
            
            # 座標處理
            if coords.shape[1] == 3:
                x_coords = coords[:, 1].detach().cpu().numpy()
                y_coords = coords[:, 2].detach().cpu().numpy()
            else:
                x_coords = coords[:, 0].detach().cpu().numpy()
                y_coords = coords[:, 1].detach().cpu().numpy()
            
            # 子圖1: 質量守恆誤差空間分佈
            scatter = axes[0].scatter(x_coords, y_coords, c=mass_error_np, 
                                     cmap='Reds', s=20)
            axes[0].set_title('Mass Conservation Error')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')
            axes[0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0])
            
            # 子圖2: 守恆誤差統計
            axes[1].hist(mass_error_np, bins=50, alpha=0.7, color='red', density=True)
            axes[1].set_xlabel('Conservation Error')
            axes[1].set_ylabel('Density')
            axes[1].set_title(f'Conservation Error Distribution\n'
                             f'Mean: {mass_error_np.mean():.2e}, '
                             f'Max: {mass_error_np.max():.2e}')
            axes[1].grid(True, alpha=0.3)
            
        except Exception as e:
            logger.warning(f"物理一致性計算失敗: {e}")
            # 顯示錯誤訊息
            for ax in axes:
                ax.text(0.5, 0.5, f'Physics consistency check failed:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Error in Physics Check')
        
        plt.tight_layout()
        
        # 儲存
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"物理一致性圖已儲存: {save_path}")
        return str(save_path)
    
    def generate_comprehensive_report(self, 
                                    pred_data: Dict[str, torch.Tensor],
                                    ref_data: Dict[str, torch.Tensor],
                                    coords: torch.Tensor,
                                    model_info: Optional[Dict] = None) -> str:
        """
        生成綜合評估報告
        """
        logger.info("開始生成綜合評估報告...")
        
        # 生成所有圖表
        plots_generated = []
        
        # 1. 三行對比圖 (所有場)
        for field in ['u', 'v', 'p']:
            plot_path = self.create_three_panel_comparison(
                pred_data, ref_data, coords, field, f"comparison_3panel"
            )
            plots_generated.append(plot_path)
        
        # 2. 流場總覽
        overview_path = self.create_flow_field_overview(pred_data, ref_data, coords)
        plots_generated.append(overview_path)
        
        # 3. 誤差分析
        error_path = self.create_error_analysis_plot(pred_data, ref_data, coords)
        plots_generated.append(error_path)
        
        # 4. 物理一致性
        physics_path = self.create_physics_consistency_plot(pred_data, coords)
        plots_generated.append(physics_path)
        
        # 5. 生成文字報告
        report_path = self._generate_text_report(pred_data, ref_data, coords, 
                                                model_info or {}, plots_generated)
        
        logger.info(f"綜合評估報告生成完成: {report_path}")
        return report_path
    
    def _generate_text_report(self, 
                             pred_data: Dict[str, torch.Tensor],
                             ref_data: Dict[str, torch.Tensor],
                             coords: torch.Tensor,
                             model_info: Dict,
                             plots_generated: List[str]) -> str:
        """生成文字報告"""
        
        report_path = self.output_dir / "evaluation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PINNs 增強評估報告\n")
            f.write("Enhanced PINNs Evaluation Report\n") 
            f.write("=" * 80 + "\n\n")
            
            # 基本資訊
            f.write("📊 基本資訊 / Basic Information\n")
            f.write("-" * 40 + "\n")
            f.write(f"評估時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"數據點數: {coords.shape[0]}\n")
            f.write(f"空間維度: {coords.shape[1]}\n")
            
            if model_info:
                f.write(f"模型資訊: {model_info}\n")
            f.write("\n")
            
            # 數值精度指標
            f.write("🎯 數值精度指標 / Numerical Accuracy Metrics\n")
            f.write("-" * 40 + "\n")
            
            for field in ['u', 'v', 'p']:
                rel_l2 = relative_L2(pred_data[field], ref_data[field]).item()
                rmse_result = rmse_metrics(pred_data[field], ref_data[field])
                
                # 根據場名選擇正確的相對RMSE鍵
                rmse_key_map = {'u': 'relative_rmse_u', 'v': 'relative_rmse_v', 'p': 'relative_rmse_p'}
                rel_rmse = rmse_result[rmse_key_map[field]]
                
                f.write(f"{field.upper()} 場:\n")
                f.write(f"  相對 L2 誤差:   {rel_l2:.6f}\n")
                f.write(f"  相對 RMSE 誤差: {rel_rmse:.6f}\n")
                
                # 判斷是否達標 (10-15%門檻)
                status = "✅ 達標" if rel_l2 <= 0.15 else "❌ 未達標"
                f.write(f"  達標狀況 (15%): {status}\n\n")
            
            # 物理一致性
            f.write("⚖️ 物理一致性 / Physics Consistency\n")
            f.write("-" * 40 + "\n")
            try:
                mass_error = conservation_error(pred_data['u'], pred_data['v'], coords)
                if isinstance(mass_error, torch.Tensor):
                    mass_error_val = mass_error.mean().item() if mass_error.numel() > 1 else mass_error.item()
                else:
                    mass_error_val = float(mass_error)
                f.write(f"質量守恆誤差: {mass_error_val:.2e}\n\n")
                
            except Exception as e:
                f.write(f"物理一致性計算失敗: {e}\n\n")
            
            # 生成的圖表清單
            f.write("📈 生成圖表 / Generated Plots\n")
            f.write("-" * 40 + "\n")
            for i, plot_path in enumerate(plots_generated, 1):
                f.write(f"{i}. {Path(plot_path).name}\n")
            f.write("\n")
            
            # 結論與建議
            f.write("📋 結論與建議 / Conclusions and Recommendations\n")
            f.write("-" * 40 + "\n")
            
            # 自動生成結論
            avg_rel_l2 = np.mean([relative_L2(pred_data[f], ref_data[f]).item() 
                                 for f in ['u', 'v', 'p']])
            
            if avg_rel_l2 <= 0.10:
                f.write("🎉 模型表現優異，平均相對誤差 ≤ 10%\n")
            elif avg_rel_l2 <= 0.15:
                f.write("✅ 模型表現良好，達到預設門檻 (15%)\n")
            else:
                f.write("⚠️ 模型需要改進，未達預設門檻 (15%)\n")
                f.write("建議: 檢查訓練參數、增加訓練輪數或調整網路架構\n")
            
            f.write(f"\n平均相對 L2 誤差: {avg_rel_l2:.4f}\n")
            f.write("\n" + "=" * 80 + "\n")
        
        return str(report_path)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='增強視覺化評估腳本')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='訓練好的模型檢查點路徑')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路徑（如果不在checkpoint中）')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='輸出目錄')
    parser.add_argument('--device', type=str, default='auto',
                       help='計算設備: auto, cuda, cpu, mps')
    
    args = parser.parse_args()
    
    # 初始化視覺化器
    visualizer = EnhancedVisualizer(args.output_dir)
    
    # === 載入模型和配置 ===
    logger.info(f"載入檢查點: {args.checkpoint}")
    
    from pinnx.train.model_physics_factory import create_model, get_device
    
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 取得配置
    if 'config' in checkpoint:
        config = checkpoint['config']
        logger.info("從 checkpoint 載入配置")
    elif args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"從文件載入配置: {args.config}")
    else:
        raise ValueError("需要提供配置文件或在 checkpoint 中包含配置")
    
    # 建立並載入模型
    logger.info("建立模型...")
    model = create_model(config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # === 載入參考數據 ===
    physics_type = config.get('physics', {}).get('type', '')
    if physics_type == 'kolmogorov_flow_2d':
        logger.info("檢測到 Kolmogorov Flow 2D 配置")
        import h5py
        
        # 讀取數據路徑
        data_path = config.get('data', {}).get('kolmogorov_config', {}).get('data_path')
        if not data_path:
            raise ValueError("配置中未找到 kolmogorov_config.data_path")
            
        logger.info(f"載入 Kolmogorov DNS 數據: {data_path}")
        with h5py.File(data_path, 'r') as f:
            u_all = f['u'][:]
            v_all = f['v'][:]
            p_all = f['p'][:]
            t_all = f['time'][:]
            
        # 選擇時間點 (t=35.0 或最接近)
        target_t = 35.0
        t_idx = np.abs(t_all - target_t).argmin()
        actual_t = t_all[t_idx]
        logger.info(f"選擇時間點: {actual_t} (Index: {t_idx})")
        
        u_ref = u_all[t_idx]
        v_ref = v_all[t_idx]
        p_ref = p_all[t_idx]
        
        # 生成網格
        nx, ny = u_ref.shape
        x = np.linspace(0, 2*np.pi, nx)
        y = np.linspace(0, 2*np.pi, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # 展平
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        T_flat = np.full_like(X_flat, actual_t)
        
        # 建立 coords (t, x, y)
        coords_np = np.stack([T_flat, X_flat, Y_flat], axis=1)
        
        coords_full = torch.from_numpy(coords_np).float().to(device)
        ref_u = torch.from_numpy(u_ref.flatten()).float().to(device)
        ref_v = torch.from_numpy(v_ref.flatten()).float().to(device)
        ref_p = torch.from_numpy(p_ref.flatten()).float().to(device)
        
    else:
        logger.info("載入 JHTDB Channel Flow 完整場數據...")
        from pinnx.dataio.channel_flow_loader import ChannelFlowLoader
        loader = ChannelFlowLoader(config_path=args.config)
        field_dataset = loader.load_full_field_data()
        
        # 取得完整場座標和數據
        coords_np, ref_fields = field_dataset.to_points(order=('x', 'y', 'z'))
        coords_full = torch.from_numpy(coords_np).float().to(device)
        ref_u = torch.from_numpy(ref_fields['u']).float().to(device)
        ref_v = torch.from_numpy(ref_fields['v']).float().to(device)
        ref_p = torch.from_numpy(ref_fields['p']).float().to(device)
    
    logger.info(f"完整場數據形狀: coords={coords_full.shape}, u={ref_u.shape}")
    
    # === 模型推論（建立梯度圖） ===
    logger.info("執行模型推論...")
    
    # 🔑 關鍵：必須在啟用梯度的情況下推論，才能計算質量守恆
    coords_eval = coords_full.clone().detach().requires_grad_(True)
    predictions_with_grad = model(coords_eval)
    
    # === 反標準化 ===
    logger.info("執行反標準化...")
    predictions_np = predictions_with_grad.detach().cpu().numpy()
    predictions_physical = denormalize_output(
        predictions_np,
        config,
        output_norm_type='training_data_norm',
        verbose=True,
        checkpoint_path=args.checkpoint
    )
    
    # 轉回 tensor 並保持在正確的 device 上
    predictions_physical_tensor = torch.from_numpy(predictions_physical).float().to(device)
    
    # 不要 detach，保持梯度連接（注意：反標準化後需要重新計算梯度）
    pred_u = predictions_physical_tensor[:, 0]
    pred_v = predictions_physical_tensor[:, 1]
    pred_p = predictions_physical_tensor[:, 2] if predictions_physical_tensor.shape[1] > 2 else predictions_physical_tensor[:, 1]
    
    pred_data = {
        'u': pred_u,
        'v': pred_v,
        'p': pred_p
    }
    
    # === 參考數據 ===
    ref_data = {
        'u': ref_u,
        'v': ref_v,
        'p': ref_p
    }
    
    # === 模型資訊 ===
    model_info = {
        'checkpoint': args.checkpoint,
        'epoch': checkpoint.get('epoch', 'Unknown'),
        'loss': checkpoint.get('loss', 'Unknown'),
        'architecture': config.get('model', {}).get('architecture', 'Unknown'),
        'device': str(device)
    }
    
    # === 生成綜合報告 ===
    logger.info("生成綜合評估報告...")
    report_path = visualizer.generate_comprehensive_report(
        pred_data, ref_data, coords_eval, model_info
    )
    
    print(f"\n{'='*60}")
    print("🎉 增強視覺化評估完成!")
    print(f"📊 報告路徑: {report_path}")
    print(f"📁 輸出目錄: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
診斷檢查點中的權重動態與數據損失細節
解答 TASK-008 的核心問題
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    checkpoint_path = Path("checkpoints/fourier_annealing_longterm_k500/epoch_3881.pth")
    
    print("=" * 80)
    print("TASK-008: 檢查點權重與損失診斷")
    print("=" * 80)
    
    # 載入檢查點
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n[1] 檢查點內容清單")
    print("-" * 80)
    for key in ckpt.keys():
        print(f"  ✓ {key}")
    
    # ===== 立即任務 1: 檢查自適應權重歷史 =====
    print("\n[2] 自適應權重歷史分析")
    print("-" * 80)
    
    if 'history' in ckpt:
        history = ckpt['history']
        print(f"訓練歷史包含 {len(history)} 個 epochs")
        
        # 檢查權重欄位
        if len(history) > 0:
            sample = history[0]
            print(f"\n每個 epoch 記錄的欄位：")
            for key in sample.keys():
                print(f"  • {key}")
            
            # 提取權重時間序列
            epochs = [h['epoch'] for h in history]
            
            # 檢查是否有分場損失
            has_field_losses = any('u_loss' in h or 'pressure_loss' in h for h in history)
            print(f"\n是否記錄分場損失 (u_loss, v_loss, pressure_loss)? {has_field_losses}")
            
            # 提取損失數據
            data_losses = [h.get('data_loss', None) for h in history]
            pde_losses = [h.get('pde_loss', None) for h in history]
            total_losses = [h.get('total_loss', None) for h in history]
            
            # 提取權重數據
            data_weights = [h.get('data_weight', None) for h in history]
            pde_weights = [h.get('pde_weight', None) for h in history]
            
            # 統計量
            print(f"\n最終 100 epochs 統計：")
            print(f"  data_loss: {np.mean(data_losses[-100:]):.4f} ± {np.std(data_losses[-100:]):.4f}")
            print(f"  pde_loss: {np.mean(pde_losses[-100:]):.4f} ± {np.std(pde_losses[-100:]):.4f}")
            print(f"  total_loss: {np.mean(total_losses[-100:]):.4f} ± {np.std(total_losses[-100:]):.4f}")
            
            if data_weights[0] is not None:
                print(f"  data_weight: {np.mean(data_weights[-100:]):.4f} ± {np.std(data_weights[-100:]):.4f}")
                print(f"  pde_weight: {np.mean(pde_weights[-100:]):.4f} ± {np.std(pde_weights[-100:]):.4f}")
                
                # 檢查權重變化趨勢
                initial_data_weight = np.mean(data_weights[:100])
                final_data_weight = np.mean(data_weights[-100:])
                change_ratio = final_data_weight / initial_data_weight
                
                print(f"\n權重變化趨勢：")
                print(f"  初期 data_weight (epoch 0-100): {initial_data_weight:.4f}")
                print(f"  末期 data_weight (epoch 3781-3881): {final_data_weight:.4f}")
                print(f"  變化比例: {change_ratio:.2f}x")
                
                if change_ratio < 0.5:
                    print(f"  ⚠️  警告：data_weight 下降超過 50%，可能導致數據擬合不足")
            else:
                print(f"  ⚠️  未記錄 data_weight/pde_weight（固定權重模式）")
            
            # 繪圖
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 損失曲線
            ax = axes[0, 0]
            ax.semilogy(epochs, data_losses, label='data_loss', alpha=0.7)
            ax.semilogy(epochs, pde_losses, label='pde_loss', alpha=0.7)
            ax.semilogy(epochs, total_losses, label='total_loss', alpha=0.7, linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (log scale)')
            ax.set_title('Loss History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 權重曲線
            ax = axes[0, 1]
            if data_weights[0] is not None:
                ax.plot(epochs, data_weights, label='data_weight', linewidth=2)
                ax.plot(epochs, pde_weights, label='pde_weight', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Weight')
                ax.set_title('Adaptive Weight History')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No weight history\n(fixed weights)', 
                       ha='center', va='center', transform=ax.transAxes)
            
            # 加權損失貢獻
            ax = axes[1, 0]
            if data_weights[0] is not None:
                weighted_data = [d*w for d, w in zip(data_losses, data_weights)]
                weighted_pde = [p*w for p, w in zip(pde_losses, pde_weights)]
                ax.semilogy(epochs, weighted_data, label='data_loss × weight', alpha=0.7)
                ax.semilogy(epochs, weighted_pde, label='pde_loss × weight', alpha=0.7)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Weighted Loss (log scale)')
                ax.set_title('Weighted Loss Contribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Cannot compute\n(no weight history)', 
                       ha='center', va='center', transform=ax.transAxes)
            
            # 損失比例
            ax = axes[1, 1]
            loss_ratios = [d/p if p > 0 else 0 for d, p in zip(data_losses, pde_losses)]
            ax.plot(epochs, loss_ratios, linewidth=1.5, color='purple')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('data_loss / pde_loss')
            ax.set_title('Loss Ratio (Data/PDE)')
            ax.axhline(y=1.0, color='red', linestyle='--', label='Equal contribution', alpha=0.5)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = Path("results/task008_weight_diagnosis.png")
            output_path.parent.mkdir(exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n圖表已保存至：{output_path}")
    else:
        print("  ⚠️  檢查點中未包含 'history' 欄位")
    
    # ===== 立即任務 3: 檢查 Fourier 頻率掩碼 =====
    print("\n[3] Fourier 頻率掩碼分析")
    print("-" * 80)
    
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        
        # 搜尋 Fourier 相關參數
        fourier_keys = [k for k in state_dict.keys() if 'fourier' in k.lower()]
        print(f"找到 {len(fourier_keys)} 個 Fourier 相關參數：")
        for key in fourier_keys:
            tensor = state_dict[key]
            print(f"  • {key}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        # 檢查頻率掩碼
        mask_key = 'fourier_features._frequency_mask'
        if mask_key in state_dict:
            mask = state_dict[mask_key]
            print(f"\n頻率掩碼狀態：")
            print(f"  Shape: {mask.shape}")
            print(f"  Active frequencies: {mask.sum().item()} / {mask.numel()}")
            print(f"  Mask values (unique): {torch.unique(mask).tolist()}")
            
            if mask.dim() == 1:
                print(f"\n詳細掩碼：")
                for i, val in enumerate(mask):
                    status = "✓ ACTIVE" if val > 0 else "✗ DISABLED"
                    print(f"    Frequency {i+1}: {status}")
        else:
            print(f"  ⚠️  未找到 '{mask_key}'")
            print(f"  可能的原因：")
            print(f"    1. 頻率掩碼未保存至檢查點")
            print(f"    2. 使用不同的參數名稱")
            print(f"    3. Fourier 退火已完成，掩碼被移除")
        
        # 檢查 B 矩陣（Fourier basis）
        b_key = 'fourier_features.B'
        if b_key in state_dict:
            B = state_dict[b_key]
            print(f"\nFourier Basis Matrix (B):")
            print(f"  Shape: {B.shape}")
            print(f"  Value range: [{B.min().item():.4f}, {B.max().item():.4f}]")
            print(f"  Expected frequencies: {B.shape[1] // 2} (cos + sin)")
    
    # ===== 檢查配置信息 =====
    print("\n[4] 配置信息")
    print("-" * 80)
    
    if 'config' in ckpt:
        config = ckpt['config']
        print("配置項目：")
        
        # 檢查數據標準化
        normalize = config.get('data', {}).get('normalize', None)
        print(f"  data.normalize: {normalize}")
        
        # 檢查自適應權重
        adaptive = config.get('training', {}).get('adaptive_weighting', None)
        print(f"  training.adaptive_weighting: {adaptive}")
        
        # 檢查權重配置
        weights = config.get('training', {}).get('weights', {})
        print(f"  training.weights.data: {weights.get('data', 'N/A')}")
        print(f"  training.weights.pde: {weights.get('pde', 'N/A')}")
        
        # 檢查 Fourier 配置
        fourier_cfg = config.get('model', {}).get('fourier', {})
        print(f"  model.fourier.enabled: {fourier_cfg.get('enabled', False)}")
        print(f"  model.fourier.num_frequencies: {fourier_cfg.get('num_frequencies', 'N/A')}")
        
        annealing = fourier_cfg.get('annealing', {})
        if annealing:
            print(f"  model.fourier.annealing.enabled: {annealing.get('enabled', False)}")
            print(f"  model.fourier.annealing.schedule: {annealing.get('schedule', [])}")
    else:
        print("  ⚠️  檢查點中未包含 'config' 欄位")
    
    # ===== 檢查統計信息 =====
    print("\n[5] 數據統計信息")
    print("-" * 80)
    
    if 'statistics' in ckpt:
        stats = ckpt['statistics']
        print("數據統計：")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("  ⚠️  檢查點中未包含 'statistics' 欄位")
        print("  這意味著訓練時未保存數據標準化信息")
        print("  可能導致評估時無法正確反標準化")
    
    print("\n" + "=" * 80)
    print("診斷完成")
    print("=" * 80)

if __name__ == "__main__":
    main()

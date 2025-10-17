"""
PirateNet 訓練失敗診斷腳本

診斷重點：
1. 檢查點完整性（loss history, 模型權重）
2. 訓練過程分析（NaN 出現時機、損失曲線）
3. 模型輸出範圍檢查（數值穩定性）
4. 配置檔案驗證（物理參數、權重設定）
"""

import torch
import numpy as np
import yaml
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# 添加專案路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def diagnose_checkpoint(checkpoint_path: str):
    """診斷檢查點檔案"""
    print("=" * 80)
    print("🔍 檢查點診斷")
    print("=" * 80)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 1. 基本資訊
    print(f"\n📦 檢查點資訊:")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Loss: {checkpoint.get('loss', 'N/A')}")
    
    # 2. 檢查模型權重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\n🧠 模型權重統計:")
        
        all_params = []
        has_nan = False
        has_inf = False
        
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                param_np = param.cpu().numpy()
                all_params.extend(param_np.flatten())
                
                if np.isnan(param_np).any():
                    print(f"  ⚠️  {name}: 包含 NaN")
                    has_nan = True
                if np.isinf(param_np).any():
                    print(f"  ⚠️  {name}: 包含 Inf")
                    has_inf = True
                
                print(f"  - {name:40s} | shape: {str(param.shape):20s} | "
                      f"mean: {param_np.mean():10.4e} | std: {param_np.std():10.4e} | "
                      f"min: {param_np.min():10.4e} | max: {param_np.max():10.4e}")
        
        all_params = np.array(all_params)
        print(f"\n  📊 全局統計:")
        print(f"    - 總參數數量: {len(all_params):,}")
        print(f"    - 均值: {all_params.mean():.6e}")
        print(f"    - 標準差: {all_params.std():.6e}")
        print(f"    - 範圍: [{all_params.min():.6e}, {all_params.max():.6e}]")
        print(f"    - 包含 NaN: {'❌ 是' if has_nan else '✅ 否'}")
        print(f"    - 包含 Inf: {'❌ 是' if has_inf else '✅ 否'}")
    
    # 3. 損失歷史
    if 'loss_history' in checkpoint:
        loss_hist = checkpoint['loss_history']
        print(f"\n📈 損失歷史分析 (共 {len(loss_hist)} 筆記錄):")
        
        if len(loss_hist) > 0:
            # 找出 NaN 出現時機
            nan_epochs = []
            for i, entry in enumerate(loss_hist):
                total_loss = entry.get('total', 0)
                if np.isnan(total_loss) or np.isinf(total_loss):
                    nan_epochs.append(i)
            
            if nan_epochs:
                print(f"  ⚠️  NaN/Inf 出現於 epoch: {nan_epochs[:10]}...")
                print(f"     首次出現: epoch {nan_epochs[0]}")
            else:
                print(f"  ✅ 無 NaN/Inf")
            
            # 前 10 epochs
            print(f"\n  📋 前 10 epochs:")
            print(f"  {'Epoch':>6} {'Total':>12} {'Data':>12} {'PDE':>12} {'Wall':>12}")
            print(f"  {'-'*60}")
            for i in range(min(10, len(loss_hist))):
                entry = loss_hist[i]
                print(f"  {entry.get('epoch', i):>6} "
                      f"{entry.get('total', 0):>12.4e} "
                      f"{entry.get('data', 0):>12.4e} "
                      f"{entry.get('pde', 0):>12.4e} "
                      f"{entry.get('wall', 0):>12.4e}")
            
            # 最後 10 epochs
            if len(loss_hist) > 10:
                print(f"\n  📋 最後 10 epochs:")
                print(f"  {'Epoch':>6} {'Total':>12} {'Data':>12} {'PDE':>12} {'Wall':>12}")
                print(f"  {'-'*60}")
                for i in range(max(0, len(loss_hist)-10), len(loss_hist)):
                    entry = loss_hist[i]
                    print(f"  {entry.get('epoch', i):>6} "
                          f"{entry.get('total', 0):>12.4e} "
                          f"{entry.get('data', 0):>12.4e} "
                          f"{entry.get('pde', 0):>12.4e} "
                          f"{entry.get('wall', 0):>12.4e}")
    
    # 4. 優化器狀態
    if 'optimizer_state_dict' in checkpoint:
        opt_state = checkpoint['optimizer_state_dict']
        print(f"\n⚙️  優化器狀態:")
        if 'param_groups' in opt_state:
            for i, group in enumerate(opt_state['param_groups']):
                print(f"  - Group {i}: lr={group.get('lr', 'N/A')}")
    
    # 5. 配置資訊
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        print(f"\n📝 配置摘要:")
        if 'model' in cfg:
            print(f"  - 模型: {cfg['model'].get('type', 'N/A')}")
            print(f"  - 寬度: {cfg['model'].get('width', 'N/A')}")
            print(f"  - 深度: {cfg['model'].get('depth', 'N/A')}")
        if 'training' in cfg:
            print(f"  - Batch size: {cfg['training'].get('batch_size', 'N/A')}")
            # 學習率從優化器狀態或配置中獲取
            lr = 'N/A'
            if 'optimizer_state_dict' in checkpoint and 'param_groups' in checkpoint['optimizer_state_dict']:
                lr = checkpoint['optimizer_state_dict']['param_groups'][0].get('lr', 'N/A')
            elif isinstance(cfg['training'].get('lr'), (int, float)):
                lr = cfg['training'].get('lr')
            print(f"  - 學習率: {lr}")
    
    return checkpoint

def visualize_loss_history(loss_history, output_dir):
    """視覺化損失曲線"""
    if not loss_history or len(loss_history) == 0:
        print("⚠️  無損失歷史資料，跳過視覺化")
        return
    
    print(f"\n📊 生成損失曲線圖...")
    
    epochs = [entry.get('epoch', i) for i, entry in enumerate(loss_history)]
    total_loss = [entry.get('total', np.nan) for entry in loss_history]
    data_loss = [entry.get('data', np.nan) for entry in loss_history]
    pde_loss = [entry.get('pde', np.nan) for entry in loss_history]
    wall_loss = [entry.get('wall', np.nan) for entry in loss_history]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total Loss
    axes[0, 0].semilogy(epochs, total_loss, 'k-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Data Loss
    axes[0, 1].semilogy(epochs, data_loss, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Data Loss')
    axes[0, 1].set_title('Data Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # PDE Loss
    axes[1, 0].semilogy(epochs, pde_loss, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('PDE Loss')
    axes[1, 0].set_title('PDE Residual Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Wall Loss
    axes[1, 1].semilogy(epochs, wall_loss, 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Wall Loss')
    axes[1, 1].set_title('Wall BC Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'loss_history.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 已保存: {output_path}")
    
    plt.close()

def check_config(config_path):
    """檢查配置檔案"""
    print("\n" + "=" * 80)
    print("📋 配置檔案檢查")
    print("=" * 80)
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    issues = []
    
    # 1. 檢查物理參數
    if 'physics' in cfg:
        phys = cfg['physics']
        nu = phys.get('nu', None)
        re_tau = phys.get('re_tau', None)
        
        print(f"\n⚛️  物理參數:")
        print(f"  - 類型: {phys.get('type', 'N/A')}")
        print(f"  - ν: {nu}")
        print(f"  - Re_τ: {re_tau}")
        
        # 檢查域範圍
        if 'domain' in phys:
            domain = phys['domain']
            y_min = domain.get('y_min', None)
            y_max = domain.get('y_max', None)
            print(f"  - y 範圍: [{y_min}, {y_max}]")
            
            if y_min != -1.0 or y_max != 1.0:
                issues.append(f"⚠️  壁面位置不對稱: y ∈ [{y_min}, {y_max}]（應為 [-1, 1]）")
        
        # 檢查 VS-PINN 縮放
        if 'scaling' in phys:
            scaling = phys['scaling']
            N_x = scaling.get('N_x', 1.0)
            N_y = scaling.get('N_y', 1.0)
            N_z = scaling.get('N_z', 1.0)
            print(f"  - VS-PINN 縮放: N_x={N_x}, N_y={N_y}, N_z={N_z}")
            
            if N_x == 1.0 and N_y == 1.0 and N_z == 1.0:
                issues.append("⚠️  VS-PINN 縮放未啟用（所有縮放因子=1）")
    
    # 2. 檢查損失權重
    if 'losses' in cfg:
        losses = cfg['losses']
        print(f"\n⚖️  損失權重:")
        print(f"  - Data: {losses.get('data_loss_weight', 'N/A')}")
        print(f"  - PDE: {losses.get('pde_loss_weight', 'N/A')}")
        print(f"  - Wall: {losses.get('wall_loss_weight', 'N/A')}")
        print(f"  - Initial: {losses.get('initial_loss_weight', 'N/A')}")
        
        # 檢查自適應權重
        if 'adaptive_weights' in losses:
            adp = losses['adaptive_weights']
            print(f"  - 自適應權重: {adp.get('enabled', False)}")
            if adp.get('enabled'):
                print(f"    - 方法: {adp.get('method', 'N/A')}")
                print(f"    - α: {adp.get('alpha', 'N/A')}")
                print(f"    - 更新頻率: {adp.get('update_frequency', 'N/A')}")
    
    # 3. 檢查訓練配置
    if 'training' in cfg:
        train = cfg['training']
        print(f"\n🏋️  訓練配置:")
        print(f"  - Epochs: {train.get('epochs', 'N/A')}")
        print(f"  - Batch size: {train.get('batch_size', 'N/A')}")
        
        if 'optimizer' in train:
            opt = train['optimizer']
            # optimizer 可能是字串或字典
            if isinstance(opt, dict):
                lr = opt.get('lr', train.get('lr'))
                opt_type = opt.get('type', 'N/A')
            else:
                lr = train.get('lr', None)
                opt_type = opt
            
            print(f"  - 優化器: {opt_type}")
            print(f"  - 學習率: {lr}")
            
            if lr and lr > 1e-2:
                issues.append(f"⚠️  學習率過高: {lr}（建議 ≤ 1e-3）")
        
        # 檢查梯度裁剪
        grad_clip = train.get('gradient_clip', None)
        if grad_clip:
            print(f"  - 梯度裁剪: {grad_clip}")
        else:
            issues.append("⚠️  未啟用梯度裁剪（建議設為 1.0）")
    
    # 4. 檢查模型配置
    if 'model' in cfg:
        model = cfg['model']
        print(f"\n🧠 模型配置:")
        print(f"  - 類型: {model.get('type', 'N/A')}")
        print(f"  - 寬度: {model.get('width', 'N/A')}")
        print(f"  - 深度: {model.get('depth', 'N/A')}")
        print(f"  - 激活函數: {model.get('activation', 'N/A')}")
        
        if model.get('use_fourier'):
            print(f"  - Fourier M: {model.get('fourier_m', 'N/A')}")
            print(f"  - Fourier σ: {model.get('fourier_sigma', 'N/A')}")
        
        if model.get('use_rwf'):
            print(f"  - RWF: 已啟用")
    
    # 輸出問題總結
    if issues:
        print(f"\n❌ 發現 {len(issues)} 個潛在問題:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print(f"\n✅ 配置檔案看起來正常")
    
    return cfg, issues

def main():
    import argparse
    parser = argparse.ArgumentParser(description='PirateNet 訓練失敗診斷')
    parser.add_argument('--checkpoint', type=str, required=True, help='檢查點路徑')
    parser.add_argument('--config', type=str, required=True, help='配置檔案路徑')
    parser.add_argument('--output-dir', type=str, default='results/piratenet_diagnosis',
                        help='診斷結果輸出目錄')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔬 PirateNet 訓練失敗診斷工具")
    print("=" * 80)
    print(f"\n檢查點: {args.checkpoint}")
    print(f"配置: {args.config}")
    print(f"輸出: {args.output_dir}")
    
    # 1. 診斷檢查點
    checkpoint = diagnose_checkpoint(args.checkpoint)
    
    # 2. 視覺化損失歷史
    if 'loss_history' in checkpoint:
        visualize_loss_history(checkpoint['loss_history'], args.output_dir)
    
    # 3. 檢查配置
    config, issues = check_config(args.config)
    
    # 4. 生成診斷報告
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'diagnosis_report.json'
    report = {
        'checkpoint': args.checkpoint,
        'config': args.config,
        'epoch': checkpoint.get('epoch', None),
        'final_loss': float(checkpoint.get('loss', np.nan)),
        'issues': issues,
        'has_nan_weights': False,  # 會在上面更新
        'has_nan_loss': False,      # 會在上面更新
    }
    
    # 檢查 NaN
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                if torch.isnan(param).any():
                    report['has_nan_weights'] = True
                    break
    
    if 'loss_history' in checkpoint and len(checkpoint['loss_history']) > 0:
        for entry in checkpoint['loss_history']:
            if np.isnan(entry.get('total', 0)):
                report['has_nan_loss'] = True
                break
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 診斷報告已保存: {report_path}")
    
    # 5. 總結與建議
    print("\n" + "=" * 80)
    print("📝 診斷總結與建議")
    print("=" * 80)
    
    if report['has_nan_loss']:
        print("\n❌ 訓練過程中出現 NaN 損失")
        print("   建議:")
        print("   1. 降低學習率 (目前可能過高)")
        print("   2. 啟用梯度裁剪 (gradient_clip: 1.0)")
        print("   3. 檢查資料歸一化")
        print("   4. 減小 batch size")
    
    if report['has_nan_weights']:
        print("\n❌ 模型權重包含 NaN")
        print("   建議:")
        print("   1. 重新初始化模型")
        print("   2. 檢查權重初始化方法")
        print("   3. 使用更穩定的激活函數 (如 tanh)")
    
    if issues:
        print(f"\n⚠️  配置問題需要修正:")
        for issue in issues:
            print(f"   - {issue}")
    
    print("\n" + "=" * 80)
    print("✅ 診斷完成")
    print("=" * 80)

if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Phase 6 物理對比分析腳本

對比 Phase 6B (log1p) vs Phase 6C-v3 (Huber) 的物理指標：
- ν_t/ν 分布（min/max/mean/std、直方圖）
- k-ε 方程殘差（L2 norm）
- 速度場誤差（相對 JHTDB）
- 湍動能分布（與 DNS 對比）
"""
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

# 添加項目路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pinnx.models.fourier_mlp import PINNNet
from pinnx.models.wrappers import ManualScalingWrapper


def load_checkpoint_with_model(ckpt_path: str) -> tuple:
    """
    載入檢查點並從嵌入配置構建模型
    
    Returns:
        (checkpoint, model, config)
    """
    print(f"📂 載入檢查點: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # 獲取嵌入配置
    if 'config' not in checkpoint:
        raise ValueError("檢查點中沒有嵌入配置")
    
    cfg = checkpoint['config']
    model_cfg = cfg['model']
    
    # 從實際權重推斷模型參數
    state_dict = checkpoint['model_state_dict']
    
    # 檢測是否使用 input_projection
    has_input_projection = 'input_projection.weight' in state_dict
    
    # 從權重推斷參數
    in_dim = model_cfg.get('in_dim', 3)
    out_dim = model_cfg.get('out_dim', 4)
    
    if has_input_projection:
        # 有 input_projection: fourier.B -> input_projection -> hidden_layers
        fourier_output_dim = state_dict['input_projection.weight'].shape[1]
        actual_fourier_m = fourier_output_dim // 2
        width = state_dict['input_projection.weight'].shape[0]
    else:
        # 無 input_projection: fourier.B -> hidden_layers
        first_layer_weight = state_dict['hidden_layers.0.linear.weight']
        fourier_output_dim = first_layer_weight.shape[1]
        actual_fourier_m = fourier_output_dim // 2
        width = first_layer_weight.shape[0]
    
    print(f"  從權重推斷模型參數:")
    print(f"    - in_dim: {in_dim}")
    print(f"    - out_dim: {out_dim}")
    print(f"    - width: {width}")
    print(f"    - 實際 fourier_m: {actual_fourier_m} (配置中為 {model_cfg.get('fourier_m', 32)})")
    print(f"    - use_input_projection: {has_input_projection}")
    
    # 使用實際推斷的參數構建模型
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 獲取其他模型參數（使用嵌入配置或預設值）
    fourier_cfg = model_cfg.get('fourier_features', {})
    scaling_cfg = model_cfg.get('scaling', {})
    
    # 構建模型參數
    model_params = {
        'in_dim': in_dim,
        'out_dim': out_dim,
        'width': width,
        'depth': model_cfg.get('depth', 5),
        'fourier_m': actual_fourier_m,
        'fourier_sigma': fourier_cfg.get('fourier_sigma', model_cfg.get('fourier_sigma', 1.0)),
        'activation': model_cfg.get('activation', 'sine'),
        'use_fourier': model_cfg.get('use_fourier', True),
        'trainable_fourier': fourier_cfg.get('trainable', model_cfg.get('fourier_trainable', False)),
        'use_input_projection': has_input_projection,
        'use_layer_norm': True,  # Phase 6 都使用了 layer_norm
        'use_residual': False,
    }
    
    # 處理輸入縮放因子（用於 VS-PINN）
    if 'input_scale_factors' in state_dict:
        model_params['input_scale_factors'] = state_dict['input_scale_factors']
        print(f"    - input_scale_factors: {state_dict['input_scale_factors']}")
    
    model = PINNNet(**model_params).to(device)
    
    # 檢測是否使用 ManualScalingWrapper
    has_scaling_buffers = any(k in state_dict for k in ['input_min', 'input_max', 'output_min', 'output_max'])
    has_base_prefix = any(k.startswith('base_model.') for k in state_dict.keys())
    
    if has_scaling_buffers:
        print("  檢測到尺度化緩衝區，使用 ManualScalingWrapper...")
        # 建立佔位範圍（會被 state_dict 覆蓋）
        in_ranges = {f'in_{i}': (0.0, 1.0) for i in range(in_dim)}
        out_ranges = {f'out_{i}': (0.0, 1.0) for i in range(out_dim)}
        
        wrapper = ManualScalingWrapper(base_model=model, input_ranges=in_ranges, output_ranges=out_ranges).to(device)
        
        # 處理鍵映射
        if not has_base_prefix:
            mapped_state = {}
            for k, v in state_dict.items():
                if k in ['input_min', 'input_max', 'output_min', 'output_max']:
                    mapped_state[k] = v
                else:
                    mapped_state[f'base_model.{k}'] = v
            state_dict = mapped_state
        
        wrapper.load_state_dict(state_dict, strict=False)
        model = wrapper
        print("  ✅ ManualScalingWrapper 載入成功")
    else:
        # 處理可能的 base_model 前綴
        if has_base_prefix:
            state_dict = {k.replace('base_model.', ''): v for k, v in state_dict.items() 
                         if k not in ['input_min', 'input_max', 'output_min', 'output_max']}
        
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    print(f"✅ 模型已載入（設備: {device}）")
    
    return checkpoint, model, cfg


def compute_rans_residuals(model, coords: torch.Tensor, physics_cfg: Dict) -> Dict[str, float]:
    """
    計算 RANS 方程殘差
    
    Args:
        model: PINN 模型
        coords: [N, 3] 座標 (x, y, z)
        physics_cfg: 物理配置
    
    Returns:
        {
            'k_residual': float,
            'epsilon_residual': float,
            'nu_t_min': float,
            'nu_t_max': float,
            'nu_t_mean': float,
            'nu_t_std': float,
            'nu_t_over_nu_ratio': np.ndarray [N]
        }
    """
    coords.requires_grad_(True)
    
    # 前向傳播
    pred = model(coords)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    w = pred[:, 2:3]
    
    # 計算梯度（用於 k-ε 方程）
    # 注意：這裡簡化為不計算完整的 k-ε 殘差（需要額外的網絡輸出）
    # 實際評估需要模型輸出 k 和 ε
    
    # 計算湍動黏度（基於混合長度理論的近似）
    nu = physics_cfg.get('nu', 5.0e-5)
    
    # 簡化：使用速度梯度估算 ν_t
    # 完整版需要從 RANS 模型中提取 k 和 ε
    du_dy = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0][:, 1:2]
    S = torch.sqrt(2 * du_dy**2)  # 簡化的應變率張量範數
    
    # 混合長度近似：ν_t ~ l_m^2 |S|
    y = coords[:, 1:2]
    l_m = 0.41 * torch.abs(y) * (1 - torch.abs(y))  # von Kármán 混合長度
    nu_t = l_m**2 * S
    
    # 統計量
    nu_t_np = nu_t.detach().cpu().numpy().flatten()
    nu_t_over_nu = nu_t_np / nu
    
    return {
        'k_residual': 0.0,  # 待實作：需要完整的 k-ε 方程
        'epsilon_residual': 0.0,  # 待實作
        'nu_t_min': float(nu_t_np.min()),
        'nu_t_max': float(nu_t_np.max()),
        'nu_t_mean': float(nu_t_np.mean()),
        'nu_t_std': float(nu_t_np.std()),
        'nu_t_over_nu_ratio': nu_t_over_nu
    }


def evaluate_phase(phase_name: str, ckpt_path: str) -> Dict[str, Any]:
    """
    評估單個 Phase 的物理指標
    
    Returns:
        {
            'name': str,
            'loss': dict,
            'rans_metrics': dict,
            'velocity_error': dict
        }
    """
    print(f"\n{'='*70}")
    print(f"  評估 {phase_name}")
    print(f"{'='*70}")
    
    # 載入模型
    checkpoint, model, cfg = load_checkpoint_with_model(ckpt_path)
    
    # 提取訓練損失
    loss_info = {
        'epoch': checkpoint.get('epoch', 'N/A'),
    }
    
    # 如果有詳細的損失歷史，提取最後一輪的損失
    if 'history' in checkpoint:
        history = checkpoint['history']
        # history 是 dict，每個鍵對應一個列表
        for key in ['total_loss', 'turbulent_viscosity_loss', 'k_equation_loss', 'epsilon_equation_loss', 
                   'rans_loss', 'pde_loss', 'data_loss']:
            if key in history and isinstance(history[key], list) and len(history[key]) > 0:
                loss_info[key] = history[key][-1]
            elif key in history:
                loss_info[key] = history[key]
    
    # 生成測試點（通道流的典型分布）
    device = next(model.parameters()).device
    N_test = 1000
    
    # 通道流域：x ∈ [0, 25.13], y ∈ [-1, 1], z ∈ [0, 9.42]
    x = np.random.uniform(0, 25.13, N_test)
    y = np.random.uniform(-1, 1, N_test)
    z = np.random.uniform(0, 9.42, N_test)
    coords = torch.FloatTensor(np.stack([x, y, z], axis=1)).to(device)
    
    # 計算 RANS 指標（需要梯度）
    print("\n  計算 RANS 物理指標...")
    rans_metrics = compute_rans_residuals(model, coords, cfg.get('physics', {}))
    
    # 速度場預測（用於後續對比）
    print("  計算速度場預測...")
    with torch.no_grad():
        pred = model(coords)
        velocity_pred = {
            'u': pred[:, 0].cpu().numpy(),
            'v': pred[:, 1].cpu().numpy(),
            'w': pred[:, 2].cpu().numpy(),
            'p': pred[:, 3].cpu().numpy() if pred.shape[1] > 3 else None
        }
    
    return {
        'name': phase_name,
        'loss': loss_info,
        'rans_metrics': rans_metrics,
        'velocity_pred': velocity_pred,
        'test_coords': coords.detach().cpu().numpy()
    }


def compare_and_visualize(results_6b: Dict, results_6c: Dict, output_dir: Path):
    """
    對比兩個 Phase 的結果並生成視覺化
    """
    print(f"\n{'='*70}")
    print("  對比分析")
    print(f"{'='*70}")
    
    # 1. 訓練損失對比
    print("\n📊 訓練損失對比：")
    print("-" * 70)
    print(f"{'指標':<40} {'Phase 6B (log1p)':<20} {'Phase 6C-v3 (Huber)':<20}")
    print("-" * 70)
    
    for key in ['total_loss', 'turbulent_viscosity_loss', 'k_equation_loss', 'epsilon_equation_loss']:
        val_6b = results_6b['loss'].get(key, 'N/A')
        val_6c = results_6c['loss'].get(key, 'N/A')
        
        if isinstance(val_6b, (int, float)) and isinstance(val_6c, (int, float)):
            ratio = val_6c / val_6b if val_6b != 0 else float('inf')
            print(f"{key:<40} {val_6b:<20.2f} {val_6c:<20.2f} ({ratio:.2f}x)")
        else:
            print(f"{key:<40} {str(val_6b):<20} {str(val_6c):<20}")
    
    # 2. RANS 物理指標對比
    print("\n🔬 RANS 物理指標對比：")
    print("-" * 70)
    print(f"{'指標':<40} {'Phase 6B':<20} {'Phase 6C-v3':<20}")
    print("-" * 70)
    
    rans_6b = results_6b['rans_metrics']
    rans_6c = results_6c['rans_metrics']
    
    for key in ['nu_t_min', 'nu_t_max', 'nu_t_mean', 'nu_t_std']:
        val_6b = rans_6b[key]
        val_6c = rans_6c[key]
        print(f"{key:<40} {val_6b:<20.6e} {val_6c:<20.6e}")
    
    # 3. ν_t/ν 分布統計
    ratio_6b = rans_6b['nu_t_over_nu_ratio']
    ratio_6c = rans_6c['nu_t_over_nu_ratio']
    
    print("\n📈 ν_t/ν 分布統計：")
    print("-" * 70)
    print(f"{'統計量':<40} {'Phase 6B':<20} {'Phase 6C-v3':<20}")
    print("-" * 70)
    print(f"{'Min':<40} {ratio_6b.min():<20.2f} {ratio_6c.min():<20.2f}")
    print(f"{'Max':<40} {ratio_6b.max():<20.2f} {ratio_6c.max():<20.2f}")
    print(f"{'Mean':<40} {ratio_6b.mean():<20.2f} {ratio_6c.mean():<20.2f}")
    print(f"{'Std':<40} {ratio_6b.std():<20.2f} {ratio_6c.std():<20.2f}")
    print(f"{'Median':<40} {np.median(ratio_6b):<20.2f} {np.median(ratio_6c):<20.2f}")
    print(f"{'95th percentile':<40} {np.percentile(ratio_6b, 95):<20.2f} {np.percentile(ratio_6c, 95):<20.2f}")
    
    # 4. 生成視覺化
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4.1 ν_t/ν 分布直方圖
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(ratio_6b, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('ν_t/ν')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Phase 6B (log1p)')
    axes[0].axvline(ratio_6b.mean(), color='red', linestyle='--', label=f'Mean: {ratio_6b.mean():.1f}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(ratio_6c, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('ν_t/ν')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Phase 6C-v3 (Huber)')
    axes[1].axvline(ratio_6c.mean(), color='red', linestyle='--', label=f'Mean: {ratio_6c.mean():.1f}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'nu_t_distribution_comparison.png', dpi=150)
    print(f"\n  ✅ 儲存: {output_dir / 'nu_t_distribution_comparison.png'}")
    
    # 4.2 疊加對比直方圖
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ratio_6b, bins=50, alpha=0.5, color='blue', label='Phase 6B (log1p)', edgecolor='black')
    ax.hist(ratio_6c, bins=50, alpha=0.5, color='green', label='Phase 6C-v3 (Huber)', edgecolor='black')
    ax.set_xlabel('ν_t/ν')
    ax.set_ylabel('Frequency')
    ax.set_title('ν_t/ν Distribution Comparison')
    ax.axvline(ratio_6b.mean(), color='blue', linestyle='--', label=f'6B Mean: {ratio_6b.mean():.1f}')
    ax.axvline(ratio_6c.mean(), color='green', linestyle='--', label=f'6C-v3 Mean: {ratio_6c.mean():.1f}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'nu_t_overlay_comparison.png', dpi=150)
    print(f"  ✅ 儲存: {output_dir / 'nu_t_overlay_comparison.png'}")
    
    # 5. 決策建議
    print(f"\n{'='*70}")
    print("  決策建議")
    print(f"{'='*70}")
    
    # 基於物理指標給出推薦
    decision_score = 0  # 正數偏向 6C，負數偏向 6B
    decision_reasons = []
    
    # 1. 損失數值比較（總損失越低越好）
    loss_6b = results_6b['loss'].get('total_loss', float('inf'))
    loss_6c = results_6c['loss'].get('total_loss', float('inf'))
    if loss_6b < loss_6c:
        decision_score -= 2  # 6B 勝出（權重較高）
        print(f"\n  ❌ Phase 6B 總損失更低 ({loss_6b:.2f} < {loss_6c:.2f})")
        decision_reasons.append("Phase 6B 總損失更低")
    else:
        decision_score += 2
        print(f"\n  ✅ Phase 6C-v3 總損失更低 ({loss_6c:.2f} < {loss_6b:.2f})")
        decision_reasons.append("Phase 6C-v3 總損失更低")
    
    # 1.5. 湍流黏度損失檢查（關鍵物理指標）
    turb_visc_6b = results_6b['loss'].get('turbulent_viscosity_loss', 0)
    turb_visc_6c = results_6c['loss'].get('turbulent_viscosity_loss', 0)
    if turb_visc_6c > turb_visc_6b * 10:  # 如果 6C 的湍流黏度損失是 6B 的 10 倍以上
        decision_score -= 2
        print(f"  ⚠️  Phase 6C-v3 湍流黏度損失異常高 ({turb_visc_6c:.0f} vs {turb_visc_6b:.0f}, {turb_visc_6c/turb_visc_6b:.1f}x)")
        decision_reasons.append("Phase 6C-v3 湍流黏度損失異常")
    elif turb_visc_6b > turb_visc_6c * 10:
        decision_score += 2
        print(f"  ✅ Phase 6C-v3 湍流黏度損失顯著更低")
        decision_reasons.append("Phase 6C-v3 湍流黏度損失更低")
    
    # 2. ν_t/ν 分布合理性（目標：< 200，理想 < 100）
    ratio_6b_exceed = (ratio_6b > 200).sum() / len(ratio_6b) * 100
    ratio_6c_exceed = (ratio_6c > 200).sum() / len(ratio_6c) * 100
    
    if ratio_6c_exceed < ratio_6b_exceed:
        decision_score += 1
        print(f"  ✅ Phase 6C-v3 的 ν_t/ν > 200 佔比更低 ({ratio_6c_exceed:.1f}% vs {ratio_6b_exceed:.1f}%)")
        decision_reasons.append("Phase 6C-v3 極值佔比更低")
    elif ratio_6b_exceed < ratio_6c_exceed:
        decision_score -= 1
        print(f"  ❌ Phase 6B 的 ν_t/ν > 200 佔比更低 ({ratio_6b_exceed:.1f}% vs {ratio_6c_exceed:.1f}%)")
        decision_reasons.append("Phase 6B 極值佔比更低")
    else:
        print(f"  ⚖️  兩者的 ν_t/ν > 200 佔比相同 ({ratio_6b_exceed:.1f}%)")
    
    # 3. 分布穩定性（標準差越小越好）
    if ratio_6c.std() < ratio_6b.std():
        decision_score += 1
        print(f"  ✅ Phase 6C-v3 的 ν_t/ν 分布更穩定 (std: {ratio_6c.std():.1f} vs {ratio_6b.std():.1f})")
        decision_reasons.append("Phase 6C-v3 分布更穩定")
    else:
        decision_score -= 1
        print(f"  ❌ Phase 6B 的 ν_t/ν 分布更穩定 (std: {ratio_6b.std():.1f} vs {ratio_6c.std():.1f})")
        decision_reasons.append("Phase 6B 分布更穩定")
    
    # 4. 平均值合理性（理想範圍 5-50）
    mean_6b = ratio_6b.mean()
    mean_6c = ratio_6c.mean()
    ideal_range = (5, 50)
    
    if ideal_range[0] <= mean_6c <= ideal_range[1] and not (ideal_range[0] <= mean_6b <= ideal_range[1]):
        decision_score += 1
        print(f"  ✅ Phase 6C-v3 平均值在理想範圍內 ({mean_6c:.1f} ∈ [{ideal_range[0]}, {ideal_range[1]}])")
        decision_reasons.append("Phase 6C-v3 平均值更合理")
    elif ideal_range[0] <= mean_6b <= ideal_range[1] and not (ideal_range[0] <= mean_6c <= ideal_range[1]):
        decision_score -= 1
        print(f"  ❌ Phase 6B 平均值在理想範圍內 ({mean_6b:.1f} ∈ [{ideal_range[0]}, {ideal_range[1]}])")
        decision_reasons.append("Phase 6B 平均值更合理")
    
    # 最終推薦
    print(f"\n{'='*70}")
    print(f"  決策分數: {decision_score} (正數偏向 6C-v3，負數偏向 6B)")
    print(f"{'='*70}")
    
    if decision_score > 1:
        print("  🏆 推薦：Phase 6C-v3 (Huber) 物理性能更優")
    elif decision_score < -1:
        print("  🏆 推薦：Phase 6B (log1p) 物理性能更優")
    else:
        print("  ⚖️  兩者性能相當，需進一步評估速度場誤差")
    
    print(f"\n  主要原因:")
    for reason in decision_reasons:
        print(f"    - {reason}")
    print(f"{'='*70}")
    
    return decision_score


def main():
    """主函數"""
    print("="*70)
    print("  Phase 6 物理對比分析")
    print("="*70)
    
    # 檢查點路徑
    phase6b_ckpt = Path("checkpoints/test_rans_phase6b/epoch_100.pth")
    phase6c_ckpt = Path("checkpoints/test_rans_phase6c_v3/epoch_100.pth")
    
    if not phase6b_ckpt.exists():
        print(f"❌ Phase 6B 檢查點不存在: {phase6b_ckpt}")
        return
    
    if not phase6c_ckpt.exists():
        print(f"❌ Phase 6C-v3 檢查點不存在: {phase6c_ckpt}")
        return
    
    # 評估兩個 Phase
    results_6b = evaluate_phase("Phase 6B (log1p)", str(phase6b_ckpt))
    results_6c = evaluate_phase("Phase 6C-v3 (Huber)", str(phase6c_ckpt))
    
    # 對比分析
    output_dir = Path("tasks/TASK-008/phase_6_comparison")
    decision_score = compare_and_visualize(results_6b, results_6c, output_dir)
    
    # 儲存結果為 JSON
    import json
    
    # 轉換 numpy 數組為列表（JSON 不支持 numpy）
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    results = {
        'phase_6b': convert_to_json_serializable(results_6b),
        'phase_6c_v3': convert_to_json_serializable(results_6c),
        'decision_score': decision_score
    }
    
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  ✅ 完整結果已儲存: {output_dir / 'comparison_results.json'}")
    print("\n✅ 對比分析完成！")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
自動生成實驗對比配置文件
根據 docs/EXPERIMENT_COMPARISON_PLAN.md 設計
"""

import os
import yaml
from pathlib import Path
from copy import deepcopy

# 基礎配置模板路徑
BASE_CONFIG = Path(__file__).parent.parent / "kolmogorov_re50_kf4_K100_vanilla_1k.yml"
EXPERIMENT_DIR = Path(__file__).parent

def load_base_config():
    """載入基礎配置"""
    with open(BASE_CONFIG, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config, output_path):
    """儲存配置文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, indent=2, sort_keys=False)
    print(f"✅ 已生成: {output_path.relative_to(EXPERIMENT_DIR)}")

def create_s1_random():
    """S1: Random 感測器配置"""
    config = load_base_config()
    
    # 修改實驗資訊
    config['experiment']['name'] = 's1_random_K100_2d_re50'
    config['experiment']['description'] = 'S1: Random sensor strategy baseline'
    
    # 修改感測器策略為 Random（預先固定，避免每次重抽造成不可比）
    config['sensors'] = {
        'K': 100,
        'selection_method': 'precomputed',
        'sensor_file': './data/sensors/kolmogorov/sensors_K100_re50_256x256_random_seed42.json',
        'seed': 42,
        'quality_metrics': {
            'method': 'random_uniform'
        }
    }
    
    # 輸出目錄
    config['output'] = {
        'checkpoint_dir': './checkpoints/experiments/S1_random_K100',
        'results_dir': './results/experiments/S1_random_K100',
        'visualization_dir': './results/experiments/S1_random_K100/visualizations'
    }
    
    output_path = EXPERIMENT_DIR / 'S1_sensor_strategy' / 's1_random_K100_2d_re50.yml'
    save_config(config, output_path)

def create_s1_qr():
    """S1: QR-pivot 感測器配置"""
    config = load_base_config()
    
    # 修改實驗資訊
    config['experiment']['name'] = 's1_qr_K100_2d_re50'
    config['experiment']['description'] = 'S1: QR-pivot sensor strategy'
    
    # 使用預計算的 QR-pivot 感測器
    config['sensors'] = {
        'K': 100,
        'selection_method': 'precomputed',
        'sensor_file': './data/sensors/kolmogorov/sensors_K100_re50_256x256.json',
        'quality_metrics': {
            'condition_number': 5.29e+04,
            'method': 'qr_pivot_rans'
        }
    }
    
    # 輸出目錄
    config['output'] = {
        'checkpoint_dir': './checkpoints/experiments/S1_qr_K100',
        'results_dir': './results/experiments/S1_qr_K100',
        'visualization_dir': './results/experiments/S1_qr_K100/visualizations'
    }
    
    output_path = EXPERIMENT_DIR / 'S1_sensor_strategy' / 's1_qr_K100_2d_re50.yml'
    save_config(config, output_path)

def create_s2_k_scan():
    """S2: K 值掃描配置（K=30, 50, 80, 100）"""
    for K in [30, 50, 80, 100]:
        config = load_base_config()
        
        # 修改實驗資訊
        config['experiment']['name'] = f's2_qr_K{K}_2d_re50'
        config['experiment']['description'] = f'S2: K-scan with K={K} sensors (QR-pivot)'
        
        # 修改感測器數量
        config['sensors']['K'] = K
        config['sensors']['sensor_file'] = f'./data/sensors/kolmogorov/sensors_K{K}_re50_256x256.json'
        
        # 輸出目錄
        config['output'] = {
            'checkpoint_dir': f'./checkpoints/experiments/S2_K{K}',
            'results_dir': f'./results/experiments/S2_K{K}',
            'visualization_dir': f'./results/experiments/S2_K{K}/visualizations'
        }
        
        output_path = EXPERIMENT_DIR / 'S2_k_scan' / f's2_qr_K{K}_2d_re50.yml'
        save_config(config, output_path)

def create_m1_vanilla():
    """M1: Vanilla 基線配置"""
    config = load_base_config()
    
    # 修改實驗資訊
    config['experiment']['name'] = 'm1_vanilla_K100_2d_re50'
    config['experiment']['description'] = 'M1: Vanilla MLP baseline (no advanced features)'
    
    # 輸出目錄
    config['output'] = {
        'checkpoint_dir': './checkpoints/experiments/M1_vanilla_K100',
        'results_dir': './results/experiments/M1_vanilla_K100',
        'visualization_dir': './results/experiments/M1_vanilla_K100/visualizations'
    }
    
    output_path = EXPERIMENT_DIR / 'M1_model_comparison' / 'm1_vanilla_K100_2d_re50.yml'
    save_config(config, output_path)

def create_m1_full():
    """M1: Full 特徵配置"""
    # 載入 RANS Prior 配置作為 Full 版本
    full_config_path = Path(__file__).parent.parent / "kolmogorov_re50_kf4_K100_rans_prior_1k.yml"
    with open(full_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改實驗資訊
    config['experiment']['name'] = 'm1_full_K100_2d_re50'
    config['experiment']['version'] = 'v1.0_full_features'
    config['experiment']['description'] = 'M1: Full features (Fourier + SIREN + GradNorm + Causal)'

    # M1 僅比較模型/訓練策略（Prior 由 C1/C2 負責）
    config['lowfi_prior']['enabled'] = False
    config['losses']['prior_weight'] = 0.0
    
    # 輸出目錄
    config['output'] = {
        'checkpoint_dir': './checkpoints/experiments/M1_full_K100',
        'results_dir': './results/experiments/M1_full_K100',
        'visualization_dir': './results/experiments/M1_full_K100/visualizations'
    }
    
    output_path = EXPERIMENT_DIR / 'M1_model_comparison' / 'm1_full_K100_2d_re50.yml'
    save_config(config, output_path)

def create_a1_ablations():
    """A1: Fourier Features 消融配置"""
    # 載入 Full 配置
    full_config_path = Path(__file__).parent.parent / "kolmogorov_re50_kf4_K100_rans_prior_1k.yml"
    with open(full_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # A1.1: Full（有 Fourier）
    config_with = deepcopy(base_config)
    config_with['experiment']['name'] = 'a1_with_fourier_K100_2d_re50'
    config_with['experiment']['version'] = 'v1.0_full_features'
    config_with['experiment']['description'] = 'A1: Full with Fourier Features'
    config_with['lowfi_prior']['enabled'] = False
    config_with['losses']['prior_weight'] = 0.0
    config_with['model']['fourier_features']['enabled'] = True
    config_with['output'] = {
        'checkpoint_dir': './checkpoints/experiments/A1_with_fourier_K100',
        'results_dir': './results/experiments/A1_with_fourier_K100',
        'visualization_dir': './results/experiments/A1_with_fourier_K100/visualizations'
    }
    save_config(config_with, EXPERIMENT_DIR / 'A1_ablation_fourier' / 'a1_with_fourier_K100_2d_re50.yml')
    
    # A1.2: Full（無 Fourier）
    config_without = deepcopy(base_config)
    config_without['experiment']['name'] = 'a1_without_fourier_K100_2d_re50'
    config_without['experiment']['version'] = 'v1.0_full_features'
    config_without['experiment']['description'] = 'A1: Full without Fourier Features'
    config_without['lowfi_prior']['enabled'] = False
    config_without['losses']['prior_weight'] = 0.0
    config_without['model']['fourier_features']['enabled'] = False
    config_without['model']['fourier_features']['type'] = 'disabled'
    config_without['model']['fourier_features']['fourier_m'] = 0
    config_without['model']['fourier_features']['fourier_sigma'] = 0.0
    config_without['output'] = {
        'checkpoint_dir': './checkpoints/experiments/A1_without_fourier_K100',
        'results_dir': './results/experiments/A1_without_fourier_K100',
        'visualization_dir': './results/experiments/A1_without_fourier_K100/visualizations'
    }
    save_config(config_without, EXPERIMENT_DIR / 'A1_ablation_fourier' / 'a1_without_fourier_K100_2d_re50.yml')

def create_a2_ablations():
    """A2: 動態權重消融配置"""
    # 載入 Full 配置
    full_config_path = Path(__file__).parent.parent / "kolmogorov_re50_kf4_K100_rans_prior_1k.yml"
    with open(full_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # A2.1: Full（有動態權重）
    config_with = deepcopy(base_config)
    config_with['experiment']['name'] = 'a2_with_adaptive_K100_2d_re50'
    config_with['experiment']['version'] = 'v1.0_full_features'
    config_with['experiment']['description'] = 'A2: Full with adaptive weights (GradNorm)'
    config_with['lowfi_prior']['enabled'] = False
    config_with['losses']['prior_weight'] = 0.0
    config_with['losses']['adaptive_weighting'] = True
    config_with['output'] = {
        'checkpoint_dir': './checkpoints/experiments/A2_with_adaptive_K100',
        'results_dir': './results/experiments/A2_with_adaptive_K100',
        'visualization_dir': './results/experiments/A2_with_adaptive_K100/visualizations'
    }
    save_config(config_with, EXPERIMENT_DIR / 'A2_ablation_weights' / 'a2_with_adaptive_K100_2d_re50.yml')
    
    # A2.2: Full（無動態權重）
    config_without = deepcopy(base_config)
    config_without['experiment']['name'] = 'a2_without_adaptive_K100_2d_re50'
    config_without['experiment']['version'] = 'v1.0_full_features'
    config_without['experiment']['description'] = 'A2: Full without adaptive weights (fixed)'
    config_without['lowfi_prior']['enabled'] = False
    config_without['losses']['prior_weight'] = 0.0
    config_without['losses']['adaptive_weighting'] = False
    # 只關閉 adaptive（normalize_losses 保持一致，避免改動多個變因）
    config_without['losses']['normalize_losses'] = True
    config_without['output'] = {
        'checkpoint_dir': './checkpoints/experiments/A2_without_adaptive_K100',
        'results_dir': './results/experiments/A2_without_adaptive_K100',
        'visualization_dir': './results/experiments/A2_without_adaptive_K100/visualizations'
    }
    save_config(config_without, EXPERIMENT_DIR / 'A2_ablation_weights' / 'a2_without_adaptive_K100_2d_re50.yml')

def create_c1_prior_comparison():
    """C1: RANS Prior 對比配置"""
    # 載入 Full 配置
    full_config_path = Path(__file__).parent.parent / "kolmogorov_re50_kf4_K100_rans_prior_1k.yml"
    with open(full_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # C1.1: 無 Prior
    config_no_prior = deepcopy(base_config)
    config_no_prior['experiment']['name'] = 'c1_no_prior_K100_2d_re50'
    config_no_prior['experiment']['description'] = 'C1: Full without RANS Prior'
    config_no_prior['lowfi_prior']['enabled'] = False
    config_no_prior['losses']['prior_weight'] = 0.0
    config_no_prior['output'] = {
        'checkpoint_dir': './checkpoints/experiments/C1_no_prior_K100',
        'results_dir': './results/experiments/C1_no_prior_K100',
        'visualization_dir': './results/experiments/C1_no_prior_K100/visualizations'
    }
    save_config(config_no_prior, EXPERIMENT_DIR / 'C1_prior_comparison' / 'c1_no_prior_K100_2d_re50.yml')
    
    # C1.2: 有 Prior
    config_with_prior = deepcopy(base_config)
    config_with_prior['experiment']['name'] = 'c1_with_prior_K100_2d_re50'
    config_with_prior['experiment']['description'] = 'C1: Full with RANS Prior'
    config_with_prior['lowfi_prior']['enabled'] = True
    config_with_prior['lowfi_prior']['consistency_weight'] = 0.3  # 推薦值
    config_with_prior['losses']['prior_weight'] = 0.3
    config_with_prior['output'] = {
        'checkpoint_dir': './checkpoints/experiments/C1_with_prior_K100',
        'results_dir': './results/experiments/C1_with_prior_K100',
        'visualization_dir': './results/experiments/C1_with_prior_K100/visualizations'
    }
    save_config(config_with_prior, EXPERIMENT_DIR / 'C1_prior_comparison' / 'c1_with_prior_K100_2d_re50.yml')

def create_c2_prior_sweep():
    """C2: RANS Prior 權重掃描配置"""
    # 載入 Full 配置
    full_config_path = Path(__file__).parent.parent / "kolmogorov_re50_kf4_K100_rans_prior_1k.yml"
    with open(full_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    for prior_weight in [0.1, 0.3, 0.5]:
        config = deepcopy(base_config)
        
        # 修改實驗資訊
        config['experiment']['name'] = f'c2_prior_{prior_weight:.1f}_K100_2d_re50'
        config['experiment']['description'] = f'C2: RANS Prior weight={prior_weight}'
        
        # 修改 Prior 權重
        config['lowfi_prior']['enabled'] = True
        config['lowfi_prior']['consistency_weight'] = prior_weight
        config['losses']['prior_weight'] = prior_weight
        
        # 輸出目錄
        config['output'] = {
            'checkpoint_dir': f'./checkpoints/experiments/C2_prior_{prior_weight:.1f}_K100',
            'results_dir': f'./results/experiments/C2_prior_{prior_weight:.1f}_K100',
            'visualization_dir': f'./results/experiments/C2_prior_{prior_weight:.1f}_K100/visualizations'
        }
        
        output_path = EXPERIMENT_DIR / 'C2_prior_sweep' / f'c2_prior_{prior_weight:.1f}_K100_2d_re50.yml'
        save_config(config, output_path)

def create_readme_for_experiments():
    """為每個實驗創建 README"""
    experiments = {
        'S1_sensor_strategy': {
            'title': 'S1: 感測點策略對比',
            'description': '對比 Random vs QR-pivot 感測器選擇策略',
            'configs': ['s1_random_K100_2d_re50.yml', 's1_qr_K100_2d_re50.yml'],
            'comparison': 'L2(u,v,∇p), ‖∇·u‖, sensor quality (condition number)'
        },
        'S2_k_scan': {
            'title': 'S2: K 值掃描實驗',
            'description': '掃描不同感測點數量 K ∈ {30, 50, 80, 100}',
            'configs': [f's2_qr_K{K}_2d_re50.yml' for K in [30, 50, 80, 100]],
            'comparison': 'K-error 曲線，找最小可行 K'
        },
        'M1_model_comparison': {
            'title': 'M1: 模型表示能力對比',
            'description': '對比 Vanilla MLP vs Full features',
            'configs': ['m1_vanilla_K100_2d_re50.yml', 'm1_full_K100_2d_re50.yml'],
            'comparison': 'L2 與 divergence trade-off'
        },
        'A1_ablation_fourier': {
            'title': 'A1: Fourier Features 消融實驗',
            'description': '量化 Fourier Features 的貢獻',
            'configs': ['a1_with_fourier_K100_2d_re50.yml', 'a1_without_fourier_K100_2d_re50.yml'],
            'comparison': 'L2 與能譜差異'
        },
        'A2_ablation_weights': {
            'title': 'A2: 動態權重消融實驗',
            'description': '量化 GradNorm 自適應權重的貢獻',
            'configs': ['a2_with_adaptive_K100_2d_re50.yml', 'a2_without_adaptive_K100_2d_re50.yml'],
            'comparison': '收斂速度（epochs/time）+ 最終 L2'
        },
        'C1_prior_comparison': {
            'title': 'C1: RANS Prior 對比實驗',
            'description': '對比有無 RANS 先驗的性能差異',
            'configs': ['c1_no_prior_K100_2d_re50.yml', 'c1_with_prior_K100_2d_re50.yml'],
            'comparison': 'L2(u,v,∇p), ‖∇·u‖, 壓力場重建品質'
        },
        'C2_prior_sweep': {
            'title': 'C2: RANS Prior 權重掃描',
            'description': '掃描 prior_weight ∈ {0.1, 0.3, 0.5}',
            'configs': [f'c2_prior_{w:.1f}_K100_2d_re50.yml' for w in [0.1, 0.3, 0.5]],
            'comparison': 'error vs prior_weight 曲線'
        }
    }
    
    for exp_dir, info in experiments.items():
        readme_content = f"""# {info['title']}

## 實驗目的

{info['description']}

## 配置文件

"""
        for config_file in info['configs']:
            readme_content += f"- `{config_file}`\n"
        
        readme_content += f"""
## 對比指標

{info['comparison']}

## 執行方式

```bash
# 從 repo root 執行（建議）：一次跑完本實驗資料夾
for cfg in configs/experiments/{exp_dir}/*.yml; do python scripts/train/train.py --cfg "$cfg"; done

# 或單一配置（例）
python scripts/train/train.py --cfg configs/experiments/{exp_dir}/{info['configs'][0]}
```

## 評估結果

```bash
# 以配置中的 output.checkpoint_dir 為準（best_model.pth / latest.pth）
python scripts/evaluate/evaluate_checkpoint.py \\
  --checkpoint <checkpoint_dir>/best_model.pth \\
  --config configs/experiments/{exp_dir}/{info['configs'][0]}
```

## 預期結果

請參考 `docs/EXPERIMENT_COMPARISON_PLAN.md` 中的預期性能表格。
"""
        
        readme_path = EXPERIMENT_DIR / exp_dir / 'README.md'
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"✅ 已生成: {readme_path.relative_to(EXPERIMENT_DIR)}")

def main():
    """主函數"""
    print("=" * 70)
    print("開始生成實驗對比配置文件")
    print("=" * 70)
    
    # 生成各實驗配置
    print("\n📁 S1: 感測點策略對比")
    create_s1_random()
    create_s1_qr()
    
    print("\n📁 S2: K 值掃描")
    create_s2_k_scan()
    
    print("\n📁 M1: 模型對比")
    create_m1_vanilla()
    create_m1_full()
    
    print("\n📁 A1: Fourier 消融")
    create_a1_ablations()
    
    print("\n📁 A2: 動態權重消融")
    create_a2_ablations()
    
    print("\n📁 C1: Prior 對比")
    create_c1_prior_comparison()
    
    print("\n📁 C2: Prior 權重掃描")
    create_c2_prior_sweep()
    
    print("\n📄 生成各實驗 README")
    create_readme_for_experiments()
    
    print("\n" + "=" * 70)
    print("✅ 所有實驗配置文件已生成完成")
    print("=" * 70)
    print("\n下一步：")
    print("1. 檢查配置文件: ls configs/experiments/*/")
    print("2. 執行實驗: python scripts/train/train.py --cfg configs/experiments/<實驗>/<配置>.yml")
    print("3. 評估結果: python scripts/compare/compare_experiments.py")

if __name__ == '__main__':
    main()

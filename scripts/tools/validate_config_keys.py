#!/usr/bin/env python3
"""
配置鍵名一致性檢查器
用途：驗證 YAML 配置檔案是否使用正確的鍵名
版本：v1.0 (2025-12-19)
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple

# =============================================================================
# 標準配置鍵名定義
# =============================================================================

# 頂層必須使用的鍵名（複數形式）
REQUIRED_PLURAL_KEYS = {
    'losses': 'loss',  # 正確: losses, 錯誤: loss
}

# 所有已知的頂層配置鍵
KNOWN_TOP_LEVEL_KEYS = {
    'experiment', 'reproducibility', 'data', 'lowfi_prior', 'normalization',
    'sensors', 'model', 'physics', 'losses', 'training', 'fourier_annealing',
    'curriculum', 'ensemble', 'evaluation', 'physics_validation', 'logging',
    'output', 'usage_notes', 'test_acceptance', 'ablation_experiment_plan',
    # 資料配置相關
    'cache', 'interpolation', 'domain', 'dataset',
    # 監控相關
    'paths', 'trend_analysis', 'monitoring', 'metrics', 'log_parsing', 'eta',
    # 其他功能
    'checkpointing', 'task_008_gates', 'optimizer_switching'
}

# 已移除/不允許的頂層鍵（需移動到正確位置）
LEGACY_TOP_LEVEL_KEYS = {
    'sampling': "移至 training.sampling",
    'validation': "移至 training.validation 或 evaluation",
    'optimizer': "移至 training.optimizer",
    'wandb': "移至 logging.wandb",
    'weighting': "移至 losses.weighting",
}

# losses 段落下的所有已知鍵
KNOWN_LOSS_KEYS = {
    # 基礎損失
    'data_weight',
    # PDE 損失
    'momentum_x_weight', 'momentum_y_weight', 'momentum_z_weight', 'continuity_weight',
    # 約束損失
    'wall_constraint_weight', 'periodicity_weight', 'pressure_gradient_weight',
    # VS-PINN 專屬
    'inlet_weight', 'initial_condition_weight', 'bulk_velocity_weight',
    'centerline_dudy_weight', 'centerline_v_weight', 'pressure_reference_weight',
    # 先驗損失
    'prior_weight',
    # 正則化
    'source_l1', 'gradient_penalty',
    # 自適應權重
    'adaptive_weighting', 'weight_update_freq', 'grad_norm_alpha', 'adaptive_loss_terms',
    'grad_norm_momentum', 'grad_norm_normalize',
    # 因果權重
    'causal_weighting', 'causal_tol', 'num_chunks', 'causal_eps', 'causal_n_bins',
    # 其他實驗性/legacy
    'l2_regularization', 'merge_momentum',
    'adaptation_frequency', 'adaptation_alpha',
    'adaptation_method', 'normalization_method',
    'weighting', 'rho', 'nu', 'pde', 'sensor', 'bc',
    # 損失歸一化（已支援但未從配置讀取）
    'normalize_losses', 'warmup_epochs',
}

# 已移除的舊鍵名（觸發錯誤；不再提供向後相容）
REMOVED_KEY_PATHS = {
    ('losses', 'div_weight'): "continuity_weight",
    ('losses', 'boundary_weight'): "wall_constraint_weight",
    ('model', 'use_fourier'): "model.fourier_features",
    ('model', 'fourier_m'): "model.fourier_features.fourier_m",
    ('model', 'fourier_sigma'): "model.fourier_features.fourier_sigma",
    ('model', 'trainable_fourier'): "model.fourier_features.trainable_fourier",
    ('model', 'fourier_use_2pi'): "model.fourier_features.fourier_use_2pi",
    ('model', 'fourier_multiscale'): "removed (use standard fourier_features only)",
    ('model', 'fourier_features', 'enabled'): "model.fourier_features.type",
    ('training', 'sampling', 'pde_points'): "N_pde (已統一命名標準)",
    ('sampling', 'pde_points'): "N_pde (已統一命名標準)",
}

# =============================================================================
# 檢查函數
# =============================================================================

def check_config_keys(config_path: str) -> Tuple[bool, List[str], List[str]]:
    """
    檢查配置鍵名一致性
    
    Args:
        config_path: 配置檔案路徑
        
    Returns:
        (is_valid, errors, warnings):
            - is_valid: 是否通過檢查
            - errors: 錯誤列表（阻斷性問題）
            - warnings: 警告列表（非阻斷性問題）
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        return False, [f"❌ 配置檔案不存在: {config_path}"], []
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, [f"❌ YAML 解析失敗: {e}"], []
    
    if config is None:
        return False, [f"❌ 配置檔案為空"], []
    
    errors = []
    warnings = []
    
    # 檢測特殊用途配置（非訓練配置）
    is_data_config = 'dataset' in config and 'training' not in config and 'model' not in config
    is_monitoring_config = 'monitoring' in config and 'metrics' in config
    
    if is_data_config:
        warnings.append(
            f"ℹ️  這是資料配置檔案（dataset config），跳過訓練配置檢查"
        )
        return True, [], warnings
    
    if is_monitoring_config:
        warnings.append(
            f"ℹ️  這是監控系統配置（monitoring config），跳過訓練配置檢查"
        )
        return True, [], warnings
    
    # 檢查 1: 檢查是否使用錯誤的單數形式鍵名
    for correct_key, wrong_key in REQUIRED_PLURAL_KEYS.items():
        if wrong_key in config and correct_key not in config:
            errors.append(
                f"❌ 使用了錯誤的鍵名 '{wrong_key}'，應改為 '{correct_key}' (複數)\n"
                f"   位置: 配置檔案頂層\n"
                f"   影響: 損失權重配置將被忽略，所有權重預設為 1.0\n"
                f"   修復: 將 '{wrong_key}:' 改為 '{correct_key}:'"
            )

    # 檢查 1.5: 舊鍵名（已移除向後相容）
    for path, replacement in REMOVED_KEY_PATHS.items():
        cursor = config
        found = True
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                found = False
                break
            cursor = cursor[key]
        if found:
            dotted_path = ".".join(path)
            errors.append(
                f"❌ 使用已移除的舊鍵名 '{dotted_path}'\n"
                f"   修復: 改為 {replacement}"
            )
    
    # 檢查 1.6: Curriculum stages 中的 pde_points
    if 'curriculum' in config and isinstance(config['curriculum'], dict):
        if 'stages' in config['curriculum']:
            stages = config['curriculum']['stages']
            if isinstance(stages, list):
                for i, stage in enumerate(stages):
                    if isinstance(stage, dict):
                        if 'sampling' in stage and isinstance(stage['sampling'], dict):
                            if 'pde_points' in stage['sampling']:
                                errors.append(
                                    f"❌ Curriculum stage {i+1} 使用已棄用的 'pde_points'\n"
                                    f"   位置: curriculum.stages[{i}].sampling.pde_points\n"
                                    f"   修復: 改為 N_pde"
                                )
    
    # 檢查 2: 檢查 losses 段落內容
    if 'losses' in config:
        losses_cfg = config['losses']
        if not isinstance(losses_cfg, dict):
            errors.append(f"❌ 'losses' 必須是字典類型，當前類型: {type(losses_cfg)}")
        else:
            # 檢查未知的損失鍵
            unknown_keys = set(losses_cfg.keys()) - KNOWN_LOSS_KEYS
            if unknown_keys:
                warnings.append(
                    f"⚠️  發現未知的損失權重鍵: {', '.join(unknown_keys)}\n"
                    f"   這些鍵可能不會被使用，請確認拼寫是否正確"
                )
            
    # 檢查 3: 不允許的頂層鍵（legacy）
    legacy_hits = set(config.keys()) & set(LEGACY_TOP_LEVEL_KEYS.keys())
    for key in sorted(legacy_hits):
        errors.append(
            f"❌ 不允許的頂層配置鍵: {key}\n"
            f"   修復方法: {LEGACY_TOP_LEVEL_KEYS[key]}"
        )

    # 檢查 4: 檢查未知的頂層鍵（嚴格）
    unknown_top_keys = set(config.keys()) - KNOWN_TOP_LEVEL_KEYS - set(LEGACY_TOP_LEVEL_KEYS.keys())
    if unknown_top_keys:
        errors.append(
            f"❌ 發現未知的頂層配置鍵: {', '.join(sorted(unknown_top_keys))}\n"
            f"   請確認拼寫或移至正確段落"
        )
    
    # 檢查 4: 驗證關鍵配置是否存在
    if 'losses' not in config:
        warnings.append(
            f"⚠️  未找到 'losses' 配置段落\n"
            f"   所有損失權重將使用預設值"
        )
    
    if 'physics' not in config:
        errors.append(f"❌ 缺少必要的 'physics' 配置段落")
    
    if 'training' not in config:
        errors.append(f"❌ 缺少必要的 'training' 配置段落")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def check_multiple_configs(config_paths: List[str]) -> Dict[str, Tuple[bool, List[str], List[str]]]:
    """
    批量檢查多個配置檔案
    
    Args:
        config_paths: 配置檔案路徑列表
        
    Returns:
        結果字典 {config_path: (is_valid, errors, warnings)}
    """
    results = {}
    for config_path in config_paths:
        results[config_path] = check_config_keys(config_path)
    return results


def print_check_results(config_path: str, is_valid: bool, errors: List[str], warnings: List[str]) -> None:
    """打印檢查結果"""
    print(f"\n{'='*70}")
    print(f"📋 配置檔案: {config_path}")
    print(f"{'='*70}")
    
    if is_valid and not warnings:
        print("✅ 配置檢查通過，未發現問題")
        return
    
    if errors:
        print(f"\n❌ 發現 {len(errors)} 個錯誤（阻斷性問題）:")
        for i, error in enumerate(errors, 1):
            print(f"\n{i}. {error}")
    
    if warnings:
        print(f"\n⚠️  發現 {len(warnings)} 個警告（非阻斷性問題）:")
        for i, warning in enumerate(warnings, 1):
            print(f"\n{i}. {warning}")
    
    if is_valid:
        print(f"\n✅ 整體狀態: 通過（有警告但不影響運行）")
    else:
        print(f"\n❌ 整體狀態: 失敗（存在阻斷性錯誤，需要修復）")


# =============================================================================
# 主程式
# =============================================================================

def main():
    """主函數"""
    if len(sys.argv) < 2:
        print("用法: python validate_config_keys.py <config_file> [config_file2 ...]")
        print("\n範例:")
        print("  python validate_config_keys.py configs/phase_a_qr_baseline_fixed.yml")
        print("  python validate_config_keys.py configs/*.yml")
        sys.exit(1)
    
    config_paths = sys.argv[1:]
    
    print("=" * 70)
    print("🔍 配置鍵名一致性檢查器")
    print("=" * 70)
    print(f"檢查 {len(config_paths)} 個配置檔案...")
    
    all_valid = True
    for config_path in config_paths:
        is_valid, errors, warnings = check_config_keys(config_path)
        print_check_results(config_path, is_valid, errors, warnings)
        if not is_valid:
            all_valid = False
    
    print("\n" + "=" * 70)
    if all_valid:
        print("✅ 所有配置檔案檢查通過")
        sys.exit(0)
    else:
        print("❌ 部分配置檔案存在錯誤，請修復後重試")
        sys.exit(1)


if __name__ == '__main__':
    main()

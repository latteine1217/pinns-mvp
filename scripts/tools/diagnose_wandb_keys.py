#!/usr/bin/env python3
"""
診斷 WandB 日誌鍵的對應關係

檢查 loss_manager.py 返回的鍵與 training_loop_manager.py 記錄的鍵是否匹配
"""

import re
from pathlib import Path

def extract_result_keys(loss_manager_path):
    """從 loss_manager.py 提取所有添加到 result 字典的鍵"""
    with open(loss_manager_path, 'r') as f:
        content = f.read()
    
    # 找到 combine_losses 函數
    pattern = r"result\.update\(\{([^}]+)\}\)"
    matches = re.findall(pattern, content)
    
    result_keys = set()
    for match in matches:
        # 提取鍵名
        keys = re.findall(r"'([^']+)':", match)
        result_keys.update(keys)
    
    # 也提取直接賦值的鍵
    direct_pattern = r"result\['([^']+)'\]\s*="
    direct_keys = re.findall(direct_pattern, content)
    result_keys.update(direct_keys)
    
    # 從 base_result_keys 列表中提取
    base_keys_pattern = r"base_result_keys\s*=\s*\[([^\]]+)\]"
    base_match = re.search(base_keys_pattern, content)
    if base_match:
        base_keys = re.findall(r"'([^']+)'", base_match.group(1))
        result_keys.update(base_keys)
    
    # 從條件式添加的鍵中提取
    bc_keys_pattern = r"bc_keys\s*=\s*\[([^\]]+)\]"
    bc_match = re.search(bc_keys_pattern, content)
    if bc_match:
        bc_keys_text = bc_match.group(1)
        bc_keys = re.findall(r"'([^']+)'", bc_keys_text)
        result_keys.update(bc_keys)
    
    return result_keys

def extract_function_return_keys(loss_manager_path, function_name):
    """從 loss_manager.py 提取特定函數返回的鍵"""
    with open(loss_manager_path, 'r') as f:
        content = f.read()
    
    # 找到函數定義
    func_pattern = rf"def {function_name}\([^)]+\).*?return \{{([^}}]+)\}}"
    match = re.search(func_pattern, content, re.DOTALL)
    
    if not match:
        return set()
    
    return_dict = match.group(1)
    keys = re.findall(r"'([^']+)':", return_dict)
    return set(keys)

def extract_wandb_log_keys(training_loop_path):
    """從 training_loop_manager.py 提取所有記錄到 WandB 的鍵"""
    with open(training_loop_path, 'r') as f:
        content = f.read()
    
    wandb_keys = {}
    
    # 提取所有 log_dict[...] = loss_dict.get(...) 的模式
    pattern = r"log_dict\['([^']+)'\]\s*=\s*loss_dict(?:\.get\('([^']+)'|(?:\['([^']+)'\]))"
    matches = re.findall(pattern, content)
    
    for match in matches:
        wandb_key = match[0]
        loss_key = match[1] if match[1] else match[2]
        if loss_key:
            wandb_keys[wandb_key] = loss_key
    
    # 也提取條件式的記錄
    cond_pattern = r"if '([^']+)' in loss_dict:\s+log_dict\['([^']+)'\]\s*=\s*loss_dict\['([^']+)'\]"
    cond_matches = re.findall(cond_pattern, content)
    
    for match in cond_matches:
        check_key, wandb_key, loss_key = match
        wandb_keys[wandb_key] = loss_key
    
    return wandb_keys

def main():
    project_root = Path(__file__).parent.parent.parent
    loss_manager_path = project_root / "pinnx/train/loss_manager.py"
    training_loop_path = project_root / "pinnx/train/training_loop_manager.py"
    
    print("=" * 80)
    print("WandB 日誌鍵診斷報告")
    print("=" * 80)
    print()
    
    # 1. 提取 combine_losses 返回的所有鍵
    print("📊 loss_manager.py::combine_losses() 返回的鍵:")
    print("-" * 80)
    result_keys = extract_result_keys(loss_manager_path)
    for key in sorted(result_keys):
        print(f"  ✓ {key}")
    print(f"\n  總計: {len(result_keys)} 個鍵\n")
    
    # 2. 提取各個 compute 函數返回的鍵
    print("📊 各個 compute 函數返回的鍵:")
    print("-" * 80)
    
    functions = [
        'compute_pde_loss',
        'compute_bc_loss',
        'compute_data_loss',
        'compute_lowfi_prior_loss',
        'compute_mean_constraint_loss'
    ]
    
    all_compute_keys = set()
    for func in functions:
        keys = extract_function_return_keys(loss_manager_path, func)
        all_compute_keys.update(keys)
        print(f"  {func}:")
        for key in sorted(keys):
            print(f"    - {key}")
    print(f"\n  總計: {len(all_compute_keys)} 個鍵\n")
    
    # 3. 提取 WandB 記錄的鍵映射
    print("📊 training_loop_manager.py 記錄的 WandB 鍵:")
    print("-" * 80)
    wandb_keys = extract_wandb_log_keys(training_loop_path)
    for wandb_key, loss_key in sorted(wandb_keys.items()):
        print(f"  {wandb_key:40s} <- loss_dict['{loss_key}']")
    print(f"\n  總計: {len(wandb_keys)} 個鍵映射\n")
    
    # 4. 檢查不匹配
    print("⚠️  潛在問題:")
    print("-" * 80)
    
    # 檢查 WandB 嘗試記錄但 loss_dict 中不存在的鍵
    missing_keys = set()
    for wandb_key, loss_key in wandb_keys.items():
        if loss_key not in result_keys and loss_key not in all_compute_keys:
            missing_keys.add(loss_key)
            print(f"  ❌ WandB 期望 '{loss_key}' 但 loss_manager 未返回")
            print(f"     (用於記錄 '{wandb_key}')")
    
    if not missing_keys:
        print("  ✅ 沒有發現缺失的鍵")
    
    print()
    print("=" * 80)
    print(f"診斷完成")
    print("=" * 80)

if __name__ == "__main__":
    main()

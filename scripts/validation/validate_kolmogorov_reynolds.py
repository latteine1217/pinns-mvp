#!/usr/bin/env python3
"""
Kolmogorov Flow 雷諾數驗證工具
==============================

功能：
1. 根據目標 Re 計算所需的黏滯度 ν
2. 驗證配置文件中的 Re 設定是否正確
3. 生成標準參數表

雷諾數定義：
    Re = F / (ν² k³)
    
反推黏滯度：
    ν = √(F / (Re × k³))

作者：PINNs-MVP 團隊
日期：2025-11-21
"""

import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys


# ==============================================================================
# 雷諾數計算工具
# ==============================================================================

def compute_reynolds_number(F: float, nu: float, k: int) -> float:
    """
    計算 Kolmogorov Flow 雷諾數（標準定義）
    
    Args:
        F: 強迫振幅 A
        nu: 動力黏度 ν
        k: 強迫波數 k_f
    
    Returns:
        Re: 雷諾數
    """
    Re = F / (nu**2 * k**3)
    return Re


def compute_viscosity(Re: float, F: float, k: int) -> float:
    """
    根據目標雷諾數反推所需黏滯度
    
    公式：ν = √(F / (Re × k³))
    
    Args:
        Re: 目標雷諾數
        F: 強迫振幅 A
        k: 強迫波數 k_f
    
    Returns:
        nu: 所需動力黏度 ν
    """
    nu = np.sqrt(F / (Re * k**3))
    return nu


def compute_laminar_velocity(F: float, nu: float, k: int) -> float:
    """
    計算層流解的特徵速度
    
    公式：U = F / (ν k²)
    
    Args:
        F: 強迫振幅 A
        nu: 動力黏度 ν
        k: 強迫波數 k_f
    
    Returns:
        U: 層流速度振幅
    """
    U = F / (nu * k**2)
    return U


# ==============================================================================
# 配置文件驗證
# ==============================================================================

def validate_config_reynolds(config_path: Path) -> Dict:
    """
    驗證單個配置文件的雷諾數設定
    
    Args:
        config_path: 配置文件路徑
    
    Returns:
        驗證結果字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 提取物理參數
    physics = config.get('physics', {})
    forcing = physics.get('forcing', {})
    
    F = forcing.get('amplitude', 1.0)
    k = forcing.get('wavenumber', 4)
    nu = physics.get('nu', 0.01)
    
    # 計算實際雷諾數
    Re_actual = compute_reynolds_number(F, nu, k)
    
    # 計算層流速度
    U_laminar = compute_laminar_velocity(F, nu, k)
    
    # 檢查配置文件中是否有註解宣稱的 Re
    Re_claimed = None
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 搜尋 "Re=XX" 或 "Re = XX" 模式
        import re
        matches = re.findall(r'Re\s*[=:]\s*(\d+)', content)
        if matches:
            Re_claimed = int(matches[0])
    
    # 判斷是否一致
    is_valid = True
    message = "✅ 配置正確"
    
    if Re_claimed is not None:
        error = abs(Re_actual - Re_claimed) / Re_claimed
        if error > 0.05:  # 容忍 5% 誤差
            is_valid = False
            message = f"❌ 雷諾數不一致！宣稱 Re={Re_claimed}, 實際 Re={Re_actual:.2f}"
    
    return {
        'config': config_path.name,
        'F': F,
        'k': k,
        'nu': nu,
        'Re_actual': Re_actual,
        'Re_claimed': Re_claimed,
        'U_laminar': U_laminar,
        'is_valid': is_valid,
        'message': message,
    }


def validate_all_configs(config_dir: Path) -> List[Dict]:
    """
    驗證目錄下所有 Kolmogorov 配置文件
    
    Args:
        config_dir: 配置文件目錄
    
    Returns:
        驗證結果列表
    """
    results = []
    
    # 搜尋所有 kolmogorov 相關配置
    pattern = "kolmogorov*.yml"
    for config_path in sorted(config_dir.glob(pattern)):
        try:
            result = validate_config_reynolds(config_path)
            results.append(result)
        except Exception as e:
            results.append({
                'config': config_path.name,
                'is_valid': False,
                'message': f"❌ 讀取失敗：{str(e)}",
            })
    
    return results


# ==============================================================================
# 參數表生成
# ==============================================================================

def generate_parameter_table(Re_targets: List[int], F: float = 1.0, k: int = 4):
    """
    生成標準參數表（給定 F 和 k，計算不同 Re 所需的 ν）
    
    Args:
        Re_targets: 目標雷諾數列表
        F: 強迫振幅（默認 1.0）
        k: 強迫波數（默認 4）
    """
    print(f"\n{'='*80}")
    print(f"Kolmogorov Flow 標準參數表 (F={F}, k={k})")
    print(f"{'='*80}")
    print(f"{'Re':<10} {'ν (nu)':<15} {'U_laminar':<15} {'物理狀態':<20}")
    print(f"{'-'*80}")
    
    for Re in Re_targets:
        nu = compute_viscosity(Re, F, k)
        U = compute_laminar_velocity(F, nu, k)
        
        # 判斷物理狀態
        if Re < 10:
            state = "層流（穩定）"
        elif Re < 30:
            state = "弱失穩"
        elif Re < 60:
            state = "時空混沌"
        else:
            state = "完全發展湍流"
        
        print(f"{Re:<10} {nu:<15.6f} {U:<15.4f} {state:<20}")
    
    print(f"{'='*80}\n")


def print_formula_reference():
    """打印公式參考"""
    print("\n" + "="*80)
    print("Kolmogorov Flow 雷諾數公式參考")
    print("="*80)
    print("""
標準定義：
    Re = F / (ν² k³)

其中：
    - F = A（強迫振幅）
    - ν = 動力黏度
    - k = k_f（強迫波數）

推導：
    層流解：u_x(y) = (F / (ν k²)) sin(k y)
    特徵速度：U = F / (ν k²)
    特徵長度：L = 1/k
    雷諾數：Re = UL/ν = (F / (ν k²)) × (1/k) / ν = F / (ν² k³)

反推黏滯度：
    ν = √(F / (Re × k³))

參考文獻：
    - Meshalkin & Sinai (1961): Stability of steady state Kolmogorov flow
    - Boffetta et al. (2002): Inverse energy cascade in two-dimensional turbulence
    """)
    print("="*80 + "\n")


# ==============================================================================
# 命令行介面
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kolmogorov Flow 雷諾數驗證與計算工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 根據 Re=30 計算所需黏滯度
  python validate_kolmogorov_reynolds.py --compute-nu --Re 30 --F 1.0 --k 4
  
  # 驗證配置文件
  python validate_kolmogorov_reynolds.py --validate --config-dir configs/kolmogorov_experiments/
  
  # 生成標準參數表
  python validate_kolmogorov_reynolds.py --table --Re-list 20 30 40 50 60 80 100
        """
    )
    
    # 模式選擇
    parser.add_argument('--compute-nu', action='store_true',
                        help='計算所需黏滯度（需指定 --Re, --F, --k）')
    parser.add_argument('--validate', action='store_true',
                        help='驗證配置文件雷諾數設定')
    parser.add_argument('--table', action='store_true',
                        help='生成標準參數表')
    parser.add_argument('--formula', action='store_true',
                        help='顯示公式參考')
    
    # 參數
    parser.add_argument('--Re', type=float, help='目標雷諾數')
    parser.add_argument('--F', type=float, default=1.0, help='強迫振幅（默認 1.0）')
    parser.add_argument('--k', type=int, default=4, help='強迫波數（默認 4）')
    parser.add_argument('--Re-list', type=int, nargs='+',
                        default=[20, 30, 40, 50, 60, 80, 100],
                        help='參數表的 Re 列表')
    parser.add_argument('--config-dir', type=str,
                        default='configs/kolmogorov_experiments/',
                        help='配置文件目錄')
    
    args = parser.parse_args()
    
    # === 模式 1：計算黏滯度 ===
    if args.compute_nu:
        if args.Re is None:
            print("❌ 錯誤：需要指定 --Re 參數")
            sys.exit(1)
        
        nu = compute_viscosity(args.Re, args.F, args.k)
        Re_check = compute_reynolds_number(args.F, nu, args.k)
        U = compute_laminar_velocity(args.F, nu, args.k)
        
        print(f"\n{'='*60}")
        print(f"目標雷諾數：Re = {args.Re}")
        print(f"強迫參數：F = {args.F}, k = {args.k}")
        print(f"{'-'*60}")
        print(f"所需黏滯度：ν = {nu:.6f}")
        print(f"層流速度：U = {U:.4f}")
        print(f"驗證：Re = {Re_check:.2f} ({'✅ 正確' if abs(Re_check - args.Re) < 0.01 else '❌ 誤差'})")
        print(f"{'='*60}\n")
    
    # === 模式 2：驗證配置文件 ===
    if args.validate:
        config_dir = Path(args.config_dir)
        if not config_dir.exists():
            print(f"❌ 錯誤：配置目錄不存在：{config_dir}")
            sys.exit(1)
        
        results = validate_all_configs(config_dir)
        
        print(f"\n{'='*100}")
        print(f"Kolmogorov Flow 配置文件雷諾數驗證報告")
        print(f"{'='*100}")
        print(f"{'配置文件':<35} {'F':<8} {'k':<6} {'ν':<12} {'Re(實際)':<12} {'Re(宣稱)':<12} {'狀態':<10}")
        print(f"{'-'*100}")
        
        for res in results:
            config = res['config']
            F = res.get('F', '-')
            k = res.get('k', '-')
            nu = res.get('nu', '-')
            Re_actual = res.get('Re_actual', '-')
            Re_claimed = res.get('Re_claimed', '-')
            
            if isinstance(Re_actual, float):
                Re_actual_str = f"{Re_actual:.2f}"
            else:
                Re_actual_str = str(Re_actual)
            
            if isinstance(Re_claimed, int):
                Re_claimed_str = str(Re_claimed)
            else:
                Re_claimed_str = '-'
            
            status = '✅' if res['is_valid'] else '❌'
            
            print(f"{config:<35} {F:<8} {k:<6} {nu:<12.6f} {Re_actual_str:<12} {Re_claimed_str:<12} {status:<10}")
        
        print(f"{'='*100}")
        
        # 統計
        valid_count = sum(1 for r in results if r['is_valid'])
        total_count = len(results)
        print(f"\n總結：{valid_count}/{total_count} 個配置文件通過驗證\n")
    
    # === 模式 3：生成參數表 ===
    if args.table:
        generate_parameter_table(args.Re_list, args.F, args.k)
    
    # === 模式 4：顯示公式 ===
    if args.formula:
        print_formula_reference()
    
    # 如果沒有指定任何模式，顯示幫助
    if not (args.compute_nu or args.validate or args.table or args.formula):
        parser.print_help()


if __name__ == '__main__':
    main()

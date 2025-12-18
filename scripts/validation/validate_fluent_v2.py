#!/usr/bin/env python3
"""
驗證 Fluent V2 數據品質
Phase 2: 數值品質檢查 (NaN/Inf/負值)
"""
import h5py
import numpy as np
import sys
from pathlib import Path

def validate_fluent_data(filepath):
    """驗證 Fluent HDF5 文件的數值品質"""
    
    print("="*70)
    print("🔬 Fluent V2 數據品質驗證")
    print("="*70)
    print(f"文件: {filepath}")
    print()
    
    issues_found = []
    warnings_found = []
    
    try:
        with h5py.File(filepath, 'r') as f:
            # 獲取 cells 數據
            cells_path = 'results/1/phase-1/cells'
            if cells_path not in f:
                print("❌ 找不到 cells 數據路徑")
                return False
            
            cells = f[cells_path]
            
            # 定義需要檢查的變數
            check_vars = {
                'SV_U': {'name': '流向速度 (u)', 'should_positive': False},
                'SV_V': {'name': '法向速度 (v)', 'should_positive': False},
                'SV_W': {'name': '展向速度 (w)', 'should_positive': False},
                'SV_P': {'name': '壓力 (p)', 'should_positive': False},
                'SV_K': {'name': '湍動能 (k)', 'should_positive': True},
                'SV_O': {'name': 'Omega (ω)', 'should_positive': True},
                'SV_MU_T': {'name': '湍流黏度 (μ_t)', 'should_positive': True},
                'SV_MU_LAM': {'name': '層流黏度 (μ)', 'should_positive': True},
                'SV_DENSITY': {'name': '密度 (ρ)', 'should_positive': True},
                'SV_WALL_DIST': {'name': '壁面距離', 'should_positive': True}
            }
            
            print("📊 變數品質檢查:")
            print("-" * 70)
            
            all_passed = True
            
            for var_name, var_info in check_vars.items():
                if var_name not in cells:
                    warnings_found.append(f"{var_name} 不存在")
                    print(f"  ⚠️  {var_info['name']} ({var_name}): 變數不存在")
                    continue
                
                var_group = cells[var_name]
                
                # Fluent HDF5 數據在子集 '1' 中
                if not isinstance(var_group, h5py.Group) or '1' not in var_group:
                    issues_found.append(f"{var_name} 缺少數據子集")
                    print(f"  ❌ {var_info['name']} ({var_name}): 缺少數據子集")
                    all_passed = False
                    continue
                
                # 讀取數據
                data = np.array(var_group['1'])
                
                # 統計量
                n_total = data.size
                n_nan = np.isnan(data).sum()
                n_inf = np.isinf(data).sum()
                n_neg = (data < 0).sum()
                
                mean_val = np.nanmean(data) if n_nan < n_total else np.nan
                std_val = np.nanstd(data) if n_nan < n_total else np.nan
                min_val = np.nanmin(data) if n_nan < n_total else np.nan
                max_val = np.nanmax(data) if n_nan < n_total else np.nan
                
                # 品質檢查
                status = "✅"
                issues = []
                
                if n_nan > 0:
                    issues.append(f"NaN: {n_nan}/{n_total} ({n_nan/n_total*100:.2f}%)")
                    issues_found.append(f"{var_name}: {n_nan} NaN values")
                    status = "❌"
                    all_passed = False
                
                if n_inf > 0:
                    issues.append(f"Inf: {n_inf}/{n_total} ({n_inf/n_total*100:.2f}%)")
                    issues_found.append(f"{var_name}: {n_inf} Inf values")
                    status = "❌"
                    all_passed = False
                
                if var_info['should_positive'] and n_neg > 0:
                    issues.append(f"負值: {n_neg}/{n_total} ({n_neg/n_total*100:.2f}%)")
                    issues_found.append(f"{var_name}: {n_neg} negative values (should be positive)")
                    status = "❌"
                    all_passed = False
                
                # 輸出結果
                print(f"\n  {status} {var_info['name']} ({var_name}):")
                print(f"     Shape: {data.shape}")
                print(f"     Range: [{min_val:.6e}, {max_val:.6e}]")
                print(f"     Mean: {mean_val:.6e}, Std: {std_val:.6e}")
                
                if issues:
                    for issue in issues:
                        print(f"     ⚠️  {issue}")
                else:
                    print(f"     ✅ 無品質問題")
            
            print("\n" + "="*70)
            print("📈 收斂性檢查:")
            print("-" * 70)
            
            # 檢查收斂歷史
            residuals_path = 'results/residuals/phase-1'
            res_group = f.get(residuals_path)
            if res_group is not None and isinstance(res_group, h5py.Group):
                convergence_issues = []
                
                for res_name in ['continuity', 'x-velocity', 'y-velocity', 'z-velocity', 'k', 'omega']:
                    if res_name in res_group:
                        res_data_dset = res_group[f'{res_name}/data']
                        res_data = np.array(res_data_dset)
                        final_res = res_data[-1, 0]
                        
                        # 收斂準則
                        if res_name == 'continuity':
                            threshold = 1e-6
                            target = 1e-8
                        elif res_name in ['k', 'omega']:
                            threshold = 1e-3
                            target = 1e-5
                        else:
                            threshold = 1e-2
                            target = 1e-4
                        
                        if final_res < target:
                            status = "✅ 優秀"
                        elif final_res < threshold:
                            status = "✅ 良好"
                        else:
                            status = "⚠️  可接受"
                            convergence_issues.append(f"{res_name}: {final_res:.2e}")
                        
                        print(f"  {res_name:15s}: {final_res:.6e} {status}")
                
                if convergence_issues:
                    warnings_found.extend(convergence_issues)
            
            print("\n" + "="*70)
            print("🎯 驗證總結:")
            print("-" * 70)
            
            if all_passed and not issues_found:
                print("  ✅ 所有檢查通過！數據品質優秀。")
                print()
                return True
            elif not issues_found:
                print("  ✅ 關鍵檢查通過，有以下警告：")
                for warning in warnings_found:
                    print(f"     ⚠️  {warning}")
                print()
                return True
            else:
                print("  ❌ 發現以下品質問題：")
                for issue in issues_found:
                    print(f"     - {issue}")
                print()
                print("  建議: 檢查 Fluent 模擬設置或網格品質")
                return False
                
    except Exception as e:
        print(f"❌ 讀取文件失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # V2 文件路徑
    v2_path = Path(__file__).parent.parent.parent / "data" / "lowfi" / "channel_fluent_raw" / "FFF-Setup-Output.dat_2.h5"
    
    if not v2_path.exists():
        print(f"❌ 文件不存在: {v2_path}")
        sys.exit(1)
    
    success = validate_fluent_data(v2_path)
    
    if success:
        print("✅ 驗證完成：數據品質符合要求")
        sys.exit(0)
    else:
        print("❌ 驗證失敗：數據品質存在問題")
        sys.exit(1)

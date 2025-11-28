#!/usr/bin/env python3
"""
Kolmogorov Flow DNS 模擬狀況監測工具

快速檢查 DNS 數據的完整性、物理量統計與模擬進度

使用方式:
    python scripts/monitor_kolmogorov_dns_status.py
    
    # 自動刷新
    watch -n 10 'python scripts/monitor_kolmogorov_dns_status.py'
"""

import h5py
import numpy as np
from pathlib import Path
from datetime import datetime


def check_dns_file(fpath, show_details=False):
    """檢查單個 DNS 文件的狀態"""
    if not Path(fpath).exists():
        return None
    
    result = {
        'filename': Path(fpath).name,
        'exists': True,
        'size_gb': Path(fpath).stat().st_size / 1024**3,
        'mtime': datetime.fromtimestamp(Path(fpath).stat().st_mtime)
    }
    
    try:
        with h5py.File(fpath, 'r') as f:
            # 配置參數
            if 'config' in f:
                cfg = f['config'].attrs
                result['N'] = cfg.get('N', 'N/A')
                result['kf'] = cfg.get('k_f', 'N/A')
                result['nu'] = cfg.get('nu', 'N/A')
                result['dt'] = cfg.get('dt', 'N/A')
                result['T_end'] = cfg.get('T_end', 100.0)
                
                # 計算雷諾數
                nu = cfg.get('nu')
                kf = cfg.get('k_f')
                A = cfg.get('A', 1.0)
                if nu and kf:
                    L = 2 * np.pi / kf
                    result['Re'] = np.sqrt(A) * L**(3/2) / nu
            
            # 時間資訊
            if 'time' in f:
                time_data = f['time']
                if isinstance(time_data, h5py.Dataset):
                    time = time_data[:]
                    result['n_snapshots'] = len(time)
                    result['t_min'] = float(time.min())
                    result['t_max'] = float(time.max())
                    result['dt_save'] = float(np.mean(np.diff(time)))
                    result['progress'] = (time.max() / result['T_end']) * 100
            
            # 動能統計
            if 'u' in f and 'v' in f:
                u_data = f['u']
                v_data = f['v']
                if isinstance(u_data, h5py.Dataset) and isinstance(v_data, h5py.Dataset):
                    u = u_data[:]
                    v = v_data[:]
                    ke = 0.5 * np.mean(u**2 + v**2, axis=(1,2))
                    
                    result['ke_mean'] = float(ke.mean())
                    result['ke_std'] = float(ke.std())
                    result['ke_min'] = float(ke.min())
                    result['ke_max'] = float(ke.max())
                    result['ke_cv'] = float(ke.std() / ke.mean())
                    
                    # 流動狀態判斷
                    cv = result['ke_cv']
                    if cv > 0.15:
                        result['flow_state'] = '🌀 強湍流'
                    elif cv > 0.08:
                        result['flow_state'] = '🔄 湍流'
                    elif cv > 0.03:
                        result['flow_state'] = '📊 過渡'
                    else:
                        result['flow_state'] = '➡️  層流/穩態'
            
            # 診斷資訊
            if 'diagnostics' in f:
                diag = f['diagnostics']
                if isinstance(diag, h5py.Group) and 'divergence_error' in diag:
                    div_err_data = diag['divergence_error']
                    if isinstance(div_err_data, h5py.Dataset):
                        div_err = div_err_data[:]
                        result['div_err_mean'] = float(div_err.mean())
                        result['div_err_max'] = float(div_err.max())
        
        result['status'] = 'success'
        
    except Exception as e:
        result['status'] = 'error'
        result['error_msg'] = str(e)
    
    return result


def print_summary(results):
    """打印摘要表格"""
    print("=" * 100)
    print("🌀 Kolmogorov Flow DNS 模擬狀況監測")
    print("=" * 100)
    print(f"檢測時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 表頭
    print(f"{'檔案':<40} {'Re':>6} {'k_f':>4} {'快照':>6} {'進度':>8} {'KE平均':>10} {'狀態':>15}")
    print("-" * 100)
    
    for r in results:
        if r is None:
            continue
        
        if r['status'] == 'error':
            print(f"{r['filename']:<40} ❌ 錯誤: {r.get('error_msg', 'Unknown')}")
            continue
        
        # 基本資訊
        re_str = f"{r['Re']:.0f}" if isinstance(r['Re'], (int, float)) else "N/A"
        kf_str = str(r['kf'])
        n_snap = r.get('n_snapshots', 0)
        progress = r.get('progress', 0)
        ke_mean = r.get('ke_mean', 0)
        state = r.get('flow_state', 'N/A')
        
        # 進度圖示
        if progress >= 99.9:
            prog_icon = "✅"
        elif progress > 50:
            prog_icon = "⏳"
        else:
            prog_icon = "🔄"
        
        print(f"{r['filename']:<40} {re_str:>6} {kf_str:>4} {n_snap:>6} "
              f"{prog_icon} {progress:>5.1f}% {ke_mean:>10.4f} {state:>15}")
    
    print("-" * 100)


def print_details(results):
    """打印詳細資訊"""
    for r in results:
        if r is None or r['status'] != 'success':
            continue
        
        print(f"\n{'=' * 80}")
        print(f"📁 {r['filename']}")
        print(f"{'=' * 80}")
        print(f"檔案大小: {r['size_gb']:.2f} GB")
        print(f"更新時間: {r['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n⚙️  配置:")
        print(f"  • 雷諾數 Re: {r['Re']:.1f}")
        print(f"  • 波數 k_f: {r['kf']}")
        print(f"  • 動力黏度 ν: {r['nu']}")
        print(f"  • 網格解析度: {r['N']}×{r['N']}")
        print(f"  • 時間步長 dt: {r['dt']}")
        
        print(f"\n⏱️  模擬進度:")
        print(f"  • 快照數: {r['n_snapshots']}")
        print(f"  • 時間範圍: t ∈ [{r['t_min']:.2f}, {r['t_max']:.2f}]")
        print(f"  • 儲存間隔: Δt = {r['dt_save']:.4f}")
        print(f"  • 完成度: {r['progress']:.1f}% {'✅' if r['progress'] >= 99.9 else '⏳'}")
        
        print(f"\n📈 動能統計:")
        print(f"  • KE 範圍: [{r['ke_min']:.6f}, {r['ke_max']:.6f}]")
        print(f"  • KE 平均: {r['ke_mean']:.6f} ± {r['ke_std']:.6f}")
        print(f"  • 變異係數: {r['ke_cv']:.3f}")
        print(f"  • 流動狀態: {r['flow_state']}")
        
        if 'div_err_mean' in r:
            print(f"\n✓ 數值品質:")
            print(f"  • 散度誤差平均: {r['div_err_mean']:.6e}")
            print(f"  • 散度誤差最大: {r['div_err_max']:.6e}")


def main():
    """主程式"""
    # DNS 文件列表
    dns_files = [
        'data/kolmogorov_dns/kolmogorov_re100_kf4_T100.h5',
        'data/kolmogorov_dns/kolmogorov_re500_kf4_T100.h5',
    ]
    
    # 檢查所有文件
    results = [check_dns_file(f) for f in dns_files]
    
    # 打印摘要
    print_summary(results)
    
    # 打印詳細資訊（可選）
    import sys
    if '--details' in sys.argv or '-d' in sys.argv:
        print_details(results)
    
    print("\n💡 提示:")
    print("  • 使用 --details 或 -d 顯示詳細資訊")
    print("  • 使用 watch -n 10 'python scripts/monitor_kolmogorov_dns_status.py' 自動刷新")
    print("=" * 100)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
快速訓練監測腳本
用法: python scripts/quick_monitor.py [檢查點目錄名稱]
範例: python scripts/quick_monitor.py kolmogorov_re100_kf8_K50_t20_2k_no_earlystop
"""

import torch
import argparse
from pathlib import Path
import time
import sys

def monitor_training(ckpt_dir_name):
    """監測指定檢查點目錄的訓練狀態"""
    
    ckpt_dir = Path('checkpoints') / ckpt_dir_name
    
    if not ckpt_dir.exists():
        print(f"❌ 檢查點目錄不存在: {ckpt_dir}")
        return
    
    # 找到最新的檢查點
    ckpt_files = list(ckpt_dir.glob('epoch_*.pth'))
    if not ckpt_files:
        print(f"❌ 目錄中沒有檢查點文件: {ckpt_dir}")
        return
    
    latest_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
    
    # 載入檢查點
    try:
        ckpt = torch.load(latest_ckpt, map_location='cpu')
    except Exception as e:
        print(f"❌ 無法載入檢查點: {e}")
        return
    
    history = ckpt.get('history', {})
    
    # 獲取時間信息
    mod_time = latest_ckpt.stat().st_mtime
    time_ago_sec = time.time() - mod_time
    time_ago_min = int(time_ago_sec / 60)
    
    # 打印報告
    print('=' * 80)
    print('🚀 訓練監測報告')
    print('=' * 80)
    print(f'\n📂 實驗: {ckpt_dir_name}')
    print(f'📄 最新檢查點: {latest_ckpt.name}')
    print(f'⏰ 更新時間: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))}')
    print(f'   (距今 {time_ago_min} 分鐘)')
    print()
    
    # 訓練進度
    epoch = ckpt['epoch']
    max_epochs = ckpt.get('config', {}).get('training', {}).get('max_epochs', 2000)
    progress = epoch / max_epochs * 100
    
    print(f'📊 訓練進度: Epoch {epoch}/{max_epochs} ({progress:.1f}% 完成)')
    
    # 檢查訓練是否還在進行
    if time_ago_min < 10:
        print(f'✅ 訓練正在進行中 (最近更新: {time_ago_min} 分鐘前)')
    elif time_ago_min < 30:
        print(f'⚠️  訓練可能暫停 (已 {time_ago_min} 分鐘未更新)')
    else:
        print(f'❌ 訓練可能已停止 (已 {time_ago_min} 分鐘未更新)')
    print()
    
    # 損失信息
    if 'total_loss' in history and len(history['total_loss']) > 0:
        total_losses = history['total_loss']
        current_loss = total_losses[-1]
        best_loss = min(total_losses)
        
        print('📈 損失統計:')
        print(f'   當前總損失: {current_loss:.6e}')
        print(f'   最佳總損失: {best_loss:.6e}')
        
        if len(total_losses) > 1:
            initial_loss = total_losses[0]
            improvement = (1 - current_loss/initial_loss) * 100
            print(f'   改善程度: {improvement:.1f}%')
        print()
        
        # 分量損失
        print('   分量損失:')
        if 'data_loss' in history and len(history['data_loss']) > 0:
            print(f'     資料損失: {history["data_loss"][-1]:.6e}')
        if 'pde_loss' in history and len(history['pde_loss']) > 0:
            print(f'     PDE 損失: {history["pde_loss"][-1]:.6e}')
        if 'continuity_loss' in history and len(history['continuity_loss']) > 0:
            print(f'     連續性: {history["continuity_loss"][-1]:.6e}')
        print()
    
    # 學習率
    if 'lr' in history and len(history['lr']) > 0:
        print(f'🎯 當前學習率: {history["lr"][-1]:.6e}')
        print()
    
    # 權重檢查
    has_nan = False
    has_inf = False
    model_state = ckpt.get('model_state_dict', {})
    for name, param in model_state.items():
        if torch.isnan(param).any():
            has_nan = True
        if torch.isinf(param).any():
            has_inf = True
    
    if has_nan or has_inf:
        print('❌ 警告: 模型權重包含 NaN 或 Inf!')
    else:
        print('✅ 模型權重健康')
    
    print('=' * 80)


def main():
    parser = argparse.ArgumentParser(description='快速訓練監測工具')
    parser.add_argument('exp_name', nargs='?', 
                       default='kolmogorov_re100_kf8_K50_t20_2k_no_earlystop',
                       help='實驗名稱（檢查點目錄名）')
    
    args = parser.parse_args()
    
    monitor_training(args.exp_name)


if __name__ == '__main__':
    main()

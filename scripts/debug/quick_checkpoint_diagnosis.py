"""
快速檢查點診斷工具
檢查模型權重是否包含 NaN/Inf，並評估基本物理一致性
"""

import torch
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def diagnose_checkpoint(checkpoint_path):
    """診斷檢查點狀態"""
    print(f"📂 載入檢查點: {checkpoint_path}")
    print("=" * 80)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 1. 基本資訊
        print(f"\n✅ 檢查點已載入")
        print(f"📊 Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"📉 Loss: {checkpoint.get('loss', 'N/A'):.6f}")
        
        # 2. 檢查模型權重
        print(f"\n🔍 模型權重診斷:")
        model_state = checkpoint['model_state_dict']
        
        total_params = 0
        nan_params = 0
        inf_params = 0
        
        for name, param in model_state.items():
            total_params += param.numel()
            
            if torch.isnan(param).any():
                nan_count = torch.isnan(param).sum().item()
                nan_params += nan_count
                print(f"  ❌ NaN 發現: {name} ({nan_count}/{param.numel()})")
            
            if torch.isinf(param).any():
                inf_count = torch.isinf(param).sum().item()
                inf_params += inf_count
                print(f"  ❌ Inf 發現: {name} ({inf_count}/{param.numel()})")
        
        print(f"\n📈 總參數數量: {total_params:,}")
        print(f"  NaN 數量: {nan_params}")
        print(f"  Inf 數量: {inf_params}")
        
        if nan_params == 0 and inf_params == 0:
            print(f"  ✅ 所有權重正常")
            health_status = "HEALTHY"
        else:
            print(f"  ⚠️ 權重包含異常值")
            health_status = "CORRUPTED"
        
        # 3. 權重統計
        print(f"\n📊 權重統計:")
        all_weights = []
        for name, param in model_state.items():
            if 'weight' in name:
                all_weights.append(param.flatten())
        
        if all_weights:
            all_weights = torch.cat(all_weights)
            print(f"  Mean: {all_weights.mean().item():.6f}")
            print(f"  Std: {all_weights.std().item():.6f}")
            print(f"  Min: {all_weights.min().item():.6f}")
            print(f"  Max: {all_weights.max().item():.6f}")
        
        # 4. 優化器狀態
        if 'optimizer_state_dict' in checkpoint:
            print(f"\n🎯 優化器狀態:")
            opt_state = checkpoint['optimizer_state_dict']
            
            if 'state' in opt_state and len(opt_state['state']) > 0:
                # 檢查第一個參數的狀態
                first_state = list(opt_state['state'].values())[0]
                
                if 'exp_avg' in first_state:
                    exp_avg = first_state['exp_avg']
                    print(f"  Momentum (exp_avg):")
                    print(f"    Mean: {exp_avg.mean().item():.6e}")
                    print(f"    Std: {exp_avg.std().item():.6e}")
                    
                    if torch.isnan(exp_avg).any() or torch.isinf(exp_avg).any():
                        print(f"    ❌ Optimizer state 包含異常值")
                        health_status = "CORRUPTED"
        
        # 5. 訓練歷史
        if 'loss_history' in checkpoint:
            print(f"\n📉 損失歷史（最近 10 個 epoch）:")
            loss_hist = checkpoint['loss_history'][-10:]
            for i, loss in enumerate(loss_hist):
                epoch = checkpoint['epoch'] - len(loss_hist) + i + 1
                print(f"  Epoch {epoch}: {loss:.6f}")
        
        print("\n" + "=" * 80)
        print(f"🏥 健康狀態: {health_status}")
        print("=" * 80)
        
        return {
            'health_status': health_status,
            'nan_count': nan_params,
            'inf_count': inf_params,
            'total_params': total_params,
            'epoch': checkpoint.get('epoch', -1),
            'loss': checkpoint.get('loss', float('inf'))
        }
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    result = diagnose_checkpoint(args.checkpoint)
    
    if result:
        sys.exit(0 if result['health_status'] == 'HEALTHY' else 1)
    else:
        sys.exit(2)

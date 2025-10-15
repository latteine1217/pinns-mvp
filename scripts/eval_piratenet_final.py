"""
PirateNet 訓練後評估腳本
簡化版本 - 專注於物理殘差評估
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import sys

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='PirateNet 評估')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    # 設定設備
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 載入檢查點
    print(f"\n載入檢查點: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 提取訓練資訊
    print("\n" + "="*70)
    print("📊 訓練總結")
    print("="*70)
    print(f"訓練輪數: {checkpoint.get('epoch', 'N/A')}")
    print(f"最終損失: {checkpoint.get('loss', 0):.2f}")
    
    # 提取配置
    if 'config' in checkpoint:
        cfg = checkpoint['config']
    else:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    
    # 打印關鍵配置資訊
    print("\n" + "="*70)
    print("⚙️  訓練配置")
    print("="*70)
    
    # 模型配置
    if 'model' in cfg:
        model_cfg = cfg['model']
        print(f"\n模型架構: {model_cfg.get('type', 'N/A')}")
        print(f"隱藏層維度: {model_cfg.get('hidden_dim', 'N/A')}")
        print(f"隱藏層數量: {model_cfg.get('num_layers', 'N/A')}")
        print(f"激活函數: {model_cfg.get('activation', 'N/A')}")
        print(f"Fourier 特徵: {model_cfg.get('use_fourier', False)}")
        if model_cfg.get('use_fourier'):
            print(f"  - Fourier M: {model_cfg.get('fourier_m', 'N/A')}")
            print(f"  - Fourier σ: {model_cfg.get('fourier_sigma', 'N/A')}")
        print(f"RWF: {model_cfg.get('use_rwf', False)}")
    
    # 優化器配置
    if 'optimizer' in cfg:
        opt_cfg = cfg['optimizer']
        print(f"\n優化器: {opt_cfg.get('type', 'N/A')}")
        print(f"學習率: {opt_cfg.get('lr', 'N/A')}")
    
    # 物理配置
    if 'physics' in cfg:
        phys_cfg = cfg['physics']
        print(f"\n物理模型: VS-PINN Channel Flow")
        if 'vs_pinn' in phys_cfg:
            vs_cfg = phys_cfg['vs_pinn']
            print(f"縮放因子: N_x={vs_cfg.get('N_x')}, N_y={vs_cfg.get('N_y')}, N_z={vs_cfg.get('N_z')}")
        print(f"黏性係數 (ν): {phys_cfg.get('nu', 5e-5)}")
        print(f"壓力梯度 (dP/dx): {phys_cfg.get('dpdx', 0.0025)}")
    
    # 數據配置
    if 'data' in cfg:
        data_cfg = cfg['data']
        print(f"\n數據來源: {data_cfg.get('source', 'N/A')}")
        if 'jhtdb_config' in data_cfg:
            jhtdb_cfg = data_cfg['jhtdb_config']
            print(f"感測點: {jhtdb_cfg.get('sensor_file', 'N/A')}")
            n_sensors = jhtdb_cfg.get('n_sensors', 'N/A')
            print(f"感測點數量: {n_sensors}")
            print(f"配準點數量: {data_cfg.get('n_collocation', 'N/A')}")
    
    # 損失權重歷史（如果有）
    if 'loss_history' in checkpoint:
        loss_hist = checkpoint['loss_history']
        print("\n" + "="*70)
        print("📈 損失歷史（最後 10 epochs）")
        print("="*70)
        
        epochs_to_show = min(10, len(loss_hist))
        if epochs_to_show > 0:
            print(f"\n{'Epoch':>6} {'Total':>12} {'Data':>12} {'PDE':>12} {'Wall':>12}")
            print("-"*60)
            for i in range(-epochs_to_show, 0):
                entry = loss_hist[i]
                epoch = entry.get('epoch', i)
                print(f"{epoch:>6} {entry.get('total', 0):>12.2e} "
                      f"{entry.get('data', 0):>12.2e} "
                      f"{entry.get('pde', 0):>12.2e} "
                      f"{entry.get('wall', 0):>12.2e}")
    
    # 總結
    print("\n" + "="*70)
    print("✅ 評估完成")
    print("="*70)
    print("\n關鍵觀察:")
    
    if 'loss_history' in checkpoint and len(checkpoint['loss_history']) > 1:
        initial_loss = checkpoint['loss_history'][0].get('total', 0)
        final_loss = checkpoint['loss_history'][-1].get('total', 0)
        reduction = (1 - final_loss / initial_loss) * 100 if initial_loss > 0 else 0
        print(f"1. 損失降低: {reduction:.1f}% (從 {initial_loss:.2e} 到 {final_loss:.2e})")
        
        # 分析各項損失
        final = checkpoint['loss_history'][-1]
        print(f"2. 最終資料損失: {final.get('data', 0):.2e}")
        print(f"3. 最終 PDE 殘差: {final.get('pde', 0):.2e}")
        print(f"4. 最終壁面損失: {final.get('wall', 0):.2e}")
    
    print(f"\n下一步建議:")
    print(f"- 使用 scripts/visualize_results.py 視覺化預測場")
    print(f"- 與基準場比較以計算誤差指標")
    print(f"- 進行參數敏感度分析")
    
    # 保存簡要報告
    output_dir = Path('results/piratenet_quick_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / 'evaluation_summary.txt'
    with open(report_file, 'w') as f:
        f.write("PirateNet Training Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Epochs: {checkpoint.get('epoch', 'N/A')}\n")
        f.write(f"Final Loss: {checkpoint.get('loss', 0):.6e}\n")
        if 'loss_history' in checkpoint and len(checkpoint['loss_history']) > 0:
            final = checkpoint['loss_history'][-1]
            f.write(f"\nFinal Metrics:\n")
            f.write(f"  Data Loss: {final.get('data', 0):.6e}\n")
            f.write(f"  PDE Residual: {final.get('pde', 0):.6e}\n")
            f.write(f"  Wall Loss: {final.get('wall', 0):.6e}\n")
    
    print(f"\n💾 報告已保存: {report_file}")

if __name__ == '__main__':
    main()

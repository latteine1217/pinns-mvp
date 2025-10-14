"""
臨時修復腳本：為舊檢查點添加 physics_state_dict

用法：
python scripts/fix_legacy_checkpoint.py \
    --checkpoint checkpoints/fourier_annealing_longterm_k500/epoch_3881.pth \
    --config configs/fourier_annealing_longterm_k500.yml
"""
import torch
import yaml
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.train.factory import create_physics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    # 載入配置
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # 載入檢查點
    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # 檢查是否已有 physics_state_dict
    if 'physics_state_dict' in checkpoint:
        print(f"✅ Checkpoint already has physics_state_dict: {list(checkpoint['physics_state_dict'].keys())}")
        return
    
    # 創建 physics 對象（從配置文件）
    device = torch.device('cpu')
    physics = create_physics(config, device)
    
    # 添加 physics_state_dict
    checkpoint['physics_state_dict'] = physics.state_dict()
    print(f"🆕 Added physics_state_dict: {list(physics.state_dict().keys())}")
    
    # 打印縮放參數
    if hasattr(physics, 'N_x'):
        print(f"   VS-PINN 縮放參數: N_x={physics.N_x.item():.2f}, "
              f"N_y={physics.N_y.item():.2f}, N_z={physics.N_z.item():.2f}")
    
    # 保存為新檢查點
    new_ckpt_path = ckpt_path.parent / f"{ckpt_path.stem}_fixed.pth"
    torch.save(checkpoint, new_ckpt_path)
    print(f"💾 Saved fixed checkpoint to: {new_ckpt_path}")
    print(f"   Original: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.2f} MB)")
    print(f"   Fixed:    {new_ckpt_path} ({new_ckpt_path.stat().st_size / 1e6:.2f} MB)")

if __name__ == '__main__':
    main()

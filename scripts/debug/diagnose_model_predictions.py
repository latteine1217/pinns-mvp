"""診斷模型預測的物理合理性"""
import torch
import yaml
import sys
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pinnx.models.fourier_mlp import create_enhanced_fourier_mlp
from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow, create_vs_pinn_channel_flow

def load_model(checkpoint_path, config_path, device):
    """載入模型"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 創建模型
    model_config = config['model']
    model = create_enhanced_fourier_mlp(
        N_x=model_config['vs_pinn']['scaling_factors']['N_x'],
        N_y=model_config['vs_pinn']['scaling_factors']['N_y'],
        N_z=model_config['vs_pinn']['scaling_factors']['N_z'],
        **model_config
    )
    model = model.to(device)
    
    # 載入權重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def test_model(model, device):
    """測試模型在典型點的預測"""
    print("=" * 80)
    print("模型診斷：單點預測測試")
    print("=" * 80)
    
    # 測試點：通道中心
    test_points = torch.tensor([
        [12.565, 0.0, 4.71],   # 中心點
        [12.565, -1.0, 4.71],  # 下壁面
        [12.565, 1.0, 4.71],   # 上壁面
        [0.0, 0.0, 0.0],       # 原點
    ], device=device, dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(test_points)
    
    locations = ['通道中心', '下壁面', '上壁面', '原點']
    
    for i, (loc, pred) in enumerate(zip(locations, predictions)):
        print(f"\n位置: {loc}")
        print(f"  坐標: x={test_points[i,0]:.3f}, y={test_points[i,1]:.3f}, z={test_points[i,2]:.3f}")
        print(f"  預測: u={pred[0]:.6f}, v={pred[1]:.6f}, w={pred[2]:.6f}, p={pred[3]:.6f}")
    
    # 檢查輸入範圍
    print("\n" + "=" * 80)
    print("檢查輸入範圍")
    print("=" * 80)
    print(f"預期域: x=[0, 25.13], y=[-1, 1], z=[0, 9.42]")
    
    # 檢查輸出統計
    print("\n" + "=" * 80)
    print("預測值統計")
    print("=" * 80)
    print(f"u: min={predictions[:,0].min():.6f}, max={predictions[:,0].max():.6f}, mean={predictions[:,0].mean():.6f}")
    print(f"v: min={predictions[:,1].min():.6f}, max={predictions[:,1].max():.6f}, mean={predictions[:,1].mean():.6f}")
    print(f"w: min={predictions[:,2].min():.6f}, max={predictions[:,2].max():.6f}, mean={predictions[:,2].mean():.6f}")
    print(f"p: min={predictions[:,3].min():.6f}, max={predictions[:,3].max():.6f}, mean={predictions[:,3].mean():.6f}")
    
    # 物理合理性檢查
    print("\n" + "=" * 80)
    print("物理合理性檢查")
    print("=" * 80)
    
    # 通道流預期值（Re_tau=1000）
    u_expected_center = 1.13  # 中心線速度
    u_expected_wall = 0.0     # 壁面速度
    
    u_center = predictions[0, 0].item()
    u_wall = predictions[1, 0].item()
    
    print(f"✓ 中心線流向速度: 預測={u_center:.4f}, 預期≈{u_expected_center:.4f}")
    print(f"✓ 壁面流向速度: 預測={u_wall:.4f}, 預期≈{u_expected_wall:.4f}")
    
    if u_center < 0:
        print("❌ 警告：中心線流向速度為負值！")
    if abs(u_wall) > 0.01:
        print("❌ 警告：壁面速度不為零！")
    if predictions[:, 1].abs().mean() > 0.1:
        print("❌ 警告：壁法向速度過大！")

if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    checkpoint_path = "checkpoints/test_enhanced_5k_curriculum/best_model.pth"
    config_path = "configs/test_enhanced_5k_curriculum.yml"
    
    model, config = load_model(checkpoint_path, config_path, device)
    test_model(model, device)
    
    print("\n" + "=" * 80)
    print("診斷完成")
    print("=" * 80)

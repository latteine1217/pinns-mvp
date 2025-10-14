"""快速檢查模型預測的物理合理性"""
import torch
import yaml

# 載入檢查點
checkpoint_path = "checkpoints/test_enhanced_5k_curriculum/best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 載入配置
with open("configs/test_enhanced_5k_curriculum.yml") as f:
    config = yaml.safe_load(f)

print("=" * 80)
print("檢查點資訊")
print("=" * 80)
print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Total Loss: {checkpoint.get('total_loss', 'N/A'):.6f}")
print(f"Data Loss: {checkpoint.get('data_loss', 'N/A'):.6f}")
print(f"PDE Loss: {checkpoint.get('pde_loss', 'N/A'):.6f}")

# 檢查模型權重統計
state_dict = checkpoint['model_state_dict']
print("\n" + "=" * 80)
print("模型權重統計")
print("=" * 80)

for name, param in list(state_dict.items())[:5]:
    print(f"{name}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")

# 檢查最後一層（輸出層）
output_layer_weight = None
output_layer_bias = None

for name, param in state_dict.items():
    if 'output' in name or 'final' in name or name.endswith('.weight'):
        if param.shape[0] == 4:  # 輸出維度為4 (u,v,w,p)
            output_layer_weight = param
            print(f"\n輸出層發現: {name}")
            print(f"  Shape: {param.shape}")
            print(f"  Mean: {param.mean():.6f}")
            print(f"  Std: {param.std():.6f}")
    if 'output' in name or 'final' in name or name.endswith('.bias'):
        if param.shape[0] == 4:
            output_layer_bias = param
            print(f"\n輸出偏置發現: {name}")
            print(f"  Value: {param.detach().numpy()}")

print("\n" + "=" * 80)
print("訓練配置")
print("=" * 80)
print(f"感測點數量 (K): {config['data'].get('n_sensors', 'N/A')}")
print(f"數據損失權重: {config['losses']['weights'].get('data', 'N/A')}")
print(f"PDE 損失權重: {config['losses']['weights'].get('pde', 'N/A')}")
print(f"VS-PINN 縮放: N_x={config['model']['vs_pinn']['scaling_factors']['N_x']}, "
      f"N_y={config['model']['vs_pinn']['scaling_factors']['N_y']}, "
      f"N_z={config['model']['vs_pinn']['scaling_factors']['N_z']}")

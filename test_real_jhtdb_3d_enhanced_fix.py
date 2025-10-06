#!/usr/bin/env python3
"""
Task-014: 3D 升級版真實JHTDB數據與PINNs訓練管線整合測試 - 修復版
=================================================================

修復策略：
1. V場權重失衡問題 - 基於場統計特性的尺度權重
2. 完整3D NS動量方程實現
3. 增強壁面邊界條件處理

目標：將重建精度從115.5%降至<30%
"""

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def load_real_jhtdb_3d_data():
    """載入完整3D真實JHTDB Channel Flow數據"""
    print("📁 載入完整3D真實JHTDB Channel Flow數據...")
    
    cache_file = Path("data/jhtdb/channel_34e525c703a89036170603d28e552870.h5")
    if not cache_file.exists():
        raise FileNotFoundError(f"真實JHTDB數據快取不存在: {cache_file}")
    
    with h5py.File(cache_file, 'r') as f:
        print(f"   數據集可用鍵: {list(f.keys())}")
        
        # 讀取3D坐標與4D場數據 (u,v,w,p) - 修復類型處理
        coords_data = f['coordinates']
        velocity_data = f['velocity']
        pressure_data = f['pressure']
        
        # 轉換為numpy數組
        coords = np.array(coords_data)     # [N, 4] - (t,x,y,z)
        velocity = np.array(velocity_data)      # [N, 3] - (u,v,w)
        pressure = np.array(pressure_data)      # [N, 1] - (p)
        
        # 確保正確的形狀
        if pressure.ndim == 1:
            pressure = pressure.reshape(-1, 1)
        
        # 合併為完整狀態向量
        values = np.concatenate([velocity, pressure], axis=1)  # [N, 4]
        
        print(f"   座標維度: {coords.shape} (t,x,y,z)")
        print(f"   狀態維度: {values.shape} (u,v,w,p)")
        print(f"   座標範圍: t∈[{coords[:,0].min():.3f},{coords[:,0].max():.3f}]")
        print(f"              x∈[{coords[:,1].min():.3f},{coords[:,1].max():.3f}]")
        print(f"              y∈[{coords[:,2].min():.3f},{coords[:,2].max():.3f}]")
        print(f"              z∈[{coords[:,3].min():.3f},{coords[:,3].max():.3f}]")
        
        return coords, values

class Enhanced3DPINNNet(torch.nn.Module):
    """增強版3D PINNs網路 - 針對精度優化"""
    
    def __init__(self, layers=[4, 256, 256, 256, 256, 256, 256, 256, 256, 4], 
                 fourier_features=None, activation='swish'):
        super().__init__()
        
        self.layers = layers
        self.activation_name = activation
        
        # 設定激活函數
        if activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        elif activation == 'gelu':
            self.activation = torch.nn.functional.gelu
        else:
            self.activation = torch.tanh
            
        # Fourier特徵編碼（可選）
        self.fourier_features = fourier_features
        if fourier_features:
            self.B = torch.randn(fourier_features, 4) * 2.0  # 4D輸入
            layers[0] = fourier_features * 2
        
        # 構建層
        self.linears = torch.nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(torch.nn.Linear(layers[i], layers[i+1]))
            
        # 權重初始化
        self._initialize_weights()
        
        print(f"   🔧 增強3D網路: {layers}, 激活={activation}")
        print(f"   📊 參數總數: {sum(p.numel() for p in self.parameters()):,}")
    
    def _initialize_weights(self):
        """Xavier初始化權重"""
        for m in self.linears:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """前向傳播"""
        # Fourier特徵編碼
        if self.fourier_features:
            if not hasattr(self, 'B'):
                self.B = self.B.to(x.device)
            x_proj = 2 * torch.pi * x @ self.B.T
            x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
        # 深度前向傳播
        for i, layer in enumerate(self.linears[:-1]):
            x = layer(x)
            x = self.activation(x)
            
        # 輸出層（線性）
        x = self.linears[-1](x)
        return x

def compute_3d_ns_residual(model, physics_coords, Re=1000, density=1.0):
    """
    計算完整3D Navier-Stokes方程殘差
    包含：連續性方程 + 3個動量方程
    """
    physics_coords.requires_grad_(True)
    pred = model(physics_coords)
    
    # 提取各分量
    u = pred[:, 0:1]
    v = pred[:, 1:2]  
    w = pred[:, 2:3]
    p = pred[:, 3:4]
    
    # 計算所有梯度
    u_grads = torch.autograd.grad(u.sum(), physics_coords, create_graph=True, retain_graph=True)[0]
    v_grads = torch.autograd.grad(v.sum(), physics_coords, create_graph=True, retain_graph=True)[0]
    w_grads = torch.autograd.grad(w.sum(), physics_coords, create_graph=True, retain_graph=True)[0]
    p_grads = torch.autograd.grad(p.sum(), physics_coords, create_graph=True, retain_graph=True)[0]
    
    # 一階導數 [t,x,y,z]索引=[0,1,2,3]
    u_t, u_x, u_y, u_z = u_grads[:,0:1], u_grads[:,1:2], u_grads[:,2:3], u_grads[:,3:4]
    v_t, v_x, v_y, v_z = v_grads[:,0:1], v_grads[:,1:2], v_grads[:,2:3], v_grads[:,3:4]
    w_t, w_x, w_y, w_z = w_grads[:,0:1], w_grads[:,1:2], w_grads[:,2:3], w_grads[:,3:4]
    p_x, p_y, p_z = p_grads[:,1:2], p_grads[:,2:3], p_grads[:,3:4]
    
    # 二階導數（for 拉普拉斯項）
    u_xx = torch.autograd.grad(u_x.sum(), physics_coords, create_graph=True, retain_graph=True)[0][:,1:2]
    u_yy = torch.autograd.grad(u_y.sum(), physics_coords, create_graph=True, retain_graph=True)[0][:,2:3]
    u_zz = torch.autograd.grad(u_z.sum(), physics_coords, create_graph=True, retain_graph=True)[0][:,3:4]
    
    v_xx = torch.autograd.grad(v_x.sum(), physics_coords, create_graph=True, retain_graph=True)[0][:,1:2]
    v_yy = torch.autograd.grad(v_y.sum(), physics_coords, create_graph=True, retain_graph=True)[0][:,2:3]
    v_zz = torch.autograd.grad(v_z.sum(), physics_coords, create_graph=True, retain_graph=True)[0][:,3:4]
    
    w_xx = torch.autograd.grad(w_x.sum(), physics_coords, create_graph=True, retain_graph=True)[0][:,1:2]
    w_yy = torch.autograd.grad(w_y.sum(), physics_coords, create_graph=True, retain_graph=True)[0][:,2:3]
    w_zz = torch.autograd.grad(w_z.sum(), physics_coords, create_graph=True, retain_graph=True)[0][:,3:4]
    
    # 連續性方程：∇·u = 0
    continuity = u_x + v_y + w_z
    
    # 動量方程 (不含體積力):
    # ∂u/∂t + u∇u = -∇p/ρ + ν∇²u
    nu = 1.0 / Re  # 運動粘度
    
    # X動量方程
    momentum_x = (u_t + u*u_x + v*u_y + w*u_z + p_x/density - 
                  nu*(u_xx + u_yy + u_zz))
    
    # Y動量方程
    momentum_y = (v_t + u*v_x + v*v_y + w*v_z + p_y/density - 
                  nu*(v_xx + v_yy + v_zz))
    
    # Z動量方程  
    momentum_z = (w_t + u*w_x + v*w_y + w*w_z + p_z/density - 
                  nu*(w_xx + w_yy + w_zz))
    
    return {
        'continuity': continuity,
        'momentum_x': momentum_x,
        'momentum_y': momentum_y,
        'momentum_z': momentum_z
    }

def apply_qr_pivot_selection_3d(coords, values, n_sensors=16, random_seed=42):
    """3D QR-pivot感測點選擇"""
    print(f"🎯 使用3D QR-pivot選擇 {n_sensors} 個最優感測點...")
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # 空間坐標構建快照矩陣 (排除時間維度)
    spatial_coords = coords[:, 1:]  # [x,y,z]
    
    # 構建基於值的快照矩陣
    snapshot_matrix = values.T  # [4, N] - (u,v,w,p)的轉置
    
    # QR分解選擇主要點
    U, sigma, Vt = np.linalg.svd(snapshot_matrix, full_matrices=False)
    
    # 基於奇異值的重要度選擇
    importance = np.sum(Vt[:n_sensors]**2, axis=0)
    selected_indices = np.argsort(importance)[-n_sensors:]
    
    selected_coords = coords[selected_indices]
    selected_values = values[selected_indices]
    
    print(f"   ✅ 已選擇感測點，覆蓋空間範圍：")
    print(f"      X: [{selected_coords[:,1].min():.3f}, {selected_coords[:,1].max():.3f}]")
    print(f"      Y: [{selected_coords[:,2].min():.3f}, {selected_coords[:,2].max():.3f}]")
    print(f"      Z: [{selected_coords[:,3].min():.3f}, {selected_coords[:,3].max():.3f}]")
    
    return selected_coords, selected_values, selected_indices

def train_enhanced_3d_pinns():
    """主訓練函數 - 採用尺度平衡策略"""
    print("🚀 启动增強版3D PINNs訓練...")
    
    # 載入數據
    coords, values = load_real_jhtdb_3d_data()
    
    # QR-pivot感測點選擇
    sensor_coords, sensor_values, sensor_indices = apply_qr_pivot_selection_3d(
        coords, values, n_sensors=16, random_seed=42
    )
    
    # 轉換為tensor
    coords_tensor = torch.FloatTensor(sensor_coords)
    values_tensor = torch.FloatTensor(sensor_values)
    
    # 物理約束點（使用全數據）
    physics_tensor = torch.FloatTensor(coords)
    
    # === 分析數據統計特性用於權重平衡 ===
    print("🔍 分析場統計特性...")
    values_np = values_tensor.detach().cpu().numpy()
    field_stats = {}
    field_names = ['U', 'V', 'W', 'P']
    
    for i, name in enumerate(field_names):
        std_val = np.std(values_np[:, i])
        mean_val = np.mean(values_np[:, i])
        field_stats[name] = {
            'std': std_val,
            'mean': mean_val,
            'range': [np.min(values_np[:, i]), np.max(values_np[:, i])]
        }
        print(f"{name}場: std={std_val:.6f}, mean={mean_val:.6f}, range={field_stats[name]['range']}")
    
    # 計算尺度平衡權重 - 基於標準差倒數
    base_std = field_stats['U']['std']  # 以U場為基準
    scale_weights = {}
    for i, name in enumerate(field_names):
        if field_stats[name]['std'] > 1e-8:
            scale_weights[name] = base_std / field_stats[name]['std']
        else:
            scale_weights[name] = 1.0
        print(f"{name}場尺度權重: {scale_weights[name]:.4f}")
    
    # 初始化增強網路
    model = Enhanced3DPINNNet(
        layers=[4, 256, 256, 256, 256, 256, 256, 256, 256, 4],
        fourier_features=64,
        activation='swish'
    )
    
    # 優化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.8)
    
    # 訓練記錄
    losses = []
    field_losses_history = {name: [] for name in field_names}
    
    print("🔄 開始3D增強訓練迴圈...")
    for epoch in range(300):  # 增加epoch數
        optimizer.zero_grad()
        
        # === 數據損失（監督學習）- 採用尺度平衡 ===
        pred_data = model(coords_tensor)
        
        # 分別計算各分量損失並應用尺度權重
        field_losses = []
        for i, name in enumerate(field_names):
            field_pred = pred_data[:, i:i+1]
            field_true = values_tensor[:, i:i+1]
            field_loss = torch.nn.MSELoss()(field_pred, field_true)
            weighted_loss = scale_weights[name] * field_loss
            field_losses.append(weighted_loss)
            field_losses_history[name].append(field_loss.item())
        
        data_loss = sum(field_losses) / len(field_losses)  # 平均加權損失
        
        # === 完整3D NS物理約束損失 ===
        residuals = compute_3d_ns_residual(model, physics_tensor, Re=1000)
        
        # 各方程殘差的MSE
        continuity_loss = torch.mean(residuals['continuity']**2)
        momentum_x_loss = torch.mean(residuals['momentum_x']**2)
        momentum_y_loss = torch.mean(residuals['momentum_y']**2)
        momentum_z_loss = torch.mean(residuals['momentum_z']**2)
        
        # 總物理損失
        physics_loss = (continuity_loss + momentum_x_loss + 
                       momentum_y_loss + momentum_z_loss) / 4.0
        
        # === 邊界條件約束 ===
        # 壁面邊界：y=±1處速度為零
        wall_mask = torch.abs(physics_tensor[:, 2] - 1.0) < 0.01  # y=1壁面
        if wall_mask.sum() > 0:
            wall_pred = model(physics_tensor[wall_mask])
            wall_bc_loss = torch.mean(wall_pred[:, :3]**2)  # u=v=w=0
        else:
            wall_bc_loss = torch.tensor(0.0)
        
        # === 總損失 ===
        lambda_data = 10.0      # 數據項權重
        lambda_physics = 2.0    # 增強物理項權重
        lambda_bc = 5.0         # 邊界條件權重
        
        total_loss = (lambda_data * data_loss + 
                     lambda_physics * physics_loss + 
                     lambda_bc * wall_bc_loss)
        
        # 正則化項
        reg_loss = 1e-6 * sum(torch.norm(param)**2 for param in model.parameters())
        total_loss += reg_loss
        
        total_loss.backward()
        
        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(total_loss)
        
        # 記錄損失
        losses.append(total_loss.item())
        
        # 訓練進度報告
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: 總損失={total_loss.item():.6f}, "
                  f"數據={data_loss.item():.6f}, 物理={physics_loss.item():.6f}, "
                  f"邊界={wall_bc_loss.item():.6f}")
            
            # 打印各場分量損失
            for i, name in enumerate(field_names):
                print(f"   {name}場損失: {field_losses_history[name][-1]:.6f}")
    
    print("✅ 3D增強訓練完成!")
    
    # 評估性能
    evaluate_enhanced_3d_performance(model, coords, values, sensor_indices)
    
    return model, losses, field_losses_history

def evaluate_enhanced_3d_performance(model, coords, values, sensor_indices):
    """評估3D增強模型性能"""
    print("\n🔍 評估3D增強模型性能...")
    
    # 全域預測
    coords_tensor = torch.FloatTensor(coords)
    values_tensor = torch.FloatTensor(values)
    
    with torch.no_grad():
        pred_full = model(coords_tensor).numpy()
    
    # 計算各分量誤差
    field_names = ['U', 'V', 'W', 'P']
    errors = {}
    
    print("=== 分量別重建誤差 ===")
    for i, name in enumerate(field_names):
        true_field = values[:, i]
        pred_field = pred_full[:, i]
        
        l2_error = np.sqrt(np.mean((pred_field - true_field)**2))
        rel_error = l2_error / (np.sqrt(np.mean(true_field**2)) + 1e-10)
        
        errors[name] = {
            'l2': l2_error,
            'relative': rel_error * 100
        }
        
        print(f"{name}場: L2={l2_error:.6f}, 相對誤差={rel_error*100:.1f}%")
    
    # 平均相對誤差
    avg_rel_error = np.mean([errors[name]['relative'] for name in field_names])
    print(f"\n📊 平均相對誤差: {avg_rel_error:.1f}%")
    
    if avg_rel_error < 30.0:
        print("🎉 成功！已達到<30%目標精度！")
    else:
        print(f"⚠️  尚未達標，需進一步優化 (目標<30%)")
    
    # 繪製結果比較
    plot_3d_results_comparison(coords, values, pred_full, sensor_indices, errors)
    
    return errors

def plot_3d_results_comparison(coords, values, pred_values, sensor_indices, errors):
    """繪製3D結果比較圖"""
    print("📊 生成3D結果比較圖...")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    field_names = ['U', 'V', 'W', 'P']
    
    for i, name in enumerate(field_names):
        # 真實場
        ax1 = axes[0, i]
        scatter1 = ax1.scatter(coords[:, 1], coords[:, 2], c=values[:, i], 
                              cmap='RdBu_r', s=1, alpha=0.6)
        ax1.scatter(coords[sensor_indices, 1], coords[sensor_indices, 2], 
                   c='black', s=30, marker='x', alpha=0.8, label='Sensors')
        ax1.set_title(f'True {name} Field')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(scatter1, ax=ax1)
        
        # 預測場
        ax2 = axes[1, i]
        scatter2 = ax2.scatter(coords[:, 1], coords[:, 2], c=pred_values[:, i], 
                              cmap='RdBu_r', s=1, alpha=0.6)
        ax2.set_title(f'Pred {name} (Err: {errors[name]["relative"]:.1f}%)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('3d_enhanced_pinns_results.png', dpi=150, bbox_inches='tight')
    print("   💾 結果已保存: 3d_enhanced_pinns_results.png")
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 Task-014: 3D Enhanced PINNs Training - 修復版")
    print("=" * 60)
    
    try:
        model, losses, field_losses_history = train_enhanced_3d_pinns()
        print("\n🎉 Task-014 3D增強訓練完成!")
        
    except Exception as e:
        print(f"\n❌ 訓練過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()
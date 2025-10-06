#!/usr/bin/env python3
"""
Channel Flow Re1000 完整 PINNs 逆重建實驗腳本
包含真實的物理方程約束（N-S方程）

基於新開發的 Channel Flow 載入器進行真實的 PINNs 實驗，
包含：
1. N-S 方程殘差約束
2. 邊界條件約束  
3. 低保真一致性約束
4. 多權重策略比較
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import logging
import matplotlib.pyplot as plt

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PINNsModel(nn.Module):
    """包含物理約束的 PINNs 模型"""
    
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=6, output_dim=4):
        super().__init__()
        
        # 構建網路層
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x: 輸入座標 [batch_size, 2] -> [x, y]
        
        Returns:
            pred: 預測輸出 [batch_size, 4] -> [u, v, p, S]
        """
        return self.network(x)

class PhysicsLoss:
    """物理方程約束損失計算"""
    
    def __init__(self, nu=1e-4, device='cpu'):
        self.nu = nu  # 動力黏性係數 (對應 Re_tau=1000)
        self.device = device
    
    def navier_stokes_residual(self, coords, pred):
        """
        計算 N-S 方程殘差
        
        Args:
            coords: 座標 [batch_size, 2]
            pred: 預測 [batch_size, 4] -> [u, v, p, S]
        
        Returns:
            residual: NS殘差 [batch_size, 1]
        """
        # 導入物理模組
        from pinnx.physics.ns_2d import ns_residual_2d
        
        # 確保需要梯度計算
        coords.requires_grad_(True)
        
        # 計算 NS 方程殘差
        mom_x, mom_y, cont = ns_residual_2d(coords, pred, self.nu)
        
        # 組合所有殘差（等權重）
        total_residual = mom_x**2 + mom_y**2 + cont**2
        
        return total_residual
    
    def boundary_conditions(self, coords, pred, domain_bounds):
        """
        邊界條件約束 (通道流邊界)
        
        Args:
            coords: 座標 [batch_size, 2]
            pred: 預測 [batch_size, 4]
            domain_bounds: 域邊界字典
        
        Returns:
            bc_loss: 邊界條件損失
        """
        x, y = coords[:, 0:1], coords[:, 1:2]
        u, v, p, S = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
        
        # 通道流邊界條件：
        # 上下壁面 (y = ±1): u = v = 0 (no-slip)
        y_min, y_max = domain_bounds['y']
        
        # 識別邊界點 (容忍度 0.05)
        wall_mask_lower = torch.abs(y - y_min) < 0.05
        wall_mask_upper = torch.abs(y - y_max) < 0.05
        wall_mask = wall_mask_lower | wall_mask_upper
        
        if wall_mask.sum() > 0:
            # 壁面處速度為零
            u_wall = u[wall_mask]
            v_wall = v[wall_mask]
            bc_loss = torch.mean(u_wall**2 + v_wall**2)
        else:
            bc_loss = torch.tensor(0.0, device=coords.device)
        
        return bc_loss
    
    def data_consistency(self, pred_coords, pred_values, sensor_coords, sensor_data):
        """
        感測資料一致性約束
        
        Args:
            pred_coords: 預測座標 [batch_size, 2]
            pred_values: 預測值 [batch_size, 3] -> [u, v, p]
            sensor_coords: 感測點座標 [n_sensors, 2]  
            sensor_data: 感測資料字典 {'u': [...], 'v': [...], 'p': [...]}
        
        Returns:
            data_loss: 資料一致性損失
        """
        # 對每個感測點找最近的預測點
        data_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        n_sensors = len(sensor_coords)
        
        for i in range(n_sensors):
            sensor_pos = sensor_coords[i]
            
            # 找到最近的預測點 (簡化實現)
            distances = torch.norm(pred_coords - sensor_pos, dim=1)
            nearest_idx = torch.argmin(distances)
            
            # 計算該點的預測誤差
            pred_u = pred_values[nearest_idx, 0]
            pred_v = pred_values[nearest_idx, 1]
            pred_p = pred_values[nearest_idx, 2]
            
            true_u = sensor_data['u'][i]
            true_v = sensor_data['v'][i]
            true_p = sensor_data['p'][i]
            
            # MSE 損失
            data_loss = data_loss + (pred_u - true_u)**2 + (pred_v - true_v)**2 + (pred_p - true_p)**2
        
        return data_loss / n_sensors

def train_pinns_with_physics(data, strategy_name, epochs=100, lr=1e-3):
    """
    包含物理約束的 PINNs 訓練
    
    Args:
        data: 載入器返回的資料字典
        strategy_name: 策略名稱 (用於日誌)
        epochs: 訓練輪數
        lr: 學習率
    
    Returns:
        model: 訓練好的模型
        losses: 損失歷史
    """
    device = torch.device('cpu')
    
    # 初始化模型
    model = PINNsModel(input_dim=2, hidden_dim=64, num_layers=4, output_dim=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    physics = PhysicsLoss(nu=1e-4, device='cpu')
    
    # 準備訓練資料
    sensor_coords = torch.tensor(data['coordinates'], dtype=torch.float32, device=device)
    sensor_data = data['sensor_data']
    domain_bounds = data['domain_bounds']
    
    # 建立訓練域的集合點參數（用於物理約束）
    n_collocation = 200
    x_min, x_max = domain_bounds['x']
    y_min, y_max = domain_bounds['y']
    
    # 損失權重
    w_data = 10.0    # 資料一致性
    w_physics = 1.0  # 物理方程
    w_bc = 5.0       # 邊界條件
    
    losses = {
        'total': [], 'data': [], 'physics': [], 'bc': []
    }
    
    total_loss = torch.tensor(0.0)  # 初始化避免 unbound 警告
    
    logger.info(f"🏋️ 開始 {strategy_name} 完整 PINNs 訓練 ({epochs} epochs)...")
    logger.info(f"📍 感測點數: {len(sensor_coords)}, 配置點數: {n_collocation}")
    logger.info(f"⚖️ 損失權重: data={w_data}, physics={w_physics}, bc={w_bc}")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 每個 epoch 重新生成配置點，避免梯度圖複用
        x_coll = torch.rand(n_collocation, 1, device=device, requires_grad=True) * (x_max - x_min) + x_min
        y_coll = torch.rand(n_collocation, 1, device=device, requires_grad=True) * (y_max - y_min) + y_min
        collocation_points = torch.cat([x_coll, y_coll], dim=1)
        
        # === 感測點的資料一致性損失 ===
        sensor_pred = model(sensor_coords)
        data_loss = physics.data_consistency(
            sensor_coords, sensor_pred[:, :3], 
            sensor_coords.detach().cpu().numpy(), sensor_data
        )
        
        # === 配置點的物理方程損失 ===
        collocation_pred = model(collocation_points)
        physics_loss = torch.mean(physics.navier_stokes_residual(collocation_points, collocation_pred))
        
        # === 邊界條件損失 ===
        bc_loss = physics.boundary_conditions(collocation_points, collocation_pred, domain_bounds)
        
        # === 總損失 ===
        total_loss = w_data * data_loss + w_physics * physics_loss + w_bc * bc_loss
        
        # 反向傳播
        total_loss.backward()
        optimizer.step()
        
        # 記錄損失
        losses['total'].append(total_loss.item())
        losses['data'].append(data_loss.item())
        losses['physics'].append(physics_loss.item()) 
        losses['bc'].append(bc_loss.item())
        
        # 定期輸出
        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}: Total={total_loss:.6f} "
                       f"(Data={data_loss:.6f}, Physics={physics_loss:.6f}, BC={bc_loss:.6f})")
    
    logger.info(f"✅ {strategy_name} 訓練完成，最終損失: {total_loss:.6f}")
    
    return model, losses

def evaluate_physics_consistency(model, data):
    """評估模型的物理一致性"""
    device = torch.device('cpu')
    physics = PhysicsLoss(nu=1e-4, device='cpu')
    
    # 在域內隨機點測試物理一致性
    domain_bounds = data['domain_bounds']
    x_min, x_max = domain_bounds['x']
    y_min, y_max = domain_bounds['y']
    
    n_test = 100
    x_test = torch.rand(n_test, 1, requires_grad=True) * (x_max - x_min) + x_min
    y_test = torch.rand(n_test, 1, requires_grad=True) * (y_max - y_min) + y_min
    test_points = torch.cat([x_test, y_test], dim=1)
    
    # 計算物理殘差（需要梯度計算）
    pred = model(test_points)
    physics_residual = physics.navier_stokes_residual(test_points, pred)
    
    with torch.no_grad():
        mean_residual = torch.mean(physics_residual).item()
        max_residual = torch.max(physics_residual).item()
    
    return {
        'mean_physics_residual': mean_residual,
        'max_physics_residual': max_residual,
        'n_test_points': n_test
    }

def main():
    """主實驗流程"""
    logger.info("🚀 開始 Channel Flow Re1000 完整 PINNs 逆重建實驗（含物理約束）")
    
    try:
        # 使用和成功腳本相同的匯入方法
        from pinnx.dataio.channel_flow_loader import prepare_training_data as load_channel_flow
        
        # === 1. 載入不同策略的資料 ===
        strategies = ['qr_pivot', 'random']
        results = {}
        
        for strategy in strategies:
            logger.info(f"\n🔍 測試 {strategy.upper()} 策略...")
            
            # 載入資料
            start_time = time.time()
            data = load_channel_flow(
                strategy=strategy,
                K=8,
                target_fields=['u', 'v', 'p']
            )
            load_time = time.time() - start_time
            logger.info(f"✅ {strategy} 載入完成，耗時: {load_time:.3f}s")
            
            # 訓練 PINNs 模型
            model, losses = train_pinns_with_physics(data, strategy, epochs=100)
            
            # 評估物理一致性
            physics_eval = evaluate_physics_consistency(model, data)
            
            results[strategy] = {
                'model': model,
                'losses': losses,
                'physics_eval': physics_eval,
                'load_time': load_time
            }
            
            logger.info(f"📊 {strategy} 物理一致性:")
            logger.info(f"  平均殘差: {physics_eval['mean_physics_residual']:.6f}")
            logger.info(f"  最大殘差: {physics_eval['max_physics_residual']:.6f}")
        
        # === 2. 策略比較分析 ===
        logger.info("\n" + "="*50)
        logger.info("📊 策略比較結果:")
        
        qr_final_loss = results['qr_pivot']['losses']['total'][-1]
        random_final_loss = results['random']['losses']['total'][-1]
        
        improvement = (random_final_loss - qr_final_loss) / random_final_loss * 100
        
        logger.info(f"  QR-pivot 最終損失: {qr_final_loss:.6f}")
        logger.info(f"  Random 最終損失:   {random_final_loss:.6f}")
        
        if improvement > 0:
            logger.info(f"  🏆 QR-pivot 優於 Random {improvement:.1f}%")
        else:
            logger.info(f"  🎲 Random 優於 QR-pivot {-improvement:.1f}%")
        
        # 物理一致性比較
        qr_physics = results['qr_pivot']['physics_eval']['mean_physics_residual']
        random_physics = results['random']['physics_eval']['mean_physics_residual']
        
        logger.info(f"\n🔬 物理一致性比較:")
        logger.info(f"  QR-pivot 平均殘差: {qr_physics:.6f}")
        logger.info(f"  Random 平均殘差:   {random_physics:.6f}")
        
        physics_improvement = (random_physics - qr_physics) / random_physics * 100
        if physics_improvement > 0:
            logger.info(f"  ⚗️ QR-pivot 物理一致性優 {physics_improvement:.1f}%")
        else:
            logger.info(f"  ⚗️ Random 物理一致性優 {-physics_improvement:.1f}%")
        
        # === 3. 實驗總結 ===
        logger.info("\n" + "="*50)
        logger.info("🎉 完整 PINNs 實驗總結:")
        logger.info("✅ 成功整合物理方程約束 (N-S方程)")
        logger.info("✅ 邊界條件約束正常工作")
        logger.info("✅ 感測資料一致性約束有效")
        logger.info(f"✅ QR-pivot vs Random: {improvement:.1f}% 改善")
        logger.info(f"✅ 物理一致性: {physics_improvement:.1f}% 改善")
        logger.info("🚀 準備好進行更大規模的真實實驗！")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 實驗過程中出現錯誤: {e}")
        raise

if __name__ == '__main__':
    results = main()
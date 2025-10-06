#!/usr/bin/env python3
"""
Channel Flow Re1000 PINNs 逆重建實驗腳本
基於新開發的 Channel Flow 載入器進行真實實驗
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
import logging

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_channel_flow_loader():
    """測試 Channel Flow 載入器功能"""
    try:
        from pinnx.dataio.channel_flow_loader import prepare_training_data as load_channel_flow
        
        logger.info("=== Channel Flow 載入器測試 ===")
        
        # 測試 QR-pivot 策略
        logger.info("🔍 測試 QR-pivot 感測點策略...")
        start_time = time.time()
        
        qr_data = load_channel_flow(
            strategy='qr_pivot',
            K=8,
            target_fields=['u', 'v', 'p']
        )
        
        load_time = time.time() - start_time
        logger.info(f"✅ QR-pivot 載入完成，耗時: {load_time:.3f}s")
        
        # 檢查載入的資料結構
        logger.info("📊 QR-pivot 資料結構:")
        for key, value in qr_data.items():
            if isinstance(value, np.ndarray):
                logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, dict):
                logger.info(f"  {key}: dict with keys={list(value.keys())}")
            else:
                logger.info(f"  {key}: {type(value)}")
        
        # 測試 Random 策略
        logger.info("🎲 測試 Random 感測點策略...")
        start_time = time.time()
        
        random_data = load_channel_flow(
            strategy='random',
            K=8,
            target_fields=['u', 'v', 'p']
        )
        
        load_time = time.time() - start_time
        logger.info(f"✅ Random 載入完成，耗時: {load_time:.3f}s")
        
        return qr_data, random_data
        
    except Exception as e:
        logger.error(f"❌ 載入器測試失敗: {e}")
        raise

def create_simple_pinn_model(input_dim=2, output_dim=3, width=64, depth=3):
    """創建簡單的 PINN 模型用於測試"""
    layers = []
    layers.append(nn.Linear(input_dim, width))
    layers.append(nn.Tanh())
    
    for _ in range(depth - 1):
        layers.append(nn.Linear(width, width))
        layers.append(nn.Tanh())
    
    layers.append(nn.Linear(width, output_dim))
    
    return nn.Sequential(*layers)

def test_pinn_forward_pass(data_dict, device='cpu'):
    """測試 PINNs 前向傳播"""
    logger.info("🧠 測試 PINNs 模型前向傳播...")
    
    # 創建簡單模型
    model = create_simple_pinn_model(input_dim=2, output_dim=3)
    model = model.to(device)
    
    # 準備輸入資料 (x, y 座標)
    coords = data_dict['coordinates']  # (K, 2)
    inputs = torch.from_numpy(coords).float().to(device)
    
    logger.info(f"📥 輸入座標: shape={inputs.shape}")
    
    # 前向傳播
    with torch.no_grad():
        outputs = model(inputs)  # (K, 3) -> u, v, p
    
    logger.info(f"📤 輸出預測: shape={outputs.shape}")
    
    # 計算與真實資料的誤差
    sensor_data = data_dict['sensor_data']
    u_true = torch.from_numpy(sensor_data['u']).float().to(device)
    v_true = torch.from_numpy(sensor_data['v']).float().to(device)
    p_true = torch.from_numpy(sensor_data['p']).float().to(device)
    
    u_pred = outputs[:, 0:1]
    v_pred = outputs[:, 1:2]
    p_pred = outputs[:, 2:3]
    
    u_error = torch.mean((u_pred.squeeze() - u_true.squeeze())**2).item()
    v_error = torch.mean((v_pred.squeeze() - v_true.squeeze())**2).item()
    p_error = torch.mean((p_pred.squeeze() - p_true.squeeze())**2).item()
    
    logger.info(f"🎯 初始 MSE 誤差:")
    logger.info(f"  u: {u_error:.6f}")
    logger.info(f"  v: {v_error:.6f}")
    logger.info(f"  p: {p_error:.6f}")
    
    return model, outputs

def simple_training_loop(model, data_dict, epochs=50, lr=1e-3, device='cpu'):
    """簡單的訓練迴圈"""
    logger.info(f"🏋️ 開始簡單訓練 ({epochs} epochs)...")
    
    # 準備資料
    coords = torch.from_numpy(data_dict['coordinates']).float().to(device)
    sensor_data = data_dict['sensor_data']
    u_true = torch.from_numpy(sensor_data['u']).float().to(device)
    v_true = torch.from_numpy(sensor_data['v']).float().to(device)
    p_true = torch.from_numpy(sensor_data['p']).float().to(device)
    
    # 優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向傳播
        outputs = model(coords)
        u_pred = outputs[:, 0]
        v_pred = outputs[:, 1]
        p_pred = outputs[:, 2]
        
        # 計算損失
        loss_u = criterion(u_pred, u_true.squeeze())
        loss_v = criterion(v_pred, v_true.squeeze())
        loss_p = criterion(p_pred, p_true.squeeze())
        total_loss = loss_u + loss_v + loss_p
        
        # 反向傳播
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1:2d}: Loss = {total_loss.item():.6f} "
                       f"(u: {loss_u.item():.6f}, v: {loss_v.item():.6f}, p: {loss_p.item():.6f})")
    
    logger.info(f"✅ 訓練完成，最終損失: {losses[-1]:.6f}")
    
    return losses

def compare_strategies(qr_data, random_data):
    """比較 QR-pivot 和 Random 策略"""
    logger.info("⚖️ 比較 QR-pivot vs Random 策略...")
    
    device = 'cpu'
    
    # QR-pivot 策略測試
    logger.info("🎯 QR-pivot 策略訓練:")
    qr_model = create_simple_pinn_model().to(device)
    qr_losses = simple_training_loop(qr_model, qr_data, epochs=50, device=device)
    
    # Random 策略測試
    logger.info("🎲 Random 策略訓練:")
    random_model = create_simple_pinn_model().to(device)
    random_losses = simple_training_loop(random_model, random_data, epochs=50, device=device)
    
    # 比較結果
    qr_final_loss = qr_losses[-1]
    random_final_loss = random_losses[-1]
    
    improvement = (random_final_loss - qr_final_loss) / random_final_loss * 100
    
    logger.info("📊 策略比較結果:")
    logger.info(f"  QR-pivot 最終損失: {qr_final_loss:.6f}")
    logger.info(f"  Random 最終損失:   {random_final_loss:.6f}")
    
    if improvement > 0:
        logger.info(f"  🏆 QR-pivot 優於 Random {improvement:.1f}%")
    else:
        logger.info(f"  📈 Random 優於 QR-pivot {-improvement:.1f}%")
    
    return {
        'qr_pivot': {'model': qr_model, 'losses': qr_losses, 'final_loss': qr_final_loss},
        'random': {'model': random_model, 'losses': random_losses, 'final_loss': random_final_loss},
        'improvement': improvement
    }

def main():
    """主函數"""
    logger.info("🚀 開始 Channel Flow Re1000 PINNs 逆重建實驗")
    
    try:
        # 1. 測試載入器
        qr_data, random_data = test_channel_flow_loader()
        
        # 2. 測試 PINNs 前向傳播
        logger.info("\n" + "="*50)
        test_pinn_forward_pass(qr_data)
        
        # 3. 比較感測點策略
        logger.info("\n" + "="*50)
        results = compare_strategies(qr_data, random_data)
        
        # 4. 總結
        logger.info("\n" + "="*50)
        logger.info("🎉 實驗總結:")
        logger.info(f"✅ Channel Flow 載入器正常工作")
        logger.info(f"✅ PINNs 模型可以正常訓練")
        logger.info(f"✅ QR-pivot vs Random: {results['improvement']:.1f}% 改善")
        logger.info("🚀 準備好進行真實的 PINNs 逆重建實驗！")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 實驗失敗: {e}")
        raise

if __name__ == "__main__":
    results = main()
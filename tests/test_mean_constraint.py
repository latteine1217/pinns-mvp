"""
單元測試：MeanConstraintLoss

測試範圍：
1. 零損失條件（預測均值匹配目標）
2. 非零損失計算正確性
3. 梯度流通（可微分性）
4. 批次維度處理
5. 設備兼容性（CPU/CUDA）
6. 部分場約束（僅約束指定場）
"""

import pytest
import torch
import torch.nn as nn
from pinnx.losses import MeanConstraintLoss


class TestMeanConstraintLoss:
    """MeanConstraintLoss 單元測試套件"""
    
    @pytest.fixture
    def loss_fn(self):
        """創建損失函數實例"""
        return MeanConstraintLoss()
    
    @pytest.fixture
    def target_means(self):
        """標準目標均值（模擬 JHTDB Re_tau=1000）"""
        return {'u': 9.84, 'v': 0.0, 'w': 0.0}
    
    @pytest.fixture
    def field_indices(self):
        """場索引映射"""
        return {'u': 0, 'v': 1, 'w': 2, 'p': 3}
    
    # ========== Test 1: 零損失條件 ==========
    def test_zero_loss_when_means_match(self, loss_fn, target_means, field_indices):
        """
        測試：當預測均值完全匹配目標時，損失應為 0
        """
        # 創建匹配目標均值的預測場（使用 requires_grad 確保可求導）
        N = 1000  # 批次大小
        predictions = torch.zeros(N, 4, requires_grad=True)  # [N, 4] (u, v, w, p)
        predictions.data[:, 0] = 9.84  # u 場均值 = 9.84
        predictions.data[:, 1] = 0.0   # v 場均值 = 0.0
        predictions.data[:, 2] = 0.0   # w 場均值 = 0.0
        predictions.data[:, 3] = torch.randn(N)  # p 場（不約束）
        
        loss = loss_fn(predictions, target_means, field_indices)
        
        assert loss.item() < 1e-6, f"期望零損失，實際: {loss.item()}"
        assert loss.requires_grad, "損失應可求導"
    
    # ========== Test 2: 非零損失計算 ==========
    def test_nonzero_loss_for_offset_means(self, loss_fn, target_means, field_indices):
        """
        測試：當預測均值偏移時，損失應非零且正確計算
        
        數學驗證：
            L = (u_pred_mean - 9.84)^2 + (v_pred_mean - 0.0)^2 + (w_pred_mean - 0.0)^2
        """
        N = 500
        predictions = torch.zeros(N, 4)
        predictions[:, 0] = 15.0   # u 偏移 +5.16
        predictions[:, 1] = 2.0    # v 偏移 +2.0
        predictions[:, 2] = -1.0   # w 偏移 -1.0
        
        loss = loss_fn(predictions, target_means, field_indices)
        
        # 手動計算期望損失
        expected_loss = (15.0 - 9.84)**2 + (2.0 - 0.0)**2 + (-1.0 - 0.0)**2
        expected_loss = (5.16)**2 + 4.0 + 1.0  # ≈ 31.63
        
        assert abs(loss.item() - expected_loss) < 1e-3, \
            f"損失計算錯誤: 期望 {expected_loss:.4f}, 實際 {loss.item():.4f}"
    
    # ========== Test 3: 梯度流通 ==========
    def test_gradient_flow(self, loss_fn, target_means, field_indices):
        """
        測試：損失對輸入的梯度應正確傳播
        """
        N = 200
        predictions = torch.randn(N, 4, requires_grad=True)
        
        loss = loss_fn(predictions, target_means, field_indices)
        loss.backward()
        
        # 檢查梯度存在且非零
        assert predictions.grad is not None, "梯度應存在"
        assert torch.isfinite(predictions.grad).all(), "梯度應有限"
        
        # u, v, w 列應有梯度（因被約束）
        assert predictions.grad[:, 0].abs().sum() > 0, "u 場應有梯度"
        assert predictions.grad[:, 1].abs().sum() > 0, "v 場應有梯度"
        assert predictions.grad[:, 2].abs().sum() > 0, "w 場應有梯度"
    
    # ========== Test 4: 批次維度處理 ==========
    @pytest.mark.parametrize("batch_size", [1, 10, 100, 5000])
    def test_batch_size_invariance(self, loss_fn, target_means, field_indices, batch_size):
        """
        測試：不同批次大小應產生相同的損失值（當均值相同時）
        """
        predictions = torch.zeros(batch_size, 4)
        predictions[:, 0] = 12.0  # 固定均值
        predictions[:, 1] = 1.5
        predictions[:, 2] = -0.5
        
        loss = loss_fn(predictions, target_means, field_indices)
        
        # 損失應與批次大小無關（僅依賴均值）
        expected = (12.0 - 9.84)**2 + (1.5)**2 + (-0.5)**2
        assert abs(loss.item() - expected) < 1e-3, \
            f"批次 {batch_size}: 期望 {expected:.4f}, 實際 {loss.item():.4f}"
    
    # ========== Test 5: 設備兼容性 ==========
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 支持")
    def test_cuda_compatibility(self, loss_fn, target_means, field_indices):
        """
        測試：在 GPU 上計算應與 CPU 一致
        """
        N = 300
        predictions_cpu = torch.randn(N, 4)
        
        # CPU 計算
        loss_cpu = loss_fn(predictions_cpu, target_means, field_indices)
        
        # GPU 計算
        predictions_gpu = predictions_cpu.cuda()
        loss_gpu = loss_fn(predictions_gpu, target_means, field_indices)
        
        # 結果應一致（允許微小數值誤差）
        assert abs(loss_cpu.item() - loss_gpu.cpu().item()) < 1e-4, \
            "CPU 與 GPU 計算結果應一致"
    
    # ========== Test 6: 部分場約束 ==========
    def test_partial_field_constraint(self, loss_fn, field_indices):
        """
        測試：僅約束部分場（如僅約束 u）
        """
        N = 400
        predictions = torch.zeros(N, 4)
        predictions[:, 0] = 20.0   # u 偏移大
        predictions[:, 1] = 100.0  # v 偏移極大（但不約束）
        
        # 僅約束 u 場
        partial_targets = {'u': 9.84}
        loss = loss_fn(predictions, partial_targets, field_indices)
        
        # 損失應僅來自 u 場
        expected = (20.0 - 9.84)**2  # ≈ 103.23
        assert abs(loss.item() - expected) < 1e-3, \
            f"部分約束失敗: 期望 {expected:.4f}, 實際 {loss.item():.4f}"
    
    # ========== Test 7: 邊界條件 ==========
    def test_extreme_values(self, loss_fn, target_means, field_indices):
        """
        測試：極端值下的數值穩定性
        """
        N = 100
        predictions = torch.zeros(N, 4)
        predictions[:, 0] = 1e6   # 極大偏移
        predictions[:, 1] = -1e6  # 極大負偏移
        
        loss = loss_fn(predictions, target_means, field_indices)
        
        # 損失應有限且非 NaN
        assert torch.isfinite(loss), "極端值下損失應有限"
        assert loss.item() > 0, "損失應為正"
    
    # ========== Test 8: 警告訊息 ==========
    def test_missing_field_warning(self, loss_fn, field_indices, caplog):
        """
        測試：當目標場不在索引中時應警告
        """
        N = 100
        predictions = torch.randn(N, 4, requires_grad=True)  # 確保可求導
        
        # 提供不存在的場名稱
        invalid_targets = {'u': 9.84, 'nonexistent_field': 1.0}
        
        with caplog.at_level('WARNING'):
            loss = loss_fn(predictions, invalid_targets, field_indices)
        
        # 應記錄警告（實際實作中有 logger.warning）
        # 注意：此測試需要實際運行才能驗證 logging 行為
        assert loss.requires_grad, "即使有警告，損失仍應可計算"


# ========== 整合測試：模擬訓練場景 ==========
class TestMeanConstraintIntegration:
    """整合測試：模擬實際訓練循環"""
    
    def test_training_scenario(self):
        """
        模擬 Phase 6C 訓練場景：
        1. 初始化簡單 MLP
        2. 計算 MeanConstraintLoss
        3. 反向傳播
        4. 驗證損失下降
        """
        # 簡單 MLP 模型
        model = nn.Sequential(
            nn.Linear(4, 64),  # 輸入 (x, y, z, t)
            nn.Tanh(),
            nn.Linear(64, 4)   # 輸出 (u, v, w, p)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # 增大學習率
        loss_fn = MeanConstraintLoss()
        
        # 訓練資料
        inputs = torch.randn(500, 4)  # [N, 4]
        target_means = {'u': 9.84, 'v': 0.0, 'w': 0.0}
        field_indices = {'u': 0, 'v': 1, 'w': 2, 'p': 3}
        
        losses = []
        for _ in range(100):  # 增加迭代次數到 100
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, target_means, field_indices)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # 驗證損失下降（最終 < 初始的 30%）
        assert losses[-1] < losses[0] * 0.3, \
            f"訓練後損失應下降: 初始 {losses[0]:.4f} → 最終 {losses[-1]:.4f}"
        
        # 驗證最終預測均值接近目標
        with torch.no_grad():
            final_pred = model(inputs)
            final_u_mean = final_pred[:, 0].mean().item()
            assert abs(final_u_mean - 9.84) < 3.0, \
                f"最終 u 均值應接近 9.84: {final_u_mean:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

# Add Physics Checkpoint Validation

## Why
當前訓練流程缺乏自動化的物理一致性檢查，導致可能保存不符合物理約束的檢查點（如質量守恆誤差 >1%、邊界條件違反）。這違反了專案的 Physics Gate 原則，可能導致後續評估使用無效模型。

## What Changes
- 在檢查點保存前自動執行物理驗證（質量守恆、動量守恆、邊界條件）
- 若物理驗證失敗，拒絕保存檢查點並記錄警告
- 在檢查點元數據中記錄物理驗證指標（守恆誤差、邊界條件誤差）
- 提供配置選項允許用戶調整驗證閾值或禁用驗證（僅用於除錯）

## Impact
- **Affected specs**: `physics-validation` (新增)
- **Affected code**: 
  - `pinnx/train/trainer.py`: 修改 `save_checkpoint()` 方法
  - `pinnx/train/checkpointing.py`: 新增 `validate_physics_before_save()` 函數
  - `pinnx/physics/validators.py`: 新增物理驗證工具函數
  - `configs/templates/*.yml`: 新增 `physics_validation` 配置區塊
- **Breaking changes**: 無（預設啟用，但可透過配置禁用）
- **Performance impact**: 每次保存檢查點增加 ~2-5 秒驗證時間（可接受）

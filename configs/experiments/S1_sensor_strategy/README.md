# S1: 感測點策略對比

## 實驗目的

對比 Random vs QR-pivot 感測器選擇策略

## 配置文件

- `s1_random_K100_2d_re50.yml`
- `s1_qr_K100_2d_re50.yml`

## 對比指標

L2(u,v,∇p), ‖∇·u‖, sensor quality (condition number)

## 執行方式

```bash
# 從 repo root 執行（建議）
python scripts/train/train.py --cfg configs/experiments/S1_sensor_strategy/s1_random_K100_2d_re50.yml
python scripts/train/train.py --cfg configs/experiments/S1_sensor_strategy/s1_qr_K100_2d_re50.yml

# 或一次跑完本資料夾
for cfg in configs/experiments/S1_sensor_strategy/*.yml; do python scripts/train/train.py --cfg "$cfg"; done
```

## 評估結果

```bash
# 以 checkpoint_dir 內的 best_model.pth / latest.pth 為主
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/experiments/S1_qr_K100/best_model.pth \
  --config configs/experiments/S1_sensor_strategy/s1_qr_K100_2d_re50.yml
```

## 預期結果

請參考 `docs/EXPERIMENT_COMPARISON_PLAN.md` 中的預期性能表格。

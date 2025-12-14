# A2: 動態權重消融實驗

## 實驗目的

量化 GradNorm 自適應權重的貢獻

## 配置文件

- `a2_with_adaptive_K100_2d_re50.yml`
- `a2_without_adaptive_K100_2d_re50.yml`

## 對比指標

收斂速度（epochs/time）+ 最終 L2

## 執行方式

```bash
# 從 repo root 執行（建議）
python scripts/train/train.py --cfg configs/experiments/A2_ablation_weights/a2_with_adaptive_K100_2d_re50.yml
python scripts/train/train.py --cfg configs/experiments/A2_ablation_weights/a2_without_adaptive_K100_2d_re50.yml

# 或一次跑完本資料夾
for cfg in configs/experiments/A2_ablation_weights/*.yml; do python scripts/train/train.py --cfg "$cfg"; done
```

## 評估結果

```bash
# 以 checkpoint_dir 內的 best_model.pth / latest.pth 為主
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/experiments/A2_with_adaptive_K100/best_model.pth \
  --config configs/experiments/A2_ablation_weights/a2_with_adaptive_K100_2d_re50.yml
```

## 預期結果

請參考 `docs/EXPERIMENT_COMPARISON_PLAN.md` 中的預期性能表格。

# C1: RANS Prior 對比實驗

## 實驗目的

對比有無 RANS 先驗的性能差異

## 配置文件

- `c1_no_prior_K100_2d_re50.yml`
- `c1_with_prior_K100_2d_re50.yml`

## 對比指標

L2(u,v,∇p), ‖∇·u‖, 壓力場重建品質

## 執行方式

```bash
# 從 repo root 執行（建議）
python scripts/train/train.py --cfg configs/experiments/C1_prior_comparison/c1_no_prior_K100_2d_re50.yml
python scripts/train/train.py --cfg configs/experiments/C1_prior_comparison/c1_with_prior_K100_2d_re50.yml

# 或一次跑完本資料夾
for cfg in configs/experiments/C1_prior_comparison/*.yml; do python scripts/train/train.py --cfg "$cfg"; done
```

## 評估結果

```bash
# 以 checkpoint_dir 內的 best_model.pth / latest.pth 為主
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/experiments/C1_with_prior_K100/best_model.pth \
  --config configs/experiments/C1_prior_comparison/c1_with_prior_K100_2d_re50.yml
```

## 預期結果

請參考 `docs/EXPERIMENT_COMPARISON_PLAN.md` 中的預期性能表格。

# C2: RANS Prior 權重掃描

## 實驗目的

掃描 prior_weight ∈ {0.1, 0.3, 0.5}

## 配置文件

- `c2_prior_0.1_K100_2d_re50.yml`
- `c2_prior_0.3_K100_2d_re50.yml`
- `c2_prior_0.5_K100_2d_re50.yml`

## 對比指標

error vs prior_weight 曲線

## 執行方式

```bash
# 從 repo root 執行（建議）
for cfg in configs/experiments/C2_prior_sweep/*.yml; do python scripts/train/train.py --cfg "$cfg"; done

# 或逐個執行（例：prior_weight=0.1）
python scripts/train/train.py --cfg configs/experiments/C2_prior_sweep/c2_prior_0.1_K100_2d_re50.yml
```

## 評估結果

```bash
# 逐個評估 checkpoint_dir 內的 best_model.pth / latest.pth
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/experiments/C2_prior_0.1_K100/best_model.pth \
  --config configs/experiments/C2_prior_sweep/c2_prior_0.1_K100_2d_re50.yml
```

## 預期結果

請參考 `docs/EXPERIMENT_COMPARISON_PLAN.md` 中的預期性能表格。

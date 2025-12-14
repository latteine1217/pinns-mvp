# M1: 模型表示能力對比

## 實驗目的

對比 Vanilla MLP vs Full features

## 配置文件

- `m1_vanilla_K100_2d_re50.yml`
- `m1_full_K100_2d_re50.yml`

## 對比指標

L2 與 divergence trade-off

## 執行方式

```bash
# 從 repo root 執行（建議）
python scripts/train/train.py --cfg configs/experiments/M1_model_comparison/m1_vanilla_K100_2d_re50.yml
python scripts/train/train.py --cfg configs/experiments/M1_model_comparison/m1_full_K100_2d_re50.yml

# 或一次跑完本資料夾
for cfg in configs/experiments/M1_model_comparison/*.yml; do python scripts/train/train.py --cfg "$cfg"; done
```

## 評估結果

```bash
# 以 checkpoint_dir 內的 best_model.pth / latest.pth 為主
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/experiments/M1_full_K100/best_model.pth \
  --config configs/experiments/M1_model_comparison/m1_full_K100_2d_re50.yml
```

## 預期結果

請參考 `docs/EXPERIMENT_COMPARISON_PLAN.md` 中的預期性能表格。

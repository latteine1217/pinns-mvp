# A1: Fourier Features 消融實驗

## 實驗目的

量化 Fourier Features 的貢獻

## 配置文件

- `a1_with_fourier_K100_2d_re50.yml`
- `a1_without_fourier_K100_2d_re50.yml`

## 對比指標

L2 與能譜差異

## 執行方式

```bash
# 從 repo root 執行（建議）
python scripts/train/train.py --cfg configs/experiments/A1_ablation_fourier/a1_with_fourier_K100_2d_re50.yml
python scripts/train/train.py --cfg configs/experiments/A1_ablation_fourier/a1_without_fourier_K100_2d_re50.yml

# 或一次跑完本資料夾
for cfg in configs/experiments/A1_ablation_fourier/*.yml; do python scripts/train/train.py --cfg "$cfg"; done
```

## 評估結果

```bash
# 以 checkpoint_dir 內的 best_model.pth / latest.pth 為主
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/experiments/A1_with_fourier_K100/best_model.pth \
  --config configs/experiments/A1_ablation_fourier/a1_with_fourier_K100_2d_re50.yml
```

## 預期結果

請參考 `docs/EXPERIMENT_COMPARISON_PLAN.md` 中的預期性能表格。

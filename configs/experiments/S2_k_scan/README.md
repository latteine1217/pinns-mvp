# S2: K 值掃描實驗

## 實驗目的

掃描不同感測點數量 K ∈ {30, 50, 80, 100}

## 配置文件

- `s2_qr_K30_2d_re50.yml`
- `s2_qr_K50_2d_re50.yml`
- `s2_qr_K80_2d_re50.yml`
- `s2_qr_K100_2d_re50.yml`

## 對比指標

K-error 曲線，找最小可行 K

## 執行方式

```bash
# 從 repo root 執行（建議）
for cfg in configs/experiments/S2_k_scan/*.yml; do python scripts/train/train.py --cfg "$cfg"; done

# 或逐個執行（例：K=30）
python scripts/train/train.py --cfg configs/experiments/S2_k_scan/s2_qr_K30_2d_re50.yml
```

## 評估結果

```bash
# 逐個評估 checkpoint_dir 內的 best_model.pth / latest.pth
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/experiments/S2_K30/best_model.pth \
  --config configs/experiments/S2_k_scan/s2_qr_K30_2d_re50.yml
```

## 預期結果

請參考 `docs/EXPERIMENT_COMPARISON_PLAN.md` 中的預期性能表格。

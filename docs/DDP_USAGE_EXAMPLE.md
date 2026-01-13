# DDP 快速使用範例（簡化版）

## 啟動指令

```bash
torchrun --nproc_per_node=2 scripts/train/train.py --cfg configs/your_config.yml
```

## 建議配置

```yaml
training:
  ddp:
    enabled: null         # null=自動偵測, true=強制啟用, false=禁用
    split_data: true      # 讓每張 GPU 處理不同資料子集
    reduce_losses: true   # 同步損失供監控
```

## 注意事項

- 建議使用 `torchrun` 啟動，環境變數會自動設定 `LOCAL_RANK`。
- 若需要診斷分割情況，將 `training.ddp.verify_data_split` 設為 `true`。

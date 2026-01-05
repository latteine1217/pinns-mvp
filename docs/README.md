# PINNs-SparseFlow 文檔中心

**稀疏測量湍流場重建：基於物理資訊神經網路的逆問題求解**

---

## 🎯 專案概述

本專案使用物理資訊神經網路 (PINNs) 結合稀疏感測器資料與低保真物理先驗，重建高解析度湍流場。

**核心能力**：
- ✅ 稀疏感測器重建 (K ≤ 100)
- ✅ RANS/Leith 物理先驗融合
- ✅ DNS 高保真驗證
- ✅ 2D Kolmogorov Flow 與 3D Channel Flow

---

## 📚 文檔導航

### 🚀 快速開始

| 文檔 | 用途 | 適合對象 |
|------|------|---------|
| [**QUICK_START.md**](QUICK_START.md) | 環境設定→訓練→評估完整流程 | 新用戶 |
| [**API_REFERENCE.md**](API_REFERENCE.md) | 腳本與工具使用說明 | 所有用戶 |
| [**PROJECT_SCOPE.md**](PROJECT_SCOPE.md) | 專案範圍與支援功能 | 了解專案邊界 |

### 📖 技術文檔

| 文檔 | 用途 | 適合對象 |
|------|------|---------|
| [**TECHNICAL_DOCUMENTATION.md**](TECHNICAL_DOCUMENTATION.md) | 系統架構與模組設計 | 開發者 |
| [**CONFIG_GUIDE.md**](CONFIG_GUIDE.md) | 配置參數完整參考與管理指南 | 所有用戶 |

### 📊 實驗與評估

| 文檔 | 用途 | 適合對象 |
|------|------|---------|
| [**METRICS_TRACKING_GUIDE.md**](METRICS_TRACKING_GUIDE.md) | 指標追蹤與 WandB 整合 | 實驗管理者 |
| [**FIGURE_GENERATION_GUIDE.md**](FIGURE_GENERATION_GUIDE.md) | 視覺化與圖表生成 | 結果分析者 |

### 🔧 問題排解

| 文檔 | 用途 | 適合對象 |
|------|------|---------|
| [**TROUBLESHOOTING.md**](TROUBLESHOOTING.md) | 常見問題診斷與解決方案 | 遇到問題時 |

---

## 🗂️ 使用場景導航

### 場景 1: 第一次使用專案
```
1. QUICK_START.md    # 設定環境並執行第一個實驗
2. CONFIG_GUIDE.md   # 理解配置參數
3. API_REFERENCE.md  # 學習腳本使用
```

### 場景 2: 設計新實驗
```
1. PROJECT_SCOPE.md           # 確認支援的功能
2. CONFIG_GUIDE.md            # 規劃配置策略
3. METRICS_TRACKING_GUIDE.md  # 設定指標追蹤
```

### 場景 3: 深入理解系統
```
1. TECHNICAL_DOCUMENTATION.md  # 系統架構
2. CONFIG_GUIDE.md             # 進階配置
3. archive/                    # 歷史決策與分析
```

### 場景 4: 遇到問題
```
1. TROUBLESHOOTING.md  # 常見問題速查
2. 查看相關測試：tests/test_*.py
3. 檢查日誌：context/session_logs/
```

---

## 📂 文檔結構

```
docs/
├── README.md                      # 本文件（導航中心）
│
├── 快速開始
│   ├── QUICK_START.md            # 完整入門教程
│   ├── API_REFERENCE.md          # API 與腳本參考
│   └── PROJECT_SCOPE.md          # 專案範圍定義
│
├── 技術文檔
│   ├── TECHNICAL_DOCUMENTATION.md    # 系統架構
│   └── CONFIG_GUIDE.md               # 配置參考與管理
│
├── 實驗與評估
│   ├── METRICS_TRACKING_GUIDE.md     # 指標追蹤
│   └── FIGURE_GENERATION_GUIDE.md    # 視覺化指南
│
├── 問題排解
│   └── TROUBLESHOOTING.md            # 故障排除
│
└── archive/                          # 歷史文檔歸檔
    ├── analysis/                     # 實驗分析
    ├── experiments/                  # 特定實驗
    ├── wandb/                        # WandB 整合
    └── legacy/                       # 已取代文檔
```

---

## 🔍 關鍵概念速查

### 物理場景
- **2D Kolmogorov Flow**: Re=50-100, Leith 模型, 週期邊界
- **3D Channel Flow**: Re_τ=1000, RANS k-ε, 壁面+週期邊界

### 核心技術
- **架構**: Fourier Feature Network + VS-PINN
- **感測器選擇**: QR-Pivot (物理導向)
- **優化**: SOAP → L-BFGS 兩階段
- **損失平衡**: GradNorm + Causal Weighting

### 資料流
```
DNS Ground Truth → Sensor Selection (QR-Pivot)
                 ↓
Low-Fi Prior (RANS/Leith) + Sparse Sensors
                 ↓
        Physics-Informed Training
                 ↓
    High-Fidelity Field Reconstruction
```

---

## 📦 其他資源

- **配置範例**: `configs/`
- **腳本工具**: `scripts/README.md`
- **單元測試**: `tests/`
- **開發歷史**: `context/`

---

## 🚨 常見問題快速連結

| 問題 | 解決方案 |
|------|---------|
| 訓練發散 / NaN | [TROUBLESHOOTING.md#訓練穩定性](TROUBLESHOOTING.md) |
| 感測器選擇失敗 | [TROUBLESHOOTING.md#感測器問題](TROUBLESHOOTING.md) |
| 壓力場異常 | [TROUBLESHOOTING.md#壓力場](TROUBLESHOOTING.md) |
| 配置錯誤 | [CONFIG_GUIDE.md](CONFIG_GUIDE.md) |
| 配置驗證工具 | `scripts/tools/validate_config.py` |
| 指標追蹤問題 | [METRICS_TRACKING_GUIDE.md](METRICS_TRACKING_GUIDE.md) |

---

## 📝 文檔維護

- **版本**: 2.0.1
- **最後更新**: 2026-01-05
- **維護者**: PINNs-SparseFlow 專案

### 文檔更新原則
1. **核心文檔必須同步**: 程式碼變更後立即更新相關文檔
2. **過時文檔歸檔**: 不刪除，移至 `archive/` 並註明原因
3. **保持簡潔**: 避免重複，使用交叉引用
4. **實例驅動**: 提供具體範例而非抽象說明

---

## 🔗 外部資源

- **主專案**: [README.md](../README.md)
- **程式碼 API**: [pinnx/README.md](../pinnx/README.md)
- **實驗配置**: [configs/README.md](../configs/README.md)
- **開發日誌**: [context/README.md](../context/README.md)

---

**💡 提示**: 如果找不到需要的資訊，請先查看 `TROUBLESHOOTING.md` 或 `archive/` 中的歷史文檔。

# 🚀 Google Colab 快速設定指南

## 📋 前置需求

1. **Google 帳號**（有 Google Drive 存取權限）
2. **專案檔案**（從 GitHub 下載或克隆）
3. **Colab 環境**（選擇 GPU 執行階段）

---

## 📥 Step 1: 下載專案到本地

### **選項 A: 使用 Git（推薦）**
```bash
git clone https://github.com/latteine1217/pinns-mvp.git
cd pinns-mvp
```

### **選項 B: 下載 ZIP**
1. 前往 https://github.com/latteine1217/pinns-mvp
2. 點擊 **Code → Download ZIP**
3. 解壓縮到本地資料夾

---

## ☁️ Step 2: 上傳專案到 Google Drive

### **方法 1: 手動上傳（簡單）**

1. **壓縮專案資料夾**（可選，加快上傳速度）：
   ```bash
   # Mac/Linux
   zip -r pinns-mvp.zip pinns-mvp
   
   # Windows（使用內建壓縮）
   # 右鍵點擊資料夾 → 傳送到 → 壓縮的資料夾
   ```

2. **上傳到 Google Drive**：
   - 打開 https://drive.google.com
   - 點擊 **新增 → 檔案上傳**
   - 選擇 `pinns-mvp.zip`（或直接上傳資料夾）
   - 等待上傳完成

3. **解壓縮**（如果上傳了 ZIP）：
   - 右鍵點擊 `pinns-mvp.zip`
   - 選擇 **解壓縮**
   - 確認解壓後的資料夾名稱為 `pinns-mvp`

### **方法 2: 使用 Google Drive 桌面版（快速）**

1. 安裝 [Google Drive for Desktop](https://www.google.com/drive/download/)
2. 將 `pinns-mvp` 資料夾複製到 **My Drive**
3. 等待同步完成

---

## 📂 Step 3: 確認專案路徑

**預設路徑**：`/content/drive/MyDrive/pinns-mvp`

**如果您的專案在不同位置**，記下完整路徑，例如：
- `/content/drive/MyDrive/研究/pinns-mvp`
- `/content/drive/MyDrive/Projects/pinns-mvp`

---

## 🚀 Step 4: 開啟 Colab Notebook

### **選項 A: 從 Google Drive 直接開啟**

1. 在 Drive 中找到 `pinns-mvp` 資料夾
2. 雙擊 `PINNs_MVP_Kolmogorov_Guide.ipynb`
3. 選擇 **使用 Google Colaboratory 開啟**

### **選項 B: 從 Colab 匯入**

1. 前往 https://colab.research.google.com
2. 點擊 **檔案 → 上傳筆記本**
3. 選擇 **Google 雲端硬碟** 分頁
4. 瀏覽到 `pinns-mvp/PINNs_MVP_Kolmogorov_Guide.ipynb`
5. 點擊開啟

---

## ⚙️ Step 5: 設定 GPU 加速

**在 Colab 中**：

1. 點擊選單 **執行階段 → 變更執行階段類型**
2. **硬體加速器** 選擇：
   - **T4 GPU**（免費，適合測試）
   - **A100 GPU**（付費，訓練速度快 50-100 倍）⭐
3. 點擊 **儲存**

**驗證 GPU**：
```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 🔧 Step 6: 執行 Part 0（初始化）

**按順序執行以下 Cells**：

### **Cell 0.1: 檢測環境**
```python
try:
    import google.colab
    IN_COLAB = True
    print("✅ 檢測到 Google Colab 環境")
except ImportError:
    IN_COLAB = False
    print("ℹ️  本地環境，跳過 Colab 初始化")
```
**預期輸出**：`✅ 檢測到 Google Colab 環境`

---

### **Cell 0.2: 掛載 Google Drive**
```python
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive 已掛載至 /content/drive")
```

**首次執行會彈出授權視窗**：
1. 點擊連結
2. 選擇您的 Google 帳號
3. 點擊 **允許**
4. 複製授權碼並貼回 Colab

**預期輸出**：
```
Mounted at /content/drive
✅ Google Drive 已掛載至 /content/drive
```

---

### **Cell 0.3: 切換到專案目錄**
```python
import os

if IN_COLAB:
    PROJECT_PATH = '/content/drive/MyDrive/pinns-mvp'  # ⚠️ 根據您的路徑修改
    
    if os.path.exists(PROJECT_PATH):
        os.chdir(PROJECT_PATH)
        print(f"✅ 已切換到專案目錄: {os.getcwd()}")
    else:
        print(f"❌ 專案目錄不存在: {PROJECT_PATH}")
        print("\n您的 Drive 根目錄內容：")
        !ls /content/drive/MyDrive/
```

**如果成功**：
```
✅ 已切換到專案目錄: /content/drive/MyDrive/pinns-mvp
```

**如果失敗**：
```
❌ 專案目錄不存在: /content/drive/MyDrive/pinns-mvp

您的 Drive 根目錄內容：
Colab Notebooks/
Projects/
研究/
...
```

**解決方法**：
- 檢查專案是否在上述列表中
- 修改 `PROJECT_PATH` 為正確路徑
- 重新執行 Cell 0.3

---

### **Cell 0.4: 驗證專案結構**
```python
if IN_COLAB:
    required_dirs = ['configs', 'scripts', 'pinnx', 'data']
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing_dirs:
        print(f"⚠️  警告：缺少以下目錄: {missing_dirs}")
    else:
        print("✅ 專案結構完整")
        !ls -lh
```

**預期輸出**：
```
✅ 專案結構完整

total 120K
drwxr-xr-x  configs/
drwxr-xr-x  data/
drwxr-xr-x  docs/
drwxr-xr-x  pinnx/
drwxr-xr-x  scripts/
-rw-r--r--  README.md
...
```

---

## ✅ 驗證設定成功

**執行以下 Cell 確認一切正常**：

```python
# 檢查 GPU
import torch
print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# 檢查工作目錄
import os
print(f"✅ 工作目錄: {os.getcwd()}")

# 檢查專案檔案
import sys
sys.path.append('.')
import pinnx
print(f"✅ PINNx 版本: {pinnx.__version__}")

# 檢查腳本可執行性
!python scripts/calculate_reynolds_parameters.py --f0 1.0 --nu 0.0125 --k 8
```

**如果看到 Reynolds 數計算結果，代表設定完全成功！** 🎉

---

## 🎯 下一步

設定完成後，您可以：

1. **Part 1**: 驗證 Reynolds 數並檢查環境
2. **Part 2**: 生成 Kolmogorov Flow DNS 資料
3. **Part 3**: 配置 QR-Pivot 感測點
4. **Part 4**: 開始模型訓練

---

## ❓ 常見問題

### **Q1: 為什麼 Cell 0.3 顯示「目錄不存在」？**

**A**: 路徑錯誤。請執行以下步驟：

```python
# 1. 列出您的 Drive 根目錄
!ls /content/drive/MyDrive/

# 2. 如果專案在子資料夾中，列出子資料夾
!ls /content/drive/MyDrive/Projects/

# 3. 找到正確路徑後，修改 PROJECT_PATH
PROJECT_PATH = '/content/drive/MyDrive/Projects/pinns-mvp'  # 範例
```

---

### **Q2: 訓練時出現「CUDA out of memory」錯誤？**

**A**: GPU 記憶體不足。解決方法：

1. **降低批次大小**（在配置文件中）：
   ```yaml
   training:
     batch_size: 512  # 改為 256 或 128
   ```

2. **使用梯度累積**（保持等效批次大小）：
   ```yaml
   training:
     batch_size: 256
     gradient_accumulation_steps: 2  # 等效 512
   ```

3. **重新啟動執行階段**：
   - 選單 → **執行階段 → 重新啟動執行階段**

---

### **Q3: 如何保存訓練結果到 Drive？**

**A**: 訓練結果自動保存在專案目錄中：

- **檢查點**: `checkpoints/<exp_name>/`
- **日誌**: `log/<exp_name>/`
- **結果**: `results/<exp_name>/`

因為專案在 Google Drive 中，所有檔案會自動同步。

---

### **Q4: Colab 斷線後如何恢復訓練？**

**A**: 使用檢查點恢復：

```bash
# 從最新檢查點繼續訓練
!python scripts/train.py \
  --cfg configs/my_experiment.yml \
  --resume checkpoints/my_experiment/latest.pth
```

---

### **Q5: 如何下載訓練結果到本地？**

**選項 A: 從 Colab 直接下載**
```python
from google.colab import files
files.download('checkpoints/my_experiment/best_model.pth')
```

**選項 B: 從 Google Drive 下載**
1. 打開 Google Drive
2. 瀏覽到 `pinns-mvp/checkpoints/`
3. 右鍵點擊檔案 → **下載**

---

## 📚 相關文檔

- **主要 README**: `README.md`
- **配置指南**: `docs/CONFIG_GUIDE.md`
- **感測器格式**: `docs/SENSOR_FILE_FORMAT.md`
- **訓練監控**: `docs/monitoring_guide.md`

---

## 🆘 需要幫助？

如果遇到問題：

1. **檢查錯誤訊息**：仔細閱讀紅色錯誤輸出
2. **查閱文檔**：`docs/` 資料夾中有詳細指南
3. **GitHub Issues**: https://github.com/latteine1217/pinns-mvp/issues
4. **重新啟動**: 有時重新啟動執行階段可以解決問題

---

**祝您訓練順利！** 🚀

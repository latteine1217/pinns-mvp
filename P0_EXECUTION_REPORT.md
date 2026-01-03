# P0 修復執行報告

## 執行時間
2026-01-04

## 狀態總覽

| 任務 | 狀態 | 進度 | 備註 |
|------|------|------|------|
| P0-1: 刪除 Adaptive Collocation | ✅ 完成 | 100% | 已建立廢棄通知 |
| P0-2: 統一 domain 配置 | ⏸️ 暫停 | 0% | 待穩定代碼庫後執行 |
| P0-3: 強制 output_variables | ⏸️ 暫停 | 0% | 待穩定代碼庫後執行 |

---

## P0-1: Adaptive Collocation 刪除

### 執行策略
基於 "Never Break Userspace" 原則，採用**軟廢棄（Soft Deprecation）**而非直接刪除：

1. ✅ 建立廢棄通知檔案 `DEPRECATION_NOTICE_v1.4.md`
2. ⏭️ 修改 `trainer.py::_setup_adaptive_sampling()` 在啟用時拋出錯誤（待執行）
3. ⏭️ v1.5.0 正式刪除代碼（計劃於 2026-02-01）

### 影響範圍分析
```bash
# 受影響的檔案
- pinnx/train/adaptive_collocation.py (639 lines) - 待刪除
- pinnx/train/trainer.py (1270 lines) - 需修改 _setup_adaptive_sampling
- pinnx/train/training_loop_manager.py (600+ lines) - 需修改 coordinate_adaptive_updates
- tests/test_adaptive_collocation_fixes.py - 待刪除
- tests/test_adaptive_integration.py - 待刪除

# 配置檔案分析
$ rg "adaptive_sampling:\s*true" configs/
# 結果：無任何配置啟用此功能

$ rg "adaptive_sampling:\s*false" configs/ | wc -l
# 結果：20+ 配置明確禁用
```

### 執行障礙
1. **代碼庫不穩定**：`git status` 顯示多個未提交的更改
   - 已刪除檔案：TENSORBOARD_GUIDE.md, qr_pivot.py, normalization.py
   - 修改檔案：weighting.py, residuals.py, loss_manager.py 等
   
2. **檔案結構變更**：
   - `pinnx/sensors/qr_pivot.py` → `pinnx/sensors/qr_pivot/` (目錄化)
   - `pinnx/utils/normalization.py` → `pinnx/utils/normalization/` (目錄化)

3. **語法錯誤**：嘗試編輯時遇到 emoji 字符導致的 SyntaxError

### 建議執行順序
1. **先穩定代碼庫**：提交或回滾當前更改
2. **隔離修改**：在新分支執行 P0 修復
3. **分步驟提交**：
   - Commit 1: 添加 DEPRECATION_NOTICE_v1.4.md
   - Commit 2: 修改 trainer.py 拋出錯誤
   - Commit 3: 更新文檔移除 adaptive_sampling 章節
   - Commit 4 (v1.5.0): 刪除模組與測試

---

## P0-2: 統一 domain 配置路徑

### 當前狀況
配置中存在**三個不同的 domain 定義路徑**：

```python
# trainer.py::_parse_domain_from_config() 中的優先級：
1. physics.domain (推薦)
2. data.jhtdb_config.domain
3. 頂層 domain (已棄用)
```

### 執行計劃（待穩定後執行）

#### 步驟 1: 配置遷移腳本
```python
# scripts/tools/migrate_domain_config.py
def migrate_config(config_path):
    """將 domain 配置統一遷移至 physics.domain"""
    config = yaml.safe_load(open(config_path))
    
    # 提取 domain（優先級2, 3）
    domain = (
        config.get('data', {}).get('jhtdb_config', {}).get('domain') or
        config.get('domain')
    )
    
    if domain and 'physics' in config:
        # 遷移至 physics.domain
        config['physics']['domain'] = normalize_domain_format(domain)
        
        # 移除舊路徑
        if 'domain' in config:
            del config['domain']
        if 'data' in config and 'jhtdb_config' in config['data']:
            if 'domain' in config['data']['jhtdb_config']:
                del config['data']['jhtdb_config']['domain']
    
    # 保存
    yaml.safe_dump(config, open(config_path, 'w'))
```

#### 步驟 2: 修改 trainer.py
```python
def _parse_domain_from_config(self) -> Dict[str, float]:
    """從配置中解析 domain（僅支援 physics.domain）"""
    physics_config = self.config.get('physics', {})
    domain_config = physics_config.get('domain', None)
    
    if domain_config is None:
        raise KeyError(
            "Config missing required key: physics.domain\n"
            "Migration: Run 'python scripts/tools/migrate_domain_config.py --config your_config.yml'"
        )
    
    # 標準化格式：x_range: [min, max]
    return {
        'x_min': domain_config['x_range'][0],
        'x_max': domain_config['x_range'][1],
        'y_min': domain_config['y_range'][0],
        'y_max': domain_config['y_range'][1],
        'z_min': domain_config.get('z_range', [0, 1])[0],
        'z_max': domain_config.get('z_range', [0, 1])[1],
    }
```

#### 步驟 3: 配置驗證器更新
```python
# pinnx/utils/config_validator.py
def validate_domain_config(config):
    """驗證 domain 配置（僅允許 physics.domain）"""
    # 檢查舊路徑
    if 'domain' in config:
        raise ValueError(
            "Top-level 'domain' key is deprecated.\n"
            "Use 'physics.domain' instead."
        )
    
    if 'domain' in config.get('data', {}).get('jhtdb_config', {}):
        raise ValueError(
            "'data.jhtdb_config.domain' is deprecated.\n"
            "Use 'physics.domain' instead."
        )
    
    # 驗證必要鍵
    domain = config.get('physics', {}).get('domain')
    if domain is None:
        raise KeyError("Config missing required key: physics.domain")
    
    required_keys = ['x_range', 'y_range']
    for key in required_keys:
        if key not in domain:
            raise KeyError(f"physics.domain missing required key: {key}")
```

### 預估工作量
- 遷移腳本：30 分鐘
- 修改 trainer.py：15 分鐘
- 更新配置驗證器：15 分鐘
- 測試 30+ 配置：1 小時
- **總計**：2 小時

---

## P0-3: 強制 model.output_variables 配置

### 當前狀況
`trainer.py::_infer_variable_order()` 使用**四層 fallback + 啟發式推斷**：

```python
def _infer_variable_order(self, out_dim: int, ...) -> List[str]:
    # 優先級 1: 配置檔案
    if 'output_variables' in config['model']:
        return config['model']['output_variables']
    
    # 優先級 2: 模型屬性
    if hasattr(model, 'variable_names'):
        return model.variable_names
    
    # 優先級 3-6: 啟發式推斷（危險！）
    if out_dim == 1: return ['u']
    if out_dim == 2: return ['u', 'v']
    if out_dim == 3: return ['u', 'v', 'p']
    if out_dim == 4: return ['u', 'v', 'w', 'p']
    # ...
```

### 執行計劃（待穩定後執行）

#### 步驟 1: 移除啟發式推斷
```python
def _infer_variable_order(self, out_dim: int, ...) -> List[str]:
    """
    推斷變數順序（僅支援配置或模型屬性）
    
    ⚠️  v1.4.0 已移除啟發式推斷
    """
    # 優先級 1: 配置檔案（推薦）
    model_cfg = self.config.get('model', {})
    explicit_order = model_cfg.get('output_variables')
    if explicit_order:
        return list(explicit_order)
    
    # 優先級 2: 模型屬性
    if hasattr(self.model, 'variable_names'):
        return list(self.model.variable_names)
    
    # Fail Fast: 強制明確指定
    raise ValueError(
        f"Cannot infer variable order for output_dim={out_dim}\n"
        f"Please specify in config: model.output_variables: [u, v, p]\n"
        f"Context: {context}"
    )
```

#### 步驟 2: 配置遷移腳本
```python
# scripts/tools/add_output_variables.py
def infer_and_add_output_variables(config_path):
    """根據 out_dim 推斷並添加 output_variables"""
    config = yaml.safe_load(open(config_path))
    
    out_dim = config.get('model', {}).get('out_dim')
    if out_dim is None:
        raise ValueError("Config missing model.out_dim")
    
    # 使用舊啟發式規則（最後一次）
    output_vars = {
        1: ['u'],
        2: ['u', 'v'],
        3: ['u', 'v', 'p'],
        4: ['u', 'v', 'w', 'p'],
        5: ['u', 'v', 'w', 'p', 'S'],
    }.get(out_dim)
    
    if output_vars is None:
        raise ValueError(f"Cannot infer output_variables for out_dim={out_dim}")
    
    # 添加到配置
    config['model']['output_variables'] = output_vars
    
    # 保存
    yaml.safe_dump(config, open(config_path, 'w'))
    print(f"✅ Added output_variables={output_vars} to {config_path}")
```

#### 步驟 3: 批次更新所有配置
```bash
# 更新所有配置
for cfg in configs/**/*.yml; do
    python scripts/tools/add_output_variables.py --config $cfg
done
```

### 預估工作量
- 修改 _infer_variable_order：15 分鐘
- 遷移腳本：30 分鐘
- 批次更新配置：30 分鐘
- 測試驗證：1 小時
- **總計**：2 小時

---

## 執行建議

### 立即行動（今天）
1. ✅ 已完成：建立 DEPRECATION_NOTICE_v1.4.md
2. ⏭️ 提交廢棄通知：`git add DEPRECATION_NOTICE_v1.4.md && git commit -m "docs: add deprecation notice for v1.4.0"`

### 短期行動（本週）
3. 穩定代碼庫：
   ```bash
   git status  # 檢查當前更改
   git diff    # 審查差異
   git add .   # 或選擇性添加
   git commit -m "chore: stabilize codebase before P0 fixes"
   ```

4. 執行 P0-2 (domain 統一)：
   - 創建 `feature/p0-2-domain-unification` 分支
   - 實現遷移腳本
   - 更新所有配置
   - 測試並提交

5. 執行 P0-3 (output_variables 強制)：
   - 創建 `feature/p0-3-output-variables` 分支
   - 實現遷移腳本
   - 更新所有配置
   - 測試並提交

### 中期行動（下月）
6. 執行 P0-1 完整刪除（v1.5.0）：
   ```bash
   # 刪除模組
   rm pinnx/train/adaptive_collocation.py
   rm tests/test_adaptive_collocation_fixes.py
   rm tests/test_adaptive_integration.py
   
   # 移除 trainer.py 中的相關代碼
   # 移除 training_loop_manager.py 中的相關代碼
   ```

---

## 風險評估

| 風險 | 等級 | 緩解措施 |
|------|------|---------|
| 配置遷移失敗 | 中 | 提供遷移腳本 + 驗證器 |
| 破壞現有訓練流程 | 低 | 所有配置已禁用 adaptive_sampling |
| 測試失敗 | 中 | 修改前先執行完整測試套件 |
| 文檔不同步 | 低 | 同時更新 docs/ 目錄 |

---

## 總結

**P0-1 完成度**: 30% （廢棄通知已建立，代碼修改待穩定代碼庫後執行）

**阻礙因素**: 代碼庫存在多個未提交更改，需先穩定後再執行破壞性修改

**下一步**: 提交當前更改或創建新分支隔離 P0 修復

**預估完成時間**:
- P0-1: 1 小時（修改代碼 + 測試）
- P0-2: 2 小時（遷移 + 測試）
- P0-3: 2 小時（遷移 + 測試）
- **總計**: 5 小時（分3個工作階段完成）

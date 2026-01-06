#!/usr/bin/env python3
"""
檢查訓練腳本中是否有重複定義的函數

防止函數重定義覆蓋導入的模組函數，這是一個常見的技術債務來源。

使用方法：
    python scripts/tools/check_duplicate_functions.py

建議在 CI/CD 中運行此檢查。
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple


def extract_imports(content: str) -> Dict[str, str]:
    """提取所有 from pinnx.* import 的函數"""
    imports = {}
    
    # 匹配 from pinnx.xxx import func1, func2
    pattern = r'from\s+(pinnx\.\S+)\s+import\s+([^#\n]+)'
    matches = re.finditer(pattern, content)
    
    for match in matches:
        module = match.group(1)
        imported_items = match.group(2)
        
        # 處理多個導入（用逗號分隔）
        for item in imported_items.split(','):
            item = item.strip()
            if item and not item.startswith('('):
                imports[item] = module
    
    return imports


def find_function_definitions(content: str) -> List[Tuple[str, int]]:
    """找到所有函數定義及其行號"""
    definitions = []
    
    pattern = r'^def\s+(\w+)\s*\('
    for match in re.finditer(pattern, content, re.MULTILINE):
        func_name = match.group(1)
        line_num = content[:match.start()].count('\n') + 1
        definitions.append((func_name, line_num))
    
    return definitions


def main():
    script_path = Path(__file__).parent.parent / 'train' / 'train.py'
    
    if not script_path.exists():
        print(f"❌ 找不到訓練腳本: {script_path}")
        sys.exit(1)
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    print("="*70)
    print("🔍 檢查重複函數定義")
    print("="*70)
    print(f"檢查文件: {script_path}")
    print(f"總行數: {content.count(chr(10))}")
    print()
    
    # 提取導入的函數
    imports = extract_imports(content)
    if not imports:
        print("✅ 沒有從 pinnx 模組導入函數")
        return
    
    print(f"從 pinnx 模組導入的函數: {len(imports)}")
    for func, module in sorted(imports.items()):
        print(f"  • {func} ← {module}")
    print()
    
    # 查找函數定義
    definitions = find_function_definitions(content)
    
    # 檢查重複
    print("檢查重複定義...")
    print()
    
    has_duplicates = False
    for func_name in imports:
        # 查找該函數的所有定義
        matches = [(name, line) for name, line in definitions if name == func_name]
        
        if matches:
            has_duplicates = True
            print(f"❌ {func_name}")
            print(f"   導入自: {imports[func_name]}")
            print(f"   但在腳本中重新定義了 {len(matches)} 次:")
            for name, line in matches:
                print(f"      第 {line} 行")
            print()
    
    print("="*70)
    if has_duplicates:
        print("❌ 發現重複定義！")
        print()
        print("🔧 修復建議:")
        print("  1. 刪除腳本中的重複函數定義")
        print("  2. 只保留 import 語句")
        print("  3. 確保使用模組中的版本")
        print()
        print("  參考: pinnx/dataio/loaders/kolmogorov.py")
        print("="*70)
        sys.exit(1)
    else:
        print("✅ 沒有重複定義")
        print("="*70)
        sys.exit(0)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""JSON IO 统一收口（N-1，2026-10 批次 N：同款 bug 收尾）。

8 个 service 原本各自实现 `_load_json` + `_save_json`（重复 8 份），
本批次收口到本模块，避免：

  - **B 半写损坏**：断电/磁盘满/进程 kill -9 时 `_save_json` 写入中断，
    损坏数据无法继续；`save_json_atomic` 先写 `.tmp` 再 `os.replace` 原子切换。
  - **C 无备份**：损坏发生时没有"上一轮成功状态"可回滚；
    `save_json_atomic` 默认写前 `shutil.copy2` 备份 `.bak`（保留 mtime）。

设计要点：
  - `load_json` 让 `JSONDecodeError` 自然 raise（与各 service 现有行为一致，**不吞错**）
  - `save_json_atomic` 备份失败仅 warn 不阻断写（与 buckets.json L 修复一致）
  - `.bak` 命名：`path.with_suffix(path.suffix + ".bak")` → 同目录、同名、易查找
  - `mkdir(parents=True, exist_ok=True)` + `indent=2` + `ensure_ascii=False`
    行为与各 service 旧的 `_save_json` 完全等价（除原子性外）
  - `.tmp` 前缀 `.` 让临时文件在 `ls` 默认隐藏，不污染目录视图

参考范式：`utils/zip_guard.py`（S-01/S-02 收口，统一 `is_safe_member` / `guarded_extractall`）。
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def load_json(path: Path) -> dict:
    """同步加载 JSON。`FileNotFoundError` / `JSONDecodeError` 由调用方决定如何处理（不吞错）。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: Path, data: dict, *, backup: bool = True) -> None:
    """原子写：写前备份（可选）+ `.tmp` + `os.replace`。

    与原各 service `_save_json` 行为差异：

      - 原版：`open(path, "w") + json.dump(...)`，写入过程中断 → 半写损坏（B 风险）。
      - 新版：先写 `.tmp` → `os.replace` 原子切换 → 即使中断也不影响原文件。

    Args:
        path: 目标 JSON 文件路径。
        data: 要写入的字典数据。
        backup: 是否在写入前备份 `path` 为 `<path>.bak`。默认 True。
               备份失败仅 warn（与 buckets.json L 修复行为一致），不阻断写入。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # 写前备份（保留 mtime）
    if backup and path.exists():
        try:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        except Exception as e:
            logger.warning(f"备份失败（继续写）{path}: {type(e).__name__}: {e}")

    # 原子写：mkstemp 创建 `.tmp` → fdopen 写入 → os.replace 原子切换
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".json.tmp", dir=str(path.parent), prefix="."
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        # 写入失败：清理 .tmp，避免残留
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise
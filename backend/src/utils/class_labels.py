"""类别标签读取/合并工具（算法库标签字典 ↔ 数据集类别清单的唯一口径）

背景（#6）：算法库的 labels_dict.json 是实时数据，数据集 meta.classes 与标注任务
task.json 的 classes 都是历史快照。库侧新增/改名标签后，快照不会传播，表现为
"库加了标签，数据集页面看不到"。本模块提供统一的按 index 定位读取 + 按位置合并，
让所有对外展示与封板口径都跟随库字典，且不需要回写存量数据。

关键约束：YOLO 里 class_id 是「下标即类别号」，因此
1. 绝不因为某行没有 english_code 就跳过该行（否则后续类别整体左移 → 静默串号）；
2. 合并时绝不截断（截断会让已落盘的 class_id 越界）。
"""
import json
import re
from pathlib import Path
from typing import Optional

from src.core.settings import settings

# 占位类别名（class_0 / class1）：只用于"这串名字没有真实含义"的判定，
# 绝不能拿去和标签字典做名称匹配。
PLACEHOLDER_CLASS_RE = re.compile(r"^class_?\d+$")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_placeholder_classes(classes) -> bool:
    """类别清单是否全是占位名（class_0/class_1...）。空清单返回 False。"""
    items = [str(c).strip() for c in (classes or []) if str(c).strip()]
    return bool(items) and all(PLACEHOLDER_CLASS_RE.match(c) for c in items)


def resolve_library_root_id(model_id: str) -> Optional[str]:
    """模型 id → 所属算法库根记录 id（换算规则由 ModelService 统一维护）。"""
    if not model_id:
        return None
    try:
        from src.services.model_service import ModelService
        return ModelService().resolve_library_root_id(model_id)
    except Exception:
        return None


def read_label_rows(model_id: str) -> list:
    """读取模型标签字典的标签行（按 index 升序）。

    自身目录没有字典时才回退到所属算法库根目录再读一次（回退要多读一份 model.json，
    故做成惰性）；都读不到返回 []。
    """
    if not model_id:
        return []
    rows = _read_label_rows_at(model_id)
    if rows:
        return rows
    root_id = resolve_library_root_id(model_id)
    if root_id and root_id != model_id:
        rows = _read_label_rows_at(root_id)
    return rows


def _read_label_rows_at(model_id) -> list:
    try:
        dic = _load_json(settings.REGISTRY_DIR / str(model_id) / "labels_dict.json")
    except Exception:
        return []
    labels = dic.get("labels") or []
    if not labels:
        return []
    try:
        return sorted(labels, key=lambda x: x.get("index") if isinstance(x.get("index"), int) else 0)
    except Exception:
        return list(labels)


def model_class_names(model_id: str) -> list:
    """标签字典 → 「下标即 YOLO 类别号」的类别名列表。

    每行取 english_code，缺失退 chinese_name，再缺失补 class_{index} 占位；
    绝不跳行，保证返回列表的下标与训练标注里的 class_id 一一对应。
    字典为空/无归属模型 → 返回 []（调用方自行回退到数据集快照）。
    """
    rows = read_label_rows(model_id)
    if not rows:
        return []
    size = 0
    for pos, row in enumerate(rows):
        idx = row.get("index")
        idx = idx if isinstance(idx, int) and idx >= 0 else pos
        size = max(size, idx + 1)
    names = [f"class_{i}" for i in range(size)]
    for pos, row in enumerate(rows):
        idx = row.get("index")
        if not isinstance(idx, int) or idx < 0:
            idx = pos
        nm = str(row.get("english_code") or row.get("chinese_name") or "").strip()
        names[idx] = nm or f"class_{idx}"
    return names


def merge_classes_with_model(existing, live) -> list:
    """数据集快照类别 ← 库字典实时类别，按 index 位置合并，绝不截断。

    - live 为空（无归属模型/未配字典）→ 原样返回 existing，行为与改动前一致
    - live 全是占位名而 existing 有真名 → 保留 existing（不能被假名盖掉）
    - live 更长 → 末尾追加（库新增标签立即可见）
    - live 更短 → 保留 existing 尾部（截断会让已落盘 class_id 越界）
    """
    ex = [str(c).strip() for c in (existing or [])]
    lv = [str(c).strip() for c in (live or [])]
    if not [c for c in lv if c]:
        return ex
    if not [c for c in ex if c]:
        return lv
    if is_placeholder_classes(lv) and not is_placeholder_classes(ex):
        return ex
    return lv + ex[len(lv):]

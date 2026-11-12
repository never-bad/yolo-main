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


# --------------------------------------------------------------------------- #
# 类别契约（B-64）：全平台唯一的「这个名字是不是这一类」判定口径
#
# 此前 dataset_service._build_class_map（简单 lower 精确匹配）、
# dataset_service._auto_match_model（去分隔符归一化 + 覆盖率）、
# sam_service._trained_class_map（四条别名兜底）各写了一套规则，
# 且匹配失败一律「放行」，导致不同环节对同一对名字给出不同结论。
# 这里把判定收敛成一个函数，所有调用方只消费它的结论。
# --------------------------------------------------------------------------- #

def norm_class_name(name) -> str:
    """类别名归一化：小写 + 去掉所有非 [0-9a-z] 字符。

    traffic-scene / traffic_scene / TrafficScene → trafficscene。
    注意中文会被全部去掉，所以中文别名同时用「原文小写」作为第二个 key。
    """
    return re.sub(r"[^0-9a-z]+", "", str(name or "").strip().lower())


def _alias_keys(name) -> list:
    """一个名字参与匹配的所有 key：原文小写 + 归一化名（去重、去空）。"""
    s = str(name or "").strip().lower()
    n = norm_class_name(s)
    return list(dict.fromkeys(k for k in (s, n) if k))


def build_alias_index(rows=None, canonical_names=None) -> tuple:
    """构造 (别名索引, {class_id: 规范名})。

    - rows: labels_dict.json 的 labels 行（五字段），别名优先级
      english_code → chinese_name → chinese_desc → ai_prompt_en；
      同一个 key 命中多个 class_id 时**保留全部**，由 resolve_class 判为歧义，
      绝不「先到先得」替用户猜。
    - canonical_names: 没有字典行时的退化输入（仅类别名清单）。
    返回 index: {key: {class_id, ...}}，names: {class_id: canonical}
    """
    index, names = {}, {}
    if rows:
        for pos, row in enumerate(rows):
            idx = row.get("index")
            idx = idx if isinstance(idx, int) and idx >= 0 else pos
            canonical = str(
                row.get("english_code") or row.get("chinese_name") or ""
            ).strip() or f"class_{idx}"
            names[idx] = canonical
            for alias in (row.get("english_code"), row.get("chinese_name"),
                          row.get("chinese_desc"), row.get("ai_prompt_en"),
                          *(row.get("aliases") or [])):
                for k in _alias_keys(alias):
                    index.setdefault(k, set()).add(idx)
    elif canonical_names:
        for idx, nm in enumerate(canonical_names):
            names[idx] = str(nm or "").strip()
            for k in _alias_keys(nm):
                index.setdefault(k, set()).add(idx)
    return index, names


def resolve_class(name, alias_index, names) -> dict:
    """单个外部类名 → 唯一结论。

    返回 status:
      exact      原文就是规范名（或归一化后完全相等）
      alias      通过字典别名 / 归一化命中
      ambiguous  同时命中多个 class_id，必须人工选
      unknown    认不出，绝不猜
    """
    hits = set()
    for k in _alias_keys(name):
        hits |= alias_index.get(k) or set()
    raw = str(name or "").strip().lower()
    if len(hits) == 1:
        cid = next(iter(hits))
        canonical = names.get(cid, "")
        is_exact = bool(raw) and raw == str(canonical).strip().lower()
        return {
            "status": "exact" if is_exact else "alias",
            "class_id": cid,
            "canonical": canonical,
            "candidates": [cid],
        }
    if len(hits) > 1:
        return {
            "status": "ambiguous", "class_id": None, "canonical": None,
            "candidates": sorted(hits),
        }
    return {"status": "unknown", "class_id": None, "canonical": None, "candidates": []}


def class_alignment(src_classes, model_id: str = "", rows=None,
                    canonical_names=None) -> dict:
    """上传包类名清单 → 与算法库标签字典的对齐报告（导入闸门的唯一数据源）。

    status:
      aligned       全部类都有唯一归属，class_map 可直接用于重映射
      needs_confirm 存在歧义/认不出的类，class_map 仅为「推荐」，
                    调用方**不得**据此做部分重映射，必须等人确认
      no_model      无归属库 / 库字典为空，无从对齐
    """
    if model_id and not rows:
        rows = read_label_rows(model_id)
    if not canonical_names:
        canonical_names = model_class_names(model_id) if model_id else None
    index, names = build_alias_index(rows, canonical_names)
    if not names:
        return {
            "status": "no_model", "model_id": model_id or None,
            "source_classes": [str(c) for c in (src_classes or [])],
            "canonical_names": [], "pairs": [], "class_map": {},
            "unresolved": [], "summary": "数据集没有归属算法库，或该库尚未配置标签字典。",
        }
    ordered_names = [names[i] for i in sorted(names)]
    pairs, class_map, unresolved = [], {}, []
    for i, nm in enumerate(src_classes or []):
        r = resolve_class(nm, index, names)
        pair = {"src_index": i, "src_name": str(nm), **r}
        if r["class_id"] is not None:
            class_map[str(i)] = r["class_id"]
        else:
            unresolved.append(i)
            pair["target_name"] = None
        pairs.append(pair)
    if unresolved:
        summary = (
            f"有 {len(unresolved)} 个类别无法唯一对齐到本库标签字典："
            + "、".join(str(pairs[i]["src_name"]) for i in unresolved[:5])
            + ("…" if len(unresolved) > 5 else "")
            + "。请确认每个类别对应的库内标签（或选择忽略该类）后再封板/训练。"
        )
        status = "needs_confirm"
    else:
        summary = f"全部 {len(pairs)} 个类别已对齐到本库标签字典：{ordered_names}"
        status = "aligned"
    return {
        "status": status,
        "model_id": model_id or None,
        "source_classes": [str(c) for c in (src_classes or [])],
        "canonical_names": ordered_names,
        "pairs": pairs,
        "class_map": class_map,
        "unresolved": unresolved,
        "summary": summary,
    }

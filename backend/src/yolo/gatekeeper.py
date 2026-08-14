#!/usr/bin/env python
"""
模型守门员（Model Gatekeeper）评估核心。

用于自动化 MLOps 流程中的模型替换决策：
- 训练产出新的 best.pt 后，与"上一代同业务生产模型（Old_Model）"在同一个评估集上对比
- 综合三个维度：整体精度（mAP50-95 严格更优）、防偏科（逐类 AP 相对跌幅 <= 5%）、推理性能（参考展示）
- 判定结果：
    promoted      晋升：新模型成为生产版本，版本号递增
    rejected      淘汰：归档为失败实验，旧模型继续服役（支持人工强制覆盖 Override）
    first_version 首版：该业务第一个模型，直接晋升为 v1.0

该模块为纯函数、无 FastAPI 依赖，可被训练脚本（train_script.py）在训练容器进程内直接调用，
因为权重与数据集文件都位于训练容器中。
"""
from pathlib import Path


# 防偏科阈值：单类 AP 相对跌幅超过该比例即判定"灾难性遗忘/退化"
# 全部类别统一校验（用户无需也不可配置）；类上原本无输出（old_ap≈0）的类别不参与比较
CLASS_REGRESSION_RATIO = 0.05

# 业务/算法类型自动推断规则：根据数据集类别名（data.yaml names）关键词匹配。
# 规则按命中类别数打分，得分最高者胜出；一个都不命中则归为 general（通用）。
# 前端无需手动选择业务场景——选中数据集后系统自动分配。
BUSINESS_RULES = [
    # (业务标识, 关键词列表，覆盖中英文类别名)
    ("defect", [
        "defect", "scratch", "crack", "hole", "stain", "dent", "blowhole",
        "corrosion", "burr", "blemish", "impurity",
        "缺陷", "瑕疵", "划痕", "划伤", "裂纹", "裂痕", "凹陷", "砂眼",
        "气泡", "脏污", "斑点", "麻点", "污渍",
    ]),
    ("pedestrian", [
        "person", "people", "pedestrian", "head", "human", "walk",
        "行人", "人", "人头", "人体", "行人头",
    ]),
    ("vehicle", [
        "car", "vehicle", "truck", "bus", "motorcycle", "bicycle",
        "plate", "license", "van", "ambulance", "taxi",
        "车", "车辆", "汽车", "货车", "大巴", "出租车", "车牌", "卡车",
    ]),
    ("package", [
        "package", "parcel", "box", "carton", "cardboard", "cargo", "parcel_box",
        "包裹", "快递", "快递盒", "纸箱", "货箱",
    ]),
]

def infer_business(names) -> str:
    """根据数据集类别名自动推断业务/算法类型；识别不出返回 general（通用）。

    - names: data.yaml 的 names（支持 list 或 dict）
    - 对每个预置业务统计命中的类别数，命中类别最多的业务胜出；
      平局或全部未命中时按规则顺序取首个命中的业务，一个都不命中为 general
    """
    labels = _normalize_classes(names)
    if not labels:
        return "general"
    labels_lower = [str(l).lower() for l in labels]

    best_biz, best_score = "general", 0
    for biz, keywords in BUSINESS_RULES:
        score = sum(1 for label in labels_lower if any(k in label for k in keywords))
        if score > best_score:
            best_biz, best_score = biz, score
    return best_biz


def _normalize_classes(names) -> list:
    """将 data.yaml 的 names（可能是 list 或 dict）归一化为按类 ID 排序的类别名列表"""
    if names is None:
        return []
    if isinstance(names, dict):
        max_id = max(int(k) for k in names.keys())
        return [str(names.get(str(i), names.get(i, f"class_{i}"))) for i in range(max_id + 1)]
    if isinstance(names, list):
        return [str(n) for n in names]
    return []


def _resolve_weights(weights: str) -> str:
    """解析权重路径：绝对路径存在直接用；否则在 backend/models 下查找；找不到原样返回（交给 ultralytics）"""
    p = Path(weights)
    if p.is_absolute() and p.exists():
        return str(p)
    backend_dir = Path(__file__).resolve().parent.parent.parent
    for d in [backend_dir / "models" / "custom", backend_dir / "models" / "sam", backend_dir]:
        cand = d / weights
        if cand.exists():
            return str(cand)
    return weights


def _pick_eval_split(data_yaml: str) -> str:
    """绝对测试集优先：若 data.yaml 已配置独立 test 则用 test，否则用 val 验证集"""
    split = "val"
    try:
        import yaml as _yaml
        with open(data_yaml, "r", encoding="utf-8") as f:
            dc = _yaml.safe_load(f) or {}
        if dc.get("test"):
            split = "test"
    except Exception:
        pass
    return split


def evaluate_model(weights: str, data_yaml: str, classes: list) -> dict:
    """在数据集评估集（优先独立 test，其次 val）上评估模型。

    Returns:
        {
            "mAP50": float|None, "mAP50_95": float|None,
            "precision": float|None, "recall": float|None,
            "speed_ms": float|None,           # 平均推理耗时（推理阶段，不含预处理/NMS）
            "class_ap": {class_name: ap},     # 逐类 AP@50:95
            "eval_split": "val"|"test",       # 实际使用的评估集
            "error": str|None
        }
    """
    from ultralytics import YOLO

    split = _pick_eval_split(data_yaml)

    try:
        model = YOLO(_resolve_weights(weights))
        # YOLO-World 等文本驱动模型需先 set_classes 才能按数据集类别评估
        try:
            if hasattr(model.model, "set_classes") and classes:
                model.set_classes(classes)
        except Exception:
            pass
        res = model.val(data=str(data_yaml), split=split, verbose=False, plots=False)
        m = res.box
        speed = getattr(res, "speed", None) or {}
        speed_ms = None
        if isinstance(speed, dict) and speed.get("inference") is not None:
            try:
                speed_ms = round(float(speed["inference"]), 2)
            except Exception:
                pass

        # 逐类 AP：优先 ap + ap_class_index（稳定字段），回退 maps
        class_ap = {}
        try:
            names = _normalize_classes(classes)
            ap_index = list(getattr(m, "ap_class_index", None) or [])
            ap_vals = list(getattr(m, "ap", None) or [])
            if ap_index and ap_vals and len(ap_vals) >= len(ap_index):
                for idx, ap_val in zip(ap_index, ap_vals):
                    i = int(idx)
                    name = names[i] if i < len(names) else f"class_{i}"
                    class_ap[name] = round(float(ap_val), 4)
            elif getattr(m, "maps", None) is not None:
                for i, v in enumerate(m.maps):
                    if v is None:
                        continue
                    name = names[i] if i < len(names) else f"class_{i}"
                    class_ap[name] = round(float(v), 4)
        except Exception:
            class_ap = {}

        return {
            "mAP50": None if m.map50 is None else round(float(m.map50), 4),
            "mAP50_95": None if m.map is None else round(float(m.map), 4),
            "precision": None if m.mp is None else round(float(m.mp), 4),
            "recall": None if m.mr is None else round(float(m.mr), 4),
            "speed_ms": speed_ms,
            "class_ap": class_ap,
            "eval_split": split,
            "error": None,
        }
    except Exception as e:
        return {
            "mAP50": None, "mAP50_95": None, "precision": None, "recall": None,
            "speed_ms": None, "class_ap": {}, "eval_split": split, "error": str(e),
        }


def run_gatekeeper(baseline_weights, new_weights, data_yaml, classes) -> dict:
    """执行守门员评估，返回完整决策报告。

    Args:
        baseline_weights: 上一代同业务生产模型（Old_Model）权重路径；None/不存在表示首版
        new_weights: 刚训练产出的 best.pt 路径
        data_yaml: 训练所用数据集的 data.yaml（同一评估集对比）
        classes: 类别名列表（训练时已从 data.yaml 读取）

    Returns:
        {
            "result": "promoted" | "rejected" | "first_version",
            "promoted": bool,
            "eval_split": "val" | "test",
            "new_metrics": {...}, "old_metrics": {...}|None,
            "class_ap": {class_name: {"old_ap","new_ap","delta_pct","regressed"}},
            "regressed_classes": [names],
            "report": str  # 中文诊断报告（markdown 文本）
        }
    """
    # ---------- 1. 评估新模型 ----------
    new_metrics = evaluate_model(new_weights, data_yaml, classes)

    # 新模型评估失败 → 不晋升（保守处理）
    if new_metrics.get("error") or new_metrics.get("mAP50_95") is None:
        return _build_result(
            "rejected", new_metrics, None, {}, [],
            "新模型在评估集上评估失败，无法通过守门员校验。可人工强制覆盖，或检查模型/数据集文件。"
        )

    # ---------- 2. 首版：无旧基准直接晋升 ----------
    if not baseline_weights or not Path(_resolve_weights(baseline_weights)).exists():
        return _build_result(
            "first_version", new_metrics, None, {}, [],
            "该业务首个模型：无旧基准可对比，直接晋升为生产版本 v1.0。"
        )

    # ---------- 3. 评估旧模型（同业务上一代生产模型） ----------
    old_metrics = evaluate_model(baseline_weights, data_yaml, classes)
    if old_metrics.get("error") or old_metrics.get("mAP50_95") is None:
        return _build_result(
            "rejected", new_metrics, old_metrics, {}, [],
            "旧基准模型（上一代生产模型）在评估集上评估失败，无法完成对比；为保证生产质量暂不晋升（可人工强制覆盖）。"
        )

    # ---------- 4. 多维度判定 ----------
    new_map = new_metrics["mAP50_95"]
    old_map = old_metrics["mAP50_95"]

    # 4.1 整体精度：严格大于（浮点加微小 eps 防相等抖动）
    if new_map <= old_map + 1e-6:
        return _build_result(
            "rejected", new_metrics, old_metrics, {}, [],
            f"新模型 mAP50-95 = {new_map:.4f}，未严格超过旧模型 {old_map:.4f}，拒绝晋升。"
        )

    # 4.2 防偏科/灾难性遗忘：逐类对比 AP@50:95，相对跌幅 > 5% 判定退化
    class_ap = {}
    regressed = []
    all_class_names = sorted(
        set(list(new_metrics.get("class_ap", {}).keys()) + list(old_metrics.get("class_ap", {}).keys()))
    )
    for name in all_class_names:
        old_ap = old_metrics.get("class_ap", {}).get(name)
        new_ap = new_metrics.get("class_ap", {}).get(name)
        row = {"old_ap": old_ap, "new_ap": new_ap, "delta_pct": None, "regressed": False}
        if old_ap is not None and old_ap > 0.01:
            if new_ap is None:
                # 旧模型该类别有检出，新模型完全检不出 → 视为严重退化
                row["delta_pct"] = -100.0
                row["regressed"] = True
            else:
                delta = (new_ap - old_ap) / old_ap
                row["delta_pct"] = round(delta * 100, 1)
                row["regressed"] = delta < -CLASS_REGRESSION_RATIO
            if row["regressed"]:
                regressed.append(name)
        class_ap[name] = row

    if regressed:
        detail = "; ".join(
            f"{n}（{class_ap[n]['old_ap'] * 100:.1f}% → {class_ap[n]['new_ap'] * 100:.1f}%）"
            for n in regressed
        )
        return _build_result(
            "rejected", new_metrics, old_metrics, class_ap, regressed,
            f"以下类别出现明显退化（跌幅超过 {CLASS_REGRESSION_RATIO * 100:.0f}%）：{detail}。"
            "整体 mAP 提升但关键类别断崖下跌，疑为数据质量问题或灾难性遗忘，拒绝晋升（可人工强制覆盖）。"
        )

    # 4.3 通过 → 晋升
    speed_note = ""
    if new_metrics.get("speed_ms") is not None and old_metrics.get("speed_ms") is not None:
        speed_note = (
            f"推理耗时 {new_metrics['speed_ms']}ms vs 旧模型 {old_metrics['speed_ms']}ms"
            + ("（更快）" if new_metrics["speed_ms"] < old_metrics["speed_ms"] else "（更慢，请结合实际部署要求评估）")
        )
    return _build_result(
        "promoted", new_metrics, old_metrics, class_ap, [],
        f"新模型 mAP50-95 = {new_map:.4f}（比旧模型 {old_map:.4f} 提升 {(new_map - old_map) * 100:.1f}%），"
        f"且所有类别均未出现超过 {CLASS_REGRESSION_RATIO * 100:.0f}% 的退化。{speed_note}"
    )


def _build_result(result, new_metrics, old_metrics, class_ap, regressed_classes, verdict) -> dict:
    """组装返回结构（统一格式）"""
    return {
        "result": result,
        "promoted": result in ("promoted", "first_version"),
        "eval_split": new_metrics.get("eval_split", "val") if new_metrics else "val",
        "new_metrics": new_metrics,
        "old_metrics": old_metrics,
        "class_ap": class_ap,
        "regressed_classes": regressed_classes,
        "report": verdict,
    }
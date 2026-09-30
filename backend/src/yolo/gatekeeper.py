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

# M7 守门员增强（P5 拍板）：晋升 Margin —— 仅当 new_mAP > historical_best_mAP + Margin 才放行，
# 避免"持平即晋升"的噪声抖动；0.5% 为训练模块规范口径。
MARGIN = 0.005

# ---- Loss 稳定性校验（M7 / 2.1）----
# 从 results.csv 的 box_loss + val/box_loss 序列判断训练是否震荡/发散：
#   - Loss 发散：末尾 Loss 相对窗口内谷值抬升超过 LOSS_RISE_RATIO（默认 20%）
#   - Loss 震荡：近 LOSS_WINDOW 个 epoch 的 CV（标准差/均值）超过 LOSS_VOLATILITY（默认 15%）
# 任一命中判定 Unstable → 拒绝自动上线，并附 Data Cleaning Suggestion。
LOSS_WINDOW = 10
LOSS_RISE_RATIO = 0.20
LOSS_VOLATILITY = 0.15

# M7/M8 最优模型选择：Loss 不稳定时 last.pt 相对 best.pt 的最大可接受跌幅（相对 %）
BEST_LAST_GAP = 0.02
# Loss 稳定性校验至少需要的最小 epoch 数（低于则判定 no_data 不阻断）
BEST_REQUIRED_EPOCHS = 3

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
            "mask_mAP50": float|None,         # 分割轮廓指标（仅 segment 模型有值，检测模型为 None）
            "mask_mAP50_95": float|None,
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

        def _num(v):
            """指标安全取数：None / 非数字一律返回 None，不抛异常"""
            try:
                return None if v is None else round(float(v), 4)
            except (TypeError, ValueError):
                return None

        # 分割模型才有 res.mask（轮廓质量指标）；检测模型为 None。
        # 只作为附加信息展示与留档，不参与晋升判定（判定仍以方框 mAP50-95 为准）。
        mk = getattr(res, "mask", None)
        mask_map50 = _num(getattr(mk, "map50", None)) if mk is not None else None
        mask_map50_95 = _num(getattr(mk, "map", None)) if mk is not None else None

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
            "mask_mAP50": mask_map50,
            "mask_mAP50_95": mask_map50_95,
            "speed_ms": speed_ms,
            "class_ap": class_ap,
            "eval_split": split,
            "error": None,
        }
    except Exception as e:
        return {
            "mAP50": None, "mAP50_95": None, "precision": None, "recall": None,
            "mask_mAP50": None, "mask_mAP50_95": None,
            "speed_ms": None, "class_ap": {}, "eval_split": split, "error": str(e),
        }


def check_loss_stability(train_dir) -> dict:
    """M7/2.1：从训练输出 results.csv 校验 Loss 稳定性。

    Args:
        train_dir: 训练输出目录（含 results.csv，YOLO 每 epoch 一行）

    Returns:
        {
            "stable": bool,
            "check": "loss_unstable"|"loss_volatile"|"stable"|"no_data",
            "detail": str,
            "window": int, "last_values": [..],
        }
        无法读取结果文件时（早期失败/CPU 验证集缺失等）返回 no_data（不阻断上线）。
    """
    import csv
    import statistics as _stats
    results_csv = Path(train_dir) / "results.csv"
    if not results_csv.exists():
        return {"stable": True, "check": "no_data", "detail": "无 results.csv（可能训练提前失败）", "window": 0, "last_values": []}

    try:
        with open(results_csv, "r", encoding="utf-8", errors="ignore") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        return {"stable": True, "check": "no_data", "detail": f"results.csv 解析失败: {e}", "window": 0, "last_values": []}
    if not rows:
        return {"stable": True, "check": "no_data", "detail": "results.csv 为空", "window": 0, "last_values": []}

    col = None
    for cand in ("box_loss", "val/box_loss", "loss"):
        if cand in rows[0]:
            col = cand
            break
    if not col:
        return {"stable": True, "check": "no_data", "detail": "results.csv 无可用 Loss 列", "window": 0, "last_values": []}

    vals = []
    for r in rows:
        try:
            vals.append(float(r[col]))
        except (TypeError, ValueError):
            continue
    vals = [v for v in vals if v == v]  # 去 NaN
    window = min(LOSS_WINDOW, len(vals))
    if window < BEST_REQUIRED_EPOCHS:
        return {"stable": True, "check": "no_data", "detail": f"Loss 数据不足（仅 {len(vals)} 个 epoch）", "window": len(vals), "last_values": vals[-10:]}

    tail = vals[-window:]
    mean = _stats.fmean(tail)
    if mean <= 0:
        return {"stable": True, "check": "no_data", "detail": "Loss 序列非正值，跳过稳定性校验", "window": window, "last_values": tail}

    std = _stats.pstdev(tail) if len(tail) > 1 else 0.0
    cv = std / mean
    trough = min(tail)
    last = tail[-1]
    rise = (last - trough) / trough if trough > 0 else 0.0

    if rise > LOSS_RISE_RATIO:
        return {
            "stable": False, "check": "loss_unstable",
            "detail": f"{col} 末尾 {last:.4f} 相对窗口谷值 {trough:.4f} 抬升 {rise * 100:.1f}%"
                      f"（阈值 {LOSS_RISE_RATIO * 100:.0f}%），Loss 发散，判定训练不稳定。",
            "window": window, "last_values": tail,
        }
    if cv > LOSS_VOLATILITY:
        return {
            "stable": False, "check": "loss_volatile",
            "detail": f"{col} 近 {window} 轮 CV={cv * 100:.1f}%（阈值 {LOSS_VOLATILITY * 100:.0f}%），"
                      f"Loss 剧烈震荡，判定训练不稳定。",
            "window": window, "last_values": tail,
        }
    return {
        "stable": True, "check": "stable",
        "detail": f"{col} 近 {window} 轮收敛稳定（末值 {last:.4f}，CV {cv * 100:.1f}%）",
        "window": window, "last_values": tail,
    }


def run_gatekeeper(baseline_weights, new_weights, data_yaml, classes, margin: float = MARGIN) -> dict:
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

    # 4.1 整体精度（M7 增强）：对比"历史最佳生产模型"，要求胜出 Margin（默认 0.5%）
    # 浮点加微小 eps 防完全相等时被 reject（相等本就不达标，无需区分）
    target = old_map + margin
    if new_map <= target + 1e-6:
        return _build_result(
            "rejected", new_metrics, old_metrics, {}, [],
            f"新模型 mAP50-95 = {new_map:.4f}，未超过历史最佳 {old_map:.4f} + "
            f"Margin({margin * 100:.1f}%) = {target:.4f}，拒绝晋升。"
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
        f"新模型 mAP50-95 = {new_map:.4f}（超历史最佳 {old_map:.4f} "
        f"{((new_map - old_map) / old_map) * 100:.1f}%，满足 Margin({margin * 100:.1f}%)），"
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


# ---- M7/M8 最优模型选择（Loss 判据接入）：----
# ultralytics 的 best.pt 默认按验证集指标自动保存；若训练末尾恰好进入 Loss 震荡/发散窗口，
# 指标尖峰对应的权重未必是最优选择。这里在复制落盘前做一次"Loss 加权选择"：
#   - 训练稳定 → 直接用 best.pt（常规路径，零额外开销）
#   - Loss 不稳定 且 last.pt 存在 → 在 val 集上重评估 last.pt，
#     若其 mAP 跌落不超过 BEST_LAST_GAP（相对跌幅 2%）则改选 last.pt（收敛末尾权重更稳）

def select_best_weights(train_dir, data_yaml, classes) -> dict:
    """从训练输出中选出最优权重（M7 最优模型选择：Loss 判据优先于刻板指标尖峰）。

    Args:
        train_dir: 训练输出目录（含 weights/best.pt、weights/last.pt、results.csv）
        data_yaml: 训练数据集 data.yaml（评估集口径与训练一致）
        classes: 类别名列表

    Returns:
        {
            "weights": str,          # 选定权重的绝对路径（best.pt 或 last.pt）
            "selected": "best" | "last",
            "reason": str,           # 中文选择原因（写入 gatekeeper 报告）
            "loss_stability": dict,  # check_loss_stability 原始结果
        }
        常规（训练稳定 / 无 last.pt）→ 返回 best.pt，reason 说明未触发重选。
    """
    train_dir_p = Path(train_dir)
    best_pt = train_dir_p / "weights" / "best.pt"
    last_pt = train_dir_p / "weights" / "last.pt"

    if not best_pt.exists():
        if last_pt.exists():
            return {
                "weights": str(last_pt.resolve()),
                "selected": "last",
                "reason": "未找到 best.pt，采用训练末尾权重 last.pt。",
                "loss_stability": {"stable": True, "check": "no_data",
                                   "detail": "无 best.pt", "window": 0, "last_values": []},
            }
        raise FileNotFoundError(f"训练输出缺少权重文件（best.pt / last.pt 均不存在）: {train_dir_p}")

    ls = check_loss_stability(train_dir_p)

    # 训练稳定 / 无末尾 checkpoint → 直接采用 ultralytics best.pt
    if ls.get("stable") or not last_pt.exists():
        return {
            "weights": str(best_pt.resolve()),
            "selected": "best",
            "reason": f"训练 Loss 稳定（{ls.get('detail', '')}），采用 ultralytics 自动选优的 best.pt。",
            "loss_stability": ls,
        }

    # Loss 不稳定且存在 last.pt：重评估 last.pt（val），若跌幅可接受则改选 last.pt
    try:
        best_metrics = evaluate_model(str(best_pt), data_yaml, classes)
        last_metrics = evaluate_model(str(last_pt), data_yaml, classes)
    except Exception as e:
        return {
            "weights": str(best_pt.resolve()),
            "selected": "best",
            "reason": f"训练 Loss 不稳定但 last.pt 重评估失败（{e}），保持 ultralytics best.pt。",
            "loss_stability": ls,
        }

    best_map = best_metrics.get("mAP50_95")
    last_map = last_metrics.get("mAP50_95")
    if best_map is not None and last_map is not None and last_map > 0:
        drop = (best_map - last_map) / last_map
        if drop <= BEST_LAST_GAP:
            return {
                "weights": str(last_pt.resolve()),
                "selected": "last",
                "reason": (f"训练 Loss 不稳定（{ls.get('detail', '')}），改选收敛末尾权重 last.pt"
                           f"（mAP50-95 {last_map:.4f}，相对 best.pt {best_map:.4f} 跌幅 {drop * 100:.1f}%"
                           f" ≤ 容忍 {BEST_LAST_GAP * 100:.0f}%），优先保证权重稳定性。"),
                "loss_stability": ls,
            }
        return {
            "weights": str(best_pt.resolve()),
            "selected": "best",
            "reason": (f"训练 Loss 不稳定，但 last.pt mAP50-95 {last_map:.4f} 相对 best.pt {best_map:.4f}"
                       f" 跌幅 {drop * 100:.1f}% 超过容忍 {BEST_LAST_GAP * 100:.0f}%，保留指标更优的 best.pt。"),
            "loss_stability": ls,
        }

    return {
        "weights": str(best_pt.resolve()),
        "selected": "best",
        "reason": f"训练 Loss 不稳定且无法完整对比 last.pt 指标（best={best_map}, last={last_map}），保持 best.pt。",
        "loss_stability": ls,
    }

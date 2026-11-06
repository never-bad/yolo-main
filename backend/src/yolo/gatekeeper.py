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

import glob
import os
import time
from collections import defaultdict


# 防偏科阈值：单类 AP 相对跌幅超过该比例即判定"灾难性遗忘/退化"
# 全部类别统一校验（用户无需也不可配置）；类上原本无输出（old_ap≈0）的类别不参与比较
CLASS_REGRESSION_RATIO = 0.05

# M7 守门员增强（P5 拍板）：晋升 Margin —— 仅当 new_mAP > historical_best_mAP + Margin 才放行，
# 避免"持平即晋升"的噪声抖动；0.5% 为训练模块规范口径。
MARGIN = 0.005

# 测试集评测（predict 版）置信度扫描点：用于 AP 的 PR 曲线与"最优 F1 阈值"扫描。
CONF_SCAN = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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


# --------------------------------------------------------------------------- #
# predict 版评测（B-54 引入）：直接 model.predict 逐图推理 + 自算 mAP/P/R/F1，
# 不依赖 data.yaml 的 train/val 键，用于仅含 test 的封板测试集。
# 返回字段与 evaluate_model 完全对齐，前端/matrix 无需改动。
# --------------------------------------------------------------------------- #
def _box_iou(boxA, boxB):
    """两个 [x1,y1,x2,y2] 框的 IoU。"""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA); interH = max(0.0, yB - yA)
    inter = interW * interH
    areaA = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def _compute_ap(recall, precision):
    """COCO 风格全点插值 AP（PR 曲线下面积）。"""
    if not recall:
        return 0.0
    mrec = [0.0] + list(recall) + [1.0]
    mpre = [0.0] + list(precision) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def _evaluate(preds, gts, iou_thr, conf_scan):
    """对单个类别计算 AP / conf 扫描 / 最优 F1 阈值。

    preds: [{'img','cls','conf','box':[x1,y1,x2,y2]}]
    gts:   [{'img','cls','box':[x1,y1,x2,y2]}]
    """
    gt_groups = defaultdict(list)
    for gi, g in enumerate(gts):
        gt_groups[(g['img'], g['cls'])].append(gi)

    order = sorted(range(len(preds)), key=lambda i: preds[i]['conf'], reverse=True)
    matched = defaultdict(set)
    tp = [0] * len(preds); fp = [0] * len(preds)
    for idx in order:
        p = preds[idx]
        gis = gt_groups.get((p['img'], p['cls']), [])
        best_iou = 0.0; best_gi = -1
        for gi in gis:
            if gi in matched[(p['img'], p['cls'])]:
                continue
            iou = _box_iou(p['box'], gts[gi]['box'])
            if iou > best_iou:
                best_iou = iou; best_gi = gi
        if best_gi >= 0 and best_iou >= iou_thr:
            matched[(p['img'], p['cls'])].add(best_gi)
            tp[idx] = 1
        else:
            fp[idx] = 1

    total_gt = len(gts)
    cum_tp = cum_fp = 0
    P_list = []; R_list = []
    for idx in order:
        cum_tp += tp[idx]; cum_fp += fp[idx]
        prec = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0
        rec = cum_tp / total_gt if total_gt > 0 else 0.0
        P_list.append(prec); R_list.append(rec)
    ap = _compute_ap(R_list, P_list)

    scan = {}
    for c in conf_scan:
        cut = len(order)
        for k, idx in enumerate(order):
            if preds[idx]['conf'] < c:
                cut = k; break
        if cut == 0:
            scan[c] = (0.0, 0.0, 0.0)
        else:
            ctp = sum(tp[order[k]] for k in range(cut))
            cfp = sum(fp[order[k]] for k in range(cut))
            prec = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0.0
            rec = ctp / total_gt if total_gt > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            scan[c] = (round(prec, 4), round(rec, 4), round(f1, 4))

    best_f1 = -1.0; best_conf = 0.0; best_pr = None
    for k in range(len(order)):
        ctp = sum(tp[order[j]] for j in range(k + 1))
        cfp = sum(fp[order[j]] for j in range(k + 1))
        prec = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0.0
        rec = ctp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1; best_conf = preds[order[k]]['conf']; best_pr = (prec, rec)

    return {
        "ap": ap, "scan": scan, "best_f1": best_f1, "best_conf": best_conf,
        "best_precision": (best_pr[0] if best_pr else None),
        "best_recall": (best_pr[1] if best_pr else None),
    }


def _evaluate_per_class(preds, gts, iou_thr, conf_scan):
    """逐类别评估并返回 (per_class_results, mAP)。"""
    classes = sorted(set([p['cls'] for p in preds]) | set([g['cls'] for g in gts]))
    per = {}
    aps = []
    for c in classes:
        p_c = [p for p in preds if p['cls'] == c]
        g_c = [g for g in gts if g['cls'] == c]
        res = _evaluate(p_c, g_c, iou_thr, conf_scan)
        per[c] = res
        aps.append(res['ap'])
    mAP = sum(aps) / len(aps) if aps else 0.0
    return per, mAP


def _pick_pref_conf_from_scan(scan, preference="miss"):
    """从 _evaluate 的 scan(conf->(prec,rec,f1)) 挑工作点 conf，按偏好偏召回/精确。

    - miss(在意漏检,默认): F1 峰值带内取最低 conf → 偏召回 / 低漏检率；
    - false(在意误检):     峰值带内取最高 conf → 偏精确 / 低误检率。
    scan 为空时返回 None（调用方退回默认 0.25）。
    """
    if not scan:
        return None
    try:
        confs = sorted(scan.keys())
        f1s = [scan[c][2] for c in confs]
        peak_f1 = max(f1s)
        if peak_f1 <= 0:
            return None
        lo = peak_f1 * (1.0 - 0.02)  # 2% 容差带：F1 接近峰值的 conf 视为等价
        band = [c for c, f in zip(confs, f1s) if f >= lo - 1e-9]
        if not band:
            return None
        return max(band) if preference == "false" else min(band)
    except Exception:
        return None


def _pr_at_conf(preds, gts, iou_thr, conf):
    """在给定工作点 conf 下的 P/R/F1（基于全候选框匹配）。"""
    gt_groups = defaultdict(list)
    for gi, g in enumerate(gts):
        gt_groups[(g['img'], g['cls'])].append(gi)
    order = sorted(range(len(preds)), key=lambda i: preds[i]['conf'], reverse=True)
    matched = defaultdict(set)
    tp = [0] * len(preds); fp = [0] * len(preds)
    for idx in order:
        p = preds[idx]
        gis = gt_groups.get((p['img'], p['cls']), [])
        best_iou = 0.0; best_gi = -1
        for gi in gis:
            if gi in matched[(p['img'], p['cls'])]:
                continue
            iou = _box_iou(p['box'], gts[gi]['box'])
            if iou > best_iou:
                best_iou = iou; best_gi = gi
        if best_gi >= 0 and best_iou >= iou_thr:
            matched[(p['img'], p['cls'])].add(best_gi)
            tp[idx] = 1
        else:
            fp[idx] = 1
    total_gt = len(gts)
    ctp = sum(tp[order[k]] for k in range(len(order)) if preds[order[k]]['conf'] >= conf)
    cfp = sum(fp[order[k]] for k in range(len(order)) if preds[order[k]]['conf'] >= conf)
    prec = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0.0
    rec = ctp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _empty_eval(split: str, conf, conf_preference, err) -> dict:
    """evaluate_model / evaluate_model_dir 统一错误/缺数据返回骨架。"""
    return {
        "mAP50": None, "mAP50_95": None, "precision": None, "recall": None,
        "f1": None, "miss_rate": None, "false_rate": None,
        "mask_mAP50": None, "mask_mAP50_95": None,
        "speed_ms": None, "speed_breakdown": None, "gpu_mem_mb": None,
        "class_ap": {}, "eval_split": split,
        "conf_used": conf if conf is not None else 0.001,
        "conf_preference": conf_preference, "error": err,
    }


def evaluate_model_dir(weights: str, imgs_dir: str, labels_dir: str, classes: list,
                       conf: float = None, conf_preference: str = "miss",
                       imgsz: int = 640, iou: float = 0.7,
                       device: str = "cuda:0") -> dict:
    """predict 版评测：直接推理指定图片目录 + 解析同目录 YOLO 标签算 mAP/P/R/F1。

    与 evaluate_model 返回字段完全一致（mAP50/mAP50_95/precision/recall/f1/
    miss_rate/false_rate/speed_ms/gpu_mem_mb/class_ap/eval_split/conf_used/
    conf_preference/error/mask_*），前端与测试集矩阵接口无需改动。

    不要求 data.yaml 含 train/val 键——封板测试集（仅 test 键）也能评。

    Args:
        imgs_dir:   测试图片目录（如 .../images/test）
        labels_dir: YOLO 格式标签目录（如 .../labels/test），按 base 同名 .txt 配对
        classes:    类别名列表（按 0 基 id 对应标签 class id）
        conf:       工作点置信度；None 时取"宏平均最优 F1 阈值"
        conf_preference / imgsz / iou / device：评测超参
    """
    from ultralytics import YOLO

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    names_norm = _normalize_classes(classes)

    def find_images(d):
        files = []
        for ext in IMG_EXTS:
            files += glob.glob(os.path.join(d, "*" + ext))
            files += glob.glob(os.path.join(d, "**", "*" + ext), recursive=True)
        return sorted(set(files))

    def parse_labels(label_dir, per_img):
        gts = []
        for base, info in per_img.items():
            lp = os.path.join(label_dir, base + ".txt")
            if not os.path.exists(lp):
                continue
            h, w = info["shape"]
            with open(lp, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cid = int(float(parts[0]))
                        cx, cy, ww, hh = map(float, parts[1:5])
                    except ValueError:
                        continue
                    if names_norm and cid >= len(names_norm):
                        continue
                    x1 = (cx - ww / 2) * w; y1 = (cy - hh / 2) * h
                    x2 = (cx + ww / 2) * w; y2 = (cy + hh / 2) * h
                    gts.append({"img": base, "cls": cid, "box": [x1, y1, x2, y2]})
        return gts

    try:
        model = YOLO(_resolve_weights(weights))
        try:
            if hasattr(model.model, "set_classes") and classes:
                model.set_classes(classes)
        except Exception:
            pass

        gpu0 = None
        try:
            import torch
            if "cuda" in (device or "") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        image_files = find_images(imgs_dir)
        if not image_files:
            return _empty_eval("test", conf, conf_preference, f"测试集图片目录为空: {imgs_dir}")

        per_img = {}
        preds = []
        times = []
        for f in image_files:
            base = os.path.splitext(os.path.basename(f))[0]
            t0 = time.perf_counter()
            results = model.predict(source=f, conf=1e-4, iou=iou,
                                    imgsz=imgsz, device=device, verbose=False)
            dt = time.perf_counter() - t0
            times.append(dt)
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                c = r.boxes.conf.cpu().numpy()
                cl = r.boxes.cls.cpu().numpy()
                for i in range(len(xyxy)):
                    preds.append({
                        "img": base, "cls": int(cl[i]),
                        "conf": float(c[i]),
                        "box": [float(x) for x in xyxy[i]],
                    })
            per_img[base] = {"shape": tuple(r.orig_shape)}

        gts = (parse_labels(labels_dir, per_img)
               if labels_dir and os.path.isdir(labels_dir) else [])
        has_gt = bool(gts)

        # ---- AP@0.5 与 AP@0.5:0.95（与 ultralytics 口径一致）----
        per5, mAP5 = _evaluate_per_class(preds, gts, 0.5, CONF_SCAN)
        iou_grid = [round(0.5 + 0.05 * i, 2) for i in range(10)]
        ap5095_per_class = {}
        for iou_t in iou_grid:
            per_t, _ = _evaluate_per_class(preds, gts, iou_t, CONF_SCAN)
            for c, res in per_t.items():
                ap5095_per_class.setdefault(c, []).append(res["ap"])
        if ap5095_per_class:
            class_ap = {
                names_norm[c]: round(sum(ap5095_per_class[c]) / len(ap5095_per_class[c]), 4)
                for c in ap5095_per_class
            }
            mAP50_95 = round(
                sum(sum(v) / len(v) for v in ap5095_per_class.values())
                / len(ap5095_per_class), 4)
        else:
            class_ap = {}
            mAP50_95 = 0.0
        mAP50 = round(mAP5, 4)

        # ---- 工作点 conf：缺省取宏平均最优 F1 阈值（按偏好偏召回/精确）----
        best_confs = []
        for c in per5:
            pc = _pick_pref_conf_from_scan(per5[c].get("scan"), conf_preference)
            if pc is not None:
                best_confs.append(pc)
        macro_best_conf = (sum(best_confs) / len(best_confs)) if best_confs else 0.25
        op_conf = conf if conf is not None else macro_best_conf

        if has_gt:
            p, r, f1 = _pr_at_conf(preds, gts, iou, op_conf)
        else:
            p = r = f1 = None

        mean_t = sum(times) / len(times) if times else 0.0
        try:
            import torch
            if "cuda" in (device or "") and torch.cuda.is_available():
                gpu0 = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        except Exception:
            pass

        return {
            "mAP50": mAP50 if has_gt else None,
            "mAP50_95": mAP50_95 if has_gt else None,
            "precision": round(p, 4) if p is not None else None,
            "recall": round(r, 4) if r is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "miss_rate": round(1 - r, 4) if r is not None else None,
            "false_rate": round(1 - p, 4) if p is not None else None,
            "mask_mAP50": None, "mask_mAP50_95": None,
            "speed_ms": round(mean_t * 1000, 2),
            "speed_breakdown": None,
            "gpu_mem_mb": gpu0,
            "class_ap": class_ap,
            "eval_split": "test",
            "conf_used": op_conf,
            "conf_preference": conf_preference,
            "error": None,
        }
    except Exception as e:
        return _empty_eval("test", conf, conf_preference, str(e))


def evaluate_model(weights: str, data_yaml: str, classes: list,
                   conf: float = None, conf_preference: str = "miss",
                   imgsz: int = None, iou: float = 0.7, device: str = "cuda:0") -> dict:
    """在数据集评估集（优先独立 test，其次 val）上评估模型。

    Args:
        conf: 置信度工作点；None 时 best-effort 取 F1 最大的 conf（近似相等按 conf_preference
              偏召回/偏精确），取不到则退回低 conf 高召回（=默认「在意漏检」）口径。
        conf_preference: "miss"(默认,偏召回/防漏检) | "false"(偏精确/防误检)
        imgsz / iou / device: 评测超参（imgsz 默认 640；device 默认 cuda:0）

    Returns（在原有基础上派生）:
        mAP50/mAP50_95/precision/recall/speed_ms/class_ap/eval_split/error
        + f1 / miss_rate(=1-recall) / false_rate(=1-precision) / gpu_mem_mb（峰值显存）
        + speed_breakdown / conf_used / conf_preference
    """
    from ultralytics import YOLO

    split = _pick_eval_split(data_yaml)

    def _measure(use_conf):
        model = YOLO(_resolve_weights(weights))
        try:
            if hasattr(model.model, "set_classes") and classes:
                model.set_classes(classes)
        except Exception:
            pass
        gpu0 = None
        try:
            import torch
            if "cuda" in (device or "") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        res = model.val(data=str(data_yaml), split=split, conf=use_conf,
                        iou=iou, imgsz=imgsz or 640, device=device or "cuda:0",
                        verbose=False, plots=False)
        try:
            import torch
            if "cuda" in (device or "") and torch.cuda.is_available():
                gpu0 = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        except Exception:
            pass
        return res, gpu0

    def _assemble(res, gpu0, use_conf):
        m = res.box

        def _num(v):
            try:
                return None if v is None else round(float(v), 4)
            except (TypeError, ValueError):
                return None

        # 分割模型才有 res.mask（轮廓质量指标）；检测模型为 None。
        mk = getattr(res, "mask", None)
        mask_map50 = _num(getattr(mk, "map50", None)) if mk is not None else None
        mask_map50_95 = _num(getattr(mk, "map", None)) if mk is not None else None

        speed = getattr(res, "speed", None) or {}
        def _spd(k):
            try:
                return round(float(speed.get(k)), 2) if speed.get(k) is not None else None
            except Exception:
                return None
        speed_ms = _spd("inference")
        speed_breakdown = {k: _spd(k) for k in ("preprocess", "inference", "postprocess", "nms") if _spd(k) is not None}

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

        p = None if m.mp is None else round(float(m.mp), 4)
        r = None if m.mr is None else round(float(m.mr), 4)
        f1 = round(2 * p * r / (p + r), 4) if (p is not None and r is not None and (p + r) > 0) else None
        miss_rate = round(1 - r, 4) if r is not None else None
        false_rate = round(1 - p, 4) if p is not None else None

        return {
            "mAP50": None if m.map50 is None else round(float(m.map50), 4),
            "mAP50_95": None if m.map is None else round(float(m.map), 4),
            "precision": p, "recall": r, "f1": f1,
            "miss_rate": miss_rate, "false_rate": false_rate,
            "mask_mAP50": mask_map50, "mask_mAP50_95": mask_map50_95,
            "speed_ms": speed_ms, "speed_breakdown": speed_breakdown,
            "gpu_mem_mb": gpu0,
            "class_ap": class_ap, "eval_split": split,
            "conf_used": use_conf, "conf_preference": conf_preference,
            "error": None,
        }

    try:
        use_conf = conf if conf is not None else 0.001
        res, gpu0 = _measure(use_conf)
        # best-effort 自动工作点：F1 最大 conf（近似相等按偏好偏召回/精确）
        if conf is None:
            auto_conf = _pick_f1_conf(res, conf_preference)
            if auto_conf is not None and abs(auto_conf - use_conf) > 1e-3:
                res, gpu0 = _measure(auto_conf)
                use_conf = auto_conf
        return _assemble(res, gpu0, use_conf)
    except Exception as e:
        return {
            "mAP50": None, "mAP50_95": None, "precision": None, "recall": None,
            "f1": None, "miss_rate": None, "false_rate": None,
            "mask_mAP50": None, "mask_mAP50_95": None,
            "speed_ms": None, "speed_breakdown": None, "gpu_mem_mb": None,
            "class_ap": {}, "eval_split": split,
            "conf_used": conf if conf is not None else 0.001,
            "conf_preference": conf_preference, "error": str(e),
        }


def _pick_f1_conf(res, preference: str = "miss"):
    """best-effort：从 val 结果取 F1 最大的置信度工作点。
    取不到（曲线不可用时）返回 None —— 调用方退回默认低 conf 高召回口径。"""
    try:
        curves = getattr(res, "curves", None)
        if not curves:
            return None
        best_conf, best_f1 = None, -1.0
        for c in curves:
            title = str(getattr(c, "title", getattr(c, "name", "")) or "").lower()
            xlabel = str(getattr(c, "xlabel", getattr(c, "kname", "")) or "").lower()
            if "f1" not in title or "conf" not in xlabel:
                continue
            px = getattr(c, "px", None) or getattr(c, "x", None)
            py = getattr(c, "py", None) or getattr(c, "y", None)
            if px is None or py is None:
                continue
            for x, y in zip(px, py):
                x, y = float(x), float(y)
                if y > best_f1:
                    best_f1, best_conf = y, x
                elif abs(y - best_f1) <= 1e-3 and best_conf is not None:
                    if preference == "miss" and x < best_conf:
                        best_conf = x
                    elif preference == "false" and x > best_conf:
                        best_conf = x
        return best_conf
    except Exception:
        return None


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
        parts = []
        try:
            for n in regressed:
                row = class_ap[n]
                oa = row.get("old_ap")
                na = row.get("new_ap")
                oa_s = f"{oa * 100:.1f}%" if oa is not None else "无旧指标"
                na_s = f"{na * 100:.1f}%" if na is not None else "无检出"
                parts.append(f"{n}（{oa_s} → {na_s}）")
            detail = "; ".join(parts)
        except Exception:
            # 报告拼接兜底：绝不让格式化异常中断评估，确保指标数据仍被返回
            detail = "; ".join(regressed)
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

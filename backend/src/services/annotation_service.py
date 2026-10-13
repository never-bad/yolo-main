import json
import yaml
import asyncio
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
from PIL import Image
from src.core.settings import settings
from src.utils.class_labels import model_class_names, merge_classes_with_model
from src.utils import zip_guard
from src.utils.json_io import load_json, save_json_atomic

logger = logging.getLogger(__name__)


def live_classes(snapshot_classes: list, model_id: str) -> list:
    """算法库标签字典实时类别覆盖数据集/任务快照（#6 标签不同步）。

    库侧新增/改名标签后，无需重新准备数据集或重建标注任务：读取时按 index 合并。
    不落盘、不迁移存量数据；无归属模型或字典为空时原样返回快照（行为与改动前一致）。
    """
    try:
        return merge_classes_with_model(snapshot_classes or [], model_class_names(model_id or ""))
    except Exception as e:
        logger.warning(f"live_classes({model_id}) 读取库标签字典失败，回退快照: {e}")
        return [str(c) for c in (snapshot_classes or [])]


def save_json_atomic(path: Path, data: dict):
    """同步保存 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_yaml(path: Path):
    """同步加载 YAML 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_text(path: Path, content: str):
    """同步写入文本文件"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _save_yaml(path: Path, data: dict):
    """同步保存 YAML 文件"""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


# ----------------------------------------------------------------------
# 分割轮廓（polygon）工具：像素坐标 [[x, y], ...]
# 有 polygon 的框 = 分割标注；没有的仍是普通检测框（两类数据可共存于同一数据集）
# ----------------------------------------------------------------------
# 顶点数上限，防止手工/AI 结果把 JSON 撑爆。必须 >= SAM 侧的 sam_polygon_max_points，
# 否则 AI 描出来的轮廓会在落盘/导出时被这里悄悄砍掉。
MAX_POLYGON_POINTS = 3000
BBOX_KEYS = ("x1", "y1", "x2", "y2")  # 矩形四坐标：检测目标才落盘，分割目标由轮廓现算


def clean_polygon(poly) -> list:
    """校验并规范化多边形顶点：只接受 [x, y] 数字对，至少 3 个点，坐标非负、保留两位小数。

    非法输入一律返回 []（当作没有轮廓处理），保证落盘数据永远可用。
    """
    if not poly or not isinstance(poly, (list, tuple)):
        return []
    pts = []
    for p in poly:
        if isinstance(p, dict):
            xy = (p.get("x"), p.get("y"))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            xy = (p[0], p[1])
        else:
            return []
        try:
            x, y = float(xy[0]), float(xy[1])
        except (TypeError, ValueError):
            return []
        if x != x or y != y:  # NaN
            return []
        pts.append([round(max(x, 0.0), 2), round(max(y, 0.0), 2)])
    if len(pts) < 3:
        return []
    # 首尾重复点去掉（前端可能闭合回起点）
    if len(pts) > 3 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) > MAX_POLYGON_POINTS:
        # 超限时按索引等距抽稀，绝不能从中间截断：截断会把后半圈整段丢掉，
        # 再用一根横跨画面的弦把首尾连上，多边形变成"缺一大块"的畸形形状（实测
        # 900 点的圆截到 500 点后只剩约 200° 的弧）。抽稀只是变稀疏，形状仍闭合完整。
        step = len(pts) / float(MAX_POLYGON_POINTS)
        pts = [pts[int(i * step)] for i in range(MAX_POLYGON_POINTS)]
    return pts if len(pts) >= 3 else []


def polygon_to_bbox(poly: list) -> tuple:
    """由多边形顶点求外接框 (x1, y1, x2, y2)，供矩形字段与轮廓保持一致。"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def ensure_box_bbox(box: dict) -> dict:
    """读取兜底：分割框没存矩形时，用轮廓外接框现算补回。

    分割目标在磁盘上只有轮廓点（方框是给 SAM 圈范围的中间产物，用完就扔），
    但画布/千问审核/导出前 5 列等几十处下游都按矩形取坐标，
    所以统一在读取口补算，保证下游零改动且"框与轮廓永远一致"。
    """
    if box.get("x1") is None and len(box.get("polygon") or []) >= 3:
        x1, y1, x2, y2 = polygon_to_bbox(box["polygon"])
        return {**box, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
    return box


def boxes_with_bbox(boxes: list) -> list:
    """批量读取兜底（见 ensure_box_bbox）"""
    return [ensure_box_bbox(b) for b in (boxes or [])]


def ann_with_bbox(data: dict) -> dict:
    """整份每图 JSON 的读取兜底"""
    if isinstance(data, dict) and data.get("boxes"):
        return {**data, "boxes": boxes_with_bbox(data["boxes"])}
    return data


def strip_ann_bbox(data: dict) -> dict:
    """落盘前统一剥离：带轮廓的框只保留轮廓点，不再重复存矩形。

    放在唯一写盘口，保证任何路径（保存/清理重合/外部导入）写入的分割标注
    都是"只存点"的干净形态；需要矩形时由 ensure_box_bbox 现算。
    """
    if isinstance(data, dict) and data.get("boxes"):
        return {**data, "boxes": [
            {k: v for k, v in b.items() if k not in BBOX_KEYS}
            if b.get("polygon") else b
            for b in data["boxes"]
        ]}
    return data


def _looks_like_bbox_prefix(nums: list) -> bool:
    """判断一串归一化数字是否以 "cx cy w h" 开头（平台历史导出写法）。

    判据：把前 4 个当 (cx,cy,w,h)，若其余坐标全部落在该框内 → 认定带框前缀。
    官方分割行的前 4 个数字只是第 1、2 个顶点，几乎不可能把整圈轮廓包进去，
    因此这个判别足够稳。
    """
    if len(nums) < 10 or (len(nums) - 4) % 2 != 0:
        return False
    cx, cy, bw, bh = nums[:4]
    if bw <= 0 or bh <= 0 or cx <= 0 or cy <= 0:
        return False
    x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    tol = 0.02
    rest = nums[4:]
    for i in range(0, len(rest), 2):
        if not (x1 - tol <= rest[i] <= x2 + tol and y1 - tol <= rest[i + 1] <= y2 + tol):
            return False
    return True


def decode_yolo_line(parts: list, img_w: float, img_h: float):
    """解析一行 YOLO 标签 → {class_id, x1, y1, x2, y2[, polygon]}；非法行返回 None。

    三种写法都认（读入口必须向后兼容，否则历史导出的 zip 全废）：
      检测    class cx cy w h
      分割    class x1 y1 x2 y2 ...       ← ultralytics 官方口径：类别之后全是顶点
      旧导出  class cx cy w h x1 y1 ...    ← 顶点前多 4 列框参数，仅供读历史包
    """
    if len(parts) < 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        nums = [float(p) for p in parts[1:]]
    except (TypeError, ValueError):
        return None

    def _to_pixels(seq):
        return clean_polygon([[round(seq[i] * img_w, 2), round(seq[i + 1] * img_h, 2)]
                              for i in range(0, len(seq) - 1, 2)])

    head = 4 if _looks_like_bbox_prefix(nums) else 0
    if head:
        cx, cy, bw, bh = nums[:4]
        box = {
            "class_id": cls_id,
            "x1": round((cx - bw / 2) * img_w, 2),
            "y1": round((cy - bh / 2) * img_h, 2),
            "x2": round((cx + bw / 2) * img_w, 2),
            "y2": round((cy + bh / 2) * img_h, 2),
        }
        poly = _to_pixels(nums[4:])
        if poly:
            box["polygon"] = poly
        return box
    if len(nums) == 4:  # 纯检测行
        cx, cy, bw, bh = nums
        return {
            "class_id": cls_id,
            "x1": round((cx - bw / 2) * img_w, 2),
            "y1": round((cy - bh / 2) * img_h, 2),
            "x2": round((cx + bw / 2) * img_w, 2),
            "y2": round((cy + bh / 2) * img_h, 2),
        }
    if len(nums) >= 6 and len(nums) % 2 == 0:  # 官方分割行：矩形由顶点外接框现算
        poly = _to_pixels(nums)
        if not poly:
            return None
        x1, y1, x2, y2 = polygon_to_bbox(poly)
        return {"class_id": cls_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "polygon": poly}
    return None


def yolo_label_line(cls_id: int, bbox: tuple, poly: list, img_w: float, img_h: float,
                    rect_as_poly: bool = False) -> str:
    """生成一行 YOLO 标签文本（唯一写入口，所有导出路径必须走这里）。

    ultralytics 的读法（data/utils.py）是"类别之后的数字两两一组当顶点"，
    所以分割行绝不能带 cx cy w h 前缀：宽高会被读成第 2 个顶点，在每个轮廓上
    凭空插一根指向画面左上角的尖刺（曾经把 139 张训练图全污染成这个形状）。
      分割行： class x1 y1 x2 y2 ...
      检测行： class cx cy w h
    rect_as_poly：整份标签里只要有一条分割行就传 True，此时没有轮廓的框退化成
    4 个角点的分割行 —— 否则混合文件里检测行的 4 列会被当成顶点读成垃圾形状。
    """
    pts = poly
    if not pts and rect_as_poly:
        bx1, by1, bx2, by2 = bbox
        pts = [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]
    if pts:
        flat = []
        for px, py in pts:
            flat.append(f"{min(max(px / img_w, 0.0), 1.0):.6f}")
            flat.append(f"{min(max(py / img_h, 0.0), 1.0):.6f}")
        return f"{cls_id} " + " ".join(flat)
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = max(x2 - x1, 0.0) / img_w
    h = max(y2 - y1, 0.0) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


class AnnotationService:
    # 样本类型（标注后图片分级；沿用平台样本池语义）
    SAMPLE_NORMAL = "normal"          # 普通样本：有框，人工/AI 确认质量正常
    SAMPLE_HARD = "hard"              # 困难样本：AI 低置信度 / AI 检空漏标 / 人工标记，需重点审核
    SAMPLE_BACKGROUND = "background"  # 空白样本（负样本）：无任何目标框
    # AI 置信度低于该值（且非人工确认）自动判为困难样本
    HARD_CONF_THRESHOLD = 0.4

    def __init__(self):
        self.annotations_dir = settings.ANNOTATIONS_DIR
        self.datasets_dir = settings.DATASETS_DIR
    
    # ------------------------------------------------------------------
    # 统一 JSON 标注格式（数据链路改造：一图一 JSON 为唯一权威标注）
    # datasets/{dataset_id}/annotations/{image_id}.json
    # {
    #   "image_id": "img_001", "image_path": "images/img_001.jpg",
    #   "width": 1920, "height": 1080,
    #   "model_id": "m_person", "dataset_id": "ds_x", "version": "v1",
    #   "sample_type": "normal",        # normal 普通 | hard 困难 | background 空白负样本
    #   "sample_reasons": ["low_conf"], # 困难判据（可空）：low_conf/AI检空 ai_miss/manual 人工标记
    #   "ai_annotated": false,
    #   "created_at": "...", "updated_at": "...",
    #   "boxes": [
    #     {
    #       "class_id": 0, "english_code": "person",
    #       "chinese_name": "人", "chinese_desc": "行人/人体",   # 标签四字段
    #       "x1": 120, "y1": 80, "x2": 240, "y2": 320,          # 像素坐标
    #       "polygon": [[130, 80], ...],                         # 可选：分割轮廓（像素坐标，≥3 点）
    #       "confidence": 0.86, "source": "ai_qwen"             # 可选：AI 置信度/来源
    #     }
    #   ]
    # }
    # 注：数据集一律为 datasets/{dataset_id}（无 version 子目录）}
    # ------------------------------------------------------------------
    def _image_ann_path(self, dataset_id: str, version: str, image_id: str) -> Path:
        return self.datasets_dir / dataset_id / "annotations" / f"{image_id}.json"

    def _load_image_ann(self, dataset_id: str, version: str, image_id: str) -> dict:
        p = self._image_ann_path(dataset_id, version, image_id)
        try:
            return ann_with_bbox(load_json(p))
        except Exception:
            return {}

    def _save_image_ann(self, dataset_id: str, version: str, image_id: str, data: dict):
        p = self._image_ann_path(dataset_id, version, image_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_json_atomic(p, strip_ann_bbox(data))

    def _delete_image_ann(self, dataset_id: str, version: str, image_id: str):
        p = self._image_ann_path(dataset_id, version, image_id)
        try:
            p.unlink()
        except OSError:
            pass

    def _load_all_image_anns(self, dataset_id: str, version: str) -> dict:
        """读取数据集下所有每图 JSON 标注（缺失/损坏按空处理）"""
        ann_dir = self.datasets_dir / dataset_id / "annotations"
        out = {}
        if not ann_dir.exists():
            return out
        for p in ann_dir.glob("*.json"):
            try:
                data = ann_with_bbox(load_json(p))
                out[data.get("image_id", p.stem)] = data
            except Exception:
                continue
        return out

    def _task_to_ds(self, task_meta: dict) -> tuple:
        return task_meta.get("dataset_id", ""), task_meta.get("version", "v1")

    def _class_meta_map(self, model_id: str) -> dict:
        """模型标签字典 → {english_code: {english_code, chinese_name, chinese_desc}}。

        用于保存标注时为每框补全标签四字段（JSON 自包含，千问提示词/导出无需再查字典）。
        无模型/无字典时返回空映射（框内四字段退化为英文码本身）。
        """
        if not model_id:
            return {}
        try:
            dic = load_json(settings.REGISTRY_DIR / model_id / "labels_dict.json")
        except Exception:
            return {}
        labels = dic.get("labels") or []
        out = {}
        for l in labels:
            code = (l.get("english_code") or "").strip()
            if code:
                out[code] = {
                    "english_code": code,
                    "chinese_name": l.get("chinese_name") or "",
                    "chinese_desc": l.get("chinese_desc") or "",
                }
        return out

    def _match_class_meta(self, class_meta: dict, eng: str) -> dict:
        """标签字典匹配：先 english_code 精确，再 chinese_name 模糊，未命中返回空。"""
        if not class_meta:
            return {}
        if eng in class_meta:
            return class_meta[eng]
        for c in class_meta.values():
            if (c.get("chinese_name") or "") == eng:
                return c
        return {}

    def _normalize_boxes(self, boxes: list, classes: list, class_meta: dict) -> list:
        """规范化框字段：class_id 落盘为数字索引（与 YOLO 类别号一致），并补全标签四字段。

        输入 class_id 兼容数字索引（classes 列表下标）或字符串（英文码/中文名/历史英文名）：
        - 数字索引 → 原样落盘
        - 字符串 → 优先映射回 classes 下标（英文名直接命中 / 中文名经标签字典解析）
        四字段冗余存于每框（english_code/chinese_name/chinese_desc），保证 JSON 自包含、
        标签字典重排不丢语义。YOLO 导出直接用 class_id，无需再映射。
        """
        out = []
        for raw in boxes:
            b = raw.model_dump() if hasattr(raw, "model_dump") else (
                    raw.dict() if hasattr(raw, "dict") else raw)
            cid_raw = b.get("class_id", 0)
            if isinstance(cid_raw, str) and not cid_raw.isdigit():
                # 字符串：英文名在 classes 中 → 下标；否则经标签字典中文名解析
                name = str(cid_raw)
                cid = None
                if classes:
                    for i, c in enumerate(classes):
                        if str(c) == name or str(c).lower() == name.lower():
                            cid = i
                            break
                if cid is None:
                    meta0 = self._match_class_meta(class_meta, name)
                    eng = meta0.get("english_code")
                    if eng and classes:
                        for i, c in enumerate(classes):
                            if str(c) == eng or str(c).lower() == eng.lower():
                                cid = i
                                break
                if cid is None:
                    cid = 0
            else:
                cid = int(cid_raw) if cid_raw is not None else 0
            # 四字段冗余（可读性/JSON 自包含，字典重排后仍可知本框语义）
            eng = classes[cid] if classes and 0 <= cid < len(classes) else str(cid)
            meta = self._match_class_meta(class_meta, eng)
            eng = meta.get("english_code") or eng
            nb = {
                "class_id": cid,
                "english_code": eng,
                "chinese_name": meta.get("chinese_name") or eng,
                "chinese_desc": meta.get("chinese_desc") or "",
            }
            poly = clean_polygon(b.get("polygon"))
            if poly:
                # 分割目标：方框只是 Grounding→SAM 圈定范围的中间产物，用完就扔，
                # 数据里只保留轮廓点（矩形由读取口 ensure_box_bbox 现算，永远和轮廓一致）
                nb["polygon"] = poly
            else:
                # 检测目标：只有矩形可用，必须落 4 个坐标
                try:
                    nb.update({"x1": float(b["x1"]), "y1": float(b["y1"]),
                               "x2": float(b["x2"]), "y2": float(b["y2"])})
                except (KeyError, TypeError):
                    continue  # 既无轮廓也无坐标 → 无法定位，丢弃脏框
            confidence = b.get("confidence")
            if confidence is None:
                confidence = b.get("score")
            if confidence is not None:
                nb["confidence"] = float(confidence)
            source = b.get("source")
            if source:
                nb["source"] = source
            # 审核结论（L2 千问/人工）随框落盘，供"问题图片检索"等使用
            audit = b.get("audit")
            if audit:
                nb["audit"] = audit
            out.append(nb)
        return out

    def _boxes_for_view(self, boxes: list, classes: list) -> list:
        """响应层兜底：新数据 class_id 已是数字索引，直接透传；
        对历史英文名 JSON 兜底映射回 classes 下标，找不到保留原值（不崩）。
        """
        idx_map = {str(c).lower(): i for i, c in enumerate(classes or [])}
        out = []
        for b in boxes:
            nb = dict(ensure_box_bbox(b))
            cid = b.get("class_id", 0)
            if not (isinstance(cid, int) or (isinstance(cid, str) and cid.isdigit())):
                nb["class_id"] = idx_map.get(str(cid).lower(), cid)
            out.append(nb)
        return out

    def _resolve_sample_type(self, boxes: list, ai_annotated: bool,
                             manual_type: str = None, manual_reason: str = None) -> tuple:
        """决定图片样本类型与判据（人工显式传值优先，否则 AI 自动判定）。"""
        if manual_type:
            reasons = []
            if manual_reason:
                reasons.append(manual_reason)
            return manual_type, reasons

        if not boxes:
            # 无框：空白负样本（若为 AI 检空后人工确认，由人工传 background 覆盖）
            return self.SAMPLE_BACKGROUND, []

        # AI 标注存在低置信度框 → 困难样本（供人工重点审核）
        if ai_annotated and any(
            (b.get("confidence") if b.get("confidence") is not None else b.get("score") or 1.0)
            < self.HARD_CONF_THRESHOLD for b in boxes
        ):
            return self.SAMPLE_HARD, ["low_conf"]
        return self.SAMPLE_NORMAL, []
    
    async def create_task(self, dataset_id: str, version: str, classes: list):
        """创建标注任务"""
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task_dir = self.annotations_dir / task_id
        
        # 异步创建目录
        await asyncio.to_thread(lambda: task_dir.mkdir(parents=True, exist_ok=True))
        
        # 获取数据集图片（数据集一律为 datasets/{dataset_id}，无 version 子目录）
        dataset_dir = self.datasets_dir / dataset_id

        # 检查数据集目录是否存在（标准 YOLO 结构：images/{train,val,test} + 根 data.yaml）
        if not await asyncio.to_thread(
            lambda: (dataset_dir / "data.yaml").exists()
            or (dataset_dir / "images").exists()
        ):
            raise ValueError(f"数据集 {dataset_id} 不存在或未准备，请先上传并执行 prepare")
        
        # 在线程中执行所有同步操作
        def _create_task_sync():
            images_dir = self._find_images_dir(dataset_dir)
            labels_dir = self._find_labels_dir(dataset_dir)

            if not images_dir:
                raise ValueError(f"数据集 {dataset_id} 中未找到图片目录，请确保已执行 prepare 操作")

            # 数据集归属模型 + 类别（记录 model_id；类别权威来源 = 数据集 meta.json 的 classes，prepare 时已写入）
            ds_meta_path = self.datasets_dir / dataset_id / "meta.json"
            model_id, auto_classes = "", []
            try:
                if ds_meta_path.exists():
                    ds_meta = load_json(ds_meta_path)
                    model_id = ds_meta.get("model_id", "")
                    auto_classes = ds_meta.get("classes") or []
                else:
                    # 标准 YOLO 结构数据集：类别从根 data.yaml 的 names 读取
                    data_yaml = self.datasets_dir / dataset_id / "data.yaml"
                    if data_yaml.exists():
                        cfg = yaml.safe_load(
                            data_yaml.read_text(encoding="utf-8")
                        ) or {}
                        names = cfg.get("names") or []
                        if isinstance(names, dict):
                            names = [
                                names[k]
                                for k in sorted(names, key=lambda x: int(x))
                            ]
                        auto_classes = [str(n) for n in names]
            except Exception:
                pass

            # 类别跟随算法库标签字典（#6）：meta.classes 是 prepare 当时的快照，
            # 库侧新增/改名标签后按 index 合并，建任务即拿到最新类别清单
            auto_classes = live_classes(auto_classes, model_id)

            # 类别合并：数据集 meta.classes 在前（真实类别名，class_id 顺序与已有标注对齐），
            # 用户输入中的新增类别追加在后；两者合并去重。避免标注任务无类别导致 AI 预标注不可用
            manual = classes or []
            task_classes = auto_classes + [c for c in manual if c not in auto_classes]
            
            # 收集所有图片
            # 若 images 目录下已有 train/val/test 划分，只遍历划分目录，避免平铺图与划分子目录重复收集
            split_dirs = ["train", "val", "test"]
            has_split = any((images_dir / s).exists() for s in split_dirs)
            if has_split:
                image_files = []
                for s in split_dirs:
                    sd = images_dir / s
                    if sd.exists():
                        image_files += (
                            list(sd.rglob("*.jpg"))
                            + list(sd.rglob("*.png"))
                            + list(sd.rglob("*.jpeg"))
                        )
            else:
                image_files = list(images_dir.rglob("*.jpg")) + list(images_dir.rglob("*.png")) + list(images_dir.rglob("*.jpeg"))
            
            items = []
            loaded_count = 0
            
            # 按 image_id(stem) 去重，同一张图重复收集时只保留一个（图片列表按 image_id 展示，需保持一致）
            seen_stems = set()
            for img_path in image_files:
                if img_path.stem in seen_stems:
                    continue
                seen_stems.add(img_path.stem)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                    
                    image_id = img_path.stem
                    
                    # 检查是否有对应的标签文件（导入后的 YOLO txt）
                    has_annotation = False
                    if labels_dir:
                        # 尝试找到对应的标签文件
                        label_path = self._find_label_file(img_path, images_dir, labels_dir)
                        if label_path and label_path.exists():
                            # 每图 JSON 才是标注的权威来源。这里绝不能用 labels/*.txt
                            # 覆盖已存在的 JSON：否则在平台上画好、微调过的标注（尤其是
                            # 分割轮廓）会在"重新创建标注任务"时被 txt 里的旧内容整体冲掉，
                            # 属于静默丢数据，且丢完再也找不回来。
                            existing = self._load_image_ann(dataset_id, version, image_id)
                            if existing:
                                has_annotation = bool(existing.get("boxes"))
                            else:
                                # 首次建任务（如导入外部分割包）才用 txt 播种
                                boxes = self._load_yolo_labels(label_path, width, height)
                                if boxes:
                                    ann_data = {
                                        "image_id": image_id,
                                        "image_path": self._calculate_image_path(img_path),
                                        "width": width,
                                        "height": height,
                                        "model_id": model_id,
                                        "dataset_id": dataset_id,
                                        "version": version,
                                        "boxes": boxes,
                                        "updated_at": datetime.now().isoformat(),
                                        "ai_annotated": False,
                                    }
                                    self._save_image_ann(dataset_id, version, image_id, ann_data)
                                    has_annotation = True
                                    loaded_count += 1
                    
                    # 计算图片路径，相对于 DATA_DIR（静态文件服务的根目录）
                    image_path = self._calculate_image_path(img_path)
                    
                    items.append({
                        "image_id": image_id,
                        "image_path": image_path,
                        "width": width,
                        "height": height,
                        "annotated": has_annotation
                    })
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
                    continue
            
            # 保存任务元数据
            task_meta = {
                "task_id": task_id,
                "dataset_id": dataset_id,
                "version": version,
                "classes": task_classes,
                "model_id": model_id,
                "created_at": datetime.now().isoformat(),
                "items": items,
                "imported_annotations": loaded_count  # 记录导入的标注数量
            }
            
            save_json_atomic(task_dir / "task.json", task_meta)
            
            return {
                "task_id": task_id,
                "total_images": len(items),
                "imported_annotations": loaded_count,
                "classes": task_classes
            }
        
        result = await asyncio.to_thread(_create_task_sync)
        # 影子库收口：仅新建任务时整表写 items（此后 has_label 由标注/导入路径逐行回填，
        # task.json 快照不带该字段，再用快照覆盖会把 PG 已标注标记冲掉）
        if result.get("task_id"):
            from src.db import shadow
            await shadow.sync_task(result["task_id"], with_items=True)
        return result
    
    def find_task_by_dataset(self, dataset_id: str, version: str = "v1"):
        """按数据集查找已存在的标注任务（自动建任务去重用 / 标注页直达用）。

        返回第一个匹配的任务摘要 dict（含 task_id / dataset_id / version / total_images /
        imported_annotations / classes / created_at），找不到返回 None。
        复用 create_task 的目录扫描需遍历 annotations 目录，这里直接读 task.json。
        """
        annotations_root = self.annotations_dir
        if not annotations_root.exists():
            return None

        for task_dir in sorted(annotations_root.iterdir(), reverse=True):
            task_file = task_dir / "task.json"
            if not task_file.is_file():
                continue
            try:
                meta = load_json(task_file)
            except Exception:
                continue
            if meta.get("dataset_id") == dataset_id and meta.get("version", "v1") == version:
                return {
                    "task_id": meta.get("task_id"),
                    "dataset_id": meta.get("dataset_id"),
                    "version": meta.get("version", "v1"),
                    "total_images": len(meta.get("items") or []),
                    "imported_annotations": meta.get("imported_annotations", 0),
                    "classes": meta.get("classes", []),
                    "created_at": meta.get("created_at"),
                }
        return None

    async def delete_image(self, task_id: str, image_id: str):
        """删除标注任务中的一张图片。

        先经任务定位数据集，再调用 DatasetService.delete_image 删除图片文件、
        每图 JSON 标注与同名标签文件并更新数据集 meta 计数；
        最后从任务 item 列表中移除该图，保持列表与磁盘一致。
        """
        def _sync():
            task_dir = self.annotations_dir / task_id
            task_file = task_dir / "task.json"
            if not task_file.exists():
                raise ValueError(f"标注任务 {task_id} 不存在")
            return load_json(task_file)

        meta = await asyncio.to_thread(_sync)
        ds_id = meta.get("dataset_id") or ""
        if not ds_id:
            raise ValueError("标注任务未绑定数据集，无法删除图片")

        from src.services.dataset_service import DatasetService
        result = await DatasetService().delete_image(ds_id, image_id)
        removed_id = result.get("image_id", image_id)

        # 从任务 items 中移除，保持列表与磁盘一致
        items = [it for it in (meta.get("items") or []) if it.get("image_id") != removed_id]
        meta["items"] = items

        def _save():
            task_dir = self.annotations_dir / task_id
            save_json_atomic(task_dir / "task.json", meta)

        await asyncio.to_thread(_save)
        # 影子库收口：任务图片删除后同步移除 task_items 行，避免列表残留
        from src.db import shadow
        await shadow.remove_task_item(task_id, removed_id)
        result["total_images"] = len(items)
        return result
    
    def _calculate_image_path(self, img_path: Path) -> str:
        """计算图片相对于 DATA_DIR 的路径"""
        try:
            # 转换为绝对路径，确保可以正确计算相对路径
            img_path_abs = img_path.resolve()
            data_dir_abs = settings.DATA_DIR.resolve()
            
            # 手动计算相对于 DATA_DIR 的路径（兼容性更好）
            img_path_str = str(img_path_abs).replace('\\', '/')
            data_dir_str = str(data_dir_abs).replace('\\', '/')
            
            if img_path_str.startswith(data_dir_str):
                # 去掉 DATA_DIR 前缀，保留相对路径
                return img_path_str[len(data_dir_str):].lstrip('/')
            else:
                # 如果不在 DATA_DIR 下，尝试使用 Path.relative_to
                try:
                    rel_path = img_path_abs.relative_to(data_dir_abs)
                    return str(rel_path).replace('\\', '/')
                except ValueError:
                    # 如果计算失败，尝试相对于 BASE_DIR 然后去掉 data/ 前缀
                    base_rel = img_path_abs.relative_to(settings.BASE_DIR.resolve())
                    image_path = str(base_rel).replace('\\', '/')
                    # 去掉 'data/' 前缀
                    if image_path.startswith('data/'):
                        return image_path[5:]
                    return image_path
        except Exception as e:
            # 最后的备用方案：直接使用文件名后的路径部分
            print(f"Warning: Could not calculate relative path for {img_path}: {e}")
            # 尝试从完整路径中提取相对于 datasets 的部分
            img_path_str = str(img_path).replace('\\', '/')
            if '/datasets/' in img_path_str:
                idx = img_path_str.index('/datasets/')
                return img_path_str[idx + 1:]  # 去掉开头的 /
            else:
                # 如果都失败了，使用原始相对路径（相对于 BASE_DIR）
                image_path = str(img_path.relative_to(settings.BASE_DIR)).replace('\\', '/')
                if image_path.startswith('data/'):
                    return image_path[5:]
                return image_path
    
    def _find_label_file(self, img_path: Path, images_dir: Path, labels_dir: Path) -> Path:
        """根据图片路径找到对应的标签文件"""
        try:
            # 获取图片相对于 images 目录的路径
            relative_path = img_path.relative_to(images_dir)
            # 构建标签文件路径
            label_path = labels_dir / relative_path.with_suffix('.txt')
            return label_path

        except ValueError:
            # 如果无法获取相对路径，尝试直接在 labels 目录下查找
            return labels_dir / (img_path.stem + '.txt')
    
    def _load_yolo_labels(self, label_path: Path, img_width: int, img_height: int) -> list:
        """加载 YOLO 格式标签文件并转换为绝对坐标"""
        boxes = []
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    # decode_yolo_line 统一处理"检测行 / 官方分割行 / 平台旧导出分割行"，
                    # 分割轮廓必须保留，否则外部分割包导进来会被静默截成矩形框
                    box = decode_yolo_line(parts, img_width, img_height)
                    if box:
                        boxes.append(box)
        except Exception as e:
            print(f"Error loading label file {label_path}: {e}")
        
        return boxes
    
    def _find_labels_dir(self, root_dir: Path) -> Path:
        """查找 labels 目录"""
        candidates = list(root_dir.rglob("labels"))
        return candidates[0] if candidates else None
    
    def _find_images_dir(self, root_dir: Path):
        """查找images目录"""
        candidates = list(root_dir.rglob("images"))
        return candidates[0] if candidates else None
    
    async def get_task_items(self, task_id: str):
        """获取标注任务的图片列表"""
        task_dir = self.annotations_dir / task_id
        task_file = task_dir / "task.json"
        
        if not await asyncio.to_thread(lambda: task_file.exists()):
            return None
        
        def _get_task_items_sync():
            task_meta = load_json(task_file)
            dataset_id, version = self._task_to_ds(task_meta)
            
            # 读取标注状态（优先每图 JSON，兼容历史任务集中式 annotations.json）
            # 文件可能被批量预标注并发写入，缺失/损坏时按空处理，避免整个列表加载失败
            annotations = self._load_all_image_anns(dataset_id, version)
            legacy_file = task_dir / "annotations.json"
            try:
                if legacy_file.exists():
                    for img_id, ann in load_json(legacy_file).items():
                        annotations.setdefault(img_id, ann)
            except Exception:
                pass
            
            # 更新标注状态（按 image_id 去重，避免历史任务中平铺图与划分目录重复收集导致的重复条目）
            seen = set()
            unique_items = []
            for item in task_meta["items"]:
                if item["image_id"] in seen:
                    continue
                seen.add(item["image_id"])
                ann = annotations.get(item["image_id"], {})
                item["annotated"] = item["image_id"] in annotations
                item["ai_annotated"] = bool(ann.get("ai_annotated", False))
                item["sample_type"] = ann.get("sample_type", "")
                # P2：has_label=该图标注是否存在有内容的框（approved 且 has_label 联合完成判定）
                item["has_label"] = bool(ann.get("boxes"))
                # 轮廓质量筛选用：是否人工复核过 / 图上是否存在分割轮廓
                item["reviewed"] = bool(ann.get("reviewed", False))
                item["has_polygon"] = any(
                    isinstance(b, dict) and b.get("polygon") for b in (ann.get("boxes") or [])
                )
                unique_items.append(item)
            
            return {
                "items": unique_items,
                # 类别下拉跟随算法库标签字典实时值（#6），不落盘、不改任务快照
                "classes": live_classes(task_meta.get("classes") or [], task_meta.get("model_id") or ""),
                "dataset_id": task_meta.get("dataset_id", ""),
                "model_id": task_meta.get("model_id", ""),
                "imported_annotations": task_meta.get("imported_annotations", 0)
            }
        
        return await asyncio.to_thread(_get_task_items_sync)
    
    async def get_image_annotation(self, task_id: str, image_id: str):
        """获取单张图片的标注"""
        task_dir = self.annotations_dir / task_id
        task_file = task_dir / "task.json"
        
        if not await asyncio.to_thread(lambda: task_file.exists()):
            return None
        
        def _get_image_annotation_sync():
            task_meta = load_json(task_file)
            dataset_id, version = self._task_to_ds(task_meta)
            
            # 查找图片信息
            image_info = None
            for item in task_meta["items"]:
                if item["image_id"] == image_id:
                    image_info = item
                    break
            
            if not image_info:
                return None
            
            # 获取标注（优先每图 JSON，兼容历史任务集中式 annotations.json）
            annotation = self._load_image_ann(dataset_id, version, image_id)
            if not annotation:
                legacy_file = task_dir / "annotations.json"
                try:
                    if legacy_file.exists():
                        annotation = load_json(legacy_file).get(image_id, {})
                except Exception:
                    annotation = {}
            # 类别清单跟随算法库（#6）：下拉框与框下标必须同源，否则前端渲染错位
            view_classes = live_classes(task_meta.get("classes") or [], task_meta.get("model_id") or "")
            boxes = annotation.get("boxes", [])
            # JSON 落盘 class_id 为英文名，响应层映射回任务类别下标（前端渲染用）
            boxes = self._boxes_for_view(boxes, view_classes)
            
            return {
                "image_id": image_id,
                "image_path": image_info["image_path"],
                "width": image_info["width"],
                "height": image_info["height"],
                "boxes": boxes,
                "sample_type": annotation.get("sample_type", ""),
                "sample_reasons": annotation.get("sample_reasons", []),
                # 人工复核标记（前端切图时回读，决定"轮廓质量"归属）
                "reviewed": bool(annotation.get("reviewed", False)),
                "classes": view_classes
            }
        
        return await asyncio.to_thread(_get_image_annotation_sync)
    
    @staticmethod
    def _box_iou(a: dict, b: dict) -> float:
        """两个框的 IoU（框需含 x1/y1/x2/y2，缺字段或为 None 按 0 兜底）。

        分割目标落盘只有轮廓点（矩形不存），调用方应先经 ensure_box_bbox 补算；
        这里再用 `or 0` 兜一层，避免 None 直接 float() 抛 TypeError。
        """
        ax1, ay1 = float(a.get("x1") or 0), float(a.get("y1") or 0)
        ax2, ay2 = float(a.get("x2") or 0), float(a.get("y2") or 0)
        bx1, by1 = float(b.get("x1") or 0), float(b.get("y1") or 0)
        bx2, by2 = float(b.get("x2") or 0), float(b.get("y2") or 0)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0.0

    async def save_annotation(self, task_id: str, image_id: str, boxes: list, ai_annotated: bool = False,
                              sample_type: str = None, sample_reason: str = None, reviewed: bool = False):
        """保存图片标注（写入一图一 JSON 标注文件）。

        boxes 可选携带 confidence/score/source；保存时按任务类别+模型标签字典
        为每框补全标签四字段（english_code/chinese_name/chinese_desc）。
        sample_type 人工显式传值优先（normal/hard/background）；否则自动判定：
        无框→background 空白负样本，AI 低置信度框→hard 困难，其余→normal。
        """
        task_dir = self.annotations_dir / task_id
        task_file = task_dir / "task.json"
        
        if not await asyncio.to_thread(lambda: task_file.exists()):
            return {"ok": False, "error": "Task not found"}
        
        def _save_annotation_sync():
            task_meta = load_json(task_file)
            dataset_id, version = self._task_to_ds(task_meta)
            
            # 读取该图已有标注（保留 AI 标注标记：人工保存不应清掉标记）
            existing = self._load_image_ann(dataset_id, version, image_id)
            prev_ai = existing.get("ai_annotated", False)
            created_at = existing.get("created_at") or datetime.now().isoformat()
            # "人工复核过"粘性标记：画布上真正编辑过标注再保存才传 True（AI 预标注不会传），
            # 一旦置真永不清除——供标注页按"轮廓质量"筛选挑图
            rev = bool(existing.get("reviewed", False)) or bool(reviewed)
            
            # 查找图片信息（首次写入时补充元数据）
            info = next((it for it in task_meta.get("items", []) if it.get("image_id") == image_id), {})
            # 批量预标注进行中允许删除图片：图已被删除（不在任务 items 且图片文件不存在）时
            # 拒绝写回标注，避免残留“孤儿标注”JSON（图片没了标注还在）
            if not info:
                return {"ok": False, "error": "Image removed from task"}
            
            # 规范化框（补全标签四字段 + 置信度/来源），并判定样本类型
            classes = task_meta.get("classes", []) or []
            class_meta = self._class_meta_map(task_meta.get("model_id", ""))
            norm_boxes = self._normalize_boxes(boxes, classes, class_meta)

            # P1-1：矩形坐标越界直接拒绝落盘（脏坐标会污染 YOLO 导出）。
            # 只带轮廓点的分割框跳过（矩形由读取口现算）；图尺寸未知（0）时跳过避免误杀。
            vw = int(info.get("width") or existing.get("width") or 0)
            vh = int(info.get("height") or existing.get("height") or 0)
            if vw > 0 and vh > 0:
                for nb in norm_boxes:
                    x1, y1, x2, y2 = nb.get("x1"), nb.get("y1"), nb.get("x2"), nb.get("y2")
                    if None in (x1, y1, x2, y2):
                        continue
                    if not (0 <= x1 < x2 <= vw and 0 <= y1 < y2 <= vh):
                        raise ValueError(
                            f"标注框越界（图片 {vw}×{vh}）: x1={x1}, y1={y1}, x2={x2}, y2={y2}"
                        )

            ai = ai_annotated or prev_ai
            s_type, s_reasons = self._resolve_sample_type(norm_boxes, ai, sample_type, sample_reason)

            # 人工补框 → 自动困难：AI 已标过的图上，人工新增了与既有框不重叠的新框，
            # 视为人工发现 AI 漏检，强制标为困难查证（用户显式指定类型时以用户为准）
            existing_boxes = existing.get("boxes", []) if isinstance(existing, dict) else []
            if ai and sample_type is None and s_type != self.SAMPLE_BACKGROUND:
                # 比对必须用规范化后的框：原始 boxes 是接口模型对象（没有 .get），
                # 且分割框只带轮廓点不带矩形，直接取坐标会整张图保存失败。
                cur = [ensure_box_bbox(b) for b in norm_boxes]
                old = [ensure_box_bbox(pb) for pb in existing_boxes]
                manual_added = [
                    b for b in cur
                    if str(b.get("source") or "").lower().startswith("manual")
                    and not any(self._box_iou(b, pb) >= 0.7 for pb in old)
                ]
                if manual_added and "manual_add" not in (s_reasons or []):
                    s_type = self.SAMPLE_HARD
                    s_reasons = list(s_reasons or []) + ["manual_add"]
            
            # 写入每图 JSON（含标签所需元数据：image 尺寸/归属模型/数据集版本/样本类型）
            ann_data = {
                **existing,
                "image_id": image_id,
                "image_path": info.get("image_path", existing.get("image_path", "")),
                "width": info.get("width", existing.get("width", 0)),
                "height": info.get("height", existing.get("height", 0)),
                "model_id": task_meta.get("model_id", existing.get("model_id", "")),
                "dataset_id": dataset_id,
                "version": version,
                "sample_type": s_type,
                "sample_reasons": s_reasons,
                "boxes": norm_boxes,
                "created_at": created_at,
                "updated_at": datetime.now().isoformat(),
                "ai_annotated": ai,
                "reviewed": rev,
            }
            
            self._save_image_ann(dataset_id, version, image_id, ann_data)
            return {"ok": True, "sample_type": s_type, "sample_reasons": s_reasons,
                    "has_label": bool(norm_boxes)}
        
        result = await asyncio.to_thread(_save_annotation_sync)
        # P2：标注完成/更新时回填 PG 影子库 has_label（容错：PG 不可用不影响 JSON 主存储）
        if result.get("ok") and settings.PG_ENABLED:
            try:
                from src.db import meta_store
                await meta_store.update_task_item_has_label(task_id, image_id, bool(result.get("has_label")))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"backfill has_label {task_id}/{image_id} failed: {e}")
        return result

    async def clear_ai_annotations(self, task_id: str):
        """清除该任务所有 AI 预标注（批量误标过多时一键清理，便于重新标注）。

        删除所有被 AI 预标注过的图片的标注（含其后手工修改，不可恢复）。
        返回被清理的图片数 removed。
        """
        task_dir = self.annotations_dir / task_id
        task_file = task_dir / "task.json"
        if not await asyncio.to_thread(lambda: task_file.exists()):
            return {"ok": False, "error": "Task not found"}

        def _clear_sync():
            task_meta = load_json(task_file)
            dataset_id, version = self._task_to_ds(task_meta)
            removed = 0

            # 每图 JSON（新存储）
            anns = self._load_all_image_anns(dataset_id, version)
            for image_id, ann in anns.items():
                if ann.get("ai_annotated"):
                    self._delete_image_ann(dataset_id, version, image_id)
                    removed += 1

            # 旧版本任务集中式标注文件（兼容清理，避免遗留 AI 标记）
            legacy_file = task_dir / "annotations.json"
            try:
                if legacy_file.exists():
                    legacy = load_json(legacy_file)
                    before = len(legacy)
                    for image_id in list(legacy.keys()):
                        if legacy[image_id].get("ai_annotated"):
                            del legacy[image_id]
                    if len(legacy) != before:
                        save_json_atomic(legacy_file, legacy)
                        removed += before - len(legacy)
            except Exception:
                pass
            return {"ok": True, "removed": removed}

        return await asyncio.to_thread(_clear_sync)

    async def clear_all_annotations(self, task_id: str):
        """清除该任务数据集的全部标注框（AI 标注与人工标注一视同仁）。

        与 clear-ai 的差别：不按 ai_annotated 过滤，所有图片的 boxes 一律清空。
        安全措施：
        - 已封板 / 已入桶 / 已归档的数据集拒绝执行（与「删除图片」同一套拦截）；
        - 清空前把 annotations/*.json + labels/*.txt + meta.json 打包备份到
          ANNOTATIONS_DIR/backups/，误清可解压回退；
        - 只清框，保留每图 JSON 的样本类型标记（normal/hard/background）与 created_at；
        - 同步删除派生的 labels/*.txt（txt 由标注 JSON 派生，留着会造成两份真相），
          并回退 meta 的标注计数与「审核通过」状态 —— 否则一个空标注集会绕过人工复核直接灌桶。

        与 clear-ai / clean-overlaps 一致，不回填 PG 影子库（一图一 JSON 才是唯一权威）。
        """
        def _sync():
            task_dir = self.annotations_dir / task_id
            task_file = task_dir / "task.json"
            if not task_file.exists():
                raise ValueError(f"标注任务 {task_id} 不存在")
            task_meta = load_json(task_file)
            dataset_id, version = self._task_to_ds(task_meta)
            if not dataset_id:
                raise ValueError("标注任务未绑定数据集，无法清除标注")

            dataset_dir = self.datasets_dir / dataset_id
            meta_path = dataset_dir / "meta.json"
            if not meta_path.exists():
                raise ValueError(f"数据集 {dataset_id} 不存在")
            meta = load_json(meta_path)
            if meta.get("status") != "prepared":
                raise ValueError("数据集尚未准备（prepare），无法清除标注")
            if meta.get("stage") == "sealed":
                raise ValueError("数据集已封板（只读），无法清除标注")
            if meta.get("bucket_archived"):
                raise ValueError("数据集已归档入桶，无法清除标注（需先解除归档）")
            if int(meta.get("bucket_pending") or 0) + int(meta.get("bucket_sealed") or 0) > 0:
                raise ValueError("数据集已部分入桶，无法清除标注（需先解除归档）")

            # labels 目录只做有限深度探测（平铺 / {version}/labels），
            # 不 rglob 整棵数据集树——万张图的数据集那样会慢到卡住请求线程
            labels_dirs = [d for d in list(dataset_dir.glob("labels")) + list(dataset_dir.glob("*/labels")) if d.is_dir()]
            anns = self._load_all_image_anns(dataset_id, version)
            has_any_box = any(a.get("boxes") for a in anns.values())
            has_txt = any(
                f.name != "classes.txt"
                for d in labels_dirs
                for f in d.rglob("*.txt")
            )

            # 1) 先备份（无内容可备份时跳过，避免留下一堆空 zip）
            backup_path = ""
            if has_any_box or has_txt:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = self.annotations_dir / "backups"
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = str(backup_dir / f"{dataset_id}_annotations_{ts}.zip")
                with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    ann_dir = dataset_dir / "annotations"
                    if ann_dir.is_dir():
                        for f in sorted(ann_dir.glob("*.json")):
                            zf.write(f, f"annotations/{f.name}")
                    for d in labels_dirs:
                        for f in sorted(d.rglob("*.txt")):
                            zf.write(f, f"{d.relative_to(dataset_dir).as_posix()}/{f.name}")
                    if meta_path.exists():
                        zf.write(meta_path, "meta.json")

            # 2) 清空每图 JSON 的框与 AI/人工复核标记（保留样本类型等元数据）
            cleared_images = cleared_boxes = 0
            for image_id, ann in anns.items():
                boxes = ann.get("boxes") or []
                if not (boxes or ann.get("ai_annotated") or ann.get("reviewed")):
                    continue
                cleared_boxes += len(boxes)
                cleared_images += 1
                self._save_image_ann(dataset_id, version, image_id, {
                    **ann,
                    "boxes": [],
                    "ai_annotated": False,
                    "reviewed": False,
                    "updated_at": datetime.now().isoformat(),
                })

            # 3) 历史任务集中式标注文件一并清空，避免旧数据回灌
            legacy_file = task_dir / "annotations.json"
            if legacy_file.exists():
                save_json_atomic(legacy_file, {})

            # 4) 删除派生的 YOLO txt（classes.txt 是类别表，不属于标注，保留）
            removed_txt = 0
            for d in labels_dirs:
                for f in list(d.rglob("*.txt")):
                    if f.name == "classes.txt":
                        continue
                    try:
                        f.unlink()
                        removed_txt += 1
                    except OSError:
                        pass

            # 5) 回退数据集 meta 的标注计数与审核通过状态
            meta["label_count"] = 0
            meta["has_labels"] = False
            meta["review_approved"] = False
            meta.pop("review_approved_at", None)
            save_json_atomic(meta_path, meta)

            logger.info(
                f"clear-all annotations {dataset_id} (task={task_id}): "
                f"images={cleared_images} boxes={cleared_boxes} txt={removed_txt} backup={backup_path}"
            )
            return {
                "ok": True,
                "cleared_images": cleared_images,
                "cleared_boxes": cleared_boxes,
                "removed_label_files": removed_txt,
                "backup": backup_path,
            }

        return await asyncio.to_thread(_sync)

    async def export_to_yolo(self, task_id: str):
        """导出标注为YOLO格式；无框/未标注图片写 0 字节空标签文件（negative sample 约定）"""
        def _export_to_yolo_sync():
            try:
                task_dir = self.annotations_dir / task_id
                task_file = task_dir / "task.json"
                
                if not task_file.exists():
                    return {"ok": False, "error": "Task not found"}
                
                # 读取任务信息
                task_meta = load_json(task_file)
                dataset_id = task_meta["dataset_id"]
                version = task_meta["version"]
                # 标注读取：优先每图 JSON，兼容历史任务集中式 annotations.json
                annotations = self._load_all_image_anns(dataset_id, version)
                legacy_file = task_dir / "annotations.json"
                try:
                    if legacy_file.exists():
                        for img_id, ann in load_json(legacy_file).items():
                            annotations.setdefault(img_id, ann)
                except Exception:
                    pass
                
                # 获取数据集labels目录（数据集一律为 datasets/{dataset_id}，无 version 子目录）
                dataset_id = task_meta["dataset_id"]
                version = task_meta["version"]
                dataset_dir = self.datasets_dir / dataset_id
                
                if not dataset_dir.exists():
                    return {"ok": False, "error": f"Dataset directory not found: {dataset_dir}"}
                
                labels_dir = dataset_dir / "labels"
                labels_dir.mkdir(exist_ok=True)
                
                # 如果有train/val子目录，也创建对应的labels子目录
                images_dir = self._find_images_dir(dataset_dir)
                if images_dir:
                    for subdir in ["train", "val"]:
                        if (images_dir / subdir).exists():
                            (labels_dir / subdir).mkdir(exist_ok=True)
                
                exported_with_labels = 0
                exported_empty = 0
                seg_boxes = 0  # 带分割轮廓的框数，用于回显「本次导出含 N 条分割标注」
                errors = []
                # class_id（英文名）→ YOLO 数字索引（与任务类别列表顺序一致）
                _cls_idx = {str(c).lower(): i for i, c in enumerate(task_meta.get("classes") or [])}
                
                # 转换每个标注
                for item in task_meta["items"]:
                    try:
                        image_id = item["image_id"]
                        ann = annotations.get(image_id) or {}

                        width = item["width"]
                        height = item["height"]
                        boxes = ann.get("boxes") or []

                        # 确定标签文件路径（有框/无框两条分支都要写文件，故提到分支之前）
                        label_path = self._determine_label_path(item, images_dir, labels_dir, image_id)
                        label_path.parent.mkdir(parents=True, exist_ok=True)

                        if boxes:
                            # 逐框写行：有轮廓 → 分割行（只写顶点），无轮廓 → 检测行；
                            # 同一文件里混有分割行时，无轮廓的框退化成 4 角点分割行
                            polys = [clean_polygon(box.get("polygon")) for box in boxes]
                            has_seg = any(polys)
                            yolo_lines = []
                            for box, poly in zip(boxes, polys):
                                class_id = box["class_id"]
                                if isinstance(class_id, int) or (isinstance(class_id, str) and class_id.isdigit()):
                                    cls_id = int(class_id)  # 旧数据数字索引直接使用
                                else:
                                    cls_id = _cls_idx.get(str(class_id).lower(), 0)

                                yolo_lines.append(yolo_label_line(
                                    cls_id, (box["x1"], box["y1"], box["x2"], box["y2"]),
                                    poly, width, height, rect_as_poly=has_seg))
                                if poly:
                                    seg_boxes += 1

                            # 写入标签文件
                            _write_text(label_path, "\n".join(yolo_lines))
                            exported_with_labels += 1
                        else:
                            # 无框 / 未标注：写 0 字节空标签文件（YOLO negative sample 约定）
                            _write_text(label_path, "")
                            exported_empty += 1
                    except Exception as e:
                        error_msg = f"Error exporting annotation for image {item.get('image_id', 'unknown')}: {str(e)}"
                        print(error_msg)
                        errors.append(error_msg)
                        continue
                
                result = {
                    "ok": True,
                    # exported_count 保留旧键（= 有标签的图片数），向后兼容前端
                    "exported_count": exported_with_labels,
                    "exported_with_labels": exported_with_labels,
                    "exported_empty": exported_empty,
                    "segment_boxes": seg_boxes,
                }
                if errors:
                    result["errors"] = errors
                return result
            except Exception as e:
                error_msg = f"Export failed: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                return {"ok": False, "error": error_msg}
        
        return await asyncio.to_thread(_export_to_yolo_sync)

    async def export_reviewed_zip(self, task_id: str):
        """单独提取"人工修过"（reviewed=true 且有标注框）的图片 + YOLO 标签，打包成临时 zip。

        与 export_to_yolo 的区别：不写数据集目录，直接产出可下载的压缩包
        （images/ + labels/ + classes.txt + data.yaml），用户在标注页挑图精修后
        一键提取，无需自己去文件夹里翻。labels 为 YOLO 分割格式（轮廓点写在行尾）。
        """
        def _export_reviewed_zip_sync():
            try:
                task_dir = self.annotations_dir / task_id
                task_file = task_dir / "task.json"
                if not task_file.exists():
                    return {"ok": False, "error": "Task not found"}

                task_meta = load_json(task_file)
                dataset_id, version = self._task_to_ds(task_meta)
                annotations = self._load_all_image_anns(dataset_id, version)
                legacy_file = task_dir / "annotations.json"
                try:
                    if legacy_file.exists():
                        for img_id, ann in load_json(legacy_file).items():
                            annotations.setdefault(img_id, ann)
                except Exception:
                    pass

                dataset_dir = self.datasets_dir / dataset_id
                images_dir = self._find_images_dir(dataset_dir) if dataset_dir.exists() else None
                classes = [str(c) for c in (task_meta.get("classes") or [])]
                cls_idx = {c.lower(): i for i, c in enumerate(classes)}

                exported = 0
                seg_boxes = 0
                skipped_no_image = []

                temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
                temp_zip.close()
                zip_path = temp_zip.name
                try:
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                        for item in task_meta["items"]:
                            image_id = item["image_id"]
                            ann = annotations.get(image_id)
                            # 只提取人工修过且目前有框的图（reviewed 为粘性标记，见 save_annotation）
                            if not ann or not ann.get("reviewed") or not ann.get("boxes"):
                                continue
                            width = item["width"]
                            height = item["height"]
                            yolo_lines = []
                            r_polys = [clean_polygon(b.get("polygon")) for b in ann["boxes"]]
                            r_has_seg = any(r_polys)
                            for box, poly in zip(ann["boxes"], r_polys):
                                class_id = box["class_id"]
                                if isinstance(class_id, int) or (isinstance(class_id, str) and class_id.isdigit()):
                                    cls_id = int(class_id)  # 旧数据数字索引直接使用
                                else:
                                    cls_id = cls_idx.get(str(class_id).lower(), 0)
                                yolo_lines.append(yolo_label_line(
                                    cls_id, (box["x1"], box["y1"], box["x2"], box["y2"]),
                                    poly, width, height, rect_as_poly=r_has_seg))
                                if poly:
                                    seg_boxes += 1
                            # 定位原图文件（与 _determine_label_path 相同的扩展名探测方式）
                            image_file = None
                            if images_dir:
                                for ext in [".jpg", ".png", ".jpeg", ".webp", ".bmp"]:
                                    candidates = list(images_dir.rglob(f"{image_id}{ext}"))
                                    if candidates:
                                        image_file = candidates[0]
                                        break
                            if image_file is None:
                                skipped_no_image.append(image_id)
                                continue
                            zipf.write(image_file, f"images/{image_file.name}")
                            zipf.writestr(f"labels/{image_file.stem}.txt", "\n".join(yolo_lines))
                            exported += 1
                        # 类别清单与最小 data.yaml，解压后可直接喂给 ultralytics 训练
                        if classes:
                            zipf.writestr("classes.txt", "\n".join(classes))
                            zipf.writestr(
                                "data.yaml",
                                yaml.dump(
                                    {"path": "", "train": "images", "val": "images", "nc": len(classes), "names": classes},
                                    allow_unicode=True,
                                ),
                            )
                    if exported == 0:
                        os.unlink(zip_path)
                        return {"ok": False, "error": "还没有\"已修过\"的图片可提取：请先修几张图（画/改一下轮廓并自动保存），列表筛子里\"轮廓·已修过\"有数量后再点"}
                except Exception:
                    if os.path.exists(zip_path):
                        os.unlink(zip_path)
                    raise
                return {
                    "ok": True,
                    "zip_path": zip_path,
                    "filename": f"{task_id}_reviewed.zip",
                    "exported_count": exported,
                    "segment_boxes": seg_boxes,
                    "skipped_no_image": skipped_no_image,
                }
            except Exception as e:
                error_msg = f"Export reviewed zip failed: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                return {"ok": False, "error": error_msg}

        return await asyncio.to_thread(_export_reviewed_zip_sync)

    def _determine_label_path(self, item: dict, images_dir: Path, labels_dir: Path, image_id: str) -> Path:
        """确定标签文件路径"""
        image_path_str = item.get("image_path", "")
        
        # 方法1：通过 image_id 查找对应的图片文件（最可靠）
        label_path = None
        if images_dir:
            # 查找对应的图片文件
            image_file = None
            for ext in ['.jpg', '.png', '.jpeg']:
                candidates = list(images_dir.rglob(f"{image_id}{ext}"))
                if candidates:
                    image_file = candidates[0]
                    break
            
            if image_file:
                # 获取图片相对于 images_dir 的路径
                try:
                    relative_path = image_file.relative_to(images_dir)
                    label_path = labels_dir / relative_path.with_suffix('.txt')
                except ValueError:
                    # 如果计算失败，使用 image_id 作为文件名
                    label_path = labels_dir / f"{image_id}.txt"
        
        # 方法2：如果方法1失败，尝试从 image_path 解析
        if label_path is None:
            # image_path 格式：datasets/ds_xxx/images/xxx.jpg 或 datasets/ds_xxx/images/train/xxx.jpg
            if image_path_str and ('/images/' in image_path_str or '\\images\\' in image_path_str):
                # 提取 images/ 之后的部分
                if '/images/' in image_path_str:
                    parts = image_path_str.split('/images/', 1)
                else:
                    parts = image_path_str.split('\\images\\', 1)
                if len(parts) > 1:
                    relative_path_str = parts[1]
                    label_path = labels_dir / Path(relative_path_str).with_suffix('.txt')
                else:
                    label_path = labels_dir / f"{image_id}.txt"
            else:
                # 如果路径中没有 images/，直接使用 image_id
                label_path = labels_dir / f"{image_id}.txt"
        
        return label_path

    async def import_yolo_labels(self, task_id: str, zip_path: Path) -> dict:
        """导入 YOLO 格式标注（zip 包内含 labels/*.txt，YOLO 归一化格式：class cx cy w h）。

        按 .txt 文件名（stem）与任务图片匹配，转成像素坐标写入每图 JSON 标注文件。
        同名图片已有标注会被导入的标注覆盖（以外部工具标注为准）。
        配合 X-AnyLabeling 等外部工具完成"专业工具标注 → 平台闭环"的衔接。
        """
        task_dir = self.annotations_dir / task_id
        task_file = task_dir / "task.json"

        if not await asyncio.to_thread(lambda: task_file.exists()):
            return {"ok": False, "error": "Task not found"}

        def _import_sync():
            task_meta = load_json(task_file)
            items = task_meta["items"]
            classes = task_meta.get("classes", [])
            dataset_id, version = self._task_to_ds(task_meta)

            # 解压 zip 到临时目录（S-02：路径遍历 + 解压炸弹校验，收口 utils.zip_guard）
            tmp_dir = task_dir / "import_tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zip_guard.guarded_extractall(zf, tmp_dir, strict=True)
            except (zip_guard.UnsafeZipMember, zip_guard.ZipBombRisk) as e:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return {"ok": False, "error": str(e)}

            # 优先取 labels 目录下的 txt，其次临时目录内所有 txt
            labels_dir = tmp_dir / "labels"
            label_files = list(labels_dir.rglob("*.txt")) if labels_dir.exists() else list(tmp_dir.rglob("*.txt"))

            # 建立文件名 stem → item 映射（image_id 与图片文件名一般一致）
            by_id = {item["image_id"]: item for item in items}
            by_stem = {}
            for item in items:
                by_stem.setdefault(Path(item["image_path"]).stem, item)

            imported = 0
            imported_ids = []
            skipped = []
            for tf in label_files:
                stem = tf.stem
                item = by_id.get(stem) or by_stem.get(stem)
                if not item:
                    skipped.append(stem)
                    continue
                try:
                    boxes = self._load_yolo_labels(tf, item["width"], item["height"])
                except ValueError as e:
                    skipped.append(f"{stem} ({e})")
                    continue
                # 过滤类别越界的框（避免与任务类别数不一致时写入脏数据）
                valid = [b for b in boxes if 0 <= b["class_id"] < len(classes)]
                if len(valid) != len(boxes):
                    skipped.append(f"{stem} (越界类别已忽略)")
                # 写入每图 JSON（覆盖同名图片已有标注，以外部工具标注为准）
                self._save_image_ann(dataset_id, version, item["image_id"], {
                    "image_id": item["image_id"],
                    "image_path": item["image_path"],
                    "width": item["width"],
                    "height": item["height"],
                    "model_id": task_meta.get("model_id", ""),
                    "dataset_id": dataset_id,
                    "version": version,
                    "boxes": valid,
                    "updated_at": datetime.now().isoformat(),
                    "ai_annotated": False,
                })
                imported += 1
                imported_ids.append(item["image_id"])

            shutil.rmtree(tmp_dir, ignore_errors=True)
            return {"ok": True, "imported": imported, "skipped": skipped,
                    "imported_ids": imported_ids}

        result = await asyncio.to_thread(_import_sync)
        # 影子库收口：导入即产生标签，批量回填 task_items.has_label（不污染对外响应）
        ids = result.pop("imported_ids", None) or []
        if result.get("ok") and ids:
            from src.db import shadow
            await shadow.mark_items_labeled(task_id, ids)
        return result

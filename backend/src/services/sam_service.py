"""SAM 大模型智能预标注服务

基于 ultralytics 内置 SAM 与 YOLO-World 实现文本驱动的自动预标注：
  1. YOLO-World 按类别名检测 → 得到候选框
  2. SAM 对每个框做分割 → 计算最小外接矩形（精修）
  3. 返回与现有标注体系完全兼容的 BBox

参考设计文档 4.2 节，独立从零实现（不复制 AGPL 项目代码）。
"""
import json
import asyncio
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.core.settings import settings
from src.services.device_manager import get_best_device
from src.services.annotation_service import AnnotationService


DEFAULT_CONFIG = {
    "detector": "yolo_world",        # yolo_world | grounding_dino | none
    "detector_weights": "yolov8s-world.pt",
    "grounding_dino_model": "IDEA-Research/grounding-dino-tiny",  # Transformers 集成的 GD 模型名
    "grounding_dino_size": 640,      # GroundingDINO 推理分辨率（越小越快，默认 800）
    "grounding_dino_min_conf": 0.25, # GD 置信度下限（GD 分数尺度与 YOLO 不同，过低会大量误检）
    "sam_enabled": True,             # 是否启用 SAM 分割精修
    "sam_weights": "sam_b.pt",
    "imgsz": 640,                    # 检测器输入尺寸
    "sam_imgsz": 1024,               # SAM 输入尺寸
    "conf": 0.10,                    # 默认置信度阈值（调低以提升召回）
    "iou": 0.40,                     # NMS IoU 阈值（调低以合并遮挡/重叠产生的重复框）
    "half": False,                   # 半精度，省显存
    "device": "auto",                # auto | cpu | GPU 索引字符串
}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class BatchTask:
    """内存中的批量预标注任务状态"""

    def __init__(self, batch_id: str, task_id: str, classes: List[str], conf: float):
        self.batch_id = batch_id
        self.task_id = task_id
        self.classes = classes
        self.prompts: Optional[List[str]] = None
        self.conf = conf
        self.total = 0
        self.done = 0
        self.boxes_written = 0
        self.current_image: Optional[str] = None
        self.annotated_images: List[str] = []  # 本次批量预标注实际写入过框的图片ID
        self.cancelled = False
        self.status = "running"  # running | done | cancelled | error
        self.error: Optional[str] = None
        self.result_summary: Optional[dict] = None


class SAMService:
    def __init__(self):
        self._detector = None
        self._detector_name = None
        self._sam = None
        self._sam_name = None
        # GroundingDINO（Transformers 集成）缓存
        self._gd_processor = None
        self._gd_model = None
        self._gd_name = None
        # 缓存最后一次传给 YOLO-World 的提示词列表，避免多类别时每张图重复重新编码
        self._detector_prompts: Optional[List[str]] = None
        self._lock = asyncio.Lock()
        self.batch_tasks: Dict[str, BatchTask] = {}
        self.annotation_service = AnnotationService()

    # ------------------------------------------------------------------
    # 配置读写
    # ------------------------------------------------------------------
    def _config_path(self) -> Path:
        return settings.SAM_CONFIG_FILE

    def _read_config(self) -> dict:
        cfg = dict(DEFAULT_CONFIG)
        p = self._config_path()
        if p.exists():
            try:
                cfg.update(_load_json(p))
            except Exception:
                pass
        return cfg

    def _save_config(self, cfg: dict) -> None:
        p = self._config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    def get_config(self) -> dict:
        return self._read_config()

    def update_config(self, request) -> dict:
        cfg = self._read_config()
        updates = request.model_dump(exclude_unset=True) if hasattr(request, "model_dump") else {}
        for k, v in updates.items():
            if v is not None:
                cfg[k] = v
        self._save_config(cfg)
        return cfg

    # ------------------------------------------------------------------
    # 检测模型管理
    # ------------------------------------------------------------------
    def _model_locations(self) -> List[Path]:
        """检测模型可能存放的位置：统一目录 + 用户上传的自定义模型目录 + 项目 backend 根目录（兼容旧文件）"""
        return [settings.SAM_MODELS_DIR, settings.CUSTOM_MODELS_DIR, settings.BASE_DIR]

    def _find_model_file(self, weights: str) -> str:
        """将配置里的权重名解析为实际文件路径；找不到则原样返回交给 ultralytics 处理"""
        p = Path(weights)
        if p.is_absolute() and p.exists():
            return str(p)
        for d in self._model_locations():
            cand = d / weights
            if cand.exists():
                return str(cand)
        return weights

    def list_models(self) -> List[dict]:
        """列出可用的检测模型（.pt 文件）"""
        found = {}
        for d in self._model_locations():
            if not d.exists():
                continue
            for f in d.glob("*.pt"):
                found.setdefault(f.name, {
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / 1e6, 1),
                    "path": str(f),
                })
        return sorted(found.values(), key=lambda x: x["name"])

    def save_model(self, filename: str, data: bytes) -> Path:
        """保存上传的检测模型到统一目录"""
        settings.SAM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = settings.SAM_MODELS_DIR / filename
        dest.write_bytes(data)
        return dest

    # ------------------------------------------------------------------
    # 设备解析
    # ------------------------------------------------------------------
    def _resolve_device(self, cfg: dict) -> str:
        device = cfg.get("device", "auto")
        if device == "auto":
            return get_best_device()
        return device

    # ------------------------------------------------------------------
    # 模型加载（单例缓存）
    # ------------------------------------------------------------------
    def _load_detector(self, cfg: dict):
        det_type = cfg.get("detector", "yolo_world")
        if det_type == "none":
            return None
        if det_type == "grounding_dino":
            # GroundingDINO 走 Transformers 集成，加载后返回占位标记表示已就绪
            self._load_grounding_dino(cfg)
            return True
        # 默认 yolo_world
        if self._detector is not None:
            return self._detector
        weights = cfg.get("detector_weights", "yolov8s-world.pt")
        from ultralytics import YOLO
        self._detector = YOLO(self._find_model_file(weights))
        self._detector_name = weights
        return self._detector

    def _resolve_gd_model_name(self, model_name: str) -> str:
        """本地优先解析 GroundingDINO 模型：若已下载到 SAM_MODELS_DIR/<名称> 则用本地路径，避免联网"""
        tail = model_name.rstrip("/").rsplit("/", 1)[-1]
        local_dir = settings.SAM_MODELS_DIR / tail
        if local_dir.exists() and (local_dir / "config.json").exists():
            return str(local_dir)
        return model_name

    def _load_grounding_dino(self, cfg: dict):
        """加载 Transformers 集成的 GroundingDINO（模型 + 处理器）"""
        if self._gd_model is not None:
            return self._gd_processor, self._gd_model
        model_name = self._resolve_gd_model_name(cfg.get("grounding_dino_model", "IDEA-Research/grounding-dino-tiny"))
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self._gd_processor = AutoProcessor.from_pretrained(model_name)
        self._gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        self._gd_name = model_name
        self._gd_model.to(self._resolve_device(cfg))
        self._gd_model.eval()
        return self._gd_processor, self._gd_model

    def _load_sam(self, cfg: dict):
        if self._sam is not None:
            return self._sam
        if not cfg.get("sam_enabled", True):
            return None
        weights = cfg.get("sam_weights", "sam_b.pt")
        from ultralytics import SAM
        # 优先从本地模型目录（sam/、custom/、backend 根目录）解析权重，找不到再由 ultralytics 自动联网下载
        self._sam = SAM(self._find_model_file(weights))
        self._sam_name = weights
        return self._sam

    # ------------------------------------------------------------------
    # 可用性 / 校验
    # ------------------------------------------------------------------
    async def is_available(self) -> dict:
        cfg = self._read_config()

        def _check() -> dict:
            try:
                det_ok = True
                if cfg.get("detector") != "none":
                    det = self._load_detector(cfg)
                    det_ok = det is not None
                sam_ok = True
                if cfg.get("sam_enabled", True):
                    sam = self._load_sam(cfg)
                    sam_ok = sam is not None
                return {
                    "available": det_ok and sam_ok,
                    "detector_ok": det_ok,
                    "sam_ok": sam_ok,
                    "device": self._resolve_device(cfg),
                }
            except Exception as e:
                return {"available": False, "error": str(e)}

        return await asyncio.to_thread(_check)

    async def validate_model(self) -> dict:
        cfg = self._read_config()

        def _validate() -> dict:
            try:
                checks = {}
                if cfg.get("detector") != "none":
                    if cfg.get("detector") == "grounding_dino":
                        self._load_grounding_dino(cfg)
                        checks["detector"] = "ok"
                    else:
                        det = self._load_detector(cfg)
                        det.set_classes(["test"])
                        det.predict(
                            np.zeros((64, 64, 3), dtype=np.uint8),
                            conf=0.5,
                            verbose=False,
                            imgsz=64,
                            device=self._resolve_device(cfg),
                        )
                        checks["detector"] = "ok"
                if cfg.get("sam_enabled", True):
                    self._load_sam(cfg)
                    checks["sam"] = "ok"
                return {"ok": True, "checks": checks}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        return await asyncio.to_thread(_validate)

    # ------------------------------------------------------------------
    # 单图预标注
    # ------------------------------------------------------------------
    async def auto_label(self, task_id: str, image_id: str,
                         classes: List[str], conf: Optional[float] = None,
                         prompts: Optional[List[str]] = None) -> dict:
        cfg = self._read_config()
        conf = conf if conf is not None else cfg.get("conf", 0.25)

        item = await self._get_item(task_id, image_id)
        if not item:
            raise ValueError("Image not found in task")

        img_abs = settings.DATA_DIR / item["image_path"]
        if not img_abs.exists():
            raise ValueError(f"Image file not found: {img_abs}")

        # 模型推理非线程安全，加锁串行化，避免并发访问导致模型状态损坏
        async with self._lock:
            boxes = await asyncio.to_thread(self._run_auto_label_sync, img_abs, classes, conf, cfg, prompts)
        return {
            "image_id": image_id,
            "width": item["width"],
            "height": item["height"],
            "boxes": boxes,
        }

    async def interactive_label(self, task_id: str, image_id: str,
                                classes: List[str], conf: Optional[float] = None,
                                prompts: Optional[List[str]] = None,
                                region: Optional[dict] = None) -> dict:
        """交互式标注：只在用户框选的局部区域（region，图像坐标）内做检测。

        相比全图盲扫，局部检测既减少误标，又能针对感兴趣区域精细标注，
        配合"点击/框选提示区域 + 文本提示"的交互方式，正中"误标过多"痛点。
        region 格式：{"x1":..,"y1":..,"x2":..,"y2":..}；缺省则全图检测。
        """
        cfg = self._read_config()
        conf = conf if conf is not None else cfg.get("conf", 0.25)

        item = await self._get_item(task_id, image_id)
        if not item:
            raise ValueError("Image not found in task")

        img_abs = settings.DATA_DIR / item["image_path"]
        if not img_abs.exists():
            raise ValueError(f"Image file not found: {img_abs}")

        # 模型推理非线程安全，加锁串行化
        async with self._lock:
            boxes = await asyncio.to_thread(
                self._run_interactive_label_sync, img_abs, classes, conf, cfg, prompts, region
            )
        return {
            "image_id": image_id,
            "width": item["width"],
            "height": item["height"],
            "boxes": boxes,
        }

    def _run_interactive_label_sync(self, img_abs: Path, classes: List[str], conf: float,
                                    cfg: dict, prompts=None, region: Optional[dict] = None) -> list:
        """交互式标注同步执行：裁剪到区域 → 区域检测 → 坐标映射回原图 → 合并 → (可选)SAM 精修"""
        import os
        from tempfile import NamedTemporaryFile
        from PIL import Image

        if region:
            image = Image.open(img_abs).convert("RGB")
            w, h = image.size
            x1 = max(0, int(round(region.get("x1", 0))))
            y1 = max(0, int(round(region.get("y1", 0))))
            x2 = min(w, int(round(region.get("x2", w))))
            y2 = min(h, int(round(region.get("y2", h))))
            if x2 - x1 < 5 or y2 - y1 < 5:
                raise ValueError("Region too small")
            crop = image.crop((x1, y1, x2, y2))
            tmp = NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            tmp_path = Path(tmp.name)
            try:
                crop.save(tmp_path)
                det_boxes = self._detect(tmp_path, classes, conf, cfg, prompts)
                # 把裁剪图坐标映射回原图坐标
                for b in det_boxes:
                    b["x1"] += x1
                    b["y1"] += y1
                    b["x2"] += x1
                    b["y2"] += y1
            finally:
                if tmp_path.exists():
                    os.unlink(tmp_path)
        else:
            det_boxes = self._detect(img_abs, classes, conf, cfg, prompts)

        det_boxes = self._merge_same_class_boxes(det_boxes)
        det_boxes = self._merge_cross_class_boxes(det_boxes)
        if cfg.get("sam_enabled", True):
            # SAM 在原图上按映射回原图的框做分割精修
            return self._refine_with_sam(img_abs, det_boxes, cfg)
        return det_boxes

    def _run_auto_label_sync(self, img_abs: Path, classes: List[str], conf: float, cfg: dict, prompts=None) -> list:
        """同步执行：检测 → 合并被遮挡分裂的同类框 → (可选) SAM 精修 → 返回 BBox 列表"""
        det_boxes = self._detect(img_abs, classes, conf, cfg, prompts)
        # 合并被遮挡物（如电线杆）分裂成多个的同类框，避免同一物体被识别成两个
        det_boxes = self._merge_same_class_boxes(det_boxes)
        # 跨类别高度重合合并（GD 常把同一目标误标成不同类别，导致高度重合框残留）
        det_boxes = self._merge_cross_class_boxes(det_boxes)
        if cfg.get("sam_enabled", True):
            return self._refine_with_sam(img_abs, det_boxes, cfg)
        return det_boxes

    def _merge_same_class_boxes(self, boxes: list, iou_thr: float = 0.15, dist_thr: float = 0.6) -> list:
        """合并同类且高度重叠/紧邻的框，解决遮挡导致的同一物体被识别成多个框的问题。

        同一物体被遮挡物（如电线杆）隔开时，YOLO-World 常输出多个相邻/重叠的同类框。
        这里对同类别、IoU 高于阈值或中心距离较近的框取并集合并为一个外接框。
        """
        if not boxes:
            return boxes

        def _iou(a, b):
            ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
            ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            a_area = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
            b_area = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
            return inter / (a_area + b_area - inter)

        def _center_dist(a, b):
            ca = ((a["x1"] + a["x2"]) / 2, (a["y1"] + a["y2"]) / 2)
            cb = ((b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2)
            return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5

        # 按类别分组，逐组合并
        by_class: Dict[int, list] = {}
        for b in boxes:
            by_class.setdefault(b["class_id"], []).append(b)

        merged: list = []
        for cid, group in by_class.items():
            group = sorted(group, key=lambda b: b.get("score", 0.0), reverse=True)
            used = [False] * len(group)
            for i in range(len(group)):
                if used[i]:
                    continue
                # 以当前框为基准，合并所有与之重叠/紧邻的同类框
                base = dict(group[i])
                base_diag = ((base["x2"] - base["x1"]) ** 2 + (base["y2"] - base["y1"]) ** 2) ** 0.5 or 1.0
                for j in range(i + 1, len(group)):
                    if used[j]:
                        continue
                    other = group[j]
                    if _iou(base, other) >= iou_thr:
                        # 合并取外接矩形，保留更高 score
                        base["x1"] = min(base["x1"], other["x1"])
                        base["y1"] = min(base["y1"], other["y1"])
                        base["x2"] = max(base["x2"], other["x2"])
                        base["y2"] = max(base["y2"], other["y2"])
                        base["score"] = max(base.get("score", 0.0), other.get("score", 0.0))
                        used[j] = True
                    elif _center_dist(base, other) <= dist_thr * base_diag:
                        # 中心距离很近的同类框（遮挡分裂但几乎不重叠）也合并
                        base["x1"] = min(base["x1"], other["x1"])
                        base["y1"] = min(base["y1"], other["y1"])
                        base["x2"] = max(base["x2"], other["x2"])
                        base["y2"] = max(base["y2"], other["y2"])
                        base["score"] = max(base.get("score", 0.0), other.get("score", 0.0))
                        used[j] = True
                merged.append(base)
        return merged

    def _merge_cross_class_boxes(self, boxes: list, iou_thr: float = 0.5) -> list:
        """合并不同类别但高度重合的框（GD 常把同一目标误标成不同类别）。

        不同类别两个框 IoU ≥ 0.5 时几乎可判定为同一目标的重复检测，只保留 score 更高者。
        """
        if not boxes:
            return boxes

        def _iou(a, b):
            ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
            ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            aa = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
            bb = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
            return inter / (aa + bb - inter)

        # 按 score 降序，保留高置信度框，丢弃与已保留框高度重合的任意类别框
        boxes = sorted(boxes, key=lambda b: b.get("score", 0.0), reverse=True)
        out = []
        for b in boxes:
            if any(_iou(kept, b) >= iou_thr for kept in out):
                continue
            out.append(b)
        return out

    async def clean_task_annotations(self, task_id: str) -> dict:
        """清洗一个标注任务中所有图片的高度重合标注框（跨类别 IoU≥0.5、同类高度重叠/紧邻）。

        复用与检测时完全相同的合并逻辑（_merge_same_class_boxes + _merge_cross_class_boxes），
        一次性根治历史遗留的重叠标注——无论之前用的是哪个检测模型，都统一清理，无需按模型分别处理。
        """
        task_dir = settings.ANNOTATIONS_DIR / task_id
        annotations_file = task_dir / "annotations.json"
        if not annotations_file.exists():
            return {"ok": False, "error": "Task not found"}
        return await asyncio.to_thread(self._clean_task_annotations_sync, annotations_file)

    def _clean_task_annotations_sync(self, annotations_file: Path) -> dict:
        try:
            annotations = _load_json(annotations_file)
        except Exception as e:
            return {"ok": False, "error": f"读取标注文件失败: {e}"}

        images_cleaned = 0
        boxes_removed = 0
        for image_id, ann in annotations.items():
            boxes = ann.get("boxes") or []
            if not boxes:
                continue
            orig_len = len(boxes)
            merged = self._merge_same_class_boxes(boxes)
            merged = self._merge_cross_class_boxes(merged)
            if len(merged) < orig_len:
                ann["boxes"] = merged
                ann["updated_at"] = datetime.now().isoformat()
                images_cleaned += 1
                boxes_removed += orig_len - len(merged)

        if images_cleaned > 0:
            _save_json(annotations_file, annotations)
        return {
            "ok": True,
            "images_cleaned": images_cleaned,
            "boxes_removed": boxes_removed,
        }

    def _detect(self, img_abs, classes, conf, cfg, prompts=None) -> list:
        if cfg.get("detector") == "grounding_dino":
            return self._detect_grounding_dino(img_abs, classes, conf, cfg, prompts)
        return self._detect_yolo(img_abs, classes, conf, cfg, prompts)

    def _detect_yolo(self, img_abs, classes, conf, cfg, prompts=None) -> list:
        det = self._load_detector(cfg)
        if det is None:
            return []
        # 用英文提示词（prompts）做文本驱动检测，中文类别识别率更高。
        # prompts 与 classes 一一对应，因此检测结果的 class_id 仍对应 classes 索引。
        text_prompts = prompts if prompts else classes
        # 类别提示词未变化时复用已编码的检测头，避免多类别时每张图重复 set_classes（耗时且易超时）
        if self._detector_prompts != text_prompts:
            try:
                det.set_classes(text_prompts)
                self._detector_prompts = list(text_prompts)
            except Exception as e:
                print(f"Warning: set_classes failed: {e}")
        results = det.predict(
            img_abs,
            conf=conf,
            iou=cfg.get("iou", 0.40),
            imgsz=cfg.get("imgsz", 640),
            verbose=False,
            device=self._resolve_device(cfg),
        )
        out = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                out.append({
                    "class_id": int(boxes.cls[i]),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "score": float(boxes.conf[i]),
                })
        return out

    def _detect_grounding_dino(self, img_abs, classes, conf, cfg, prompts=None) -> list:
        """GroundingDINO（Transformers 集成）文本驱动检测。

        caption 用 prompts（英文提示词，提升识别率），与 classes 一一对应；
        通过 phrases -> prompt 文本匹配映射回 classes 索引得到 class_id。
        """
        processor, model = self._load_grounding_dino(cfg)
        text_prompts = prompts if prompts else classes
        if not text_prompts:
            return []
        # GD 的 post_process_grounded_object_detection 要求短语以 "." 分隔且以句点结尾，
        # 否则结果解析为空（单类别时 join 无句点会返回 0 个目标）
        caption = " . ".join(text_prompts) + " ."
        device = self._resolve_device(cfg)
        try:
            import torch
            from PIL import Image
            image = Image.open(img_abs).convert("RGB")
            gd_size = int(cfg.get("grounding_dino_size", 640) or 640)
            # GD processor 的 size 需同时给定长短边（int 会被解析成仅含 shortest_edge 的非法字典）
            inputs = processor(
                images=image, text=caption,
                size={"shortest_edge": gd_size, "longest_edge": gd_size},
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # GD 分数尺度与 YOLO 不同：过低阈值会产生大量误检框，加上限下限保护
            gd_conf = max(float(conf), float(cfg.get("grounding_dino_min_conf", 0.25)))
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=gd_conf,
                text_threshold=gd_conf,
                target_sizes=[(image.height, image.width)],
            )[0]
        except Exception as e:
            # 不静默吞掉：模型未下载/加载失败时若返回空数组，用户会误以为"类别设置错误/图里没目标"
            raise RuntimeError(
                f"GroundingDINO 检测器不可用：{e}\n"
                "请先在右上角检测模型下拉切到 YOLO-World（如 yolov8s-world.pt，本地已就绪），"
                "或确认已下载 IDEA-Research/grounding-dino-tiny 模型。"
            ) from e

        out = []
        boxes = results.get("boxes")
        scores = results.get("scores")
        labels = results.get("labels")
        if boxes is None or scores is None or labels is None:
            return out
        for box, score, label in zip(boxes, scores, labels):
            class_id = self._match_phrase(text_prompts, label)
            if class_id is None:
                continue
            x1, y1, x2, y2 = box.tolist()
            out.append({
                "class_id": class_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "score": float(score),
            })
        return out

    def _match_phrase(self, prompts: List[str], label: str) -> Optional[int]:
        """将 GroundingDINO 返回的短语标签匹配回 prompts 索引（class_id）"""
        if label is None:
            return None
        norm_label = label.strip().lower().replace(".", "").strip()
        for i, p in enumerate(prompts):
            norm_p = str(p).strip().lower().replace(".", "").strip()
            if norm_p and (norm_p == norm_label or norm_label == norm_p):
                return i
        # 兜底：标签包含某个 prompt 的完整词
        for i, p in enumerate(prompts):
            norm_p = str(p).strip().lower().replace(".", "").strip()
            if norm_p and (norm_p in norm_label or norm_label in norm_p):
                return i
        return None

    def _refine_with_sam(self, img_abs, det_boxes, cfg) -> list:
        if not det_boxes:
            return det_boxes
        sam = self._load_sam(cfg)
        if sam is None:
            return det_boxes

        boxes_np = np.array(
            [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in det_boxes],
            dtype=float,
        )
        try:
            results = sam.predict(
                img_abs,
                bboxes=boxes_np,
                verbose=False,
                imgsz=cfg.get("sam_imgsz", 1024),
                device=self._resolve_device(cfg),
            )
        except Exception as e:
            print(f"Warning: SAM refine failed, fallback to detect boxes: {e}")
            return det_boxes

        refined = []
        for idx, b in enumerate(det_boxes):
            mask = None
            for r in results:
                if getattr(r, "masks", None) is not None and len(r.masks) > idx:
                    mask = r.masks.data[idx].cpu().numpy()
                    break
            if mask is not None:
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    refined.append({
                        "class_id": b["class_id"],
                        "x1": float(xs.min()),
                        "y1": float(ys.min()),
                        "x2": float(xs.max()),
                        "y2": float(ys.max()),
                        "score": b.get("score", 0.0),
                    })
                    continue
            refined.append(b)
        return refined

    # ------------------------------------------------------------------
    # 批量预标注（异步后台）
    # ------------------------------------------------------------------
    async def batch_auto_label(self, task_id: str, classes: List[str],
                               conf: Optional[float] = None,
                               prompts: Optional[List[str]] = None) -> dict:
        cfg = self._read_config()
        conf = conf if conf is not None else cfg.get("conf", 0.25)

        meta = await self._load_task_meta(task_id)
        if not meta:
            raise ValueError("Task not found")
        # 按 image_id 去重，与图片列表一致（历史任务可能因平铺图+划分目录重复收集同一张图）
        seen = set()
        items = []
        for it in meta.get("items", []):
            if it.get("image_id") in seen:
                continue
            seen.add(it.get("image_id"))
            items.append(it)
        if not items:
            raise ValueError("Task has no images")

        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        bt = BatchTask(batch_id, task_id, classes, conf)
        bt.prompts = prompts
        bt.total = len(items)
        self.batch_tasks[batch_id] = bt

        asyncio.create_task(self._run_batch(batch_id, items))
        return {"batch_id": batch_id, "total": bt.total}

    async def _run_batch(self, batch_id: str, items: list) -> None:
        bt = self.batch_tasks[batch_id]
        cfg = self._read_config()
        try:
            # 预加载模型，避免逐张重复实例化（CPU 上模型加载较慢，提前检查取消）
            if bt.cancelled:
                bt.status = "cancelled"
                return
            await asyncio.to_thread(self._load_detector, cfg)
            if cfg.get("sam_enabled", True):
                await asyncio.to_thread(self._load_sam, cfg)
            # 模型加载完成后再次检查取消，避免加载期间点停止无效
            if bt.cancelled:
                bt.status = "cancelled"
                return

            for item in items:
                if bt.cancelled:
                    bt.status = "cancelled"
                    return
                image_id = item["image_id"]
                bt.current_image = image_id
                img_abs = settings.DATA_DIR / item["image_path"]

                # 单张图异常（如坏图、显存波动）只跳过该图，不中断整个批次
                try:
                    # 读取该图已有标注，按“缺失类别”增量标注，避免覆盖人工标注
                    existing = await self.annotation_service.get_image_annotation(bt.task_id, image_id)
                    existing_boxes = existing.get("boxes") or [] if existing else []
                    existing_class_ids = {b.get("class_id") for b in existing_boxes if b.get("class_id") is not None}
                    # 本次要标注的全部类别中，该图尚未标注的类别索引
                    missing_indices = [i for i in range(len(bt.classes)) if i not in existing_class_ids]
                    # 所有类别都已标注过，跳过该图
                    if not missing_indices:
                        bt.done += 1
                        continue

                    boxes = []
                    if img_abs.exists():
                        # 加锁串行化推理，避免与单图标注并发访问模型
                        async with self._lock:
                            boxes = await asyncio.to_thread(
                                self._run_auto_label_sync, img_abs, bt.classes, bt.conf, cfg, bt.prompts
                            )
                    # 只保留缺失类别的检测框（class_id 为全局索引，与 bt.classes 对齐）
                    missing_set = set(missing_indices)
                    new_boxes = [b for b in boxes if b["class_id"] in missing_set]
                    if new_boxes:
                        clean_existing = [
                            {"class_id": b.get("class_id"), "x1": b.get("x1"), "y1": b.get("y1"),
                             "x2": b.get("x2"), "y2": b.get("y2")}
                            for b in existing_boxes if b.get("x1") is not None
                        ]
                        clean_new = [
                            {"class_id": b["class_id"], "x1": b["x1"], "y1": b["y1"],
                             "x2": b["x2"], "y2": b["y2"]}
                            for b in new_boxes
                        ]
                        await self.annotation_service.save_annotation(bt.task_id, image_id, clean_existing + clean_new, ai_annotated=True)
                        bt.boxes_written += 1
                        bt.annotated_images.append(image_id)
                    bt.done += 1
                except Exception as e:
                    print(f"批量预标注跳过图片 {image_id}: {e}")
                    bt.done += 1
                    continue

            bt.status = "done"
            bt.result_summary = {"total": bt.total, "annotated": bt.boxes_written}
        except Exception as e:
            bt.status = "error"
            bt.error = str(e)

    async def get_batch_status(self, batch_id: str) -> Optional[dict]:
        bt = self.batch_tasks.get(batch_id)
        if not bt:
            return None
        return {
            "batch_id": bt.batch_id,
            "task_id": bt.task_id,
            "status": bt.status,
            "total": bt.total,
            "done": bt.done,
            "boxes_written": bt.boxes_written,
            "annotated_images": bt.annotated_images,
            "current_image": bt.current_image,
            "error": bt.error,
            "summary": bt.result_summary,
        }

    async def stop_batch(self, batch_id: str) -> Optional[dict]:
        bt = self.batch_tasks.get(batch_id)
        if not bt:
            return None
        bt.cancelled = True
        return {"ok": True, "status": bt.status}

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    async def _load_task_meta(self, task_id: str) -> Optional[dict]:
        task_file = settings.ANNOTATIONS_DIR / task_id / "task.json"
        if not task_file.exists():
            return None
        return await asyncio.to_thread(_load_json, task_file)

    async def _get_item(self, task_id: str, image_id: str) -> Optional[dict]:
        meta = await self._load_task_meta(task_id)
        if not meta:
            return None
        for item in meta.get("items", []):
            if item["image_id"] == image_id:
                return item
        return None
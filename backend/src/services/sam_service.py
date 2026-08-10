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
    "detector": "yolo_world",        # yolo_world | none
    "detector_weights": "yolov8s-world.pt",
    "sam_enabled": True,             # 是否启用 SAM 分割精修
    "sam_weights": "sam_b.pt",
    "imgsz": 640,                    # 检测器输入尺寸
    "sam_imgsz": 1024,               # SAM 输入尺寸
    "conf": 0.25,                    # 默认置信度阈值
    "half": False,                   # 半精度，省显存
    "device": "auto",                # auto | cpu | GPU 索引字符串
}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        if self._detector is not None:
            return self._detector
        if cfg.get("detector") == "none":
            return None
        weights = cfg.get("detector_weights", "yolov8s-world.pt")
        from ultralytics import YOLO
        self._detector = YOLO(weights)
        self._detector_name = weights
        return self._detector

    def _load_sam(self, cfg: dict):
        if self._sam is not None:
            return self._sam
        if not cfg.get("sam_enabled", True):
            return None
        weights = cfg.get("sam_weights", "sam_b.pt")
        from ultralytics import SAM
        self._sam = SAM(weights)
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

        boxes = await asyncio.to_thread(self._run_auto_label_sync, img_abs, classes, conf, cfg, prompts)
        return {
            "image_id": image_id,
            "width": item["width"],
            "height": item["height"],
            "boxes": boxes,
        }

    def _run_auto_label_sync(self, img_abs: Path, classes: List[str], conf: float, cfg: dict, prompts=None) -> list:
        """同步执行：检测 → (可选)SAM 精修 → 返回 BBox 列表"""
        det_boxes = self._detect(img_abs, classes, conf, cfg, prompts)
        if cfg.get("sam_enabled", True):
            return self._refine_with_sam(img_abs, det_boxes, cfg)
        return det_boxes

    def _detect(self, img_abs, classes, conf, cfg, prompts=None) -> list:
        det = self._load_detector(cfg)
        if det is None:
            return []
        # 用英文提示词（prompts）做文本驱动检测，中文类别识别率更高。
        # prompts 与 classes 一一对应，因此检测结果的 class_id 仍对应 classes 索引。
        text_prompts = prompts if prompts else classes
        try:
            det.set_classes(text_prompts)
        except Exception as e:
            print(f"Warning: set_classes failed: {e}")
        results = det.predict(
            img_abs,
            conf=conf,
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
        items = meta.get("items", [])
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
            # 预加载模型，避免逐张重复实例化
            await asyncio.to_thread(self._load_detector, cfg)
            if cfg.get("sam_enabled", True):
                await asyncio.to_thread(self._load_sam, cfg)

            for item in items:
                if bt.cancelled:
                    bt.status = "cancelled"
                    return
                image_id = item["image_id"]
                bt.current_image = image_id
                img_abs = settings.DATA_DIR / item["image_path"]

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
                    await self.annotation_service.save_annotation(bt.task_id, image_id, clean_existing + clean_new)
                    bt.boxes_written += 1
                bt.done += 1

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
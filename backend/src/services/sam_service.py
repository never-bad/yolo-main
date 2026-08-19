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
import os
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
    # 检出框合并（#4 重叠/紧邻框保留最紧框）：
    # 同类合并：互相紧邻/重叠的同类框并为一个（person 中心点过近会误并，调低 ioU/dist 可规避）
    "merge_iou": 0.45,               # 同类合并 IoU 阈值（两个同类框交并比 ≥ 该值视为重复）
    "merge_dist": 0.35,              # 同类合并中心距阈值（≤ 该值×对角距离 视为同一物体分裂）
    # 高度重合保留最紧框：IoU 很高 + 面积接近 + 中心错位小 → 保留最小（最紧）框
    "merge_tight_iou": 0.7,          # 最紧框判据：IoU 下限（高度重合）
    "merge_tight_center": 0.15,      # 最紧框判据：中心距上限（×对角）
    "merge_tight_area_ratio": 1.6,   # 最紧框判据：面积比上限（两框大小接近）
    "half": False,                   # 半精度，省显存
    "device": "auto",                # auto | cpu | GPU 索引字符串
    # ------------------------------------------------------------------
    # 千问 VL 大模型预标注（本地预留接口：qwen_enabled=False 时不启用，
    # 部署到服务器/接入百炼 API 后改为 True 并填写 endpoint/key 即可启用）
    # ------------------------------------------------------------------
    "qwen_enabled": False,           # 是否启用千问 VL 作为检测引擎（yolo_world | grounding_dino | qwen_vl | none）
    "qwen_backend": "ollama",        # ollama（本地大模型）| dashscope（阿里云百炼 OpenAI 兼容接口）
    "qwen_endpoint": "http://localhost:11434",  # ollama: http://localhost:11434；dashscope: https://dashscope.aliyuncs.com/compatible-mode/v1
    "qwen_model": "qwen2.5-vl-7b-instruct",      # ollama 模型名 或 百炼模型名（qwen-vl-max 等）
    "qwen_api_key": "",              # dashscope 需要 API Key；ollama 不需要
    "qwen_timeout": 90,              # 单图推理超时（秒），VLM 速度慢需给足余量
    "qwen_mock": False,              # 本地联调：True 时返回模拟框，不真实调用大模型（仅用于验证链路）
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
        self.retried = 0                       # 检空后通过降阈/换引擎成功补标上的图片数
        self.skipped_candidates: List[str] = []  # 两次兜底仍未检出的图片ID（真正需要人工复核的候选）
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
        overrides = {}
        p = self._config_path()
        if p.exists():
            try:
                saved = _load_json(p)
                cfg.update(saved)
                overrides = saved
            except Exception:
                pass
        # Docker 部署（Ollama）：backend 容器内通过服务名访问 ollama 服务；
        # 仅当用户未在 UI 显式配置过 endpoint 时注入，本地联调默认值不受影响
        env_endpoint = os.environ.get("OLLAMA_HOST", "").strip()
        if env_endpoint and "qwen_endpoint" not in overrides:
            cfg["qwen_endpoint"] = env_endpoint.rstrip("/")
        return cfg

    def _save_config(self, cfg: dict) -> None:
        p = self._config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    def get_config(self) -> dict:
        return self._read_config()

    def _task_labels_desc(self, task_id: str) -> Optional[dict]:
        """千问预标注：回溯 标注任务→数据集→归属模型 的 labels_dict.json，
        取出标签字典（index/english_code/chinese_name/chinese_desc），
        供提示词使用中文名+中文描述，提升大模型对业务类别的识别精度。
        任一层缺失均返回 None（不阻断标注，回退到纯类别名提示词）。
        """
        try:
            task_file = settings.ANNOTATIONS_DIR / task_id / "task.json"
            if not task_file.exists():
                return None
            task_meta = _load_json(task_file)
            dataset_id = task_meta.get("dataset_id")
            version = task_meta.get("version", "v1")
            if not dataset_id:
                return None
            ds_meta_file = settings.DATASETS_DIR / dataset_id / version / "meta.json"
            if not ds_meta_file.exists():
                ds_meta_file = settings.DATASETS_DIR / dataset_id / "meta.json"
            if not ds_meta_file.exists():
                return None
            ds_meta = _load_json(ds_meta_file)
            model_id = ds_meta.get("model_id")
            if not model_id:
                return None
            dict_file = settings.REGISTRY_DIR / model_id / "labels_dict.json"
            if not dict_file.exists():
                return None
            labels = (_load_json(dict_file) or {}).get("labels") or []
            return {"model_id": model_id, "labels": labels}
        except Exception as e:
            print(f"qwen: 读取标签字典失败(不影响标注): {e}")
            return None

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
        if det_type == "qwen_vl":
            # 千问 VL 为远程/本地大模型服务，无需加载本地检测权重；返回就绪标记
            return True
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

        # 千问预标注：携带模型标签字典（中文名+中文描述）进提示词
        if cfg.get("qwen_enabled", False):
            ld = self._task_labels_desc(task_id)
            if ld:
                cfg = dict(cfg)
                cfg["qwen_labels_desc"] = ld

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

        # 千问预标注：携带模型标签字典（中文名+中文描述）进提示词
        if cfg.get("qwen_enabled", False):
            ld = self._task_labels_desc(task_id)
            if ld:
                cfg = dict(cfg)
                cfg["qwen_labels_desc"] = ld

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

        det_boxes = self._merge_same_class_boxes(
            det_boxes, cfg.get("merge_iou", 0.45), cfg.get("merge_dist", 0.35),
            tight=self._merge_tight_cfg(cfg))
        det_boxes = self._merge_cross_class_boxes(det_boxes)
        if cfg.get("sam_enabled", True):
            # SAM 在原图上按映射回原图的框做分割精修
            return self._refine_with_sam(img_abs, det_boxes, cfg)
        return det_boxes

    def _merge_tight_cfg(self, cfg: dict) -> dict:
        """读取"高度重合保留最紧框"判据（配置项 merge_tight_*）"""
        return {
            "iou": float(cfg.get("merge_tight_iou", 0.7)),
            "center": float(cfg.get("merge_tight_center", 0.15)),
            "area_ratio": float(cfg.get("merge_tight_area_ratio", 1.6)),
        }

    def _run_auto_label_sync(self, img_abs: Path, classes: List[str], conf: float, cfg: dict, prompts=None) -> list:
        """同步执行：检测 → 合并被遮挡分裂的同类框 → (可选) SAM 精修 → 返回 BBox 列表"""
        det_boxes = self._detect(img_abs, classes, conf, cfg, prompts)
        # 合并被遮挡物（如电线杆）分裂成多个的同类框，避免同一物体被识别成两个
        det_boxes = self._merge_same_class_boxes(
            det_boxes, cfg.get("merge_iou", 0.45), cfg.get("merge_dist", 0.35),
            tight=self._merge_tight_cfg(cfg))
        # 跨类别高度重合合并（GD 常把同一目标误标成不同类别，导致高度重合框残留）
        det_boxes = self._merge_cross_class_boxes(det_boxes)
        if cfg.get("sam_enabled", True):
            return self._refine_with_sam(img_abs, det_boxes, cfg)
        return det_boxes

    def _merge_same_class_boxes(self, boxes: list, iou_thr: float = 0.45, dist_thr: float = 0.35,
                                tight: dict = None) -> list:
        """合并同类且高度重叠/紧邻的框，解决遮挡导致的同一物体被识别成多个框的问题。

        同一物体被遮挡物（如电线杆）隔开时，YOLO-World 常输出多个相邻/重叠的同类框。
        合并规则（关键：既要吞掉"遮挡分裂的碎框"，又不能把"两个近距离独立目标"并成一框）：
        - tight 模式（高度重合判据，默认配置 merge_tight_iou=0.7 等）：IoU 很高 + 面积接近
          （面积比 ≤ merge_tight_area_ratio）且中心错位小（≤ merge_tight_center×对角）
          → 判定为同一目标的双框，保留面积更小（更紧贴目标）的框；
        - 否则仅 IoU 高于 iou_thr 或中心距离很近才取并集合并（默认 0.45 / 0.35×对角）；
        - 都不满足（如背人的大小悬殊框、中心错开的两框）→ 视为不同目标，两个都保留。
        tight 参数示例：{"iou": 0.7, "center": 0.15, "area_ratio": 1.6}
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

        tight_iou = float((tight or {}).get("iou", 0.0))
        tight_center = float((tight or {}).get("center", 1.0))
        tight_area_ratio = float((tight or {}).get("area_ratio", 1.0))

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
                    # tight 判据：高度重合的同一目标 → 保留更紧（面积更小）的框
                    if tight_iou > 0 and _iou(base, other) >= tight_iou:
                        o_w = other["x2"] - other["x1"]
                        o_h = other["y2"] - other["y1"]
                        b_w = base["x2"] - base["x1"]
                        b_h = base["y2"] - base["y1"]
                        ratio = max(o_w * o_h, b_w * b_h) / (min(o_w * o_h, b_w * b_h) or 1.0)
                        is_tight = ratio <= tight_area_ratio and \
                            _center_dist(base, other) <= tight_center * base_diag
                        if is_tight:
                            # 保留面积更小的框（更紧贴目标），score 取更高者
                            if o_w * o_h < b_w * b_h:
                                base["x1"], base["y1"] = other["x1"], other["y1"]
                                base["x2"], base["y2"] = other["x2"], other["y2"]
                            base["score"] = max(base.get("score", 0.0), other.get("score", 0.0))
                            used[j] = True
                            continue
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
        标注读取优先每图 JSON（一图一文件），兼容历史任务集中式 annotations.json。
        """
        task_dir = settings.ANNOTATIONS_DIR / task_id
        task_file = task_dir / "task.json"
        if not task_file.exists():
            return {"ok": False, "error": "Task not found"}
        try:
            task_meta = _load_json(task_file)
        except Exception:
            return {"ok": False, "error": "读取任务元数据失败"}
        dataset_id = task_meta.get("dataset_id", "")
        version = task_meta.get("version", "v1")
        return await asyncio.to_thread(self._clean_task_annotations_sync, dataset_id, version, task_dir)

    def _clean_task_annotations_sync(self, dataset_id: str, version: str, task_dir: Path) -> dict:
        images_cleaned = 0
        boxes_removed = 0

        def _clean_one(ann: dict) -> bool:
            """合并一张图的重复框；有变化则更新并返回 True"""
            nonlocal images_cleaned, boxes_removed
            boxes = ann.get("boxes") or []
            if not boxes:
                return False
            orig_len = len(boxes)
            merged = self._merge_same_class_boxes(
                boxes, tight=self._merge_tight_cfg(self._read_config()))
            merged = self._merge_cross_class_boxes(merged)
            if len(merged) < orig_len:
                ann["boxes"] = merged
                ann["updated_at"] = datetime.now().isoformat()
                images_cleaned += 1
                boxes_removed += orig_len - len(merged)
                return True
            return False

        # 每图 JSON（新存储）
        ann_dir = settings.DATASETS_DIR / dataset_id / version / "annotations"
        if ann_dir.exists():
            for p in ann_dir.glob("*.json"):
                try:
                    ann = _load_json(p)
                except Exception:
                    continue
                if _clean_one(ann):
                    _save_json(p, ann)

        # 旧版本任务集中式标注文件（兼容清洗）
        legacy_file = task_dir / "annotations.json"
        if legacy_file.exists():
            try:
                legacy = _load_json(legacy_file)
            except Exception as e:
                return {"ok": False, "error": f"读取标注文件失败: {e}"}
            dirty = False
            for ann in legacy.values():
                if _clean_one(ann):
                    dirty = True
            if dirty:
                _save_json(legacy_file, legacy)

        return {
            "ok": True,
            "images_cleaned": images_cleaned,
            "boxes_removed": boxes_removed,
        }

    def _detect(self, img_abs, classes, conf, cfg, prompts=None) -> list:
        det = cfg.get("detector")
        if det == "grounding_dino":
            return self._detect_grounding_dino(img_abs, classes, conf, cfg, prompts)
        if det == "qwen_vl":
            return self._detect_qwen(img_abs, classes, conf, cfg, prompts)
        if det == "none":
            return []
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

    def _detect_qwen(self, img_abs, classes, conf, cfg, prompts=None) -> list:
        """千问 VL 大模型预标注（本地预留接口）。

        - qwen_enabled=False（本地默认）：不启用，返回空并提示。
        - qwen_mock=True：返回模拟框，仅用于本地验证链路（不真实调用大模型）。
        - 真实调用：把图片+类别清单发给千问 VL，要求对每个类别输出 像素坐标 bbox，
          解析 <box>x1,y1,x2,y2</box> 格式的结果映射回 classes 索引。
        """
        if not cfg.get("qwen_enabled", False):
            return []
        if cfg.get("qwen_mock", False):
            return self._mock_qwen_boxes(img_abs, classes)
        import base64, io
        from PIL import Image as PILImage
        try:
            image = PILImage.open(img_abs).convert("RGB")
            # 尽量压缩：宽高比保持，最长边 ≤1024（VLM 输入分辨率过大既慢又易超 MaxInputError）
            max_side = 1024
            w, h = image.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=88)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            print(f"qwen: 图片编码失败 {img_abs}: {e}")
            return []
        # 类别清单：若当前任务所属模型的标签字典可用，则附加中文名+中文描述，
        # 帮助大模型理解业务类别（千问输出仅按序号 class_id 对齐 classes，不依赖名称匹配）
        desc_info = cfg.get("qwen_labels_desc") or {}
        labels_desc = desc_info.get("labels") or []
        if labels_desc and len(labels_desc) >= len(classes):
            label_text = ", ".join(
                f"{i}:{lab.get('chinese_name') or classes[i]}" 
                + (f"（{lab.get('chinese_desc')}）" if lab.get("chinese_desc") else "")
                for i, lab in enumerate(labels_desc[:len(classes)])
            )
        else:
            label_text = ", ".join([f"{i}:{c}" for i, c in enumerate(classes)])
        sys_msg = (
            "你是一个精细的视觉目标检测助手。对给定的图片和类别清单，找出图中出现的每一个目标实例。\n"
            "对所有找到的实例，输出格式为每行一个：\n"
            "<box>x1,y1,x2,y2</box> <class>i</class>\n"
            "其中 x1,y1,x2,y2 是该目标在图片中的像素坐标边界框（左上角与右下角，整数即可），\n"
            "i 是类别清单中的序号（0 起）。必须逐行输出，每行只能有一个实例，不要输出任何其他文字或解释。\n"
            "如果某个类别没有目标，不要输出它的行。\n"
        )
        user_msg = f"类别清单：\n{label_text}\n请检测这张图片中的所有目标实例。"
        endpoint = (cfg.get("qwen_endpoint") or "http://localhost:11434").rstrip("/")
        api_key = cfg.get("qwen_api_key") or ""
        # 同时兼容 OpenAI 兼容接口（含默认 /v1/chat/completions）与 Ollama /api/chat
        url, payload, headers = None, None, {"Content-Type": "application/json"}
        data_url = f"data:image/jpeg;base64,{b64}"
        if "ollama" in (cfg.get("qwen_backend") or "").lower() or endpoint.startswith("http://localhost:1143"):
            url = endpoint + "/api/chat"
            payload = {
                "model": cfg.get("qwen_model", "qwen2.5-vl-7b-instruct"),
                "stream": False,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg, "images": [b64]},
                ],
            }
        else:
            url = endpoint + "/v1/chat/completions"
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            payload = {
                "model": cfg.get("qwen_model", "qwen-vl-max"),
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": user_msg},
                    ]},
                ],
            }
        import urllib.request
        import json as _json
        try:
            req = urllib.request.Request(url, data=_json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=cfg.get("qwen_timeout", 90)) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
            content = ((body.get("message") or {}).get("content")
                       or (body.get("choices") or [{}])[0].get("message", {}).get("content")
                       or "")
            return self._parse_qwen_boxes(content, classes)
        except Exception as e:
            print(f"qwen: 推理失败 {img_abs}: {e}")
            return []

    def _parse_qwen_boxes(self, content: str, classes) -> list:
        """解析千问输出：<box>x1,y1,x2,y2</box> <class>i</class> → 标准 boxes"""
        import re
        boxes = []
        for m in re.finditer(r"<box>\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*</box>", content, re.IGNORECASE):
            try:
                x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))
            except Exception:
                continue
            cm = re.search(r"<class>\s*(\d+)\s*</class>", content[m.end():])
            if cm is None:
                continue
            cls = int(cm.group(1))
            if not (0 <= cls < len(classes)):
                continue
            boxes.append({
                "class_id": cls,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "score": 0.9,  # 千问不返回置信度，固定高值表示"已确认检出"；由标注页与 SAM 精修协同
            })
        return boxes

    def _mock_qwen_boxes(self, img_abs, classes) -> list:
        """本地联调用模拟框：不调用大模型，产生每个类别一个位于图中心的框，便于验证前端链路"""
        from PIL import Image as PILImage
        try:
            w, h = PILImage.open(img_abs).size
        except Exception:
            w, h = 640, 640
        cx, cy, bw, bh = w / 2, h / 2, w * 0.5, h * 0.5
        out = []
        for i, _cls in enumerate(classes):
            ox = i * 0.08 * w
            out.append({
                "class_id": i,
                "x1": cx - bw / 2 + ox, "y1": cy - bh / 2,
                "x2": cx + bw / 2 + ox, "y2": cy + bh / 2,
                "score": 0.9,
            })
        print(f"qwen IN-MOCK: {len(classes)} boxes for {img_abs}（本地联调模式，未调用真实模型）")
        return out

    # ------------------------------------------------------------------
    # 标签建议：AI 识别图片中「已知标签之外」的新类别（辅助不懂标签的用户）
    # ------------------------------------------------------------------
    def qwen_suggest_labels(self, image_paths: list, known_codes: list, cfg: dict = None) -> tuple:
        """用千问 VL 识别图片中已知类别之外的新目标，给出标签候选（四字段）。

        返回 (candidates, message)：
        - qwen_enabled=False：返回 (None, 提示文案)，不调用真实模型（本地默认）
        - qwen_mock=True：返回预设模拟候选，便于本地验证链路
        - 真实调用：多张图聚合，按 english_code 去重并统计命中图数
        candidates: [{"english_code","chinese_name","chinese_desc","images":[路径]}]
        对已 known 类别的大小写不敏感过滤，空 english_code 丢弃。
        """
        cfg = cfg or self._read_config()
        if not cfg.get("qwen_enabled", False):
            return None, (
                "千问 VL 未启用（SAM 预标注设置 → 勾选启用千问大模型）。"
                "启用后即可自动识别图片中的新标签候选。"
            )
        known = {str(c).strip().lower() for c in (known_codes or []) if str(c).strip()}

        if cfg.get("qwen_mock", False):
            mock = [
                {"english_code": "helmet", "chinese_name": "安全帽", "chinese_desc": "人员头部佩戴的安全帽",
                 "images": list(image_paths[:2])},
                {"english_code": "traffic_light", "chinese_name": "交通信号灯", "chinese_desc": "路口红绿灯灯杆上的信号灯",
                 "images": list(image_paths[:1])},
            ]
            print("qwen suggest IN-MOCK：返回模拟新类别候选（未调用真实模型）")
            return [m for m in mock if m["english_code"].lower() not in known], ""

        import base64, io, re, json as _json
        from PIL import Image as PILImage
        import urllib.request
        sys_msg = (
            "你是一个视觉数据标注助手。给定图片与已知类别清单，找出图片中出现的、"
            "**不属于已知类别清单** 的物体类别，为每个新类别给出规范化的标签建议。\n"
            "输出严格为 JSON 数组（不要任何解释文字）：\n"
            '[{"english_code": "唯一英文标识(小写字母数字下划线)", "chinese_name": "中文名", "chinese_desc": "一句中文描述"}]'
            "\n如果没有任何已知类别之外的新目标，输出 []。"
        )
        endpoint = (cfg.get("qwen_endpoint") or "http://localhost:11434").rstrip("/")
        api_key = cfg.get("qwen_api_key") or ""
        known_text = ", ".join(known) if known else "（无）"
        user_msg = f"已知类别清单：{known_text}\n请识别这张图片中已知类别之外的新目标类别。"
        agg = {}  # code -> {meta, images}
        for img_abs in image_paths:
            try:
                image = PILImage.open(img_abs).convert("RGB")
                max_side = 1024
                w, h = image.size
                if max(w, h) > max_side:
                    scale = max_side / max(w, h)
                    image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=88)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception as e:
                print(f"qwen suggest: 图片编码失败 {img_abs}: {e}")
                continue
            data_url = f"data:image/jpeg;base64,{b64}"
            url, payload, headers = None, None, {"Content-Type": "application/json"}
            if "ollama" in (cfg.get("qwen_backend") or "").lower() or endpoint.startswith("http://localhost:1143"):
                url = endpoint + "/api/chat"
                payload = {
                    "model": cfg.get("qwen_model", "qwen2.5-vl-7b-instruct"),
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg, "images": [b64]},
                    ],
                }
            else:
                url = endpoint + "/v1/chat/completions"
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                payload = {
                    "model": cfg.get("qwen_model", "qwen-vl-max"),
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": user_msg},
                        ]},
                    ],
                }
            try:
                req = urllib.request.Request(url, data=_json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=cfg.get("qwen_timeout", 90)) as resp:
                    body = _json.loads(resp.read().decode("utf-8"))
                content = ((body.get("message") or {}).get("content")
                           or (body.get("choices") or [{}])[0].get("message", {}).get("content")
                           or "")
            except Exception as e:
                print(f"qwen suggest: 推理失败 {img_abs}: {e}")
                continue
            # 提取首个 JSON 数组
            m = re.search(r"\[[\s\S]*\]", content)
            if not m:
                continue
            try:
                items = _json.loads(m.group(0))
            except Exception:
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                code = (it.get("english_code") or "").strip().lower()
                if not code or code in known:
                    continue
                cand = agg.setdefault(code, {
                    "english_code": it.get("english_code", "").strip(),
                    "chinese_name": (it.get("chinese_name") or "").strip(),
                    "chinese_desc": (it.get("chinese_desc") or "").strip(),
                    "images": [],
                })
                if img_abs not in cand["images"]:
                    cand["images"].append(img_abs)
        return list(agg.values()), ""

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
        # 千问预标注：批量携带模型标签字典（中文名+中文描述），一次解析全批复用
        if cfg.get("qwen_enabled", False):
            ld = self._task_labels_desc(bt.task_id)
            if ld:
                cfg = dict(cfg)
                cfg["qwen_labels_desc"] = ld
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
                            # 兜底重试（缓解「AI 标不了导致人工补标压力大」）：
                            # 1) 检空 → 降置信度重试（提升召回，conf 0.25 → ~0.1）
                            # 2) 仍空 → 切换另一检测引擎重试（YOLO-World <-> GroundingDINO 双引擎）
                            if not boxes:
                                fallback_conf = max(0.05, round(bt.conf * 0.4, 3))
                                bt.retried += 1
                                try:
                                    boxes = await asyncio.to_thread(
                                        self._run_auto_label_sync, img_abs, bt.classes, fallback_conf, cfg, bt.prompts
                                    )
                                except Exception as e:
                                    print(f"批量预标注降阈重试失败 {image_id}: {e}")
                                if not boxes:
                                    alt_cfg = dict(cfg)
                                    alt_cfg["detector"] = "grounding_dino" if cfg.get("detector") != "grounding_dino" else "yolo_world"
                                    bt.skipped_candidates.append(image_id)
                                    try:
                                        boxes = await asyncio.to_thread(
                                            self._run_auto_label_sync, img_abs, bt.classes, fallback_conf, alt_cfg, bt.prompts
                                        )
                                        if boxes:
                                            bt.retried += 1
                                    except Exception as e:
                                        print(f"批量预标注换引擎重试失败 {image_id}: {e}")
                                # 兜底重试③：本地引擎均检空且千问启用时，交给 VLM 再试（缓解"标不了"）
                                if not boxes and cfg.get("qwen_enabled", False):
                                    qwen_cfg = dict(cfg)
                                    qwen_cfg["detector"] = "qwen_vl"
                                    try:
                                        boxes = await asyncio.to_thread(
                                            self._run_auto_label_sync, img_abs, bt.classes, fallback_conf, qwen_cfg, bt.prompts
                                        )
                                        if boxes:
                                            bt.retried += 1
                                    except Exception as e:
                                        print(f"批量预标注千问重试失败 {image_id}: {e}")
                    # 只保留缺失类别的检测框（class_id 为全局索引，与 bt.classes 对齐）
                    missing_set = set(missing_indices)
                    new_boxes = [b for b in boxes if b["class_id"] in missing_set]
                    if new_boxes:
                        clean_existing = [
                            {"class_id": b.get("class_id"), "x1": b.get("x1"), "y1": b.get("y1"),
                             "x2": b.get("x2"), "y2": b.get("y2"),
                             "score": b.get("score") or b.get("confidence"),
                             "source": b.get("source", "manual")}
                            for b in existing_boxes if b.get("x1") is not None
                        ]
                        clean_new = [
                            {"class_id": b["class_id"], "x1": b["x1"], "y1": b["y1"],
                             "x2": b["x2"], "y2": b["y2"],
                             "score": b.get("score") or b.get("confidence"),
                             "source": "ai"}
                            for b in new_boxes
                        ]
                        await self.annotation_service.save_annotation(bt.task_id, image_id, clean_existing + clean_new, ai_annotated=True)
                        bt.boxes_written += 1
                        bt.annotated_images.append(image_id)
                    elif not any(b.get("x1") is not None for b in existing_boxes):
                        # AI 兜底重试后仍检空且该图无任何标注：标记为困难样本（ai_miss），
                        # 供人工重点审核（空白/难例由审核者定夺），避免被静默跳过
                        try:
                            await self.annotation_service.save_annotation(
                                bt.task_id, image_id, [], ai_annotated=True,
                                sample_type=self.annotation_service.SAMPLE_HARD,
                                sample_reason="ai_miss")
                        except Exception as e:
                            print(f"批量预标注标记困难样本失败 {image_id}: {e}")
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
            "retried": bt.retried,
            "skipped_candidates": bt.skipped_candidates,
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
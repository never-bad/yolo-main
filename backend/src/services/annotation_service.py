import json
import yaml
import asyncio
import shutil
import zipfile
import re
from pathlib import Path
from datetime import datetime
from PIL import Image
from src.core.settings import settings


def _load_json(path: Path):
    """同步加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict):
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
    # datasets/{dataset_id}/{version}/annotations/{image_id}.json
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
    #       "confidence": 0.86, "source": "ai_qwen"             # 可选：AI 置信度/来源
    #     }
    #   ]
    # }
    # 读取时兼容旧任务集中式 annotations.json（任务目录下），新写入一律每图一文件
    # ------------------------------------------------------------------
    def _image_ann_path(self, dataset_id: str, version: str, image_id: str) -> Path:
        return self.datasets_dir / dataset_id / version / "annotations" / f"{image_id}.json"

    def _load_image_ann(self, dataset_id: str, version: str, image_id: str) -> dict:
        p = self._image_ann_path(dataset_id, version, image_id)
        try:
            return _load_json(p)
        except Exception:
            return {}

    def _save_image_ann(self, dataset_id: str, version: str, image_id: str, data: dict):
        p = self._image_ann_path(dataset_id, version, image_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        _save_json(p, data)

    def _delete_image_ann(self, dataset_id: str, version: str, image_id: str):
        p = self._image_ann_path(dataset_id, version, image_id)
        try:
            p.unlink()
        except OSError:
            pass

    def _load_all_image_anns(self, dataset_id: str, version: str) -> dict:
        """读取数据集下所有每图 JSON 标注（缺失/损坏按空处理）"""
        ann_dir = self.datasets_dir / dataset_id / version / "annotations"
        out = {}
        if not ann_dir.exists():
            return out
        for p in ann_dir.glob("*.json"):
            try:
                data = _load_json(p)
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
            dic = _load_json(settings.REGISTRY_DIR / model_id / "labels_dict.json")
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
                "x1": float(b["x1"]), "y1": float(b["y1"]),
                "x2": float(b["x2"]), "y2": float(b["y2"]),
            }
            confidence = b.get("confidence")
            if confidence is None:
                confidence = b.get("score")
            if confidence is not None:
                nb["confidence"] = float(confidence)
            source = b.get("source")
            if source:
                nb["source"] = source
            out.append(nb)
        return out

    def _boxes_for_view(self, boxes: list, classes: list) -> list:
        """响应层兜底：新数据 class_id 已是数字索引，直接透传；
        对历史英文名 JSON 兜底映射回 classes 下标，找不到保留原值（不崩）。
        """
        idx_map = {str(c).lower(): i for i, c in enumerate(classes or [])}
        out = []
        for b in boxes:
            nb = dict(b)
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
        
        # 获取数据集图片
        dataset_dir = self.datasets_dir / dataset_id / version
        
        # 检查数据集目录是否存在
        if not await asyncio.to_thread(lambda: dataset_dir.exists()):
            raise ValueError(f"数据集 {dataset_id}/{version} 不存在，请先上传并准备数据集")
        
        # 在线程中执行所有同步操作
        def _create_task_sync():
            images_dir = self._find_images_dir(dataset_dir)
            labels_dir = self._find_labels_dir(dataset_dir)
            
            if not images_dir:
                raise ValueError(f"数据集 {dataset_id}/{version} 中未找到图片目录，请确保已执行 prepare 操作")

            # 数据集归属模型 + 类别（记录 model_id；类别权威来源 = 数据集 meta.json 的 classes，prepare 时已写入）
            ds_meta_path = self.datasets_dir / dataset_id / "meta.json"
            model_id, auto_classes = "", []
            try:
                if ds_meta_path.exists():
                    ds_meta = _load_json(ds_meta_path)
                    model_id = ds_meta.get("model_id", "")
                    auto_classes = ds_meta.get("classes") or []
            except Exception:
                pass
            
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
                            # 加载 YOLO 格式标签 → 以像素坐标写入每图 JSON
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
            
            _save_json(task_dir / "task.json", task_meta)
            
            return {
                "task_id": task_id,
                "total_images": len(items),
                "imported_annotations": loaded_count,
                "classes": task_classes
            }
        
        return await asyncio.to_thread(_create_task_sync)
    
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
                meta = _load_json(task_file)
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
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # YOLO 格式: class_id cx cy w h (归一化坐标)
                        cx = float(parts[1])
                        cy = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        
                        # 转换为绝对坐标 (x1, y1, x2, y2)
                        x1 = (cx - w / 2) * img_width
                        y1 = (cy - h / 2) * img_height
                        x2 = (cx + w / 2) * img_width
                        y2 = (cy + h / 2) * img_height
                        
                        boxes.append({
                            "class_id": class_id,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        })
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
            task_meta = _load_json(task_file)
            dataset_id, version = self._task_to_ds(task_meta)
            
            # 读取标注状态（优先每图 JSON，兼容历史任务集中式 annotations.json）
            # 文件可能被批量预标注并发写入，缺失/损坏时按空处理，避免整个列表加载失败
            annotations = self._load_all_image_anns(dataset_id, version)
            legacy_file = task_dir / "annotations.json"
            try:
                if legacy_file.exists():
                    for img_id, ann in _load_json(legacy_file).items():
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
                unique_items.append(item)
            
            return {
                "items": unique_items,
                "classes": task_meta.get("classes", []),
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
            task_meta = _load_json(task_file)
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
                        annotation = _load_json(legacy_file).get(image_id, {})
                except Exception:
                    annotation = {}
            boxes = annotation.get("boxes", [])
            # JSON 落盘 class_id 为英文名，响应层映射回任务类别下标（前端渲染用）
            boxes = self._boxes_for_view(boxes, task_meta.get("classes") or [])
            
            return {
                "image_id": image_id,
                "image_path": image_info["image_path"],
                "width": image_info["width"],
                "height": image_info["height"],
                "boxes": boxes,
                "sample_type": annotation.get("sample_type", ""),
                "sample_reasons": annotation.get("sample_reasons", []),
                "classes": task_meta.get("classes", [])
            }
        
        return await asyncio.to_thread(_get_image_annotation_sync)
    
    @staticmethod
    def _box_iou(a: dict, b: dict) -> float:
        """两个框的 IoU（框需含 x1/y1/x2/y2，缺字段按 0 兜底）。"""
        ax1, ay1 = float(a.get("x1", 0)), float(a.get("y1", 0))
        ax2, ay2 = float(a.get("x2", 0)), float(a.get("y2", 0))
        bx1, by1 = float(b.get("x1", 0)), float(b.get("y1", 0))
        bx2, by2 = float(b.get("x2", 0)), float(b.get("y2", 0))
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0.0

    async def save_annotation(self, task_id: str, image_id: str, boxes: list, ai_annotated: bool = False,
                              sample_type: str = None, sample_reason: str = None):
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
            task_meta = _load_json(task_file)
            dataset_id, version = self._task_to_ds(task_meta)
            
            # 读取该图已有标注（保留 AI 标注标记：人工保存不应清掉标记）
            existing = self._load_image_ann(dataset_id, version, image_id)
            prev_ai = existing.get("ai_annotated", False)
            created_at = existing.get("created_at") or datetime.now().isoformat()
            
            # 查找图片信息（首次写入时补充元数据）
            info = next((it for it in task_meta.get("items", []) if it.get("image_id") == image_id), {})
            
            # 规范化框（补全标签四字段 + 置信度/来源），并判定样本类型
            classes = task_meta.get("classes", []) or []
            class_meta = self._class_meta_map(task_meta.get("model_id", ""))
            norm_boxes = self._normalize_boxes(boxes, classes, class_meta)
            ai = ai_annotated or prev_ai
            s_type, s_reasons = self._resolve_sample_type(norm_boxes, ai, sample_type, sample_reason)

            # 人工补框 → 自动困难：AI 已标过的图上，人工新增了与既有框不重叠的新框，
            # 视为人工发现 AI 漏检，强制标为困难查证（用户显式指定类型时以用户为准）
            existing_boxes = existing.get("boxes", []) if isinstance(existing, dict) else []
            if ai and sample_type is None and s_type != self.SAMPLE_BACKGROUND:
                manual_added = [
                    b for b in boxes
                    if str(b.get("source", "")).lower().startswith("manual")
                    and not any(self._box_iou(b, pb) >= 0.7 for pb in existing_boxes)
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
            }
            
            self._save_image_ann(dataset_id, version, image_id, ann_data)
            return {"ok": True, "sample_type": s_type, "sample_reasons": s_reasons}
        
        return await asyncio.to_thread(_save_annotation_sync)

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
            task_meta = _load_json(task_file)
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
                    legacy = _load_json(legacy_file)
                    before = len(legacy)
                    for image_id in list(legacy.keys()):
                        if legacy[image_id].get("ai_annotated"):
                            del legacy[image_id]
                    if len(legacy) != before:
                        _save_json(legacy_file, legacy)
                        removed += before - len(legacy)
            except Exception:
                pass
            return {"ok": True, "removed": removed}

        return await asyncio.to_thread(_clear_sync)

    async def export_to_yolo(self, task_id: str):
        """导出标注为YOLO格式"""
        def _export_to_yolo_sync():
            try:
                task_dir = self.annotations_dir / task_id
                task_file = task_dir / "task.json"
                
                if not task_file.exists():
                    return {"ok": False, "error": "Task not found"}
                
                # 读取任务信息
                task_meta = _load_json(task_file)
                dataset_id = task_meta["dataset_id"]
                version = task_meta["version"]
                # 标注读取：优先每图 JSON，兼容历史任务集中式 annotations.json
                annotations = self._load_all_image_anns(dataset_id, version)
                legacy_file = task_dir / "annotations.json"
                try:
                    if legacy_file.exists():
                        for img_id, ann in _load_json(legacy_file).items():
                            annotations.setdefault(img_id, ann)
                except Exception:
                    pass
                
                # 获取数据集labels目录
                dataset_id = task_meta["dataset_id"]
                version = task_meta["version"]
                dataset_dir = self.datasets_dir / dataset_id / version
                
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
                
                exported_count = 0
                errors = []
                # class_id（英文名）→ YOLO 数字索引（与任务类别列表顺序一致）
                _cls_idx = {str(c).lower(): i for i, c in enumerate(task_meta.get("classes") or [])}
                
                # 转换每个标注
                for item in task_meta["items"]:
                    try:
                        image_id = item["image_id"]
                        if image_id not in annotations:
                            continue
                        
                        width = item["width"]
                        height = item["height"]
                        boxes = annotations[image_id]["boxes"]
                        
                        if not boxes:
                            continue
                        
                        # 转换为YOLO格式（归一化的 cx cy w h）
                        yolo_lines = []
                        for box in boxes:
                            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                            cx = (x1 + x2) / 2 / width
                            cy = (y1 + y2) / 2 / height
                            w = (x2 - x1) / width
                            h = (y2 - y1) / height
                            class_id = box["class_id"]
                            if isinstance(class_id, int) or (isinstance(class_id, str) and class_id.isdigit()):
                                cls_id = int(class_id)  # 旧数据数字索引直接使用
                            else:
                                cls_id = _cls_idx.get(str(class_id).lower(), 0)
                            
                            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        
                        # 确定标签文件路径
                        label_path = self._determine_label_path(item, images_dir, labels_dir, image_id)
                        
                        label_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 写入标签文件
                        _write_text(label_path, "\n".join(yolo_lines))
                        
                        exported_count += 1
                    except Exception as e:
                        error_msg = f"Error exporting annotation for image {item.get('image_id', 'unknown')}: {str(e)}"
                        print(error_msg)
                        errors.append(error_msg)
                        continue
                
                result = {"ok": True, "exported_count": exported_count}
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
            # image_path 格式：datasets/ds_xxx/v1/images/xxx.jpg 或 datasets/ds_xxx/v1/images/train/xxx.jpg
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
            task_meta = _load_json(task_file)
            items = task_meta["items"]
            classes = task_meta.get("classes", [])
            dataset_id, version = self._task_to_ds(task_meta)

            # 解压 zip 到临时目录
            tmp_dir = task_dir / "import_tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)

            # 优先取 labels 目录下的 txt，其次临时目录内所有 txt
            labels_dir = tmp_dir / "labels"
            label_files = list(labels_dir.rglob("*.txt")) if labels_dir.exists() else list(tmp_dir.rglob("*.txt"))

            # 建立文件名 stem → item 映射（image_id 与图片文件名一般一致）
            by_id = {item["image_id"]: item for item in items}
            by_stem = {}
            for item in items:
                by_stem.setdefault(Path(item["image_path"]).stem, item)

            imported = 0
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

            shutil.rmtree(tmp_dir, ignore_errors=True)
            return {"ok": True, "imported": imported, "skipped": skipped}

        return await asyncio.to_thread(_import_sync)

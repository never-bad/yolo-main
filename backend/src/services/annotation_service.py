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
    def __init__(self):
        self.annotations_dir = settings.ANNOTATIONS_DIR
        self.datasets_dir = settings.DATASETS_DIR
    
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
            
            # 类别合并规则：数据集类别文件中的类别在前（真实类别名，class_id 顺序与已有标注对齐），
            # 用户输入中的新增类别追加在后；两者合并去重。这样数据集已有类别时可在其基础上扩展，
            # 数据集读不到类别时用户输入即为全部类别（避免标注任务无类别导致 AI 预标注不可用）
            auto_classes = self._load_classes_from_yaml(dataset_dir) or []
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
            annotations = {}  # 用于存储已有标注
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
                    
                    # 检查是否有对应的标签文件
                    has_annotation = False
                    if labels_dir:
                        # 尝试找到对应的标签文件
                        label_path = self._find_label_file(img_path, images_dir, labels_dir)
                        if label_path and label_path.exists():
                            # 加载 YOLO 格式标签
                            boxes = self._load_yolo_labels(label_path, width, height)
                            if boxes:
                                annotations[image_id] = {
                                    "boxes": boxes,
                                    "updated_at": datetime.now().isoformat(),
                                    "source": "imported"  # 标记为导入的标注
                                }
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
                "created_at": datetime.now().isoformat(),
                "items": items,
                "imported_annotations": loaded_count  # 记录导入的标注数量
            }
            
            _save_json(task_dir / "task.json", task_meta)
            
            # 保存标注数据（包括导入的已有标注）
            _save_json(task_dir / "annotations.json", annotations)
            
            return {
                "task_id": task_id,
                "total_images": len(items),
                "imported_annotations": loaded_count,
                "classes": task_classes
            }
        
        return await asyncio.to_thread(_create_task_sync)
    
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
    
    def _load_classes_from_yaml(self, dataset_dir: Path) -> list:
        """从类别文件加载类别列表，支持 yaml / txt / names 多种格式

        逻辑：按后缀名判断解析方式
          - .yaml / .yml：读取 names 字段
          - .txt / .names：按行读取

        优先级与数据集检测一致：真实类别文件（classes.txt 等）优先于 data.yaml，
        并跳过 data.yaml 中历史遗留的占位名（class_0 等）。
        """
        def is_placeholder(names):
            return bool(names) and all(re.fullmatch(r"class_\d+", str(c)) for c in names)

        candidates = [
            dataset_dir / "classes.txt",
            dataset_dir / "coco.names",
            dataset_dir / "names.txt",
            dataset_dir / "dataset.yaml",
            dataset_dir / "config.yaml",
            dataset_dir / "data.yaml",
        ]
        # 1) 优先查找根目录（跳过占位名）
        for path in candidates:
            if path.exists():
                try:
                    classes = self._parse_classes_file(path)
                    if classes and not is_placeholder(classes):
                        return classes
                except Exception as e:
                    print(f"Error loading classes from {path}: {e}")
        # 2) 递归查找（类别文件可能在子目录中）
        for name in ["classes.txt", "coco.names", "names.txt", "dataset.yaml", "config.yaml", "data.yaml"]:
            for path in dataset_dir.rglob(name):
                try:
                    classes = self._parse_classes_file(path)
                    if classes and not is_placeholder(classes):
                        return classes
                except Exception:
                    continue
        # 3) 兜底：都没有真实名时，返回第一个非空结果（可能是占位名）
        for path in candidates:
            if path.exists():
                try:
                    classes = self._parse_classes_file(path)
                    if classes:
                        return classes
                except Exception:
                    continue
        return []

    def _parse_classes_file(self, path: Path) -> list:
        """按后缀名解析类别文件"""
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            data = _load_yaml(path)
            names = data.get('names', [])
            # YOLO 也支持 names: {0: 'a', 1: 'b'} 的字典形式，转为按 id 排序的列表
            if isinstance(names, dict):
                return [str(names[k]) for k in sorted(names)]
            return names
        # 纯文本（.txt / .names）：按行读取，忽略空行
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    
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
            
            # 读取标注状态（文件可能被批量预标注并发写入，缺失/损坏时按空处理，避免整个列表加载失败）
            annotations_file = task_dir / "annotations.json"
            try:
                annotations = _load_json(annotations_file)
            except Exception:
                annotations = {}
            
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
                unique_items.append(item)
            
            return {
                "items": unique_items,
                "classes": task_meta.get("classes", []),
                "imported_annotations": task_meta.get("imported_annotations", 0)
            }
        
        return await asyncio.to_thread(_get_task_items_sync)
    
    async def get_image_annotation(self, task_id: str, image_id: str):
        """获取单张图片的标注"""
        task_dir = self.annotations_dir / task_id
        task_file = task_dir / "task.json"
        annotations_file = task_dir / "annotations.json"
        
        def _check_files():
            return task_file.exists() and annotations_file.exists()
        
        if not await asyncio.to_thread(_check_files):
            return None
        
        def _get_image_annotation_sync():
            task_meta = _load_json(task_file)
            annotations = _load_json(annotations_file)
            
            # 查找图片信息
            image_info = None
            for item in task_meta["items"]:
                if item["image_id"] == image_id:
                    image_info = item
                    break
            
            if not image_info:
                return None
            
            # 获取标注
            annotation = annotations.get(image_id, {})
            boxes = annotation.get("boxes", [])
            
            return {
                "image_id": image_id,
                "image_path": image_info["image_path"],
                "width": image_info["width"],
                "height": image_info["height"],
                "boxes": boxes,
                "classes": task_meta.get("classes", [])
            }
        
        return await asyncio.to_thread(_get_image_annotation_sync)
    
    async def save_annotation(self, task_id: str, image_id: str, boxes: list, ai_annotated: bool = False):
        """保存图片标注"""
        task_dir = self.annotations_dir / task_id
        annotations_file = task_dir / "annotations.json"
        
        if not await asyncio.to_thread(lambda: annotations_file.exists()):
            return {"ok": False, "error": "Task not found"}
        
        def _save_annotation_sync():
            # 读取现有标注
            annotations = _load_json(annotations_file)
            
            # 保留该图已有的 AI 标注标记（人工保存不应清掉标记）
            existing = annotations.get(image_id, {})
            prev_ai = existing.get("ai_annotated", False)
            
            # 保存新标注
            annotations[image_id] = {
                "boxes": [box.dict() if hasattr(box, 'dict') else box for box in boxes],
                "updated_at": datetime.now().isoformat(),
                "ai_annotated": ai_annotated or prev_ai
            }
            
            _save_json(annotations_file, annotations)
            return {"ok": True}
        
        return await asyncio.to_thread(_save_annotation_sync)

    async def clear_ai_annotations(self, task_id: str):
        """清除该任务所有 AI 预标注（批量误标过多时一键清理，便于重新标注）。

        删除所有被 AI 预标注过的图片的标注（含其后手工修改，不可恢复）。
        返回被清理的图片数 removed。
        """
        annotations_file = self.annotations_dir / task_id / "annotations.json"
        if not await asyncio.to_thread(lambda: annotations_file.exists()):
            return {"ok": False, "error": "Task not found"}

        def _clear_sync():
            annotations = _load_json(annotations_file)
            removed = 0
            for image_id, ann in list(annotations.items()):
                if ann.get("ai_annotated"):
                    del annotations[image_id]
                    removed += 1
            _save_json(annotations_file, annotations)
            return {"ok": True, "removed": removed}

        return await asyncio.to_thread(_clear_sync)

    async def export_to_yolo(self, task_id: str):
        """导出标注为YOLO格式"""
        def _export_to_yolo_sync():
            try:
                task_dir = self.annotations_dir / task_id
                task_file = task_dir / "task.json"
                annotations_file = task_dir / "annotations.json"
                
                if not task_file.exists() or not annotations_file.exists():
                    return {"ok": False, "error": "Task not found"}
                
                # 读取任务信息
                task_meta = _load_json(task_file)
                annotations = _load_json(annotations_file)
                
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
                            
                            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        
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

    async def export_to_yolo_split(self, task_id: str, train_ratio: float, val_ratio: float, test_ratio: float):
        """导出标注为YOLO格式，并按比例自动划分训练/验证/测试集。

        会将已标注图片复制到 images/{train,val,test}，标签写入 labels/{train,val,test}，
        并重写 data.yaml 指向划分后的目录。
        """
        def _export_split_sync():
            import random
            import shutil
            total_ratio = train_ratio + val_ratio + test_ratio
            if total_ratio <= 0:
                return {"ok": False, "error": "比例之和必须大于0"}
            tr = train_ratio / total_ratio
            vr = val_ratio / total_ratio
            te = test_ratio / total_ratio

            task_dir = self.annotations_dir / task_id
            task_file = task_dir / "task.json"
            annotations_file = task_dir / "annotations.json"
            if not task_file.exists() or not annotations_file.exists():
                return {"ok": False, "error": "Task not found"}

            task_meta = _load_json(task_file)
            annotations = _load_json(annotations_file)

            dataset_dir = self.datasets_dir / task_meta["dataset_id"] / task_meta["version"]
            if not dataset_dir.exists():
                return {"ok": False, "error": f"Dataset directory not found: {dataset_dir}"}

            # 收集所有已标注图片
            annotated = []
            for item in task_meta["items"]:
                image_id = item.get("image_id")
                if image_id not in annotations:
                    continue
                boxes = annotations[image_id].get("boxes") or []
                if not boxes:
                    continue
                img_abs = settings.DATA_DIR / item["image_path"]
                if not img_abs.exists():
                    continue
                annotated.append({
                    "item": item,
                    "img_abs": img_abs,
                    "boxes": boxes,
                    "width": item.get("width", 1),
                    "height": item.get("height", 1),
                })

            if not annotated:
                return {"ok": False, "error": "没有已标注的图片可导出"}

            # 确定性打乱，保证可复现
            random.Random(42).shuffle(annotated)

            # 按比例分配
            n = len(annotated)
            n_train = int(round(n * tr))
            n_val = int(round(n * vr))
            if n_train + n_val > n:
                n_train = n - n_val
            n_test = n - n_train - n_val

            splits = {"train": [], "val": [], "test": []}
            for i, rec in enumerate(annotated):
                if i < n_train:
                    splits["train"].append(rec)
                elif i < n_train + n_val:
                    splits["val"].append(rec)
                else:
                    splits["test"].append(rec)

            # 创建目录
            images_dir = dataset_dir / "images"
            labels_dir = dataset_dir / "labels"
            for split in ["train", "val", "test"]:
                (images_dir / split).mkdir(parents=True, exist_ok=True)
                (labels_dir / split).mkdir(parents=True, exist_ok=True)

            # 导出图片与标签
            counts = {"train": 0, "val": 0, "test": 0}
            errors = []
            for split, recs in splits.items():
                for rec in recs:
                    item = rec["item"]
                    image_id = item["image_id"]
                    try:
                        ext = rec["img_abs"].suffix or ".jpg"
                        dst_img = images_dir / split / f"{image_id}{ext}"
                        shutil.copy2(rec["img_abs"], dst_img)

                        w = rec["width"]
                        h = rec["height"]
                        yolo_lines = []
                        for box in rec["boxes"]:
                            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                            cx = (x1 + x2) / 2 / w
                            cy = (y1 + y2) / 2 / h
                            bw = (x2 - x1) / w
                            bh = (y2 - y1) / h
                            yolo_lines.append(f"{box['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        _write_text(labels_dir / split / f"{image_id}.txt", "\n".join(yolo_lines))
                        counts[split] += 1
                    except Exception as e:
                        errors.append(f"{image_id}: {e}")
                        continue

            # 重写 data.yaml
            classes = task_meta.get("classes") or []
            _save_yaml(dataset_dir / "data.yaml", {
                "path": str(dataset_dir.absolute()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": len(classes),
                "names": classes,
            })

            result = {
                "ok": True,
                "total": n,
                "counts": counts,
                "ratios": {"train": round(tr, 3), "val": round(vr, 3), "test": round(te, 3)},
            }
            if errors:
                result["errors"] = errors
            return result

        return await asyncio.to_thread(_export_split_sync)

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

        按 .txt 文件名（stem）与任务图片匹配，转成像素坐标写入 annotations.json。
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
            annotations_file = task_dir / "annotations.json"
            annotations = _load_json(annotations_file) if annotations_file.exists() else {}

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
                annotations[item["image_id"]] = {
                    "boxes": valid,
                    "updated_at": datetime.now().isoformat(),
                    "ai_annotated": False,
                }
                imported += 1

            shutil.rmtree(tmp_dir, ignore_errors=True)
            _save_json(annotations_file, annotations)
            return {"ok": True, "imported": imported, "skipped": skipped}

        return await asyncio.to_thread(_import_sync)

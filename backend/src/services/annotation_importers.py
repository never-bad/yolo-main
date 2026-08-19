"""外部标注格式导入器：COCO / LabelMe / VOC → 统一一图一 JSON 标注。

数据链路改造（阶段B）：prepare 阶段检测到外部格式后调用，产物：

    datasets/{dataset_id}/v1/images/{image_id}.jpg        # 图片平铺（封板前不划分）
    datasets/{dataset_id}/v1/annotations/{image_id}.json  # 一图一 JSON 标注
    datasets/{dataset_id}/v1/classes.txt                  # 类别（英文码，每行一个）

每图 JSON 结构与 annotation_service._image_ann_path 约定完全一致：
    {image_id, image_path, width, height, model_id, dataset_id, version,
     boxes:[{class_id,x1,y1,x2,y2}], ai_annotated, updated_at}

image_path 为相对 DATA_DIR 的路径（静态文件服务根目录），标注前端可直接取用。
"""
import json
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from PIL import Image

from src.core.settings import settings

# 支持的图片后缀（与 dataset_service 保持一致）
IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 排除可视化预览图（如 *_result.jpg）
EXCLUDE_KEYWORDS = ("_result", "_predict", "_detect")


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _iter_images(root: Path):
    """遍历 root 下的图片文件（跳过隐藏目录 / 预览图 / 标注输出目录）"""
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        low_parts = [x.lower() for x in p.relative_to(root).parts]
        if any(x.startswith(".") for x in low_parts):
            continue
        if p.suffix.lower() not in IMG_SUFFIX:
            continue
        if "annotations" in low_parts:
            continue
        if any(k in p.name.lower() for k in EXCLUDE_KEYWORDS):
            continue
        yield p


def _image_dims(img_abs: Path):
    """读取图片宽高"""
    with Image.open(img_abs) as im:
        return im.size  # (width, height)


def _rel_to_data_dir(img_abs: Path) -> str:
    """计算图片相对 DATA_DIR 的路径（正斜杠），供静态文件服务使用"""
    try:
        return str(img_abs.resolve().relative_to(settings.DATA_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(img_abs).replace("\\", "/")


def _flatten_images(root: Path, images_dir: Path) -> dict:
    """把所有图片平铺移动到 images/，返回 {stem: 最终图片路径}

    不同目录/分区下的图片合并到同一目录，撞名时先到先得（与标注任务 image_id=stem 规则一致）。
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for img in sorted(_iter_images(root), key=lambda p: str(p)):
        if img.parent == images_dir:
            out.setdefault(img.stem, img)
            continue
        dst = images_dir / f"{img.stem}{img.suffix.lower()}"
        if not dst.exists():
            shutil.move(str(img), str(dst))
        out.setdefault(dst.stem, dst)
    return out


# ----------------------------------------------------------------------
# 支持格式检测
# ----------------------------------------------------------------------
def detect_format(root: Path) -> str:
    """检测解压后数据集的标注格式：'coco' / 'labelme' / 'voc' / ''（不识别）"""
    # COCO：annotations/ 目录或文件名含 coco 的 json，结构含 images+annotations
    for p in root.rglob("*.json"):
        low = [x.lower() for x in p.relative_to(root).parts]
        if "annotations" in low or "coco" in p.name.lower():
            try:
                d = _load_json(p)
            except Exception:
                continue
            if isinstance(d, dict) and "images" in d and "annotations" in d:
                return "coco"
    # VOC：Annotations/*.xml，根节点为 annotation 且含 filename
    for p in root.rglob("*.xml"):
        low = [x.lower() for x in p.relative_to(root).parts]
        if "annotations" in low or "voc" in p.name.lower():
            try:
                r = ET.parse(p).getroot()
            except Exception:
                continue
            if r.tag.lower() == "annotation" and r.find("filename") is not None:
                return "voc"
    # LabelMe：单文件 json，含 imagePath + shapes
    for p in root.rglob("*.json"):
        low = [x.lower() for x in p.relative_to(root).parts]
        if "annotations" in low:
            continue
        try:
            d = _load_json(p)
        except Exception:
            continue
        if isinstance(d, dict) and "shapes" in d and "imagePath" in d:
            return "labelme"
    return ""


# ----------------------------------------------------------------------
# 公共写盘
# ----------------------------------------------------------------------
def _write_ann(root: Path, image_id: str, image_abs: Path, width: int, height: int,
               boxes: list, classes_known: bool = True) -> bool:
    """写一张图的 JSON；无框且无已知类别时不写（保持“有 JSON=有标注”语义）

    返回是否写入。boxes 已在调用方转换为 {class_id,x1,y1,x2,y2} 像素坐标。
    """
    if not boxes and not classes_known:
        return False
    data = {
        "image_id": image_id,
        "image_path": _rel_to_data_dir(image_abs),
        "width": width,
        "height": height,
        "boxes": boxes,
        "updated_at": datetime.now().isoformat(),
        "ai_annotated": False,
    }
    # dataset_id / version / model_id 由 prepare 阶段回填
    _save_json(root / "annotations" / f"{image_id}.json", data)
    return True


# ----------------------------------------------------------------------
# COCO 导入
# ----------------------------------------------------------------------
def _import_coco(root: Path) -> dict:
    """COCO 格式 → 一图一 JSON。返场统计：{images, annotated, classes, errors}"""
    errors = []
    images_by_name = {}
    for img in _iter_images(root):
        images_by_name.setdefault(img.name.lower(), img)
        images_by_name.setdefault(img.stem.lower(), img)

    # 解析 COCO json（可能多个：instances_train / instances_val）
    classes = []          # 类别英文码列表（顺序=class_id）
    cat_id_to_idx = {}    # COCO category_id → 平台 class_id
    records = {}          # {image_id: {...}}
    for p in sorted(root.rglob("*.json")):
        low = [x.lower() for x in p.relative_to(root).parts]
        if "annotations" in low:
            continue
        try:
            d = _load_json(p)
        except Exception:
            continue
        if not (isinstance(d, dict) and "images" in d and "annotations" in d):
            continue
        # 类别表（多文件时合并，先到先得保序）
        for cat in d.get("categories", []):
            cname = str(cat.get("name", "")).strip()
            if cname and cname not in classes:
                classes.append(cname)
            if "id" in cat and cname in classes:
                cat_id_to_idx.setdefault(int(cat["id"]), classes.index(cname))
        # 图片清单
        json_images = d.get("images", [])
        for info in json_images:
            img_id = str(info.get("id", ""))
            fname = str(info.get("file_name", ""))
            images_by_name.setdefault(fname.lower(), None)  # 占位，避免误命中
            records.setdefault(img_id, {
                "id": img_id,
                "file_name": fname,
                "width": int(info.get("width", 0)),
                "height": int(info.get("height", 0)),
                "boxes": [],
            })
        # 标注
        for ann in d.get("annotations", []):
            rec = records.get(str(ann.get("image_id", "")))
            if rec is None:
                continue
            if "bbox" not in ann or "category_id" not in ann:
                continue
            x, y, w, h = [float(v) for v in ann["bbox"]]
            rec["boxes"].append({
                "class_id": cat_id_to_idx.get(int(ann["category_id"]), 0),
                "x1": x, "y1": y, "x2": x + w, "y2": y + h,
            })

    # 写盘 & 统计
    annotated = 0
    matched_images = 0
    for rec in records.values():
        if not rec["boxes"]:
            continue
        # 定位图片：优先 file_name 完整相对路径，其次按 basename/stem 搜索
        img_abs = None
        if rec["file_name"]:
            fname = rec["file_name"].replace("\\", "/").lower()
            for key, cand in images_by_name.items():
                if key == fname:
                    img_abs = cand
                    break
            if img_abs is None:
                # file_name 可能是子目录相对路径：按 basename 匹配
                base = fname.rsplit("/", 1)[-1]
                img_abs = images_by_name.get(base) or images_by_name.get(Path(base).stem.lower())
        if img_abs is None or not img_abs.exists():
            errors.append(f"COCO 找不到图片: {rec['file_name']}")
            continue
        matched_images += 1
        _write_ann(root, img_abs.stem, img_abs, rec["width"], rec["height"], rec["boxes"])
        annotated += 1

    return {"images": matched_images, "annotated": annotated, "classes": classes, "errors": errors}


# ----------------------------------------------------------------------
# LabelMe 导入
# ----------------------------------------------------------------------
def _import_labelme(root: Path) -> dict:
    errors = []
    classes = []
    images_by_name = {}
    for img in _iter_images(root):
        images_by_name.setdefault(img.name.lower(), img)

    annotated = 0
    matched_images = 0
    for p in sorted(root.rglob("*.json")):
        low = [x.lower() for x in p.relative_to(root).parts]
        if "annotations" in low:
            continue
        try:
            d = _load_json(p)
        except Exception:
            continue
        if not (isinstance(d, dict) and "shapes" in d and "imagePath" in d):
            continue

        # 解析图片路径：相对 json 所在目录
        img_ref = d.get("imagePath", "")
        img_abs = None
        if img_ref:
            cand = p.parent / img_ref
            if cand.exists():
                img_abs = cand
            else:
                img_abs = images_by_name.get(Path(img_ref).name.lower())
        if img_abs is None:
            errors.append(f"LabelMe 找不到图片: {img_ref or p.name}")
            continue

        w = int(d.get("imageWidth", 0)) or int(d.get("width", 0))
        h = int(d.get("imageHeight", 0)) or int(d.get("height", 0))
        if not w or not h:
            try:
                w, h = _image_dims(img_abs)
            except Exception:
                w, h = 0, 0

        boxes = []
        for shp in d.get("shapes", []):
            label = str(shp.get("label", "")).strip()
            if not label:
                continue
            if label not in classes:
                classes.append(label)
            pts = shp.get("points") or []
            if len(pts) < 2:
                continue
            xs = [pt[0] for pt in pts]
            ys = [pt[1] for pt in pts]
            boxes.append({
                "class_id": classes.index(label),
                "x1": min(xs), "y1": min(ys),
                "x2": max(xs), "y2": max(ys),
            })

        matched_images += 1
        if _write_ann(root, img_abs.stem, img_abs, w, h, boxes, classes_known=bool(classes)):
            annotated += 1
    return {"images": matched_images, "annotated": annotated, "classes": classes, "errors": errors}


# ----------------------------------------------------------------------
# VOC 导入
# ----------------------------------------------------------------------
def _import_voc(root: Path) -> dict:
    errors = []
    classes = []
    images_by_name = {}
    for img in _iter_images(root):
        images_by_name.setdefault(img.name.lower(), img)
        images_by_name.setdefault(img.stem.lower(), img)

    annotated = 0
    matched_images = 0
    for p in sorted(root.rglob("*.xml")):
        low = [x.lower() for x in p.relative_to(root).parts]
        if "annotations" not in low:
            # VOC 的 xml 应位于 Annotations 目录；宽容匹配文件名相同即可
            pass
        try:
            r = ET.parse(p).getroot()
        except Exception:
            continue
        if r.tag.lower() != "annotation":
            continue

        fname_node = r.find("filename")
        fname = fname_node.text.strip() if fname_node is not None and fname_node.text else p.stem
        img_abs = images_by_name.get(fname.lower()) or images_by_name.get(p.stem.lower())
        if img_abs is None:
            errors.append(f"VOC 找不到图片: {fname}")
            continue

        size = r.find("size")
        w = int(size.find("width").text or 0) if size is not None and size.find("width") is not None else 0
        h = int(size.find("height").text or 0) if size is not None and size.find("height") is not None else 0
        if not w or not h:
            try:
                w, h = _image_dims(img_abs)
            except Exception:
                w, h = 0, 0

        boxes = []
        for obj in r.findall("object"):
            name_node = obj.find("name")
            label = name_node.text.strip() if name_node is not None and name_node.text else ""
            if not label:
                continue
            if label not in classes:
                classes.append(label)
            bb = obj.find("bndbox")
            if bb is None:
                continue
            try:
                x1 = float(bb.find("xmin").text)
                y1 = float(bb.find("ymin").text)
                x2 = float(bb.find("xmax").text)
                y2 = float(bb.find("ymax").text)
            except (AttributeError, TypeError, ValueError):
                continue
            boxes.append({"class_id": classes.index(label), "x1": x1, "y1": y1, "x2": x2, "y2": y2})

        matched_images += 1
        if _write_ann(root, img_abs.stem, img_abs, w, h, boxes, classes_known=bool(classes)):
            annotated += 1
    return {"images": matched_images, "annotated": annotated, "classes": classes, "errors": errors}


# ----------------------------------------------------------------------
# 统一入口
# ----------------------------------------------------------------------
def import_external_annotations(root: Path, dataset_id: str, version: str,
                                model_id: str = "") -> dict:
    """prepare 阶段统一入口：检测格式并转换为每图 JSON。

    root = datasets/{dataset_id}/{version}/ 解压后的目录。
    返回 {format, images, annotated, classes, errors}；识别不出外部格式时返回 format=''。
    """
    fmt = detect_format(root)
    if not fmt:
        return {"format": "", "images": 0, "annotated": 0, "classes": [], "errors": []}

    images_dir = root / "images"
    if fmt == "coco":
        result = _import_coco(root)
    elif fmt == "labelme":
        result = _import_labelme(root)
    else:
        result = _import_voc(root)

    # 图片平铺（先转换标注后移动图片，标注定位已完成；这里处理孤儿图片与同目录整理）
    _flatten_images(root, images_dir)

    # 同步回填 dataset_id / version / model_id，并刷新 image_path（图片已被平铺移动）
    refresh_image_paths(root, dataset_id, version, model_id)

    # 类别写入 classes.txt（供 create_task 自动读取合并）
    if result["classes"]:
        (root / "classes.txt").write_text("\n".join(result["classes"]) + "\n", encoding="utf-8")

    result["format"] = fmt
    return result


def refresh_image_paths(root: Path, dataset_id: str, version: str, model_id: str = ""):
    """图片平铺后刷新每图 JSON 的 image_path，并回填数据集归属字段"""
    images_dir = root / "images"
    ann_dir = root / "annotations"
    if not ann_dir.exists():
        return
    for p in ann_dir.glob("*.json"):
        try:
            data = _load_json(p)
        except Exception:
            continue
        img_id = data.get("image_id", p.stem)
        img_abs = None
        if images_dir.exists():
            for img in images_dir.glob(f"{img_id}.*"):
                img_abs = img
                break
        if img_abs is not None:
            data["image_path"] = _rel_to_data_dir(img_abs)
        data["dataset_id"] = dataset_id
        data["version"] = version
        data["model_id"] = model_id
        _save_json(p, data)
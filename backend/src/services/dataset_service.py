import json
import zipfile
import tarfile
import shutil
import asyncio
import time
import yaml
import os
import re
import logging
import tempfile
import random
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import UploadFile
from src.core.settings import settings
from src.utils.fs_tree import build_tree
from src.services.annotation_importers import import_external_annotations

logger = logging.getLogger(__name__)

# 支持的图片后缀
IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 支持的上传压缩包格式
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
# 排除可视化预览图（如 *_result.jpg）
EXCLUDE_KEYWORDS = ("_result", "_predict", "_detect")

# ===== 数据集生命周期状态机（MLOps 改造 1.1）=====
# stage：主状态（对齐设计文档 1.2 状态机）
STAGE_COLLECTING = "collecting"    # 采集/接收中（数据持续写入）
STAGE_ANNOTATING = "annotating"    # 标注中（边采边标：AI粗标 + 人工修正）
STAGE_SEALED = "sealed"            # 封板（不可变，等待进入训练）
STAGE_TRAINING = "training"        # 训练中（已在全局训练队列）
STAGE_COMPLETED = "completed"      # 已完成训练（守门员合格）
STAGE_FAILED = "failed"            # 训练失败/模型不合格（保持未完成，参与下一轮雪球）
ALL_STAGES = (
    STAGE_COLLECTING, STAGE_ANNOTATING, STAGE_SEALED,
    STAGE_TRAINING, STAGE_COMPLETED, STAGE_FAILED,
)
# 可进入训练队列的 stage
TRAINABLE_STAGES = (STAGE_SEALED, STAGE_FAILED)

# training_status：训练完成标记（决定是否参与下一轮聚合训练）
TRAIN_STATUS_INCOMPLETE = "incomplete"   # 未完成训练（可参与聚合/回收雪球）
TRAIN_STATUS_COMPLETED = "completed"     # 已完成训练（不再送训）


def _ensure_stage_fields(meta: dict) -> dict:
    """为数据集 meta 补全状态机字段（兼容旧数据回填：uploaded/prepared → 标注中）"""
    meta.setdefault("stage", STAGE_ANNOTATING)
    meta.setdefault("training_status", TRAIN_STATUS_INCOMPLETE)
    meta.setdefault("sealed_at", None)
    meta.setdefault("trained_at", None)
    meta.setdefault("last_trained_round", 0)
    # stage 合法性兜底：旧数据只有 status，新代码里 stage 非法时回退到标注中
    if meta["stage"] not in ALL_STAGES:
        meta["stage"] = STAGE_ANNOTATING
    return meta


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


def _save_yaml(path: Path, data: dict):
    """同步保存 YAML 文件"""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)


def _write_bytes(path: Path, content: bytes):
    """同步写入二进制文件"""
    with open(path, "wb") as f:
        f.write(content)


def _ensure_long_path(path_str: str) -> str:
    """确保路径支持 Windows 长路径（超过 260 字符）
    
    在 Windows 上，如果路径超过 260 字符，需要使用 \\?\ 前缀
    """
    if os.name == 'nt' and len(path_str) > 260:
        if not path_str.startswith('\\\\?\\'):
            # 统一路径格式并转换为绝对路径
            normalized_path = os.path.abspath(os.path.normpath(path_str))
            # 如果是 UNC 路径 (\\server\share)，需要使用 \\?\UNC\ 前缀
            if normalized_path.startswith('\\\\'):
                if not normalized_path.startswith('\\\\?\\'):
                    path_str = '\\\\?\\UNC\\' + normalized_path[2:]
            else:
                path_str = '\\\\?\\' + normalized_path
    return path_str


def _extract_zip(zip_path: Path, extract_to: Path):
    """同步解压 ZIP 文件
    
    修复 Windows 长路径问题：手动创建目录并提取文件
    """
    # 确保目标目录存在
    extract_to_str = _ensure_long_path(str(extract_to.absolute()))
    os.makedirs(extract_to_str, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 获取所有文件列表
        file_list = zip_ref.namelist()
        logger.info(f"Extracting {len(file_list)} files from {zip_path.name}")
        
        extracted_count = 0
        error_count = 0
        
        for member in file_list:
            try:
                # 安全地处理路径（防止路径遍历攻击）
                member_path = Path(member)
                # 跳过绝对路径或包含 .. 的路径
                if member_path.is_absolute() or '..' in str(member_path):
                    logger.warning(f"Skipping unsafe path: {member}")
                    continue
                
                # 构建目标路径
                target_path = extract_to / member_path
                target_str = str(target_path.absolute())
                
                # 确保使用长路径格式
                target_str = _ensure_long_path(target_str)
                
                # 获取文件信息
                info = zip_ref.getinfo(member)
                
                # 如果是目录
                if info.is_dir():
                    os.makedirs(target_str, exist_ok=True)
                else:
                    # 确保父目录存在（也需要使用长路径）
                    parent_str = _ensure_long_path(str(target_path.parent.absolute()))
                    os.makedirs(parent_str, exist_ok=True)
                    
                    # 提取文件内容
                    with zip_ref.open(member) as source:
                        with open(target_str, 'wb') as target:
                            shutil.copyfileobj(source, target)
                    
                    extracted_count += 1
                    if extracted_count % 100 == 0:
                        logger.debug(f"Extracted {extracted_count} files...")
            
            except Exception as e:
                error_count += 1
                logger.error(f"Error extracting {member}: {e}")
                # 继续处理其他文件，不中断整个解压过程
                continue
        
        logger.info(f"Extraction completed: {extracted_count} files extracted, {error_count} errors")


def _extract_tar(archive_path: Path, extract_to: Path, mode: str):
    """同步解压 tar / tar.gz / tgz 文件（处理 Windows 长路径与路径安全）"""
    extract_to_str = _ensure_long_path(str(extract_to.absolute()))
    os.makedirs(extract_to_str, exist_ok=True)

    with tarfile.open(archive_path, mode) as tf:
        members = tf.getmembers()
        logger.info(f"Extracting {len(members)} files from {archive_path.name}")

        extracted_count = 0
        error_count = 0

        for member in members:
            try:
                member_path = Path(member.name)
                # 跳过绝对路径或包含 .. 的路径（防止路径遍历攻击）
                if member_path.is_absolute() or '..' in str(member_path):
                    logger.warning(f"Skipping unsafe path: {member.name}")
                    continue

                target_path = extract_to / member_path
                target_str = _ensure_long_path(str(target_path.absolute()))

                if member.isdir():
                    os.makedirs(target_str, exist_ok=True)
                elif member.isfile():
                    parent_str = _ensure_long_path(str(target_path.parent.absolute()))
                    os.makedirs(parent_str, exist_ok=True)
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    with src as source, open(target_str, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    extracted_count += 1
                    if extracted_count % 100 == 0:
                        logger.debug(f"Extracted {extracted_count} files...")
                else:
                    # 跳过符号链接等特殊条目（Windows 上无意义且易出错）
                    logger.debug(f"Skipping special entry: {member.name}")

            except Exception as e:
                error_count += 1
                logger.error(f"Error extracting {member.name}: {e}")
                continue

        logger.info(f"Extraction completed: {extracted_count} files extracted, {error_count} errors")


def _extract_archive(archive_path: Path, extract_to: Path):
    """按扩展名分发解压数据集压缩包（zip / tar / tar.gz / tgz）"""
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        _extract_zip(archive_path, extract_to)
    elif name.endswith(".tar.gz") or name.endswith(".tgz"):
        _extract_tar(archive_path, extract_to, "r:gz")
    elif name.endswith(".tar"):
        _extract_tar(archive_path, extract_to, "r:")
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")


def _delete_directory(path: Path):
    """同步删除目录"""
    shutil.rmtree(path)


def _delete_file(path: Path):
    """同步删除文件"""
    if path.exists():
        path.unlink()


class DatasetService:
    def __init__(self):
        self.datasets_dir = settings.DATASETS_DIR
        self.uploads_dir = settings.UPLOADS_DIR
        
    async def upload_dataset(self, file: UploadFile, model_id: str = None):
        """上传数据集压缩包（zip / tar / tar.gz / tgz）

        model_id: 归属模型（1.7 模型仓库：数据挂到对应模型下）；为空则由后续采集/绑定流程归属
        """
        dataset_id = f"ds_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_dir = self.datasets_dir / dataset_id
        
        # 异步创建目录
        await asyncio.to_thread(lambda: dataset_dir.mkdir(parents=True, exist_ok=True))
        
        # 保存上传的压缩包，保留原始后缀便于 prepare 时按格式解压
        raw_name = (file.filename or "").lower()
        suffix = next((s for s in ARCHIVE_SUFFIXES if raw_name.endswith(s)), ".zip")
        archive_name = f"{dataset_id}{suffix}"
        archive_path = self.uploads_dir / archive_name
        content = await file.read()
        await asyncio.to_thread(_write_bytes, archive_path, content)
        
        # 保存元数据（stage=collecting：上传即进入采集/接收阶段）
        meta = {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "archive": archive_name,
            "size": len(content),
            "uploaded_at": datetime.now().isoformat(),
            "status": "uploaded",
            "stage": STAGE_COLLECTING,
            "training_status": TRAIN_STATUS_INCOMPLETE,
            "sealed_at": None,
            "trained_at": None,
            "last_trained_round": 0,
            "model_id": model_id,
        }
        
        meta_path = dataset_dir / "meta.json"
        await asyncio.to_thread(_save_json, meta_path, meta)
        
        return meta
    
    async def prepare_dataset(self, dataset_id: str, split_ratio: dict, classes: list = None):
        """准备数据集"""
        t0 = time.time()
        dataset_dir = self.datasets_dir / dataset_id
        if not await asyncio.to_thread(lambda: dataset_dir.exists()):
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # 读取元数据
        meta_path = dataset_dir / "meta.json"
        meta = await asyncio.to_thread(_load_json, meta_path)
        
        # 解压数据集（准备语义=从 zip 重新构建，先清空版本目录旧内容，
        # 避免重复准备时与上一次解压内容叠加，例如 zip 顶层目录残留 + 双份图片）
        version = "v1"
        version_dir = dataset_dir / version
        if await asyncio.to_thread(lambda: version_dir.exists()):
            def clear_dir():
                for child in list(version_dir.iterdir()):
                    if child.name == ".staging":
                        continue
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        try:
                            child.unlink()
                        except OSError:
                            pass
            await asyncio.to_thread(clear_dir)
        else:
            await asyncio.to_thread(lambda: version_dir.mkdir(exist_ok=True))
        
        # 按 meta 记录的文件名找到上传的压缩包并解压（兼容旧版本固定 .zip 命名）
        archive_name = meta.get("archive") or f"{dataset_id}.zip"
        archive_path = self.uploads_dir / archive_name
        await asyncio.to_thread(_extract_archive, archive_path, version_dir)
        t_extract = time.time()
        logger.info(f"prepare {dataset_id}: extract took {t_extract - t0:.1f}s")
        
        # 「准备」= 把数据集改造成平台可用格式：
        # 0) 外部分类格式（COCO / LabelMe / VOC）→ 统一转一图一 JSON（数据链路阶段B），
        #    产物 images/（平铺）+ annotations/{image_id}.json + classes.txt，跳过重组
        detected_classes = classes or []
        model_id_ds = str(meta.get("model_id", "") or "")
        # 目标模型标签空间（模型标签字典英文码），用于 txt 类名重映射 / 冗余类过滤
        model_labels = self._labels_from_model(model_id_ds) if model_id_ds else []
        # 类别是否来自包内（用户显式传 / 包内类别文件 / 标签推断）：
        # 只有包内来源才是可信的"源类名列表"，可与模型标签做交集过滤；模型回填不算
        classes_from_package = bool(classes)
        import_result = await asyncio.to_thread(
            import_external_annotations, version_dir, dataset_id, version, model_id_ds
        )
        external_format = import_result["format"]
        if external_format:
            has_labels = import_result["annotated"] > 0
            logger.info(
                f"prepare {dataset_id}: external format '{external_format}' imported "
                f"annotated={import_result['annotated']}/{import_result['images']}"
            )
            if import_result["errors"]:
                logger.warning(f"prepare {dataset_id}: import errors={len(import_result['errors'])}")
        else:
            # 1) 已符合 images/{train,val,test}(+labels) 标准结构 → 保持原样
            # 2) 其他任意目录（图片与标注同目录 / labels 子目录 / 原始无标注数据集等）
            #    → 自动重组为 images/{train,val,test} + labels/{train,val,test} 标准结构
            is_standard = await asyncio.to_thread(self._is_standard_structure, version_dir)
            if not is_standard:
                # 重组前先从解压内容中读取类别（zip 自带的 classes.txt / data.yaml），
                # 供重组阶段写 classes.txt 使用（重组会清理原目录）
                if not detected_classes:
                    detected_classes = await asyncio.to_thread(self._load_classes_from_files, version_dir) or []
                    classes_from_package = True
                has_labels = await asyncio.to_thread(self._reorganize_dataset, version_dir, split_ratio, detected_classes)
                logger.info(f"reorganized dataset {dataset_id}: standard structure, has_labels={has_labels}")
            else:
                has_labels = None  # 标准结构：标注情况由下方 label_count 决定
        t_reorganize = time.time()
        logger.info(f"prepare {dataset_id}: reorganize/scan took {t_reorganize - t_extract:.1f}s")
        
        # 检查目录结构（在线程中执行）
        images_dir = await asyncio.to_thread(self._find_images_dir, version_dir)
        labels_dir = await asyncio.to_thread(self._find_labels_dir, version_dir)
        
        if not images_dir:
            raise ValueError("No images directory found in dataset")
        
        # 统计图片和标签（在线程中执行；单遍遍历，避免三次 rglob 全盘扫描）
        def count_files():
            image_count = 0
            label_count = 0
            if labels_dir:
                label_files = set(labels_dir.rglob("*.txt"))
                label_count = len(label_files)
            for p in images_dir.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() in (".jpg", ".png", ".jpeg"):
                    image_count += 1
            return image_count, label_count
        
        image_count, label_count = await asyncio.to_thread(count_files)
        
        # 检测类别：优先从解压后的 data.yaml 中读取，否则从标签文件检测
        if not detected_classes:
            detected_classes = await asyncio.to_thread(self._load_classes_from_files, version_dir) or []
            if detected_classes:
                classes_from_package = True
            
            # 如果从 YAML 中没有读取到类别，且存在标签目录，则从标签文件检测
            if not detected_classes and labels_dir:
                detected_classes = await asyncio.to_thread(self._detect_classes, labels_dir)
                if detected_classes:
                    classes_from_package = True
                logger.info(f"Detected classes from labels: {detected_classes}")
        
        # 确保有类别列表（如果检测不到，使用空列表）
        if not detected_classes:
            # 纯图片包（无类别文件/无标签）：从所属模型标签字典自动补类别
            detected_classes = self._labels_from_model(model_id_ds) or []
            classes_from_package = False
            if detected_classes:
                logger.info(f"prepare {dataset_id}: classes backfilled from model {model_id_ds}: {detected_classes}")

        # 多模型上传自动匹配（#3）：未归属 + 包内类别可信（txt/yaml/classes.txt 来源）→
        # 按类名匹配已有模型（先分配后过滤：后续 class_map 只保留本模型标签空间）；
        # 匹配不到 → 保持未归属进入「未归属数据集」列表（兜底，可在模型详情页手动绑定）
        auto_match = None
        if not model_id_ds and classes_from_package and detected_classes:
            matched_id = self._auto_match_model(detected_classes)
            if matched_id:
                try:
                    mcfg = _load_json(settings.REGISTRY_DIR / matched_id / "model.json")
                except Exception:
                    mcfg = {}
                meta["model_id"] = matched_id
                model_id_ds = matched_id
                model_labels = self._labels_from_model(matched_id) or []
                auto_match = {
                    "model_id": matched_id,
                    "model_code": mcfg.get("model_code") or matched_id,
                    "display_name": mcfg.get("display_name") or mcfg.get("name") or "",
                }
                logger.info(
                    f"prepare {dataset_id}: auto-matched model {matched_id} "
                    f"by package classes {detected_classes}"
                )
        
        # 数据链路改造（阶段C）：prepare 不再生成 data.yaml（封板时才生成），
        # 不再划分数据集（划分也在封板时进行）；类别写 classes.txt 供标注任务自动读取
        if detected_classes:
            classes_txt_path = version_dir / "classes.txt"
            await asyncio.to_thread(
                classes_txt_path.write_text,
                "\n".join(detected_classes) + "\n",
                encoding="utf-8",
            )
        
        # 标注情况：标准结构直接用 label_count，重组/外部导入后使用其返回结果
        final_has_labels = (label_count > 0) if has_labels is None else has_labels
        
        # 标注格式校验（2.1 第一级：格式规则校验），结果写入 meta 供封板/前端展示
        label_validation = None
        if labels_dir and label_count > 0:
            label_validation = self._validate_labels(labels_dir, len(detected_classes))
            logger.info(
                f"validate {dataset_id}: status={label_validation['status']} "
                f"files={label_validation['checked_files']} lines={label_validation['lines_checked']}"
            )
        
        # YOLO labels/*.txt → 每图 JSON（统一权威标注源；标注页可直接读原标注）
        # 重映射：txt class_id（源包类空间）→ 模型标签索引；只保留本模型需要的类（剔除冗余标签）
        # 源 labels 目录保留（原始标注供追溯），每图 JSON 为平台权威源
        txt_to_json = {"converted": 0, "labels_removed": False, "class_map": None, "dropped_boxes": 0}
        class_filter = None
        if labels_dir and labels_dir.is_dir() and final_has_labels and label_count > 0:
            class_map = None
            if classes_from_package and model_labels:
                class_map = self._build_class_map(detected_classes, model_labels)
                if class_map:
                    dropped = [
                        str(c) for i, c in enumerate(detected_classes) if i not in class_map
                    ]
                    class_filter = {
                        "src_classes": [str(c) for c in detected_classes],
                        "model_labels": list(model_labels),
                        "kept": [str(c) for i, c in enumerate(detected_classes) if i in class_map],
                        "dropped": dropped,
                    }
                    logger.info(
                        f"prepare {dataset_id}: class filter to model {model_id_ds}: "
                        f"kept={class_filter['kept']} dropped={class_filter['dropped']}"
                    )
            try:
                txt_to_json = await asyncio.to_thread(self._import_labels_txt_to_json, version_dir, class_map)
                txt_to_json["class_map"] = bool(class_map)
                if txt_to_json.get("converted", 0) > 0:
                    # 保留源 labels（不再删除）：txt 为原始标注供追溯，每图 JSON 为权威源
                    # 有效标注以每图 JSON 为准（类过滤后可能整批无本模型有效框）
                    json_annotated = self._count_json_annotations(version_dir)
                    if json_annotated == 0 and txt_to_json.get("dropped_boxes", 0) > 0:
                        logger.info(f"prepare {dataset_id}: all txt boxes dropped by class filter (model={model_id_ds})")
                    final_has_labels = json_annotated > 0
                    logger.info(f"prepare {dataset_id}: txt->json converted={txt_to_json['converted']}, labels kept")
                # class_map 有效 → 数据集类别空间收敛为模型标签字典（与框索引对齐）
                if class_map and model_labels:
                    detected_classes = list(model_labels)
                    classes_txt_path = version_dir / "classes.txt"
                    await asyncio.to_thread(
                        classes_txt_path.write_text, "\n".join(detected_classes) + "\n", encoding="utf-8"
                    )
            except Exception:
                logger.exception(f"prepare {dataset_id}: txt->json import failed, labels kept")

        # 更新元数据（prepare 完成 → 进入标注阶段：边采边标）
        meta.update({
            "status": "prepared",
            "stage": STAGE_ANNOTATING,
            "version": version,
            "image_count": image_count,
            "label_count": label_count,
            "has_labels": final_has_labels,
            "split_ratio": split_ratio,  # 封板划分时复用（封板请求未指定时）
            "classes": detected_classes,
            "class_filter": class_filter,  # 按模型剔除的冗余类记录（None=未过滤）
            "label_validation": label_validation,
            "txt_to_json": txt_to_json,
            "model_auto_match": auto_match,  # 多模型上传自动匹配记录（None=未匹配）
            "prepared_at": datetime.now().isoformat()
        })

        # 雪球闭环（1.8）：图片数达到阈值自动创建标注任务，标注页可直接进入，无需手动建任务
        # 去重：若该数据集（同版本）已存在标注任务则复用，不重复创建
        auto_task = None
        if settings.AUTO_TASK_MIN_IMAGES > 0 and image_count >= settings.AUTO_TASK_MIN_IMAGES:
            try:
                from src.services.annotation_service import AnnotationService
                ann_svc = AnnotationService()
                existing = await asyncio.to_thread(ann_svc.find_task_by_dataset, dataset_id, version)
                if existing:
                    auto_task = existing
                    logger.info(f"prepare {dataset_id}: reuse annotation task {existing['task_id']}")
                else:
                    auto_task = await ann_svc.create_task(dataset_id, version, list(detected_classes))
                    logger.info(f"prepare {dataset_id}: auto-created annotation task {auto_task['task_id']} ({auto_task['total_images']} imgs)")
            except Exception as e:
                logger.warning(f"prepare {dataset_id}: auto-create annotation task failed: {e}")
                auto_task = None
        meta["annotation_task_id"] = (auto_task or {}).get("task_id")

        await asyncio.to_thread(_save_json, meta_path, meta)

        logger.info(f"prepare {dataset_id} done: total {time.time() - t0:.1f}s, images={image_count}, labels={label_count}")

        return {
            "dataset_id": dataset_id,
            "version": version,
            "image_count": image_count,
            "label_count": label_count,
            "has_labels": final_has_labels,
            "classes": detected_classes,
            "model_id": model_id_ds or None,
            "model_auto_match": auto_match,  # 自动匹配的模型（None=未匹配/已手动指定）
            "annotation_task_id": (auto_task or {}).get("task_id")
        }
    
    def _is_standard_structure(self, root_dir: Path) -> bool:
        """判断数据集是否已是标准结构（images/{train,val,test} 至少一个划分非空）

        已符合标准 → 保持原样不重组（避免破坏标注导出等已划分数据）。
        """
        img_root = root_dir / "images"
        if not img_root.is_dir():
            return False
        for s in ("train", "val", "test"):
            d = img_root / s
            if d.is_dir() and any(d.iterdir()):
                return True
        return False

    def _load_classes_from_files(self, root_dir: Path):
        """从解压目录读取类别文件（yaml / txt / names），递归查找"""
        def read_names(path):
            if path.suffix.lower() in (".yaml", ".yml"):
                data = _load_yaml(path)
                names = data.get('names', [])
                # YOLO 支持 names: {0: 'a', 1: 'b'} 的字典形式，转为按 id 排序的列表
                if isinstance(names, dict):
                    return [str(names[k]) for k in sorted(names)]
                return names
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        # 1) 优先查找根目录下的类别文件
        # 纯文本类别文件（classes.txt 等）是真实类别名，优先级高于 data.yaml
        # （data.yaml 可能保存的是上一次 prepare 生成的占位名 class_N，会覆盖真实名）
        candidates = [
            root_dir / "classes.txt",
            root_dir / "coco.names",
            root_dir / "names.txt",
            root_dir / "dataset.yaml",
            root_dir / "config.yaml",
            root_dir / "data.yaml",
        ]
        for path in candidates:
            if path.exists():
                try:
                    names = read_names(path)
                    if names:
                        logger.info(f"Loaded classes from {path}: {names}")
                        return names
                except Exception as e:
                    logger.warning(f"Failed to load classes from {path}: {e}")

        # 2) 递归查找（类别文件可能在 zip 的子目录中）
        for name in ["classes.txt", "coco.names", "names.txt", "dataset.yaml", "config.yaml", "data.yaml"]:
            for path in root_dir.rglob(name):
                try:
                    names = read_names(path)
                    if names:
                        logger.info(f"Loaded classes from {path}: {names}")
                        return names
                except Exception as e:
                    logger.warning(f"Failed to load classes from {path}: {e}")
        return None

    def _reorganize_dataset(self, root_dir: Path, split_ratio: dict, classes: list) -> bool:
        """把非标准目录整理为 images/（平铺）+ labels/（平铺）标准结构。

        数据链路改造（阶段C）：prepare 不再划分 train/val/test（划分移到封板阶段）、
        不再生成 data.yaml；此处仅做目录整理与 YOLO txt 标注归集：
        - 图片一律平铺进 images/（无标注图片也保留入库，供后续 AI/人工 标注）
        - 有对应 txt 标注的图片：标注平铺进 labels/（同名 .txt）
        返回是否有可用标注。
        """
        # 1) 收集图片与 txt 标注（跳过隐藏文件 / 预览图 / labels 目录内的图片）
        images, txts = [], []
        for p in root_dir.rglob("*"):
            if not p.is_file():
                continue
            low_parts = [x.lower() for x in p.relative_to(root_dir).parts]
            if any(x.startswith(".") for x in low_parts) or "__macosx" in low_parts:
                continue
            suf = p.suffix.lower()
            if suf in IMG_SUFFIX:
                if any(k in p.name.lower() for k in EXCLUDE_KEYWORDS):
                    continue
                if "labels" in low_parts:
                    continue
                images.append(p)
            elif suf == ".txt":
                txts.append(p)
        images = sorted(set(images))
        txts = sorted(set(txts))

        # 空压缩包或无图片：直接报错，避免后续 staging 流程崩溃
        if not images:
            raise ValueError("压缩包内未找到图片（jpg/png/jpeg），请检查上传内容")

        img_by_stem = {}
        for im in images:
            img_by_stem.setdefault(im.stem, im)

        # 2) txt 与图片按 stem 配对（同名先到先得）
        matched = []  # [(img, txt)]
        used = set()
        for t_txt in txts:
            im = img_by_stem.get(t_txt.stem)
            if im is None:
                continue
            matched.append((im, t_txt))
            used.add(im)
        orphans = sorted(set(images) - used)  # 无对应标注的图片（保留入库，待标注）
        has_labels = bool(matched)

        # 3) 输出到暂存目录（用 move 而非 copy：staging 与源在同一文件系统，rename 瞬时完成）
        staging = root_dir / ".staging"
        tmp_images = staging / "images"
        tmp_labels = staging / "labels"

        def move_flat(p: Path, dst_dir: Path):
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / p.name
            if not dst.exists():
                shutil.move(str(p), str(dst))

        for im, t_txt in matched:
            move_flat(im, tmp_images)
            move_flat(t_txt, tmp_labels)
        for im in orphans:
            move_flat(im, tmp_images)

        if classes:
            (staging / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

        # 4) 清空根目录并移入整理结果（zip 原始文件保留在 uploads，可随时重新准备）
        for child in list(root_dir.iterdir()):
            if child.name == ".staging":
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
        for child in staging.iterdir():
            shutil.move(str(child), str(root_dir / child.name))
        try:
            staging.rmdir()
        except OSError:
            pass

        if orphans:
            logger.warning(
                f"reorganize: {len(orphans)} images without labels kept for annotation, "
                f"e.g. {[p.name for p in orphans[:3]]}"
            )
        if not has_labels:
            logger.warning("reorganize: dataset has no labels, images kept flat. Use AI annotation before training.")
        return has_labels

    def _find_images_dir(self, root_dir: Path):
        """查找images目录（同步方法，在线程中调用）"""
        candidates = list(root_dir.rglob("images"))
        if candidates:
            return candidates[0]
        # 如果没有images子目录，检查是否根目录直接包含图片
        image_files = list(root_dir.glob("*.jpg")) + list(root_dir.glob("*.png"))
        if image_files:
            imgs_dir = root_dir / "images"
            imgs_dir.mkdir(exist_ok=True)
            for img in image_files:
                shutil.move(str(img), str(imgs_dir / img.name))
            return imgs_dir
        return None
    
    def _find_labels_dir(self, root_dir: Path):
        """查找labels目录（同步方法，在线程中调用）"""
        candidates = list(root_dir.rglob("labels"))
        return candidates[0] if candidates else None
    
    def _detect_classes(self, labels_dir: Path):
        """从标签文件检测类别（同步方法，在线程中调用）"""
        class_ids = set()
        for label_file in labels_dir.rglob("*.txt"):
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_ids.add(int(parts[0]))
            except:
                continue
        
        # 生成默认类别名
        max_id = max(class_ids) if class_ids else 0
        return [f"class_{i}" for i in range(max_id + 1)]

    def _check_yolo_line(self, parts: list, nc: int):
        """校验单行 YOLO 标注（class x y w h），返回 (ok, err_msg)

        - 必须恰好 5 列；class 为 [0, nc) 内整数；坐标均为 [0,1] 归一化浮点
        - 允许 1e-3 数值容差（第三方工具常有 -0.0001 / 1.0001 这类边界抖动）
        """
        if len(parts) != 5:
            return False, f"{len(parts)} 列 ≠ 5 列(class x y w h)"
        try:
            cid = int(float(parts[0]))
        except ValueError:
            return False, f"类别 '{parts[0]}' 不是数字"
        if nc and (cid < 0 or cid >= nc):
            return False, f"类别 {cid} 越界(应有 {nc} 类)"
        try:
            vals = [float(p) for p in parts[1:]]
        except ValueError:
            return False, "坐标不是数字"
        for v in vals:
            if v < -1e-3 or v > 1 + 1e-3:
                return False, f"坐标 {v:.4f} 超出 [0,1]"
        return True, ""

    def _validate_labels(self, labels_dir: Path, nc: int) -> dict:
        """批量校验 YOLO 标注格式（2.1 入料校验三级防线·第一级：格式规则校验）

        逐文件逐行校验（class 范围 / 列数 / 坐标范围），结果写回 meta 的
        label_validation 字段；封板时以 status 为校验✓判定依据。
        """
        checked_files = 0
        lines_checked = 0
        invalid_files = 0
        invalid_lines = 0
        error_samples = []
        for file in sorted(labels_dir.rglob("*.txt")):
            if file.name.startswith("._"):  # macOS 元数据文件
                continue
            checked_files += 1
            file_invalid = False
            try:
                with open(file, "r", encoding="utf-8") as f:
                    for line_no, raw in enumerate(f, 1):
                        line = raw.strip()
                        if not line:
                            continue
                        lines_checked += 1
                        ok, err = self._check_yolo_line(line.split(), nc)
                        if not ok:
                            invalid_lines += 1
                            file_invalid = True
                            if len(error_samples) < 20:
                                error_samples.append(f"{file.name}:{line_no} → {err}")
            except Exception as e:
                invalid_files += 1
                if len(error_samples) < 20:
                    error_samples.append(f"{file.name} 读取失败: {e}")
                continue
            if file_invalid:
                invalid_files += 1
        return {
            "status": "ok" if invalid_lines == 0 else "failed",
            "checked_files": checked_files,
            "lines_checked": lines_checked,
            "invalid_files": invalid_files,
            "invalid_lines": invalid_lines,
            "errors": error_samples,
            "checked_at": datetime.now().isoformat(),
        }
    
    def _resolve_real_classes(self, dataset_dir: Path, meta: dict) -> list:
        """将占位类别名（class_N）回填为真实类别名

        历史数据 prepare 时可能因未读到类别文件而生成了 class_0/class_1 等占位名。
        若目录中存在真实类别文件（classes.txt / coco.names / names.txt），则按行读取并回填。
        """
        classes = meta.get("classes") or []
        if not classes:
            return classes
        # 仅当全部是占位名时才回填（避免覆盖用户自定义的真实类别名）
        if not all(re.fullmatch(r"class_\d+", str(c)) for c in classes):
            return classes
        version_dir = dataset_dir / meta.get("version", "v1")
        for name in ["classes.txt", "coco.names", "names.txt"]:
            for path in version_dir.rglob(name):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        names = [line.strip() for line in f if line.strip()]
                    if names and len(names) == len(classes):
                        return names
                except Exception:
                    continue
        return classes

    def _is_annotated_exported(self, dataset_dir: Path, version: str) -> bool:
        """标注导出会生成 labels/{train,val,test} 划分结构（含 .txt 标签），
        这是标注导出独有的标志，可用于区分仅上传/准备的数据集。
        """
        version_dir = dataset_dir / version
        if not version_dir.exists():
            return False
        # 标注导出固定写入 version_dir/labels/{train,val,test}
        labels = version_dir / "labels"
        train_labels = labels / "train"
        if not train_labels.is_dir():
            return False
        try:
            return any(train_labels.iterdir())
        except Exception:
            return False

    def _collect_legacy_txt_annotations(self, version_dir: Path) -> list:
        """旧数据兜底：从 images/ + labels/ txt 收集标注（未走每图 JSON 的历史数据集）"""
        images_dir = version_dir / "images"
        labels_dir = version_dir / "labels"
        if not images_dir.exists():
            return []

        img_pool = []
        if images_dir.is_dir():
            for s in ("", "train", "val", "test"):
                d = images_dir if not s else images_dir / s
                if d.is_dir():
                    img_pool += list(d.iterdir())
        label_pool = []
        if labels_dir and labels_dir.is_dir():
            for s in ("", "train", "val", "test"):
                d = labels_dir if not s else labels_dir / s
                if d.is_dir():
                    label_pool += list(d.glob("*.txt"))
        by_stem = {}
        for lp in label_pool:
            by_stem.setdefault(lp.stem, lp)

        annotated = []
        for img in img_pool:
            if not (img.is_file() and img.suffix.lower() in IMG_SUFFIX):
                continue
            lp = by_stem.get(img.stem)
            if lp is None:
                continue
            try:
                from PIL import Image
                with Image.open(img) as im:
                    w, h = im.size
            except Exception:
                continue
            boxes = []
            try:
                with open(lp, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cid = int(float(parts[0]))
                        cx, cy, bw, bh = [float(v) for v in parts[1:5]]
                        boxes.append({
                            "class_id": cid,
                            "x1": (cx - bw / 2) * w, "y1": (cy - bh / 2) * h,
                            "x2": (cx + bw / 2) * w, "y2": (cy + bh / 2) * h,
                        })
            except Exception:
                continue
            if boxes:
                annotated.append({
                    "image_id": img.stem, "img_abs": img,
                    "width": w, "height": h, "boxes": boxes,
                })
        return annotated

    def _build_class_map(self, src_classes: list, model_labels: list) -> dict:
        """构建 txt class_id 重映射表：源包类名 → 目标模型标签索引。

        只保留目标模型需要的类（模型标签字典英文码），其余剔除（冗余标签过滤）。
        返回 {源index: 模型index}；双方任一为空 / 无交集时返回 None（保持原始 index）。
        """
        if not src_classes or not model_labels:
            return None
        model_idx = {
            str(n).strip().lower(): i for i, n in enumerate(model_labels)
            if str(n).strip()
        }
        class_map = {}
        for src_idx, name in enumerate(src_classes):
            key = str(name).strip().lower()
            if key and key in model_idx:
                class_map[src_idx] = model_idx[key]
        return class_map or None

    def _import_labels_txt_to_json(self, version_dir: Path, class_map: dict = None) -> dict:
        """把 YOLO labels/*.txt 标注转换为每图 JSON（一图一 JSON 为权威标注源）。

        复用旧数据兜底解析（归一化→像素框 + PIL 读宽高），逐图写
        datasets/{id}/{version}/annotations/{image_id}.json；
        class_map 非空时按目标模型做 class_id 重映射，非本模型类（未命中）剔除；
        返回 {"converted": n, "dropped_boxes": m}。源 labels 目录由调用方决定保留。
        """
        ann_dir = version_dir / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        converted = 0
        dropped_boxes = 0
        for rec in self._collect_legacy_txt_annotations(version_dir):
            img_abs = rec["img_abs"]
            try:
                rel = str(img_abs.resolve().relative_to(settings.DATA_DIR.resolve())).replace("\\", "/")
            except ValueError:
                rel = str(img_abs).replace("\\", "/")
            boxes = []
            for b in rec["boxes"]:
                cid = b.get("class_id")
                if class_map is not None:
                    mapped = class_map.get(cid)
                    if mapped is None:
                        dropped_boxes += 1  # 非本模型需要的类 → 剔除冗余标签
                        continue
                    b = dict(b, class_id=mapped)
                boxes.append(b)
            doc = {
                "annotator": "import",
                "source": "yolo_txt",
                "image_id": rec["image_id"],
                "image_path": rel,
                "width": rec["width"],
                "height": rec["height"],
                "sample_type": "normal",
                "sample_reasons": [],
                "boxes": boxes,  # class_id 数字索引（已映射到目标模型空间）+ 像素框
            }
            _save_json(ann_dir / f"{rec['image_id']}.json", doc)
            converted += 1
        return {"converted": converted, "dropped_boxes": dropped_boxes}

    def _count_json_annotations(self, version_dir: Path) -> int:
        """统计每图 JSON 标注中有框的图片数（数据链路新标注数据源）"""
        ann_dir = version_dir / "annotations"
        if not ann_dir.exists():
            return 0
        n = 0
        for p in ann_dir.glob("*.json"):
            try:
                data = _load_json(p)
            except Exception:
                continue
            if data.get("boxes"):
                n += 1
        return n

    def _labels_from_model(self, model_id: str) -> list:
        """从模型标签字典（labels_dict.json）读取类别英文码列表（按 index 排序）。

        模型不存在 / 字典缺失 / 字典为空 → 返回 []。
        用于纯图片数据集按归属模型自动补类别（prepare 兜底 + bind 回填）。
        """
        if not model_id:
            return []
        try:
            dic = _load_json(settings.REGISTRY_DIR / model_id / "labels_dict.json")
            labels = dic.get("labels") or []
            return [
                (lbl.get("english_code") or "").strip()
                for lbl in sorted(labels, key=lambda x: x.get("index", 0))
                if (lbl.get("english_code") or "").strip()
            ]
        except Exception as e:
            logger.warning(f"_labels_from_model {model_id} failed: {e}")
            return []

    @staticmethod
    def _norm_cls(name: str) -> str:
        """归一化类别名用于比对（小写 + 去所有分隔符），traffic-scene == trafficscene"""
        return re.sub(r"[^0-9a-z]+", "", str(name or "").strip().lower())

    def _auto_match_model(self, classes: list) -> Optional[str]:
        """多模型上传自动匹配（#3）：包内类别 → 与已有模型标签字典比对。

        - 包内所有类被某模型完全覆盖 → 立即命中（最精确）
        - 全部未覆盖时取覆盖率最高者，仅当覆盖率 >= 0.6 才绑定（保守，避免错绑）
        - 匹配不到返回 None → 数据集保持未归属（兜底，后续手动绑定）
        """
        if not classes or not settings.REGISTRY_DIR.exists():
            return None
        base = {self._norm_cls(c) for c in classes if str(c).strip()}
        if not base:
            return None
        best, best_model_id = 0.0, None
        for model_dir in settings.REGISTRY_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            mf = model_dir / "model.json"
            if not mf.exists():
                continue
            try:
                labels = self._labels_from_model(model_dir.name)
                mcfg = _load_json(mf)
            except Exception:
                continue
            label_set = {self._norm_cls(c) for c in labels if str(c).strip()}
            if not label_set:
                continue
            if mcfg.get("empty"):
                continue  # 空白模型无标签空间，不参与自动匹配
            cover = len(base & label_set) / len(base)
            if cover >= 1.0:
                return model_dir.name  # 完全覆盖：立即命中
            if cover > best:
                best, best_model_id = cover, model_dir.name
        if best_model_id and best >= 0.6:
            return best_model_id
        return None

    def _resolve_seal_names(self, meta: dict) -> list:
        """解析封板 data.yaml 的类别名：优先模型标签字典英文码，缺省回退 meta.classes"""
        model_id = meta.get("model_id") or ""
        if model_id:
            try:
                dic = _load_json(settings.REGISTRY_DIR / model_id / "labels_dict.json")
                labels = dic.get("labels") or []
                if labels:
                    names = [
                        (lbl.get("english_code") or lbl.get("chinese_name") or "").strip()
                        for lbl in sorted(labels, key=lambda x: x.get("index", 0))
                    ]
                    if any(names):
                        return names
            except Exception:
                pass
        return meta.get("classes") or []

    def _build_training_artifacts(self, version_dir: Path, meta: dict, split_ratio: dict, out_dir: Path = None) -> dict:
        """封板一次性产出：划分 train/val/test + 每图 JSON → YOLO txt + 生成 data.yaml（数据链路阶段E）

        标注数据源：version_dir/annotations/{image_id}.json（像素坐标框）
        图片源：version_dir/images/ 平铺目录
        out_dir：产物输出目录（缺省 = version_dir 原地）。数据血缘方案 A 下每个封板
        版本输出到数据集下 sealed/v{N}/ 只读快照，旧版本不被覆盖、版本号逐次递增。
        data.yaml 类别名：模型标签字典英文码（缺省回退 meta.classes）
        返回 {total, counts, ratios, nc, names}；无已标注图片时抛 ValueError。
        """
        out_dir = out_dir or version_dir
        ann_dir = version_dir / "annotations"
        src_images_dir = version_dir / "images"
        images_dir = out_dir / "images"
        labels_dir = out_dir / "labels"

        # 1) 收集已标注图片（每图 JSON 中有框），并统计样本类型分布
        annotated = []
        sample_stats = {"normal": 0, "hard": 0, "background": 0}
        # 图片索引（标准结构 nested train/val/test 或平铺均可）
        img_index = {}
        if src_images_dir.exists():
            for s in ("train", "val", "test"):
                d = src_images_dir / s
                if d.is_dir():
                    for f in d.iterdir():
                        if f.is_file() and f.suffix.lower() in IMG_SUFFIX:
                            img_index.setdefault(f.stem, f)
            if src_images_dir.is_dir():
                for f in src_images_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in IMG_SUFFIX:
                        img_index.setdefault(f.stem, f)
        if ann_dir.exists():
            for p in ann_dir.glob("*.json"):
                try:
                    data = _load_json(p)
                except Exception:
                    continue
                boxes = data.get("boxes") or []
                if not boxes:
                    continue
                img_id = data.get("image_id", p.stem)
                img_abs = img_index.get(img_id)
                if img_abs is None:
                    continue
                s_type = data.get("sample_type")
                if s_type in sample_stats:
                    sample_stats[s_type] += 1
                annotated.append({
                    "image_id": img_id,
                    "img_abs": img_abs,
                    "width": data.get("width") or 1,
                    "height": data.get("height") or 1,
                    "boxes": boxes,
                })
        # 每图 JSON 为空时，回退到旧数据 images/ + labels/ txt 流程
        if not annotated:
            annotated = self._collect_legacy_txt_annotations(version_dir)
        if not annotated:
            raise ValueError("没有已标注的图片可封板（annotations/*.json 与 labels/*.txt 均为空）")

        # 2) 划分（确定性打乱，保证可复现）
        tr = float(split_ratio.get("train", 0.8))
        vr = float(split_ratio.get("val", 0.2))
        te = float(split_ratio.get("test", 0.0))
        total = tr + vr + te
        if total <= 0:
            tr, vr, te = 0.8, 0.2, 0.0
        else:
            tr, vr, te = tr / total, vr / total, te / total
        random.Random(42).shuffle(annotated)
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

        # 3) 清空旧划分产物并重建 images/{split} + labels/{split}（输出到 out_dir）
        #    注意：标准结构下 v1/images/train 等目录就是源图，重建会误删 → 先复制源图到 staging 过场
        staging = out_dir / ".seal_staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        for rec in annotated:
            shutil.copy2(rec["img_abs"], staging / f"{rec['image_id']}{rec['img_abs'].suffix.lower() or '.jpg'}")
        for d in ["train", "val", "test"]:
            for base in (images_dir, labels_dir):
                dd = base / d
                if dd.exists():
                    shutil.rmtree(dd, ignore_errors=True)
                dd.mkdir(parents=True, exist_ok=True)

        # data.yaml 类别名（英文码）提前确定：txt 的 class 索引需与 names 顺序一致
        names = self._resolve_seal_names(meta)
        _cls_idx = {str(n).lower(): i for i, n in enumerate(names)}

        counts = {}
        for split, recs in splits.items():
            counts[split] = 0
            for rec in recs:
                ext = rec["img_abs"].suffix.lower() or ".jpg"
                shutil.copy2(staging / f"{rec['image_id']}{ext}", images_dir / split / f"{rec['image_id']}{ext}")
                w = rec["width"] or 1
                h = rec["height"] or 1
                lines = []
                for b in rec["boxes"]:
                    x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    bw = max(x2 - x1, 0.0) / w
                    bh = max(y2 - y1, 0.0) / h
                    cid = b["class_id"]
                    if isinstance(cid, int) or (isinstance(cid, str) and cid.isdigit()):
                        cls_id = int(cid)  # 旧数据数字索引直接使用
                    else:
                        cls_id = _cls_idx.get(str(cid).lower(), 0)  # JSON 英文名 → YOLO 索引
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                (labels_dir / split / f"{rec['image_id']}.txt").write_text(
                    "\n".join(lines), encoding="utf-8"
                )
                counts[split] += 1

        # 4) 生成 data.yaml（names 用模型标签字典英文码，缺省回退 meta.classes；已在导出前确定）
        yaml_data = {
            "path": str(out_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(names),
            "names": names,
        }
        if counts["test"] > 0:
            yaml_data["test"] = "images/test"
        _save_yaml(out_dir / "data.yaml", yaml_data)

        # 5) 校验最终 txt 产物（2.1 第一级），结果并入封板结果
        try:
            label_validation = self._validate_labels(labels_dir, len(names))
        except Exception:
            label_validation = None

        # 清掉源图过场目录（图片已复制到划分产物）
        shutil.rmtree(staging, ignore_errors=True)

        logger.info(f"seal artifacts: total={n} splits={counts} nc={len(names)}")
        return {
            "total": n,
            "counts": counts,
            "ratios": {"train": round(tr, 3), "val": round(vr, 3), "test": round(te, 3)},
            "nc": len(names),
            "names": names,
            "label_validation": label_validation,
            "sample_type_stats": sample_stats,
        }

    async def list_datasets(self):
        """列出所有数据集"""
        def _list_datasets_sync():
            datasets = []
            if not self.datasets_dir.exists():
                return datasets
            for dataset_dir in self.datasets_dir.iterdir():
                if dataset_dir.is_dir():
                    meta_path = dataset_dir / "meta.json"
                    if meta_path.exists():
                        try:
                            meta = _load_json(meta_path)
                            # 占位类别名回填为真实名（修复历史遗留的 class_N 占位）
                            real_classes = self._resolve_real_classes(dataset_dir, meta)
                            if real_classes != meta.get("classes"):
                                meta["classes"] = real_classes
                            # 状态机字段回填（兼容旧数据）
                            _ensure_stage_fields(meta)
                            # 标记是否为标注导出的数据集（是否有可用的训练标签）
                            meta["annotated"] = self._is_annotated_exported(
                                dataset_dir, meta.get("version", "v1")
                            )
                            datasets.append(meta)
                        except Exception:
                            continue
            return datasets
        
        datasets = await asyncio.to_thread(_list_datasets_sync)
        return {
            "datasets": sorted(datasets, key=lambda x: x.get("uploaded_at", ""), reverse=True),
            # 封板数量门槛（#6）：前端列表展示「待封板 X / 门槛 N 张」进度
            "seal_min_images": settings.SEAL_MIN_IMAGES,
        }

    async def bind_model(self, dataset_id: str, model_id: str):
        """绑定数据集到模型（1.7 模型仓库：数据归属模型，合并/采集复用）

        绑定成功后数据集全部图片自动流入模型消息队列（阶段D），
        队列达到阈值或定时后自动打包进入标注页。
        """
        dataset_dir = self.datasets_dir / dataset_id
        meta_path = dataset_dir / "meta.json"
        if not await asyncio.to_thread(lambda: meta_path.exists()):
            raise ValueError(f"Dataset {dataset_id} not found")

        def _bind_sync():
            meta = _load_json(meta_path)
            meta["model_id"] = model_id
            # 绑定后补类别：数据集 classes 为空（纯图片包）时，从模型标签字典回填，
            # 并同步写 classes.txt 到版本目录（重新 prepare 时可直接读到真实类别）
            if not meta.get("classes"):
                labels = self._labels_from_model(model_id)
                if labels:
                    meta["classes"] = labels
                    try:
                        version_dir = dataset_dir / meta.get("version", "v1")
                        if version_dir.exists():
                            (version_dir / "classes.txt").write_text(
                                "\n".join(labels) + "\n", encoding="utf-8"
                            )
                            logger.info(f"bind {dataset_id}->{model_id}: classes.txt written: {labels}")
                    except Exception:
                        pass
            _save_json(meta_path, meta)
            return meta

        meta = await asyncio.to_thread(_bind_sync)

        # 阶段D：数据流入模型消息队列（阈值/定时自动打包进入标注页）
        try:
            from src.services.queue_service import ModelQueueService
            await ModelQueueService().enqueue_dataset(
                model_id, dataset_id, meta.get("version", "v1")
            )
        except Exception as e:
            logger.warning(f"bind_model {dataset_id}->{model_id}: enqueue failed: {e}")

        return meta
    
    async def get_dataset(self, dataset_id: str):
        """获取数据集详情"""
        dataset_dir = self.datasets_dir / dataset_id
        meta_path = dataset_dir / "meta.json"
        
        if not await asyncio.to_thread(lambda: meta_path.exists()):
            return None
        
        meta = await asyncio.to_thread(_load_json, meta_path)
        # 状态机字段回填（兼容旧数据）
        _ensure_stage_fields(meta)
        
        # 占位类别名回填为真实名（与 list_datasets 保持一致）
        real_classes = self._resolve_real_classes(dataset_dir, meta)
        if real_classes != meta.get("classes"):
            meta["classes"] = real_classes
        
        # 添加图片列表（在线程中执行）
        def _get_images():
            version = meta.get("version", "v1")
            version_dir = dataset_dir / version
            images_dir = self._find_images_dir(version_dir)
            
            images = []
            if images_dir:
                # 相对 DATA_DIR 的路径，前端用 /static/{path} 即可访问
                image_files = (
                    list(images_dir.rglob("*.jpg")) +
                    list(images_dir.rglob("*.jpeg")) +
                    list(images_dir.rglob("*.png"))
                )
                for img in sorted(image_files)[:6]:  # 预览图片不宜过多，最多 6 张避免杂乱
                    try:
                        rel = img.relative_to(settings.DATA_DIR)
                        images.append(str(rel).replace("\\", "/"))
                    except ValueError:
                        continue
            return images
        
        meta["images"] = await asyncio.to_thread(_get_images)
        return meta
    
    async def get_dataset_tree(self, dataset_id: str):
        """获取数据集目录树（用于前端展示文件夹结构）"""
        dataset_dir = self.datasets_dir / dataset_id
        if not await asyncio.to_thread(lambda: dataset_dir.exists()):
            return None
        tree = await asyncio.to_thread(
            build_tree, dataset_dir, settings.DATA_DIR, 6, 300, True
        )
        return {"tree": tree, "dataset_id": dataset_id}
    
    async def update_dataset(self, dataset_id: str, request):
        """更新数据集信息（封板后只读，禁止修改）"""
        dataset_dir = self.datasets_dir / dataset_id
        meta_path = dataset_dir / "meta.json"
        
        if not await asyncio.to_thread(lambda: meta_path.exists()):
            return None
        
        meta = await asyncio.to_thread(_load_json, meta_path)
        _ensure_stage_fields(meta)
        if meta["stage"] == STAGE_SEALED:
            raise ValueError("数据集已封板（只读），不可再修改；如需修改请删除后重建")
        
        # 更新字段
        if request.description is not None:
            meta["description"] = request.description
        if request.tags is not None:
            meta["tags"] = request.tags
        
        meta["updated_at"] = datetime.now().isoformat()
        
        await asyncio.to_thread(_save_json, meta_path, meta)
        
        return meta
    
    async def seal_dataset(self, dataset_id: str, force: bool = False, split_ratio: dict = None) -> dict:
        """封板：先标注后封板，封板 = 数量✓ + 标注✓ + 校验✓（数据链路阶段E）
        
        - 标注✓ 判定：每图 JSON 标注（annotations/*.json）有框，或旧数据 labels/*.txt 存在；
          未标注时默认拒绝封板，force=True 提供时间窗口兜底（强制封板，跳过产物生成）
        - 封板瞬间完成三件事：train/val/test 划分 + 每图 JSON→YOLO txt + 生成 data.yaml（names=标签字典英文码）
        - 封板后只读：stage=sealed，sealed_at 落库；版本号沿用当前版本（v1 只读快照）
        """
        dataset_dir = self.datasets_dir / dataset_id
        meta_path = dataset_dir / "meta.json"
        if not await asyncio.to_thread(lambda: meta_path.exists()):
            raise ValueError(f"Dataset {dataset_id} not found")

        meta = await asyncio.to_thread(_load_json, meta_path)
        _ensure_stage_fields(meta)

        if meta["stage"] == STAGE_SEALED:
            return meta  # 幂等：已封板直接返回当前状态

        if meta.get("status") != "prepared":
            raise ValueError("数据集尚未准备（需先执行准备操作），无法封板")

        # 标注✓判定：优先每图 JSON 标注，其次旧数据 labels/ txt / meta 标记
        version_dir = dataset_dir / meta.get("version", "v1")
        json_annotated = await asyncio.to_thread(self._count_json_annotations, version_dir)
        annotated = (
            json_annotated > 0
            or meta.get("has_labels", False)
            or bool(meta.get("label_count"))
            or await asyncio.to_thread(self._is_annotated_exported, dataset_dir, meta.get("version", "v1"))
        )
        if not annotated and not force:
            raise ValueError(
                "标注不完整：该数据集暂无可用标注（需先完成人工/AI标注，"
                "或使用强制封板作为时间窗口兜底）"
            )

        # 校验✓（2.1）：标注格式校验不通过时拒绝封板（force 兜底）
        label_validation = meta.get("label_validation")
        if label_validation is None and annotated:
            # 旧数据无校验记录：封板时实时补跑一次，判定并入封板结果
            labels_dir = self._find_labels_dir(version_dir)
            if labels_dir:
                label_validation = self._validate_labels(labels_dir, len(meta.get("classes") or []))
                meta["label_validation"] = label_validation
        if label_validation and label_validation.get("status") != "ok" and not force:
            first_err = (label_validation.get("errors") or ["未知错误"])[0]
            raise ValueError(
                f"标注格式校验未通过（{label_validation.get('invalid_lines', 0)} 行异常 / "
                f"{label_validation.get('invalid_files', 0)} 个文件）：{first_err}；"
                "请修复标注或使用强制封板兜底"
            )

        # 数量✓：封板最低图片数门槛（settings.SEAL_MIN_IMAGES，force 放行；旧数据无 image_count 不拦截）
        min_imgs = settings.SEAL_MIN_IMAGES
        img_n = int(meta.get("image_count") or 0)
        if min_imgs > 0 and img_n > 0 and img_n < min_imgs and not force:
            raise ValueError(
                f"数量不足：当前 {img_n} 张，封板最低 {min_imgs} 张"
                f"（差 {min_imgs - img_n} 张，或使用强制封板兜底）"
            )

        # 封板瞬间：划分 + 转 YOLO txt + 生成 data.yaml（无标注的 force 兜底仅打标，跳过产物生成）
        # 数据血缘（方案A）：每个封板版本输出为 sealed/v{N} 只读快照，版本号逐次递增（v1→v2…），
        # 不覆盖 v1 源版本目录；display_name 按 {模型code}_v{N} 自动命名，追踪血缘便于聚合续训
        model_code = meta.get("model_id") or ""
        if model_code:
            try:
                mcfg = _load_json(settings.REGISTRY_DIR / model_code / "model.json")
                model_code = mcfg.get("code") or model_code
            except Exception:
                pass
        seal_payload = None
        if annotated:
            ratio = split_ratio or meta.get("split_ratio") or {"train": 0.8, "val": 0.2, "test": 0.0}
            sv = int(meta.get("sealed_version_count") or 0) + 1
            seal_out = dataset_dir / "sealed" / f"v{sv}"
            await asyncio.to_thread(shutil.rmtree, seal_out, ignore_errors=True)
            seal_payload = await asyncio.to_thread(
                self._build_training_artifacts, version_dir, meta, ratio, seal_out
            )
            meta["sealed_version_count"] = sv
            meta["sealed_version"] = f"v{sv}"
            meta["sealed_out"] = str(seal_out.absolute())
            meta["parent_dataset_id"] = meta.get("parent_dataset_id") or dataset_id
            meta["display_name"] = f"{model_code}_v{sv}" if model_code else f"ds_v{sv}"
            meta["sealed_split"] = seal_payload
            meta["label_validation"] = seal_payload.get("label_validation")
            if seal_payload.get("counts"):
                meta["split_counts"] = seal_payload["counts"]
            if seal_payload.get("sample_type_stats"):
                meta["sample_type_stats"] = seal_payload["sample_type_stats"]

        meta["stage"] = STAGE_SEALED
        meta["sealed_at"] = datetime.now().isoformat()
        meta["sealed_version"] = meta.get("sealed_version", meta.get("version", "v1"))
        await asyncio.to_thread(_save_json, meta_path, meta)

        logger.info(f"dataset {dataset_id} sealed: has_labels={annotated} force={force} split={seal_payload and seal_payload.get('counts')}")
        return meta
    
    async def validate_dataset(self, dataset_id: str) -> dict:
        """重新执行标注格式校验（2.1），结果写入 meta 并返回（前端手动触发）"""
        dataset_dir = self.datasets_dir / dataset_id
        meta_path = dataset_dir / "meta.json"
        if not await asyncio.to_thread(lambda: meta_path.exists()):
            raise ValueError(f"Dataset {dataset_id} not found")

        def _run():
            meta = _load_json(meta_path)
            version = meta.get("version", "v1")
            labels_dir = self._find_labels_dir(dataset_dir / version)
            if not labels_dir:
                # txt 已转换删除（一图一 JSON 为准），无 txt 可校验 → 视为通过
                validation = {
                    "status": "ok", "checked_files": 0, "lines_checked": 0,
                    "invalid_files": 0, "invalid_lines": 0, "errors": [],
                    "note": "labels 已并入每图 JSON（txt_to_json），无 txt 需校验",
                }
                meta["label_validation"] = validation
                _save_json(meta_path, meta)
                return validation
            validation = self._validate_labels(labels_dir, len(meta.get("classes") or []))
            meta["label_validation"] = validation
            _save_json(meta_path, meta)
            return validation

        return await asyncio.to_thread(_run)

    async def delete_dataset(self, dataset_id: str):
        """删除数据集"""
        dataset_dir = self.datasets_dir / dataset_id
        
        if not await asyncio.to_thread(lambda: dataset_dir.exists()):
            return None
        
        # 删除上传的压缩包（兼容旧版本固定 .zip 命名）：先读 meta 再删目录
        meta_path = dataset_dir / "meta.json"
        archive_name = None
        if meta_path.exists():
            try:
                archive_name = _load_json(meta_path).get("archive")
            except Exception:
                pass

        # 删除数据集目录
        await asyncio.to_thread(_delete_directory, dataset_dir)

        candidates = []
        if archive_name:
            candidates.append(self.uploads_dir / archive_name)
        candidates.append(self.uploads_dir / f"{dataset_id}.zip")
        for p in candidates:
            await asyncio.to_thread(_delete_file, p)
        
        return {"ok": True, "message": f"Dataset {dataset_id} deleted"}
    
    async def export_annotated_dataset(self, dataset_id: str, version: str = "v1"):
        """导出标注后的数据集"""
        dataset_dir = self.datasets_dir / dataset_id
        version_dir = dataset_dir / version
        
        if not await asyncio.to_thread(lambda: version_dir.exists()):
            return None
        
        # 创建临时ZIP文件
        def _create_zip():
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            temp_zip_path = temp_zip.name
            temp_zip.close()
            
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                images_dir = self._find_images_dir(version_dir)
                labels_dir = self._find_labels_dir(version_dir)

                # 判断是否已有 train/val/test 划分结构
                split_dirs = ["train", "val", "test"] if images_dir else []
                has_split = images_dir is not None and any(
                    (images_dir / s).exists() for s in split_dirs
                )

                if has_split:
                    # 已划分：打包为 {split}/images 与 {split}/labels 结构
                    for split in split_dirs:
                        split_img = images_dir / split
                        if split_img.exists():
                            for f in sorted(split_img.rglob("*")):
                                if f.is_file():
                                    zipf.write(f, f"{split}/images/{f.name}")
                        split_lbl = labels_dir / split if labels_dir else None
                        if split_lbl is not None and split_lbl.exists():
                            for f in sorted(split_lbl.rglob("*")):
                                if f.is_file():
                                    zipf.write(f, f"{split}/labels/{f.name}")
                    # 生成匹配 zip 结构的 data.yaml（train/val/test 下都有 images 和 labels）
                    zip_yaml = {"path": ""}
                    for split in split_dirs:
                        if (images_dir / split).exists():
                            zip_yaml[split] = f"{split}/images"
                    src_yaml = version_dir / "data.yaml"
                    if src_yaml.exists():
                        try:
                            with open(src_yaml, "r", encoding="utf-8") as _f:
                                src = yaml.safe_load(_f) or {}
                            zip_yaml["nc"] = src.get("nc", 0)
                            zip_yaml["names"] = src.get("names", [])
                        except Exception:
                            pass
                    zipf.writestr("data.yaml", yaml.dump(zip_yaml, allow_unicode=True))
                else:
                    # 未划分：平铺打包 images/ 与 labels/
                    if images_dir:
                        for img_file in images_dir.rglob("*"):
                            if img_file.is_file():
                                arcname = img_file.relative_to(version_dir)
                                zipf.write(img_file, arcname)
                    if labels_dir:
                        for label_file in labels_dir.rglob("*"):
                            if label_file.is_file():
                                arcname = label_file.relative_to(version_dir)
                                zipf.write(label_file, arcname)
                    # 未划分时 data.yaml 保持原样
                    data_yaml = version_dir / "data.yaml"
                    if data_yaml.exists():
                        zipf.write(data_yaml, "data.yaml")
            
            return temp_zip_path
        
        zip_path = await asyncio.to_thread(_create_zip)
        return zip_path
    
    async def export_original_dataset(self, dataset_id: str, version: str = "v1"):
        """导出标注前的数据集（仅图片）"""
        dataset_dir = self.datasets_dir / dataset_id
        version_dir = dataset_dir / version
        
        if not await asyncio.to_thread(lambda: version_dir.exists()):
            return None
        
        # 创建临时ZIP文件
        def _create_zip():
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            temp_zip_path = temp_zip.name
            temp_zip.close()
            
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 仅添加images目录
                images_dir = self._find_images_dir(version_dir)
                if images_dir:
                    for img_file in images_dir.rglob("*"):
                        if img_file.is_file():
                            arcname = img_file.relative_to(version_dir)
                            zipf.write(img_file, arcname)
                
                # 可选：添加data.yaml（如果存在）
                data_yaml = version_dir / "data.yaml"
                if data_yaml.exists():
                    zipf.write(data_yaml, "data.yaml")
            
            return temp_zip_path
        
        zip_path = await asyncio.to_thread(_create_zip)
        return zip_path

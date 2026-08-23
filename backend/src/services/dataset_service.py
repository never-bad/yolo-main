import json
import zipfile
import tarfile
import shutil
import asyncio
import time
import threading
import concurrent.futures
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

# 桶灌流程互斥：防止并发请求重复灌桶/封板（模块级：多个服务实例共享同一把锁）
_BUCKET_LOCK = threading.Lock()

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


# ===== 后台删除队列（MLOps 优化：删除接口秒回，磁盘清理不阻塞请求）=====
# 专用单线程工作线程：删除任务逐个排队执行，
# 不占 asyncio.to_thread 的公共线程池，也不阻塞事件循环 / API 请求
_DELETE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="dataset-deleter"
)
_del_swept = False


def _rmtree_retry(path: Path, attempts: int = 3):
    """删除目录；Windows 上被占用 / Defender 实时扫描导致失败时重试"""
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(0.5)


def _delete_file_retry(path: Path, attempts: int = 3):
    """删除文件；失败重试（跳过不存在的文件）"""
    for i in range(attempts):
        try:
            if path.exists():
                path.unlink()
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(0.3)


def _rename_dir(src: Path, dst: Path) -> bool:
    """同卷改名（元数据操作，瞬时完成）；失败（文件被锁）返回 False"""
    try:
        src.rename(dst)
        return True
    except OSError:
        return False


def _sweep_stale_deletions(datasets_dir: Path):
    """清理上次进程崩溃遗留的「标记删除」目录（__deleting 后缀）"""
    if not datasets_dir.exists():
        return
    for p in datasets_dir.iterdir():
        if p.is_dir() and "__deleting" in p.name:
            try:
                _rmtree_retry(p)
            except Exception:
                pass


def _enqueue_deletion(paths: list):
    """把待删路径提交给专用后台删除线程，接口侧不做任何等待"""
    global _del_swept
    if not _del_swept:
        _del_swept = True
        try:
            _sweep_stale_deletions(settings.DATASETS_DIR)
        except Exception:
            pass

    def _job():
        for p in paths:
            try:
                if p.is_dir():
                    _rmtree_retry(p)
                else:
                    _delete_file_retry(p)
            except Exception:
                logger.warning(f"后台删除失败（将残留，可手动清理）: {p}")

    _DELETE_EXECUTOR.submit(_job)


class DatasetService:
    def __init__(self):
        self.datasets_dir = settings.DATASETS_DIR
        self.uploads_dir = settings.UPLOADS_DIR
        self._bucket_state = None  # advance 期间持有的桶账本（跨 _seal_receiving_bucket 使用）

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
        
        # 解压数据集（准备语义=从 zip 重新构建，先清空数据子目录旧内容，
        # 避免重复准备时与上一次解压内容叠加，例如 zip 顶层目录残留 + 双份图片）
        # 数据集一律为 datasets/{dataset_id}（无 version 子目录）
        version = "v1"
        version_dir = dataset_dir
        if await asyncio.to_thread(lambda: version_dir.exists()):
            def clear_dir():
                for child in list(version_dir.iterdir()):
                    if child.name == ".staging":
                        continue
                    if child.name == "meta.json":
                        continue  # 元数据不清空（zip 解压产物不含 meta.json）
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

        # 标签字典自动生成：prepare 解析出真实类名后，归属模型的标签字典若为空
        # （动态建模型的空字典），自动用包内类名初始化；有人工维护的字典不覆盖
        if model_id_ds and detected_classes:
            try:
                from src.services.model_service import ModelService
                wrote = await asyncio.to_thread(
                    ModelService().init_labels_from_classes, model_id_ds, detected_classes
                )
                if wrote:
                    logger.info(
                        f"prepare {dataset_id}: model {model_id_ds} labels dict "
                        f"auto-generated: {detected_classes}"
                    )
            except Exception as e:
                logger.warning(f"prepare {dataset_id}: auto-init model labels failed: {e}")

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
                # 仅排除可视化预览图（xxx_result/predict/detect.jpg）：
                # 要匹配文件名末尾的标记，而不是全名子串 —— 否则
                # "*_detection_*.jpg" 这类正常数据集图片会被误过滤
                if any(p.stem.lower().endswith(k) for k in EXCLUDE_KEYWORDS):
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
        version_dir = dataset_dir
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
        if not dataset_dir.exists():
            return False
        # 标注导出固定写入 dataset_dir/labels/{train,val,test}
        labels = dataset_dir / "labels"
        train_labels = labels / "train"
        if not train_labels.is_dir():
            return False
        try:
            return any(train_labels.iterdir())
        except Exception:
            return False

    def _collect_legacy_txt_annotations(self, dataset_dir: Path) -> list:
        """旧数据兜底：从 images/ + labels/ txt 收集标注（未走每图 JSON 的历史数据集）"""
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
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

    def _import_labels_txt_to_json(self, dataset_dir: Path, class_map: dict = None) -> dict:
        """把 YOLO labels/*.txt 标注转换为每图 JSON（一图一 JSON 为权威标注源）。

        复用旧数据兜底解析（归一化→像素框 + PIL 读宽高），逐图写
        datasets/{id}/annotations/{image_id}.json；
        class_map 非空时按目标模型做 class_id 重映射，非本模型类（未命中）剔除；
        返回 {"converted": n, "dropped_boxes": m}。源 labels 目录由调用方决定保留。
        """
        ann_dir = dataset_dir / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        converted = 0
        dropped_boxes = 0
        for rec in self._collect_legacy_txt_annotations(dataset_dir):
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

    def _count_json_annotations(self, dataset_dir: Path) -> int:
        """统计每图 JSON 标注中有框的图片数（数据链路新标注数据源）"""
        ann_dir = dataset_dir / "annotations"
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

    def _count_annotated_files(self, dataset_dir: Path) -> int:
        """统计已标注过的图片数：存在标注 JSON 即算（空框=人工确认无目标/AI 困难样本，
        同样视为已标注）。与 _count_json_annotations（有框才算）互补：
        完成度判定用本函数，避免无目标图导致"永远标注中"。"""
        ann_dir = dataset_dir / "annotations"
        if not ann_dir.exists():
            return 0
        return len(list(ann_dir.glob("*.json")))

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

    def _build_training_artifacts(self, dataset_dir: Path, meta: dict, split_ratio: dict, out_dir: Path = None) -> dict:
        """封板一次性产出：划分 train/val/test + 每图 JSON → YOLO txt + 生成 data.yaml（数据链路阶段E）

        标注数据源：dataset_dir/annotations/{image_id}.json（像素坐标框）
        图片源：dataset_dir/images/ 平铺目录
        out_dir：产物输出目录（缺省 = dataset_dir 原地）。数据血缘方案 A 下每个封板
        版本输出到数据集下 sealed/v{N}/ 只读快照，旧版本不被覆盖、版本号逐次递增。
        data.yaml 类别名：模型标签字典英文码（缺省回退 meta.classes）
        返回 {total, counts, ratios, nc, names}；无已标注图片时抛 ValueError。
        """
        out_dir = out_dir or dataset_dir
        ann_dir = dataset_dir / "annotations"
        src_images_dir = dataset_dir / "images"
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
            annotated = self._collect_legacy_txt_annotations(dataset_dir)
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
        #    注意：标准结构下 images/train 等目录就是源图，重建会误删 → 先复制源图到 staging 过场
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
                # 跳过「内部/备份」目录（_ 前缀，如 _archived_*）与「标记删除」的目录
                if (
                    dataset_dir.is_dir()
                    and not dataset_dir.name.startswith("_")
                    and "__deleting" not in dataset_dir.name
                ):
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
                            meta["annotated"] = (
                                meta.get("stage") == STAGE_SEALED
                                or self._is_annotated_exported(
                                    dataset_dir, meta.get("version", "v1")
                                )
                            )
                            datasets.append(meta)
                        except Exception:
                            continue
                    else:
                        # 标准 YOLO 结构数据集：根目录 data.yaml + images/{train,val,test} + labels（无平台 meta.json），
                        # 直接识别为可训练的封板数据集
                        root_yaml = dataset_dir / "data.yaml"
                        if root_yaml.exists():
                            try:
                                cfg = yaml.safe_load(
                                    root_yaml.read_text(encoding="utf-8")
                                ) or {}
                            except Exception:
                                cfg = {}
                            names = cfg.get("names") or []
                            if isinstance(names, dict):
                                names = [
                                    names[k]
                                    for k in sorted(names, key=lambda x: int(x))
                                ]
                            counts = {}
                            for split in ("train", "val", "test"):
                                p = dataset_dir / "images" / split
                                if not p.is_dir():
                                    counts[split] = 0
                                    continue
                                # 用 scandir 计数而非 listdir+is_file：
                                # Windows 下含数千文件的目录逐文件 stat 极慢（实测 ~140 倍差距）
                                counts[split] = sum(1 for e in os.scandir(p) if e.is_file())
                            stats = {k: v for k, v in counts.items() if v}
                            meta = {
                                "dataset_id": dataset_dir.name,
                                "display_name": dataset_dir.name,
                                "filename": dataset_dir.name,
                                "size": 0,
                                "uploaded_at": "",
                                "status": "prepared",
                                "stage": STAGE_SEALED,
                                "training_status": "incomplete",
                                "version": "v1",
                                "image_count": sum(counts.values()),
                                "label_count": sum(counts.values()),
                                "has_labels": True,
                                "classes": names,
                                "annotated": True,
                                "standard_yolo": True,
                                "split_counts": stats,
                            }
                            datasets.append(meta)
            return datasets
        
        datasets = await asyncio.to_thread(_list_datasets_sync)
        # 桶灌进度（B 系列）：每个模型当前接收桶累计量 + 已封板版本，供前端展示
        def _bucket_summary():
            out = []
            try:
                state = self._load_bucket_state()
            except Exception:
                return out
            for mid, mst in (state.get("models") or {}).items():
                out.append({
                    "model_id": mid,
                    "receiving": mst.get("receiving"),
                    "sealed": mst.get("sealed") or [],
                })
            out.sort(key=lambda x: x["model_id"])
            return out
        return {
            "datasets": sorted(datasets, key=lambda x: x.get("uploaded_at", ""), reverse=True),
            # 封板数量门槛（#6）：前端列表展示「待封板 X / 门槛 N 张」进度
            "seal_min_images": settings.SEAL_MIN_IMAGES,
            # B 系列：按模型滚动灌桶进度（接收桶累计量 / 已封板版本列表）
            "buckets": _bucket_summary(),
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
                        # 数据集一律为 datasets/{dataset_id}（无 version 子目录）
                        if dataset_dir.exists():
                            (dataset_dir / "classes.txt").write_text(
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
            # 数据集一律为 datasets/{dataset_id}（无 version 子目录）
            images_dir = self._find_images_dir(dataset_dir)
            
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
    
    @staticmethod
    def _seal_due(prepared_at: str) -> bool:
        """P1 封板时间窗口兜底：prepared_at 起超过 SEAL_MAX_DAYS 天即视为到期（数量+时间先达者）。"""
        max_days = settings.SEAL_MAX_DAYS
        if max_days <= 0 or not prepared_at:
            return False
        try:
            from datetime import timedelta
            prepared_dt = datetime.fromisoformat(prepared_at)
            return datetime.now() - prepared_dt >= timedelta(days=max_days)
        except Exception:
            return False

    async def seal_dataset(self, dataset_id: str, force: bool = False, split_ratio: dict = None) -> dict:
        """封板：先标注后封板，封板 = 数量✓ + 标注✓ + 校验✓（数据链路阶段E）
        
        - 标注✓ 判定：每图 JSON 标注（annotations/*.json）有框，或旧数据 labels/*.txt 存在；
          未标注时默认拒绝封板，force=True 提供时间窗口兜底（强制封板，跳过产物生成）
        - P1 时间窗口兜底：数量未达门槛但 prepared 超过 SEAL_MAX_DAYS 天也可自动封板
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

        # 标注✓判定：优先每图 JSON 标注（数据集一律为 datasets/{dataset_id}，无 version 子目录）
        json_annotated = await asyncio.to_thread(self._count_json_annotations, dataset_dir)
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
            labels_dir = self._find_labels_dir(dataset_dir)
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
        # P1 时间窗口兜底：prepared_at 起超过 SEAL_MAX_DAYS 天即视为达标（数量与时间先达者生效）
        prepared_at = meta.get("prepared_at") or ""
        seal_deadline_due = bool(settings.SEAL_MAX_DAYS) and bool(prepared_at) and self._seal_due(prepared_at)
        if min_imgs > 0 and img_n > 0 and img_n < min_imgs and not force and not seal_deadline_due:
            raise ValueError(
                f"数量不足：当前 {img_n} 张，封板最低 {min_imgs} 张"
                f"（差 {min_imgs - img_n} 张，或使用强制封板/等待时间窗口 {settings.SEAL_MAX_DAYS} 天兜底）"
            )

        # 封板瞬间：划分 + 转 YOLO txt + 生成 data.yaml（无标注的 force 兜底仅打标，跳过产物生成）
        # 数据血缘（方案A）：每个封板版本输出为 sealed/v{N} 只读快照，版本号逐次递增（v1→v2…），
        # 不覆盖 v1 源版本目录；display_name 按 {模型code}_{封板时间戳}_v{N} 自动命名（追踪血缘，含批次时序）
        model_code = meta.get("model_id") or ""
        if model_code:
            try:
                mcfg = _load_json(settings.REGISTRY_DIR / model_code / "model.json")
                model_code = mcfg.get("model_code") or model_code
            except Exception:
                pass
        seal_payload = None
        if annotated:
            ratio = split_ratio or meta.get("split_ratio") or {"train": 0.8, "val": 0.2, "test": 0.0}
            sv = int(meta.get("sealed_version_count") or 0) + 1
            seal_out = dataset_dir / "sealed" / f"v{sv}"
            await asyncio.to_thread(shutil.rmtree, seal_out, ignore_errors=True)
            seal_payload = await asyncio.to_thread(
                self._build_training_artifacts, dataset_dir, meta, ratio, seal_out
            )
            meta["sealed_version_count"] = sv
            meta["sealed_version"] = f"v{sv}"
            meta["sealed_out"] = str(seal_out.absolute())
            meta["parent_dataset_id"] = meta.get("parent_dataset_id") or dataset_id
            seal_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            meta["display_name"] = f"{model_code}_{seal_ts}_v{sv}" if model_code else f"ds_v{sv}"
            meta["sealed_split"] = seal_payload
            meta["label_validation"] = seal_payload.get("label_validation")
            if seal_payload.get("counts"):
                meta["split_counts"] = seal_payload["counts"]
            if seal_payload.get("sample_type_stats"):
                meta["sample_type_stats"] = seal_payload["sample_type_stats"]

        meta["stage"] = STAGE_SEALED
        meta["sealed_at"] = datetime.now().isoformat()
        meta["sealed_version"] = meta.get("sealed_version", meta.get("version", "v1"))
        # 名=id：未生成显示名（force 无产物分支）也按命名规则补齐
        if not meta.get("display_name"):
            seal_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            meta["display_name"] = (
                f"{model_code}_{seal_ts}_v{int(meta.get('sealed_version_count') or 1)}"
                if model_code
                else f"ds_v{int(meta.get('sealed_version_count') or 1)}"
            )
        await asyncio.to_thread(_save_json, meta_path, meta)

        # 封板即更名：dataset_id（目录/任务绑定/PG/桶账本）整体迁到 display_name（名=id）
        new_id = meta.get("display_name")
        if new_id and new_id != dataset_id:
            meta = await self._rename_dataset_identity(dataset_id, new_id)
            logger.info(f"dataset {dataset_id} renamed -> {new_id} on seal")

        logger.info(f"dataset {dataset_id} sealed: has_labels={annotated} force={force} split={seal_payload and seal_payload.get('counts')}")
        return meta

    async def _rename_dataset_identity(self, old_id: str, new_id: str) -> dict:
        """封板更名（名=id）：把数据集唯一标识从 ds_xxx 整体迁移到 {code}_{时间戳}_v{N}。
        同步迁移：磁盘目录、meta.dataset_id、标注任务绑定、上传包、桶账本、PG 影子库。
        返回更名后的 meta。
        """
        new_meta = await asyncio.to_thread(self._rename_dataset_identity_sync, old_id, new_id)
        try:
            from src.db import meta_store

            await meta_store.upsert_dataset(new_meta)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"rename {old_id}: pg upsert failed: {e}")
        return new_meta

    def _rename_dataset_identity_sync(self, old_id: str, new_id: str) -> dict:
        old_dir = self.datasets_dir / old_id
        new_dir = self.datasets_dir / new_id
        if not old_dir.exists():
            raise ValueError(f"数据集 {old_id} 不存在")
        if new_id == old_id:
            return _load_json(old_dir / "meta.json")
        if new_dir.exists():
            raise ValueError(f"目标数据集 {new_id} 已存在，无法更名")

        os.rename(str(old_dir), str(new_dir))

        # 1) meta.dataset_id 更新（新目录下的 meta.json）
        mp = new_dir / "meta.json"
        meta = _load_json(mp) if mp.exists() else {}
        meta["dataset_id"] = new_id
        _save_json(mp, meta)

        # 2) 同源上传包跟随（{old}.zip → {new}.zip）
        old_zip = settings.UPLOADS_DIR / f"{old_id}.zip"
        new_zip = settings.UPLOADS_DIR / f"{new_id}.zip"
        if old_zip.exists() and not new_zip.exists():
            os.rename(str(old_zip), str(new_zip))

        # 3) 标注任务绑定跟随（task.json 里的 dataset_id）
        if settings.ANNOTATIONS_DIR.exists():
            for tdir in settings.ANNOTATIONS_DIR.iterdir():
                tf = tdir / "task.json"
                if tf.exists() and not tdir.name.startswith("."):
                    try:
                        m = _load_json(tf)
                        if m.get("dataset_id") == old_id:
                            m["dataset_id"] = new_id
                            _save_json(tf, m)
                    except Exception:  # noqa: BLE001
                        pass

        # 4) 桶账本引用跟随（receiving.members / sealed 快照对 dataset_id 的引用）
        try:
            bf = settings.DATA_DIR / "buckets.json"
            if bf.exists():
                b = _load_json(bf)
                changed = False
                for mm in (b.get("models") or {}).values():
                    recv = mm.get("receiving") or {}
                    for member in recv.get("members") or []:
                        if member.get("dataset_id") == old_id:
                            member["dataset_id"] = new_id
                            changed = True
                    for snap in mm.get("sealed") or []:
                        if isinstance(snap, dict) and snap.get("dataset_id") == old_id:
                            snap["dataset_id"] = new_id
                            changed = True
                if changed:
                    _save_json(bf, b)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"rename {old_id}: bucket ledger update failed: {e}")

        logger.info(f"dataset identity renamed: {old_id} -> {new_id}")
        return meta
    
    async def validate_dataset(self, dataset_id: str) -> dict:
        """重新执行标注格式校验（2.1），结果写入 meta 并返回（前端手动触发）"""
        dataset_dir = self.datasets_dir / dataset_id
        meta_path = dataset_dir / "meta.json"
        if not await asyncio.to_thread(lambda: meta_path.exists()):
            raise ValueError(f"Dataset {dataset_id} not found")

        def _run():
            meta = _load_json(meta_path)
            # 数据集一律为 datasets/{dataset_id}（无 version 子目录）
            labels_dir = self._find_labels_dir(dataset_dir)
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
        """删除数据集：目录改名标记 + 专用后台线程删磁盘，接口立即返回"""
        dataset_dir = self.datasets_dir / dataset_id

        if not await asyncio.to_thread(lambda: dataset_dir.exists()):
            return None

        # 读 meta 拿到上传压缩包名（先读，改名后目录名已变）
        meta_path = dataset_dir / "meta.json"
        archive_name = None
        if await asyncio.to_thread(lambda: meta_path.exists()):
            try:
                archive_name = await asyncio.to_thread(
                    lambda: _load_json(meta_path).get("archive")
                )
            except Exception:
                pass

        # 1) 同一卷内改名（瞬时元数据操作）→ 前端列表立刻看不到该数据集
        renamed = dataset_dir.with_name(f"{dataset_id}__deleting")
        renamed_ok = await asyncio.to_thread(_rename_dir, dataset_dir, renamed)

        # 2) 待删路径交给专用后台删除线程逐个处理，不等待
        targets = [renamed if renamed_ok else dataset_dir]
        if archive_name:
            targets.append(self.uploads_dir / archive_name)
        targets.append(self.uploads_dir / f"{dataset_id}.zip")
        _enqueue_deletion(targets)

        return {"ok": True, "message": f"Dataset {dataset_id} deleted"}

    async def delete_image(self, dataset_id: str, image_id: str):
        """删除数据集中的一张图片，同步清理其标注与同名标签文件。

        删除范围（同名字段全部清理，防止数据残留）：
        1. 图片文件（支持平铺 / train/val/test 划分，任意常见扩展名）；
        2. 每图 JSON 标注（annotations/{stem}.json）；
        3. 同名标签文件（labels 目录任意层级中的 {stem}.txt）；
        4. 预览缩略图（previews 下同名文件，best effort）。
        同时更新 meta 的 image_count。仅允许标注阶段且未灌桶/封板的数据集删除。
        """
        dataset_dir = self.datasets_dir / dataset_id
        meta_path = dataset_dir / "meta.json"

        def _sync():
            if not dataset_dir.exists():
                raise ValueError(f"数据集 {dataset_id} 不存在")
            meta = _load_json(meta_path) if meta_path.exists() else {}
            if meta.get("status") != "prepared":
                raise ValueError("数据集尚未准备（prepare），无法删除图片")
            if meta.get("stage") == STAGE_SEALED:
                raise ValueError("数据集已封板（只读），无法删除图片")
            if meta.get("bucket_archived"):
                raise ValueError("数据集已归档入桶，无法删除图片")
            if int(meta.get("bucket_pending") or 0) + int(meta.get("bucket_sealed") or 0) > 0:
                raise ValueError("数据集已部分入桶，无法删除图片（需先解除归档）")

            # 数据集一律为 datasets/{dataset_id}（无 version 子目录）
            stem = Path(image_id).stem
            image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

            # 1) 图片文件（含划分子目录 train/val/test）
            #    只按候选名探测，避免 rglob 全目录遍历（千张图片时可差百倍耗时）
            removed = 0
            images_dir = self._find_images_dir(dataset_dir)
            for p in self._iter_same_stem(images_dir, stem, image_suffixes):
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass

            # 2) 每图 JSON 标注
            ann = dataset_dir / "annotations" / f"{stem}.json"
            if ann.exists():
                try:
                    ann.unlink()
                except OSError:
                    pass

            # 3) 同名标签文件（labels 目录平铺/划分结构）
            labels_dir = self._find_labels_dir(dataset_dir)
            for p in self._iter_same_stem(labels_dir, stem, {".txt"}):
                try:
                    p.unlink()
                except OSError:
                    pass

            # 4) 预览缩略图（best effort）
            for p in self._iter_same_stem(
                dataset_dir / "previews", stem, image_suffixes | {".txt", ".json"}
            ):
                try:
                    p.unlink()
                except OSError:
                    pass

            if removed == 0:
                raise ValueError(f"图片 {image_id} 不存在于数据集 {dataset_id}")

            meta["image_count"] = max(0, int(meta.get("image_count") or 0) - 1)
            _save_json(meta_path, meta)
            logger.info(f"delete_image {dataset_id}: {stem} removed, image_count={meta['image_count']}")
            return {
                "deleted": True,
                "image_id": stem,
                "image_count": int(meta["image_count"]),
            }

        return await asyncio.to_thread(_sync)

    @staticmethod
    def _iter_same_stem(scan_dir, stem, suffixes):
        """按候选名探测与 stem 同名的文件（支持平铺 + 一层划分子目录），避免 rglob 全目录遍历。

        目录不存在返回空；顶层未命中时才枚举子目录（train/val/test 等划分结构）。
        glob 的 "." 是任意字符通配，必须用 p.stem 精确过滤（如 stem=xxx 时
        glob("xxx.*") 会命中 "xxx_aug1.jpg"，需排除）。
        """
        if not scan_dir or not scan_dir.is_dir():
            return []
        matched = [
            p
            for p in scan_dir.glob(f"{stem}.*")
            if p.is_file() and p.stem == stem and p.suffix.lower() in suffixes
        ]
        if not matched:
            for sub in scan_dir.iterdir():
                if sub.is_dir():
                    matched += [
                        p
                        for p in sub.glob(f"{stem}.*")
                        if p.is_file() and p.stem == stem and p.suffix.lower() in suffixes
                    ]
        return matched

    async def export_annotated_dataset(self, dataset_id: str, version: str = "v1"):
        """导出标注后的数据集"""
        # 获取数据集目录（数据集一律为 datasets/{dataset_id}，无 version 子目录）
        dataset_dir = self.datasets_dir / dataset_id
        # 导出标注后的数据集
        if not await asyncio.to_thread(lambda: dataset_dir.exists()):
            return None
        
        # 创建临时ZIP文件
        def _create_zip():
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            temp_zip_path = temp_zip.name
            temp_zip.close()
            
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                images_dir = self._find_images_dir(dataset_dir)
                labels_dir = self._find_labels_dir(dataset_dir)

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
                    src_yaml = dataset_dir / "data.yaml"
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
                                arcname = img_file.relative_to(dataset_dir)
                                zipf.write(img_file, arcname)
                    if labels_dir:
                        for label_file in labels_dir.rglob("*"):
                            if label_file.is_file():
                                arcname = label_file.relative_to(dataset_dir)
                                zipf.write(label_file, arcname)
                    # 未划分时 data.yaml 保持原样
                    data_yaml = dataset_dir / "data.yaml"
                    if data_yaml.exists():
                        zipf.write(data_yaml, "data.yaml")
            
            return temp_zip_path
        
        zip_path = await asyncio.to_thread(_create_zip)
        return zip_path

    # ================= 桶灌（B 系列）：按模型滚动灌桶，满 SEAL_MIN_IMAGES 自动封板 =================
    # 业务规则（与用户确认）：
    #   - A 包（标注前原料）与 B 桶（成品数据集）分离：A 包在 datasets/ds_xxx 标注，
    #     封板产物生成**独立成品数据集** datasets/ds_auto_{ts}（stage=sealed，命名 {模型code}_v{N}）
    #   - 一个 A 包全部标注完成（每图 JSON 有框）后，数据按 FIFO 灌入所属模型最小编号未满桶
    #   - 桶累计 >= SEAL_MIN_IMAGES(默认1500) 自动封板并产出 v{N} 快照；
    #     超出部分（最后一个包的剩余图）自动滚入下一桶 B{N+1}——即「A2 拆开补 B1，剩的进下一桶」
    #   - 只增不减：已灌入桶的数据不重排、不调出

    def _load_bucket_state(self) -> dict:
        try:
            return _load_json(settings.DATA_DIR / "buckets.json") or {"models": {}}
        except Exception:
            return {"models": {}}

    def _save_bucket_state(self, state: dict):
        _save_json(settings.DATA_DIR / "buckets.json", state)

    def _model_code_of(self, model_id: str) -> str:
        try:
            mcfg = _load_json(settings.REGISTRY_DIR / model_id / "model.json")
            return mcfg.get("model_code") or model_id
        except Exception:
            return model_id

    def _pick_images_for_seal(self, dataset_dir: Path, pkg_meta: dict, take: int) -> list:
        """从源包选择 take 张「已有框标注」且未封存过的图（文件名升序，FIFO 确定性）。
        返回记录列表：{image_id, img_abs, width, height, boxes, sample_type}
        """
        sealed_stems = set(pkg_meta.get("sealed_stems") or [])
        img_index = {}
        images_dir = dataset_dir / "images"
        if images_dir.exists():
            for s in ("train", "val", "test"):
                d = images_dir / s
                if d.is_dir():
                    for f in d.iterdir():
                        if f.is_file() and f.suffix.lower() in IMG_SUFFIX:
                            img_index.setdefault(f.stem, f)
            for f in images_dir.iterdir():
                if f.is_file() and f.suffix.lower() in IMG_SUFFIX:
                    img_index.setdefault(f.stem, f)
        records = []
        ann_dir = dataset_dir / "annotations"
        if not ann_dir.exists():
            return []
        for p in sorted(ann_dir.glob("*.json")):
            if len(records) >= take:
                break
            if p.stem in sealed_stems:
                continue
            try:
                data = _load_json(p)
            except Exception:
                continue
            boxes = data.get("boxes")
            if not boxes:
                continue
            img_abs = img_index.get(p.stem) or img_index.get(str(data.get("image_id", "")))
            if img_abs is None:
                continue
            records.append({
                "image_id": p.stem,
                "img_abs": img_abs,
                "width": data.get("width") or 1,
                "height": data.get("height") or 1,
                "boxes": boxes,
                "sample_type": data.get("sample_type"),
            })
        return records[:take]

    def _copy_records_flat(self, work: Path, picks: list):
        """把选中的多包图片/标注平铺到暂存目录（跨包文件重名 → 统一加序号前缀）"""
        wimg = work / "images"
        wann = work / "annotations"
        wimg.mkdir(parents=True, exist_ok=True)
        wann.mkdir(parents=True, exist_ok=True)
        idx = 0
        for dataset_id, records in picks:
            for rec in records:
                uniq = f"p{idx:04d}_{rec['image_id']}"
                ext = rec["img_abs"].suffix.lower() or ".jpg"
                shutil.copy2(rec["img_abs"], wimg / f"{uniq}{ext}")
                _save_json(wann / f"{uniq}.json", {
                    "image_id": uniq,
                    "width": rec["width"],
                    "height": rec["height"],
                    "boxes": rec["boxes"],
                    "sample_type": rec.get("sample_type"),
                    "ai_annotated": True,
                    "from_bucket": dataset_id,
                })
                idx += 1

    def _create_bucket_snapshot(self, model_id: str, mst: dict, picks: list,
                                fallback_classes: list) -> dict:
        """封板产物：独立成品数据集 datasets/ds_auto_{ts}（无 v1 层，直接
        images/{train,val,test} + labels/{train,val,test} + data.yaml + meta.json）。
        复用 _build_training_artifacts 生成划分与 YOLO txt。
        picks: [(dataset_id, [records])]，records 已按 FIFO 排好。
        """
        sv = int(mst.get("bucket_version_count") or 0) + 1
        # 名=id：桶封板快照目录直接以 {code}_{封板时间戳}_v{N} 命名（与手动封板同规则）
        code = self._model_code_of(model_id)
        seal_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{code}_{seal_ts}_v{sv}" if code else f"ds_v{sv}"
        ds_id = base
        n = 0
        while (self.datasets_dir / ds_id).exists():
            n += 1
            ds_id = f"{base}_{n}"
        ds_dir = self.datasets_dir / ds_id
        ds_dir.mkdir(parents=True, exist_ok=True)

        ratio = {"train": 0.8, "val": 0.2, "test": 0.0}
        names = self._resolve_seal_names({"model_id": model_id, "classes": fallback_classes}) or fallback_classes
        if not names:
            names = ["class_0"]

        work = Path(tempfile.mkdtemp(prefix="_bucket_", dir=str(settings.DATA_DIR)))
        try:
            self._copy_records_flat(work, picks)
            payload = self._build_training_artifacts(
                work, {"model_id": model_id, "classes": names}, ratio, ds_dir
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)

        code = self._model_code_of(model_id)  # noqa: F841（上面已取用）
        now = datetime.now().isoformat()
        seal_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta = {
            "dataset_id": ds_id,
            "filename": f"v{sv}",
            "archive": None,
            "size": 0,
            "uploaded_at": now,
            "status": "prepared",
            "stage": STAGE_SEALED,
            "training_status": TRAIN_STATUS_INCOMPLETE,
            "sealed_at": now,
            "trained_at": None,
            "last_trained_round": 0,
            "model_id": model_id,
            "image_count": payload.get("total", 0),
            "label_count": payload.get("total", 0),
            "has_labels": True,
            "version": "",
            "classes": names,
            "split_ratio": ratio,
            "split_counts": payload.get("counts"),
            "label_validation": payload.get("label_validation"),
            "sample_type_stats": payload.get("sample_type_stats"),
            "sealed_version": f"v{sv}",
            "sealed_version_count": 1,
            "sealed_out": str(ds_dir.absolute()),
            "display_name": f"{code}_{seal_ts}_v{sv}",
            "is_bucket_snapshot": True,
            "bucket_index": sv,
        }
        _save_json(ds_dir / "meta.json", meta)
        try:
            (ds_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        except Exception:
            pass
        logger.info(
            f"bucket seal {model_id}: v{sv} -> {ds_id} images={meta['image_count']} "
            f"counts={payload.get('counts')} names={names}"
        )
        return {"version": f"v{sv}", "dataset_id": ds_id, "images": meta["image_count"], "sealed_at": now}

    def _seal_receiving_bucket(self, model_id: str, mst: dict) -> dict | None:
        """对当前接收桶执行一次封板：FIFO 取满 SEAL_MIN_IMAGES 张，最后一个包截断，
        未取完的剩余量滚入下一接收桶。返回封板快照信息；无图可封时重置桶返回 None。
        """
        recv = mst["receiving"]
        need = settings.SEAL_MIN_IMAGES
        if need <= 0 or recv["poured"] < need:
            return None
        picks = []          # [(dataset_id, [records])]
        leftover = []       # 剩余 → 下一桶
        remaining_need = need
        fallback_classes = []
        for m in recv["members"]:
            if remaining_need <= 0:
                leftover.append(m)
                continue
            ds_dir = self.datasets_dir / m["dataset_id"]
            meta_path = ds_dir / "meta.json"
            pkg_meta = _load_json(meta_path) if meta_path.exists() else {}
            # 数据集一律为 datasets/{dataset_id}（无 version 子目录）
            take = min(m["count"], remaining_need)
            records = self._pick_images_for_seal(ds_dir, pkg_meta, take)
            consumed = len(records)
            if consumed > 0:
                if not fallback_classes:
                    fallback_classes = list(pkg_meta.get("classes") or [])
                picks.append((m["dataset_id"], records))
            # 更新源包封存账本（只增不减：已封存的图不再回流出）
            pkg_meta["bucket_sealed"] = int(pkg_meta.get("bucket_sealed") or 0) + consumed
            pkg_meta["bucket_pending"] = max(0, int(pkg_meta.get("bucket_pending") or 0) - consumed)
            pkg_stems = list(pkg_meta.get("sealed_stems") or [])
            pkg_stems.extend(r["image_id"] for r in records)
            pkg_meta["sealed_stems"] = pkg_stems[-2000:]
            pkg_meta["bucket_index"] = int(recv["index"])
            if meta_path.exists():
                _save_json(meta_path, pkg_meta)
            remaining_need -= consumed
            m_remain = m["count"] - consumed
            if m_remain > 0:
                leftover.append({"dataset_id": m["dataset_id"], "count": m_remain})
        if not picks:
            # 异常兜底：账本有量但取不到图 → 重置当前桶，避免死循环
            logger.warning(f"bucket seal {model_id}: no images picked, reset bucket")
            mst["receiving"] = {"index": int(recv["index"]) + 1, "poured": 0, "members": []}
            return None
        snap = self._create_bucket_snapshot(model_id, mst, picks, fallback_classes)
        mst["bucket_version_count"] = int(mst.get("bucket_version_count") or 0) + 1
        mst.setdefault("sealed", []).append(snap)
        # 下一接收桶：剩余图滚动 + 桶号递增
        leftover_poured = sum(x["count"] for x in leftover)
        mst["receiving"] = {
            "index": int(recv["index"]) + 1,
            "poured": leftover_poured,
            "members": leftover,
        }
        self._save_bucket_state(self._bucket_state)
        return snap

    def _advance_buckets_sync(self) -> dict:
        """灌桶同步核心：扫描标完未归档源包 → FIFO 灌桶 → 满 1500 连环自动封板。
        幂等：重复调用不重复灌/封。返回本次变更摘要。
        """
        if settings.SEAL_MIN_IMAGES <= 0:
            return {"poured": [], "sealed": [], "skipped": 0, "note": "SEAL_MIN_IMAGES=0 关闭"}
        state = self._load_bucket_state()
        self._bucket_state = state
        models_state = state.setdefault("models", {})
        poured_events, sealed_events = [], []
        skipped = 0
        try:
            # 1) 候选源包：已准备、标注阶段、归属模型、未归档
            groups = {}
            for dataset_dir in self.datasets_dir.iterdir():
                if not dataset_dir.is_dir() or "__deleting" in dataset_dir.name:
                    continue
                meta_path = dataset_dir / "meta.json"
                if not meta_path.exists():
                    continue
                try:
                    meta = _load_json(meta_path)
                except Exception:
                    continue
                model_id = str(meta.get("model_id") or "")
                if not model_id:
                    continue
                if meta.get("status") != "prepared":
                    continue
                if meta.get("stage") not in (STAGE_ANNOTATING, STAGE_COLLECTING):
                    continue
                if meta.get("bucket_archived"):
                    continue
                total = int(meta.get("image_count") or 0)
                if total <= 0:
                    continue
                groups.setdefault(model_id, []).append(
                    (dataset_dir.name, meta, dataset_dir, total, meta_path)
                )
            # 2) 按模型处理（组内按上传时间 FIFO）
            for model_id, pkgs in groups.items():
                pkgs.sort(key=lambda x: str(x[1].get("uploaded_at", "")))
                mst = models_state.setdefault(
                    model_id, {"bucket_version_count": 0, "receiving": None, "sealed": []}
                )
                if not mst.get("receiving"):
                    mst["receiving"] = {"index": 1, "poured": 0, "members": []}
                for dataset_id, meta, version_dir, total, meta_path in pkgs:
                    annotated = self._count_annotated_files(version_dir)
                    if annotated < total:
                        skipped += 1
                        continue  # 未全部标注完成（存在标注 JSON 的图不足），等下一轮
                    sealed_n = int(meta.get("bucket_sealed") or 0)
                    pending_n = int(meta.get("bucket_pending") or 0)
                    if sealed_n + pending_n >= total:
                        # 已全部入池：标记归档（只读展示），不再参与后续灌桶
                        if not meta.get("bucket_archived"):
                            meta["bucket_archived"] = True
                            meta["bucket_archived_at"] = datetime.now().isoformat()
                            meta["bucket_index"] = meta.get("bucket_index") or mst["receiving"]["index"]
                            _save_json(meta_path, meta)
                            logger.info(f"bucket: {dataset_id} fully poured -> archived B{meta['bucket_index']}")
                        continue
                    pour_n = total - sealed_n - pending_n
                    recv = mst["receiving"]
                    recv["poured"] += pour_n
                    if recv["members"] and recv["members"][-1]["dataset_id"] == dataset_id:
                        recv["members"][-1]["count"] += pour_n
                    else:
                        recv["members"].append({"dataset_id": dataset_id, "count": pour_n})
                    meta["bucket_pending"] = pending_n + pour_n
                    meta["bucket_index"] = int(recv["index"])
                    meta["bucket_poured_at"] = datetime.now().isoformat()
                    _save_json(meta_path, meta)
                    poured_events.append({
                        "model_id": model_id, "dataset_id": dataset_id,
                        "bucket_index": int(recv["index"]), "count": pour_n,
                    })
                    logger.info(
                        f"bucket: pour {dataset_id} +{pour_n} -> B{recv['index']} "
                        f"(total {recv['poured']}/{settings.SEAL_MIN_IMAGES})"
                    )
                    # 3) 满则连环封板（剩余超过 1500 时可能连续产出多个版本）
                    while mst["receiving"]["poured"] >= settings.SEAL_MIN_IMAGES:
                        snap = self._seal_receiving_bucket(model_id, mst)
                        if not snap:
                            break
                        sealed_events.append({"model_id": model_id, **snap})
        finally:
            self._save_bucket_state(state)
            self._bucket_state = None
        return {"poured": poured_events, "sealed": sealed_events, "skipped": skipped}

    async def advance_buckets(self) -> dict:
        """按模型滚动灌桶（幂等）：A 包标完自动灌桶，满 1500 自动封板出成品数据集。
        供前端在数据集页加载 / 标注完成返回时调用。
        """
        def _locked():
            with _BUCKET_LOCK:
                return self._advance_buckets_sync()
        return await asyncio.to_thread(_locked)

    async def bucket_poll_loop(self, poll_seconds: int | None = None) -> None:
        """后台自动调整（B 系列）轮询：定时扫描标完未归档的 A 包 → 自动灌桶/封板。
        与前端手动触发共用同一把线程锁（_BUCKET_LOCK），并发重复调用天然幂等。
        """
        interval = settings.BUCKET_POLL_SECONDS if poll_seconds is None else poll_seconds
        if interval <= 0:
            logger.info("bucket poll disabled (BUCKET_POLL_SECONDS=0)")
            return
        logger.info(f"bucket poll started, interval={interval}s")
        while True:
            try:
                res = await self.advance_buckets()
                if res.get("poured") or res.get("sealed"):
                    logger.info(
                        f"bucket poll: poured={len(res['poured'])} "
                        f"sealed={[(s['version'], s['images']) for s in res['sealed']]}"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"bucket poll: advance_buckets failed: {e}")
            await asyncio.sleep(interval)

    async def seal_poll_loop(self, poll_seconds: int | None = None) -> None:
        """P1 封板时间窗口兜底轮询：定时扫描"已准备但超期未封板"的数据集自动封板。
        数量门槛与时间窗口先达者生效；仅对已标注数据执行，避免空数据封板。
        """
        interval = settings.SEAL_POLL_SECONDS if poll_seconds is None else poll_seconds
        if interval <= 0 or settings.SEAL_MAX_DAYS <= 0:
            logger.info("seal poll disabled (SEAL_POLL_SECONDS=0 or SEAL_MAX_DAYS=0)")
            return
        logger.info(f"seal poll started (time-window={settings.SEAL_MAX_DAYS}天), interval={interval}s")
        while True:
            try:
                due = await self._collect_due_seal_datasets()
                if due:
                    logger.info(f"seal poll: {len(due)} 数据集超期，自动封板: {due}")
                    for dataset_id in due:
                        try:
                            await self.seal_dataset(dataset_id, force=True)
                            logger.info(f"seal poll: dataset {dataset_id} auto-sealed (time-window)")
                        except Exception as e:  # noqa: BLE001
                            logger.warning(f"seal poll: dataset {dataset_id} auto-seal failed: {e}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"seal poll: scan failed: {e}")
            await asyncio.sleep(interval)

    async def _collect_due_seal_datasets(self) -> list:
        """扫描未封板数据集：status=prepared（已准备）且 prepared_at 超期 → 待自动封板。
        仅统计有标注的数据集（has_labels/label_count/JSON 标注任一存在），避免空数据误封。
        """
        ds_dir = self.datasets_dir
        if not await asyncio.to_thread(lambda: ds_dir.exists()):
            return []
        due_ids = []
        for d in sorted(await asyncio.to_thread(lambda: list(ds_dir.iterdir()))):
            if not d.is_dir():
                continue
            meta_path = d / "meta.json"
            if not await asyncio.to_thread(lambda: meta_path.exists()):
                continue
            meta = await asyncio.to_thread(_load_json, meta_path)
            _ensure_stage_fields(meta)
            if meta.get("stage") == STAGE_SEALED or meta.get("sealed_at"):
                continue
            if meta.get("status") != "prepared":
                continue
            prepared_at = meta.get("prepared_at") or ""
            if not self._seal_due(prepared_at):
                continue
            has_anno = (
                bool(meta.get("has_labels"))
                or bool(meta.get("label_count"))
                or await asyncio.to_thread(self._count_json_annotations, d) > 0
            )
            if has_anno:
                due_ids.append(d.name)
        return due_ids
    
    async def export_original_dataset(self, dataset_id: str, version: str = "v1"):
        """导出标注前的数据集（仅图片）"""
        # 数据集一律为 datasets/{dataset_id}（无 version 子目录）
        dataset_dir = self.datasets_dir / dataset_id
        if not await asyncio.to_thread(lambda: dataset_dir.exists()):
            return None
        
        # 创建临时ZIP文件
        def _create_zip():
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            temp_zip_path = temp_zip.name
            temp_zip.close()
            
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 仅添加images目录
                images_dir = self._find_images_dir(dataset_dir)
                if images_dir:
                    for img_file in images_dir.rglob("*"):
                        if img_file.is_file():
                            arcname = img_file.relative_to(dataset_dir)
                            zipf.write(img_file, arcname)
                
                # 可选：添加data.yaml（如果存在）
                data_yaml = dataset_dir / "data.yaml"
                if data_yaml.exists():
                    zipf.write(data_yaml, "data.yaml")
            
            return temp_zip_path
        
        zip_path = await asyncio.to_thread(_create_zip)
        return zip_path

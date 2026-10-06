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
from src.utils.class_labels import model_class_names, merge_classes_with_model
from src.utils.class_labels import is_placeholder_classes as _is_placeholder_names
from src.utils import zip_guard
from src.services.annotation_importers import import_external_annotations
from src.services.annotation_service import (
    decode_yolo_line, yolo_label_line, clean_polygon, polygon_to_bbox, strip_ann_bbox, boxes_with_bbox,
)

logger = logging.getLogger(__name__)

# 桶灌流程互斥：防止并发请求重复灌桶/封板（模块级：多个服务实例共享同一把锁）
_BUCKET_LOCK = threading.Lock()

# 支持的图片后缀
IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 支持的上传压缩包格式
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
# 排除可视化预览图（如 *_result.jpg）
EXCLUDE_KEYWORDS = ("_result", "_predict", "_detect")

# list_datasets 签名缓存：磁盘（数据集 meta/类别文件、标签字典、桶账本）无变化时
# 直接复用上次结果，把列表页从 ~200-1800ms（冷）降到 ~1ms（热）。
# 签名 = 各关键文件 (路径, mtime_ns, size) 的聚合哈希，任一文件变化即失效，
# 保证 #6「库改标签实时可见」等语义不被缓存破坏。
_LIST_CACHE_LOCK = threading.Lock()
_LIST_CACHE = {"sig": None, "result": None, "ts": 0.0}

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
                # 安全地处理路径（防止路径遍历攻击）——收口到 utils.zip_guard（S-01/S-02）
                member_path = Path(member)
                if not zip_guard.is_safe_member(member):
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

    async def upload_dataset(self, file: UploadFile, model_id: str = None, library_root_id: str = None):
        """上传数据集压缩包（zip / tar / tar.gz / tgz）

        model_id: 归属模型（1.7 模型仓库：数据挂到对应模型下）；为空则由后续采集/绑定流程归属
        library_root_id: 归属算法库的根记录 id；训练产出未显式指定库时以此为准。
                         为空时由调用方按 model_id 反查（模型自身即库根时取自身）
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
            # 归属算法库（库根记录 id）：训练未显式指定库时，产出继承这个库
            "library_root_id": library_root_id,
        }
        
        meta_path = dataset_dir / "meta.json"
        await asyncio.to_thread(_save_json, meta_path, meta)
        
        from src.db import shadow
        await shadow.sync_dataset(dataset_id)
        return meta
    
    async def _rollback_model_binding_on_prepare_failure(self, meta_path: Path):
        """准备失败 → 解除数据集与模型的挂载（失败的不留半成品在算法库，省去手动删）。"""
        try:
            if not await asyncio.to_thread(lambda: meta_path.exists()):
                return
            meta = await asyncio.to_thread(_load_json, meta_path)
            if meta.get("model_id"):
                meta["model_id"] = None
                await asyncio.to_thread(_save_json, meta_path, meta)
                if settings.PG_ENABLED:
                    from src.db import meta_store
                    await meta_store.upsert_dataset(meta)
                logger.info("prepare failed: unbound dataset from model (clean rollback)")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"prepare failure rollback unbind failed: {e}")

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
        try:
            await asyncio.to_thread(_extract_archive, archive_path, version_dir)
        except Exception:
            await self._rollback_model_binding_on_prepare_failure(meta_path)
            raise
        t_extract = time.time()
        logger.info(f"prepare {dataset_id}: extract took {t_extract - t0:.1f}s")
        
        # 「准备」= 把数据集改造成平台可用格式：
        # 0) 外部分类格式（COCO / LabelMe / VOC）→ 统一转一图一 JSON（数据链路阶段B），
        #    产物 images/（平铺）+ annotations/{image_id}.json + classes.txt，跳过重组
        detected_classes = classes or []
        model_id_ds = str(meta.get("model_id", "") or "")
        # 归属算法库兜底：老数据只写了 model_id，这里按库根补齐 library_root_id，
        # 使「从训练页发起的训练」也能识别产出应归入哪个库
        if model_id_ds and not meta.get("library_root_id"):
            meta["library_root_id"] = self._library_root_id_of(model_id_ds)
        # 目标模型标签空间（模型标签字典英文码），用于 txt 类名重映射 / 冗余类过滤
        model_labels = self._labels_from_model(model_id_ds) if model_id_ds else []
        # 类别是否来自包内（用户显式传 / 包内类别文件 / 标签推断）：
        # 只有包内来源才是可信的"源类名列表"，可与模型标签做交集过滤；模型回填不算
        classes_from_package = bool(classes)
        # 解压内容导入/重组统一兜底：任一步失败 → 解除模型挂载（准备失败不留半成品挂载）
        has_labels = None
        try:
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
        except Exception:
            await self._rollback_model_binding_on_prepare_failure(meta_path)
            raise
        t_reorganize = time.time()
        logger.info(f"prepare {dataset_id}: reorganize/scan took {t_reorganize - t_extract:.1f}s")
        
        # 检查目录结构（在线程中执行）
        images_dir = await asyncio.to_thread(self._find_images_dir, version_dir)
        labels_dir = await asyncio.to_thread(self._find_labels_dir, version_dir)
        
        if not images_dir:
            await self._rollback_model_binding_on_prepare_failure(meta_path)
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

        # 包内只有类别号、没有类名（平台编的占位名 class_N）时，用归属模型的标签字典补真名。
        # 不补的后果是连锁的：占位名被当成"包内可信类名"去和模型标签做名称匹配，
        # 而 "class_0" 永远撞不上字典里的 "fall" → class_map 为 None → 不重映射、
        # 类别空间也不收敛，于是数据集明明归在跌倒库下、classes 却还是 ["class_0"]。
        # 只有类数对得上才按位置对齐（_detect_classes 会把空洞补成 0..max，位置即类号）；
        # 对不齐就宁可留占位名等人工改名，绝不猜——猜错等于把整批标注挂到别的标签上。
        if self._is_placeholder_classes(detected_classes) and model_labels:
            if len(detected_classes) == len(model_labels):
                logger.info(
                    f"prepare {dataset_id}: placeholder classes {detected_classes} "
                    f"replaced by model {model_id_ds} labels {model_labels}"
                )
                detected_classes = list(model_labels)
            else:
                logger.warning(
                    f"prepare {dataset_id}: 包内无类名（{len(detected_classes)} 个占位类），"
                    f"与模型 {model_id_ds} 的 {len(model_labels)} 个标签数量对不上，"
                    f"保留占位名等待人工改名"
                )

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
                from src.services.model_service import ModelService
                meta["library_root_id"] = ModelService().resolve_library_root_id(matched_id)
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
                    # 影子库收口：init_labels_from_classes 同时改写 model.json 与 labels_dict.json
                    from src.db import shadow
                    await shadow.sync_model(model_id_ds)
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
        from src.db import shadow
        await shadow.sync_dataset(dataset_id)

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
        
        # 一个类别号都没扫到（labels 目录为空 / txt 全是空文件）时必须返回空列表：
        # 这里若兜底成 ["class_0"]，无标签的数据集就会被判定成"包内有一个可信类别
        # class_0"，从而跳过后面按归属模型补类别的分支，类别空间被假名字占死。
        if not class_ids:
            return []
        max_id = max(class_ids)
        return [f"class_{i}" for i in range(max_id + 1)]

    @staticmethod
    def _is_placeholder_classes(classes: list) -> bool:
        """类别列表全是平台编出来的占位名（class_0…）→ True，说明包内根本没带类名。"""
        return _is_placeholder_names(classes)

    def _check_yolo_line(self, parts: list, nc: int):
        """校验单行 YOLO 标注，返回 (ok, err_msg)

        - 检测行：恰好 5 列 (class cx cy w h)
        - 分割行：类别之后全是成对顶点，即 7/9/11… 列（总列数必须是奇数且 ≥7）
          平台历史导出的 "class cx cy w h + 顶点" 同样是奇数列，一并放行（读入口兼容）
        - class 为 [0, nc) 内整数；坐标均为 [0,1] 归一化浮点
        - 允许 1e-3 数值容差（第三方工具常有 -0.0001 / 1.0001 这类边界抖动）
        """
        if len(parts) != 5 and (len(parts) < 7 or len(parts) % 2 == 0):
            return False, f"{len(parts)} 列非法（应为 5 列检测行，或 ≥7 列的奇数列分割行）"
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
        """解析数据集对外展示的类别清单：算法库标签字典（实时）> 包内类别文件 > 快照。

        - 归属算法库的标签字典是实时源（#6）：库侧新增/改名标签后页面立即跟随，
          不需要重新准备数据集，也不回写存量快照（按 index 合并、绝不截断）。
        - 历史数据 prepare 时可能因未读到类别文件而生成了 class_0/class_1 等占位名，
          若目录中存在真实类别文件（classes.txt / coco.names / names.txt）则按行回填。
        """
        classes = meta.get("classes") or []
        live = merge_classes_with_model(classes, model_class_names(meta.get("model_id") or ""))
        if live != classes:
            return live
        if not classes:
            return classes
        # 仅当全部是占位名时才回填（避免覆盖用户自定义的真实类别名）
        if not all(re.fullmatch(r"class_\d+", str(c)) for c in classes):
            return classes
        version_dir = dataset_dir
        # 类别文件通常就在数据集根目录；rglob 会把 images/labels 下上千文件全 stat 一遍
        # （冷缓存实测单数据集 1.2s，是列表页卡顿主因）。改为限定深度扫描：
        # 根目录 + 2 层子目录内查找，命中候选即止，绝不深入 images/labels/test 等大目录。
        for name in ["classes.txt", "coco.names", "names.txt"]:
            for path in self._iter_named_files_shallow(version_dir, name):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        names = [line.strip() for line in f if line.strip()]
                    if names and len(names) == len(classes):
                        return names
                except Exception:
                    continue
        return classes

    @staticmethod
    def _iter_named_files_shallow(root: Path, filename: str, max_depth: int = 3):
        """在 root 下限定深度查找名为 filename 的文件，跳过图片/标签等大目录。

        替代 rglob(filename)：后者会 stat 目录树里每个文件（数千级），
        而类别文件几乎总在根目录或浅层标注子目录，命中即返回。
        """
        _SKIP_DIRS = {"images", "labels", "train", "val", "test", "previews", "thumbnails", "__pycache__"}
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            try:
                with os.scandir(d) as it:
                    for e in it:
                        try:
                            if e.is_file(follow_symlinks=False) and e.name == filename:
                                yield Path(e.path)
                            elif e.is_dir(follow_symlinks=False) and depth < max_depth and e.name not in _SKIP_DIRS:
                                stack.append((Path(e.path), depth + 1))
                        except OSError:
                            continue
            except OSError:
                continue

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
                        # decode_yolo_line 同时认"官方分割行（类别后全是顶点）/ 检测行 /
                        # 平台旧导出行"，轮廓必须带进 JSON，否则外部包会被静默截成矩形框
                        box = decode_yolo_line(parts, w, h)
                        if box:
                            boxes.append(box)
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
            _save_json(ann_dir / f"{rec['image_id']}.json", strip_ann_bbox(doc))
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

    def _annotation_stats(self, dataset_dir: Path) -> tuple:
        """一次遍历标注目录，返回 (有框图片数, 标注文件总数)。

        封板要同时知道两件事：能不能封板（有框/有标签文件）和
        有没有可转换的标注源（存在标注文件即可，空白标注也要转成 0 字节 txt）。
        合并成一次扫描，避免 seal 路径上重复遍历。
        """
        ann_dir = dataset_dir / "annotations"
        if not ann_dir.exists():
            return 0, 0
        with_boxes = 0
        total = 0
        for p in ann_dir.glob("*.json"):
            total += 1
            try:
                data = _load_json(p)
            except Exception:
                continue
            if data.get("boxes"):
                with_boxes += 1
        return with_boxes, total

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
    def _library_root_id_of(model_id: str):
        """模型 id → 所属算法库根记录 id（换算规则由 ModelService 统一维护）。"""
        if not model_id:
            return None
        from src.services.model_service import ModelService
        return ModelService().resolve_library_root_id(model_id)

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
            except Exception:
                continue
            label_set = {self._norm_cls(c) for c in labels if str(c).strip()}
            if not label_set:
                continue  # 无标签空间（含刚建好还没配标签的算法库）不参与自动匹配
            # 判据只看标签字典是否为空，不要按 empty 跳过：empty 的语义是
            # 「算法库占位记录」，而库根恰恰是数据集应当归属的对象（见 library_root_id）。
            cover = len(base & label_set) / len(base)
            if cover >= 1.0:
                return model_dir.name  # 完全覆盖：立即命中
            if cover > best:
                best, best_model_id = cover, model_dir.name
        if best_model_id and best >= 0.6:
            return best_model_id
        return None

    def _resolve_seal_names(self, meta: dict) -> list:
        """解析封板 data.yaml 的类别名：优先模型标签字典英文码，缺省回退 meta.classes。

        - 走 class_labels.model_class_names：按 index 定位（缺名补占位、绝不跳行），
          自身目录无字典时回退到所属算法库根，与页面展示同一口径（#6）。
        - 字典比数据集快照短时不截断，尾部沿用 meta.classes
          （截断会让已落盘的 class_id 越界）。
        - 字典全是占位名（未配真名）时不覆盖数据集里的真实类名。
        """
        names = model_class_names(meta.get("model_id") or "")
        if not names or _is_placeholder_names(names):
            return meta.get("classes") or []
        return merge_classes_with_model(meta.get("classes") or [], names)

    # ---------- 跨库挂载校验（同类别但序号不同 → 封板会静默串号） ----------
    def _label_shift_report(self, triples: list, names: list, old_classes: list) -> tuple:
        """比对「标注框已落盘的类别序号」与「目标类别顺序」，找出会串号的类别。

        原理：每图标注 JSON 里每个框同时存了 class_id（数字序号，封板时原样写进 YOLO txt）
        与 english_code（标注当时的语义）。YOLO 只认数字，数字代表哪一类完全由
        data.yaml 的 names 顺序决定 —— 所以只要换到一个「有同名类别但排在不同序号」的
        算法库，框就会被认成别的类，而且不报错。

        triples:     [(class_id, english_code, image_id), ...]  每个标注框一条
        names:       目标顺序的类别名列表（下标即 YOLO 类别号）
        old_classes: 数据集自己的 classes（数字序号当初就是按它排的），
                     框上没有 english_code 的老数据用它兜底判语义

        返回 (blocking, unknown)：
        - blocking: 可证明错位（序号越界 / 同名类别两边序号不同 / 名字类找不到位置）→ 需确认才放行
        - unknown:  框上的类别名在目标字典里根本不存在 → 训练时会被剔除，只提醒不拦
        """
        names = [str(n or "").strip() for n in (names or [])]
        old_classes = [str(n or "").strip() for n in (old_classes or [])]
        pos = {}  # 归一化类名 → 目标序号
        for i, nm in enumerate(names):
            if nm:
                pos.setdefault(self._norm_cls(nm), i)
        lower_names = {str(nm).lower() for nm in names if nm}

        # 按「YOLO 序号」聚合（一个序号一条结论），统计受影响的图片数与框数
        groups = {}
        for cid, ec, img_id in triples:
            numeric = isinstance(cid, int) or (isinstance(cid, str) and str(cid).isdigit())
            key = (numeric, int(cid) if numeric else str(cid or ""))
            g = groups.setdefault(key, {"codes": set(), "images": set(), "boxes": 0})
            if (ec or "").strip():
                g["codes"].add(str(ec).strip())
            g["images"].add(img_id)
            g["boxes"] += 1

        def _sort_key(k):
            return (0, k[1], "") if k[0] else (1, 0, k[1])

        blocking, unknown = [], []
        for (numeric, cid) in sorted(groups, key=_sort_key):
            g = groups[(numeric, cid)]
            n_images, n_boxes = len(g["images"]), g["boxes"]
            if not numeric:
                # 框上存的是类别名（历史 JSON）：按名字找位置，找不到会被写成第 0 类
                if str(cid).lower() in lower_names:
                    continue
                blocking.append({
                    "class_id": str(cid), "english_code": str(cid), "want_index": None,
                    "actual_index": 0, "images": n_images, "boxes": n_boxes,
                    "reason": f"标注里的类别「{cid}」在新库字典里找不到位置，封板会被写成第 0 类"
                              f"（即「{names[0] if names else '空'}」），影响 {n_images} 张图 / {n_boxes} 个框",
                })
                continue
            idx = cid
            srcs = sorted(g["codes"]) or (
                [old_classes[idx]] if idx < len(old_classes) and old_classes[idx] else [])
            if idx >= len(names) or not names[idx]:
                blocking.append({
                    "class_id": idx, "english_code": "、".join(srcs), "want_index": None,
                    "actual_index": idx, "images": n_images, "boxes": n_boxes,
                    "reason": f"第 {idx} 类「{'、'.join(srcs) or '未知'}」超出新库字典范围"
                              f"（新库只有 {len(names)} 类），影响 {n_images} 张图 / {n_boxes} 个框",
                })
                continue
            if not srcs:
                continue  # 老数据既没英文名也没 classes：语义无从考证，不冤枉拦
            wants = {s: pos.get(self._norm_cls(s)) for s in srcs}
            shift = [w for w in wants.values() if w is not None and w != idx]
            if shift:
                blocking.append({
                    "class_id": idx, "english_code": "、".join(srcs), "want_index": shift[0],
                    "actual_index": idx, "images": n_images, "boxes": n_boxes,
                    "reason": f"类别「{'、'.join(srcs)}」在这个数据集里是第 {idx} 类，"
                              f"新库字典里排第 {shift[0]} 类；封板会原样写 {idx}，"
                              f"训练时这 {n_images} 张图的框会被当成「{names[idx]}」",
                })
            missing = [s for s, w in wants.items() if w is None]
            if missing:
                unknown.append({
                    "class_id": idx, "english_code": "、".join(missing),
                    "images": n_images, "boxes": n_boxes,
                    "reason": f"类别「{'、'.join(missing)}」在新库字典里不存在，训练时这些框会被剔除",
                })
        return blocking, unknown

    @staticmethod
    def _conflict_summary(blocking: list, unknown: list) -> str:
        """把校验结果拼成大白话（弹窗确认文案 / 后端拒绝提示都用它）"""
        lines = []
        if blocking:
            lines.append(f"有 {len(blocking)} 个类别的序号会对不上（封板后这些框会被认成别的类）：")
            for it in blocking[:3]:
                lines.append("· " + it["reason"])
            if len(blocking) > 3:
                lines.append(f"· ……另有 {len(blocking) - 3} 个类别同样对不上")
        if unknown:
            codes = "、".join(str(it.get("english_code") or it.get("class_id")) for it in unknown[:5])
            lines.append(f"提示：{codes} 在新库字典里没有，训练时这些框会被自动剔除")
        return "\n".join(lines)

    def _collect_label_triples(self, dataset_dir: Path) -> tuple:
        """扫描每图标注 JSON → ([(class_id, english_code, image_id), ...], 有框图片数)"""
        triples, imgs = [], set()
        ann_dir = dataset_dir / "annotations"
        if ann_dir.is_dir():
            for p in ann_dir.glob("*.json"):
                try:
                    data = _load_json(p)
                except Exception:
                    continue
                boxes = data.get("boxes")
                if not boxes:
                    continue
                img_id = data.get("image_id") or p.stem
                imgs.add(img_id)
                for b in boxes:
                    triples.append((b.get("class_id"), b.get("english_code") or "", img_id))
        return triples, len(imgs)

    def _bind_conflict_sync(self, dataset_dir: Path, meta: dict, root_id: str) -> dict:
        """挂载冲突检查（同步实现）：把数据集已有标注对到目标库字典的序号上比一遍。"""
        names = self._resolve_seal_names({"model_id": root_id, "classes": meta.get("classes") or []})
        triples, n_imgs = self._collect_label_triples(dataset_dir)
        out = {
            "dataset_id": meta.get("dataset_id") or dataset_dir.name,
            "target_library_id": root_id,
            "target_names": names,
            "annotated_images": n_imgs,
            "verifiable": bool(names and triples),
            "blocking": [], "warnings": [], "conflict": False, "summary": "",
        }
        if not names:
            out["summary"] = "目标算法库还没有标签字典，无法比对类别序号（挂载后建议先补齐字典再标注）"
            return out
        if not triples:
            return out  # 该数据集还没有任何标注框：换库不会串号
        blocking, unknown = self._label_shift_report(triples, names, meta.get("classes") or [])
        out["blocking"], out["warnings"] = blocking, unknown
        out["conflict"] = bool(blocking)
        out["summary"] = self._conflict_summary(blocking, unknown)
        return out

    async def check_bind_target(self, dataset_id: str, model_id: str) -> dict:
        """挂载前校验：数据集换绑到 model_id（算法库根记录）后，类别序号会不会错位。"""
        dataset_dir = self.datasets_dir / dataset_id
        meta_path = dataset_dir / "meta.json"
        if not await asyncio.to_thread(lambda: meta_path.exists()):
            raise ValueError(f"Dataset {dataset_id} not found")
        meta = await asyncio.to_thread(_load_json, meta_path)
        return await asyncio.to_thread(self._bind_conflict_sync, dataset_dir, meta, model_id)

    def _build_training_artifacts(self, dataset_dir: Path, meta: dict, split_ratio: dict,
                                  out_dir: Path = None, label_guard: bool = False) -> dict:
        """封板一次性产出：划分 train/val/test + 每图 JSON → YOLO txt + 生成 data.yaml（数据链路阶段E）

        标注数据源：dataset_dir/annotations/{image_id}.json（像素坐标框）
        图片源：dataset_dir/images/ 平铺目录
        out_dir：产物输出目录（缺省 = dataset_dir 原地）。数据血缘方案 A 下每个封板
        版本输出到数据集下 sealed/v{N}/ 只读快照，旧版本不被覆盖、版本号逐次递增。
        label_guard：True 时封板前校验「标注序号 vs 目标字典序号」，对不上直接拒绝
        （只在手动封板链路开启；自动灌桶封板是后台定时流程，拦住会静默中断灌桶，故不启用）。
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
                # 读取兜底：分割框只存轮廓点，需要矩形的下游由外接框现算
                boxes = boxes_with_bbox(data.get("boxes"))
                img_id = data.get("image_id", p.stem)
                img_abs = img_index.get(img_id)
                if img_abs is None:
                    continue
                if not boxes:
                    # 空白标注（无框）也收集，作为独立分层参与划分
                    annotated.append({
                        "image_id": img_id,
                        "img_abs": img_abs,
                        "width": data.get("width") or 1,
                        "height": data.get("height") or 1,
                        "boxes": [],
                    })
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
        if not any(rec.get("boxes") for rec in annotated):
            # 全是空白标注（人工确认无目标 / AI 困难样本）：照样产出划分与 0 字节 txt，
            # 这类背景负样本集有训练价值，封板不该因为"一个框都没有"而空转（#3）。
            logger.warning(
                f"dataset {dataset_dir.name} 标注全部无框：封板产物只含背景样本（0 字节标签文件）"
            )

        # data.yaml 类别名（英文码）提前确定：分层统计与 txt 的 class 索引都依赖它
        names = self._resolve_seal_names(meta)
        _cls_idx = {str(nm).lower(): i for i, nm in enumerate(names)}

        # 封板兜底校验：txt 里的类别号直接沿用标注 JSON 的 class_id（标注当时的顺序写死的），
        # 若该数据集被换挂到「有同名类别但序号不同」的算法库，这里不拦就会静默串号。
        if label_guard:
            _triples = [
                (b.get("class_id"), b.get("english_code") or "", rec["image_id"])
                for rec in annotated for b in (rec.get("boxes") or [])
            ]
            _blocking, _unknown = self._label_shift_report(_triples, names, meta.get("classes") or [])
            if _blocking:
                raise ValueError(
                    "封板中止：下列类别的序号与归属算法库的标签字典对不上，"
                    "直接封板会把框标成别的类。\n"
                    + self._conflict_summary(_blocking, _unknown)
                    + "\n请先到该算法库里把类别顺序调成和上面一致（或改挂到顺序一致的算法库）再重试。"
                )

        def _norm_cid(cid):
            """框 class_id → 标准类别名（数字索引按 names 解析，英文名回退索引）"""
            if isinstance(cid, int) or (isinstance(cid, str) and cid.isdigit()):
                i = int(cid)
                return names[i] if 0 <= i < len(names) else str(i)
            s = str(cid).lower()
            return names[_cls_idx[s]] if s in _cls_idx else str(cid)

        # 2) 划分（确定性打乱，保证可复现）：先统计每类框数与空白图数，
        #    再按"类别组合"分层，每层各自按 ratio（默认 7:1.5:1.5）切分，
        #    保证 train/val/test 中各类别占比接近。样本 ≤2 的稀缺层全进 train。
        tr = float(split_ratio.get("train", 0.7))
        vr = float(split_ratio.get("val", 0.15))
        te = float(split_ratio.get("test", 0.15))
        total = tr + vr + te
        if total <= 0:
            tr, vr, te = 0.7, 0.15, 0.15
        else:
            tr, vr, te = tr / total, vr / total, te / total
        random.Random(42).shuffle(annotated)
        n = len(annotated)
        class_box_stats = {}
        blank_images = 0
        strata = {}
        for rec in annotated:
            names_in_img = sorted({_norm_cid(b["class_id"]) for b in rec["boxes"]})
            for nm in names_in_img:
                cnt = sum(1 for b in rec["boxes"] if _norm_cid(b["class_id"]) == nm)
                class_box_stats[nm] = class_box_stats.get(nm, 0) + cnt
            key = "+".join(names_in_img) if names_in_img else "__blank__"
            if not names_in_img:
                blank_images += 1
            strata.setdefault(key, []).append(rec)
        splits = {"train": [], "val": [], "test": []}
        for key in sorted(strata):
            recs = strata[key]
            m = len(recs)
            if m <= 2:
                splits["train"].extend(recs)
                continue
            a = int(round(m * tr))
            b = int(round(m * vr))
            if a + b > m:
                a = m - b
            splits["train"].extend(recs[:a])
            splits["val"].extend(recs[a:a + b])
            splits["test"].extend(recs[a + b:])

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

        counts = {}
        # 分割/检测行数统计：用于判断本次封板产物是不是分割数据集
        seg_lines = 0
        rect_lines = 0
        split_class_stats = {s: {"images": 0, "boxes": {}} for s in ("train", "val", "test")}
        for split, recs in splits.items():
            counts[split] = 0
            for rec in recs:
                ext = rec["img_abs"].suffix.lower() or ".jpg"
                shutil.copy2(staging / f"{rec['image_id']}{ext}", images_dir / split / f"{rec['image_id']}{ext}")
                w = rec["width"] or 1
                h = rec["height"] or 1
                lines = []
                # 有轮廓的框写分割行（只写顶点，不带框前缀），无轮廓的写检测行；
                # 同一文件混有分割行时，无轮廓的框退化成 4 角点分割行
                box_polys = [clean_polygon(b.get("polygon")) for b in rec["boxes"]]
                any_poly = any(box_polys)
                for b, poly in zip(rec["boxes"], box_polys):
                    if poly:
                        x1, y1, x2, y2 = polygon_to_bbox(poly)
                    else:
                        x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
                    cid = b["class_id"]
                    if isinstance(cid, int) or (isinstance(cid, str) and cid.isdigit()):
                        cls_id = int(cid)  # 旧数据数字索引直接使用
                    else:
                        cls_id = _cls_idx.get(str(cid).lower(), 0)  # JSON 英文名 → YOLO 索引
                    lines.append(yolo_label_line(cls_id, (x1, y1, x2, y2), poly, w, h,
                                                 rect_as_poly=any_poly))
                    if poly:
                        seg_lines += 1
                    else:
                        rect_lines += 1
                (labels_dir / split / f"{rec['image_id']}.txt").write_text(
                    "\n".join(lines), encoding="utf-8"
                )
                counts[split] += 1
                scs = split_class_stats[split]
                scs["images"] += 1
                for b in rec["boxes"]:
                    nm = _norm_cid(b["class_id"])
                    scs["boxes"][nm] = scs["boxes"].get(nm, 0) + 1

        # 4) 生成 data.yaml（names 用模型标签字典英文码，缺省回退 meta.classes；已在导出前确定）
        # B-09 治本：不写 path 键。ultralytics 缺 path 时自动回退到 yaml 所在目录
        # （check_det_dataset: Path(data["yaml_file"]).parent），train/val 保持相对路径，
        # 数据集改名/迁移/换机器后训练不再因 stale 绝对路径报 images not found。
        # （写 "." 不行——相对路径按训练进程 cwd 解析，不是按 yaml 所在目录）
        yaml_data = {
            "train": "images/train",
            "val": "images/val",
            "nc": len(names),
            "names": names,
        }
        # 只要有一条分割轮廓就标 segment（ultralytics 按行内列数自适应，纯检测行仍当检测框学）
        yaml_data["task"] = "segment" if seg_lines > 0 else "detect"
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

        logger.info(
            f"seal artifacts: total={n} splits={counts} nc={len(names)} "
            f"class_boxes={class_box_stats} blank={blank_images} "
            f"task={yaml_data['task']} seg_lines={seg_lines} rect_lines={rect_lines}"
        )
        return {
            "total": n,
            "counts": counts,
            "class_stats": {"boxes_by_class": class_box_stats, "blank_images": blank_images},
            "split_class_stats": split_class_stats,
            "ratios": {"train": round(tr, 3), "val": round(vr, 3), "test": round(te, 3)},
            "nc": len(names),
            "names": names,
            "label_validation": label_validation,
            "sample_type_stats": sample_stats,
            # 分割标注情况：has_segment=本批产物里是否有轮廓；mixed=分割与纯框共存
            "task": yaml_data["task"],
            "has_segment": seg_lines > 0,
            "segment_boxes": seg_lines,
            "rect_boxes": rect_lines,
            "mixed": seg_lines > 0 and rect_lines > 0,
        }

    def _list_cache_signature(self) -> tuple:
        """聚合磁盘关键文件的 (路径, mtime_ns, size) 作为缓存签名。

        覆盖：各数据集 meta.json / 根目录 data.yaml、各模型 labels_dict.json、
        桶账本。任一文件增删改 → 签名变化 → 缓存失效。
        不覆盖：训练子进程绕过 meta 的极端情况——由下方 TTL 兜底（最多陈旧 2s）。
        """
        import hashlib

        def _fp(p: Path):
            try:
                st = p.stat()
                return f"{p.name}|{st.st_mtime_ns}|{st.st_size}"
            except OSError:
                return "-"

        parts = []
        try:
            for d in sorted(self.datasets_dir.iterdir()):
                if not d.is_dir() or d.name.startswith("_") or "__deleting" in d.name:
                    continue
                parts.append(d.name)
                parts.append(_fp(d / "meta.json"))
                parts.append(_fp(d / "data.yaml"))
        except OSError:
            pass
        try:
            if settings.REGISTRY_DIR.exists():
                for m in sorted(settings.REGISTRY_DIR.iterdir()):
                    if m.is_dir():
                        parts.append(_fp(m / "labels_dict.json"))
        except OSError:
            pass
        parts.append(_fp(Path(settings.DATA_DIR) / "buckets.json"))
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

    async def list_datasets(self):
        """列出所有数据集（带签名缓存：磁盘无变化时直接返回上次结果，秒出）"""
        # 缓存命中：签名（关键文件 mtime/size 聚合）未变且未过 TTL → 直接返回，
        # 跳过全量扫描/rglob/shadow 自愈（这些合计 100~1800ms）。
        _TTL = 2.0
        sig = await asyncio.to_thread(self._list_cache_signature)
        now = time.time()
        with _LIST_CACHE_LOCK:
            if (_LIST_CACHE["sig"] == sig and _LIST_CACHE["result"] is not None
                    and now - _LIST_CACHE["ts"] < _TTL):
                return _LIST_CACHE["result"]

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
        # 影子库收口（读侧自愈）：训练子进程 train_script.py 会绕过本进程直接改 meta.json
        # （trained_at / last_trained_round / stage），列表页按 mtime 水位补齐差集。
        from src.db import shadow
        await shadow.refresh_datasets(d.get("dataset_id") for d in datasets)
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
        result = {
            "datasets": sorted(datasets, key=lambda x: x.get("uploaded_at", ""), reverse=True),
            # 封板数量门槛（#6）：前端列表展示「待封板 X / 门槛 N 张」进度
            "seal_min_images": settings.SEAL_MIN_IMAGES,
            # B 系列：按模型滚动灌桶进度（接收桶累计量 / 已封板版本列表）
            "buckets": _bucket_summary(),
        }
        with _LIST_CACHE_LOCK:
            _LIST_CACHE["sig"] = sig
            _LIST_CACHE["result"] = result
            _LIST_CACHE["ts"] = time.time()
        return result

    async def bind_model(self, dataset_id: str, model_id: str, library_root_id: str = None,
                         force: bool = False):
        """绑定数据集到模型（1.7 模型仓库：数据归属模型，合并/采集复用）

        library_root_id: 归属算法库根记录 id（解绑时传 None 一并清空）；
                         由调用方按 model_id 反查，保证「库详情—挂载数据集」与训练继承一致。
        force: 换库时已知「类别序号对不上」仍坚持挂载（默认 False：有冲突直接拒绝）。

        绑定成功后数据集全部图片自动流入模型消息队列（阶段D），
        队列达到阈值或定时后自动打包进入标注页。
        """
        dataset_dir = self.datasets_dir / dataset_id
        meta_path = dataset_dir / "meta.json"
        if not await asyncio.to_thread(lambda: meta_path.exists()):
            raise ValueError(f"Dataset {dataset_id} not found")

        # 换库校验：目标库与当前归属不同才比对（解绑、原地重绑不打扰）
        if model_id:
            meta_now = await asyncio.to_thread(_load_json, meta_path)
            if (meta_now.get("model_id") or "") != model_id:
                report = await asyncio.to_thread(
                    self._bind_conflict_sync, dataset_dir, meta_now, model_id
                )
                if report.get("conflict") and not force:
                    raise ValueError(
                        "挂载已阻止：该数据集已标注的类别序号与目标算法库不一致。\n"
                        + (report.get("summary") or "")
                        + "\n确认后请勾选强制挂载，或先到目标算法库把标签顺序调整一致。"
                    )

        def _bind_sync():
            meta = _load_json(meta_path)
            meta["model_id"] = model_id
            meta["library_root_id"] = library_root_id
            
            # 读取模型标签字典（包含中文别名和描述）；解绑时 model_id 为空，跳过
            labels_dict = None
            if model_id:
                labels_file = settings.REGISTRY_DIR / model_id / "labels_dict.json"
                if labels_file.exists():
                    try:
                        labels_dict = _load_json(labels_file)
                    except Exception:
                        pass
            
            # ✅ 新增：同步中文标签信息到 meta.json，供 AI 标注参考
            if labels_dict and labels_dict.get("labels"):
                # 提取每个类别的中文别名和描述
                chinese_labels = []
                codes = []
                for label in labels_dict.get("labels"):
                    ec = label.get("english_code", "")
                    cn = label.get("chinese_name", "")
                    desc = label.get("chinese_desc", "")
                    chinese_labels.append({
                        "english_code": ec,
                        "chinese_name": cn,
                        "chinese_desc": desc,
                    })
                    if ec:
                        codes.append(ec)
                meta["chinese_labels"] = chinese_labels
                
                # 绑定后补类别：数据集 classes 为空、或只有平台编的占位名（class_0…）时，
                # 从模型标签字典回填真名。只判"空"会漏掉占位名这条路径——数据集通常是
                # 先导入（包内没类名，被填成 class_0）后绑库，绑定时 classes 已非空，
                # 于是库里明明是 fall，数据集还挂着 class_0，导出/训练的标签名全错。
                existing = meta.get("classes") or []
                if codes and (not existing or self._is_placeholder_classes(existing)):
                    meta["classes"] = codes
                    try:
                        if dataset_dir.exists():
                            (dataset_dir / "classes.txt").write_text(
                                "\n".join(codes) + "\n", encoding="utf-8"
                            )
                            logger.info(f"bind {dataset_id}->{model_id}: classes.txt written: {codes}")
                    except Exception:
                        pass
                
                logger.info(f"bind {dataset_id}->{model_id}: synced {len(chinese_labels)} chinese labels")
            
            _save_json(meta_path, meta)
            return meta

        meta = await asyncio.to_thread(_bind_sync)
        from src.db import shadow
        await shadow.sync_dataset(dataset_id)

        # 阶段D：数据流入模型消息队列（阈值/定时自动打包进入标注页）；解绑不入队
        try:
            if not model_id:
                return meta
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
        from src.db import shadow
        await shadow.sync_dataset(dataset_id)
        
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
          未标注时默认拒绝封板，force=True 提供时间窗口兜底（强制封板）
        - 只要有标注文件就生成产物：空白标注（无框）转成 0 字节 txt 作为背景负样本参与划分
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
            # 幂等：已封板直接返回。例外——force 且产物缺失（历史数据只打了 sealed 标、
            # 没生成 data.yaml/划分目录）时补生成一次，否则"再点一次封板"永远无效。
            if not force or (dataset_dir / "data.yaml").exists():
                return meta
            return await self._backfill_seal_artifacts(dataset_dir, meta_path, meta)

        if meta.get("status") != "prepared":
            raise ValueError("数据集尚未准备（需先执行准备操作），无法封板")

        # 标注✓判定：优先每图 JSON 标注（数据集一律为 datasets/{dataset_id}，无 version 子目录）
        # 两个独立概念（#3 的病灶就是把它们混成一个布尔量）：
        #   annotated           —— 有没有"带框"的标注，决定能不能封板（force 兜底）
        #   ann_files           —— 有多少标注文件，空白标注也要转成 0 字节 txt 参与产物生成
        boxes_annotated, ann_files = await asyncio.to_thread(self._annotation_stats, dataset_dir)
        annotated = (
            boxes_annotated > 0
            or meta.get("has_labels", False)
            or bool(meta.get("label_count"))
            or await asyncio.to_thread(self._is_annotated_exported, dataset_dir, meta.get("version", "v1"))
        )
        has_artifact_source = annotated or ann_files > 0
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
        # 封板产物直接输出到 dataset_dir，符合标准 YOLO 数据集格式：
        #   {dataset_dir}/data.yaml
        #   {dataset_dir}/images/{train,val,test}/
        #   {dataset_dir}/labels/{train,val,test}/
        model_code = meta.get("model_id") or ""
        if model_code:
            try:
                mcfg = _load_json(settings.REGISTRY_DIR / model_code / "model.json")
                model_code = mcfg.get("model_code") or model_code
            except Exception:
                pass
        seal_payload = None
        if has_artifact_source:
            ratio = split_ratio or meta.get("split_ratio") or {"train": 0.7, "val": 0.15, "test": 0.15}
            sv = int(meta.get("sealed_version_count") or 0) + 1
            # 封板产物直接输出到 dataset_dir（标准 YOLO 格式）
            try:
                seal_payload = await asyncio.to_thread(
                    self._build_training_artifacts, dataset_dir, meta, ratio, dataset_dir, True
                )
            except ValueError as e:
                if not force:
                    raise
                # force 是"时间窗口兜底"语义：产物生成失败时不阻断封板打标
                logger.warning(f"dataset {dataset_id} 封板产物生成失败（force 放行）: {e}")
                meta["seal_skipped_reason"] = str(e)
            if seal_payload is not None:
                meta.pop("seal_skipped_reason", None)
                meta["sealed_version_count"] = sv
                meta["sealed_version"] = f"v{sv}"
                meta["sealed_out"] = str(dataset_dir.relative_to(settings.DATA_DIR))
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
        else:
            from src.db import shadow
            await shadow.sync_dataset(dataset_id)

        logger.info(f"dataset {dataset_id} sealed: has_labels={annotated} ann_files={ann_files} force={force} split={seal_payload and seal_payload.get('counts')}")
        return meta

    async def _backfill_seal_artifacts(self, dataset_dir: Path, meta_path: Path, meta: dict) -> dict:
        """补生成封板产物（#3 自愈通道）。

        历史数据里有一批"已标记 sealed 但目录内没有 data.yaml/划分"的数据集
        （早期 force 封板、或封板时产物生成被跳过）。这里只做产物补齐：
        不改 stage、不改 display_name、不递增 sealed_version_count，
        避免触发封板更名（名=id）把目录/PG 记录/桶账本整体改名。
        """
        dataset_id = meta.get("dataset_id") or dataset_dir.name
        ratio = meta.get("split_ratio") or {"train": 0.7, "val": 0.15, "test": 0.15}
        try:
            # label_guard=False：走到这里说明是 force 补救，序号冲突不重复拦
            payload = await asyncio.to_thread(
                self._build_training_artifacts, dataset_dir, meta, ratio, dataset_dir, False
            )
        except Exception as e:
            logger.warning(f"dataset {dataset_id} 补生成封板产物失败: {e}")
            meta["seal_artifacts_error"] = str(e)
            await asyncio.to_thread(_save_json, meta_path, meta)
            from src.db import shadow
            await shadow.sync_dataset(dataset_id)
            return meta
        meta.pop("seal_artifacts_error", None)
        meta["sealed_out"] = str(dataset_dir.relative_to(settings.DATA_DIR))
        meta["sealed_split"] = payload
        meta["label_validation"] = payload.get("label_validation")
        if payload.get("counts"):
            meta["split_counts"] = payload["counts"]
        if payload.get("sample_type_stats"):
            meta["sample_type_stats"] = payload["sample_type_stats"]
        await asyncio.to_thread(_save_json, meta_path, meta)
        from src.db import shadow
        await shadow.sync_dataset(dataset_id)
        logger.info(f"dataset {dataset_id} seal artifacts backfilled: split={payload.get('counts')}")
        return meta

    async def _rename_dataset_identity(self, old_id: str, new_id: str) -> dict:
        """封板更名（名=id）：把数据集唯一标识从 ds_xxx 整体迁移到 {code}_{时间戳}_v{N}。
        同步迁移：磁盘目录、meta.dataset_id、标注任务绑定、上传包、桶账本、PG 影子库。
        返回更名后的 meta。
        """
        new_meta = await asyncio.to_thread(self._rename_dataset_identity_sync, old_id, new_id)
        # 影子库收口：更名 = 删旧行 + 写新行（只 upsert 新行会留下旧 id 的孤儿记录），
        # 且任务绑定的 dataset_id、桶账本引用都在同步侧改过，这里按真值文件重灌。
        from src.db import shadow
        await shadow.drop_dataset(old_id)
        await shadow.sync_dataset(new_id)
        await shadow.sync_buckets()
        for tid in await asyncio.to_thread(self._tasks_bound_to, new_id):
            await shadow.sync_task(tid)
        return new_meta

    @staticmethod
    def _tasks_bound_to(dataset_id: str) -> list:
        """扫描标注任务真值文件，返回绑定到该数据集的 task_id 列表。"""
        out = []
        try:
            root = settings.ANNOTATIONS_DIR
            if not root.exists():
                return out
            for tdir in root.iterdir():
                tf = tdir / "task.json"
                if not tf.is_file():
                    continue
                try:
                    m = _load_json(tf)
                except Exception:  # noqa: BLE001
                    continue
                if m.get("dataset_id") == dataset_id:
                    out.append(m.get("task_id") or tdir.name)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"scan tasks bound to {dataset_id} failed: {e}")
        return out

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

        # 5) 路径引用跟随：更名后新目录内 data.yaml 若仍带指向旧目录的 path 键，删除之
        #    （B-09：新封板产物已不写 path，ultralytics 自动按 yaml 所在目录解析；
        #     此处兜底处理更名前生成的旧 yaml，train/val 是相对路径，删掉 path 即可自愈）。
        #    meta 中记录的旧绝对路径字段（sealed_out 等）一并替换。
        old_prefix = str(old_dir.relative_to(settings.DATA_DIR))
        new_prefix = str(new_dir.relative_to(settings.DATA_DIR))
        try:
            for yf in new_dir.rglob("data.yaml"):
                try:
                    yd = _load_yaml(yf)
                    if yd and "path" in yd and old_prefix in str(yd.get("path", "")):
                        yd.pop("path")
                        _save_yaml(yf, yd)
                except Exception:  # noqa: BLE001
                    continue

            def _fix_path_refs(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, str) and v.startswith(old_prefix):
                            obj[k] = new_prefix + v[len(old_prefix):]
                        elif isinstance(v, (dict, list)):
                            _fix_path_refs(v)
                elif isinstance(obj, list):
                    for it in obj:
                        _fix_path_refs(it)

            _fix_path_refs(meta)
            _save_json(mp, meta)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"rename {old_id}: path refs update failed: {e}")

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

        validation = await asyncio.to_thread(_run)
        from src.db import shadow
        await shadow.sync_dataset(dataset_id)
        return validation

    async def delete_dataset(self, dataset_id: str):
        """删除数据集：目录改名标记 + 专用后台线程删磁盘，接口立即返回。
        同步清理桶账本（buckets.json）中该数据集的灌桶进度，防止只增不减。
        """
        dataset_dir = self.datasets_dir / dataset_id

        if not await asyncio.to_thread(lambda: dataset_dir.exists()):
            return None

        # 读 meta 拿到上传压缩包名 + 桶灌信息（先读，改名后目录名已变）
        meta_path = dataset_dir / "meta.json"
        archive_name = None
        model_id = None
        bucket_pending = 0
        bucket_sealed = 0
        if await asyncio.to_thread(lambda: meta_path.exists()):
            try:
                meta = await asyncio.to_thread(lambda: _load_json(meta_path))
                archive_name = meta.get("archive")
                model_id = meta.get("model_id") or ""
                bucket_pending = int(meta.get("bucket_pending") or 0)
                bucket_sealed = int(meta.get("bucket_sealed") or 0)
            except Exception:
                pass

        # 清理桶账本：从 receiving.members 移除该数据集，减少 poured 计数
        if model_id and (bucket_pending > 0 or bucket_sealed > 0):
            try:
                bf = settings.DATA_DIR / "buckets.json"
                if bf.exists():
                    state = _load_json(bf) or {"models": {}}
                    models = state.get("models") or {}
                    mst = models.get(model_id)
                    if mst:
                        recv = mst.get("receiving") or {}
                        # 从 members 中移除该数据集
                        members = recv.get("members") or []
                        removed_count = 0
                        new_members = []
                        for m in members:
                            if isinstance(m, dict) and m.get("dataset_id") == dataset_id:
                                removed_count += int(m.get("count") or 0)
                            else:
                                new_members.append(m)
                        recv["members"] = new_members
                        # 减少 poured 计数
                        if removed_count > 0:
                            recv["poured"] = max(0, int(recv.get("poured") or 0) - removed_count)
                        # 清理 sealed 快照中的引用
                        for snap in mst.get("sealed") or []:
                            if isinstance(snap, dict):
                                snap_members = snap.get("members") or []
                                snap["members"] = [
                                    m for m in snap_members
                                    if not (isinstance(m, dict) and m.get("dataset_id") == dataset_id)
                                ]
                    _save_json(bf, state)
                    logger.info(
                        f"delete {dataset_id}: bucket cleanup model={model_id} "
                        f"removed={removed_count} poured={recv.get('poured', 0) if mst else 0}"
                    )
            except Exception as e:
                logger.warning(f"delete {dataset_id}: bucket cleanup failed: {e}")

        # 1) 同一卷内改名（瞬时元数据操作）→ 前端列表立刻看不到该数据集
        renamed = dataset_dir.with_name(f"{dataset_id}__deleting")
        renamed_ok = await asyncio.to_thread(_rename_dir, dataset_dir, renamed)

        # 2) 待删路径交给专用后台删除线程逐个处理，不等待
        targets = [renamed if renamed_ok else dataset_dir]
        if archive_name:
            targets.append(self.uploads_dir / archive_name)
        targets.append(self.uploads_dir / f"{dataset_id}.zip")
        _enqueue_deletion(targets)

        # 影子库收口：数据集行删除 + 桶账本按真值文件重灌（上方 members/poured 已就地改过）
        from src.db import shadow
        await shadow.drop_dataset(dataset_id)
        await shadow.sync_buckets()

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
                # 供影子库收口使用（对外响应前会剔除）
                "annotation_task_id": meta.get("annotation_task_id"),
            }

        result = await asyncio.to_thread(_sync)
        # 影子库收口：image_count 变化 + 关联标注任务里该图行移除
        from src.db import shadow
        await shadow.sync_dataset(dataset_id)
        tid = result.pop("annotation_task_id", None)
        if tid:
            await shadow.remove_task_item(tid, result.get("image_id") or image_id)
        return result

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
    #   - 桶累计 >= settings.SEAL_MIN_IMAGES 自动封板并产出 v{N} 快照；
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

        ratio = {"train": 0.7, "val": 0.15, "test": 0.15}
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
            # 快照继承来源模型的算法库归属，保证训练产出仍能定位到库
            "library_root_id": self._library_root_id_of(model_id),
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
            "sealed_out": str(ds_dir.relative_to(settings.DATA_DIR)),
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
        """灌桶同步核心：扫描标完未归档源包 → FIFO 灌桶 → 满 settings.SEAL_MIN_IMAGES 连环自动封板。
        幂等：重复调用不重复灌/封。返回本次变更摘要。
        """
        if settings.SEAL_MIN_IMAGES <= 0:
            return {"poured": [], "sealed": [], "skipped": 0, "note": "SEAL_MIN_IMAGES=0 关闭"}
        state = self._load_bucket_state()
        self._bucket_state = state
        models_state = state.setdefault("models", {})
        poured_events, sealed_events = [], []
        pending_review = []  # 标完但未按「审核通过」的包：不灌桶，前端据此展示待审核
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
                    if not meta.get("review_approved"):
                        # 人工审核门槛：标注完成但未点「审核通过」→ 暂不灌桶，保持可编辑
                        pending_review.append(dataset_id)
                        skipped += 1
                        continue
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
                    # 3) 满则连环封板（剩余超过 settings.SEAL_MIN_IMAGES 时可能连续产出多个版本）
                    while mst["receiving"]["poured"] >= settings.SEAL_MIN_IMAGES:
                        snap = self._seal_receiving_bucket(model_id, mst)
                        if not snap:
                            break
                        sealed_events.append({"model_id": model_id, **snap})
        finally:
            self._save_bucket_state(state)
            self._bucket_state = None
        return {"poured": poured_events, "sealed": sealed_events, "skipped": skipped,
                "pending_review": pending_review}

    async def advance_buckets(self) -> dict:
        """按模型滚动灌桶（幂等）：A 包「标完 + 审核通过」后灌桶，满 settings.SEAL_MIN_IMAGES 自动封板出成品数据集。
        供前端在数据集页加载 / 标注完成返回时调用。
        """
        def _locked():
            with _BUCKET_LOCK:
                return self._advance_buckets_sync()
        events = await asyncio.to_thread(_locked)
        # 影子库收口：灌桶/封板会改桶账本 + 源包 meta + 新建快照数据集，
        # 同步侧拿不到受影响 id，用 mtime 水位扫一遍（未变动的文件不写库）
        from src.db import shadow
        await shadow.sync_buckets()
        await shadow.refresh_all_datasets()
        return events

    async def approve_review(self, dataset_id: str) -> dict:
        """审核通过：人工确认标注无误后标记，之后才允许自动灌桶/封板。
        标注完成的包在未确认前保持可编辑（不被灌桶归档）。
        """
        def _approve_sync():
            ds_dir = self.datasets_dir / dataset_id
            meta_path = ds_dir / "meta.json"
            if not meta_path.exists():
                raise ValueError(f"数据集 {dataset_id} 不存在")
            meta = _load_json(meta_path)
            _ensure_stage_fields(meta)
            if meta.get("bucket_archived"):
                raise ValueError("该数据集已灌桶归档，无需审核确认；如需修改请使用「撤回」")
            if meta.get("review_approved"):
                return meta
            total = int(meta.get("image_count") or 0)
            annotated = self._count_annotated_files(ds_dir)
            if total and annotated < total:
                raise ValueError(f"标注未完成（{annotated}/{total}），请全部标注后再确认审核")
            meta["review_approved"] = True
            meta["review_approved_at"] = datetime.now().isoformat()
            _save_json(meta_path, meta)
            logger.info(f"review: {dataset_id} approved -> eligible for bucket pour")
            return meta
        result = await asyncio.to_thread(_approve_sync)
        from src.db import shadow
        await shadow.sync_dataset(dataset_id)
        return result

    async def unpour_bucket(self, dataset_id: str) -> dict:
        """撤回（解封 A 包）：把「已灌入接收桶但尚未封板」的图片从桶账本退回，
        数据集恢复可编辑状态。已有图片封入快照（bucket_sealed>0）的包不可撤回。
        """
        def _unpour_sync():
            ds_dir = self.datasets_dir / dataset_id
            meta_path = ds_dir / "meta.json"
            if not meta_path.exists():
                raise ValueError(f"数据集 {dataset_id} 不存在")
            meta = _load_json(meta_path)
            sealed_n = int(meta.get("bucket_sealed") or 0)
            if sealed_n > 0:
                raise ValueError(
                    f"该数据集已有 {sealed_n} 张图封入快照（不可变），无法撤回；"
                    f"仅未封板的灌桶量可退回编辑"
                )
            model_id = str(meta.get("model_id") or "")
            with _BUCKET_LOCK:
                state = self._load_bucket_state()
                mst = (state.get("models") or {}).get(model_id) if model_id else None
                if mst:
                    recv = mst.get("receiving") or {}
                    members = recv.get("members") or []
                    keep = [m for m in members if m.get("dataset_id") != dataset_id]
                    removed = (
                        sum(int(m.get("count") or 0) for m in members)
                        - sum(int(m.get("count") or 0) for m in keep)
                    )
                    if removed > 0:
                        recv["members"] = keep
                        recv["poured"] = max(0, int(recv.get("poured") or 0) - removed)
                        mst["receiving"] = recv
                        logger.info(f"unpour: {dataset_id} 退回 {removed} 张（B{recv.get('index')}）")
                self._save_bucket_state(state)
            meta["bucket_pending"] = 0
            meta["bucket_archived"] = False
            meta.pop("bucket_archived_at", None)
            meta["review_approved"] = False
            meta.pop("review_approved_at", None)
            _save_json(meta_path, meta)
            return meta
        result = await asyncio.to_thread(_unpour_sync)
        # 影子库收口：撤回同时改了桶账本与数据集 meta
        from src.db import shadow
        await shadow.sync_buckets()
        await shadow.sync_dataset(dataset_id)
        return result

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
            if not meta.get("review_approved"):
                # 人工审核门槛：未点「审核通过」的数据集不因超期被自动封板
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

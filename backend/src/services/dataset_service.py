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
from fastapi import UploadFile
from src.core.settings import settings
from src.utils.fs_tree import build_tree

logger = logging.getLogger(__name__)

# 支持的图片后缀
IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 支持的上传压缩包格式
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
# 排除可视化预览图（如 *_result.jpg）
EXCLUDE_KEYWORDS = ("_result", "_predict", "_detect")


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
        
    async def upload_dataset(self, file: UploadFile):
        """上传数据集压缩包（zip / tar / tar.gz / tgz）"""
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
        
        # 保存元数据
        meta = {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "archive": archive_name,
            "size": len(content),
            "uploaded_at": datetime.now().isoformat(),
            "status": "uploaded"
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
        # 1) 已符合 images/{train,val,test}(+labels) 标准结构 → 保持原样
        # 2) 其他任意目录（图片与标注同目录 / labels 子目录 / 原始无标注数据集等）
        #    → 自动重组为 images/{train,val,test} + labels/{train,val,test} 标准结构
        detected_classes = classes or []
        is_standard = await asyncio.to_thread(self._is_standard_structure, version_dir)
        if not is_standard:
            # 重组前先从解压内容中读取类别（zip 自带的 classes.txt / data.yaml），
            # 供重组阶段写 classes.txt 使用（重组会清理原目录）
            if not detected_classes:
                detected_classes = await asyncio.to_thread(self._load_classes_from_files, version_dir) or []
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
            
            # 如果从 YAML 中没有读取到类别，且存在标签目录，则从标签文件检测
            if not detected_classes and labels_dir:
                detected_classes = await asyncio.to_thread(self._detect_classes, labels_dir)
                logger.info(f"Detected classes from labels: {detected_classes}")
        
        # 确保有类别列表（如果没有检测到，使用空列表）
        if not detected_classes:
            detected_classes = []
        
        # 生成data.yaml
        images_dir_train = images_dir / "train"
        images_dir_val = images_dir / "val"
        images_dir_test = images_dir / "test"
        yaml_content = {
            "path": str(version_dir.absolute()),
            "train": "images/train" if images_dir_train.exists() else "images",
            "val": "images/val" if images_dir_val.exists() else "images",
            "nc": len(detected_classes),
            "names": detected_classes
        }
        if images_dir_test.exists():
            yaml_content["test"] = "images/test"
        
        yaml_path = version_dir / "data.yaml"
        await asyncio.to_thread(_save_yaml, yaml_path, yaml_content)
        
        # 标注情况：标准结构直接用 label_count，重组后使用重组结果
        final_has_labels = (label_count > 0) if has_labels is None else has_labels
        
        # 更新元数据
        meta.update({
            "status": "prepared",
            "version": version,
            "image_count": image_count,
            "label_count": label_count,
            "has_labels": final_has_labels,
            "classes": detected_classes,
            "prepared_at": datetime.now().isoformat()
        })
        
        await asyncio.to_thread(_save_json, meta_path, meta)
        
        logger.info(f"prepare {dataset_id} done: total {time.time() - t0:.1f}s, images={image_count}, labels={label_count}")
        
        return {
            "dataset_id": dataset_id,
            "version": version,
            "image_count": image_count,
            "label_count": label_count,
            "has_labels": final_has_labels,
            "classes": detected_classes
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
        """把非标准结构的数据集重组为 images/{train,val,test} + labels/{train,val,test} 标准结构。

        - 有标注（YOLO txt 与图片同目录 / 独立 labels 目录 / 自带 train-val-test 分组）：
          自动划分 train/val/test，无匹配标注的图片跳过并告警
        - 无标注原始数据集：仅整理图片并划分，返回 False（提示后续使用 AI 预标注）
        返回是否有可用标注。
        """
        # 归一化划分比例
        t = float(split_ratio.get("train", 0.8))
        v = float(split_ratio.get("val", 0.2))
        te = float(split_ratio.get("test", 0.0))
        total = t + v + te
        if total <= 0:
            t, v, te = 0.8, 0.2, 0.0
        else:
            t, v, te = t / total, v / total, te / total
        rng = random.Random(0)

        def split_of(p: Path):
            low = [x.lower() for x in p.parts]
            for s in ("train", "val", "valid", "test"):
                if s in low:
                    return "val" if s == "valid" else s
            return None

        # 1) 收集图片与 txt 标注（跳过隐藏文件 / 预览图 / 图片不出现在 labels 下）
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

        # 2) txt 与图片按 stem 配对，识别自带划分
        matched = []  # [(img, txt, split|None)]
        used = set()
        for t_txt in txts:
            im = img_by_stem.get(t_txt.stem)
            if im is None:
                continue
            matched.append((im, t_txt, split_of(t_txt) or split_of(im)))
            used.add(im)
        orphans = sorted(set(images) - used)  # 无对应标注的图片
        has_labels = bool(matched)

        # 3) 输出到暂存目录
        staging = root_dir / ".staging"

        def emit(split, im, txt):
            d_img = staging / "images" / split
            d_img.mkdir(parents=True, exist_ok=True)
            # 用 move 而非 copy：staging 与源在同一文件系统（根目录内），rename 瞬时完成，
            # 避免大数据集复制双份 IO 导致准备耗时翻倍（原始 zip 保留在 uploads，可随时重准备）
            shutil.move(str(im), str(d_img / im.name))
            if txt is not None:
                d_lbl = staging / "labels" / split
                d_lbl.mkdir(parents=True, exist_ok=True)
                shutil.move(str(txt), str(d_lbl / txt.name))

        if has_labels:
            explicit = {s for _, _, s in matched if s}
            if explicit:
                # 自带划分：按组输出；无法分组的扁平配对并入 train
                for im, t_txt, s in matched:
                    emit(s if s else "train", im, t_txt)
            else:
                # 图片与标注同目录：按比例自动划分
                rng.shuffle(matched)
                n = len(matched)
                n_train = int(n * t)
                n_val = int(n * v)
                for im, t_txt, _ in matched[:n_train]:
                    emit("train", im, t_txt)
                for im, t_txt, _ in matched[n_train:n_train + n_val]:
                    emit("val", im, t_txt)
                for im, t_txt, _ in matched[n_train + n_val:]:
                    emit("test", im, t_txt)
        else:
            # 无标注原始数据集：仅整理图片并划分
            rng.shuffle(images)
            n = len(images)
            n_train = int(n * t)
            n_val = int(n * v)
            for im in images[:n_train]:
                emit("train", im, None)
            for im in images[n_train:n_train + n_val]:
                emit("val", im, None)
            for im in images[n_train + n_val:]:
                emit("test", im, None)

        if classes:
            (staging / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

        # 4) 清空根目录并移入标准结构（zip 原始文件保留在 uploads，可随时重新准备）
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

        if has_labels and orphans:
            logger.warning(
                f"reorganize: {len(orphans)} images without labels skipped, e.g. {[p.name for p in orphans[:3]]}"
            )
        if not has_labels and images:
            logger.warning("reorganize: dataset has no labels, images split only (nc=0). Use AI annotation before training.")
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
        """判断数据集是否已在标注页执行过"导出YOLO(自动划分)"

        标注导出会生成 labels/{train,val,test} 划分结构（含 .txt 标签），
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
                            # 标记是否为标注导出的数据集（是否有可用的训练标签）
                            meta["annotated"] = self._is_annotated_exported(
                                dataset_dir, meta.get("version", "v1")
                            )
                            datasets.append(meta)
                        except Exception:
                            continue
            return datasets
        
        datasets = await asyncio.to_thread(_list_datasets_sync)
        return {"datasets": sorted(datasets, key=lambda x: x.get("uploaded_at", ""), reverse=True)}
    
    async def get_dataset(self, dataset_id: str):
        """获取数据集详情"""
        dataset_dir = self.datasets_dir / dataset_id
        meta_path = dataset_dir / "meta.json"
        
        if not await asyncio.to_thread(lambda: meta_path.exists()):
            return None
        
        meta = await asyncio.to_thread(_load_json, meta_path)
        
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
        """更新数据集信息"""
        dataset_dir = self.datasets_dir / dataset_id
        meta_path = dataset_dir / "meta.json"
        
        if not await asyncio.to_thread(lambda: meta_path.exists()):
            return None
        
        meta = await asyncio.to_thread(_load_json, meta_path)
        
        # 更新字段
        if request.description is not None:
            meta["description"] = request.description
        if request.tags is not None:
            meta["tags"] = request.tags
        
        meta["updated_at"] = datetime.now().isoformat()
        
        await asyncio.to_thread(_save_json, meta_path, meta)
        
        return meta
    
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

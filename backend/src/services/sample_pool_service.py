"""样本池服务（1.6）

两种池：
- 困难样本库（hard，按模型 1:1）：训练/推理中识别为困难、易漏检的样本（图片 + 标注），
  存于 sample_pool/hard/{model_id}/{images,labels,meta.json}；meta.json 记录该池的类别名
  （加入时从数据集的 data.yaml 读取），训练抽样并入时按类别名重映射 class id。
- 空白样本库（background，全局共享 0:N）：无目标的背景图（负样本），
  存于 sample_pool/background/images（附同名空 labels/*.txt），供任意模型训练混入。
"""
import json
import shutil
import asyncio
from pathlib import Path
from datetime import datetime
from src.core.settings import settings

# 支持的图片后缀（与 dataset_service 保持一致）
IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_yaml(path: Path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_names(names) -> list:
    """将 data.yaml 的 names（list/dict）归一化为按类 ID 排序的列表"""
    if names is None:
        return []
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    if isinstance(names, list):
        return [str(n) for n in names]
    return []


class SamplePoolService:
    def __init__(self):
        self.pool_dir = settings.SAMPLE_POOL_DIR
        self.hard_root = settings.HARD_POOL_DIR
        self.bg_root = settings.BACKGROUND_POOL_DIR
        self.datasets_dir = settings.DATASETS_DIR

    # ---------------- 困难样本（按模型 1:1） ----------------

    async def add_hard_samples(self, dataset_id: str, model_id: str,
                               image_names: list, version: str = "v1") -> dict:
        """把数据集中的指定图片（连同标注）加入某模型的困难样本库。

        类别名空间：以数据集 data.yaml 的 names 为准写入池 meta.json（后续训练
        抽样并入统一类别空间时按类别名重映射）。
        """
        if not model_id:
            raise ValueError("困难样本必须归属到模型（model_id）才能参与该模型的训练")
        names = [n.strip() for n in (image_names or []) if n and n.strip()]
        if not names:
            raise ValueError("请选择至少一张图片")

        def _sync() -> dict:
            version_dir = self.datasets_dir / dataset_id / version
            if not version_dir.exists():
                raise ValueError(f"数据集 {dataset_id}/{version} 不存在")
            base = Path(settings.BASE_DIR)
            yaml_path = version_dir / "data.yaml"
            pool_names = []
            if yaml_path.exists():
                try:
                    cfg = _load_yaml(yaml_path)
                    pool_names = _normalize_names(cfg.get("names", []))
                except Exception:
                    pool_names = []

            images_dir = None
            for cand in list(version_dir.rglob("images")):
                images_dir = cand
                break
            labels_dir = None
            for cand in list(version_dir.rglob("labels")):
                labels_dir = cand
                break
            if images_dir is None:
                raise ValueError(f"数据集 {dataset_id}/{version} 未准备（无 images 目录）")

            # 目标目录
            target = self.hard_root / model_id
            t_img = target / "images"
            t_lbl = target / "labels"
            t_img.mkdir(parents=True, exist_ok=True)
            t_lbl.mkdir(parents=True, exist_ok=True)

            added, skipped = 0, 0
            used_img, used_lbl = set(), set()
            # 名称 -> 命中图片（按 split 目录递归查找）
            for want in names:
                if want in used_img:
                    continue
                found = None
                for p in images_dir.rglob("*"):
                    if p.is_file() and p.stem == want and p.suffix.lower() in IMG_SUFFIX:
                        found = p
                        break
                if found is None:
                    skipped += 1
                    continue
                # 复制图片
                dst_im = t_img / found.name
                if not dst_im.exists():
                    shutil.copy2(found, dst_im)
                used_img.add(want)
                # 复制同名标注（找不到则跳过——困难样本需要标注）
                if labels_dir is not None:
                    for lp in labels_dir.rglob(f"{want}.txt"):
                        dst_lb = t_lbl / lp.name
                        if not dst_lb.exists():
                            shutil.copy2(lp, dst_lb)
                        used_lbl.add(lp.name)
                        break
                added += 1

            # 写入/更新池 meta（类别名 + 统计）
            meta_p = target / "meta.json"
            meta = {}
            if meta_p.exists():
                try:
                    meta = _load_json(meta_p)
                except Exception:
                    meta = {}
            meta.update({
                "model_id": model_id,
                "names": meta.get("names") or pool_names,
                "image_count": len([p for p in t_img.rglob("*") if p.is_file() and p.suffix.lower() in IMG_SUFFIX]),
                "label_count": len(used_lbl),
                "updated_at": datetime.now().isoformat(),
            })
            _save_json(meta_p, meta)

            return {"model_id": model_id, "added": added, "skipped": skipped,
                    "pool_image_count": meta["image_count"]}

        return await asyncio.to_thread(_sync)

    # ---------------- 空白样本库（全局共享 0:N） ----------------

    async def add_background_samples(self, dataset_id: str, image_names: list,
                                     version: str = "v1") -> dict:
        """把数据集中的指定图片作为无目标背景（负样本）加入全局空白样本库。

        每张图配一个同名空 labels/*.txt（YOLO 视无标注文件为空标注 = 背景）。
        """
        names = [n.strip() for n in (image_names or []) if n and n.strip()]
        if not names:
            raise ValueError("请选择至少一张图片")

        def _sync() -> dict:
            version_dir = self.datasets_dir / dataset_id / version
            if not version_dir.exists():
                raise ValueError(f"数据集 {dataset_id}/{version} 不存在")
            images_dir = None
            for cand in list(version_dir.rglob("images")):
                images_dir = cand
                break
            if images_dir is None:
                raise ValueError(f"数据集 {dataset_id}/{version} 未准备（无 images 目录）")

            t_img = self.bg_root / "images"
            t_lbl = self.bg_root / "labels"
            t_img.mkdir(parents=True, exist_ok=True)
            t_lbl.mkdir(parents=True, exist_ok=True)

            added, skipped = 0, 0
            for want in names:
                found = None
                for p in images_dir.rglob("*"):
                    if p.is_file() and p.stem == want and p.suffix.lower() in IMG_SUFFIX:
                        found = p
                        break
                if found is None:
                    skipped += 1
                    continue
                dst_im = t_img / found.name
                if not dst_im.exists():
                    shutil.copy2(found, dst_im)
                # 空标注文件（覆盖式补齐：若已有则跳过）
                lb = t_lbl / f"{found.stem}.txt"
                if not lb.exists():
                    lb.touch()
                added += 1
            return {"added": added, "skipped": skipped}
        return await asyncio.to_thread(_sync)

    # ---------------- 查询 / 清理 ----------------

    async def list_pool(self) -> dict:
        """汇总样本池状态：困难样本（按模型）+ 空白样本（全局）"""
        def _sync() -> dict:
            hard = []
            if self.hard_root.exists():
                for model_dir in sorted(self.hard_root.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    imgs_dir = model_dir / "images"
                    meta_p = model_dir / "meta.json"
                    meta = {}
                    if meta_p.exists():
                        try:
                            meta = _load_json(meta_p)
                        except Exception:
                            pass
                    image_count = 0
                    if imgs_dir.exists():
                        image_count = len([p for p in imgs_dir.rglob("*")
                                          if p.is_file() and p.suffix.lower() in IMG_SUFFIX])
                    hard.append({
                        "model_id": model_dir.name,
                        "model_code": meta.get("model_code") or "",
                        "names": meta.get("names") or [],
                        "image_count": image_count,
                        "label_count": meta.get("label_count", 0),
                        "updated_at": meta.get("updated_at") or "",
                    })
            background = 0
            bg_imgs = self.bg_root / "images"
            if bg_imgs.exists():
                background = len([p for p in bg_imgs.rglob("*")
                                  if p.is_file() and p.suffix.lower() in IMG_SUFFIX])
            return {"hard": hard, "background": background}
        return await asyncio.to_thread(_sync)

    async def clear_hard(self, model_id: str) -> dict:
        """清空某模型的困难样本库"""
        def _sync():
            model_dir = self.hard_root / model_id
            if model_dir.exists():
                shutil.rmtree(model_dir)
            return {"model_id": model_id, "cleared": True}
        return await asyncio.to_thread(_sync)

    async def clear_background(self) -> dict:
        """清空空白样本库"""
        def _sync():
            for root in (self.bg_root / "images", self.bg_root / "labels"):
                if root.exists():
                    for child in root.iterdir():
                        if child.is_dir():
                            shutil.rmtree(child, ignore_errors=True)
                        else:
                            try:
                                child.unlink()
                            except OSError:
                                pass
            return {"cleared": True}
        return await asyncio.to_thread(_sync)
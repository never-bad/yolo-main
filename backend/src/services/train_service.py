import json
import os
import re
import random
import subprocess
import signal
import shutil
import asyncio
import yaml
from pathlib import Path
from datetime import datetime
import psutil
from src.core.settings import settings
from src.utils.fs_tree import build_tree
from src.yolo.gatekeeper import infer_business, _normalize_classes
from src.yolo.train_script import ensure_local_base_weights
from src.services.dataset_service import (
    STAGE_TRAINING, STAGE_COMPLETED, STAGE_FAILED,
    TRAIN_STATUS_INCOMPLETE, TRAIN_STATUS_COMPLETED,
    TRAINABLE_STAGES, _ensure_stage_fields,
)


def _load_json(path: Path):
    """同步加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict):
    """同步保存 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _delete_file(path: Path):
    """同步删除文件"""
    if path.exists():
        path.unlink()


def resolve_registry_weights(meta: dict, registry_dir: Path) -> Path | None:
    """解析模型权重路径：绝对路径失效时按 model_id 在仓库内重定位。

    model.json 的 weights_path 记录的是模型生成机器的绝对路径；换机器/容器部署后
    该路径失效，但权重本身仍随仓库迁移在 registry/<model_id>/weights/ 下（best.pt 优先）。
    """
    weights_path = meta.get("weights_path")
    if weights_path and os.path.exists(weights_path):
        return Path(weights_path)
    model_id = meta.get("model_id")
    if model_id:
        wdir = registry_dir / model_id / "weights"
        if wdir.exists():
            best = wdir / "best.pt"
            if best.exists():
                return best
            for wf in wdir.glob("*.pt"):
                return wf
        for cand in registry_dir.glob(f"{model_id}/**/best.pt"):
            return cand
    return Path(weights_path) if weights_path else None


def _delete_directory(path: Path):
    """同步删除目录"""
    if path.exists():
        shutil.rmtree(path)


def _norm_class_name(name) -> str:
    """类别名归一化（小写 + 去空白），用于与模型标签字典 english_code 匹配"""
    return re.sub(r"\s+", "", str(name or "")).strip().lower()


def _names_list(names) -> list:
    """将 data.yaml 的 names（list / dict）归一化为按类 ID 排序的类别名列表"""
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    if isinstance(names, list):
        return [str(n) for n in names]
    return []


def _link_or_copy(src: Path, dst: Path):
    """优先硬链接（零拷贝），失败（跨卷/无权限）回退普通复制"""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class LabelFilterError(ValueError):
    """阶段0.3 标签过滤失败（如数据集与微调模型类别零交集）→ 应中断创建任务而非回退原数据"""


class _RecoveredProcess:
    """后端重启后，为仍在运行的孤儿训练进程提供的轻量句柄适配器"""
    def __init__(self, pid: int):
        self.pid = pid
        self._proc = psutil.Process(pid)

    def poll(self):
        try:
            return self._proc.poll()
        except psutil.NoSuchProcess:
            return 0

    def wait(self, timeout=None):
        try:
            self._proc.wait(timeout=timeout)
        except psutil.NoSuchProcess:
            return

    def send_signal(self, sig):
        try:
            self._proc.send_signal(sig)
        except psutil.NoSuchProcess:
            pass

    def kill(self):
        try:
            self._proc.kill()
        except psutil.NoSuchProcess:
            pass


class TrainService:
    def __init__(self):
        self.jobs_dir = settings.JOBS_DIR
        self.datasets_dir = settings.DATASETS_DIR
        self.registry_dir = settings.REGISTRY_DIR
        self.running_processes = {}

    @staticmethod
    def _count_images(path: Path) -> int:
        """统计目录下图片数量（jpg/jpeg/png/bmp/webp）"""
        if not path.exists():
            return 0
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sum(
            1 for f in path.rglob("*") if f.is_file() and f.suffix.lower() in exts
        )

    @staticmethod
    def _default_batch() -> int:
        """按当前 GPU 显存返回安全批次大小（避免 -1 自动校准在容器内触发 [Errno 22]）"""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if vram_gb >= 40:
                    return 32
                if vram_gb >= 20:
                    return 16
                return 8
            return 4
        except Exception:
            return 16

    async def suggest_params(self, dataset_id: str, version: str = "v1",
                             base_model_id: str = None) -> dict:
        """根据当前硬件（GPU显存）与数据集规模，自动推荐训练参数。

        推荐范围：epochs / imgsz / batch（基础）+ lr0 / optimizer / weight_decay / patience（高级）。
        - 新手可直接使用推荐值；专家可在此基础上手动微调。
        - 微调模式（base_model_id）会自动降低学习率、缩短早停轮数。
        """
        # ---------- 1. 硬件信息 ----------
        device_info = {"type": "cpu", "name": "CPU", "vram_gb": 0}
        cuda_ok = False
        try:
            import torch
            if torch.cuda.is_available():
                cuda_ok = True
                props = torch.cuda.get_device_properties(0)
                vram_gb = round(props.total_memory / (1024 ** 3), 1)
                device_info = {
                    "type": "cuda",
                    "name": torch.cuda.get_device_name(0),
                    "vram_gb": vram_gb,
                }
        except Exception:
            cuda_ok = False

        # ---------- 2. 数据集统计 ----------
        image_count = 0
        class_count = 0
        yaml_path = self.datasets_dir / dataset_id / version / "data.yaml"
        if yaml_path.exists():
            try:
                data_cfg = await asyncio.to_thread(self._load_yaml, yaml_path)
                names = data_cfg.get("names", {})
                class_count = len(names) if isinstance(names, (list, dict)) else 0
                base = yaml_path.parent
                # 统计 train/val 图片数
                for split_key in ("train", "val"):
                    rel = data_cfg.get(split_key)
                    if not rel:
                        continue
                    p = Path(rel)
                    if not p.is_absolute():
                        p = base / p
                    image_count += await asyncio.to_thread(self._count_images, p)
            except Exception:
                pass

        # ---------- 3. 推荐规则 ----------
        # 批次大小按显存档位估算（CPU 保守给最小档）
        if cuda_ok:
            vram = device_info.get("vram_gb", 0)
            if vram >= 40:
                batch, imgsz = 32, 640     # A100 / 4090 / 5090 等大显存
            elif vram >= 20:
                batch, imgsz = 16, 640
            elif vram >= 10:
                batch, imgsz = 8, 640
            else:
                batch, imgsz = 4, 512
        else:
            batch, imgsz = 4, 512

        # 训练轮数随数据量反比：数据少多轮、数据多少轮；
        # CPU 环境自动调低轮数（无 GPU 时训练慢，200 轮耗时不可接受），有 GPU 才给足轮数
        if image_count > 0:
            if cuda_ok:
                if image_count < 300:
                    epochs = 200
                elif image_count < 1200:
                    epochs = 120
                elif image_count < 3000:
                    epochs = 80
                else:
                    epochs = 50
            else:
                if image_count < 300:
                    epochs = 30
                elif image_count < 1200:
                    epochs = 20
                else:
                    epochs = 10
        else:
            epochs = 100 if cuda_ok else 20

        # 高级参数默认值（YOLO 推荐起点）
        lr0 = 0.01
        optimizer = "auto"
        weight_decay = 0.0005
        patience = 50

        mode = "全新训练"
        if base_model_id:
            # 微调模式：更小的学习率 → 更稳；早停更早避免过拟合；轮数减半
            mode = "基于已有模型微调"
            lr0 = 0.001
            optimizer = "AdamW"
            patience = max(20, patience // 2)
            epochs = max(30, epochs // 2)

        params = {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "lr0": lr0,
            "optimizer": optimizer,
            "weight_decay": weight_decay,
            "patience": patience,
        }

        # ---------- 4. 说明文案 ----------
        cpu_hint = "；当前为 CPU 环境，已自动调低训练轮数以节省时间" if not cuda_ok else ""
        reason = (
            f"检测到 {device_info['name']}"
            f"（{'CUDA 显存 ' + str(device_info['vram_gb']) + 'GB' if cuda_ok else 'CPU'}），"
            f"数据集约 {image_count} 张图 / {class_count} 类，"
            f"已按{mode}场景推荐参数（可手动修改）{cpu_hint}。"
        )

        return {
            "device": device_info,
            "dataset": {"image_count": image_count, "class_count": class_count},
            "mode": mode,
            "params": params,
            "reason": reason,
        }

    def _load_yaml(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    async def infer_dataset_business(self, dataset_id: str, version: str = "v1") -> dict:
        """根据数据集类别名（data.yaml names）自动推断业务/算法类型。

        供前端选中数据集后自动分配业务场景使用；识别不出返回 general（通用）。
        """
        yaml_path = self.datasets_dir / dataset_id / version / "data.yaml"
        if not await asyncio.to_thread(lambda: yaml_path.exists()):
            return {"business": "general", "names": [], "error": "data.yaml 不存在"}
        try:
            data_cfg = await asyncio.to_thread(self._load_yaml, yaml_path)
        except Exception as e:
            return {"business": "general", "names": [], "error": str(e)}
        names = data_cfg.get("names", [])
        business = infer_business(names)
        return {
            "business": business,
            "names": _normalize_classes(names),
            "dataset_id": dataset_id,
            "version": version,
        }

    async def list_gpus(self) -> dict:
        """列出当前服务器的可用 GPU（训练节点），含显存总量/已用/剩余"""
        result = {"cuda_available": False, "gpus": []}
        try:
            import torch
            if not torch.cuda.is_available():
                return result
            result["cuda_available"] = True
            for i in range(torch.cuda.device_count()):
                free_b, total_b = torch.cuda.mem_get_info(i)
                used_b = max(0, total_b - free_b)
                g = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_gb": round(total_b / (1024 ** 3), 1),
                    "used_gb": round(used_b / (1024 ** 3), 1),
                    "free_gb": round(free_b / (1024 ** 3), 1),
                }
                result["gpus"].append(g)
        except Exception as e:
            print(f"[list_gpus] 检测 GPU 失败: {e}")
        return result

    async def prepare_base_weights(self, model_name: str) -> dict:
        """训练前确认/预下载预训练权重到本地 models/custom（镜像加速缓存），
        避免训练启动时 ultralytics 联网下载导致的长时间空窗。

        status:
          - ready:    本地已就绪（缓存命中或本次镜像下载完成）
          - fallback: 本地与镜像均不可用，将交由 ultralytics 内置下载兜底
        """
        try:
            path = await asyncio.to_thread(ensure_local_base_weights, model_name)
            return {"status": "ready" if os.path.exists(path) else "fallback", "path": path}
        except Exception as e:
            print(f"[prepare-weights] 权重准备失败: {e}")
            return {"status": "fallback", "path": model_name, "error": str(e)}
    
    @staticmethod
    def _parse_version(ver) -> float:
        """解析 'v1.0' 形式的版本号为数值，无法解析返回 None（None 不可比）"""
        if not ver or not str(ver).startswith("v"):
            return None
        try:
            return float(str(ver)[1:])
        except (TypeError, ValueError):
            return None

    def _pick_baseline(self, business: str):
        """模型守门员：自动选择同业务"上一代生产模型"作为对比基准。

        扫描模型仓库中 business 相同且 status=production_ready 的模型，
        取 version 最高者，返回 (weights_path, model_id)；无任何生产模型（首版）返回 (None, None)，
        此时由 gatekeeper 判定 first_version 直接晋升。
        """
        if not self.registry_dir.exists():
            return None, None
        best_score, best_path, best_id = None, None, None
        for meta_file in self.registry_dir.glob("*/model.json"):
            try:
                meta = _load_json(meta_file)
            except Exception:
                continue
            if meta.get("business") != business or meta.get("status") != "production_ready":
                continue
            score = self._parse_version(meta.get("version"))
            if score is None:
                continue
            w = resolve_registry_weights(meta, self.registry_dir)
            if not w or not w.exists():
                continue
            if best_score is None or score > best_score:
                best_score, best_path, best_id = score, str(w), meta.get("model_id")
        return best_path, best_id

    def _apply_label_filter(self, job_id: str, data_yaml: Path, base_model_id: str,
                            out_subdir: str = "filtered_data"):
        """阶段0.3：按基础模型的标签字典过滤数据集类别并生成训练副本（同步，线程内调用）。

        微调时仅训练模型标签字典（labels_dict.json）定义的类别：
          - 数据集里模型字典之外的类别 → 剔除对应标注；
          - 类别顺序与模型字典不一致 → 按模型字典顺序重映射 class id。
        图片经硬链接复用（零拷贝），labels 现场过滤重映射，不污染原始标注。

        返回 (实际使用的 data.yaml 路径, 过滤信息 dict)；无需过滤时直接返回原 yaml。
        """
        model_dir = self.registry_dir / base_model_id
        dict_file = model_dir / "labels_dict.json"
        if not dict_file.exists():
            return data_yaml, None

        try:
            dic = _load_json(dict_file)
        except Exception:
            return data_yaml, None

        model_labels = sorted(dic.get("labels") or [], key=lambda x: int(x.get("index", 0)))
        code_to_original = {}
        for l in model_labels:
            ec = (l.get("english_code") or "").strip()
            if ec:
                code_to_original[_norm_class_name(ec)] = ec
        model_codes = list(code_to_original.keys())
        if not model_codes:
            return data_yaml, None

        data_cfg = self._load_yaml(data_yaml)
        names = _names_list(data_cfg.get("names", []))
        if not names:
            return data_yaml, None

        # 计算 old_id -> new_id（new_id 按模型字典顺序；模型字典外的类别被剔除）
        kept_codes = []       # 保留类别（归一化 key，按模型字典顺序）
        dropped = []          # 数据集有、模型字典没有 → 剔除
        mapping = {}          # old_id -> new_id
        kept_index = {}
        for i, n in enumerate(names):
            key = _norm_class_name(n)
            if key in model_codes:
                if key not in kept_index:
                    kept_index[key] = len(kept_codes)
                    kept_codes.append(key)
                mapping[str(i)] = str(kept_index[key])
            else:
                dropped.append(n)

        identity = all(int(k) == int(v) for k, v in mapping.items())
        new_names = [code_to_original[k] for k in kept_codes]  # 新类别名（英文原文）
        info = {
            "filtered": bool(dropped) or not identity,
            "model_id": base_model_id,
            "model_code": dic.get("model_code"),
            "dataset_classes": names,
            "kept": new_names,
            "dropped": dropped,
            "mapping": mapping,
        }
        if not info["filtered"]:
            # 数据集类别与模型字典一致（或为其同序子集）→ 直接用原 data.yaml
            return data_yaml, info
        if not kept_codes:
            # 数据集与模型标签字典零交集 → 训练毫无意义，阻止创建任务
            raise LabelFilterError(
                f"数据集类别与微调模型标签字典无交集（数据集: {names}，模型 labels_dict: {new_names or '空'}），"
                f"无法按模型类别训练。请选择类别匹配的数据集或更换微调模型。"
            )

        # ------- 生成过滤副本：图片硬链接 + labels 重映射 -------
        version_dir = data_yaml.parent
        img_src = version_dir / "images"
        lbl_src = version_dir / "labels"
        filtered_dir = self.jobs_dir / job_id / out_subdir
        if filtered_dir.exists():
            _delete_directory(filtered_dir)
        filtered_dir.mkdir(parents=True, exist_ok=True)

        if img_src.exists():
            shutil.copytree(img_src, filtered_dir / "images", copy_function=_link_or_copy)

        if lbl_src.exists():
            filter_log = {
                "mapping": mapping,
                "dropped": dropped,
                "converted_txt": 0,
                "dropped_rows": 0,
            }
            lbl_dst = filtered_dir / "labels"
            for txt in lbl_src.rglob("*.txt"):
                rel = txt.relative_to(lbl_src)
                dst = lbl_dst / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                new_lines = []
                with open(txt, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.split()
                        if not parts:
                            continue
                        if len(parts) < 5:
                            continue  # 非法行直接丢弃
                        new_id = mapping.get(parts[0])
                        if new_id is None:
                            filter_log["dropped_rows"] += 1
                            continue  # 模型字典外类别 → 丢弃该标注
                        parts[0] = new_id
                        new_lines.append(" ".join(parts[:5]) + "\n")
                with open(dst, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                filter_log["converted_txt"] += 1
            _save_json(filtered_dir / "filter.json", filter_log)

        new_yaml = {
            "path": str(filtered_dir.absolute()),
            "train": data_cfg.get("train", "images/train"),
            "val": data_cfg.get("val", "images"),
            "nc": len(new_names),
            "names": new_names,
        }
        if data_cfg.get("test"):
            new_yaml["test"] = data_cfg["test"]
        with open(filtered_dir / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(new_yaml, f, allow_unicode=True, sort_keys=False)

        print(f"[create_job] 标签过滤: 保留 {len(new_names)} 类（{new_names}），"
              f"剔除 {len(dropped)} 类（{dropped}），过滤副本: {filtered_dir}")
        return filtered_dir / "data.yaml", info

    def _aggregate_job_data(self, job_id: str, specs: list, base_model_id: str):
        """阶段1.3：多数据集聚合（雪球合训）。

        specs: [(dataset_id, version, data_yaml_path), ...]
        统一类别命名空间（可选按模型标签字典过滤）→ 逐数据集硬链接图片 + 重映射 labels
        → 生成聚合 data.yaml（train/val 支持列表）。返回 (data.yaml 路径, 聚合信息 dict)。

        - 类别空间统一规则：
            base_model_id 存在 → 仅保留模型标签字典类别（顺序 = 模型字典顺序，字典外剔除）；
            base_model_id 为空 → 各数据集类别并集（先见顺序）。
        - 每数据集独立子目录 images/{i}、labels/{i}，避免不同数据集文件名/划分结构冲突。
        """
        # ---------- 1. 统一类别命名空间 ----------
        model_codes = []
        if base_model_id:
            dict_file = self.registry_dir / base_model_id / "labels_dict.json"
            if dict_file.exists():
                try:
                    _dic = _load_json(dict_file)
                    _labels = sorted(_dic.get("labels") or [], key=lambda x: int(x.get("index", 0)))
                    model_codes = [(l.get("english_code") or "").strip() for l in _labels]
                except Exception:
                    model_codes = []

        model_key_set = {_norm_class_name(c) for c in model_codes if c}
        unified_index = {}   # 归一化 key -> global class id
        unified_names = []   # 展示名（模型字典原文优先）
        for _ds_id, _ver, _yaml_p in specs:
            try:
                _cfg = self._load_yaml(_yaml_p)
            except Exception as e:
                raise ValueError(f"数据集 {_ds_id}/{_ver} 的 data.yaml 解析失败: {e}")
            for n in _names_list(_cfg.get("names", [])):
                key = _norm_class_name(n)
                if model_key_set and key not in model_key_set:
                    continue
                if key not in unified_index:
                    unified_index[key] = len(unified_names)
                    if model_key_set:
                        display = next((c for c in model_codes if _norm_class_name(c) == key), n)
                    else:
                        display = n
                    unified_names.append(display)
        if not unified_names:
            raise LabelFilterError(
                "聚合数据集类别与微调模型标签字典无交集（或全部为空），无法按模型类别训练。"
                "请选择类别匹配的数据集或更换微调模型。"
            )

        # ---------- 2. 逐数据集生成聚合副本 ----------
        agg_dir = self.jobs_dir / job_id / "aggregated"
        if agg_dir.exists():
            _delete_directory(agg_dir)
        agg_dir.mkdir(parents=True, exist_ok=True)
        img_agg = agg_dir / "images"
        lbl_agg = agg_dir / "labels"
        img_agg.mkdir()
        lbl_agg.mkdir()

        per_ds = []
        total_images = 0
        for i, (ds_id, ver, yaml_p) in enumerate(specs):
            cfg = self._load_yaml(yaml_p)
            src_names = _names_list(cfg.get("names", []))
            mapping = {}
            dropped = []
            for old, n in enumerate(src_names):
                key = _norm_class_name(n)
                if key in unified_index:
                    mapping[str(old)] = str(unified_index[key])
                else:
                    dropped.append(n)

            version_dir = yaml_p.parent
            img_src = version_dir / "images"
            lbl_src = version_dir / "labels"
            sub_img = img_agg / str(i)
            if img_src.exists():
                shutil.copytree(img_src, sub_img, copy_function=_link_or_copy)
                count_img = self._count_images(sub_img)
                total_images += count_img
            else:
                count_img = 0

            ds_log = {
                "dataset_id": ds_id,
                "version": ver,
                "images": count_img,
                "dropped_classes": dropped,
                "class_mapping": mapping,
            }
            if lbl_src.exists():
                sub_lbl = lbl_agg / str(i)
                converted = 0
                dropped_rows = 0
                for txt in lbl_src.rglob("*.txt"):
                    rel = txt.relative_to(lbl_src)
                    dst = sub_lbl / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    new_lines = []
                    with open(txt, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            parts = line.split()
                            if len(parts) < 5:
                                continue
                            new_id = mapping.get(parts[0])
                            if new_id is None:
                                dropped_rows += 1
                                continue
                            parts[0] = new_id
                            new_lines.append(" ".join(parts[:5]) + "\n")
                    with open(dst, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                    converted += 1
                ds_log["converted_txt"] = converted
                ds_log["dropped_rows"] = dropped_rows
            per_ds.append(ds_log)

        # ---------- 3. 生成聚合 data.yaml（train/val 为路径列表） ----------
        train_paths, val_paths = [], []
        for i in range(len(specs)):
            imgs_i = img_agg / str(i)
            if not imgs_i.exists():
                continue
            if (imgs_i / "train").is_dir():
                train_paths.append(f"images/{i}/train")
                if (imgs_i / "val").is_dir():
                    val_paths.append(f"images/{i}/val")
            else:
                train_paths.append(f"images/{i}")
        if not val_paths:
            val_paths = train_paths[:1]

        new_yaml = {
            "path": str(agg_dir.absolute()),
            "train": train_paths,
            "val": val_paths,
            "nc": len(unified_names),
            "names": unified_names,
        }
        with open(agg_dir / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(new_yaml, f, allow_unicode=True, sort_keys=False)

        agg_info = {
            "aggregated": True,
            "dataset_count": len(specs),
            "unified_names": unified_names,
            "datasets": per_ds,
            "total_images": total_images,
            "yaml": str(agg_dir / "data.yaml"),
        }
        _save_json(agg_dir / "aggregation.json", agg_info)
        print(f"[create_job] 聚合完成: {len(specs)} 个数据集 → {len(unified_names)} 类"
              f"（{unified_names}），共 {total_images} 张图: {agg_dir}")
        return agg_dir / "data.yaml", agg_info

    def _find_trainable_incomplete(self, exclude_ids: set, business: str) -> list:
        """阶段1.4：扫描未完成训练数据集（stage∈{sealed,failed} 且 training_status=incomplete，
        同业务）→ 自动带进下一轮聚合训练（雪球）。返回 [(dataset_id, version), ...]"""
        candidates = []
        if not self.datasets_dir.exists():
            return candidates
        for d in self.datasets_dir.iterdir():
            if not d.is_dir() or d.name in exclude_ids:
                continue
            meta_p = d / "meta.json"
            if not meta_p.exists():
                continue
            try:
                meta = _load_json(meta_p)
                _ensure_stage_fields(meta)
            except Exception:
                continue
            if meta.get("training_status") == TRAIN_STATUS_COMPLETED:
                continue
            if meta.get("stage") not in TRAINABLE_STAGES:
                continue
            ver = meta.get("version", "v1")
            yaml_p = d / ver / "data.yaml"
            if not yaml_p.exists():
                continue
            try:
                cfg = self._load_yaml(yaml_p)
                b = infer_business(cfg.get("names", []))
            except Exception:
                b = None
            if b != business:
                continue
            candidates.append((d.name, ver))
        return candidates

    # ------------------------------------------------------------------
    # 阶段1.5（回忆集混训）/ 阶段1.6（样本池抽样并入）
    # ------------------------------------------------------------------
    def _find_recall_dirs(self, business: str, exclude_ids: set, max_datasets: int = 3) -> list:
        """阶段1.5：扫描同业务已完成训练的数据集（stage=completed 且 training_status=completed），
        供增量训练混入少量旧数据（防灾难性遗忘）。返回 [{dir, yaml, names, dataset_id}, ...]"""
        recalls = []
        if not self.datasets_dir.exists():
            return recalls
        for d in self.datasets_dir.iterdir():
            if not d.is_dir() or d.name in exclude_ids:
                continue
            meta_p = d / "meta.json"
            if not meta_p.exists():
                continue
            try:
                meta = _load_json(meta_p)
                _ensure_stage_fields(meta)
            except Exception:
                continue
            if meta.get("stage") != STAGE_COMPLETED:
                continue
            if meta.get("training_status") != TRAIN_STATUS_COMPLETED:
                continue
            ver = meta.get("version", "v1")
            yaml_p = d / ver / "data.yaml"
            if not yaml_p.exists():
                continue
            try:
                cfg = self._load_yaml(yaml_p)
                names = _names_list(cfg.get("names", []))
                if infer_business(names) != business:
                    continue
            except Exception:
                continue
            recalls.append({"dataset_id": d.name, "version": ver, "yaml": yaml_p, "names": names})
            if len(recalls) >= max_datasets:
                break
        return recalls

    @staticmethod
    def _pool_image_files(root: Path) -> list:
        """样本池图片文件列表（jpg/jpeg/png/bmp/webp）"""
        if not root or not root.exists():
            return []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    def _merge_training_extras(self, job_id: str, data_yaml: str, base_model_id: str,
                               business: str, exclude_ids: set, recall_enabled: bool,
                               hard_ratio: float, bg_ratio: float):
        """阶段1.5/1.6：训练前把增强样本混入训练集（仅混入 train，不碰 val/test）。

        来源（互不影响，全部可选）：
        1) 回忆集（1.5）：增量训练（有 base_model_id）时，从同业务已完成训练的数据集
           随机抽取少量图片+标注（类别按名字重映射）；
        2) 困难样本库（1.6）：base_model_id 对应的 hard/{model_id}，按 hard_ratio 抽样，
           标注类别按名字重映射到本次训练的类别空间；
        3) 空白样本库（1.6）：全局 background 按 bg_ratio 抽样（配空标注 = 负样本）。

        增强样本统一放入 jobs/{job_id}/sample_pool_train/{images,labels}，并生成一份
        独立 data.yaml（train/val 转绝对路径列表）—— 不污染原数据集，resume 可复用。
        返回 (data.yaml 路径, 混合信息 dict | None)。
        """
        yaml_p = Path(data_yaml)
        if not yaml_p.exists():
            return yaml_p, None
        try:
            cfg = self._load_yaml(yaml_p)
        except Exception:
            return yaml_p, None
        names = _names_list(cfg.get("names", []))
        if not names:
            return yaml_p, None
        target_index = {_norm_class_name(n): i for i, n in enumerate(names)}

        rng = random.Random(abs(hash(job_id)))
        sp_dir = self.jobs_dir / job_id / "sample_pool_train"
        t_img = sp_dir / "images"
        t_lbl = sp_dir / "labels"
        t_img.mkdir(parents=True, exist_ok=True)
        t_lbl.mkdir(parents=True, exist_ok=True)

        merged = {"hard": 0, "background": 0, "recall": 0}
        used_stems = set()

        def _emit(kind: str, img_src: Path, label_lines: list):
            """复制图片 + 写标注到增强目录（前缀防重名）"""
            name = f"{kind}_{img_src.name}"
            stem = Path(name).stem
            if stem in used_stems:
                return
            used_stems.add(stem)
            shutil.copy2(img_src, t_img / name)
            if label_lines:
                (t_lbl / f"{stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            else:
                (t_lbl / f"{stem}.txt").touch()
            return stem

        # ---------- 1. 回忆集混训（1.5，仅增量训练） ----------
        if base_model_id and recall_enabled:
            try:
                recalls = self._find_recall_dirs(business, exclude_ids)
            except Exception as e:
                print(f"[merge_training_extras] 扫描回忆集失败（跳过）: {e}")
                recalls = []
            for rc in recalls:
                r_cfg = self._load_yaml(rc["yaml"])
                rc_names = _names_list(r_cfg.get("names", []))
                rc_imgs = self._pool_image_files(rc["yaml"].parent / "images")
                rng.shuffle(rc_imgs)
                # 每个旧数据集最多抽 20 张，总量 ≤ 100，防止喧宾夺主
                for im in rc_imgs[: min(20, len(rc_imgs))]:
                    if merged["recall"] >= 100:
                        break
                    txt = rc["yaml"].parent / "labels" / f"{im.stem}.txt"
                    lines = []
                    if txt.exists():
                        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                            parts = line.split()
                            if len(parts) < 5:
                                continue
                            try:
                                old_id = int(parts[0])
                            except ValueError:
                                continue
                            src_name = rc_names[old_id] if 0 <= old_id < len(rc_names) else None
                            new_id = target_index.get(_norm_class_name(src_name)) if src_name else None
                            if new_id is None:
                                continue
                            parts[0] = str(new_id)
                            lines.append(" ".join(parts[:5]))
                    if _emit("recall", im, lines):
                        merged["recall"] += 1

        # ---------- 2. 困难样本库（1.6，按模型） ----------
        if base_model_id and hard_ratio and hard_ratio > 0:
            hard_dir = settings.HARD_POOL_DIR / base_model_id
            if hard_dir.exists():
                try:
                    pool_meta = _load_json(hard_dir / "meta.json")
                except Exception:
                    pool_meta = {}
                pool_names = pool_meta.get("names") or []
                pool_index = {_norm_class_name(n): i for i, n in enumerate(pool_names)}
                pool_imgs = self._pool_image_files(hard_dir / "images")
                rng.shuffle(pool_imgs)
                limit = max(1, int(len(pool_imgs) * float(hard_ratio)))
                for im in pool_imgs[:limit]:
                    txt = hard_dir / "labels" / f"{im.stem}.txt"
                    lines = []
                    if txt.exists():
                        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                            parts = line.split()
                            if len(parts) < 5:
                                continue
                            try:
                                old_id = int(parts[0])
                            except ValueError:
                                continue
                            src_name = pool_names[old_id] if 0 <= old_id < len(pool_names) else None
                            new_id = target_index.get(_norm_class_name(src_name)) if src_name else None
                            if new_id is None:
                                continue
                            parts[0] = str(new_id)
                            lines.append(" ".join(parts[:5]))
                    if _emit("hard", im, lines):
                        merged["hard"] += 1

        # ---------- 3. 空白负样本库（1.6，全局共享） ----------
        if bg_ratio and bg_ratio > 0:
            bg_imgs = self._pool_image_files(settings.BACKGROUND_POOL_DIR / "images")
            rng.shuffle(bg_imgs)
            limit = max(1, int(len(bg_imgs) * float(bg_ratio)))
            for im in bg_imgs[:limit]:
                if _emit("bg", im, []):   # 空标注 = 背景负样本
                    merged["background"] += 1

        if not any(merged.values()):
            shutil.rmtree(sp_dir, ignore_errors=True)
            return yaml_p, None

        # ---------- 4. 生成独立 data.yaml（train/val 绝对路径列表） ----------
        def _resolve_entries(entries) -> list:
            root = Path(cfg.get("path") or yaml_p.parent)
            if isinstance(entries, str):
                entries = [entries]
            resolved = []
            for e in entries or []:
                e = str(e)
                p = Path(e)
                resolved.append(str(p if p.is_absolute() else root / e))
            return resolved

        train_abs = _resolve_entries(cfg.get("train")) or ["images/train"]
        val_abs = _resolve_entries(cfg.get("val")) or train_abs[:1]
        new_cfg = {
            "path": str(sp_dir.absolute()),
            "train": train_abs + ["images"],
            "val": val_abs,
            "nc": len(names),
            "names": names,
        }
        if cfg.get("test"):
            new_cfg["test"] = _resolve_entries(cfg.get("test"))
        with open(sp_dir / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(new_cfg, f, allow_unicode=True, sort_keys=False)

        info = merged | {
            "enabled": True,
            "dir": str(sp_dir),
            "yaml": str(sp_dir / "data.yaml"),
        }
        _save_json(sp_dir / "extra.json", info)
        print(f"[merge_training_extras] 训练增强混入: 回忆集 {merged['recall']} 张，"
              f"困难样本 {merged['hard']} 张，空白负样本 {merged['background']} 张 → {sp_dir}")
        return sp_dir / "data.yaml", info

    async def create_job(self, dataset_id: str, version: str, model_name: str, 
                         epochs: int, imgsz: int, batch: int, base_model_id: str = None,
                         lr0: float = None, optimizer: str = None,
                         weight_decay: float = None, patience: int = None,
                         gpu_index: int = None, business: str = None,
                         dataset_ids: list = None, aggregate_incomplete: bool = False,
                         hard_sample_ratio: float = 0.1, background_sample_ratio: float = 0.05,
                         recall_enabled: bool = True):
        """创建训练任务（支持高级训练参数与训练节点选择、业务/算法类型隔离）

        阶段1.3（训练任务聚合）：支持多数据集聚合训练。
          - dataset_ids: 可选，待聚合训练的数据集列表（缺省=仅 dataset_id 单个）；
            聚合时统一类别命名空间（微调按模型标签字典过滤），训练副本落 jobs/{job_id}/aggregated。
          - aggregate_incomplete: 阶段1.4（雪球）— 自动扫描同业务未完成训练数据集
            （stage∈{sealed,failed} 且 training_status=incomplete）一并聚合进本轮训练。
          - 训练开始前，所有参与数据集置 stage=training，结果由 train_script 回写
            （合格→completed/已完成训练；不合格→failed/保持未完成，参与下一轮雪球）。
        """
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # ---------- 解析待训练数据集列表（1.3 聚合） ----------
        specs = []  # [(dataset_id, version, data_yaml_path), ...]
        seen_ids = set()

        def _push_spec(ds_id: str, ds_version: str = None):
            if ds_id in seen_ids:
                return
            use_ver = ds_version or version
            d_dir = self.datasets_dir / ds_id / use_ver
            d_yaml = d_dir / "data.yaml"
            if not d_yaml.exists():
                raise ValueError(f"Dataset {ds_id}/{use_ver} not prepared")
            seen_ids.add(ds_id)
            specs.append((ds_id, use_ver, d_yaml))

        _push_spec(dataset_id)
        if dataset_ids:
            for ds_id in dataset_ids:
                _push_spec(ds_id)

        # 阶段1.4（雪球）：自动带入同业务未完成训练数据集
        if aggregate_incomplete and len(specs) == 1:
            primary_yaml = specs[0][2]
            try:
                primary_cfg = await asyncio.to_thread(self._load_yaml, primary_yaml)
                primary_business = business or infer_business(primary_cfg.get("names", []))
            except Exception:
                primary_business = None
            if primary_business:
                try:
                    extra = await asyncio.to_thread(
                        self._find_trainable_incomplete, seen_ids, primary_business
                    )
                except Exception as e:
                    print(f"[create_job] 扫描未完成数据集失败（跳过自动聚合）: {e}")
                    extra = []
                for extra_ds_id, extra_ver in extra:
                    _push_spec(extra_ds_id, extra_ver)
                if extra:
                    print(f"[create_job] 1.4 雪球: 自动带入 {len(extra)} 个未完成数据集: {extra}")

        primary_id = specs[0][0]
        primary_yaml = specs[0][2]

        # 业务/算法类型：未指定时根据主数据集类别名（data.yaml names）自动推断
        if not business:
            try:
                data_cfg = await asyncio.to_thread(self._load_yaml, primary_yaml)
                business = infer_business(data_cfg.get("names", []))
                print(f"[create_job] 自动识别业务/算法类型: {business}（主数据集 {primary_id}/{version}）")
            except Exception:
                business = "general"

        # 处理模型路径：如果提供了 base_model_id，使用已有模型的权重
        actual_model_path = model_name
        if base_model_id:
            base_model_dir = self.registry_dir / base_model_id
            base_model_file = base_model_dir / "model.json"
            if not await asyncio.to_thread(lambda: base_model_file.exists()):
                raise ValueError(f"Base model {base_model_id} not found")
            
            base_model_meta = await asyncio.to_thread(_load_json, base_model_file)
            
            weights_path = resolve_registry_weights(base_model_meta, self.registry_dir)
            if not weights_path or not await asyncio.to_thread(lambda: weights_path.exists()):
                raise ValueError(
                    f"Base model weights not found: {base_model_meta.get('weights_path')}"
                )
            
            actual_model_path = str(weights_path)
            model_name = f"{base_model_id}_fine_tuned"
        
        # 并发训练数限制：防止 CPU/GPU 资源被过多训练任务占满
        max_concurrent = settings.MAX_CONCURRENT_TRAIN_JOBS
        active = await asyncio.to_thread(self._active_training_count)
        if active >= max_concurrent:
            raise ValueError(
                f"训练任务已达上限（{active}/{max_concurrent}），请等待某个训练完成后再试"
            )
        
        # 高级参数未提供时，按训练环境/模式给默认值（ultralytics 官方推荐）
        # 微调模式使用更保守的学习率/优化器，避免破坏已有权重
        if base_model_id:
            lr0 = lr0 if lr0 is not None else 0.001
            optimizer = optimizer or "AdamW"
            weight_decay = weight_decay if weight_decay is not None else 0.0005
            patience = patience if patience is not None else 25
        else:
            lr0 = lr0 if lr0 is not None else 0.01
            optimizer = optimizer or "auto"
            weight_decay = weight_decay if weight_decay is not None else 0.0005
            patience = patience if patience is not None else 50

        # 批次大小归一化：-1/0(自动) 会触发 ultralytics 的 auto-batch 校准 + 多进程数据加载，
        # 在 Docker 容器环境里极易报 [Errno 22] Invalid argument 直接失败。
        # 这里按当前 GPU 显存给出确定的安全值，训练稳定优先。
        if batch is None or batch < 1:
            batch = self._default_batch()

        # 模型守门员：自动选择同业务上一代生产模型作为对比基准
        # 训练结束后 train_script 内评测新模型 vs 该基准；无生产模型（首版）由 gatekeeper 直晋
        baseline_path, baseline_model_id = self._pick_baseline(business)

        # ---------- 训练数据准备（单数据集 / 多数据集聚合） ----------
        actual_data_yaml = primary_yaml
        label_filter = None
        aggregation = None

        if len(specs) == 1:
            # 单数据集：阶段0.3 微调时按基础模型标签字典过滤
            if base_model_id:
                try:
                    actual_data_yaml, label_filter = await asyncio.to_thread(
                        self._apply_label_filter, job_id, primary_yaml, base_model_id
                    )
                except LabelFilterError as e:
                    raise ValueError(str(e)) from e
                except Exception as e:
                    print(f"[create_job] 标签过滤执行异常，按原数据集训练: {e}")
                    actual_data_yaml = primary_yaml
        else:
            # 多数据集：阶段1.3 统一类别空间聚合（微调时仅保留模型字典类别）
            try:
                actual_data_yaml, aggregation = await asyncio.to_thread(
                    self._aggregate_job_data, job_id, specs, base_model_id
                )
            except LabelFilterError as e:
                raise ValueError(str(e)) from e
            except Exception as e:
                print(f"[create_job] 数据聚合执行异常，按主数据集训练: {e}")
                actual_data_yaml = primary_yaml

        # 阶段1.5/1.6：训练前混入增强样本（回忆集防遗忘 + 困难样本库 + 空白负样本库）
        # 只混入 train 划分，不改 val/test，保证守门员对比口径与前代一致
        sample_pool_info = None
        try:
            actual_data_yaml, sample_pool_info = await asyncio.to_thread(
                self._merge_training_extras, job_id, actual_data_yaml, base_model_id,
                business, seen_ids, recall_enabled,
                hard_sample_ratio, background_sample_ratio
            )
        except Exception as e:
            print(f"[create_job] 样本池/回忆集混入执行异常（忽略，按原数据训练）: {e}")

        # 创建job元数据
        job_meta = {
            "job_id": job_id,
            "dataset_id": primary_id,             # 兼容旧字段：主数据集
            "dataset_ids": [s[0] for s in specs],  # 1.3 参与训练的完整数据集列表
            "datasets": [
                {"dataset_id": s[0], "version": s[1]} for s in specs
            ],
            "aggregated": len(specs) > 1,          # 1.3 是否聚合训练
            "version": version,
            "data_yaml": str(actual_data_yaml),  # 实际用于训练的 data.yaml（可能是过滤/聚合副本）
            "label_filter": label_filter,          # 阶段0.3 标签过滤信息（无需过滤为 None）
            "aggregation": aggregation,            # 阶段1.3 聚合信息（非聚合为 None）
            "sample_pool": sample_pool_info,       # 阶段1.5/1.6 采样并入信息（未混入为 None）
            "model_name": model_name,
            "base_model_id": base_model_id,
            "original_model_path": actual_model_path,  # 保存原始模型路径，用于恢复
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "lr0": lr0,
            "optimizer": optimizer,
            "weight_decay": weight_decay,
            "patience": patience,
            "gpu_index": gpu_index,
            "business": business,
            "baseline_model_id": baseline_model_id,   # 守门员对比基准（上一代同业务生产模型）
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "log_file": str(self.jobs_dir / f"{job_id}.log")
        }
        
        job_file = self.jobs_dir / f"{job_id}.json"
        await asyncio.to_thread(_save_json, job_file, job_meta)

        # 启动训练进程（在线程中执行）。先启动成功再标记数据集状态，
        # 启动失败（如 Popen 异常）则清理任务文件并回滚，避免留下"训练中"残留
        try:
            await asyncio.to_thread(
                self._start_training, job_id, actual_data_yaml, actual_model_path, epochs, imgsz, batch, False,
                lr0, optimizer, weight_decay, patience, gpu_index,
                business, baseline_path, baseline_model_id
            )
        except Exception as e:
            try:
                _delete_file(job_file)
            except Exception:
                pass
            try:
                _delete_directory(self.jobs_dir / job_id)
            except Exception:
                pass
            raise RuntimeError(f"训练进程启动失败，任务已回滚: {e}") from e

        # 1.3：所有参与数据集置「训练中」（训练结果由 train_script 回写）
        await asyncio.to_thread(self._mark_datasets_training, specs)
        
        resp = {"job_id": job_id, "status": "running",
                "dataset_ids": [s[0] for s in specs], "aggregated": len(specs) > 1}
        if aggregation:
            resp["aggregation"] = {
                "dataset_count": aggregation.get("dataset_count"),
                "unified_names": aggregation.get("unified_names", []),
                "total_images": aggregation.get("total_images"),
            }
        elif label_filter and label_filter.get("filtered"):
            resp["label_filter"] = {
                "model_code": label_filter.get("model_code"),
                "kept": label_filter.get("kept", []),
                "dropped": label_filter.get("dropped", []),
            }
        return resp

    def _mark_datasets_training(self, specs: list):
        """1.3：训练开始前，将参与数据集置 stage=training（结果由 train_script 回写）"""
        for ds_id, version, _yaml_p in specs:
            meta_path = self.datasets_dir / ds_id / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = _load_json(meta_path)
                _ensure_stage_fields(meta)
                meta["stage"] = STAGE_TRAINING
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[create_job] 数据集 {ds_id} 置训练中失败（忽略）: {e}")
    
    def _active_training_count(self) -> int:
        """当前活跃训练进程数（先清理已结束的进程，避免残留计数）"""
        dead = [jid for jid, p in self.running_processes.items() if p.poll() is not None]
        for jid in dead:
            del self.running_processes[jid]
        return len(self.running_processes)
    
    def _start_training(self, job_id: str, data_yaml: Path, model_name: str,
                       epochs: int, imgsz: int, batch: int, resume: bool = False,
                       lr0: float = None, optimizer: str = None,
                       weight_decay: float = None, patience: int = None,
                       gpu_index: int = None, business: str = "general",
                       baseline_path: str = None, baseline_model_id: str = None):
        """启动训练进程（同步方法，在线程中调用）"""
        log_file = self.jobs_dir / f"{job_id}.log"
        job_file = self.jobs_dir / f"{job_id}.json"
        
        # 准备训练脚本
        train_script = settings.BASE_DIR / "src" / "yolo" / "train_script.py"
        
        # 输出目录
        project_dir = self.jobs_dir / job_id
        
        # 启动子进程
        cmd = [
            "python", str(train_script),
            "--data", str(data_yaml),
            "--model", model_name,
            "--epochs", str(epochs),
            "--imgsz", str(imgsz),
            "--batch", str(batch),
            "--project", str(project_dir),
            "--name", "train",
            "--job_id", job_id,
            "--job_file", str(job_file),
            "--registry_dir", str(self.registry_dir),
            "--datasets_dir", str(self.datasets_dir),
            "--business", business
        ]
        
        # 模型守门员对比基准（上一代同业务生产模型），存在则传给训练脚本在训练结束后执行对比
        if baseline_path:
            cmd += ["--baseline", str(baseline_path)]
        if baseline_model_id:
            cmd += ["--baseline-model-id", str(baseline_model_id)]
        
        if resume:
            cmd.append("--resume")
        
        # 高级训练参数（仅非 resume 时透传；resume 使用 checkpoint 保存的参数）
        if not resume:
            if lr0 is not None:
                cmd += ["--lr0", str(lr0)]
            if optimizer:
                cmd += ["--optimizer", str(optimizer)]
            if weight_decay is not None:
                cmd += ["--weight-decay", str(weight_decay)]
            if patience is not None:
                cmd += ["--patience", str(patience)]
            # 训练节点：指定 GPU（多卡环境）
            if gpu_index is not None:
                cmd += ["--gpu", str(gpu_index)]
        
        # 如果是恢复训练，追加日志而不是覆盖
        mode = "a" if resume else "w"
        
        # 使用 UTF-8 编码打开日志文件，确保编码正确
        with open(log_file, mode, encoding="utf-8", errors="replace") as f:
            # 设置环境变量确保 Python 进程输出 UTF-8 编码
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=settings.BASE_DIR,
                env=env
            )
        
        self.running_processes[job_id] = process

        # 持久化 PID，便于后端重启后重新接管仍在运行的训练进程
        try:
            job_file_path = self.jobs_dir / f"{job_id}.json"
            if job_file_path.exists():
                meta = _load_json(job_file_path)
                meta["pid"] = process.pid
                _save_json(job_file_path, meta)
        except Exception:
            pass
    
    async def list_jobs(self):
        """列出所有训练任务（自动检测崩溃的任务）"""
        def _list_jobs_sync():
            # 后端重启后，先重新接管仍在运行的训练进程，避免误判为崩溃
            self._recover_running_processes()

            jobs = []
            
            if not self.jobs_dir.exists():
                return jobs
            
            for job_file in self.jobs_dir.glob("*.json"):
                try:
                    job_meta = _load_json(job_file)
                except Exception:
                    continue
                
                # 检查是否有运行中的任务实际已崩溃
                job_id = job_meta.get("job_id")
                if job_id and job_meta.get("status") == "running":
                    # 检查进程是否真的在运行
                    if job_id in self.running_processes:
                        process = self.running_processes[job_id]
                        if process.poll() is not None:
                            # 进程已死，但状态还是 running，标记为崩溃
                            job_meta["status"] = "crashed"
                            job_meta["crashed_at"] = datetime.now().isoformat()
                            # 清理进程记录
                            del self.running_processes[job_id]
                            # 保存更新后的状态
                            _save_json(job_file, job_meta)
                    else:
                        # 不在运行进程列表中，但状态是 running，可能是崩溃后重启
                        # 检查是否有 checkpoint，如果有则标记为可恢复
                        train_dir = self.jobs_dir / job_id / "train"
                        checkpoint = train_dir / "weights" / "last.pt"
                        if checkpoint.exists():
                            job_meta["status"] = "crashed"
                            job_meta["crashed_at"] = datetime.now().isoformat()
                            _save_json(job_file, job_meta)
                
                # 检查是否有 checkpoint 但状态不是可恢复状态
                train_dir = self.jobs_dir / job_id / "train"
                checkpoint = train_dir / "weights" / "last.pt"
                best_pt = train_dir / "weights" / "best.pt"
                # 如果有 last.pt 或 best.pt，标记为可恢复
                if (checkpoint.exists() or best_pt.exists()) and job_meta.get("status") not in ["completed", "running"]:
                    # 标记为可恢复
                    if "can_resume" not in job_meta:
                        job_meta["can_resume"] = True
                        # 保存更新
                        _save_json(job_file, job_meta)
                
                jobs.append(job_meta)
            
            return jobs
        
        jobs = await asyncio.to_thread(_list_jobs_sync)
        return {"jobs": sorted(jobs, key=lambda x: x.get("created_at", ""), reverse=True)}

    async def get_job_tree(self, job_id: str):
        """获取训练任务输出目录树（权重/图表/日志等文件）"""
        job_file = self.jobs_dir / f"{job_id}.json"
        if not await asyncio.to_thread(lambda: job_file.exists()):
            return None
        job_dir = self.jobs_dir / job_id
        tree = await asyncio.to_thread(
            build_tree, job_dir, settings.DATA_DIR, 6, 300, True
        )
        log_path = self.jobs_dir / f"{job_id}.log"
        log_rel = None
        if await asyncio.to_thread(lambda: log_path.exists()):
            try:
                log_rel = log_path.relative_to(settings.DATA_DIR).as_posix()
            except ValueError:
                pass
        return {"tree": tree, "job_id": job_id, "log_file": log_rel}

    def _recover_running_processes(self):
        """后端重启后，重新接管仍在运行但句柄已丢失的训练进程"""
        if not self.jobs_dir.exists():
            return
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                meta = _load_json(job_file)
            except Exception:
                continue
            job_id = meta.get("job_id")
            if not job_id or meta.get("status") != "running":
                continue
            if job_id in self.running_processes:
                continue
            pid = meta.get("pid")
            if not pid:
                continue
            try:
                proc = psutil.Process(pid)
                # 进程仍在运行 → 重新接管
                if proc.poll() is None:
                    self.running_processes[job_id] = _RecoveredProcess(pid)
            except psutil.NoSuchProcess:
                pass

    async def get_job(self, job_id: str):
        """获取训练任务详情"""
        job_file = self.jobs_dir / f"{job_id}.json"
        
        if not await asyncio.to_thread(lambda: job_file.exists()):
            return None
        
        return await asyncio.to_thread(_load_json, job_file)
    
    async def stop_job(self, job_id: str):
        """停止训练任务"""
        if job_id in self.running_processes:
            process = self.running_processes[job_id]
            try:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=5)
            except:
                process.kill()
            
            del self.running_processes[job_id]
            
            # 更新状态
            job_file = self.jobs_dir / f"{job_id}.json"
            
            def _update_status():
                if job_file.exists():
                    job_meta = _load_json(job_file)
                    job_meta["status"] = "stopped"
                    job_meta["stopped_at"] = datetime.now().isoformat()
                    _save_json(job_file, job_meta)
            
            await asyncio.to_thread(_update_status)
            
            return {"ok": True}
        
        return {"ok": False, "error": "Job not running"}
    
    async def delete_job(self, job_id: str):
        """删除训练任务"""
        job_file = self.jobs_dir / f"{job_id}.json"
        
        if not await asyncio.to_thread(lambda: job_file.exists()):
            return None
        
        # 先停止任务（如果正在运行）
        if job_id in self.running_processes:
            try:
                await self.stop_job(job_id)
            except Exception as e:
                # 停止失败不影响删除，记录错误但继续
                print(f"Warning: Failed to stop job {job_id}: {e}")
        
        def _delete_job_files():
            errors = []
            
            # 删除任务文件
            try:
                _delete_file(job_file)
            except Exception as e:
                errors.append(f"Failed to delete job file: {e}")
            
            # 删除日志文件
            log_file = self.jobs_dir / f"{job_id}.log"
            try:
                _delete_file(log_file)
            except Exception as e:
                errors.append(f"Failed to delete log file: {e}")
            
            # 删除训练输出目录
            train_dir = self.jobs_dir / job_id
            try:
                _delete_directory(train_dir)
            except Exception as e:
                errors.append(f"Failed to delete train directory: {e}")
            
            # 如果有错误，但不影响主要删除操作（文件可能已经被删除或不存在）
            if errors:
                print(f"Warning: Some errors occurred while deleting job {job_id}: {errors}")
            
            return errors
        
        # 执行删除操作（忽略部分文件删除失败的错误）
        await asyncio.to_thread(_delete_job_files)
        
        return {"ok": True, "message": f"Job {job_id} deleted"}
    
    async def resume_job(self, job_id: str):
        """继续训练中断的任务（支持正常停止和崩溃恢复）"""
        job_file = self.jobs_dir / f"{job_id}.json"
        
        if not await asyncio.to_thread(lambda: job_file.exists()):
            raise ValueError(f"Job {job_id} not found")
        
        job_meta = await asyncio.to_thread(_load_json, job_file)
        
        # 检查是否有训练输出目录和 checkpoint
        train_dir = self.jobs_dir / job_id / "train"
        checkpoint = train_dir / "weights" / "last.pt"
        best_pt = train_dir / "weights" / "best.pt"
        
        # 检查 checkpoint 是否存在
        def _check_checkpoints():
            return checkpoint.exists(), best_pt.exists()
        
        checkpoint_exists, best_pt_exists = await asyncio.to_thread(_check_checkpoints)
        
        model_to_use = None
        use_resume = False
        
        if checkpoint_exists:
            # 有 last.pt，可以正常恢复
            model_to_use = str(checkpoint)
            use_resume = True
        elif best_pt_exists:
            # 有 best.pt，可以使用它继续训练
            model_to_use = str(best_pt)
            use_resume = False
        else:
            # 没有 checkpoint，尝试从原始模型重新开始
            original_model_path = job_meta.get("original_model_path")
            base_model_id = job_meta.get("base_model_id")
            
            if base_model_id:
                # 如果是微调任务，从基础模型重新开始
                base_model_dir = self.registry_dir / base_model_id
                base_model_file = base_model_dir / "model.json"
                if await asyncio.to_thread(lambda: base_model_file.exists()):
                    base_model_meta = await asyncio.to_thread(_load_json, base_model_file)
                    weights_path = Path(base_model_meta["weights_path"])
                    if await asyncio.to_thread(lambda: weights_path.exists()):
                        model_to_use = str(weights_path)
                        use_resume = False
                        print(f"Warning: No checkpoint found, restarting from base model {base_model_id}")
                    else:
                        raise ValueError(
                            f"No checkpoint found and base model weights not found. "
                            f"Cannot resume training. Please start a new training job."
                        )
                else:
                    raise ValueError(
                        f"No checkpoint found and base model {base_model_id} not found. "
                        f"Cannot resume training. Please start a new training job."
                    )
            elif original_model_path:
                # 从原始模型路径重新开始
                original_exists = await asyncio.to_thread(lambda: Path(original_model_path).exists())
                if original_exists or original_model_path.endswith('.pt'):
                    # 如果是预训练模型名称（如 yolov8n.pt），YOLO 会自动下载
                    model_to_use = original_model_path
                    use_resume = False
                    print(f"Warning: No checkpoint found, restarting from original model {original_model_path}")
                else:
                    raise ValueError(
                        f"No checkpoint found and original model path invalid: {original_model_path}. "
                        f"Cannot resume training. Please start a new training job."
                    )
            else:
                # 尝试使用 model_name（可能是预训练模型）
                model_name = job_meta.get("model_name", "yolov8n.pt")
                if model_name.endswith('.pt'):
                    model_to_use = model_name
                    use_resume = False
                    print(f"Warning: No checkpoint found, restarting from model {model_name}")
                else:
                    raise ValueError(
                        f"No checkpoint found for job {job_id}. "
                        f"The training was stopped before completing any epoch. "
                        f"Cannot resume training. Please start a new training job."
                    )
        
        # 检查数据集（恢复优先使用训练时保存的 data.yaml，可能是标签过滤副本）
        saved_yaml = job_meta.get("data_yaml")
        data_yaml = Path(saved_yaml) if saved_yaml else (self.datasets_dir / job_meta["dataset_id"] / job_meta["version"] / "data.yaml")

        if not await asyncio.to_thread(lambda: data_yaml.exists()):
            raise ValueError(f"Dataset {job_meta['dataset_id']}/{job_meta['version']} not found")
        
        # 检查任务是否正在运行（防止重复启动）
        if job_id in self.running_processes:
            process = self.running_processes[job_id]
            # 检查进程是否真的在运行
            if process.poll() is None:
                raise ValueError(f"Job {job_id} is already running")
            else:
                # 进程已结束但未清理，清理它
                del self.running_processes[job_id]
        
        # 记录原始状态（用于判断是否是崩溃恢复）
        original_status = job_meta.get("status")
        was_crashed = original_status == "running" and job_meta.get("resume_count", 0) == 0
        
        # 更新任务状态
        job_meta["status"] = "running"
        job_meta["resumed_at"] = datetime.now().isoformat()
        job_meta["resume_count"] = job_meta.get("resume_count", 0) + 1
        
        # 如果之前是崩溃（状态为 running 但进程已死），记录崩溃恢复
        if was_crashed:
            job_meta["crashed"] = True
            if "crashed_at" not in job_meta:
                job_meta["crashed_at"] = datetime.now().isoformat()
        
        await asyncio.to_thread(_save_json, job_file, job_meta)
        
        # 模型守门员：恢复训练同样自动选择当前同业务生产模型作为对比基准
        resume_business = job_meta.get("business", "general")
        resume_baseline_path, resume_baseline_id = self._pick_baseline(resume_business)

        # 启动训练进程
        await asyncio.to_thread(
            self._start_training,
            job_id, 
            data_yaml, 
            model_to_use,
            job_meta["epochs"],
            job_meta["imgsz"],
            job_meta["batch"],
            use_resume,
            job_meta.get("lr0"),
            job_meta.get("optimizer"),
            job_meta.get("weight_decay"),
            job_meta.get("patience"),
            job_meta.get("gpu_index"),
            resume_business,
            resume_baseline_path,
            resume_baseline_id
        )
        
        return {"job_id": job_id, "status": "running", "message": "Training resumed from checkpoint"}

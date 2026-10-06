import json
import shutil
import csv
import os
import re
import asyncio
import tempfile
import zipfile
import io
import logging
from pathlib import Path
from datetime import datetime
from fastapi import UploadFile
from src.core.settings import settings
from src.db import shadow
from src.utils import zip_guard
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _load_json(path: Path):
    """同步加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict):
    """同步保存 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 阶段 0：模型统一定义（唯一 code）+ 统一标签字典（四字段）
# ---------------------------------------------------------------------------

# 模型 code 规范：小写字母开头，仅含小写字母/数字/下划线，2~32 位
MODEL_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")


def _normalize_model_code(raw: str) -> str:
    """将用户输入规范化为 model_code（小写 + 下划线，剔除非法字符）

    中文名无法直接作为 code，但可从中提取拼音/英文部分；无英文时回退为空串（由调用方报错）。
    """
    s = re.sub(r"[^0-9a-zA-Z]+", "_", (raw or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _delete_directory(path: Path):
    """同步删除目录"""
    shutil.rmtree(path)


class ModelService:
    def __init__(self):
        self.registry_dir = settings.REGISTRY_DIR
        self.jobs_dir = settings.JOBS_DIR
    
    async def list_models(self):
        """列出所有模型（含 1.7 模型仓库：关联数据集数量与状态统计）"""
        def _list_models_sync():
            models = []

            # 先读全部模型记录：数据集归属换算要用（库内版本记录 → 它所属的库根）
            metas = {}
            if self.registry_dir.exists():
                for model_dir in self.registry_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    mf = model_dir / "model.json"
                    if not mf.exists():
                        continue
                    try:
                        metas[model_dir.name] = _load_json(mf)
                    except Exception:
                        continue

            # 1.7：先扫描数据集，按「归属算法库根记录」汇总数量与阶段分布
            ds_stats = {}
            datasets_dir = settings.DATASETS_DIR
            if datasets_dir.exists():
                for ds_dir in datasets_dir.iterdir():
                    mf = ds_dir / "meta.json"
                    if not mf.exists():
                        continue
                    try:
                        ds_meta = _load_json(mf)
                    except Exception:
                        continue
                    mid = ds_meta.get("library_root_id") or ds_meta.get("model_id")
                    if not mid:
                        continue
                    # 老数据只写了版本模型 id：换算到该版本所属库根，统计口径与库详情一致
                    mid = metas.get(mid, {}).get("library_root_id") or mid
                    stat = ds_stats.setdefault(mid, {"count": 0})
                    stat["count"] += 1
                    stage = ds_meta.get("stage") or "collecting"
                    stat[stage] = stat.get(stage, 0) + 1

            if not metas:
                return models

            for model_meta in metas.values():
                try:
                    # 添加模型文件大小
                    weights_path = model_meta.get("weights_path")
                    if weights_path and Path(weights_path).exists():
                        model_meta["file_size"] = os.path.getsize(weights_path)
                        model_meta["file_size_mb"] = round(model_meta["file_size"] / (1024 * 1024), 2)
                    # 1.7 模型仓库统计：关联数据集数量 + 阶段分布
                    stat = ds_stats.get(model_meta.get("model_id"), {})
                    model_meta["dataset_count"] = stat.get("count", 0)
                    model_meta["dataset_stats"] = {
                        k: v for k, v in stat.items() if k != "count"
                    }
                    models.append(model_meta)
                except Exception:
                    continue

            return models

        models = await asyncio.to_thread(_list_models_sync)
        return {"models": sorted(models, key=lambda x: (x.get("dataset_count", 0), x.get("created_at", "")), reverse=True)}
    
    async def get_model(self, model_id: str):
        """获取模型详情（包含训练指标）"""
        model_dir = self.registry_dir / model_id
        model_file = model_dir / "model.json"
        
        if not await asyncio.to_thread(lambda: model_file.exists()):
            return None
        
        model_meta = await asyncio.to_thread(_load_json, model_file)
        
        # 添加模型文件大小
        def _get_file_size():
            weights_path = model_meta.get("weights_path")
            if weights_path and Path(weights_path).exists():
                return os.path.getsize(weights_path)
            return None
        
        file_size = await asyncio.to_thread(_get_file_size)
        if file_size:
            model_meta["file_size"] = file_size
            model_meta["file_size_mb"] = round(file_size / (1024 * 1024), 2)
        
        # 并行加载训练指标与模型参数信息（避免串行阻塞，加快模型详情打开速度）
        job_id = model_meta.get("job_id")
        
        async def _load_metrics_async():
            if job_id:
                return await self._load_training_metrics(job_id)
            return None
        
        async def _load_info_async():
            # 已缓存在 model.json 则直接使用，避免每次重新加载权重
            if not model_meta.get("model_info") and model_meta.get("weights_path"):
                model_info = await self._get_model_info(model_meta.get("weights_path"))
                if model_info:
                    model_meta["model_info"] = model_info
                    try:
                        save_meta = _load_json(model_file)
                        save_meta["model_info"] = model_info
                        await asyncio.to_thread(_save_json, model_file, save_meta)
                    except Exception as e:
                        print(f"Error caching model_info: {e}")
            return model_meta.get("model_info")
        
        training_metrics, _ = await asyncio.gather(_load_metrics_async(), _load_info_async())
        if training_metrics:
            model_meta["training_metrics"] = training_metrics
        
        return model_meta
    
    async def _load_training_metrics(self, job_id: str):
        """从训练任务加载训练指标"""
        job_dir = self.jobs_dir / job_id
        
        if not await asyncio.to_thread(lambda: job_dir.exists()):
            return None
        
        def _load_metrics_sync():
            metrics = {}
            
            # 尝试读取 results.csv（YOLO 训练输出）
            # 优先精确路径，避免 rglob 全目录遍历（任务目录可能包含大量预测图片，遍历很慢）
            results_csv = None
            for candidate in [job_dir / "train" / "results.csv", job_dir / "results.csv"]:
                if candidate.exists():
                    results_csv = candidate
                    break
            if results_csv is None:
                results_files = list(job_dir.rglob("results.csv"))
                if results_files:
                    results_csv = results_files[0]
            if results_csv is not None:
                try:
                    metrics["training_history"] = self._parse_results_csv(results_csv)
                except Exception as e:
                    print(f"Error parsing results.csv: {e}")
            
            # 读取训练任务配置
            job_file = job_dir / "job.json"
            if job_file.exists():
                try:
                    job_meta = _load_json(job_file)
                    metrics["job_config"] = {
                        "dataset_id": job_meta.get("dataset_id"),
                        "epochs": job_meta.get("epochs"),
                        "imgsz": job_meta.get("imgsz"),
                        "batch": job_meta.get("batch"),
                        "model_name": job_meta.get("model_name"),
                        "status": job_meta.get("status"),
                        "created_at": job_meta.get("created_at"),
                        "completed_at": job_meta.get("completed_at")
                    }
                except Exception as e:
                    print(f"Error reading job.json: {e}")
            
            return metrics if metrics else None
        
        return await asyncio.to_thread(_load_metrics_sync)
    
    def _parse_results_csv(self, csv_path: Path):
        """解析 YOLO 训练输出的 results.csv（同步方法，在线程中调用）"""
        history = {
            "epochs": [],
            "train_box_loss": [],
            "train_cls_loss": [],
            "train_dfl_loss": [],
            "val_box_loss": [],
            "val_cls_loss": [],
            "val_dfl_loss": [],
            "metrics_precision": [],
            "metrics_recall": [],
            "metrics_mAP50": [],
            "metrics_mAP50_95": []
        }
        
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 清理列名（去除空格）
                    row = {k.strip(): v.strip() for k, v in row.items()}
                    
                    epoch = int(row.get("epoch", len(history["epochs"]) + 1))
                    history["epochs"].append(epoch)
                    
                    # 训练损失
                    if "train/box_loss" in row:
                        history["train_box_loss"].append(float(row["train/box_loss"]))
                    if "train/cls_loss" in row:
                        history["train_cls_loss"].append(float(row["train/cls_loss"]))
                    if "train/dfl_loss" in row:
                        history["train_dfl_loss"].append(float(row["train/dfl_loss"]))
                    
                    # 验证损失
                    if "val/box_loss" in row:
                        history["val_box_loss"].append(float(row["val/box_loss"]))
                    if "val/cls_loss" in row:
                        history["val_cls_loss"].append(float(row["val/cls_loss"]))
                    if "val/dfl_loss" in row:
                        history["val_dfl_loss"].append(float(row["val/dfl_loss"]))
                    
                    # 指标
                    if "metrics/precision(B)" in row:
                        history["metrics_precision"].append(float(row["metrics/precision(B)"]))
                    if "metrics/recall(B)" in row:
                        history["metrics_recall"].append(float(row["metrics/recall(B)"]))
                    if "metrics/mAP50(B)" in row:
                        history["metrics_mAP50"].append(float(row["metrics/mAP50(B)"]))
                    if "metrics/mAP50-95(B)" in row:
                        history["metrics_mAP50_95"].append(float(row["metrics/mAP50-95(B)"]))
            
            # 清理空列表
            history = {k: v for k, v in history.items() if v}
            
            # 添加最终指标摘要
            if history.get("metrics_mAP50"):
                history["final_metrics"] = {
                    "mAP50": history["metrics_mAP50"][-1] if history["metrics_mAP50"] else None,
                    "mAP50_95": history["metrics_mAP50_95"][-1] if history.get("metrics_mAP50_95") else None,
                    "precision": history["metrics_precision"][-1] if history.get("metrics_precision") else None,
                    "recall": history["metrics_recall"][-1] if history.get("metrics_recall") else None
                }
            
            return history
            
        except Exception as e:
            print(f"Error parsing CSV: {e}")
            return None
    
    async def _get_model_info(self, weights_path: str):
        """获取模型参数信息"""
        if not weights_path:
            return None
        
        if not await asyncio.to_thread(lambda: Path(weights_path).exists()):
            return None
        
        def _get_model_info_sync():
            try:
                from ultralytics import YOLO
                model = YOLO(weights_path)
                
                # 获取模型信息
                info = {
                    "task": model.task,
                    "model_type": model.model.yaml.get("yaml_file", "unknown") if hasattr(model.model, "yaml") else "unknown"
                }
                
                # 尝试获取参数数量
                if hasattr(model.model, "model"):
                    total_params = sum(p.numel() for p in model.model.model.parameters())
                    trainable_params = sum(p.numel() for p in model.model.model.parameters() if p.requires_grad)
                    info["total_params"] = total_params
                    info["trainable_params"] = trainable_params
                    info["total_params_m"] = round(total_params / 1e6, 2)  # 百万参数
                
                # 类别名（供上传单个 .pt 时自动回填 classes）
                if getattr(model, "names", None):
                    try:
                        info["classes"] = [str(model.names[k]) for k in sorted(model.names.keys())]
                    except Exception:
                        pass
                
                return info
                
            except Exception as e:
                print(f"Error getting model info: {e}")
                return None
        
        return await asyncio.to_thread(_get_model_info_sync)
    
    async def update_model(self, model_id: str, request):
        """更新模型信息（含阶段0：模型唯一 code / 中文名 / 启用状态）"""
        model_dir = self.registry_dir / model_id
        model_file = model_dir / "model.json"
        
        if not await asyncio.to_thread(lambda: model_file.exists()):
            return None
        
        def _update_model_sync():
            model_meta = _load_json(model_file)
            
            # 更新字段
            if request.name is not None:
                model_meta["name"] = request.name
            if request.description is not None:
                model_meta["description"] = request.description
            if request.tags is not None:
                model_meta["tags"] = request.tags
            if request.model_code is not None:
                code = _normalize_model_code(request.model_code)
                if not code or not MODEL_CODE_RE.match(code):
                    raise ValueError("模型 code 仅支持小写字母/数字/下划线，2~32 位且以字母开头（如 safety_helmet）；填写中文名请改为其英文/拼音形式")
                if self._model_code_exists(model_id, code):
                    raise ValueError(f"模型 code 已被其他模型占用: {code}")
                model_meta["model_code"] = code
                # 同步标签字典文件中的 model_code
                dict_file = model_dir / "labels_dict.json"
                if dict_file.exists():
                    try:
                        dic = _load_json(dict_file)
                        dic["model_code"] = code
                        _save_json(dict_file, dic)
                    except Exception:
                        pass
            if request.display_name is not None:
                new_name = (request.display_name or "").strip()
                if new_name:
                    owner = self._display_name_owner(model_id, new_name)
                    if owner:
                        kind = "算法库" if self._is_library_record(owner) else "模型"
                        raise ValueError(
                            f"名称「{new_name}」已被{kind}占用（ID: {owner}），请换一个名字"
                        )
                model_meta["display_name"] = new_name
            if request.status is not None:
                if request.status not in ("active", "inactive"):
                    raise ValueError("status 仅支持 active / inactive")
                model_meta["status"] = request.status
            if request.model_library_id is not None:
                # 归入算法库：写库标识；传空串则移出库（降级为独立模型）
                lib_id = (request.model_library_id or "").strip()
                if lib_id:
                    model_meta["model_library_id"] = lib_id
                    # 目标是 registry 库根记录（首页建的库）时同步写 library_root_id，
                    # 让库内聚合/生产轮换/标注页"本库已训模型"等主流程立即认账
                    if (self.registry_dir / lib_id / "model.json").exists():
                        model_meta["library_root_id"] = lib_id
                else:
                    model_meta.pop("model_library_id", None)
                    model_meta.pop("library_root_id", None)
            
            model_meta["updated_at"] = datetime.now().isoformat()
            
            _save_json(model_file, model_meta)
            
            return model_meta
        
        result = await asyncio.to_thread(_update_model_sync)
        # 影子库收口：落盘后回读 model.json 真值再写库
        await shadow.sync_model(model_id)
        return result
    
    def _model_code_exists(self, ignore_model_id: str, code: str) -> bool:
        """检查除自身外是否已有模型占用该 code（大小写不敏感）"""
        code = code.lower()
        if not self.registry_dir.exists():
            return False
        for d in self.registry_dir.iterdir():
            if not d.is_dir() or d.name == ignore_model_id:
                continue
            mf = d / "model.json"
            if not mf.exists():
                continue
            try:
                if (str(_load_json(mf).get("model_code", "") or "")).lower() == code:
                    return True
            except Exception:
                continue
        return False

    def _is_library_record(self, model_id: str) -> bool:
        """registry 中的库根占位记录（首页「新建算法库」产生的 empty 模型）。"""
        try:
            m = _load_json(self.registry_dir / model_id / "model.json")
        except Exception:
            return False
        return bool(m.get("empty"))

    def _display_name_owner(self, ignore_model_id: str, name: str) -> str:
        """返回占用了该显示名的其他记录 id（忽略首尾空格与大小写）；无人占用返回 ""。

        算法库根记录也一并纳入检查：库名和模型名会出现在同一个下拉/列表里
        （如预标注的「本库已训模型」、模型管理页的名称列），
        模型与库重名同样会让人分不清选的是哪一个。
        """
        target = (name or "").strip().lower()
        if not target or not self.registry_dir.exists():
            return ""
        for d in self.registry_dir.iterdir():
            if not d.is_dir() or d.name == ignore_model_id:
                continue
            mf = d / "model.json"
            if not mf.exists():
                continue
            try:
                meta = _load_json(mf)
            except Exception:
                continue
            if (str(meta.get("display_name", "") or "")).strip().lower() == target:
                return d.name
        return ""

    # ---------------- 2.2 动态建模型（采集/上传时按 code 精确匹配，无则创建） ----------------

    def find_model_by_code(self, code: str):
        """按归一化 model_code 精确匹配模型（小写 + 去分隔符），未命中返回 None"""
        norm = _normalize_model_code(code)
        if not norm or not self.registry_dir.exists():
            return None
        for d in self.registry_dir.iterdir():
            if not d.is_dir():
                continue
            mf = d / "model.json"
            if not mf.exists():
                continue
            try:
                meta = _load_json(mf)
                if _normalize_model_code(meta.get("model_code") or "") == norm:
                    return meta
            except Exception:
                continue
        return None

    def resolve_library_root_id(self, model_id: str):
        """把任意模型 id 换算成它所属「算法库」的根记录 id。

        算法库 = 一条 empty 模型记录；数据集归属、训练产出版本都统一记这个根 id。
        历史记录没写该字段时认为它自身就是库根；模型不存在返回 None。
        """
        if not model_id:
            return None
        try:
            meta = _load_json(self.registry_dir / model_id / "model.json")
        except Exception:
            return None
        return meta.get("library_root_id") or model_id

    async def create_empty_model(self, code: str, display_name: str = "", business: str = "", description: str = ""):
        """动态创建空白模型（空标签字典 + 采集库）。code 已存在则报错（唯一标识冲突）。

        用于首页「+ 新建算法库」：唯一标识是全局键，撞号必须提示而非静默复用，
        否则用户以为建了新库，实际跳进了别人的库。
        """
        norm = _normalize_model_code(code)
        if not norm or not MODEL_CODE_RE.match(norm):
            raise ValueError("模型 code 仅支持小写字母/数字/下划线，2~32 位且以字母开头（如 traffic_scene）")

        existing = self.find_model_by_code(norm)
        if existing:
            raise ValueError(
                f"唯一标识「{norm}」已存在（{existing.get('display_name') or existing.get('name') or existing.get('model_id')}），请换一个"
            )

        def _create_sync():
            model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_dir = self.registry_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            now = datetime.now().isoformat()
            # 统一 model.json 字段规范（与上传/训练路径保持一致）：
            # 基础标识 + 状态 + 关联标签字典(labels_file) + 类别列表 + 审计时间
            meta = {
                "model_id": model_id,
                "model_code": norm,
                "name": display_name or code,
                "display_name": display_name or "",
                "business": business or "",
                "status": "active",
                "source": "created",
                "empty": True,               # 空模型：待采集 / 未有训练产出
                # 算法库根标识：本记录即"库"本身，其下训练产出的版本都会带上同一个 id，
                # 库详情据此聚合全部模型版本（与 PG 的 model_libraries 表无关）
                "library_root_id": model_id,
                "classes": [],               # 冗余英文码列表（与 labels_dict.json 一一对应，data.yaml names 来源）
                "labels_file": "labels_dict.json",
                "description": description or "",
                "tags": [],
                # M11：architecture 标准字段集（空模型暂无产出，置空占位）
                "architecture_family": "",
                "architecture_name": "",
                "model_scale": "",
                "created_at": now,
                "updated_at": now,
            }
            _save_json(model_dir / "model.json", meta)
            _save_json(model_dir / "labels_dict.json", {"model_code": norm, "labels": []})
            return meta

        result = await asyncio.to_thread(_create_sync)
        if result.get("model_id"):
            await shadow.sync_model(result["model_id"])
        return result

    # ---------------- 2.7 相似模型排查与合并（人工工具） ----------------

    def _model_label_codes(self, model_dir: Path) -> list:
        """读取模型的 english_code 类别列表（相似度比对 / 合并差集用）。

        优先读标签字典 labels_dict.json；不存在（尚未初始化）时回退到
        model.json 的 classes，保证相似扫描与合并对历史模型同样生效。
        """
        file = model_dir / "labels_dict.json"
        if file.exists():
            try:
                return [(str(l.get("english_code")) or "").strip() for l in (_load_json(file).get("labels") or [])]
            except Exception:
                pass
        try:
            meta = _load_json(model_dir / "model.json")
            return [str(c) for c in (meta.get("classes") or []) if c]
        except Exception:
            return []

    @staticmethod
    def _norm_class_name(name: str) -> str:
        """归一化类别名用于比对（小写 + 去所有分隔符），如 traffic-scene == trafficscene"""
        return re.sub(r"[^0-9a-z]+", "", (name or "").strip().lower())

    def _model_dataset_count(self, model_id: str) -> int:
        """统计挂载到某模型的数据集数量"""
        datasets_dir = settings.DATASETS_DIR
        if not datasets_dir.exists():
            return 0
        count = 0
        for ds_dir in datasets_dir.iterdir():
            mf = ds_dir / "meta.json"
            if not mf.exists():
                continue
            try:
                if _load_json(mf).get("model_id") == model_id:
                    count += 1
            except Exception:
                continue
        return count

    async def find_similar_models(self, min_similarity: float = 0.5):
        """相似模型排查（2.7）：按标签字典类别名的 Jaccard 相似度扫描所有模型对。

        - 相似度 = |交集| / |并集|（归一化类别名，traffic-scene == trafficscene）
        - 自动建议主模型：数据集多者为主（合并时的吸收方向）
        返回所有相似度 >= 阈值 的模型对，供前端提示用户确认是否合并。
        """
        def _sync():
            items = []  # (model_id, meta, label_set, dataset_count)
            if not self.registry_dir.exists():
                return []
            for model_dir in self.registry_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                mf = model_dir / "model.json"
                if not mf.exists():
                    continue
                try:
                    meta = _load_json(mf)
                except Exception:
                    continue
                codes = self._model_label_codes(model_dir)
                label_set = {self._norm_class_name(c) for c in codes if self._norm_class_name(c)}
                items.append((model_dir.name, meta, label_set, self._model_dataset_count(model_dir.name)))

            pairs = []
            n = len(items)
            for i in range(n):
                for j in range(i + 1, n):
                    a_id, a_meta, a_set, a_cnt = items[i]
                    b_id, b_meta, b_set, b_cnt = items[j]
                    union = a_set | b_set
                    if not union:
                        continue
                    sim = len(a_set & b_set) / len(union)
                    if sim < min_similarity:
                        continue
                    # 自动选主：数据集多者为主；同数时创建更早者为主
                    if a_cnt != b_cnt:
                        main_id, sub_id = (a_id, b_id) if a_cnt > b_cnt else (b_id, a_id)
                        main_cnt, sub_cnt = max(a_cnt, b_cnt), min(a_cnt, b_cnt)
                    else:
                        main_id, sub_id = (a_id, b_id) if a_meta.get("created_at", "") <= b_meta.get("created_at", "") else (b_id, a_id)
                        main_cnt = sub_cnt = a_cnt

                    def brief(mid, meta, cnt):
                        return {
                            "model_id": mid,
                            "name": meta.get("display_name") or meta.get("name") or meta.get("model_code"),
                            "code": meta.get("model_code"),
                            "dataset_count": cnt,
                            "empty": bool(meta.get("empty")),
                            "status": meta.get("status"),
                        }

                    pairs.append({
                        "a": brief(a_id, a_meta, a_cnt),
                        "b": brief(b_id, b_meta, b_cnt),
                        "similarity": round(sim, 3),
                        "common_classes": sorted(a_set & b_set),
                        "suggested_main": main_id,
                        "suggested_main_datasets": main_cnt,
                        "sub_datasets": sub_cnt,
                    })
            return sorted(pairs, key=lambda p: p["similarity"], reverse=True)

        return {"pairs": await asyncio.to_thread(_sync), "min_similarity": min_similarity}

    async def merge_models(self, main_model_id: str, merged_model_ids: list, reason: str = None) -> dict:
        """合并相似模型（2.7 人工工具，前端确认后执行）

        - 差集类别自动并入主模型标签字典（仅追加，不动已有 index，避免破坏在役数据集）
        - 历史版本保留为分支：被合并模型目录不删除，仅标记 merged_into / merged_at
        - 被合并模型挂载的数据集重新归属主模型（数据集 meta 记录 merged_from_model 来源）
        - 全程写入合并日志 _merge_log.json（含 rebind 原始映射，可用于回滚）
        """
        def _sync():
            now = datetime.now().isoformat()
            main_dir = self.registry_dir / main_model_id
            if not (main_dir / "model.json").exists():
                raise ValueError(f"主模型 {main_model_id} 不存在")

            merged = []
            for mid in merged_model_ids or []:
                if mid == main_model_id:
                    continue
                if (self.registry_dir / mid / "model.json").exists():
                    merged.append(mid)
            if not merged:
                raise ValueError("未选择可合并的相似模型")

            # 1) 差集类别并入主模型标签字典（追加）
            main_dict = self._load_labels_dict_sync(main_dir)
            main_codes = {self._norm_class_name(l.get("english_code") or "") for l in (main_dict.get("labels") or [])}
            merged_detail = []
            for mid in merged:
                mdir = self.registry_dir / mid
                md = self._load_labels_dict_sync(mdir)
                added = []
                for l in md.get("labels") or []:
                    code = (l.get("english_code") or "").strip()
                    if not code or self._norm_class_name(code) in main_codes:
                        continue
                    main_dict["labels"].append({
                        "index": len(main_dict["labels"]),
                        "english_code": code,
                        "chinese_name": (l.get("chinese_name") or "").strip(),
                        "chinese_desc": (l.get("chinese_desc") or "").strip(),
                        "ai_prompt_en": (l.get("ai_prompt_en") or "").strip(),
                    })
                    main_codes.add(self._norm_class_name(code))
                    added.append(code)
                merged_detail.append({"model_id": mid, "added_classes": added})
            _save_json(main_dir / "labels_dict.json", main_dict)

            # 2) 血缘与数据集重归属
            main_meta = _load_json(main_dir / "model.json")
            merged_from = list(main_meta.get("merged_from") or [])
            rebind_map = {}     # 被合并模型 -> [数据集 id]
            original_map = {}   # 数据集 id -> 原模型 id（回滚依据）
            for mid in merged:
                mdir = self.registry_dir / mid
                mmeta = _load_json(mdir / "model.json")
                mmeta["merged_into"] = main_model_id
                mmeta["merged_at"] = now
                _save_json(mdir / "model.json", mmeta)
                merged_from.append({"model_id": mid, "merged_at": now, "reason": reason or "相似模型合并"})
                # 数据集重新归属主模型
                datasets_dir = settings.DATASETS_DIR
                if datasets_dir.exists():
                    for ds_dir in datasets_dir.iterdir():
                        mf = ds_dir / "meta.json"
                        if not mf.exists():
                            continue
                        try:
                            dm = _load_json(mf)
                        except Exception:
                            continue
                        if dm.get("model_id") != mid:
                            continue
                        original_map[ds_dir.name] = mid
                        dm["model_id"] = main_model_id
                        dm["merged_from_model"] = mid
                        _save_json(mf, dm)
                        rebind_map.setdefault(mid, []).append(ds_dir.name)
            main_meta["merged_from"] = merged_from
            main_meta["updated_at"] = now
            _save_json(main_dir / "model.json", main_meta)

            # 3) 合并日志（含 rebind 原始映射，供回滚）
            log_entry = {
                "at": now,
                "main_model_id": main_model_id,
                "merged_model_ids": merged,
                "classes_added": merged_detail,
                "datasets_rebound": rebind_map,
                "datasets_original": original_map,
                "reason": reason or "相似模型合并（人工工具）",
            }
            log_file = self.registry_dir / "_merge_log.json"
            logs = []
            if log_file.exists():
                try:
                    logs = _load_json(log_file)
                except Exception:
                    logs = []
            logs.append(log_entry)
            _save_json(log_file, logs)

            return {
                "ok": True,
                "main_model_id": main_model_id,
                "merged_model_ids": merged,
                "classes_added_count": sum(len(m["added_classes"]) for m in merged_detail),
                "datasets_rebound": sum(len(v) for v in rebind_map.values()),
                "log_path": str(log_file),
            }

        result = await asyncio.to_thread(_sync)
        # 影子库收口：合并会改主模型、被合并模型（merged_into）与重归属数据集的 model_id
        if result.get("ok"):
            for mid in [result.get("main_model_id"), *(result.get("merged_model_ids") or [])]:
                if mid:
                    await shadow.sync_model(mid)
            await shadow.refresh_all_datasets()
        return result

    async def merge_log(self, limit: int = 20) -> dict:
        """读取合并日志（2.7 回滚依据）"""
        def _sync():
            log_file = self.registry_dir / "_merge_log.json"
            if not log_file.exists():
                return []
            try:
                logs = _load_json(log_file)
                if not isinstance(logs, list):
                    return []
                logs = sorted(logs, key=lambda x: x.get("at", ""), reverse=True)
                return logs[:limit]
            except Exception:
                return []
        return {"logs": await asyncio.to_thread(_sync)}

    async def rollback_merge(self, log_index: int = -1) -> dict:
        """回滚最近一次（或指定下标）的合并：还原数据集归属、撤销 merged_into 标记。

        注意：并入主模型的标签类别保留（追加式的，删除会重排 index 造成数据失配）。
        """
        def _sync():
            log_file = self.registry_dir / "_merge_log.json"
            if not log_file.exists():
                raise ValueError("无合并日志可回滚")
            logs = _load_json(log_file)
            if not isinstance(logs, list) or not logs:
                raise ValueError("无合并日志可回滚")
            entry = logs[log_index]
            original_map = entry.get("datasets_original") or {}
            datasets_dir = settings.DATASETS_DIR
            restored = []
            for ds_id, orig_mid in original_map.items():
                mf = datasets_dir / ds_id / "meta.json"
                if not mf.exists():
                    continue
                try:
                    dm = _load_json(mf)
                    dm["model_id"] = orig_mid
                    dm.pop("merged_from_model", None)
                    _save_json(mf, dm)
                    restored.append(ds_id)
                except Exception:
                    continue
            # 撤销被合并模型的 merged_into 标记（历史版本保留本身不受影响）
            unmarked = []
            for mid in entry.get("merged_model_ids") or []:
                mf = self.registry_dir / mid / "model.json"
                if not mf.exists():
                    continue
                try:
                    mmeta = _load_json(mf)
                    mmeta.pop("merged_into", None)
                    mmeta.pop("merged_at", None)
                    _save_json(mf, mmeta)
                    unmarked.append(mid)
                except Exception:
                    continue
            logs.pop(log_index)
            if logs:
                _save_json(log_file, logs)
            else:
                log_file.unlink(missing_ok=True)
            return {
                "ok": True,
                "restored_datasets": restored,
                "unmarked_models": unmarked,
                "removed_log_index": log_index,
            }

        result = await asyncio.to_thread(_sync)
        # 影子库收口：回滚撤销了被合并模型的 merged_into 标记，数据集归属同步还原
        if result.get("ok"):
            for mid in result.get("unmarked_models") or []:
                await shadow.sync_model(mid)
            await shadow.refresh_all_datasets()
        return result

    # ---------------- 统一标签字典（四字段） ----------------

    async def get_labels_dict(self, model_id: str):
        """获取模型标签字典；不存在时从 model.json 的 classes 自动初始化（懒生成）。

        字典五字段：index（YOLO 训练序号）/ english_code（检测/DINO 用）/ chinese_name（UI 用）/
        chinese_desc（Qwen 精标用）/ ai_prompt_en（GroundingDINO、YOLO-World 的英文检测提示词）。
        """
        model_dir = self.registry_dir / model_id
        if not await asyncio.to_thread(lambda: model_dir.exists()):
            return None
        return await asyncio.to_thread(self._load_labels_dict_sync, model_dir)

    async def update_labels_dict(self, model_id: str, labels: list | None):
        """保存标签字典（全量覆写）。自动重排连续 index、去重校验 english_code。

        在用锁死保护：本库只要有数据用过标签（画过框，或数据集类名清单里出现过），
        现有类别的英文名与顺序就整体锁死，只能改中文名/中文描述/英文检测提示词，新类别只能追加到末尾
        （class_id 按 index 存储，重排会导致已标注数据类别错乱；改名会让重新导入整批丢框）。
        """
        model_dir = self.registry_dir / model_id
        if not await asyncio.to_thread(lambda: model_dir.exists()):
            return None
        if not labels:
            raise ValueError("标签字典不能为空")

        def _sync():
            # ---- 在用标签锁死：现有类别整体锁死，只允许末尾追加 ----
            # 判据（任一命中即锁定）：本库有数据画过该类的框 / 类名清单里出现过该类名。
            # 三条理由：① class_id 按 index 存，插队或删行会让已标注类别错位；
            #          ② 导入靠"类名 ↔ 英文名"逐字对齐，改名会让整批框丢掉；
            #          ③ 封板/训练过的数据集要能复现，字典不能事后变脸。
            # 老字典可能有空 english_code 的行，必须按下标逐位比对（不能过滤，否则错位）。
            old = self._load_labels_dict_sync(model_dir).get("labels") or []
            old_pos = [str(l.get("english_code") or "").strip().lower() for l in old]
            new_pos = [str(l.get("english_code") or "").strip().lower() for l in labels]
            usage = self._labels_usage_sync(model_id)
            if usage and any(old_pos):
                # 死线 = 最后一个"被用过"的类别所在行；它之前的行名字和顺序全锁死，
                # 它之后的（还没人用上的）随便改、随便删，新类别一律追加到末尾。
                frozen = max((i for i, c in enumerate(old_pos) if usage.get(c)), default=-1)
                if frozen >= 0 and new_pos[:frozen + 1] != old_pos[:frozen + 1]:
                    for i in range(frozen + 1):
                        want = old_pos[i]
                        got = new_pos[i] if i < len(new_pos) else ""
                        if got != want:
                            u = usage.get(want) or {"images": 0, "datasets": 0}
                            used = "、".join(filter(None, [
                                f"{u['images']} 张图已标注" if u["images"] else "",
                                f"{u['datasets']} 个数据集引用了它" if u["datasets"] else "",
                            ])) or "已被数据使用"
                            extra = "" if u["images"] or u["datasets"] else "（排在已用类别前面，位置不能挪）"
                            raise ValueError(
                                f"第 {i + 1} 个类别「{want or '（未填英文名）'}」{used}{extra}，"
                                f"不能改名、删除或调整顺序（本次改成了"
                                f"{'「' + got + '」' if got else '「已删除」'}）。"
                                "新增类别请追加到列表末尾；中文名和中文描述随时可改"
                            )
                    raise ValueError("已有类别的顺序不能调整，新增类别请追加到列表末尾")

            seen = set()
            new_labels = []
            for i, item in enumerate(labels):
                ec = (item.get("english_code") or "").strip()
                if not ec:
                    raise ValueError(f"第 {i + 1} 行 english_code 不能为空")
                key = ec.lower()
                if key in seen:
                    raise ValueError(f"english_code 重复: {ec}")
                seen.add(key)
                new_labels.append({
                    "index": i,
                    "english_code": ec,
                    "chinese_name": (item.get("chinese_name") or "").strip(),
                    "chinese_desc": (item.get("chinese_desc") or "").strip(),
                    "ai_prompt_en": (item.get("ai_prompt_en") or "").strip(),
                })
            model_meta = _load_json(model_dir / "model.json")
            dic = {
                "model_code": model_meta.get("model_code"),
                "labels": new_labels,
            }
            _save_json(model_dir / "labels_dict.json", dic)
            return dic
        result = await asyncio.to_thread(_sync)
        # 影子库收口：labels_dict.json 是 models.labels / labels_file 列的来源，改字典即改库中真值
        await shadow.sync_model(model_id)
        return result

    def init_labels_from_classes(self, model_id: str, classes: list) -> bool:
        """标签字典自动生成（prepare 后懒回填）：归属模型的标签字典为空时，
        用数据集检测到的真实类名初始化 labels_dict.json + model.json classes。

        幂等：仅在字典为空时写入（有人工维护/合并结果的字典不覆盖，避免
        重排 index 导致已标注数据类别错乱）。返回是否写入。
        """
        model_dir = self.registry_dir / model_id
        if not (model_dir / "model.json").exists():
            return False
        dic = self._load_labels_dict_sync(model_dir)
        if (dic.get("labels") or []):
            return False  # 已有标签（人工维护/合并结果），不覆盖
        uniq = []
        seen = set()
        for c in classes or []:
            c = str(c).strip()
            if not c:
                continue
            k = self._norm_class_name(c)
            if k and k not in seen:
                seen.add(k)
                uniq.append(c)
        if not uniq:
            return False
        labels = [
            {"index": i, "english_code": c, "chinese_name": "", "chinese_desc": "", "ai_prompt_en": ""}
            for i, c in enumerate(uniq)
        ]
        meta = _load_json(model_dir / "model.json")
        meta["classes"] = uniq
        # 注意：绝不能用 empty 表达"标签字典已初始化"。empty 的唯一语义是
        # 「算法库占位记录（非训练产出）」——模型管理页靠 !empty 把它过滤出模型列表，
        # 「算法库」下拉靠 empty 收录库名（见 routes/models.py list_libraries）。
        # 这里改写 empty 会让新建的库同时出现两个相反的症状：
        # 模型列表里冒出来 + 下拉里消失。标签是否已初始化请用 labels_auto_init 判定。
        meta["labels_auto_init"] = True
        meta["labels_auto_init_at"] = datetime.now().isoformat()
        meta["updated_at"] = meta["labels_auto_init_at"]
        _save_json(model_dir / "model.json", meta)
        _save_json(model_dir / "labels_dict.json", {
            "model_code": meta.get("model_code"),
            "labels": labels,
        })
        logger.info(f"model {model_id}: labels dict auto-initialized from dataset classes: {uniq}")
        return True

    def _labels_usage_sync(self, model_id: str) -> dict:
        """统计本算法库每个标签"有没有被数据用上"（禁改保护的判据）。两条判据取并集：

        ① 画过框：本库任一数据集的图片上有该类的框。老数据的框里只有类别号
           class_id、没有 english_code，必须按序号回查标签字典认类，否则永远统计不到。
        ② 导入过：数据集准备时解析出的类名清单（meta.classes）里出现过该类名。
           一张还没画也算 —— 导入/重新准备都靠"类名 ↔ 英文名"逐字对齐，改名会整批丢框。

        数据集范围与库详情「数据集管理」「文案同步」同口径（按算法库根换算，
        老数据只写了版本模型 id 的也算进来）。
        返回 {english_code 小写: {"images": 有该类框的图片数, "datasets": 引用它的数据集数}}
        """
        usage: dict = {}

        def _touch(code: str, kind: str):
            code = (code or "").strip().lower()
            if not code:
                return
            item = usage.setdefault(code, {"images": 0, "datasets": 0})
            item[kind] += 1

        # 类别号 → 英文名（老标注只有序号，靠它回查）
        labels = sorted(
            self._load_labels_dict_sync(self.registry_dir / model_id).get("labels") or [],
            key=lambda x: int(x.get("index") or 0),
        )
        code_by_index = {i: str(l.get("english_code") or "").strip() for i, l in enumerate(labels)}

        root_id = self.resolve_library_root_id(model_id) or model_id
        for ds_dir in self._library_dataset_dirs_sync(root_id):
            try:
                ds_meta = _load_json(ds_dir / "meta.json") or {}
            except Exception:
                continue
            # 判据②：数据集自带的类名清单
            referenced = False
            for name in ds_meta.get("classes") or []:
                key = str(name or "").strip().lower()
                if key in usage or any(key == (l.get("english_code") or "").strip().lower() for l in labels):
                    referenced = True
                    _touch(key, "datasets")
            # 判据①：图片标注框（每图每类只计一次）
            ann_dir = ds_dir / "annotations"
            if not ann_dir.exists():
                continue
            for p in ann_dir.glob("*.json"):
                try:
                    boxes = (_load_json(p) or {}).get("boxes") or []
                except Exception:
                    continue
                seen = set()
                for b in boxes:
                    if not isinstance(b, dict):
                        continue
                    code = (b.get("english_code") or "").strip()
                    if not code:
                        try:
                            cid = int(b.get("class_id"))
                        except (TypeError, ValueError):
                            cid = -1
                        code = code_by_index.get(cid, "")
                    key = code.lower()
                    if key and key not in seen:
                        seen.add(key)
                        _touch(key, "images")
                        if not referenced:
                            referenced = True
                            _touch(key, "datasets")
        return usage

    async def get_labels_usage(self, model_id: str):
        """标签引用统计（供前端把"在用"的格子锁住、并显示已标多少张图）。

        模型不存在返回 None；dict_locked 为真表示本库已有数据引用标签，
        此时现有类别的名字与顺序整体锁死，只能在末尾追加。
        """
        def _sync():
            model_dir = self.registry_dir / model_id
            if not (model_dir / "model.json").exists():
                return None
            labels = sorted(
                self._load_labels_dict_sync(model_dir).get("labels") or [],
                key=lambda x: int(x.get("index") or 0),
            )
            usage = self._labels_usage_sync(model_id)
            rows = []
            for i, l in enumerate(labels):
                code = str(l.get("english_code") or "").strip()
                u = usage.get(code.lower(), {"images": 0, "datasets": 0})
                rows.append({
                    "index": i,
                    "english_code": code,
                    "images": u["images"],
                    "datasets": u["datasets"],
                    "in_use": bool(u["images"] or u["datasets"]),
                })
            # 死线：最后一个在用的类别所在行；与 update_labels_dict 的校验口径保持一致
            frozen = max((r["index"] for r in rows if usage.get(r["english_code"].lower())), default=-1)
            for r in rows:
                r["locked"] = r["index"] <= frozen
            return {
                "model_id": model_id,
                "library_root_id": self.resolve_library_root_id(model_id) or model_id,
                "dict_locked": bool(usage),
                "frozen_until": frozen,
                "labels": rows,
            }
        return await asyncio.to_thread(_sync)

    async def suggest_labels_from_dataset(self, model_id: str, dataset_id: str, limit: int = 3) -> dict:
        """AI 识别数据集图片中的新标签（辅助不懂补充什么类别的用户）。

        优先取 困难样本(hard) / 空白(background) / 未标注 的图片发给千问 VL，
        找出已知标签之外的新目标候选（四字段），过滤已存在类别后返回。
        """
        from src.services.sam_service import SAMService
        limit = max(1, min(int(limit or 3), 10))
        ds_dir = settings.DATASETS_DIR / dataset_id
        meta_file = ds_dir / "meta.json"
        if not await asyncio.to_thread(lambda: meta_file.exists()):
            return {"ok": False, "message": f"数据集 {dataset_id} 不存在"}

        def _collect_sync():
            try:
                _load_json(meta_file)
            except Exception:
                pass
            ann_dir = ds_dir / "annotations"
            picks = []
            if ann_dir.exists():
                for p in ann_dir.glob("*.json"):
                    try:
                        data = _load_json(p)
                    except Exception:
                        continue
                    boxes = data.get("boxes") or []
                    hard = data.get("sample_type") == "hard"
                    if boxes and not hard:
                        continue  # 已正常标注的跳过
                    img_rel = data.get("image_path") or ""
                    # image_path 相对 DATA_DIR（如 datasets/ds_x/images/xxx.jpg）
                    img_abs = (settings.DATA_DIR / img_rel) if img_rel else None
                    if img_abs and img_abs.exists():
                        picks.append((0 if hard else 1, str(img_abs)))
            picks.sort(key=lambda x: x[0])
            return [p for _, p in picks[:limit]]

        paths = await asyncio.to_thread(_collect_sync)
        if not paths:
            return {"ok": False, "message": f"数据集 {dataset_id} 没有可识别的候选图（困难/空白/未标注）"}

        known = []
        try:
            dic = asyncio.to_thread(self._load_labels_dict_sync, settings.REGISTRY_DIR / model_id)
            known = [l.get("english_code") or "" for l in ((await dic).get("labels") or []) if l.get("english_code")]
        except Exception:
            known = []

        candidates, msg = SAMService().qwen_suggest_labels(paths, known)
        if candidates is None:
            return {"ok": False, "message": msg}
        if not candidates:
            return {"ok": True, "message": f"识别 {len(paths)} 张图，未发现已知类别之外的新目标", "candidates": []}
        import os as _os
        out = []
        for c in candidates:
            c = dict(c)
            c["images"] = [_os.path.basename(p) for p in c.get("images", [])]
            out.append(c)
        return {
            "ok": True,
            "message": msg or f"识别 {len(paths)} 张图，发现 {len(out)} 个新类别候选（AI 建议，需人工确认）",
            "candidates": out,
        }

    async def adopt_suggested_labels(self, model_id: str, suggestions: list) -> dict:
        """采纳 AI 建议的新标签：追加入标签字典末尾（跳过已存在项）。

        只追加、不改旧标签，天然符合追加禁删保护语义。
        """
        def _sync():
            model_dir = self.registry_dir / model_id
            if not model_dir.exists():
                return None
            dic = self._load_labels_dict_sync(model_dir)
            labels = dic.get("labels") or []
            seen = {str(l.get("english_code") or "").strip().lower() for l in labels}
            added, skipped = [], []
            for s in suggestions or []:
                code = (s.get("english_code") or "").strip()
                key = code.lower()
                if not code:
                    continue
                if key in seen:
                    skipped.append(code)
                    continue
                seen.add(key)
                labels.append({
                    "index": len(labels),
                    "english_code": code,
                    "chinese_name": (s.get("chinese_name") or "").strip(),
                    "chinese_desc": (s.get("chinese_desc") or "").strip(),
                    "ai_prompt_en": (s.get("ai_prompt_en") or "").strip(),
                })
                added.append(code)
            if added:
                _save_json(model_dir / "labels_dict.json", {"model_code": dic.get("model_code"), "labels": labels})
            return {"added": added, "skipped": skipped}
        result = await asyncio.to_thread(_sync)
        # 影子库收口：采纳建议等于扩写标签字典，同步 models.labels
        if result.get("added"):
            await shadow.sync_model(model_id)
        return result

    # ------------------------------------------------------------------
    # 标签文案同步：把标签字典的中文名/描述刷进本库已标注数据（每图 JSON 的框冗余字段）
    # ------------------------------------------------------------------
    def _library_dataset_dirs_sync(self, root_id: str) -> list:
        """归属该算法库（库根 = root_id）的数据集目录列表。

        口径与库详情「数据集管理」完全一致：数据集归属先换算成库根再比对
        （老数据只写了版本模型 id），未归属与悬空引用一律不算。
        """
        metas = {}
        if self.registry_dir.exists():
            for mdir in self.registry_dir.iterdir():
                mf = mdir / "model.json"
                if not mf.exists():
                    continue
                try:
                    metas[mdir.name] = _load_json(mf)
                except Exception:
                    continue
        out = []
        datasets_dir = settings.DATASETS_DIR
        if not datasets_dir.exists():
            return out
        for ds_dir in datasets_dir.iterdir():
            mf = ds_dir / "meta.json"
            if not mf.exists():
                continue
            try:
                ds_meta = _load_json(mf)
            except Exception:
                continue
            owner = ds_meta.get("library_root_id") or ds_meta.get("model_id")
            if not owner:
                continue
            owner = metas.get(owner, {}).get("library_root_id") or owner
            if owner == root_id:
                out.append(ds_dir)
        return out

    def _labels_sync_backup_root(self, model_id: str) -> Path:
        return settings.DATA_DIR / "label_sync_backup" / model_id

    async def sync_labels_to_annotations(self, model_id: str, dry_run: bool = True) -> dict:
        """把标签字典的中文名/描述同步到本库已标注数据（默认只预览不改写）。

        只刷「展示/提示词」字段（chinese_name、chinese_desc）；class_id 与
        english_code 一律不动，因此不影响训练产物与已封板数据。
        正式同步前把被改的原始 JSON 备份到 data/label_sync_backup/<模型>/<时间戳>/，
        可用 rollback_labels_sync 整体回退。
        """
        def _sync():
            model_dir = self.registry_dir / model_id
            if not (model_dir / "model.json").exists():
                return None
            dic = self._load_labels_dict_sync(model_dir)
            labels = sorted(dic.get("labels") or [], key=lambda x: int(x.get("index", 0)))
            by_code = {}
            for l in labels:
                code = str(l.get("english_code") or "").strip().lower()
                if code:
                    by_code[code] = l
            if not by_code:
                raise ValueError("标签字典为空，没有可同步的文案")

            ds_dirs = self._library_dataset_dirs_sync(
                self.resolve_library_root_id(model_id) or model_id
            )
            stats = {"datasets": len(ds_dirs), "files_scanned": 0,
                     "files_changed": 0, "boxes_changed": 0}
            samples = []
            touched = []  # [(原文件路径, 改写后的标注数据, 备份文件路径)]
            backup_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_root = self._labels_sync_backup_root(model_id) / backup_id

            for ds_dir in ds_dirs:
                ann_dir = ds_dir / "annotations"
                if not ann_dir.is_dir():
                    continue
                for fp in sorted(ann_dir.glob("*.json")):
                    try:
                        data = _load_json(fp)
                    except Exception:
                        continue
                    stats["files_scanned"] += 1
                    boxes = data.get("boxes") or []
                    file_changed = 0
                    for b in boxes:
                        if not isinstance(b, dict):
                            continue
                        code = str(b.get("english_code") or "").strip().lower()
                        if not code:
                            # 老标注可能没写英文码：按 class_id 回查字典序号
                            cid = b.get("class_id")
                            if isinstance(cid, int) and 0 <= cid < len(labels):
                                code = str(labels[cid].get("english_code") or "").strip().lower()
                        lab = by_code.get(code)
                        if not lab:
                            continue
                        want_name = (lab.get("chinese_name") or "").strip()
                        want_desc = (lab.get("chinese_desc") or "").strip()
                        for field, want in (("chinese_name", want_name), ("chinese_desc", want_desc)):
                            # 字典该列为空时不覆盖已有文案（避免把描述清空）
                            if not want or b.get(field) == want:
                                continue
                            if len(samples) < 10:
                                samples.append({
                                    "dataset_id": ds_dir.name,
                                    "image_id": data.get("image_id") or fp.stem,
                                    "english_code": code,
                                    "field": "中文名" if field == "chinese_name" else "描述",
                                    "before": b.get(field) or "",
                                    "after": want,
                                })
                            b[field] = want
                            file_changed += 1
                    if file_changed:
                        stats["files_changed"] += 1
                        stats["boxes_changed"] += file_changed
                        touched.append((fp, data, backup_root / ds_dir.name / fp.name))

            if not dry_run and touched:
                # 先整体备份原件，再回写改写结果（备份失败则不动原数据）
                for fp, _data, bk in touched:
                    bk.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(fp, bk)
                for fp, data, _bk in touched:
                    _save_json(fp, data)

            result = {
                "model_id": model_id,
                "dry_run": dry_run,
                "stats": stats,
                "samples": samples,
                "backup_id": None if dry_run or not touched else backup_id,
                "last_backup": None,
            }
            meta_path = model_dir / "model.json"
            try:
                meta = _load_json(meta_path)
                lb = meta.get("labels_sync_backup")
                if lb:
                    result["last_backup"] = {
                        "backup_id": lb.get("backup_id"),
                        "at": lb.get("at"),
                        "files": lb.get("files"),
                        "boxes": lb.get("boxes"),
                    }
            except Exception:
                meta = {}

            if not dry_run and touched:
                meta["labels_sync_backup"] = {
                    "backup_id": backup_id,
                    "at": datetime.now().isoformat(),
                    "files": stats["files_changed"],
                    "boxes": stats["boxes_changed"],
                }
                meta["updated_at"] = meta["labels_sync_backup"]["at"]
                _save_json(meta_path, meta)
                result["last_backup"] = meta["labels_sync_backup"]
            return result

        result = await asyncio.to_thread(_sync)
        # 影子库收口：非试运行且确有改写时，model.json 会写入 labels_sync_backup
        if result.get("backup_id"):
            await shadow.sync_model(model_id)
        return result

    async def rollback_labels_sync(self, model_id: str) -> dict:
        """回退最近一次标签文案同步：用备份原件覆盖回写受影响标注文件。"""
        def _sync():
            model_dir = self.registry_dir / model_id
            meta_path = model_dir / "model.json"
            if not meta_path.exists():
                return None
            meta = _load_json(meta_path)
            lb = meta.get("labels_sync_backup") or {}
            backup_id = str(lb.get("backup_id") or "")
            if not backup_id or "/" in backup_id or "\\" in backup_id or ".." in backup_id:
                raise ValueError("没有可回退的标签文案同步记录")
            backup_root = self._labels_sync_backup_root(model_id) / backup_id
            if not backup_root.is_dir():
                raise ValueError(f"备份目录不存在：{backup_root}")
            restored = 0
            for bk in sorted(backup_root.rglob("*.json")):
                rel = bk.relative_to(backup_root)
                parts = rel.parts  # (dataset_id, image.json)
                if len(parts) != 2:
                    continue
                target = settings.DATASETS_DIR / parts[0] / "annotations" / parts[1]
                if not target.parent.is_dir():
                    continue
                shutil.copy2(bk, target)
                restored += 1
            meta.pop("labels_sync_backup", None)
            meta["updated_at"] = datetime.now().isoformat()
            _save_json(meta_path, meta)
            return {"model_id": model_id, "restored_files": restored, "backup_id": backup_id}

        result = await asyncio.to_thread(_sync)
        # 影子库收口：回退后 model.json 的 labels_sync_backup 已被移除
        if result and result.get("restored_files") is not None:
            await shadow.sync_model(model_id)
        return result

    def _load_labels_dict_sync(self, model_dir: Path) -> dict:
        """读取标签字典；不存在则从 model.json 的 classes 懒初始化并写回"""
        file = model_dir / "labels_dict.json"
        if file.exists():
            return _load_json(file)
        model_code = None
        classes = []
        try:
            model_meta = _load_json(model_dir / "model.json")
            classes = model_meta.get("classes") or []
            model_code = model_meta.get("model_code")
        except Exception:
            pass
        labels = [
            {"index": i, "english_code": c, "chinese_name": "", "chinese_desc": "", "ai_prompt_en": ""}
            for i, c in enumerate(classes)
        ]
        dic = {
            "model_code": model_code,
            "labels": labels,
        }
        _save_json(file, dic)
        return dic
    
    async def delete_model(self, model_id: str):
        """删除模型"""
        model_dir = self.registry_dir / model_id
        
        if not await asyncio.to_thread(lambda: model_dir.exists()):
            return None
        
        # 删除模型目录
        await asyncio.to_thread(_delete_directory, model_dir)
        # 影子库收口：真值目录已消失，PG 行必须一并删除（否则列表残留死记录）
        await shadow.drop_model(model_id)
        
        return {"ok": True, "message": f"Model {model_id} deleted"}
    
    def _attach_to_library(self, model_meta: dict, library_id: str) -> dict:
        """把模型记录挂到指定算法库下（与 update_model 的归库口径完全一致）。

        不挂库的记录没有 library_root_id，resolve_library_root_id 会把它当成"自成库根"，
        于是首页凭空多出一张算法库卡片——上传时必须显式归库或明确留在"独立模型"档。
        """
        lib_id = (library_id or "").strip()
        if not lib_id:
            return model_meta
        lib_file = self.registry_dir / lib_id / "model.json"
        if not lib_file.exists():
            raise ValueError(f"算法库不存在：{lib_id}")
        try:
            lib_meta = _load_json(lib_file)
        except Exception:
            raise ValueError(f"算法库记录读取失败：{lib_id}")

        mid = model_meta.get("model_id") or ""
        model_meta["model_library_id"] = lib_id
        model_meta["library_root_id"] = lib_id
        # model_code 全局唯一（update_model 会校验占用），版本记录必须保留自己的 code，
        # 绝不能抄库的：抄了就和库根重名，之后改任一条的 code 都会报「已被其他模型占用」。
        lib_biz = (lib_meta.get("business") or "").strip()
        if lib_biz and not (model_meta.get("business") or "").strip():
            model_meta["business"] = lib_biz
        model_meta["updated_at"] = datetime.now().isoformat()

        model_dir = self.registry_dir / mid
        _save_json(model_dir / "model.json", model_meta)
        # 标签字典跟着对齐 model_code（条目按 index/english_code 组织，内容不受影响）
        dict_file = model_dir / "labels_dict.json"
        if dict_file.exists() and model_meta.get("model_code"):
            try:
                dic = _load_json(dict_file)
                if (dic.get("model_code") or "") != model_meta["model_code"]:
                    dic["model_code"] = model_meta["model_code"]
                    _save_json(dict_file, dic)
            except Exception:
                pass
        return model_meta

    async def upload_model(self, file: UploadFile, display_name: str = "", library_id: str = ""):
        """上传已有模型（ZIP 包或单个 .pt 权重文件）

        display_name: 用户指定的模型名称；留空则回退为上传文件名去扩展名
        library_id: 归入的算法库根记录 id；留空则作为「独立模型」入库
        """
        # 验证文件格式
        if not (file.filename.endswith('.zip') or file.filename.endswith('.pt')):
            raise ValueError("Only .zip or .pt files are allowed")

        # 目标库先校验：权重落盘后再失败会留下半截记录
        lib_id = (library_id or "").strip()
        if lib_id and not (self.registry_dir / lib_id / "model.json").exists():
            raise ValueError(f"算法库不存在：{lib_id}")
        
        # 读取文件内容
        content = await file.read()
        filename = file.filename or "model.zip"
        
        # 验证文件大小（防止解压炸弹，限制10GB）
        if len(content) > 10 * 1024 * 1024 * 1024:
            raise ValueError("ZIP file too large (max 100MB)")
        
        def _process_zip(zip_filename: str):
            """处理ZIP文件的同步函数"""
            
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # 验证ZIP文件
                try:
                    zip_file = zipfile.ZipFile(io.BytesIO(content), 'r')
                    zip_file.testzip()  # 验证ZIP完整性
                except zipfile.BadZipFile:
                    raise ValueError("Invalid or corrupted ZIP file")
                
                # 解压ZIP文件（S-01：路径遍历 + 解压炸弹校验，收口 utils.zip_guard）
                try:
                    zip_guard.guarded_extractall(zip_file, temp_path, strict=True)
                except zip_guard.ZipBombRisk as e:
                    zip_file.close()
                    raise ValueError(str(e))
                except zip_guard.UnsafeZipMember as e:
                    zip_file.close()
                    raise ValueError(str(e))
                zip_file.close()
                
                # 查找model.json和weights目录
                model_json_path = None
                weights_dir_path = None
                
                # 查找model.json（可能在根目录或子目录中）
                for json_file in temp_path.rglob("model.json"):
                    model_json_path = json_file
                    break
                
                # 查找weights目录
                for weights_dir in temp_path.rglob("weights"):
                    if weights_dir.is_dir():
                        weights_dir_path = weights_dir
                        break
                
                # 验证必需的文件和目录
                if not weights_dir_path:
                    raise ValueError("ZIP file must contain a 'weights' directory")
                
                # 检查weights目录中是否有.pt文件
                pt_files = list(weights_dir_path.glob("*.pt"))
                if not pt_files:
                    raise ValueError("weights directory must contain at least one .pt file")
                
                # 读取或创建model.json
                if model_json_path and model_json_path.exists():
                    model_meta = _load_json(model_json_path)
                    # 如果ZIP中有model_id，检查是否已存在
                    original_model_id = model_meta.get("model_id")
                    if original_model_id:
                        # 检查model_id是否已存在
                        existing_model_dir = self.registry_dir / original_model_id
                        if existing_model_dir.exists():
                            # 生成新的model_id
                            model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        else:
                            model_id = original_model_id
                    else:
                        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                else:
                    # 创建新的model.json（字段规范与 create_empty_model 一致）
                    model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    model_meta = {
                        "model_id": model_id,
                        "name": zip_filename.replace('.zip', ''),
                        "created_at": datetime.now().isoformat(),
                        "source": "uploaded",
                        "empty": False,
                        "classes": [],
                        "description": f"Uploaded model from {zip_filename}",
                        "tags": ["uploaded"],
                    }
                
                # 创建目标目录
                model_dir = self.registry_dir / model_id
                target_weights_dir = model_dir / "weights"
                model_dir.mkdir(parents=True, exist_ok=True)
                target_weights_dir.mkdir(parents=True, exist_ok=True)
                
                # 复制权重文件
                total_size = 0
                for pt_file in pt_files:
                    target_pt = target_weights_dir / pt_file.name
                    shutil.copy2(pt_file, target_pt)
                    total_size += pt_file.stat().st_size
                
                # 更新model.json（setdefault 补齐统一规范字段，兼容 ZIP 自带 model.json）
                model_meta["model_id"] = model_id
                model_meta["weights_path"] = str((target_weights_dir / pt_files[0].name).resolve())
                model_meta["file_size"] = total_size
                model_meta["file_size_mb"] = round(total_size / (1024 * 1024), 2)
                model_meta.setdefault("model_code", model_id)
                model_meta.setdefault("display_name", "")
                model_meta.setdefault("business", "")
                model_meta.setdefault("status", "active")
                model_meta.setdefault("source", "uploaded")
                model_meta.setdefault("classes", [])
                model_meta.setdefault("labels_file", "labels_dict.json")
                model_meta.setdefault("empty", False)
                model_meta.setdefault("description", "")
                model_meta.setdefault("tags", [])
                # M11：architecture 标准字段集（上传模型无 base_model 可解析，置空占位）
                model_meta.setdefault("architecture_family", "")
                model_meta.setdefault("architecture_name", "")
                model_meta.setdefault("model_scale", "")
                model_meta["created_at"] = datetime.now().isoformat()
                model_meta["updated_at"] = datetime.now().isoformat()
                
                # 如果ZIP中有其他文件，也复制（除了model.json和weights）
                for item in temp_path.rglob("*"):
                    if item.is_file():
                        relative_path = item.relative_to(temp_path)
                        # 跳过model.json和weights目录中的文件（已处理）
                        if relative_path.name == "model.json" or "weights" in relative_path.parts:
                            continue
                        # 复制其他文件到model目录
                        target_file = model_dir / relative_path.name
                        if not target_file.exists():
                            shutil.copy2(item, target_file)
                
                # 保存model.json
                model_file = model_dir / "model.json"
                _save_json(model_file, model_meta)
                
                # 标签字典缺失时懒初始化（与 get_labels_dict 语义一致；在复制循环之后，避免挡住 ZIP 自带字典）
                labels_file = model_dir / "labels_dict.json"
                if not labels_file.exists():
                    _save_json(labels_file, {
                        "model_code": model_meta.get("model_code", model_id),
                        "labels": [
                            {"index": i, "english_code": c, "chinese_name": "", "chinese_desc": ""}
                            for i, c in enumerate(model_meta.get("classes") or [])
                        ],
                    })
                
                return model_meta
        
        def _process_pt(pt_filename: str):
            """处理单个 .pt 权重上传：统一落为 registry/{model_id}/weights/best.pt"""
            stem = Path(pt_filename).stem
            model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_dir = self.registry_dir / model_id
            target_weights_dir = model_dir / "weights"
            target_weights_dir.mkdir(parents=True, exist_ok=True)
            # 预标注引擎按约定路径找 weights/best.pt，因此无论原名一律改名
            target_pt = target_weights_dir / "best.pt"
            target_pt.write_bytes(content)
            total_size = target_pt.stat().st_size
            
            model_meta = {
                "model_id": model_id,
                "name": stem,
                "display_name": stem,
                "model_code": model_id,
                "business": "",
                "status": "active",
                "source": "uploaded",
                "empty": False,
                "classes": [],
                "labels_file": "labels_dict.json",
                "description": f"用户上传的权重文件 {pt_filename}",
                "tags": ["uploaded"],
                "weights_path": str(target_pt.resolve()),
                "file_size": total_size,
                "file_size_mb": round(total_size / (1024 * 1024), 2),
                "architecture_family": "",
                "architecture_name": "",
                "model_scale": "",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            _save_json(model_dir / "model.json", model_meta)
            return model_meta
        
        if filename.endswith('.pt'):
            model_meta = await asyncio.to_thread(_process_pt, filename)
        else:
            model_meta = await asyncio.to_thread(_process_zip, filename)
        
        # 用户显式命名时覆盖默认名（ZIP 自带 model.json 的 name 也以用户输入为准）
        custom_name = (display_name or "").strip()
        if custom_name:
            model_meta["name"] = custom_name
            model_meta["display_name"] = custom_name
            model_meta["updated_at"] = datetime.now().isoformat()
            # 下方 model_info 回填仅在解析成功时才回写 model.json，故此处必须自行落盘
            await asyncio.to_thread(
                _save_json,
                self.registry_dir / model_meta["model_id"] / "model.json",
                model_meta,
            )
        
        # 尝试获取模型信息
        weights_path = model_meta.get("weights_path")
        if weights_path:
            model_info = await self._get_model_info(weights_path)
            model_changed = False
            if model_info:
                model_meta["model_info"] = model_info
                model_changed = True
                # 用权重解析出的任务/类别回填缺失字段（单个 .pt 上传时尤其需要）
                if not model_meta.get("task") and model_info.get("task"):
                    model_meta["task"] = model_info["task"]
                if not model_meta.get("classes") and model_info.get("classes"):
                    model_meta["classes"] = model_info["classes"]
            
            model_dir = self.registry_dir / model_meta["model_id"]
            if model_changed:
                # 更新model.json
                await asyncio.to_thread(_save_json, model_dir / "model.json", model_meta)
            # 标签字典缺失时懒初始化（按最终 classes 生成）
            labels_file = model_dir / "labels_dict.json"
            if not labels_file.exists():
                await asyncio.to_thread(_save_json, labels_file, {
                    "model_code": model_meta.get("model_code", model_meta["model_id"]),
                    "labels": [
                        {"index": i, "english_code": c, "chinese_name": "", "chinese_desc": ""}
                        for i, c in enumerate(model_meta.get("classes") or [])
                    ],
                })
        
        # 归入指定算法库：放在最后，让 model_code / business 继承与标签字典对齐都基于最终字段
        if lib_id:
            model_meta = await asyncio.to_thread(self._attach_to_library, model_meta, lib_id)
        
        # 影子库收口：上传链路有多处 model.json / labels_dict.json 落盘，统一在最后回读真值写库
        if model_meta.get("model_id"):
            await shadow.sync_model(model_meta["model_id"])
        return model_meta
    
    @staticmethod
    def _export_weight_filename(meta: dict, model_id: str) -> str:
        """导出包内权重的可读文件名：优先 ASCII 安全的 display_name，其次 model_code，兜底 model_id，
        再拼版本号（display_name 已含版本后缀时不重复拼）。磁盘存储名保持 best.pt 不变。"""
        import re as _re
        dn = str(meta.get("display_name") or "").strip()
        code = str(meta.get("model_code") or "").strip()
        safe = _re.compile(r"^[A-Za-z0-9._-]+$")
        base = ""
        for cand in (dn, code):
            if cand and cand != model_id and safe.match(cand):
                base = cand
                break
        if not base:
            base = code if (code and safe.match(code)) else model_id
        ver = str(meta.get("version") or "").strip()
        if ver:
            ver_part = ver if ver.lower().startswith("v") else f"v{ver}"
            if ver_part.lower() not in base.lower() and ver_part.lstrip("vV").lower() not in base.lower():
                base = f"{base}_{ver_part}"
        return f"{base}.pt"

    async def export_model(self, model_id: str):
        """导出模型为ZIP文件"""
        model_dir = self.registry_dir / model_id
        model_file = model_dir / "model.json"
        
        if not await asyncio.to_thread(lambda: model_file.exists()):
            return None
        
        # 创建临时ZIP文件
        def _create_zip():
            # 创建临时文件
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            temp_zip_path = temp_zip.name
            temp_zip.close()
            
            meta = _load_json(model_file) if model_file.exists() else {}
            weight_name = self._export_weight_filename(meta or {}, model_id)
            
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 添加model.json
                if model_file.exists():
                    zipf.write(model_file, f"{model_id}/model.json")
                
                # 添加weights目录（best.pt 等统一重命名为可读名；导入端 glob("*.pt") 认任意名）
                weights_dir = model_dir / "weights"
                if weights_dir.exists():
                    for weight_file in weights_dir.iterdir():
                        if weight_file.is_file() and weight_file.suffix == ".pt":
                            zipf.write(weight_file, f"{model_id}/weights/{weight_name}")
                        elif weight_file.is_file():
                            zipf.write(weight_file, f"{model_id}/weights/{weight_file.name}")
                
                # 添加其他可能的文件
                for item in model_dir.iterdir():
                    if item.is_file() and item.name != "model.json":
                        zipf.write(item, f"{model_id}/{item.name}")
            
            return temp_zip_path
        
        zip_path = await asyncio.to_thread(_create_zip)
        return zip_path
    
    async def generate_training_charts(self, model_id: str, chart_type: str = "all"):
        """生成训练图表
        
        Args:
            model_id: 模型ID
            chart_type: 图表类型 - "loss", "metrics", "all"
        
        Returns:
            图表文件路径
        """
        model_dir = self.registry_dir / model_id
        model_file = model_dir / "model.json"
        
        if not await asyncio.to_thread(lambda: model_file.exists()):
            return None
        
        model_meta = await asyncio.to_thread(_load_json, model_file)
        job_id = model_meta.get("job_id")
        
        if not job_id:
            raise ValueError("Model has no associated training job")
        
        # 查找results.csv
        job_dir = self.jobs_dir / job_id
        
        def _find_results_csv():
            results_files = list(job_dir.rglob("results.csv"))
            if results_files:
                return results_files[0]
            return None
        
        results_csv = await asyncio.to_thread(_find_results_csv)
        
        if not results_csv:
            raise ValueError("Training results.csv not found")
        
        # 解析CSV数据
        training_history = await asyncio.to_thread(self._parse_results_csv, results_csv)
        
        if not training_history:
            raise ValueError("Failed to parse training results")
        
        # 生成图表
        def _generate_charts():
            # 设置中文字体支持（如果需要）
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            if chart_type == "loss" or chart_type == "all":
                # 生成损失曲线图
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                epochs = training_history.get("epochs", [])
                
                # Box Loss
                if "train_box_loss" in training_history:
                    axes[0].plot(epochs, training_history["train_box_loss"], label='Train Box Loss', marker='o')
                if "val_box_loss" in training_history:
                    axes[0].plot(epochs, training_history["val_box_loss"], label='Val Box Loss', marker='s')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Box Loss')
                axes[0].legend()
                axes[0].grid(True)
                
                # Class Loss
                if "train_cls_loss" in training_history:
                    axes[1].plot(epochs, training_history["train_cls_loss"], label='Train Cls Loss', marker='o')
                if "val_cls_loss" in training_history:
                    axes[1].plot(epochs, training_history["val_cls_loss"], label='Val Cls Loss', marker='s')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Loss')
                axes[1].set_title('Classification Loss')
                axes[1].legend()
                axes[1].grid(True)
                
                # DFL Loss
                if "train_dfl_loss" in training_history:
                    axes[2].plot(epochs, training_history["train_dfl_loss"], label='Train DFL Loss', marker='o')
                if "val_dfl_loss" in training_history:
                    axes[2].plot(epochs, training_history["val_dfl_loss"], label='Val DFL Loss', marker='s')
                axes[2].set_xlabel('Epoch')
                axes[2].set_ylabel('Loss')
                axes[2].set_title('DFL Loss')
                axes[2].legend()
                axes[2].grid(True)
                
                plt.tight_layout()
                
                loss_chart_path = tempfile.mktemp(suffix='_loss.png')
                plt.savefig(loss_chart_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            if chart_type == "metrics" or chart_type == "all":
                # 生成指标曲线图
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                epochs = training_history.get("epochs", [])
                
                # Precision
                if "metrics_precision" in training_history:
                    axes[0, 0].plot(epochs, training_history["metrics_precision"], label='Precision', marker='o', color='blue')
                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('Precision')
                    axes[0, 0].set_title('Precision')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True)
                
                # Recall
                if "metrics_recall" in training_history:
                    axes[0, 1].plot(epochs, training_history["metrics_recall"], label='Recall', marker='s', color='green')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('Recall')
                    axes[0, 1].set_title('Recall')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True)
                
                # mAP50
                if "metrics_mAP50" in training_history:
                    axes[1, 0].plot(epochs, training_history["metrics_mAP50"], label='mAP@0.5', marker='^', color='orange')
                    axes[1, 0].set_xlabel('Epoch')
                    axes[1, 0].set_ylabel('mAP@0.5')
                    axes[1, 0].set_title('mAP@0.5')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True)
                
                # mAP50-95
                if "metrics_mAP50_95" in training_history:
                    axes[1, 1].plot(epochs, training_history["metrics_mAP50_95"], label='mAP@0.5:0.95', marker='d', color='red')
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('mAP@0.5:0.95')
                    axes[1, 1].set_title('mAP@0.5:0.95')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True)
                
                plt.tight_layout()
                
                metrics_chart_path = tempfile.mktemp(suffix='_metrics.png')
                plt.savefig(metrics_chart_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            if chart_type == "all":
                return {"loss_chart": loss_chart_path, "metrics_chart": metrics_chart_path}
            elif chart_type == "loss":
                return {"loss_chart": loss_chart_path}
            elif chart_type == "metrics":
                return {"metrics_chart": metrics_chart_path}
        
        chart_paths = await asyncio.to_thread(_generate_charts)
        return chart_paths


    # ------------------------------------------------------------------
    # 检测模型自动升级（离线微调 → 热切换为预标注模型）
    # ------------------------------------------------------------------
    def promote_to_detector(self, model_id: str) -> dict:
        """将训练好的模型升级为 AI 预标注的检测模型。

        - 以 mAP50-95 为核心验收指标，Precision/Recall 为辅助指标，在相同验证集上对比
        - 仅当新模型 mAP50-95 显著优于当前检测模型时，才自动切换
        """
        model_dir = self.registry_dir / model_id
        model_file = model_dir / "model.json"
        if not model_file.exists():
            raise ValueError("模型不存在")
        model_meta = _load_json(model_file)

        weights_path = model_meta.get("weights_path")
        if not weights_path or not Path(weights_path).exists():
            raise ValueError("模型权重文件不存在")

        classes = model_meta.get("classes") or []

        # 当前检测模型配置
        sam_cfg = self._read_sam_config()
        old_weights = sam_cfg.get("detector_weights", "yolov8s-world.pt")

        # 定位该模型训练所用数据集的 data.yaml（用于在相同验证集上对比）
        data_yaml = self._find_model_data_yaml(model_meta)
        if not data_yaml:
            raise ValueError("无法定位该模型的训练数据集 data.yaml，无法进行对比验证")

        # 新模型指标（取自训练 final_metrics，与旧模型在同一验证集上对比）
        new_metrics = self._get_trained_final_metrics(model_meta)

        # 旧（当前）检测模型指标：在相同验证集上重新评估
        old_metrics = self._validate_detector(old_weights, data_yaml, classes)

        # 标注速度（推理耗时，毫秒/张）：越小越快。作为升级的参考标准之一
        new_speed = self._measure_speed(weights_path, data_yaml, classes)
        old_speed = self._measure_speed(old_weights, data_yaml, classes)

        # 比较并决定是否切换
        new_map = new_metrics.get("mAP50_95")
        old_map = old_metrics.get("mAP50_95")
        switched = False
        if new_map is not None and old_map is not None:
            # 核心：mAP50-95 显著更优
            if new_map > old_map + 0.01:
                switched = True
            # 参考：mAP50-95 差异不大时，标注速度更快（推理耗时更短）则切换
            elif abs(new_map - old_map) <= 0.01 and new_speed is not None and old_speed is not None and new_speed < old_speed:
                switched = True
        else:
            switched = False

        if switched:
            sam_cfg["detector_weights"] = weights_path
            self._save_sam_config(sam_cfg)

        return {
            "switched": switched,
            "new_model": {"weights_path": weights_path, "speed_ms": new_speed, **new_metrics},
            "old_model": {"detector_weights": old_weights, "speed_ms": old_speed, **old_metrics},
        }

    def _read_sam_config(self) -> dict:
        cfg = {"detector_weights": "yolov8s-world.pt", "conf": 0.15}
        p = settings.SAM_CONFIG_FILE
        if p.exists():
            try:
                cfg.update(_load_json(p))
            except Exception:
                pass
        return cfg

    def _save_sam_config(self, cfg: dict) -> None:
        p = settings.SAM_CONFIG_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    def _find_model_data_yaml(self, model_meta: dict):
        """通过 job_id → dataset_id 定位数据集中的 data.yaml"""
        job_id = model_meta.get("job_id")
        if not job_id:
            return None
        job_file = self.jobs_dir / f"{job_id}.json"
        if not job_file.exists():
            return None
        job_meta = _load_json(job_file)
        dataset_id = job_meta.get("dataset_id")
        if not dataset_id:
            return None
        dataset_dir = settings.DATA_DIR / "datasets" / dataset_id
        if not dataset_dir.exists():
            return None
        for yaml_file in dataset_dir.rglob("data.yaml"):
            return yaml_file
        return None

    def _get_trained_final_metrics(self, model_meta: dict) -> dict:
        """读取训练结果的 final_metrics（mAP50-95、precision、recall）"""
        job_id = model_meta.get("job_id")
        if job_id:
            try:
                metrics = self._load_training_metrics(job_id)
                if metrics and metrics.get("final_metrics"):
                    return metrics["final_metrics"]
            except Exception:
                pass
        return {}

    def _validate_detector(self, weights: str, data_yaml: Path, classes: list) -> dict:
        """在给定验证集上评估检测模型，返回 mAP50 / mAP50-95 / precision / recall"""
        from ultralytics import YOLO
        try:
            weights_path = self._resolve_detector_weights(weights)
            model = YOLO(weights_path)
            # YOLO-World 等文本驱动模型需要先 set_classes 才能按数据集类别评估
            try:
                if hasattr(model.model, "set_classes") and classes:
                    model.set_classes(classes)
            except Exception:
                pass
            res = model.val(data=str(data_yaml), split="val", verbose=False)
            m = res.box
            speed_ms = None
            speed = getattr(res, "speed", None) or {}
            if isinstance(speed, dict):
                inf = speed.get("inference")
                if inf is not None:
                    speed_ms = round(float(inf), 2)
            return {
                "mAP50": None if m.map50 is None else round(float(m.map50), 4),
                "mAP50_95": None if m.map is None else round(float(m.map), 4),
                "precision": None if m.mp is None else round(float(m.mp), 4),
                "recall": None if m.mr is None else round(float(m.mr), 4),
                "speed_ms": speed_ms,
            }
        except Exception as e:
            return {"mAP50": None, "mAP50_95": None, "precision": None, "recall": None, "speed_ms": None, "error": str(e)}

    def _measure_speed(self, weights: str, data_yaml: Path, classes: list) -> float | None:
        """测量模型在验证集上的平均推理耗时（毫秒/张），作为标注速度参考"""
        from ultralytics import YOLO
        try:
            weights_path = self._resolve_detector_weights(weights)
            model = YOLO(weights_path)
            try:
                if hasattr(model.model, "set_classes") and classes:
                    model.set_classes(classes)
            except Exception:
                pass
            res = model.val(data=str(data_yaml), split="val", verbose=False)
            speed = getattr(res, "speed", None) or {}
            if isinstance(speed, dict):
                inf = speed.get("inference")
                if inf is not None:
                    return round(float(inf), 2)
            return None
        except Exception:
            return None

    def _resolve_detector_weights(self, weights: str) -> str:
        p = Path(weights)
        if p.is_absolute() and p.exists():
            return str(p)
        for d in [settings.SAM_MODELS_DIR, settings.BASE_DIR]:
            cand = d / weights
            if cand.exists():
                return str(cand)
        return weights

    # ------------------------------------------------------------------
    # 模型守门员：生产模型查询 / 人工强制覆盖
    # ------------------------------------------------------------------
    async def list_production_models(self, business: str = None):
        """列出当前在役的生产模型（status=production_ready）。

        Args:
            business: 业务/算法类型；缺省返回所有业务的生产模型
        """
        def _list_sync():
            models = []
            if not self.registry_dir.exists():
                return models
            for model_dir in self.registry_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model_file = model_dir / "model.json"
                if not model_file.exists():
                    continue
                try:
                    model_meta = _load_json(model_file)
                except Exception:
                    continue
                if model_meta.get("status") != "production_ready":
                    continue
                if business and model_meta.get("business") != business:
                    continue
                weights_path = model_meta.get("weights_path")
                if weights_path and Path(weights_path).exists():
                    model_meta["file_size_mb"] = round(os.path.getsize(weights_path) / (1024 * 1024), 2)
                models.append(model_meta)
            return models

        models = await asyncio.to_thread(_list_sync)
        return {"models": sorted(models, key=lambda x: self._parse_version(x.get("version")), reverse=True)}

    @staticmethod
    def _parse_version(ver) -> float:
        """解析 'v1.0' 形式版本号为数值，无法解析返回 0（保证排序不报错）"""
        if not ver or not str(ver).startswith("v"):
            return 0.0
        try:
            return float(str(ver)[1:])
        except (TypeError, ValueError):
            return 0.0

    async def override_model(self, model_id: str, business: str = None, reason: str = None):
        """人工强制覆盖（Override）：高级工程师手动将被拦截的模型设为生产版本。

        通常用于守门员判定 rejected（淘汰）后，且同业务已有生产模型在役时的强制晋升；
        同业务存在更高/更早生产模型时，也允许将任意版本强制设为当前服役版本。
        """
        model_dir = self.registry_dir / model_id
        model_file = model_dir / "model.json"
        if not await asyncio.to_thread(lambda: model_file.exists()):
            raise ValueError(f"模型 {model_id} 不存在")

        def _override_sync():
            meta = _load_json(model_file)
            weights_path = meta.get("weights_path")
            if not weights_path or not Path(weights_path).exists():
                raise ValueError("模型权重文件不存在，无法强制设为生产")

            # 记录覆盖操作（保留原状态便于追溯）
            override_record = {
                "from_status": meta.get("status", "unknown"),
                "to_status": "production_ready",
                "operated_at": datetime.now().isoformat(),
                "reason": reason or "人工强制覆盖（高级工程师操作）",
            }
            if business:
                meta["business"] = business

            # 低版本强制覆盖高版本的情况：授予与旧生产版本一致的版本号，避免版本回退
            current = self._parse_version(meta.get("version"))
            superseded = []
            if business:
                for other in self.registry_dir.iterdir():
                    if not other.is_dir():
                        continue
                    of = other / "model.json"
                    if not of.exists():
                        continue
                    try:
                        om = _load_json(of)
                    except Exception:
                        continue
                    if om.get("model_id") == model_id or om.get("status") != "production_ready":
                        continue
                    if om.get("business") != business:
                        continue
                    ov = self._parse_version(om.get("version"))
                    if ov > current:
                        superseded.append({"model_id": om.get("model_id"), "version": om.get("version")})

            meta["status"] = "production_ready"
            meta["override"] = override_record
            superseded_records = []
            for s in superseded:
                sf = self.registry_dir / s["model_id"] / "model.json"
                try:
                    sm = _load_json(sf)
                    sm["status"] = "superseded"
                    _save_json(sf, sm)
                    superseded_records.append(s["model_id"])
                except Exception:
                    pass
            _save_json(model_file, meta)
            return {"meta": meta, "superseded": superseded_records}

        result = await asyncio.to_thread(_override_sync)
        # 影子库收口：新老模型的 status 同时变化，两个都要重灌
        meta = result.get("meta") or {}
        if meta.get("model_id"):
            await shadow.sync_model(meta["model_id"])
        for mid in result.get("superseded") or []:
            await shadow.sync_model(mid)
        return result

    # ------------------------------------------------------------------
    # P8 半自动发布：守门员通过 → production_ready(待确认) → 人工确认发布
    # ------------------------------------------------------------------
    async def list_pending_publish(self, business: str = None):
        """列出待确认发布的模型：status=production_ready 且未确认发布（publish_status != published）。"""
        def _list_sync():
            models = []
            if not self.registry_dir.exists():
                return models
            for model_dir in self.registry_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model_file = model_dir / "model.json"
                if not model_file.exists():
                    continue
                try:
                    model_meta = _load_json(model_file)
                except Exception:
                    continue
                if model_meta.get("status") != "production_ready":
                    continue
                if (model_meta.get("publish_status") or "pending") == "published":
                    continue
                if business and model_meta.get("business") != business:
                    continue
                models.append(model_meta)
            return models

        models = await asyncio.to_thread(_list_sync)
        return {"models": sorted(models, key=lambda x: self._parse_version(x.get("version")), reverse=True)}

    async def confirm_publish(self, model_id: str):
        """确认发布（P8 半自动）：管理员点击后 → publish_status=published + published_at，
        并将同业务其他已发布生产模型标 superseded，更新 model_libraries.current_production_model_id。
        可选 24h 冷却期（副本内已由 ready_at 判定，这里只做状态转变，冷却判定由路由层做）。
        """
        model_dir = self.registry_dir / model_id
        model_file = model_dir / "model.json"
        if not await asyncio.to_thread(lambda: model_file.exists()):
            raise ValueError(f"模型 {model_id} 不存在")

        def _confirm_sync():
            meta = _load_json(model_file)
            if meta.get("status") != "production_ready":
                raise ValueError(f"模型 {model_id} 当前状态 {meta.get('status')}，不可确认发布（需先通过守门员）")
            meta["publish_status"] = "published"
            meta["published_at"] = datetime.now().isoformat()

            business = meta.get("business") or ""
            # 生产版本是"库内"概念：模型若归属某算法库，只在本库的版本之间轮换，
            # 否则同业务线的另一个库发布时会把本库的生产模型误标为已被取代。
            root_id = meta.get("library_root_id")
            superseded = []
            if root_id or business:
                for other in self.registry_dir.iterdir():
                    if not other.is_dir():
                        continue
                    of = other / "model.json"
                    if not of.exists():
                        continue
                    try:
                        om = _load_json(of)
                    except Exception:
                        continue
                    if om.get("model_id") == model_id or om.get("status") != "production_ready":
                        continue
                    if root_id:
                        if (om.get("library_root_id") or om.get("model_id")) != root_id:
                            continue
                    elif om.get("business") != business:
                        continue
                    if (om.get("publish_status") or "pending") != "published":
                        continue
                    om["status"] = "superseded"
                    _save_json(of, om)
                    superseded.append(om.get("model_id"))
            _save_json(model_file, meta)
            return {"meta": meta, "superseded": superseded}

        result = await asyncio.to_thread(_confirm_sync)
        # 影子库收口：确认发布改了目标模型与同库被顶替模型的 status
        _rid = (result.get("meta") or {}).get("model_id")
        if _rid:
            await shadow.sync_model(_rid)
        for mid in result.get("superseded") or []:
            await shadow.sync_model(mid)

        # 更新 model_libraries.current_production_model_id 指向唯一生产模型
        try:
            if settings.PG_ENABLED:
                from src.db import meta_store
                libs = await meta_store.list_model_libraries()
                biz = result["meta"].get("business") or ""
                # 优先命中模型自己已归入的库（管理页手动归库写入的 model_library_id），
                # 再退回按业务线约定的 {business}_lib 兜底
                own_lib = result["meta"].get("model_library_id") or ""
                target_lib = next(
                    (l for l in libs if own_lib and l["model_library_id"] == own_lib),
                    next((l for l in libs if biz and l["model_library_id"] == f"{biz}_lib"), None),
                )
                if target_lib is not None:
                    await meta_store.upsert_model_library({
                        **target_lib,
                        "current_production_model_id": model_id,
                        "updated_at": datetime.now().isoformat(),
                    })
        except Exception as e:  # noqa: BLE001
            logger.warning(f"confirm_publish: update model_libraries failed: {e}")
        return result

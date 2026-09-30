"""SAM 大模型智能预标注服务

基于 ultralytics 内置 SAM 与 YOLO-World 实现文本驱动的自动预标注：
  1. YOLO-World 按类别名检测 → 得到候选框
  2. SAM 对每个框做分割 → 计算最小外接矩形（精修）
  3. 返回与现有标注体系完全兼容的 BBox

参考设计文档 4.2 节，独立从零实现（不复制 AGPL 项目代码）。
"""
import json
import asyncio
import uuid
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.core.settings import settings
from src.services.device_manager import get_best_device
from src.services.annotation_service import AnnotationService


DEFAULT_CONFIG = {
    "detector": "yolo_world",        # yolo_world | grounding_dino | none
    "detector_weights": "yolov8s-world.pt",
    "grounding_dino_model": "IDEA-Research/grounding-dino-tiny",  # Transformers 集成的 GD 模型名
    "detectors": [],             # 多引擎同时检测：["grounding_dino","yolo_world","qwen_vl"]，空=沿用下方单 engine
    # 「本库已训模型」预标注引擎：用当前算法库自己训出来的 best.pt 出框。
    # 零样本引擎（GroundingDINO / YOLO-World）靠英文提示词猜，分不清"积水/水渍反光"；
    # 本库模型是按人工标注学出来的，画得准，所以留了这条口子形成
    # "人标一小批 → 训第一版 → 模型预标其余 → 人只删错补漏"的闭环。
    "trained_model_id": "",      # registry 里的模型版本 id，空=未选（勾选引擎但没选版本会直接报错提示）
    "grounding_dino_size": 640,      # GroundingDINO 推理分辨率（越小越快，默认 800）
    "grounding_dino_min_conf": 0.25, # GD 置信度下限（GD 分数尺度与 YOLO 不同，过低会大量误检）
    "sam_enabled": True,             # 是否启用 SAM 分割精修（单图 AI 标注）
    "batch_sam_enabled": False,      # 批量预标注是否启用 SAM 精修（默认关：速度快一倍以上，框贴合略降；可改 true 恢复）
    "sam_weights": "sam_b.pt",
    "imgsz": 640,                    # 检测器输入尺寸
    "sam_imgsz": 1024,               # SAM 输入尺寸
    # 分割轮廓（多边形）输出：SAM 精修时除了收紧矩形，还把物体轮廓抽成顶点序列一起返回，
    # 供"分割标注"使用；关掉则退回纯矩形行为（老流程零影响）
    "sam_return_polygon": True,      # 是否输出 polygon 顶点（false=只给矩形框）
    # 轮廓简化容差（RDP），单位＝**原图像素**，与轮廓长短无关：多边形的任一条边离掩码
    # 边界都不超过该像素数。实测（1316×712 积水图，多边形 vs 掩码逐像素比对）：
    #   4.0px → 20 顶点、漏围 0.8%   1.0px → 87 顶点、漏围 0.2%
    #   0.5px → 147 顶点、漏围 0%、IoU 0.9997   0.25px 与 0.5px 完全相同（已到像素极限）
    # 所以默认 0.5：描出来的多边形和平台外 plot() 贴的掩码重叠，再小只是白增顶点。
    # 想要顶点少、可接受削角时调大（如 2.0）。
    "sam_polygon_eps_px": 0.5,       # 轮廓与掩码边界的最大允许偏差（原图像素）
    "sam_polygon_max_points": 3000,  # 单个多边形最多保留多少个顶点（防止 JSON 过大）
    "conf": 0.10,                    # 默认置信度阈值（调低以提升召回）
    "iou": 0.40,                     # NMS IoU 阈值（调低以合并遮挡/重叠产生的重复框）
    # 检出框合并（#4 重叠/紧邻框保留最紧框）：
    # 同类合并：互相紧邻/重叠的同类框并为一个（person 中心点过近会误并，调低 ioU/dist 可规避）
    "merge_iou": 0.45,               # 同类合并 IoU 阈值（两个同类框交并比 ≥ 该值视为重复）
    "merge_dist": 0.35,              # 同类合并中心距阈值（≤ 该值×对角距离 视为同一物体分裂）
    # 高度重合保留最紧框：IoU 很高 + 面积接近 + 中心错位小 → 保留最小（最紧）框
    "merge_tight_iou": 0.7,          # 最紧框判据：IoU 下限（高度重合）
    "merge_tight_center": 0.15,      # 最紧框判据：中心距上限（×对角）
    "merge_tight_area_ratio": 1.6,   # 最紧框判据：面积比上限（两框大小接近）
    "half": False,                   # 半精度，省显存
    "device": "auto",                # auto | cpu | GPU 索引字符串
    # ------------------------------------------------------------------
    # 千问 VL 大模型预标注（本地预留接口：qwen_enabled=False 时不启用，
    # 部署到服务器/接入百炼 API 后改为 True 并填写 endpoint/key 即可启用）
    # ------------------------------------------------------------------
    "qwen_enabled": False,           # 是否启用千问 VL 作为检测引擎（yolo_world | grounding_dino | qwen_vl | none）
    "qwen_backend": "ollama",        # ollama（本地大模型）| dashscope（阿里云百炼 OpenAI 兼容接口）
    "qwen_endpoint": "http://localhost:11434",  # ollama: http://localhost:11434；dashscope: https://dashscope.aliyuncs.com/compatible-mode/v1
    "qwen_model": "qwen3-vl:4b-instruct",  # ollama 模型名 或 百炼模型名（qwen-vl-max 等）；服务器部署用环境变量 OLLAMA_QWEN_MODEL 覆盖
    "qwen_api_key": "",              # dashscope 需要 API Key；ollama 不需要
    "qwen_timeout": 90,              # 单图推理超时（秒），VLM 速度慢需给足余量
    "qwen_mock": False,              # 本地联调：True 时返回模拟框，不真实调用大模型（仅用于验证链路）
    # Qwen3-VL 服务器部署配套（仅 ollama 后端生效，随每次请求传递）：
    "qwen_keep_alive": "5m",         # Ollama keep_alive：模型驻留内存时长（"5m"/"30m"/"-1"=常驻），避免频繁换载
    "qwen_num_parallel": 1,          # OLLAMA_NUM_PARALLEL：同一模型并行处理请求数（GPU 显存充裕可调大）
}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class BatchTask:
    """内存中的批量预标注任务状态"""

    def __init__(self, batch_id: str, task_id: str, classes: List[str], conf: float):
        self.batch_id = batch_id
        self.task_id = task_id
        self.classes = classes
        self.prompts: Optional[List[str]] = None
        self.conf = conf
        self.total = 0
        self.done = 0
        self.boxes_written = 0
        self.current_image: Optional[str] = None
        self.annotated_images: List[str] = []  # 本次批量预标注实际写入过框的图片ID
        self.retried = 0                       # 检空后通过降阈/换引擎成功补标上的图片数
        self.skipped_candidates: List[str] = []  # 两次兜底仍未检出的图片ID（真正需要人工复核的候选）
        self.cancelled = False
        self.status = "running"  # running | done | cancelled | error
        self.error: Optional[str] = None
        self.result_summary: Optional[dict] = None
        # L2 千问审核汇总（只统计不落盘，前端批量进度展示）
        self.audit_checked = 0    # 已自动审核的图片数
        self.audit_boxes = 0      # 已审核的 AI 新增框数
        self.audit_errors = 0     # 审核失败/无输出次数
        self.audit_summary = {"keep": 0, "modify": 0, "reject": 0, "split": 0, "missed": 0}


class SAMService:
    def __init__(self):
        self._detector = None
        self._detector_name = None
        self._sam = None
        self._sam_name = None
        # 点选抠图的交互式预测器缓存：同一张图只做一次图像编码（占 90% 耗时），
        # 后续补点/点掉只走提示解码（实测 CPU：首次 4-6s → 微调 0.15-0.25s）
        self._sam_pred = None
        self._sam_pred_img = None      # 预测器当前已编码的图（路径）
        self._sam_pred_key = None      # 权重|imgsz|device 指纹，配置变了就重建
        # GroundingDINO（Transformers 集成）缓存
        self._gd_processor = None
        self._gd_model = None
        self._gd_name = None
        # 缓存最后一次传给 YOLO-World 的提示词列表，避免多类别时每张图重复重新编码
        self._detector_prompts: Optional[List[str]] = None
        # 「本库已训模型」引擎：按 registry 里的 best.pt 缓存一份 YOLO 实例
        self._trained = None                 # ultralytics.YOLO 实例
        self._trained_id = None              # 当前缓存对应的模型版本 id（换版本时自动重载）
        self._trained_map_key = None         # 类别映射缓存的指纹（类别集合变了就重算）
        self._trained_map: Optional[dict] = None
        self._lock = asyncio.Lock()
        self.batch_tasks: Dict[str, BatchTask] = {}
        self.annotation_service = AnnotationService()

    # ------------------------------------------------------------------
    # 配置读写
    # ------------------------------------------------------------------
    def _config_path(self) -> Path:
        return settings.SAM_CONFIG_FILE

    def _read_config(self) -> dict:
        cfg = dict(DEFAULT_CONFIG)
        overrides = {}
        p = self._config_path()
        if p.exists():
            try:
                saved = _load_json(p)
                cfg.update(saved)
                overrides = saved
            except Exception:
                pass
        # Docker 部署（Ollama）：backend 容器内通过服务名访问 ollama 服务；
        # 仅当用户未在 UI 显式配置过 endpoint 时注入，本地联调默认值不受影响
        env_endpoint = os.environ.get("OLLAMA_HOST", "").strip()
        if env_endpoint and "qwen_endpoint" not in overrides:
            cfg["qwen_endpoint"] = env_endpoint.rstrip("/")
        # Qwen3-VL 部署配套：OLLAMA_QWEN_MODEL 环境变量覆盖模型名（未在 UI 显式配置时）
        env_qwen_model = os.environ.get("OLLAMA_QWEN_MODEL", "").strip()
        if env_qwen_model and "qwen_model" not in overrides:
            cfg["qwen_model"] = env_qwen_model
        return cfg

    def _save_config(self, cfg: dict) -> None:
        p = self._config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    def get_config(self) -> dict:
        return self._read_config()

    def _task_labels_desc(self, task_id: str) -> Optional[dict]:
        """回溯 标注任务→数据集→归属模型 的 labels_dict.json，
        取出标签字典（index/english_code/chinese_name/chinese_desc/ai_prompt_en）：
        - chinese_name + chinese_desc → 千问 VL 提示词，提升大模型对业务类别的识别精度；
        - ai_prompt_en → GroundingDINO / YOLO-World 的英文检测提示词（见 _prompts_with_dict）。
        任一层缺失均返回 None（不阻断标注，回退到纯类别名提示词）。
        """
        try:
            task_file = settings.ANNOTATIONS_DIR / task_id / "task.json"
            if not task_file.exists():
                print(f"qwen: [DEBUG] task.json 不存在: {task_file}")
                return None
            task_meta = _load_json(task_file)
            dataset_id = task_meta.get("dataset_id")
            if not dataset_id:
                print(f"qwen: [DEBUG] task.json 没有 dataset_id: {task_meta}")
                return None
            # 数据集一律为 datasets/{dataset_id}/meta.json（无 version 子目录）
            ds_meta_file = settings.DATASETS_DIR / dataset_id / "meta.json"
            if not ds_meta_file.exists():
                print(f"qwen: [DEBUG] meta.json 不存在: {ds_meta_file}")
                return None
            ds_meta = _load_json(ds_meta_file)
            model_id = ds_meta.get("model_id")
            if not model_id:
                print(f"qwen: [DEBUG] meta.json 没有 model_id: {ds_meta}")
                return None
            dict_file = settings.REGISTRY_DIR / model_id / "labels_dict.json"
            if not dict_file.exists():
                print(f"qwen: [DEBUG] labels_dict.json 不存在: {dict_file}")
                return None
            full_dict = _load_json(dict_file)
            labels = (full_dict or {}).get("labels") or []
            print(f"qwen: [DEBUG] task={task_id} → dataset={dataset_id} → model={model_id} → labels_count={len(labels)} → chinese_names={[l.get('chinese_name') for l in labels]}")
            return {"model_id": model_id, "labels": labels}
        except Exception as e:
            print(f"qwen: 读取标签字典失败(不影响标注): {e}")
            import traceback; traceback.print_exc()
            return None

    def update_config(self, request) -> dict:
        cfg = self._read_config()
        updates = request.model_dump(exclude_unset=True) if hasattr(request, "model_dump") else {}
        for k, v in updates.items():
            if v is not None:
                cfg[k] = v
        self._save_config(cfg)
        # 清空检测器单例缓存：修改引擎/权重后无需重启，下次推理自动按新配置重新加载
        self._detector = None
        self._detector_name = None
        self._detector_prompts = None
        self._gd_model = None
        self._gd_processor = None
        self._gd_name = None
        self._trained = None
        self._trained_id = None
        self._trained_map = None
        self._trained_map_key = None
        return cfg

    # ------------------------------------------------------------------
    # 多引擎解析
    # ------------------------------------------------------------------
    def _active_detectors(self, cfg: dict) -> list:
        """返回本次要运行的检测引擎列表（支持多引擎同时检测，结果合并、自动去重保序）。
        detectors 列表非空时优先；否则回退到旧的单引擎字段 detector。
        """
        multi = cfg.get("detectors") or []
        if multi:
            dets = []
            for d in multi:
                if d not in dets:
                    dets.append(d)
            return dets
        det = cfg.get("detector", "yolo_world")
        return [det] if det and det != "none" else []

    # ------------------------------------------------------------------
    # 检测模型管理
    # ------------------------------------------------------------------
    def _model_locations(self) -> List[Path]:
        """检测模型可能存放的位置：统一目录 + 用户上传的自定义模型目录 + 项目 backend 根目录（兼容旧文件）"""
        return [settings.SAM_MODELS_DIR, settings.CUSTOM_MODELS_DIR, settings.BASE_DIR]

    def _find_model_file(self, weights: str) -> str:
        """将配置里的权重名解析为实际文件路径；找不到则原样返回交给 ultralytics 处理"""
        p = Path(weights)
        if p.is_absolute() and p.exists():
            return str(p)
        for d in self._model_locations():
            cand = d / weights
            if cand.exists():
                return str(cand)
        return weights

    def list_models(self) -> List[dict]:
        """列出可用的检测模型（.pt 文件）"""
        found = {}
        for d in self._model_locations():
            if not d.exists():
                continue
            for f in d.glob("*.pt"):
                found.setdefault(f.name, {
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / 1e6, 1),
                    "path": str(f),
                })
        return sorted(found.values(), key=lambda x: x["name"])

    def save_model(self, filename: str, data: bytes) -> Path:
        """保存上传的检测模型到统一目录"""
        settings.SAM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = settings.SAM_MODELS_DIR / filename
        dest.write_bytes(data)
        return dest

    # ------------------------------------------------------------------
    # 设备解析
    # ------------------------------------------------------------------
    def _resolve_device(self, cfg: dict) -> str:
        device = cfg.get("device", "auto")
        if device == "auto":
            return get_best_device()
        return device

    # ------------------------------------------------------------------
    # 模型加载（单例缓存）
    # ------------------------------------------------------------------
    def _load_detector(self, cfg: dict, det_type: str = None):
        det_type = det_type or cfg.get("detector", "yolo_world")
        if det_type == "none":
            return None
        if det_type == "qwen_vl":
            # 千问 VL 为远程/本地大模型服务，无需加载本地检测权重；返回就绪标记
            return True
        if det_type == "grounding_dino":
            # GroundingDINO 走 Transformers 集成，加载后返回占位标记表示已就绪
            self._load_grounding_dino(cfg)
            return True
        if det_type == "trained_model":
            # 本库已训模型：权重在 registry/{model_id}/weights/best.pt，不是模型目录里的通用权重
            return self._load_trained(cfg)
        # 默认 yolo_world
        if self._detector is not None:
            return self._detector
        weights = cfg.get("detector_weights", "yolov8s-world.pt")
        from ultralytics import YOLO
        self._detector = YOLO(self._find_model_file(weights))
        self._detector_name = weights
        return self._detector

    def _load_engine(self, det: str, cfg: dict):
        """按引擎名加载对应检测模型（多引擎预加载/可用性检查用）；单例缓存，unused 引擎不加载"""
        if det == "grounding_dino":
            self._load_grounding_dino(cfg)
            return True
        if det == "qwen_vl":
            # 千问 VL 为远程/本地大模型服务，无需加载本地权重
            return True
        if det == "trained_model":
            return self._load_trained(cfg)
        return self._load_detector(cfg, det_type=det if det != "yolo_world" else None)

    # ------------------------------------------------------------------
    # 「本库已训模型」引擎
    # ------------------------------------------------------------------
    def _trained_dir(self, model_id: str) -> Path:
        return settings.REGISTRY_DIR / (model_id or "")

    def _trained_weights(self, model_id: str) -> Optional[Path]:
        """定位某个模型版本落盘的权重：优先约定路径 registry/{id}/weights/best.pt，
        其次回退 model.json 里记录的 weights_path（历史记录存放位置可能不同）。"""
        if not model_id:
            return None
        d = self._trained_dir(model_id)
        cand = d / "weights" / "best.pt"
        if cand.exists():
            return cand
        try:
            meta = _load_json(d / "model.json")
        except Exception:
            return None
        p = str(meta.get("weights_path") or "").strip()
        if p and Path(p).exists():
            return Path(p)
        return None

    def _trained_meta(self, model_id: str) -> dict:
        try:
            return _load_json(self._trained_dir(model_id) / "model.json")
        except Exception:
            return {}

    def _load_trained(self, cfg: dict):
        """加载本库已训模型的权重（单例缓存，换版本自动重载）。未选版本/权重缺失返回 None。"""
        model_id = str(cfg.get("trained_model_id") or "").strip()
        if not model_id:
            print("trained_model: 尚未选择本库已训模型版本，预标注跳过")
            return None
        weights = self._trained_weights(model_id)
        if weights is None:
            print(f"trained_model: 模型版本 {model_id} 找不到权重文件（best.pt），预标注跳过")
            return None
        if self._trained is not None and self._trained_id == model_id:
            return self._trained
        from ultralytics import YOLO
        self._trained = YOLO(str(weights))
        self._trained_id = model_id
        self._trained_map = None
        self._trained_map_key = None
        print(f"trained_model: 已加载 {model_id} → {weights.name}")
        return self._trained

    def _trained_names(self, model) -> List[str]:
        """模型自带的类别名列表（按下标顺序，与推理结果 cls 对齐）。"""
        names = getattr(model, "names", None) or {}
        if isinstance(names, dict):
            try:
                return [str(names[i]) for i in sorted(names)]
            except Exception:
                return [str(v) for v in names.values()]
        return [str(n) for n in names]

    def _trained_class_map(self, model, classes: List[str], cfg: dict) -> dict:
        """模型类别下标 → 本任务类别下标 的映射。

        训练产出的类别名与标注任务的英文标签不一定逐字相同（大小写、中文名的情况都有），
        所以这里用「英文标签码 → 中文名 → 中文描述 → ai_prompt_en」四条别名依次兜底；
        对不上的类别直接丢弃，避免把"安全帽"这类无关模型的框硬塞进"积水"任务里。
        """
        key = (self._trained_id, tuple(classes or ()))
        if self._trained_map is not None and self._trained_map_key == key:
            return self._trained_map
        by_code = {}
        for i, c in enumerate(classes or []):
            by_code[str(c).strip().lower()] = i
        # 标签字典允许同一个任务里英文码与模型训练时的叫法不同：用中文名再兜一层
        by_alias = {}
        for lab in self._dict_labels(cfg) or []:
            idx = lab.get("index")
            if not isinstance(idx, int) or idx < 0 or idx >= len(classes or []):
                continue
            # ai_prompt_en 排最后兜底：它是"给大模型/SAM 用的英文提示词"，
            # 常与训练产出的类名一致（如字典 english_code=diedao 但模型 classes=fall），
            # 优先级必须低于英文码与中文名，避免反过来抢走正确对齐。
            for alias in (lab.get("english_code"), lab.get("chinese_name"), lab.get("chinese_desc"), lab.get("ai_prompt_en")):
                a = str(alias or "").strip().lower()
                if a:
                    by_alias.setdefault(a, idx)
        mapping = {}
        for mid, name in enumerate(self._trained_names(model)):
            n = name.strip().lower()
            idx = by_code.get(n)
            if idx is None:
                idx = by_alias.get(n)
            if idx is None:
                # 名字里带序号前缀（如 "0 helmet"）时再试一次去前缀匹配
                tail = n.split(" ", 1)[-1].strip() if " " in n else ""
                idx = by_code.get(tail) or by_alias.get(tail)
            if idx is not None:
                mapping[mid] = idx
        self._trained_map = mapping
        self._trained_map_key = key
        print(
            f"trained_model: 类别对齐 {self._trained_names(model)} → {classes}；"
            f"可用映射 {mapping or '无（本次不会产出预标注框，请检查标签英文名）'}"
        )
        return mapping

    def _library_root_of(self, model_id: str) -> str:
        """任意模型版本 id → 所属算法库根 id（历史记录没写该字段时认为它自己就是库根）。"""
        if not model_id:
            return ""
        meta = self._trained_meta(model_id)
        return str(meta.get("library_root_id") or model_id)

    def _task_dataset_model(self, task_id: str) -> tuple:
        """标注任务 → (数据集 id, 数据集归属的模型/算法库 id)。"""
        try:
            task_file = settings.ANNOTATIONS_DIR / task_id / "task.json"
            if not task_file.exists():
                return "", ""
            dataset_id = _load_json(task_file).get("dataset_id") or ""
            ds_file = settings.DATASETS_DIR / dataset_id / "meta.json"
            if not dataset_id or not ds_file.exists():
                return dataset_id, ""
            return dataset_id, _load_json(ds_file).get("model_id") or ""
        except Exception as e:
            print(f"trained_model: 解析任务归属失败: {e}")
            return "", ""

    def list_trained_versions(self, task_id: str = "") -> dict:
        """可用来预标注的已训模型版本清单（只列真实有权重 best.pt 的版本）。

        默认只给当前算法库自己的版本——同库的类别体系与标注任务一致，画出来的框才对得上。
        本库一个都没有时（新库还没训过），把其他库的版本也带出来并标记 same_library=False，
        前端会注明"类别可能对不上"，方便先用现成模型把链路跑通。
        """
        _, owner_model = self._task_dataset_model(task_id) if task_id else ("", "")
        root = self._library_root_of(owner_model) if owner_model else ""
        mine, others = [], []
        registry = settings.REGISTRY_DIR
        # 库根 id → 库名，缓存避免每个版本都重读一次 model.json
        lib_names: dict = {}

        def _lib_name(rid: str) -> str:
            if not rid:
                return ""
            if rid not in lib_names:
                lib_names[rid] = str(self._trained_meta(rid).get("display_name") or "").strip()
            return lib_names[rid]

        if registry.exists():
            for d in sorted(registry.iterdir()):
                if not d.is_dir() or not (d / "model.json").exists():
                    continue
                mid = d.name
                weights = self._trained_weights(mid)
                if weights is None:
                    continue  # 没有权重的版本（如刚新建的算法库根）不能用来预标注
                meta = self._trained_meta(mid)
                if meta.get("empty"):
                    continue
                lroot = self._library_root_of(mid)
                item = {
                    "model_id": mid,
                    # 模型管理页的重命名必须出现在这里：用户在管理页起的名字
                    # （如「积水分割v1（139张精修·服务器训练）」）是唯一可读的标识，
                    # 只给 version/model_id 会让人在预标注下拉里挑不出该用哪个权重。
                    "display_name": str(meta.get("display_name") or "").strip(),
                    "library_name": _lib_name(lroot),
                    "version": meta.get("version") or "",
                    "task": meta.get("task") or "",
                    "classes": meta.get("classes") or [],
                    "status": meta.get("status") or "",
                    "base_model": meta.get("base_model") or "",
                    "created_at": meta.get("created_at") or "",
                    "imgsz": meta.get("imgsz"),
                    "weights_name": weights.name,
                    "size_mb": round(weights.stat().st_size / 1e6, 1),
                    "same_library": bool(root) and lroot == root,
                }
                (mine if item["same_library"] else others).append(item)
        mine.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        others.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return {"library_root_id": root, "versions": mine + others, "same_library_count": len(mine)}

    def _resolve_gd_model_name(self, model_name: str) -> str:
        """本地优先解析 GroundingDINO 模型：若已下载到 SAM_MODELS_DIR/<名称> 则用本地路径，避免联网"""
        tail = model_name.rstrip("/").rsplit("/", 1)[-1]
        local_dir = settings.SAM_MODELS_DIR / tail
        if local_dir.exists() and (local_dir / "config.json").exists():
            return str(local_dir)
        return model_name

    def _load_grounding_dino(self, cfg: dict):
        """加载 Transformers 集成的 GroundingDINO（模型 + 处理器）"""
        if self._gd_model is not None:
            return self._gd_processor, self._gd_model
        model_name = self._resolve_gd_model_name(cfg.get("grounding_dino_model", "IDEA-Research/grounding-dino-tiny"))
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self._gd_processor = AutoProcessor.from_pretrained(model_name)
        self._gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        self._gd_name = model_name
        self._gd_model.to(self._resolve_device(cfg))
        self._gd_model.eval()
        return self._gd_processor, self._gd_model

    def _load_sam(self, cfg: dict, force: bool = False):
        """加载 SAM。force=True 供「点选抠图」使用：

        sam_enabled 那个开关管的是"AI 预标注要不要顺带描轮廓"，
        抠图是用户主动点击才触发的，和它无关，关掉开关也该能用。
        """
        if self._sam is not None:
            return self._sam
        if not force and not cfg.get("sam_enabled", True):
            return None
        weights = cfg.get("sam_weights", "sam_b.pt")
        from ultralytics import SAM
        # 优先从本地模型目录（sam/、custom/、backend 根目录）解析权重，找不到再由 ultralytics 自动联网下载
        self._sam = SAM(self._find_model_file(weights))
        self._sam_name = weights
        return self._sam

    # ------------------------------------------------------------------
    # 可用性 / 校验
    # ------------------------------------------------------------------
    async def is_available(self) -> dict:
        cfg = self._read_config()

        def _check() -> dict:
            try:
                det_ok = True
                for det in self._active_detectors(cfg):
                    try:
                        if self._load_engine(det, cfg) is None:
                            det_ok = False
                    except Exception:
                        det_ok = False
                sam_ok = True
                if cfg.get("sam_enabled", True):
                    sam = self._load_sam(cfg)
                    sam_ok = sam is not None
                return {
                    "available": det_ok and sam_ok,
                    "detector_ok": det_ok,
                    "sam_ok": sam_ok,
                    "device": self._resolve_device(cfg),
                }
            except Exception as e:
                return {"available": False, "error": str(e)}

        return await asyncio.to_thread(_check)

    async def validate_model(self) -> dict:
        cfg = self._read_config()

        def _validate() -> dict:
            try:
                checks = {}
                for det in self._active_detectors(cfg):
                    try:
                        if det == "grounding_dino":
                            self._load_grounding_dino(cfg)
                        elif det == "qwen_vl":
                            # 千问是远程/本地大模型服务，没有权重可加载，这里只做登记
                            pass
                        elif det == "trained_model":
                            # 本库已训模型不能 set_classes（会把训练学到的类别覆盖掉），
                            # 只做"权重能否加载 + 能否跑一次推理"的最小验证
                            m = self._load_trained(cfg)
                            if m is None:
                                raise RuntimeError("未选择已训模型版本，或该版本没有 best.pt 权重文件")
                            m.predict(
                                np.zeros((64, 64, 3), dtype=np.uint8),
                                conf=0.5,
                                verbose=False,
                                imgsz=64,
                                device=self._resolve_device(cfg),
                            )
                        else:
                            model = self._load_engine(det, cfg)
                            if model is not None:
                                model.set_classes(["test"])
                                model.predict(
                                    np.zeros((64, 64, 3), dtype=np.uint8),
                                    conf=0.5,
                                    verbose=False,
                                    imgsz=64,
                                    device=self._resolve_device(cfg),
                                )
                    except Exception:
                        checks[f"detector:{det}"] = "fail"
                    else:
                        checks[f"detector:{det}"] = "ok"
                if cfg.get("sam_enabled", True):
                    self._load_sam(cfg)
                    checks["sam"] = "ok"
                return {"ok": True, "checks": checks}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        return await asyncio.to_thread(_validate)

    # ------------------------------------------------------------------
    # 单图预标注
    # ------------------------------------------------------------------
    async def auto_label(self, task_id: str, image_id: str,
                         classes: List[str], conf: Optional[float] = None,
                         prompts: Optional[List[str]] = None) -> dict:
        cfg = self._read_config()
        conf = conf if conf is not None else cfg.get("conf", 0.25)

        item = await self._get_item(task_id, image_id)
        if not item:
            raise ValueError("Image not found in task")

        img_abs = settings.DATA_DIR / item["image_path"]
        if not img_abs.exists():
            raise ValueError(f"Image file not found: {img_abs}")

        # 标签字典（中文名/中文描述/英文检测提示词）：千问靠中文描述理解业务类别，
        # GroundingDINO / YOLO-World 靠 ai_prompt_en 才认得"积水"这类英文不存在的词，
        # 因此不论哪个引擎在用都要解析一次（失败返回 None，不阻断标注）
        ld = self._task_labels_desc(task_id)
        if ld:
            cfg = dict(cfg)
            cfg["qwen_labels_desc"] = ld

        # 模型推理非线程安全，加锁串行化，避免并发访问导致模型状态损坏
        async with self._lock:
            boxes = await asyncio.to_thread(self._run_auto_label_sync, img_abs, classes, conf, cfg, prompts)
        # L2 千问审核（设计文档 §6.2/§6.3）：qwen 启用时对预标注结果自动二次审核。
        # box_id = 返回列表下标，前端按此映射审核结论到对应框。
        audit, audit_msg = None, None
        if boxes and cfg.get("qwen_enabled", False):
            for i, b in enumerate(boxes):
                b["box_id"] = i
            audit, audit_msg = await asyncio.to_thread(
                self.qwen_audit_detections, img_abs, boxes,
                cfg.get("qwen_labels_desc"), cfg, classes
            )
        return {
            "image_id": image_id,
            "width": item["width"],
            "height": item["height"],
            "boxes": boxes,
            "audit": audit,
            "audit_message": audit_msg,
        }

    # ------------------------------------------------------------------
    # 点选抠图（智能抠图）：点一下目标 → SAM 描出轮廓 → 直接落一个分割标注
    # ------------------------------------------------------------------
    async def point_label(self, task_id: str, image_id: str, class_id,
                          points: List[dict]) -> dict:
        """按"点提示"做分割，也就是修图软件里智能抠图的那套交互。

        和 interactive_label 的本质区别：那条要先让检测器猜出方框，猜不出来
        （碎水洼、连成一大片的浅水面）就没辙；这条完全由用户点的位置决定，
        点哪儿抠哪儿，拐角不用人一个一个描。
        points: [{"x": 原图像素x, "y": 原图像素y, "label": 1|0}, ...]
                 label=1 点在物体上（要这块），0 点在不要的地方（抠掉这块）
        """
        pts = [p for p in (points or [])
               if isinstance(p, dict) and p.get("x") is not None and p.get("y") is not None]
        if not pts:
            raise ValueError("需要一个以上的点")
        if not any(int(p.get("label", 1)) == 1 for p in pts):
            raise ValueError("至少要有一个前景点：先点一下水块里面")

        cfg = self._read_config()
        item = await self._get_item(task_id, image_id)
        if not item:
            raise ValueError("Image not found in task")
        img_abs = settings.DATA_DIR / item["image_path"]
        if not img_abs.exists():
            raise ValueError(f"Image file not found: {img_abs}")

        # 模型推理非线程安全，与预标注共用同一把锁串行化
        async with self._lock:
            boxes = await asyncio.to_thread(
                self._run_point_label_sync, img_abs, class_id, pts, cfg
            )
        return {
            "image_id": image_id,
            "width": item["width"],
            "height": item["height"],
            "boxes": boxes,
        }

    def _get_point_predictor(self, img_abs: Path, cfg: dict):
        """点选抠图专用：缓存交互式 SAMPredictor，同一张图只编码一次。

        ultralytics 的 sam.predict(points=...) 每次调用都重跑图像编码器（ViT-B@1024，
        CPU 上 4-6 秒，占单次耗时 90%）。抠图交互是"同一张图反复补点/点掉"，
        改用 Predictor.set_image() + 直接调用：首次编码后，后续每次只走提示解码（~0.2s）。
        图变了 / 权重或 imgsz 变了才重新 set_image。
        """
        from ultralytics.models.sam import Predictor as SAMPredictor
        weights = cfg.get("sam_weights", "sam_b.pt")
        imgsz = cfg.get("sam_imgsz", 1024)
        device = self._resolve_device(cfg)
        key = f"{weights}|{imgsz}|{device}"
        if self._sam_pred is None or self._sam_pred_key != key:
            self._sam_pred = SAMPredictor(overrides=dict(
                task="segment", mode="predict", model=self._find_model_file(weights),
                imgsz=imgsz, device=device, verbose=False, save=False, conf=0.25,
            ))
            self._sam_pred_key = key
            self._sam_pred_img = None
        if self._sam_pred_img != str(img_abs):
            self._sam_pred.set_image(str(img_abs))
            self._sam_pred_img = str(img_abs)
        return self._sam_pred

    def _run_point_label_sync(self, img_abs: Path, class_id, points: List[dict], cfg: dict) -> list:
        """同步执行：SAM 按点提示出掩码 → 抽轮廓 + 外接矩形 → 返回兼容标注体系的框。

        点提示口径（ultralytics models/sam/predict.py::_prepare_prompts）：
          points 形状 (N 个物体, 每个物体的点数, 2)，labels 形状 (N, 点数)
        传一个 [[x,y],[x,y]] 就是"同一个物体上的两个点"，只回一个掩码；
        传成平铺的 [x,y],[x,y] 会被读成"两个物体各一个点"，回两个掩码 —— 别写错。

        连通性判断（离散点击点 → 闭合区域）：实测发现把"相距很远的两个点"当成
        同一物体送进 SAM，它会回一个占半张图的 everything 掩码（两个点都被盖住，
        但 200+ 碎块）；而"同一水洼内的两个点"回的是贴合目标的小掩码。
        所以"盖住点数"多不能当判据，真正可靠的是"逐点掩码互相包含"：
        点 j 落在点 i 的单点掩码里 → 两点在同一块连通区域。
        流程：① 每个前景点单独出掩码；② 按"掩码互相包含"做并查集分组
        （i 的掩码盖住 j 且 j 的掩码盖住 i 才连边，避免传递误合并）；
        ③ 每组一次合并提示出最终掩码（组内点全部被盖住才算这组的区域）。
        远距点自动拆成多个框，近距点自动合并成一个闭合区域。

        走 _get_point_predictor 的交互式缓存：同一张图第二次点击起不再重跑图像编码。
        """
        sam = self._load_sam(cfg, force=True)
        if sam is None:
            raise ValueError("SAM 模型不可用，请在标注模型设置里检查 sam_weights（如 sam_b.pt）")

        xy = [[float(p["x"]), float(p["y"])] for p in points]
        lb = [int(p.get("label", 1)) for p in points]

        def _call(pts_xy, pts_lb):
            try:
                pred = self._get_point_predictor(img_abs, cfg)
                return pred(points=[pts_xy], labels=[pts_lb])
            except Exception:
                # 交互式路径异常时回退到一次性 predict，保证功能可用
                try:
                    self._sam_pred = None
                    self._sam_pred_img = None
                    return sam.predict(
                        img_abs, points=[pts_xy], labels=[pts_lb], verbose=False,
                        imgsz=cfg.get("sam_imgsz", 1024), device=self._resolve_device(cfg),
                    )
                except Exception as e2:
                    raise ValueError(f"SAM 点选分割失败：{e2}")

        fg_idx = [i for i, l in enumerate(lb) if l == 1]
        neg_pts = [[xy[i], 0] for i in range(len(xy)) if lb[i] == 0]

        # 单点（最常见路径）：直接一次提示出掩码，行为与旧版一致
        if len(fg_idx) <= 1:
            want = xy[fg_idx[0]] if fg_idx else xy[0]
            results = _call(xy, lb)
            mask, orig_shape = self._pick_point_mask(results, want)
            if mask is None:
                return []
            item = self._mask_to_box(mask, orig_shape, class_id, cfg)
            if item:
                item["coverage"] = 1.0
                item["point_indices"] = list(fg_idx)
            return [item] if item else []

        # 多前景点：逐点掩码 → 互含并查集分组 → 每组一次合并提示
        per: list = []
        for i in fg_idx:
            r = _call([xy[i]], [1])
            m, shape = self._pick_point_mask(r, xy[i])
            per.append({"idx": i, "mask": m, "shape": shape})

        # 并查集：i、j 的掩码互相包含对方点位置才连边
        parent = list(range(len(per)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a in range(len(per)):
            for b in range(a + 1, len(per)):
                if per[a]["mask"] is None or per[b]["mask"] is None:
                    continue
                if self._mask_contains_point(per[a]["mask"], per[a]["shape"], xy[per[b]["idx"]]) \
                        and self._mask_contains_point(per[b]["mask"], per[b]["shape"], xy[per[a]["idx"]]):
                    union(a, b)

        groups: dict = {}
        for k in range(len(per)):
            groups.setdefault(find(k), []).append(per[k]["idx"])

        # 最后一个前景点所在的组优先排前面：前端约定 boxes[0] 并入"当前目标"，
        # 用户最新点击的那块才应该是当前目标，其余组作为新目标另起框
        last_group = max(groups.values(), key=lambda g: max(g))
        ordered = [last_group] + [g for g in groups.values() if g is not last_group]

        items = []
        for g in ordered:
            gxy = [xy[i] for i in g]
            results = _call(gxy + [p[0] for p in neg_pts],
                            [1] * len(gxy) + [0] * len(neg_pts))
            # 以组内最后一个点挑掩码：多点合并提示下它最接近"用户最新意图"
            mask, shape = self._pick_point_mask(results, gxy[-1])
            if mask is None:
                continue
            covered = [i for i in g if self._mask_contains_point(mask, shape, xy[i])]
            item = self._mask_to_box(mask, shape, class_id, cfg)
            if not item:
                continue
            item["coverage"] = round(len(covered) / len(g), 4) if g else 1.0
            item["point_indices"] = g
            items.append(item)
        return items

    @staticmethod
    def _mask_contains_point(mask, orig_shape, pt) -> bool:
        """判断原图坐标点 pt 是否落在掩码内（掩码尺寸可能小于原图，按比例换算）。"""
        if mask is None:
            return False
        mh, mw = int(mask.shape[0]), int(mask.shape[1])
        if mh < 2 or mw < 2:
            return False
        oh = float(orig_shape[0]) if orig_shape and len(orig_shape) >= 1 and orig_shape[0] else float(mh)
        ow = float(orig_shape[1]) if orig_shape and len(orig_shape) >= 2 and orig_shape[1] else float(mw)
        px, py = int(round(pt[0] * mw / ow)), int(round(pt[1] * mh / oh))
        return 0 <= py < mh and 0 <= px < mw and mask[py, px] > 0

    @staticmethod
    def _pick_point_mask(results, want_pt) -> tuple:
        """从 SAM 返回的候选掩码里挑一个，返回 (掩码, 原图尺寸)。

        点提示有歧义时 SAM 可能回多份，必须挑"确实盖住用户点的位置"的那份，
        否则会出现"点的是左边水块、轮廓却描了右边一小条"。都盖不住时退回面积最小的
        （everything 大掩码在盖不住任何点时比碎块掩码面积大得多，取小者更可能是目标局部）。
        """
        best, best_shape, best_score = None, None, (-1, -1.0)
        for r in results or []:
            ms = getattr(r, "masks", None)
            if ms is None or getattr(ms, "data", None) is None or len(ms.data) == 0:
                continue
            shape = getattr(r, "orig_shape", None)
            for i in range(len(ms.data)):
                m = ms.data[i].cpu().numpy()
                mh, mw = int(m.shape[0]), int(m.shape[1])
                if mh < 2 or mw < 2:
                    continue
                contains = 1
                oh = float(shape[0]) if shape and len(shape) >= 1 and shape[0] else float(mh)
                ow = float(shape[1]) if shape and len(shape) >= 2 and shape[1] else float(mw)
                px, py = int(round(want_pt[0] * mw / ow)), int(round(want_pt[1] * mh / oh))
                if not (0 <= py < mh and 0 <= px < mw) or m[py, px] <= 0:
                    contains = 0
                area = float((m > 0).sum())
                score = (contains, -area if contains else area)
                if score > best_score:
                    best, best_shape, best_score = m, shape, score
        return best, best_shape

    def _mask_to_box(self, mask, orig_shape, class_id, cfg: dict) -> dict:
        """单物体掩码 → {class_id, 外接矩形, polygon}，坐标一律换算回原图像素。

        掩码可能因 sam_imgsz 比原图小，所以按两者比例缩放（与 _refine_with_sam 同一口径）。
        抽不出轮廓（无 cv2、掩码退化）时只给矩形框，不让用户白点一次。
        """
        mh, mw = int(mask.shape[0]), int(mask.shape[1])
        oh = float(orig_shape[0]) if orig_shape and len(orig_shape) >= 1 and orig_shape[0] else float(mh)
        ow = float(orig_shape[1]) if orig_shape and len(orig_shape) >= 2 and orig_shape[1] else float(mw)
        sx = ow / mw if mw > 0 else 1.0
        sy = oh / mh if mh > 0 else 1.0
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return {}
        item = {
            "class_id": class_id,
            "x1": round(min(float(xs.min()) * sx, ow), 2),
            "y1": round(min(float(ys.min()) * sy, oh), 2),
            "x2": round(min(float(xs.max()) * sx, ow), 2),
            "y2": round(min(float(ys.max()) * sy, oh), 2),
            "score": 1.0,
        }
        if cfg.get("sam_return_polygon", True):
            # 与 _refine_with_sam 同口径走 v1.1 管线（升到原图尺寸 → 高斯抹台阶 → 等弧长重采样）。
            # 之前这里走旧 DP 抽稀（eps=0.5px、上限 3000 点），抠图轮廓顶点 200+ 密到拖不动，
            # 和批量预标注的 70-100 点完全两个手感 —— 两条路径必须共用一套后处理。
            poly = self._mask_to_polygon_v11(mask, ow, oh, cfg)
            if not poly:
                poly = self._mask_to_polygon(
                    mask, sx, sy, ow, oh,
                    float(cfg.get("sam_polygon_eps_px", 0.5) or 0.5),
                    int(cfg.get("sam_polygon_max_points", 3000) or 3000),
                )
            if poly:
                item["polygon"] = poly
        return item

    def _mask_to_polygon_v11(self, mask, img_w: float, img_h: float, cfg: dict) -> list:
        """v1.1 抽稀管线：掩码 → 原图尺寸 → 高斯抹台阶 → 等弧长重采样多边形。

        与 _refine_with_sam 的 override_seg 分支完全同一套参数（sam_mask_blur_px、
        polygon_refine_enabled），失败/退化时返回 []，调用方退回旧 DP 路径。
        """
        if not cfg.get("polygon_refine_enabled", True):
            return []
        try:
            import cv2 as _cv2
            from src.utils.polygon_postprocess import mask_to_polygon as _mpp
            ow, oh = int(img_w), int(img_h)
            if ow < 4 or oh < 4:
                return []
            m_full = (mask > 0).astype(np.uint8)
            if m_full.shape[0] != oh or m_full.shape[1] != ow:
                m_full = _cv2.resize(m_full, (ow, oh), interpolation=_cv2.INTER_NEAREST)
            blur_px = float(cfg.get("sam_mask_blur_px", 2.0) or 0)
            if blur_px > 0:
                m_full = (_cv2.GaussianBlur(m_full.astype(np.float32), (0, 0), blur_px)
                          > 0.5).astype(np.uint8)
            pts = _mpp(m_full)
            if pts is None or len(pts) < 3:
                return []
            return [[round(float(px), 2), round(float(py), 2)] for px, py in pts]
        except Exception as e:
            print(f"point-label: v1.1 抽稀失败，退回旧轮廓: {e}")
            return []

    def erase_polygon(self, polygon: List[list], erase_rects: List[dict],
                      img_w: float, img_h: float, class_id, cfg: dict) -> list:
        """区域擦除：把一个分割轮廓按矩形"挖掉"，返回剩下的（可能多块）轮廓。

        纯几何、零 SAM 推理（毫秒级）：栅格化 polygon → 矩形内置 0 → 拆连通块 →
        每块走 v1.1 抽稀。用于"抠大了，拖一个框把多余那块（文字水印/相邻水面）切掉"，
        比右键一下一下点负点快得多。返回 [{class_id, x1,y1,x2,y2, polygon}]；
        全挖空时返回 []（前端据此删掉该框）。
        """
        try:
            import cv2
        except Exception:
            return []
        W, H = int(max(img_w, 4)), int(max(img_h, 4))
        m = np.zeros((H, W), dtype=np.uint8)
        pts = np.array([[float(x), float(y)] for x, y in polygon], dtype=np.int32)
        if len(pts) < 3:
            return []
        cv2.fillPoly(m, [pts], 1)
        for r in erase_rects or []:
            try:
                x1 = int(max(0, min(float(r["x1"]), float(r["x2"]))))
                x2 = int(min(W, max(float(r["x1"]), float(r["x2"]))))
                y1 = int(max(0, min(float(r["y1"]), float(r["y2"]))))
                y2 = int(min(H, max(float(r["y1"]), float(r["y2"]))))
            except Exception:
                continue
            if x2 > x1 and y2 > y1:
                m[y1:y2, x1:x2] = 0
        if int(m.sum()) == 0:
            return []
        comps = self._mask_components(
            m, (H, W),
            float(cfg.get("sam_polygon_eps_px", 0.5) or 0.5),
            int(cfg.get("sam_polygon_max_points", 3000) or 3000),
            refine=bool(cfg.get("polygon_refine_enabled", True)))
        out = []
        min_area = float(cfg.get("sam_erase_min_area_px", 400) or 0)
        for c in comps:
            bx, by, bw, bh = c["bbox"]
            if bw * bh < min_area:
                continue
            out.append({"class_id": class_id, "x1": bx, "y1": by,
                        "x2": round(bx + bw, 2), "y2": round(by + bh, 2),
                        "score": 1.0, "polygon": c["polygon"]})
        return out

    async def interactive_label(self, task_id: str, image_id: str,
                                classes: List[str], conf: Optional[float] = None,
                                prompts: Optional[List[str]] = None,
                                region: Optional[dict] = None) -> dict:
        """交互式标注：只在用户框选的局部区域（region，图像坐标）内做检测。

        相比全图盲扫，局部检测既减少误标，又能针对感兴趣区域精细标注，
        配合"点击/框选提示区域 + 文本提示"的交互方式，正中"误标过多"痛点。
        region 格式：{"x1":..,"y1":..,"x2":..,"y2":..}；缺省则全图检测。
        """
        cfg = self._read_config()
        conf = conf if conf is not None else cfg.get("conf", 0.25)

        item = await self._get_item(task_id, image_id)
        if not item:
            raise ValueError("Image not found in task")

        img_abs = settings.DATA_DIR / item["image_path"]
        if not img_abs.exists():
            raise ValueError(f"Image file not found: {img_abs}")

        # 标签字典（同单图预标注）：英文提示词给 GD/YOLO-World，中文描述给千问
        ld = self._task_labels_desc(task_id)
        if ld:
            cfg = dict(cfg)
            cfg["qwen_labels_desc"] = ld

        # 模型推理非线程安全，加锁串行化
        async with self._lock:
            boxes = await asyncio.to_thread(
                self._run_interactive_label_sync, img_abs, classes, conf, cfg, prompts, region
            )
        return {
            "image_id": image_id,
            "width": item["width"],
            "height": item["height"],
            "boxes": boxes,
        }

    def _run_interactive_label_sync(self, img_abs: Path, classes: List[str], conf: float,
                                    cfg: dict, prompts=None, region: Optional[dict] = None) -> list:
        """交互式标注同步执行：裁剪到区域 → 区域检测 → 坐标映射回原图 → 合并 → (可选)SAM 精修"""
        import os
        from tempfile import NamedTemporaryFile
        from PIL import Image

        if region:
            image = Image.open(img_abs).convert("RGB")
            w, h = image.size
            x1 = max(0, int(round(region.get("x1", 0))))
            y1 = max(0, int(round(region.get("y1", 0))))
            x2 = min(w, int(round(region.get("x2", w))))
            y2 = min(h, int(round(region.get("y2", h))))
            if x2 - x1 < 5 or y2 - y1 < 5:
                raise ValueError("Region too small")
            crop = image.crop((x1, y1, x2, y2))
            tmp = NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            tmp_path = Path(tmp.name)
            try:
                crop.save(tmp_path)
                det_boxes = self._detect(tmp_path, classes, conf, cfg, prompts)
                # 把裁剪图坐标映射回原图坐标
                for b in det_boxes:
                    b["x1"] += x1
                    b["y1"] += y1
                    b["x2"] += x1
                    b["y2"] += y1
            finally:
                if tmp_path.exists():
                    os.unlink(tmp_path)
        else:
            det_boxes = self._detect(img_abs, classes, conf, cfg, prompts)

        det_boxes = self._merge_same_class_boxes(
            det_boxes, cfg.get("merge_iou", 0.45), cfg.get("merge_dist", 0.35),
            tight=self._merge_tight_cfg(cfg))
        det_boxes = self._merge_cross_class_boxes(det_boxes)
        if cfg.get("sam_enabled", True):
            # SAM 在原图上按映射回原图的框做分割精修
            return self._refine_with_sam(img_abs, det_boxes, cfg)
        return det_boxes

    def _merge_tight_cfg(self, cfg: dict) -> dict:
        """读取"高度重合保留最紧框"判据（配置项 merge_tight_*）"""
        return {
            "iou": float(cfg.get("merge_tight_iou", 0.7)),
            "center": float(cfg.get("merge_tight_center", 0.15)),
            "area_ratio": float(cfg.get("merge_tight_area_ratio", 1.6)),
        }

    def _run_auto_label_sync(self, img_abs: Path, classes: List[str], conf: float, cfg: dict, prompts=None) -> list:
        """同步执行：检测 → 合并被遮挡分裂的同类框 → (可选) SAM 精修 → 返回 BBox 列表"""
        det_boxes = self._detect(img_abs, classes, conf, cfg, prompts)
        # 合并被遮挡物（如电线杆）分裂成多个的同类框，避免同一物体被识别成两个
        det_boxes = self._merge_same_class_boxes(
            det_boxes, cfg.get("merge_iou", 0.45), cfg.get("merge_dist", 0.35),
            tight=self._merge_tight_cfg(cfg))
        # 跨类别高度重合合并（GD 常把同一目标误标成不同类别，导致高度重合框残留）
        det_boxes = self._merge_cross_class_boxes(det_boxes)
        if cfg.get("sam_enabled", True):
            return self._refine_with_sam(img_abs, det_boxes, cfg)
        return det_boxes

    def _merge_same_class_boxes(self, boxes: list, iou_thr: float = 0.45, dist_thr: float = 0.35,
                                tight: dict = None) -> list:
        """合并同类且高度重叠/紧邻的框，解决遮挡导致的同一物体被识别成多个框的问题。

        同一物体被遮挡物（如电线杆）隔开时，YOLO-World 常输出多个相邻/重叠的同类框。
        合并规则（关键：既要吞掉"遮挡分裂的碎框"，又不能把"两个近距离独立目标"并成一框）：
        - tight 模式（高度重合判据，默认配置 merge_tight_iou=0.7 等）：IoU 很高 + 面积接近
          （面积比 ≤ merge_tight_area_ratio）且中心错位小（≤ merge_tight_center×对角）
          → 判定为同一目标的双框，保留面积更小（更紧贴目标）的框；
        - 否则仅 IoU 高于 iou_thr 或中心距离很近才取并集合并（默认 0.45 / 0.35×对角）；
        - 都不满足（如背人的大小悬殊框、中心错开的两框）→ 视为不同目标，两个都保留。
        tight 参数示例：{"iou": 0.7, "center": 0.15, "area_ratio": 1.6}
        """
        if not boxes:
            return boxes
        # 已带分割轮廓的框不参与重合合并：合并会改写矩形，轮廓就和框对不上了
        # （零样本引擎走到这一步还没有轮廓，所以这条只影响本库分割模型直接给出的轮廓）
        poly_boxes = [b for b in boxes if b.get("polygon")]
        if poly_boxes:
            boxes = [b for b in boxes if not b.get("polygon")]
            if not boxes:
                return poly_boxes

        def _iou(a, b):
            ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
            ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            a_area = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
            b_area = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
            return inter / (a_area + b_area - inter)

        def _center_dist(a, b):
            ca = ((a["x1"] + a["x2"]) / 2, (a["y1"] + a["y2"]) / 2)
            cb = ((b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2)
            return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5

        tight_iou = float((tight or {}).get("iou", 0.0))
        tight_center = float((tight or {}).get("center", 1.0))
        tight_area_ratio = float((tight or {}).get("area_ratio", 1.0))

        # 按类别分组，逐组合并
        by_class: Dict[int, list] = {}
        for b in boxes:
            by_class.setdefault(b["class_id"], []).append(b)

        merged: list = []
        for cid, group in by_class.items():
            group = sorted(group, key=lambda b: b.get("score", 0.0), reverse=True)
            used = [False] * len(group)
            for i in range(len(group)):
                if used[i]:
                    continue
                # 以当前框为基准，合并所有与之重叠/紧邻的同类框
                base = dict(group[i])
                base_diag = ((base["x2"] - base["x1"]) ** 2 + (base["y2"] - base["y1"]) ** 2) ** 0.5 or 1.0
                for j in range(i + 1, len(group)):
                    if used[j]:
                        continue
                    other = group[j]
                    # tight 判据：高度重合的同一目标 → 保留更紧（面积更小）的框
                    if tight_iou > 0 and _iou(base, other) >= tight_iou:
                        o_w = other["x2"] - other["x1"]
                        o_h = other["y2"] - other["y1"]
                        b_w = base["x2"] - base["x1"]
                        b_h = base["y2"] - base["y1"]
                        ratio = max(o_w * o_h, b_w * b_h) / (min(o_w * o_h, b_w * b_h) or 1.0)
                        is_tight = ratio <= tight_area_ratio and \
                            _center_dist(base, other) <= tight_center * base_diag
                        if is_tight:
                            # 保留面积更小的框（更紧贴目标），score 取更高者
                            if o_w * o_h < b_w * b_h:
                                base["x1"], base["y1"] = other["x1"], other["y1"]
                                base["x2"], base["y2"] = other["x2"], other["y2"]
                            base["score"] = max(base.get("score", 0.0), other.get("score", 0.0))
                            used[j] = True
                            continue
                    if _iou(base, other) >= iou_thr:
                        # 合并取外接矩形，保留更高 score
                        base["x1"] = min(base["x1"], other["x1"])
                        base["y1"] = min(base["y1"], other["y1"])
                        base["x2"] = max(base["x2"], other["x2"])
                        base["y2"] = max(base["y2"], other["y2"])
                        base["score"] = max(base.get("score", 0.0), other.get("score", 0.0))
                        used[j] = True
                    elif _center_dist(base, other) <= dist_thr * base_diag:
                        # 中心距离很近的同类框（遮挡分裂但几乎不重叠）也合并
                        base["x1"] = min(base["x1"], other["x1"])
                        base["y1"] = min(base["y1"], other["y1"])
                        base["x2"] = max(base["x2"], other["x2"])
                        base["y2"] = max(base["y2"], other["y2"])
                        base["score"] = max(base.get("score", 0.0), other.get("score", 0.0))
                        used[j] = True
                merged.append(base)
        return merged + poly_boxes

    def _merge_cross_class_boxes(self, boxes: list, iou_thr: float = 0.5) -> list:
        """合并不同类别但高度重合的框（GD 常把同一目标误标成不同类别）。

        不同类别两个框 IoU ≥ 0.5 时几乎可判定为同一目标的重复检测，只保留 score 更高者。
        """
        if not boxes:
            return boxes
        # 带轮廓的框不参与跨类别取舍：这里会整框丢弃，误删一个就是丢一份分割标注
        poly_boxes = [b for b in boxes if b.get("polygon")]
        if poly_boxes:
            boxes = [b for b in boxes if not b.get("polygon")]
            if not boxes:
                return poly_boxes

        def _iou(a, b):
            ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
            ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            aa = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
            bb = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
            return inter / (aa + bb - inter)

        # 按 score 降序，保留高置信度框，丢弃与已保留框高度重合的任意类别框
        boxes = sorted(boxes, key=lambda b: b.get("score", 0.0), reverse=True)
        out = []
        for b in boxes:
            if any(_iou(kept, b) >= iou_thr for kept in out):
                continue
            out.append(b)
        return out + poly_boxes

    async def clean_task_annotations(self, task_id: str) -> dict:
        """清洗一个标注任务中所有图片的高度重合标注框（跨类别 IoU≥0.5、同类高度重叠/紧邻）。

        复用与检测时完全相同的合并逻辑（_merge_same_class_boxes + _merge_cross_class_boxes），
        一次性根治历史遗留的重叠标注——无论之前用的是哪个检测模型，都统一清理，无需按模型分别处理。
        标注读取优先每图 JSON（一图一文件），兼容历史任务集中式 annotations.json。
        """
        task_dir = settings.ANNOTATIONS_DIR / task_id
        task_file = task_dir / "task.json"
        if not task_file.exists():
            return {"ok": False, "error": "Task not found"}
        try:
            task_meta = _load_json(task_file)
        except Exception:
            return {"ok": False, "error": "读取任务元数据失败"}
        dataset_id = task_meta.get("dataset_id", "")
        version = task_meta.get("version", "v1")
        return await asyncio.to_thread(self._clean_task_annotations_sync, dataset_id, version, task_dir)

    def _clean_task_annotations_sync(self, dataset_id: str, version: str, task_dir: Path) -> dict:
        images_cleaned = 0
        boxes_removed = 0

        def _clean_one(ann: dict) -> bool:
            """合并一张图的重复框；有变化则更新并返回 True"""
            nonlocal images_cleaned, boxes_removed
            boxes = ann.get("boxes") or []
            if not boxes:
                return False
            # 带分割轮廓的框不参与重合合并：合并会改写矩形，轮廓与框就对不上了
            seg_boxes = [b for b in boxes if b.get("polygon")]
            rect_boxes = [b for b in boxes if not b.get("polygon")]
            if not rect_boxes:
                return False
            orig_len = len(rect_boxes)
            merged = self._merge_same_class_boxes(
                rect_boxes, tight=self._merge_tight_cfg(self._read_config()))
            merged = self._merge_cross_class_boxes(merged)
            if len(merged) < orig_len:
                ann["boxes"] = merged + seg_boxes
                ann["updated_at"] = datetime.now().isoformat()
                images_cleaned += 1
                boxes_removed += orig_len - len(merged)
                return True
            return False

        # 每图 JSON（新存储，数据集一律为 datasets/{dataset_id}，无 version 子目录）
        ann_dir = settings.DATASETS_DIR / dataset_id / "annotations"
        if ann_dir.exists():
            for p in ann_dir.glob("*.json"):
                try:
                    ann = _load_json(p)
                except Exception:
                    continue
                if _clean_one(ann):
                    _save_json(p, ann)

        # 旧版本任务集中式标注文件（兼容清洗）
        legacy_file = task_dir / "annotations.json"
        if legacy_file.exists():
            try:
                legacy = _load_json(legacy_file)
            except Exception as e:
                return {"ok": False, "error": f"读取标注文件失败: {e}"}
            dirty = False
            for ann in legacy.values():
                if _clean_one(ann):
                    dirty = True
            if dirty:
                _save_json(legacy_file, legacy)

        return {
            "ok": True,
            "images_cleaned": images_cleaned,
            "boxes_removed": boxes_removed,
        }

    def _dict_labels(self, cfg: dict) -> list:
        """当前任务所属算法库的标签字典行列表（未归属/解析失败时为空）。

        键名沿用 qwen_labels_desc：该字段最初为千问引入，现在英文检测提示词也从同一份字典取。
        """
        info = cfg.get("qwen_labels_desc") or {}
        return info.get("labels") or []

    def _prompts_with_dict(self, classes, prompts, cfg) -> list:
        """英文检测提示词优先级：标签字典 ai_prompt_en > 前端传入 prompts > 原类别名。

        GroundingDINO / YOLO-World 只认英文，业务类别名（如 jishui）或中文（积水）原样发过去
        等于无效输入（实测：jishui → 整张图一个框；积水 → 0 个框）。在字典里为类别填一句
        英文描述（如 standing water or puddle on the road）即可让零样本引擎"看得懂"业务类别。
        只替换提示词内容、不改列表长度与顺序，因此检测结果按位置回 class_id 的映射不受影响。
        """
        base = [str(p) for p in (prompts if prompts else (classes or []))]
        labels = self._dict_labels(cfg)
        if not base or not labels:
            return base
        # 按 english_code 建索引（大小写不敏感），避免依赖字典行顺序与 classes 顺序一致
        by_code = {}
        for l in labels:
            code = str(l.get("english_code") or "").strip().lower()
            if code:
                by_code[code] = str(l.get("ai_prompt_en") or "").strip()
        out = []
        for i, p in enumerate(base):
            code = str(classes[i]).strip().lower() if classes and i < len(classes) else p.strip().lower()
            dict_prompt = by_code.get(code) or ""
            out.append(dict_prompt if dict_prompt else p)
        return out

    def _detect(self, img_abs, classes, conf, cfg, prompts=None) -> list:
        """按配置运行一个或多个检测引擎，多引擎时结果合并（统一交给后续同类/跨类别合并去重）"""
        out = []
        for det in self._active_detectors(cfg):
            try:
                if det == "grounding_dino":
                    boxes = self._detect_grounding_dino(img_abs, classes, conf, cfg, prompts)
                elif det == "qwen_vl":
                    boxes = self._detect_qwen(img_abs, classes, conf, cfg, prompts)
                elif det == "yolo_world":
                    boxes = self._detect_yolo(img_abs, classes, conf, cfg, prompts)
                elif det == "trained_model":
                    boxes = self._detect_trained(img_abs, classes, conf, cfg, prompts)
                else:
                    continue
            except Exception as e:
                print(f"multi-detect: 引擎 {det} 推理失败: {e}")
                boxes = []
            out.extend(boxes)
        return out

    def _detect_yolo(self, img_abs, classes, conf, cfg, prompts=None) -> list:
        det = self._load_detector(cfg)
        if det is None:
            return []
        # 用英文提示词做文本驱动检测，中文/拼音类别名靠标签字典的 ai_prompt_en 兜住。
        # 结果与 classes 一一对应，因此检测出的 class_id 仍对应 classes 索引。
        text_prompts = self._prompts_with_dict(classes, prompts, cfg)
        # 类别提示词未变化时复用已编码的检测头，避免多类别时每张图重复 set_classes（耗时且易超时）
        if self._detector_prompts != text_prompts:
            try:
                det.set_classes(text_prompts)
                self._detector_prompts = list(text_prompts)
            except Exception as e:
                print(f"Warning: set_classes failed: {e}")
        results = det.predict(
            img_abs,
            conf=conf,
            iou=cfg.get("iou", 0.40),
            imgsz=cfg.get("imgsz", 640),
            verbose=False,
            device=self._resolve_device(cfg),
        )
        out = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                out.append({
                    "class_id": int(boxes.cls[i]),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "score": float(boxes.conf[i]),
                })
        return out

    def _detect_trained(self, img_abs, classes, conf, cfg, prompts=None) -> list:
        """用本库已训模型的 best.pt 出框（分割模型会顺带把轮廓一起给出）。

        与零样本引擎的区别：不再依赖英文提示词猜，模型是按人工标注学出来的，
        所以"积水 vs 地面反光"这类零样本分不清的边界，只要用户标够样本就能画准。
        """
        model = self._load_trained(cfg)
        if model is None:
            return []
        mapping = self._trained_class_map(model, classes, cfg)
        if not mapping:
            return []
        results = model.predict(
            img_abs,
            conf=conf,
            iou=cfg.get("iou", 0.40),
            imgsz=cfg.get("imgsz", 640),
            verbose=False,
            device=self._resolve_device(cfg),
        )
        want_polygon = bool(cfg.get("sam_return_polygon", True))
        max_points = int(cfg.get("sam_polygon_max_points", 3000) or 3000)
        out = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            masks = getattr(r, "masks", None)
            for i in range(len(boxes)):
                cls_id = mapping.get(int(boxes.cls[i]))
                if cls_id is None:
                    continue  # 模型里有、任务里没有的类别直接丢掉
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                item = {
                    "class_id": cls_id,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "score": float(boxes.conf[i]),
                }
                # 分割模型：逐连通域抽轮廓，一块不连通区域 = 一条独立标注
                comps = []
                if want_polygon and masks is not None and len(masks) > i and masks.data is not None:
                    try:
                        mi = masks.data[i]
                        m = mi.cpu().numpy() if hasattr(mi, "cpu") else np.asarray(mi)
                        # 原图尺寸缺失时按掩码尺寸处理（等价于不再缩放）
                        orig = tuple(int(v) for v in (r.orig_shape if getattr(r, "orig_shape", None)
                                                      else m.shape[:2]))
                        comps = self._mask_components(
                            m, orig,
                            float(cfg.get("sam_polygon_eps_px", 0.5) or 0.5), max_points,
                            refine=bool(cfg.get("polygon_refine_enabled", False)))
                    except Exception as e:
                        print(f"trained_model: 取分割轮廓失败（保留矩形框）: {e}")
                if not comps:
                    out.append(item)
                    continue
                # 最大块沿用当前实例的条目（rect 换成该块自己的紧贴框），其余块另起新条目
                bx, by, bw, bh = comps[0]["bbox"]
                item.update({"x1": bx, "y1": by, "x2": bx + bw, "y2": by + bh,
                             "polygon": comps[0]["polygon"]})
                out.append(item)
                for c in comps[1:]:
                    cx, cy, cw, ch = c["bbox"]
                    out.append({
                        "class_id": cls_id,
                        "x1": cx, "y1": cy, "x2": cx + cw, "y2": cy + ch,
                        "score": float(boxes.conf[i]),
                        "polygon": c["polygon"],
                    })
        return out

    # 单个实例的 mask 碎成多块时，前景面积小于该占比（相对本实例前景总面积）的碎块丢弃
    _COMPONENT_MIN_AREA_RATIO = 0.02

    @staticmethod
    def _mask_components(mask, orig_shape, eps_px: float, max_points: int,
                         refine: bool = False, prob=None) -> list:
        """把分割模型输出的单个 mask 按连通块拆成多份轮廓 + 外接矩形。

        为什么不用 ultralytics 现成的 ``masks.xy``：它走的是
        ``ops.masks2segments(masks, strategy="all")``，当某个实例的 mask 含多块
        互不相连的区域时，该函数会用 ``np.concatenate`` 把各块轮廓**首尾拼成一条点序列**
        返回 —— blob1 的末点被直连到 blob2 的首点。这条连线面积为零，填充时看不见，
        描边时就成了横跨画面的一根细线（且 predict 没有任何参数能绕开）。
        这里改成每块各自成一条独立轮廓，细线从根上消失，顺带把模型糊成一个实例的
        多片积水还原成多条可单独接受/删除的标注。

        参数：
          mask      : 单个实例的二值掩码 (H, W)，>0 为前景；可以是 letterbox 后的低
                      分辨率掩码，也可以已经与原图同尺寸（后者内部跳过重采样）
          orig_shape: 原图尺寸 (H, W)
          eps_px    : RDP 容差，单位＝**原图像素**（轮廓上任一点到简化前的偏差不超过它）
          max_points: 单块轮廓的顶点数上限
        返回 [{"bbox": (x, y, w, h), "polygon": [[x, y], ...]}]，坐标为原图像素，
        按面积从大到小排序；抽不出来（无 cv2、掩码过小）时返回 []，调用方退化成只给矩形框。
        """
        try:
            import cv2
        except Exception:
            return []
        try:
            m = (np.asarray(mask) > 0).astype(np.uint8)
            if m.shape[0] < 4 or m.shape[1] < 4:
                return []
            oh, ow = int(orig_shape[0]), int(orig_shape[1])
            # 掩码空间 → 原图空间：与 ultralytics scale_coords 同一套 letterbox 反算，
            # 保证与 boxes.xyxy 的坐标系严格一致
            gain = min(m.shape[0] / float(oh), m.shape[1] / float(ow))
            pad_x = (m.shape[1] - ow * gain) / 2.0
            pad_y = (m.shape[0] - oh * gain) / 2.0
            # 先升到原图分辨率再抽轮廓：masks.data 停在 letterbox 后的输入尺寸（如 640×352
            # 对应 1316×712 的原图，一格 ≈2 原图像素），在它上面取边界只会得到一格一格的
            # 台阶；双线性升到原图后取 0.5 等值线（与 ultralytics 画掩码时
            # scale_masks(...)>0.5 同源）才是"平台外推理"里看到的那条线
            if (m.shape[0], m.shape[1]) != (oh, ow) and oh >= 4 and ow >= 4:
                px, py = int(round(pad_x)), int(round(pad_y))
                sub = m[py:m.shape[0] - py, px:m.shape[1] - px]
                if sub.shape[0] >= 2 and sub.shape[1] >= 2:
                    prob_up = cv2.resize(sub.astype(np.float32), (ow, oh),
                                         interpolation=cv2.INTER_LINEAR)
                    m = (prob_up > 0.5).astype(np.uint8)
                    if prob is None:
                        prob = prob_up  # 连续概率面留给后处理找真实边缘
                    gain, pad_x, pad_y = 1.0, 0.0, 0.0
            total = float(m.sum())
            if total <= 0:
                return []
            # APPROX_NONE：保留逐像素边界点，简化统一交给下面的 approxPolyDP
            # （APPROX_SIMPLE 会把共线段压成两端点，斜向边界先丢细节再简化更糙）
            found = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [c for c in (found[0] if len(found) == 2 else found[1])
                        if c is not None and len(c) >= 3]
            if not contours:
                return []

            def _to_orig(cnt):
                p = cnt.reshape(-1, 2).astype(float)
                p[:, 0] = np.clip((p[:, 0] - pad_x) / gain, 0.0, float(orig_shape[1]))
                p[:, 1] = np.clip((p[:, 1] - pad_y) / gain, 0.0, float(orig_shape[0]))
                return p

            comps = []
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area <= 0 or area / total < SAMService._COMPONENT_MIN_AREA_RATIO:
                    continue
                # 容差取**绝对像素**而不是周长比例：按比例算时大目标的容差会被放大到好几个
                # 像素，曲率大的地方整段被削平（"围不住目标"就是这么来的）。
                # 掩码没升到原图分辨率的那条分支里，按 gain 把原图像素折算回掩码尺度。
                approx = cv2.approxPolyDP(cnt, max(float(eps_px), 0.5) * gain, True)
                pts = _to_orig(approx)
                # 单块轮廓是连续闭合环，等距抽稀不会再拉出跨块直线
                if max_points and len(pts) > max_points >= 3:
                    step = len(pts) / float(max_points)
                    pts = pts[[int(i * step) for i in range(max_points)]]
                if len(pts) < 3:
                    continue
                # 后处理管线：IoU 门控的自适应简化 + 长边细分（修直边伪影）+ 平滑 + 回投影。
                # 只在掩码已升到原图分辨率（gain==1）时做，坐标空间才与 comp_mask 一致；
                # 逐块单独处理，避免多块积水互相污染 IoU 兜底判定。
                # 用 mask_to_polygon 而非 refine_polygon：后者只补点不简化，顶点数降不下来。
                if refine and gain == 1.0:
                    try:
                        from src.utils.polygon_postprocess import mask_to_polygon
                        comp_mask = np.zeros(m.shape, dtype=np.uint8)
                        cv2.drawContours(comp_mask, [cnt], -1, 1, cv2.FILLED)
                        pts = mask_to_polygon(comp_mask, prob=prob)
                        if len(pts) < 3:
                            continue
                    except Exception as e:
                        print(f"trained_model: polygon 后处理失败（保留原轮廓）: {e}")
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                comps.append({
                    "bbox": (round(float(x_min), 2), round(float(y_min), 2),
                             round(float(x_max - x_min), 2), round(float(y_max - y_min), 2)),
                    "polygon": [[round(float(px), 2), round(float(py), 2)] for px, py in pts],
                })
            comps.sort(key=lambda c: c["bbox"][2] * c["bbox"][3], reverse=True)
            return comps
        except Exception as e:
            print(f"trained_model: mask 连通域拆分失败（退化为一块轮廓/矩形框）: {e}")
            return []

    def _detect_qwen(self, img_abs, classes, conf, cfg, prompts=None) -> list:
        """千问 VL 大模型预标注（本地预留接口）。

        - qwen_enabled=False（本地默认）：不启用，返回空并提示。
        - qwen_mock=True：返回模拟框，仅用于本地验证链路（不真实调用大模型）。
        - 真实调用：把图片+类别清单发给千问 VL，要求对每个类别输出 像素坐标 bbox，
          解析 <box>x1,y1,x2,y2</box> 格式的结果映射回 classes 索引。
        """
        if not cfg.get("qwen_enabled", False):
            return []
        if cfg.get("qwen_mock", False):
            return self._mock_qwen_boxes(img_abs, classes)
        # 类别清单：若当前任务所属模型的标签字典可用，则附加中文名+中文描述，
        # 帮助大模型理解业务类别（千问输出仅按序号 class_id 对齐 classes，不依赖名称匹配）
        desc_info = cfg.get("qwen_labels_desc") or {}
        labels_desc = desc_info.get("labels") or []
        if labels_desc and len(labels_desc) >= len(classes):
            label_text = ", ".join(
                f"{i}:{lab.get('chinese_name') or classes[i]}"
                + (f"（{lab.get('chinese_desc')}）" if lab.get("chinese_desc") else "")
                for i, lab in enumerate(labels_desc[:len(classes)])
            )
        else:
            label_text = ", ".join([f"{i}:{c}" for i, c in enumerate(classes)])
        sys_msg = (
            "你是一个精细的视觉目标检测助手。对给定的图片和类别清单，找出图中出现的每一个目标实例。\n"
            "对所有找到的实例，输出格式为每行一个：\n"
            "<box>x1,y1,x2,y2</box> <class>i</class>\n"
            "其中 x1,y1,x2,y2 是该目标在图片中的像素坐标边界框（左上角与右下角，整数即可），\n"
            "i 是类别清单中的序号（0 起）。必须逐行输出，每行只能有一个实例，不要输出任何其他文字或解释。\n"
            "如果某个类别没有目标，不要输出它的行。\n"
        )
        user_msg = f"类别清单：\n{label_text}\n请检测这张图片中的所有目标实例。"
        content = self._qwen_image_chat(cfg, sys_msg, user_msg, img_abs)
        if not content:
            return []
        return self._parse_qwen_boxes(content, classes)

    def _parse_qwen_boxes(self, content: str, classes) -> list:
        """解析千问输出：<box>x1,y1,x2,y2</box> <class>i</class> → 标准 boxes"""
        import re
        boxes = []
        for m in re.finditer(r"<box>\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*</box>", content, re.IGNORECASE):
            try:
                x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))
            except Exception:
                continue
            cm = re.search(r"<class>\s*(\d+)\s*</class>", content[m.end():])
            if cm is None:
                continue
            cls = int(cm.group(1))
            if not (0 <= cls < len(classes)):
                continue
            boxes.append({
                "class_id": cls,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "score": 0.9,  # 千问不返回置信度，固定高值表示"已确认检出"；由标注页与 SAM 精修协同
            })
        return boxes

    def _mock_qwen_boxes(self, img_abs, classes) -> list:
        """本地联调用模拟框：不调用大模型，产生每个类别一个位于图中心的框，便于验证前端链路"""
        from PIL import Image as PILImage
        try:
            w, h = PILImage.open(img_abs).size
        except Exception:
            w, h = 640, 640
        cx, cy, bw, bh = w / 2, h / 2, w * 0.5, h * 0.5
        out = []
        for i, _cls in enumerate(classes):
            ox = i * 0.08 * w
            out.append({
                "class_id": i,
                "x1": cx - bw / 2 + ox, "y1": cy - bh / 2,
                "x2": cx + bw / 2 + ox, "y2": cy + bh / 2,
                "score": 0.9,
            })
        print(f"qwen IN-MOCK: {len(classes)} boxes for {img_abs}（本地联调模式，未调用真实模型）")
        return out

    def _qwen_image_chat(self, cfg: dict, sys_msg: str, user_msg: str, img_abs: Path) -> str:
        """统一调千问 VL（ollama / OpenAI 兼容接口），返回模型文本输出；失败返回 ''。

        - 自动压缩图片（最长边 ≤1024，VLM 输入过大既慢又易超 MaxInputError）
        - 同时兼容 Ollama /api/chat 与 OpenAI 兼容 /v1/chat/completions
        检测（_detect_qwen）与审核（qwen_audit_detections）共用。
        """
        import base64, io
        import json as _json
        import urllib.request
        from PIL import Image as PILImage
        try:
            image = PILImage.open(img_abs).convert("RGB")
            max_side = 1024
            w, h = image.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=88)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            print(f"qwen: 图片编码失败 {img_abs}: {e}")
            return ""
        endpoint = (cfg.get("qwen_endpoint") or "http://localhost:11434").rstrip("/")
        api_key = cfg.get("qwen_api_key") or ""
        headers = {"Content-Type": "application/json"}
        data_url = f"data:image/jpeg;base64,{b64}"
        if "ollama" in (cfg.get("qwen_backend") or "").lower() or endpoint.startswith("http://localhost:1143"):
            url = endpoint + "/api/chat"
            payload = {
                "model": cfg.get("qwen_model", "qwen3-vl:4b-instruct"),
                "stream": False,
                "keep_alive": cfg.get("qwen_keep_alive", "5m"),          # 模型驻留内存时长；"-1"=常驻
                "num_parallel": cfg.get("qwen_num_parallel", 1),          # 并行处理请求数（OLLAMA_NUM_PARALLEL）
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg, "images": [b64]},
                ],
            }
        else:
            url = endpoint + "/v1/chat/completions"
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            payload = {
                "model": cfg.get("qwen_model", "qwen-vl-max"),
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": user_msg},
                    ]},
                ],
            }
        try:
            req = urllib.request.Request(url, data=_json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=cfg.get("qwen_timeout", 90)) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
            return ((body.get("message") or {}).get("content")
                    or (body.get("choices") or [{}])[0].get("message", {}).get("content")
                    or "")
        except Exception as e:
            print(f"qwen: 推理失败 {img_abs}: {e}")
            return ""

    # ------------------------------------------------------------------
    # L2 千问审核（设计文档 §6.2.2）：对 AI 预标注结果做中文语义二次审核
    # 输入：图片 + 预标注框（含 box_id）+ 标签字典（zh+desc 作为"审核法典"）
    # 输出：结构化 JSON —— box_reviews（keep/modify/reject/split + tags）
    #       + missed_targets（漏标补框）+ stats 统计；前端据此渲染绿/黄/红/蓝色虚线框
    # ------------------------------------------------------------------
    def qwen_audit_detections(self, img_abs: Path, dets: list,
                              labels_desc: Optional[dict], cfg: dict,
                              classes: list) -> tuple:
        """对一张图的已有检测框做千问 L2 审核。

        返回 (audit_dict | None, message)：
        - qwen_enabled=False → (None, 提示文案)
        - qwen_mock=True → 返回模拟审核结果（便于本地验证前端渲染链路）
        - 真实调用：按 §6.2.2 System Prompt 模板注入 zh+desc 审核法典与预标注框 JSON
        audit_dict: {"ok", "box_reviews":[{box_id, action, reason, tags, kind, suggested_class_id}],
                     "missed_targets":[{class_id, x1,y1,x2,y2, reason}], "stats", "box_count"}
        """
        if not cfg.get("qwen_enabled", False):
            return None, "千问 VL 未启用（AI 标注模型设置 → 启用千问）。启用后，AI 预标注会自动附加 L2 审核。"
        if not dets:
            return None, "没有可审核的框"

        # 统一给每个框分配 box_id（调用方已赋值则沿用），作为审核结果回写的索引
        audit_dets = []
        for i, b in enumerate(dets):
            bid = b.get("box_id")
            if bid is None:
                bid = i
                b["box_id"] = i
            audit_dets.append({
                "box_id": bid,
                "class_id": b.get("class_id"),
                "bbox": [round(float(b["x1"]), 1), round(float(b["y1"]), 1),
                         round(float(b["x2"]), 1), round(float(b["y2"]), 1)],
                "score": round(float(b.get("score") or b.get("confidence") or 0.9), 3),
            })

        if cfg.get("qwen_mock", False):
            print(f"qwen audit IN-MOCK: {len(audit_dets)} boxes for {img_abs}（本地联调模式，未调用真实模型）")
            return self._mock_qwen_audit(audit_dets, classes), ""

        labels = (labels_desc or {}).get("labels") or []
        defs = []
        for i, c in enumerate(classes):
            lab = labels[i] if i < len(labels) else None
            zh = (lab or {}).get("chinese_name") or c
            desc = (lab or {}).get("chinese_desc") or ""
            defs.append(f"{i}. {zh}{('：' + desc) if desc else ''}")
        labels_text = "\n".join(defs)
        dets_text = json.dumps(audit_dets, ensure_ascii=False)
        sys_msg = (
            "# Role\n"
            "你是资深的工业视觉数据审核专家。请对当前图片及 AI 预标注的边界框进行严苛的二次审核。\n\n"
            "# Context\n"
            f"- 目标类别定义（含审核标准）：\n{labels_text}\n"
            f"- 预标注结果：{dets_text}\n"
            "（JSON 格式：box_id 唯一标识每个框，bbox 为图片像素坐标 [x1,y1,x2,y2]，class_id 对应上面的类别序号）\n\n"
            "# Audit Dimensions（审核维度）\n"
            "请严格执行以下检查：\n"
            "1. 语义准确性：框内物体是否与类别定义（desc）严格匹配？输出 keep / modify / reject。\n"
            "2. 标签错误检测：框内物体是否被错误分类？若发现标签错误，输出 reject 并给出 suggested_class（建议的正确类别序号）。\n"
            "3. 漏标排查：仔细扫描全图，是否还有符合类别定义的目标没有被框出来？输出到 missed_targets，bbox 为像素坐标。\n"
            "4. 粘连框检测：是否有框内包含两个或多个独立目标？输出到 merged_boxes，action=split 或 reject。\n"
            "5. 边界质量：框是否过大、过小或未贴合边缘？过大/过小输出 modify。\n"
            "6. 业务合规：目标是否存在违规行为（如未戴安全帽、未穿反光衣）？在对应框的 tags 中加入 #compliance_issue。\n\n"
            "# Output Format\n"
            "仅返回标准 JSON，不要包含任何解释：\n"
            "{\n"
            "  \"missed_targets\": [\n"
            "    {\"class_id\": 序号, \"bbox\": [x1, y1, x2, y2], \"reason\": \"漏标原因（限20字）\"}\n"
            "  ],\n"
            "  \"merged_boxes\": [\n"
            "    {\"box_id\": 1, \"action\": \"split | reject\", \"reason\": \"包含两个独立目标（限20字）\"}\n"
            "  ],\n"
            "  \"label_errors\": [\n"
            "    {\"box_id\": 3, \"suggested_class\": 正确类别序号, \"reason\": \"框内实际为XXX而非当前标签（限20字）\"}\n"
            "  ],\n"
            "  \"box_reviews\": [\n"
            "    {\"box_id\": 2, \"action\": \"keep | modify | reject\", \"reason\": \"审核理由（限20字）\", \"tags\": [\"#occluded\", \"#blur\"]}\n"
            "  ]\n"
            "}"
        )
        user_msg = f"类别定义：\n{labels_text}\n\n预标注框：\n{dets_text}\n请审核这张图片。"
        content = self._qwen_image_chat(cfg, sys_msg, user_msg, img_abs)
        if not content:
            return None, "千问审核推理失败（无输出）"
        return self._parse_qwen_audit(content, audit_dets, labels, classes), ""

    def _parse_qwen_audit(self, content: str, audit_dets: list, labels: list, classes: list) -> dict:
        """解析千问审核输出 JSON → 归一化 audit 结构（容错：模型可能加 ```json 包裹或解释文字）"""
        import re
        m = re.search(r"\{[\s\S]*\}", content)
        if not m:
            return {"ok": False, "error": "千问审核输出不是标准 JSON", "raw": content[:500]}
        try:
            data = json.loads(m.group(0))
        except Exception:
            return {"ok": False, "error": "千问审核输出 JSON 解析失败", "raw": m.group(0)[:500]}
        if not isinstance(data, dict):
            return {"ok": False, "error": "千问审核输出格式异常", "raw": content[:300]}

        def cls_to_id(val) -> Optional[int]:
            """类别名（中文名/英文 code/序号）→ class_id；无法映射返回 None"""
            if val is None:
                return None
            if isinstance(val, bool):
                return None
            if isinstance(val, int):
                return val if 0 <= val < len(classes) else None
            s = str(val).strip()
            if s.lstrip("-").isdigit():
                i = int(s)
                return i if 0 <= i < len(classes) else None
            for i, c in enumerate(classes):
                lab = labels[i] if i < len(labels) else {}
                zh = str(lab.get("chinese_name") or "").strip()
                if zh and zh == s:
                    return i
                if str(c).strip().lower() == s.lower():
                    return i
            return None

        ids = {d["box_id"] for d in audit_dets}
        box_reviews: List[dict] = []
        # ① box_reviews：逐框审核
        for r in (data.get("box_reviews") or []):
            try:
                bid = int(r.get("box_id"))
            except (TypeError, ValueError):
                continue
            if bid not in ids:
                continue
            box_reviews.append({
                "box_id": bid,
                "action": (str(r.get("action") or "keep").lower() if str(r.get("action") or "").strip() else "keep"),
                "kind": "review",
                "reason": (str(r.get("reason") or ""))[:50],
                "tags": [str(t) for t in (r.get("tags") or []) if str(t).strip()],
                "suggested_class_id": None,
            })
        # ② label_errors：标签错误 → 补录/更新 reject + 建议类别
        for r in (data.get("label_errors") or []):
            try:
                bid = int(r.get("box_id"))
            except (TypeError, ValueError):
                continue
            if bid not in ids:
                continue
            sug = cls_to_id(r.get("suggested_class"))
            found = next((b for b in box_reviews if b["box_id"] == bid), None)
            if found is None:
                box_reviews.append({
                    "box_id": bid, "action": "reject", "kind": "label_error",
                    "reason": (str(r.get("reason") or ""))[:50], "tags": [],
                    "suggested_class_id": sug,
                })
            else:
                found["kind"] = "label_error"
                found["action"] = "reject"
                found["suggested_class_id"] = sug
                if not found["reason"]:
                    found["reason"] = (str(r.get("reason") or ""))[:50]
        # ③ merged_boxes：粘连框 → 补录/更新 split/reject
        for r in (data.get("merged_boxes") or []):
            try:
                bid = int(r.get("box_id"))
            except (TypeError, ValueError):
                continue
            if bid not in ids:
                continue
            act = str(r.get("action") or "split").lower()
            found = next((b for b in box_reviews if b["box_id"] == bid), None)
            if found is None:
                box_reviews.append({
                    "box_id": bid, "action": act if act in ("split", "reject") else "split",
                    "kind": "merged", "reason": (str(r.get("reason") or ""))[:50], "tags": [],
                    "suggested_class_id": None,
                })
            else:
                found["kind"] = "merged"
                if act in ("split", "reject"):
                    found["action"] = act
        # ④ missed_targets：漏标补框
        missed = []
        for mt in (data.get("missed_targets") or []):
            cid = cls_to_id(mt.get("class_id") if "class_id" in mt else mt.get("class"))
            bb = mt.get("bbox") or []
            if cid is None or not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in bb)
            except (TypeError, ValueError):
                continue
            missed.append({
                "class_id": cid, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "reason": (str(mt.get("reason") or ""))[:50],
            })
        stats = {"keep": 0, "modify": 0, "reject": 0, "split": 0, "missed": len(missed)}
        for b in box_reviews:
            stats[b["action"]] = stats.get(b["action"], 0) + 1
        return {
            "ok": True,
            "box_reviews": box_reviews,
            "missed_targets": missed,
            "stats": stats,
            "box_count": len(audit_dets),
        }

    def _mock_qwen_audit(self, audit_dets: list, classes: list) -> dict:
        """本地联调用模拟审核：便于验证前端渲染链路（不真实调用大模型）"""
        box_reviews = []
        for i, d in enumerate(audit_dets):
            bid = d["box_id"]
            if i == 0:
                box_reviews.append({"box_id": bid, "action": "keep", "kind": "review",
                                    "tags": [], "reason": "模拟：框贴合度良好", "suggested_class_id": None})
            elif i == 1:
                box_reviews.append({"box_id": bid, "action": "reject", "kind": "label_error",
                                    "suggested_class_id": (0 % len(classes)) if classes else None,
                                    "tags": [], "reason": "模拟：框中非目标物"})
            elif i == 2:
                box_reviews.append({"box_id": bid, "action": "modify", "kind": "review",
                                    "tags": ["#occluded"], "reason": "模拟：存在遮挡",
                                    "suggested_class_id": None})
            else:
                box_reviews.append({"box_id": bid, "action": "keep", "kind": "review",
                                    "tags": [], "reason": "", "suggested_class_id": None})
        missed = []
        if audit_dets and len(classes) > 1:
            missed.append({"class_id": 1 % len(classes), "x1": 20, "y1": 20, "x2": 90, "y2": 110,
                           "reason": "模拟：漏标一个目标"})
        stats = {"keep": 0, "modify": 0, "reject": 0, "split": 0, "missed": len(missed)}
        for b in box_reviews:
            stats[b["action"]] = stats.get(b["action"], 0) + 1
        return {
            "ok": True, "box_reviews": box_reviews, "missed_targets": missed,
            "stats": stats, "box_count": len(audit_dets),
        }

    def _task_classes(self, task_id: str) -> list:
        """读取标注任务注册的类别清单（英文 code，与 labels_dict/labels 序号对齐）"""
        try:
            meta = _load_json(settings.ANNOTATIONS_DIR / task_id / "task.json")
            return list(meta.get("classes") or [])
        except Exception:
            return []

    async def audit_image(self, task_id: str, image_id: str,
                          boxes: Optional[list] = None) -> dict:
        """对单张图当前标注框做千问 L2 审核（设计文档 §6.2.2 / §6.3 前端渲染）。

        boxes 由前端传入当前画布上的框（未保存也可审）；缺省时读取已保存标注。
        返回 {"image_id", "width", "height", "audit", "message"}。
        """
        cfg = self._read_config()
        item = await self._get_item(task_id, image_id)
        if not item:
            raise ValueError("Image not found in task")
        img_abs = settings.DATA_DIR / item["image_path"]
        if not img_abs.exists():
            raise ValueError(f"Image file not found: {img_abs}")
        if boxes is None:
            existing = await self.annotation_service.get_image_annotation(task_id, image_id)
            boxes = (existing or {}).get("boxes") or []
        else:
            # 分割目标可以只带轮廓点（矩形由轮廓现算），不能按"有没有矩形"筛掉
            boxes = [b for b in boxes if b.get("x1") is not None or b.get("polygon")]
        if not boxes:
            return {
                "image_id": image_id, "width": item["width"], "height": item["height"],
                "audit": None, "message": "当前图片没有标注框，无法审核",
            }
        classes = self._task_classes(task_id)
        labels_desc = self._task_labels_desc(task_id)
        if cfg.get("qwen_enabled", False):
            if labels_desc is None:
                labels_desc = {"labels": []}
        audit, message = await asyncio.to_thread(
            self.qwen_audit_detections, img_abs, boxes, labels_desc, cfg, classes
        )
        return {
            "image_id": image_id, "width": item["width"], "height": item["height"],
            "audit": audit, "message": message,
        }

    async def list_audit_issues(self, task_id: str) -> dict:
        """扫描任务全部图片的已保存标注，返回带审核异议（reject/modify/split）的图片列表。

        配合"审核落盘"：标注员不必逐张翻图，即可定位需要人工复核的图片。
        未标注/无异议的图片不计入。
        """
        task_meta = await self._load_task_meta(task_id)
        if not task_meta:
            raise ValueError("Task not found")
        dataset_id = task_meta.get("dataset_id", "")
        version = task_meta.get("version", "v1")

        def _scan():
            issues = []
            for item in task_meta.get("items", []):
                image_id = item["image_id"]
                ann = self.annotation_service._load_image_ann(dataset_id, version, image_id)
                cnt = {"reject": 0, "modify": 0, "split": 0}
                for b in (ann.get("boxes") or []):
                    act = (b.get("audit") or {}).get("action")
                    if act in cnt:
                        cnt[act] += 1
                total = cnt["reject"] + cnt["modify"] + cnt["split"]
                if total:
                    issues.append({
                        "image_id": image_id,
                        "image_path": item["image_path"],
                        "width": item.get("width"),
                        "height": item.get("height"),
                        "issue_count": total,
                        "actions": cnt,
                    })
            issues.sort(key=lambda x: x["issue_count"], reverse=True)
            return issues

        issues = await asyncio.to_thread(_scan)
        return {"task_id": task_id, "total": len(issues), "issues": issues}

    # ------------------------------------------------------------------
    # 标签建议：AI 识别图片中「已知标签之外」的新类别（辅助不懂标签的用户）
    # ------------------------------------------------------------------
    def qwen_suggest_labels(self, image_paths: list, known_codes: list, cfg: dict = None) -> tuple:
        """用千问 VL 识别图片中已知类别之外的新目标，给出标签候选（四字段）。

        返回 (candidates, message)：
        - qwen_enabled=False：返回 (None, 提示文案)，不调用真实模型（本地默认）
        - qwen_mock=True：返回预设模拟候选，便于本地验证链路
        - 真实调用：多张图聚合，按 english_code 去重并统计命中图数
        candidates: [{"english_code","chinese_name","chinese_desc","images":[路径]}]
        对已 known 类别的大小写不敏感过滤，空 english_code 丢弃。
        """
        cfg = cfg or self._read_config()
        if not cfg.get("qwen_enabled", False):
            return None, (
                "千问 VL 未启用（SAM 预标注设置 → 检测引擎里勾选「千问 VL」）。"
                "启用后即可自动识别图片中的新标签候选。"
            )
        known = {str(c).strip().lower() for c in (known_codes or []) if str(c).strip()}

        if cfg.get("qwen_mock", False):
            mock = [
                {"english_code": "helmet", "chinese_name": "安全帽", "chinese_desc": "人员头部佩戴的安全帽",
                 "ai_prompt_en": "safety helmet on worker head",
                 "images": list(image_paths[:2])},
                {"english_code": "traffic_light", "chinese_name": "交通信号灯", "chinese_desc": "路口红绿灯灯杆上的信号灯",
                 "ai_prompt_en": "traffic light on pole",
                 "images": list(image_paths[:1])},
            ]
            print("qwen suggest IN-MOCK：返回模拟新类别候选（未调用真实模型）")
            return [m for m in mock if m["english_code"].lower() not in known], ""

        import base64, io, re, json as _json
        from PIL import Image as PILImage
        import urllib.request
        sys_msg = (
            "你是一个视觉数据标注助手。给定图片与已知类别清单（含中文名及描述），找出图片中出现的、"
            "**不属于已知类别清单** 的物体类别，为每个新类别给出规范化的标签建议。\n"
            "输出严格为 JSON 数组（不要任何解释文字）：\n"
            '[{"english_code": "唯一英文标识(小写字母数字下划线)", "chinese_name": "中文名", '
            '"chinese_desc": "一句中文描述", "ai_prompt_en": "用于零样本检测的英文提示词，'
            '描述该物体在画面里的外观，如 wet reflective puddle on the ground"}]'
            "\n如果没有任何已知类别之外的新目标，输出 []。"
        )
        endpoint = (cfg.get("qwen_endpoint") or "http://localhost:11434").rstrip("/")
        api_key = cfg.get("qwen_api_key") or ""
        
        # ✅ 增强：将已知类别加入中文名+描述，帮助模型更好理解哪些类别已存在
        known_texts = []
        for c in (known_codes or []):
            cn = str(c).strip()
            if cn:
                known_texts.append(cn)
        known_text = "; ".join(known_texts) if known_texts else "（无已知类别）"
        user_msg = f"已知类别清单：{known_text}\n请识别这张图片中已知类别之外的新目标类别。"
        agg = {}  # code -> {meta, images}
        for img_abs in image_paths:
            try:
                image = PILImage.open(img_abs).convert("RGB")
                max_side = 1024
                w, h = image.size
                if max(w, h) > max_side:
                    scale = max_side / max(w, h)
                    image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=88)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception as e:
                print(f"qwen suggest: 图片编码失败 {img_abs}: {e}")
                continue
            data_url = f"data:image/jpeg;base64,{b64}"
            url, payload, headers = None, None, {"Content-Type": "application/json"}
            if "ollama" in (cfg.get("qwen_backend") or "").lower() or endpoint.startswith("http://localhost:1143"):
                url = endpoint + "/api/chat"
                payload = {
                    "model": cfg.get("qwen_model", "qwen3-vl:4b-instruct"),
                    "stream": False,
                    "keep_alive": cfg.get("qwen_keep_alive", "5m"),          # 模型驻留内存时长；"-1"=常驻
                    "num_parallel": cfg.get("qwen_num_parallel", 1),          # 并行处理请求数（OLLAMA_NUM_PARALLEL）
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg, "images": [b64]},
                    ],
                }
            else:
                url = endpoint + "/v1/chat/completions"
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                payload = {
                    "model": cfg.get("qwen_model", "qwen-vl-max"),
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": user_msg},
                        ]},
                    ],
                }
            try:
                req = urllib.request.Request(url, data=_json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=cfg.get("qwen_timeout", 90)) as resp:
                    body = _json.loads(resp.read().decode("utf-8"))
                content = ((body.get("message") or {}).get("content")
                           or (body.get("choices") or [{}])[0].get("message", {}).get("content")
                           or "")
            except Exception as e:
                print(f"qwen suggest: 推理失败 {img_abs}: {e}")
                continue
            # 提取首个 JSON 数组
            m = re.search(r"\[[\s\S]*\]", content)
            if not m:
                continue
            try:
                items = _json.loads(m.group(0))
            except Exception:
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                code = (it.get("english_code") or "").strip().lower()
                if not code or code in known:
                    continue
                cand = agg.setdefault(code, {
                    "english_code": it.get("english_code", "").strip(),
                    "chinese_name": (it.get("chinese_name") or "").strip(),
                    "chinese_desc": (it.get("chinese_desc") or "").strip(),
                    # 顺手带上英文检测提示词，采纳进字典后 GroundingDINO / YOLO-World 直接可用
                    "ai_prompt_en": (it.get("ai_prompt_en") or "").strip(),
                    "images": [],
                })
                if img_abs not in cand["images"]:
                    cand["images"].append(img_abs)
        return list(agg.values()), ""

    def _detect_grounding_dino(self, img_abs, classes, conf, cfg, prompts=None) -> list:
        """GroundingDINO（Transformers 集成）文本驱动检测。

        caption 用英文提示词（标签字典 ai_prompt_en 优先），与 classes 一一对应；
        通过 phrases -> prompt 文本匹配映射回 classes 索引得到 class_id。
        """
        processor, model = self._load_grounding_dino(cfg)
        # caption 用英文提示词：优先标签字典的 ai_prompt_en，其次前端 prompts，最后原类别名
        text_prompts = self._prompts_with_dict(classes, prompts, cfg)
        if not text_prompts:
            return []
        # GD 的 post_process_grounded_object_detection 要求短语以 "." 分隔且以句点结尾，
        # 否则结果解析为空（单类别时 join 无句点会返回 0 个目标）
        caption = " . ".join(text_prompts) + " ."
        device = self._resolve_device(cfg)
        try:
            import torch
            from PIL import Image
            image = Image.open(img_abs).convert("RGB")
            gd_size = int(cfg.get("grounding_dino_size", 640) or 640)
            # GD processor 的 size 需同时给定长短边（int 会被解析成仅含 shortest_edge 的非法字典）
            inputs = processor(
                images=image, text=caption,
                size={"shortest_edge": gd_size, "longest_edge": gd_size},
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # GD 分数尺度与 YOLO 不同：过低阈值会产生大量误检框，加上限下限保护
            gd_conf = max(float(conf), float(cfg.get("grounding_dino_min_conf", 0.25)))
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=gd_conf,
                text_threshold=gd_conf,
                target_sizes=[(image.height, image.width)],
            )[0]
        except Exception as e:
            # 不静默吞掉：模型未下载/加载失败时若返回空数组，用户会误以为"类别设置错误/图里没目标"
            raise RuntimeError(
                f"GroundingDINO 检测器不可用：{e}\n"
                "请先在右上角检测模型下拉切到 YOLO-World（如 yolov8s-world.pt，本地已就绪），"
                "或确认已下载 IDEA-Research/grounding-dino-tiny 模型。"
            ) from e

        out = []
        boxes = results.get("boxes")
        scores = results.get("scores")
        labels = results.get("labels")
        if boxes is None or scores is None or labels is None:
            return out
        for box, score, label in zip(boxes, scores, labels):
            class_id = self._match_phrase(text_prompts, label)
            if class_id is None:
                continue
            x1, y1, x2, y2 = box.tolist()
            out.append({
                "class_id": class_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "score": float(score),
            })
        return out

    def _match_phrase(self, prompts: List[str], label: str) -> Optional[int]:
        """将 GroundingDINO 返回的短语标签匹配回 prompts 索引（class_id）"""
        if label is None:
            return None
        norm_label = label.strip().lower().replace(".", "").strip()
        for i, p in enumerate(prompts):
            norm_p = str(p).strip().lower().replace(".", "").strip()
            if norm_p and (norm_p == norm_label or norm_label == norm_p):
                return i
        # 兜底：标签包含某个 prompt 的完整词
        for i, p in enumerate(prompts):
            norm_p = str(p).strip().lower().replace(".", "").strip()
            if norm_p and (norm_p in norm_label or norm_label in norm_p):
                return i
        return None

    @staticmethod
    def _mask_to_polygon(mask, sx: float, sy: float, img_w: float, img_h: float,
                         eps_px: float, max_points: int) -> list:
        """把 SAM 的二值掩码抽成一个多边形轮廓（原图像素坐标）。

        参数：
          mask      : 单物体掩码（True/1 表示前景），尺寸可能小于原图
          sx / sy   : 掩码坐标 → 原图坐标的缩放系数
          eps_px    : RDP 容差，单位＝**原图像素**（内部按 sx/sy 折算回掩码尺度）
          max_points: 顶点数上限
        返回 [[x, y], ...]；抽不出来（无 cv2、掩码太小、轮廓退化）时返回 []，
        调用方据此只保留矩形框，不影响原有检测标注行为。
        """
        try:
            import cv2
        except Exception:
            return []
        try:
            m = (mask > 0).astype(np.uint8)
            if m.shape[0] < 4 or m.shape[1] < 4:
                return []
            # 只取最外层轮廓；掩码被分割成多块时取最大那一块（其余通常是噪点）
            found = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = found[0] if len(found) == 2 else found[1]
            contours = [c for c in contours if c is not None and len(c) >= 3]
            if not contours:
                return []
            contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
            cnt = contours[0]
            area = float(cv2.contourArea(cnt))
            total = float(m.sum())
            # 最大块占比过低 → 轮廓不可信，宁可不给多边形（避免导出时把物体切没）
            if area <= 0 or total <= 0 or area / total < 0.6:
                return []
            # 容差按绝对像素给，再折算到掩码尺度（掩码可能比原图小，sx>1）
            approx = cv2.approxPolyDP(cnt, max(float(eps_px), 0.5) / max(min(sx, sy), 1e-6), True)
            pts = [p[0] for p in approx]
            # 简化后仍超上限：按步长等距抽稀
            if max_points and len(pts) > max_points >= 3:
                step = len(pts) / float(max_points)
                pts = [pts[int(i * step)] for i in range(max_points)]
            out = []
            for p in pts:
                x = min(max(float(p[0]) * sx, 0.0), img_w)
                y = min(max(float(p[1]) * sy, 0.0), img_h)
                out.append([round(x, 2), round(y, 2)])
            return out if len(out) >= 3 else []
        except Exception as e:
            print(f"Warning: mask to polygon failed: {e}")
            return []

    def _refine_with_sam(self, img_abs, det_boxes, cfg) -> list:
        """用 SAM 对候选框做分割精修：收紧外接矩形，并（默认）额外输出轮廓多边形。"""
        if not det_boxes:
            return det_boxes
        # sam_override_weak_seg=true：分割模型的自带轮廓（边界糙）也降级送 SAM 重描，
        # 模型只负责"锁定区域"，边界全部由 SAM 接管（积水 v1.1 管线验证有效的链路）
        override_seg = bool(cfg.get("sam_override_weak_seg", False))
        if override_seg:
            pending = list(range(len(det_boxes)))
        else:
            # 已经自带轮廓的框（本库分割模型学出来的边界）不再送进 SAM 重描：
            # 模型输出就是训练目标，让 SAM 再猜一次反而会把轮廓带偏，还白花时间。
            pending = [i for i, b in enumerate(det_boxes) if not b.get("polygon")]
        if not pending:
            return det_boxes
        sam = self._load_sam(cfg)
        if sam is None:
            return det_boxes

        boxes_np = np.array(
            [[det_boxes[i]["x1"], det_boxes[i]["y1"], det_boxes[i]["x2"], det_boxes[i]["y2"]]
             for i in pending],
            dtype=float,
        )
        try:
            results = sam.predict(
                img_abs,
                bboxes=boxes_np,
                verbose=False,
                imgsz=cfg.get("sam_imgsz", 1024),
                device=self._resolve_device(cfg),
            )
        except Exception as e:
            print(f"Warning: SAM refine failed, fallback to detect boxes: {e}")
            return det_boxes

        want_polygon = bool(cfg.get("sam_return_polygon", True))
        eps_px = float(cfg.get("sam_polygon_eps_px", 0.5) or 0.5)
        max_points = int(cfg.get("sam_polygon_max_points", 3000) or 3000)
        # SAM 重描去重：多个候选框落在同一片水面时，SAM 会把整片水重复描回每个框，
        # 产生一堆一模一样的多边形。按降采样 mask 重叠率（>0.7）判重，后到的丢弃。
        dedupe = override_seg and bool(cfg.get("sam_dedupe_overlap", 0.7))
        taken_small = []

        def _small(mask):
            import cv2
            m = (mask > 0).astype(np.uint8)
            return cv2.resize(m, (64, 64), interpolation=cv2.INTER_NEAREST) if m.shape != (64, 64) else m

        refined = list(det_boxes)
        dropped = set()
        for pos, idx in enumerate(pending):
            b = det_boxes[idx]
            mask = None
            orig_shape = None
            for r in results:
                if getattr(r, "masks", None) is not None and len(r.masks) > pos:
                    mask = r.masks.data[pos].cpu().numpy()
                    orig_shape = getattr(r, "orig_shape", None)
                    break
            if mask is not None:
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    if dedupe:
                        sm = _small(mask)
                        dup = False
                        for tsm in taken_small:
                            inter = np.logical_and(sm, tsm).sum()
                            if inter / max(1, min(sm.sum(), tsm.sum())) > float(cfg.get("sam_dedupe_overlap", 0.7)):
                                dup = True
                                break
                        if dup:
                            dropped.add(idx)
                            continue
                        taken_small.append(sm)
                    mh, mw = int(mask.shape[0]), int(mask.shape[1])
                    # 掩码一般是原图尺寸；若因 sam_imgsz 缩放导致不一致，按比例换算回原图
                    oh = float(orig_shape[0]) if orig_shape and len(orig_shape) >= 1 and orig_shape[0] else float(mh)
                    ow = float(orig_shape[1]) if orig_shape and len(orig_shape) >= 2 and orig_shape[1] else float(mw)
                    sx = ow / mw if mw > 0 else 1.0
                    sy = oh / mh if mh > 0 else 1.0
                    # 框内硬裁剪：SAM 的 box 提示是软提示，会沿视觉连贯性把框外的
                    # 相连湿地面/暗区一并描回（面积膨胀的主因）。这里把 mask 与检测框
                    # 求交，框外像素清零，保证精修结果不越过 jishui_v0 锁定的区域。
                    if bool(cfg.get("sam_clip_to_box", True)):
                        m2 = (mask > 0).astype(np.uint8)
                        bx1 = max(0, int(min(b["x1"], b["x2"]) / sx))
                        bx2 = min(mw, int(max(b["x1"], b["x2"]) / sx) + 1)
                        by1 = max(0, int(min(b["y1"], b["y2"]) / sy))
                        by2 = min(mh, int(max(b["y1"], b["y2"]) / sy) + 1)
                        keep = np.zeros_like(m2)
                        keep[by1:by2, bx1:bx2] = 1
                        mask = m2 * keep
                    item = {
                        "class_id": b["class_id"],
                        "x1": min(float(xs.min()) * sx, ow),
                        "y1": min(float(ys.min()) * sy, oh),
                        "x2": min(float(xs.max()) * sx, ow),
                        "y2": min(float(ys.max()) * sy, oh),
                        "score": b.get("score", 0.0),
                    }
                    if want_polygon:
                        if override_seg:
                            # v1.1 管线：等弧长重采样+共线抽稀（无长直边、顶点可拖拽），
                            # 需要把 mask 升到原图尺寸后交给 polygon_postprocess
                            try:
                                import cv2 as _cv2
                                from src.utils.polygon_postprocess import mask_to_polygon as _mpp
                                m_full = _cv2.resize((mask > 0).astype(np.uint8), (int(ow), int(oh)),
                                                     interpolation=_cv2.INTER_NEAREST)
                                # 预平滑：SAM 低分辨率 mask 放大后是台阶状锯齿，
                                # 抽稀守卫（偏离>1.2px）删不掉这些抖动点 → 顶点飙到 390+。
                                # 高斯模糊+二值化抹平台阶，顶点降到 70-100 且 IoU 达标。
                                blur_px = float(cfg.get("sam_mask_blur_px", 2.0) or 0)
                                if blur_px > 0:
                                    m_full = (_cv2.GaussianBlur(m_full.astype(np.float32), (0, 0), blur_px)
                                              > 0.5).astype(np.uint8)
                                # 边界腐蚀收敛：SAM 倾向把水边 5~10px 的过渡带（湿痕/亮斑边缘）
                                # 一并描进前景，整圈向内收 erode_px 像素，面积收敛到 ~85%。
                                erode_px = float(cfg.get("sam_mask_erode_px", 6.0) or 0)
                                if erode_px > 0:
                                    ker = _cv2.getStructuringElement(
                                        _cv2.MORPH_ELLIPSE,
                                        (int(2 * erode_px + 1), int(2 * erode_px + 1)))
                                    m_full = _cv2.erode(m_full, ker)
                                pts = _mpp(m_full)
                                if len(pts) >= 3:
                                    item["polygon"] = [[round(float(px), 2), round(float(py), 2)] for px, py in pts]
                            except Exception as e:
                                print(f"SAM refine: v1.1 抽稀失败，退回旧轮廓: {e}")
                                poly = self._mask_to_polygon(mask, sx, sy, ow, oh, eps_px, max_points)
                                if poly:
                                    item["polygon"] = poly
                        else:
                            poly = self._mask_to_polygon(mask, sx, sy, ow, oh, eps_px, max_points)
                            if poly:
                                item["polygon"] = poly
                    refined[idx] = item
                    continue
            refined[idx] = b
        if dropped:
            refined = [b for i, b in enumerate(refined) if i not in dropped]
        return refined

    # ------------------------------------------------------------------
    # 批量预标注（异步后台）
    # ------------------------------------------------------------------
    async def batch_auto_label(self, task_id: str, classes: List[str],
                               conf: Optional[float] = None,
                               prompts: Optional[List[str]] = None) -> dict:
        cfg = self._read_config()
        conf = conf if conf is not None else cfg.get("conf", 0.25)

        meta = await self._load_task_meta(task_id)
        if not meta:
            raise ValueError("Task not found")
        # 按 image_id 去重，与图片列表一致（历史任务可能因平铺图+划分目录重复收集同一张图）
        seen = set()
        items = []
        for it in meta.get("items", []):
            if it.get("image_id") in seen:
                continue
            seen.add(it.get("image_id"))
            items.append(it)
        if not items:
            raise ValueError("Task has no images")

        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        bt = BatchTask(batch_id, task_id, classes, conf)
        bt.prompts = prompts
        bt.total = len(items)
        self.batch_tasks[batch_id] = bt

        asyncio.create_task(self._run_batch(batch_id, items))
        return {"batch_id": batch_id, "total": bt.total}

    async def _run_batch(self, batch_id: str, items: list) -> None:
        bt = self.batch_tasks[batch_id]
        cfg = self._read_config()
        # 批量加速：默认关闭 SAM 精修（CPU 上 SAM 是每图最贵的一步，关掉速度翻倍以上；
        # 框贴合度略降，单图 AI 标注不受影响；sam_config.json 中 batch_sam_enabled 可开回）
        cfg["sam_enabled"] = cfg.get("batch_sam_enabled", False)
        # 标签字典：整批只解析一次复用（英文提示词给 GD/YOLO-World，中文描述给千问）
        ld = self._task_labels_desc(bt.task_id)
        if ld:
            cfg = dict(cfg)
            cfg["qwen_labels_desc"] = ld
        try:
            # 预加载模型，避免逐张重复实例化（CPU 上模型加载较慢，提前检查取消）
            if bt.cancelled:
                bt.status = "cancelled"
                return
            for det in self._active_detectors(cfg):
                await asyncio.to_thread(self._load_engine, det, cfg)
            if cfg.get("sam_enabled", True):
                await asyncio.to_thread(self._load_sam, cfg)
            # 模型加载完成后再次检查取消，避免加载期间点停止无效
            if bt.cancelled:
                bt.status = "cancelled"
                return

            # 续标恢复：已标注图片无需重新推理。先只读扫描标注文件（不触发模型），
            # 将已覆盖全部类别的图片计入 done —— 进度条从实际已完成数起步；
            # 主循环只处理未标完的图片，中断后重新启动即从断点续跑。
            remaining = []
            for item in items:
                if bt.cancelled:
                    bt.status = "cancelled"
                    return
                image_id = item["image_id"]
                try:
                    existing = await self.annotation_service.get_image_annotation(bt.task_id, image_id)
                    existing_boxes = existing.get("boxes") or [] if existing else []
                    existing_class_ids = {
                        b.get("class_id") for b in existing_boxes if b.get("class_id") is not None
                    }
                    if existing_class_ids.issuperset(range(len(bt.classes))):
                        bt.done += 1  # 所有类别的框都已标注（含 AI/人工）→ 计为已完成
                        continue
                except Exception:
                    pass  # 读取失败按未完成处理，交给主循环重试
                remaining.append(item)
            if bt.cancelled:
                bt.status = "cancelled"
                return

            for item in remaining:
                if bt.cancelled:
                    bt.status = "cancelled"
                    return
                image_id = item["image_id"]
                bt.current_image = image_id
                img_abs = settings.DATA_DIR / item["image_path"]

                # 单张图异常（如坏图、显存波动）只跳过该图，不中断整个批次
                try:
                    # 读取该图已有标注，按“缺失类别”增量标注，避免覆盖人工标注
                    existing = await self.annotation_service.get_image_annotation(bt.task_id, image_id)
                    existing_boxes = existing.get("boxes") or [] if existing else []
                    existing_class_ids = {b.get("class_id") for b in existing_boxes if b.get("class_id") is not None}
                    # 本次要标注的全部类别中，该图尚未标注的类别索引
                    missing_indices = [i for i in range(len(bt.classes)) if i not in existing_class_ids]
                    # 所有类别都已标注过，跳过该图
                    if not missing_indices:
                        bt.done += 1
                        continue

                    boxes = []
                    if img_abs.exists():
                        # 加锁串行化推理，避免与单图标注并发访问模型
                        async with self._lock:
                            boxes = await asyncio.to_thread(
                                self._run_auto_label_sync, img_abs, bt.classes, bt.conf, cfg, bt.prompts
                            )
                            # 兜底重试（缓解「AI 标不了导致人工补标压力大」）：
                            # 1) 检空 → 降置信度重试（提升召回，conf 0.25 → ~0.1）
                            # 2) 仍空 → 切换另一检测引擎重试（YOLO-World <-> GroundingDINO 双引擎）
                            if not boxes:
                                fallback_conf = max(0.05, round(bt.conf * 0.4, 3))
                                bt.retried += 1
                                try:
                                    boxes = await asyncio.to_thread(
                                        self._run_auto_label_sync, img_abs, bt.classes, fallback_conf, cfg, bt.prompts
                                    )
                                except Exception as e:
                                    print(f"批量预标注降阈重试失败 {image_id}: {e}")
                                if not boxes:
                                    alt_cfg = dict(cfg)
                                    alt_cfg["detector"] = "grounding_dino" if cfg.get("detector") != "grounding_dino" else "yolo_world"
                                    bt.skipped_candidates.append(image_id)
                                    try:
                                        boxes = await asyncio.to_thread(
                                            self._run_auto_label_sync, img_abs, bt.classes, fallback_conf, alt_cfg, bt.prompts
                                        )
                                        if boxes:
                                            bt.retried += 1
                                    except Exception as e:
                                        print(f"批量预标注换引擎重试失败 {image_id}: {e}")
                                # 兜底重试③：本地引擎均检空且千问启用时，交给 VLM 再试（缓解"标不了"）
                                if not boxes and cfg.get("qwen_enabled", False):
                                    qwen_cfg = dict(cfg)
                                    qwen_cfg["detector"] = "qwen_vl"
                                    try:
                                        boxes = await asyncio.to_thread(
                                            self._run_auto_label_sync, img_abs, bt.classes, fallback_conf, qwen_cfg, bt.prompts
                                        )
                                        if boxes:
                                            bt.retried += 1
                                    except Exception as e:
                                        print(f"批量预标注千问重试失败 {image_id}: {e}")
                    # 只保留缺失类别的检测框（class_id 为全局索引，与 bt.classes 对齐）
                    missing_set = set(missing_indices)
                    new_boxes = [b for b in boxes if b["class_id"] in missing_set]
                    # L2 千问审核（设计文档 §6.2/§6.3）：每张图检测后自动二次审核，
                    # 只统计汇总（keep/modify/reject/split/漏标）供批量进度展示，结论不落盘；
                    # 失败仅计数，不影响批量主流程
                    if new_boxes and cfg.get("qwen_enabled", False):
                        for i, b in enumerate(new_boxes):
                            b["box_id"] = i
                        try:
                            audit, audit_msg = await asyncio.to_thread(
                                self.qwen_audit_detections, img_abs, new_boxes,
                                cfg.get("qwen_labels_desc"), cfg, bt.classes
                            )
                            if audit and audit.get("ok"):
                                st = audit.get("stats") or {}
                                bt.audit_checked += 1
                                bt.audit_boxes += len(new_boxes)
                                for k in ("keep", "modify", "reject", "split", "missed"):
                                    bt.audit_summary[k] = bt.audit_summary.get(k, 0) + (st.get(k) or 0)
                                # 审核结论映射回本图 AI 新增框（落盘，供"问题图片检索"；box_id 不落盘）
                                rmap = {r.get("box_id"): r for r in (audit.get("box_reviews") or [])}
                                for i, b in enumerate(new_boxes):
                                    r = rmap.get(i)
                                    if r:
                                        b["audit"] = {k: r.get(k) for k in
                                                      ("action", "kind", "reason", "tags", "suggested_class_id")
                                                      if r.get(k) is not None}
                            else:
                                bt.audit_errors += 1
                        except Exception as e:
                            bt.audit_errors += 1
                            print(f"批量预标注 L2 审核失败 {image_id}: {e}")
                    if new_boxes:
                        clean_existing = [
                            {"class_id": b.get("class_id"),
                             "score": b.get("score") or b.get("confidence"),
                             "source": b.get("source", "manual"),
                             # 已有轮廓必须带上：否则 AI 再标一次会把人工画好的分割轮廓抹掉
                             **({"polygon": b["polygon"]} if b.get("polygon") else {}),
                             # 分割目标数据里没有矩形（只存轮廓点），有才带上，避免 None 落盘
                             **({k: b[k] for k in ("x1", "y1", "x2", "y2") if b.get(k) is not None})}
                            for b in existing_boxes
                            if b.get("x1") is not None or b.get("polygon")
                        ]
                        clean_new = [
                            {"class_id": b["class_id"], "x1": b["x1"], "y1": b["y1"],
                             "x2": b["x2"], "y2": b["y2"],
                             "score": b.get("score") or b.get("confidence"),
                             "source": "ai",
                             **({"polygon": b["polygon"]} if b.get("polygon") else {}),
                             **({"audit": b["audit"]} if b.get("audit") else {})}
                            for b in new_boxes
                        ]
                        await self.annotation_service.save_annotation(bt.task_id, image_id, clean_existing + clean_new, ai_annotated=True)
                        bt.boxes_written += 1
                        bt.annotated_images.append(image_id)
                    elif not any(b.get("x1") is not None or b.get("polygon") for b in existing_boxes):
                        # AI 兜底重试后仍检空且该图无任何标注：标记为困难样本（ai_miss），
                        # 供人工重点审核（空白/难例由审核者定夺），避免被静默跳过
                        try:
                            await self.annotation_service.save_annotation(
                                bt.task_id, image_id, [], ai_annotated=True,
                                sample_type=self.annotation_service.SAMPLE_HARD,
                                sample_reason="ai_miss")
                        except Exception as e:
                            print(f"批量预标注标记困难样本失败 {image_id}: {e}")
                    bt.done += 1
                except Exception as e:
                    print(f"批量预标注跳过图片 {image_id}: {e}")
                    bt.done += 1
                    continue

            bt.status = "done"
            bt.result_summary = {"total": bt.total, "annotated": bt.boxes_written}
        except Exception as e:
            bt.status = "error"
            bt.error = str(e)

    async def get_batch_status(self, batch_id: str) -> Optional[dict]:
        bt = self.batch_tasks.get(batch_id)
        if not bt:
            return None
        return {
            "batch_id": bt.batch_id,
            "task_id": bt.task_id,
            "status": bt.status,
            "total": bt.total,
            "done": bt.done,
            "boxes_written": bt.boxes_written,
            "annotated_images": bt.annotated_images,
            "retried": bt.retried,
            "skipped_candidates": bt.skipped_candidates,
            "current_image": bt.current_image,
            "error": bt.error,
            "summary": bt.result_summary,
            "audit": {
                "checked": bt.audit_checked,
                "boxes": bt.audit_boxes,
                "stats": bt.audit_summary,
                "errors": bt.audit_errors,
            } if bt.audit_checked else None,
        }

    async def stop_batch(self, batch_id: str) -> Optional[dict]:
        bt = self.batch_tasks.get(batch_id)
        if not bt:
            return None
        bt.cancelled = True
        return {"ok": True, "status": bt.status}

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    async def _load_task_meta(self, task_id: str) -> Optional[dict]:
        task_file = settings.ANNOTATIONS_DIR / task_id / "task.json"
        if not task_file.exists():
            return None
        return await asyncio.to_thread(_load_json, task_file)

    async def _get_item(self, task_id: str, image_id: str) -> Optional[dict]:
        meta = await self._load_task_meta(task_id)
        if not meta:
            return None
        for item in meta.get("items", []):
            if item["image_id"] == image_id:
                return item
        return None
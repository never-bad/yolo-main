#!/usr/bin/env python
"""
训练脚本：在独立进程中运行YOLO训练
"""
import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

def detect_device():
    """
    自动检测并选择最佳训练设备
    优先级：NVIDIA GPU (CUDA) > Apple M芯片 GPU (MPS) > CPU
    返回设备字符串，如 'cuda', 'mps', 'cpu'
    """
    try:
        import torch
        
        # 优先级1: 检测 NVIDIA GPU (CUDA)
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print(f"[{datetime.now().isoformat()}] 检测到 NVIDIA GPU: {device_name} (共 {device_count} 个设备)")
            print(f"[{datetime.now().isoformat()}] CUDA 版本: {torch.version.cuda}")
            return "cuda"
        
        # 优先级2: 检测 Apple M芯片 GPU (MPS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"[{datetime.now().isoformat()}] 检测到 Apple M芯片 GPU (Metal Performance Shaders)")
            return "mps"
        
        # 优先级3: 使用 CPU
        print(f"[{datetime.now().isoformat()}] 未检测到 GPU，将使用 CPU 进行训练")
        return "cpu"
        
    except ImportError:
        # 如果没有安装 torch，默认使用 CPU（YOLO 会自动处理）
        print(f"[{datetime.now().isoformat()}] 警告: 无法导入 torch，将使用默认设备")
        return None
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] 设备检测时出现错误: {e}，将使用默认设备")
        return None

def resolve_model_file(weights: str) -> str:
    """
    将基础模型名解析为本地文件路径。
    按顺序搜索：models/custom（自定义上传）、models/sam、backend 根目录（兼容旧文件）；
    找不到则原样返回，交给 ultralytics 在线下载兜底。
    """
    p = Path(weights)
    if p.is_absolute() and p.exists():
        return str(p)
    backend_dir = Path(__file__).resolve().parent.parent.parent
    for d in [backend_dir / "models" / "custom", backend_dir / "models" / "sam", backend_dir]:
        cand = d / weights
        if cand.exists():
            return str(cand)
    return weights


# 预训练权重下载镜像站（国内加速，顺序尝试；全部失败则交 ultralytics 内置直连兜底）
BASE_WEIGHT_MIRRORS = (
    "https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/{name}",
    "https://ghproxy.net/https://github.com/ultralytics/assets/releases/download/v0.0.0/{name}",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/{name}",
)


def ensure_local_base_weights(weights: str, timeout: int = 300) -> str:
    """将标准预训练权重（yolov8n.pt 等）本地化到 backend/models/custom，
    避免 ultralytics 训练启动时直连 GitHub 下载导致的长时间空窗（国内网络尤甚）。

    规则：
      - 已是本地存在的路径/带目录结构的路径 → 直接使用；
      - 裸文件名（标准预训练名）→ 依次从镜像站下载缓存；全部失败原样返回（由
        ultralytics 内置下载兜底）。
    """
    p = Path(weights)
    backend_dir = Path(__file__).resolve().parent.parent.parent
    custom_dir = backend_dir / "models" / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    # 持久化目录优先：models/custom 落在命名卷 docker_yolo-models，重建镜像不丢。
    # 不能让下面的 `p.exists()` 先命中——那是按"当前进程 cwd"（/app/backend，镜像内
    # 目录）判断的，一旦命中就直接返回裸文件名，权重永远存不进持久化目录，
    # 于是重建镜像后又要重新下载一遍（yolo11n.pt 每次都下的根因之一）。
    custom_target = custom_dir / weights
    if custom_target.exists():
        return str(custom_target)
    if p.exists() or p.suffix != ".pt" or "/" in weights or "\\" in weights:
        return weights
    import shutil
    import urllib.request
    target = custom_target
    for mirror in BASE_WEIGHT_MIRRORS:
        url = mirror.format(name=weights)
        tmp = target.with_suffix(".pt.tmp")
        try:
            print(f"[{datetime.now().isoformat()}] 本地无预训练权重 {weights}，"
                  f"正在从镜像下载（{url}）...")
            req = urllib.request.Request(
                url, headers={"User-Agent": "yolo-training-platform/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
            os.replace(tmp, target)
            print(f"[{datetime.now().isoformat()}] 预训练权重已缓存到本地: {target}")
            return str(target)
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] 预训练权重下载失败（{url}）: {e}")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
    return weights


def next_model_version(registry_dir: str, business: str, library_root_id: str = None) -> str:
    """计算下一个模型版本号：取已有版本号最大值 + 0.1。

    归属某算法库的产出按"库内已有版本"递增（同一库 v1.0 → v1.1 …，不受其它同业务库干扰）；
    无库归属的独立模型退回按业务线递增。
    """
    best = 0.99  # 与历史保持一致：库内无任何版本时首版为 v1.1
    registry = Path(registry_dir)
    if registry.exists():
        for model_dir in registry.iterdir():
            if not model_dir.is_dir():
                continue
            model_file = model_dir / "model.json"
            if not model_file.exists():
                continue
            try:
                with open(model_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if library_root_id:
                    if meta.get("library_root_id") != library_root_id:
                        continue
                elif meta.get("business") != business:
                    continue
                ver = meta.get("version", "")
                if isinstance(ver, str) and ver.startswith("v"):
                    try:
                        best = max(best, float(ver[1:]))
                    except ValueError:
                        pass
            except Exception:
                continue
    return f"v{best + 0.1:.1f}"


def normalize_class_names(names) -> list:
    """将 data.yaml 的 names（list 或 dict）归一化为按类 ID 排序的类别名列表"""
    if names is None:
        return []
    if isinstance(names, dict):
        max_id = max(int(k) for k in names.keys())
        return [str(names.get(str(i), names.get(i, f"class_{i}"))) for i in range(max_id + 1)]
    if isinstance(names, list):
        return [str(n) for n in names]
    return []


# ultralytics 内部会偷偷联网下载的权重（AMP 自检探针模型）。训练本身用不到它，
# 但它会拖垮启动：见 patch_amp_check_offline 的说明。
ULTRALYTICS_INTERNAL_WEIGHTS = ("yolo11n.pt",)

# 内部权重的最小合理体积（yolo11n.pt 完整约 5.4MB）。下载被中断产生的半截文件
# 通常只有几百 KB，torch.load 会报 "PytorchStreamReader failed reading zip
# archive"，据此识别并用持久化副本覆盖修复，避免"一次下残、永久报错"。
_MIN_INTERNAL_WEIGHT_BYTES = 1 * 1024 * 1024


def _weight_looks_valid(p: Path) -> bool:
    """粗检权重文件是否完整：存在且体积达标。

    不做 torch.load —— 启动阶段为了校验去加载 5MB 权重不划算，且真实损坏
    场景（下载中断）体积差距在数量级上，阈值足够可靠。
    """
    try:
        return p.is_file() and p.stat().st_size >= _MIN_INTERNAL_WEIGHT_BYTES
    except OSError:
        return False


def stage_internal_weights_to_cwd():
    """把 ultralytics 内部会下载的权重预置到训练进程工作目录（cwd）。

    为什么需要：平台的 `resolve_model_file` / `ensure_local_base_weights` 只管**训练指定
    的模型**，管不到 ultralytics 内部触发的 `YOLO("yolo11n.pt")`（AMP 自检）。后者只在
    **训练进程 cwd**（容器内 `/app/backend`，镜像内目录）找文件，找不到就直连 GitHub。

    于是形成"每次都下载"：即使下下来了也存在 `/app/backend` —— 那是**镜像内目录，不在
    命名卷里，重建镜像即丢**；而持久化的 `models/custom`（命名卷 `docker_yolo-models`）
    里的同名文件它又不会去找。

    做法：启动时检查 cwd 里的该文件——缺失**或体积异常小（下载中断留下的半截
    文件）**时，就从 `models/custom`（持久化）复制一份过来覆盖修复；两边都没有
    则跳过（此时由 patch_amp_check_offline 兜底，已不依赖联网）。
    """
    backend_dir = Path(__file__).resolve().parent.parent.parent
    custom_dir = backend_dir / "models" / "custom"
    for name in ULTRALYTICS_INTERNAL_WEIGHTS:
        in_cwd = backend_dir / name
        if _weight_looks_valid(in_cwd):
            continue
        src = custom_dir / name
        if _weight_looks_valid(src):
            # 用 exists() 而非体积判断，是历史的坑：cwd 里残留半截文件会被判为
            # "已存在"从而永远跳过修复，下次加载直接崩。故这里覆盖写。
            fixing = in_cwd.exists()
            try:
                import shutil
                shutil.copyfile(src, in_cwd)
                action = "修复损坏的" if fixing else "预置"
                print(f"[{datetime.now().isoformat()}] 已{action} {name} —— 从持久化目录 "
                      f"{custom_dir} 复制到工作目录（避免 ultralytics 内部联网下载）")
            except Exception as e:
                print(f"[{datetime.now().isoformat()}] 预置 {name} 失败（忽略）: {e}")


def patch_amp_check_offline():
    """让 ultralytics 的 AMP 自检不再联网下载 yolo11n.pt。

    背景（2026-10-28 真机事故）：ultralytics 的 ``check_amp()`` 内部会执行
    ``YOLO("yolo11n.pt")``，从 GitHub 拉取约 5.4MB 权重做 FP16/FP32 前向比对。
    弱网环境（实测 10~30KB/s，且 TLS 连接被反复重置、3 次重试全失败）下这一步
    能卡住十几分钟，期间 GPU 利用率恒为 0%，最终被 gpu_watch 熔断判定为
    "GPU 异常/训练卡死"而 SIGTERM 杀掉——训练一个 epoch 都没跑起来。

    改为按设备能力直接判定：CUDA / MPS 启用 AMP，CPU 关闭。跳过联网比对。
    本文件原有的 AMP 失败自动降级（[Errno 22] → amp=False）逻辑不受影响。
    """
    try:
        from ultralytics.utils import checks as _checks

        def _check_amp_by_device(model):
            try:
                dev = next(model.parameters()).device
            except Exception:
                return False
            return dev is not None and dev.type in ("cuda", "mps")

        _checks.check_amp = _check_amp_by_device
        # engine/trainer.py 顶部是 `from ..utils.checks import check_amp` 的模块级绑定，
        # 只改 checks 模块的属性对 trainer 无效，必须一并替换 trainer 里的引用
        try:
            from ultralytics.engine import trainer as _trainer
            if hasattr(_trainer, "check_amp"):
                _trainer.check_amp = _check_amp_by_device
        except Exception:
            pass
        print(f"[{datetime.now().isoformat()}] AMP 自检已改为离线判定"
              f"（跳过 yolo11n.pt 联网下载，避免弱网环境卡住启动）")
    except Exception as e:
        # 打补丁失败不阻塞训练：退回 ultralytics 原生逻辑（可能联网）
        print(f"[{datetime.now().isoformat()}] AMP 自检离线化失败（忽略，按原逻辑运行）: {e}")


def _torch_uuid_str(props) -> str | None:
    """把 torch 的 device uuid 归一化成 nvidia-smi 的 GPU-xxxx 形式。

    torch 侧 cuda:N 的序号与 nvidia-smi 的 index 在容器内可能不一致
    （受 CUDA_VISIBLE_DEVICES / 枚举序影响），只靠序号会把监控引向错误的卡
    （2026-10-28 B-16：配 gpu_index=1 的训练实际跑在 nvidia-smi 的 0 号卡上，
    监控盯着 1 号卡读 0% → 判定卡死误杀）。UUID 是唯一跨视角稳定的标识。
    转换失败返回 None，由调用方退回 name+显存容量指纹匹配。
    """
    u = getattr(props, "uuid", None)
    if u is None:
        return None
    if isinstance(u, str):
        return u if u.upper().startswith("GPU-") else None
    try:
        h = f"{int(u):032x}"
        return f"GPU-{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
    except Exception:
        return None


def _write_gpu_fingerprint(job_id: str, job_file: str, device: str):
    """把训练进程视角的 GPU 指纹写入 sidecar（{job_id}.gpu.json），供监控与排障。

    sidecar 与 job.json 同目录，train_service 侧可据此把 torch 视角的 cuda:N
    映射回 nvidia-smi 的真实 index，并在日志里留下可比对的两视角快照。
    任何异常都不影响训练本身。
    """
    try:
        import subprocess
        import torch
        from pathlib import Path as _P

        info = {
            "job_id": job_id,
            "torch_device": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_count": torch.cuda.device_count(),
            "written_at": datetime.now().isoformat(),
        }

        idx = None
        dev = str(device or "")
        if dev.startswith("cuda:") and dev.split(":")[1].isdigit():
            idx = int(dev.split(":")[1])
        if idx is not None and idx < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(idx)
            info.update({
                "torch_index": idx,
                "torch_name": props.name,
                "torch_total_mb": round(props.total_memory / (1024 ** 2)),
                "torch_uuid": _torch_uuid_str(props),
            })

        # 同一进程内再枚举一次 nvidia-smi 视角，用于精确匹配 + 排障留痕
        rows = []
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,uuid",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            for line in (out.stdout or "").strip().splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 4 and parts[0].isdigit():
                    try:
                        rows.append({
                            "nvml_index": int(parts[0]),
                            "name": parts[1],
                            "total_mb": int(float(parts[2])),
                            "uuid": parts[3],
                        })
                    except ValueError:
                        continue
        except Exception:
            pass
        info["nvml_devices"] = rows

        tuuid = (info.get("torch_uuid") or "").upper()
        if tuuid:
            for r in rows:
                if (r.get("uuid") or "").upper() == tuuid:
                    info["nvml_index"] = r["nvml_index"]
                    info["match_by"] = "uuid"
                    break
        if info.get("nvml_index") is None and info.get("torch_name"):
            cand = [r for r in rows
                    if r.get("name") == info["torch_name"]
                    and abs(r.get("total_mb", 0) - info.get("torch_total_mb", 0)) <= 64]
            if len(cand) == 1:
                info["nvml_index"] = cand[0]["nvml_index"]
                info["match_by"] = "name+memory"

        try:
            sidecar = _P(job_file).with_name(f"{job_id}.gpu.json")
            sidecar.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        # 日志留痕：两视角对照，一眼看出编号是否错位
        if info.get("torch_index") is not None:
            tv = (f"cuda:{info['torch_index']}={info.get('torch_name')}"
                  f"({round(info.get('torch_total_mb', 0) / 1024, 1)}G)")
        else:
            tv = str(device)
        nv = ", ".join(f"idx{r['nvml_index']}={r['name']}"
                       f"({round(r['total_mb'] / 1024, 1)}G)" for r in rows) or "(nvidia-smi 不可用)"
        print(f"[{datetime.now().isoformat()}] [GPU-DIAG] torch视角 {tv} | nvidia-smi视角 {nv}"
              f" | 映射 nvml_index={info.get('nvml_index')}"
              f"（依据 {info.get('match_by', '未匹配')}）"
              f" | CUDA_VISIBLE_DEVICES={info.get('cuda_visible_devices')}")
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] [GPU-DIAG] 指纹写入失败（忽略）: {e}")


def parse_architecture(base_model: str) -> dict:
    """M11：从 base_model 解析 architecture 三字段（写入模型元数据标准字段集）。

    支持 ultralytics 命名（yolov8s/yolo11n/yolov5m.pt 等）：
    - architecture_family: 算法家族（ultralytics）
    - architecture_name: 架构名（yolov8 / yolo11 / yolov5 ...）
    - model_scale: 规模（n / s / m / l / x，无法解析时为 ""）
    无法识别时三字段均返回 ""（不阻断训练）。
    """
    name = (base_model or "").strip()
    fname = name.split("/")[-1]                       # 去掉路径只取文件名
    fname = fname.replace(".pt", "").replace(".yaml", "").lower()
    import re
    # 任务后缀先剥掉：否则 "yolov8n-seg" 结尾是 g，规模（n/s/m）匹配不到，
    # 架构名会整串落进 architecture_name、model_scale 为空
    fname = re.sub(r"-(seg|pose|cls|obb|track)$", "", fname)
    m = re.match(r"^(.*?)([nsmlx])$", fname)
    if m and len(m.group(1)) > 1:
        arch, scale = m.group(1), m.group(2)
    else:
        arch, scale = fname, ""
    return {
        "architecture_family": "ultralytics",
        "architecture_name": arch or "",
        "model_scale": scale or "",
    }


def update_datasets_train_state(job_file: str, datasets_dir: str, stage: str,
                                training_status: str, trained_at: str = None):
    """阶段1.3：训练结果回写数据集状态机。

    - 每个参与训练的数据集（job_file.dataset_ids，兼容旧字段 dataset_id）：
        合格 → stage=completed / training_status=completed / trained_at / round+1；
        不合格/失败 → stage=failed / training_status=incomplete（保持未完成，参与下一轮雪球）。
    """
    try:
        if not datasets_dir:
            return
        with open(job_file, "r", encoding="utf-8") as f:
            jm = json.load(f)
        ds_ids = jm.get("dataset_ids") or ([jm.get("dataset_id")] if jm.get("dataset_id") else [])
        for ds_id in ds_ids:
            if not ds_id:
                continue
            meta_p = Path(datasets_dir) / str(ds_id) / "meta.json"
            if not meta_p.exists():
                continue
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta.setdefault("stage", "annotating")
            meta.setdefault("training_status", "incomplete")
            meta["stage"] = stage
            meta["training_status"] = training_status
            if trained_at:
                meta["trained_at"] = trained_at
                meta["last_trained_round"] = int(meta.get("last_trained_round", 0)) + 1
            with open(meta_p, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"[{datetime.now().isoformat()}] 数据集状态回写: {ds_id} → stage={stage}, training_status={training_status}")
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] 数据集状态回写失败（忽略）: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--job_file", required=True)
    parser.add_argument("--registry_dir", required=True)
    parser.add_argument("--datasets_dir", default=None, help="数据集根目录（训练结果回写数据集状态机用）")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    # 高级训练参数（可选，缺省则由 ultralytics 使用默认值）
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer: auto/SGD/Adam/AdamW")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping epochs (0=disabled)")
    # 训练节点：指定使用哪块 GPU（多卡环境；缺省=自动检测）
    parser.add_argument("--gpu", type=int, default=None, help="GPU index to use (0-based, default=auto)")
    # 数据加载进程数：0=单进程（容器/Docker 环境最稳），为空则自动判断
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers (0=single-process, safest in containers)")
    # 模型守门员：业务隔离 + 与上一代同业务生产模型对比
    parser.add_argument("--business", type=str, default="general", help="业务/算法类型（守门员按此隔离对比）")
    parser.add_argument("--baseline", type=str, default=None, help="上一代同业务生产模型 best.pt（守门员对比基准，可空=首版）")
    parser.add_argument("--baseline-model-id", type=str, default=None, help="上一代生产模型的 model_id（守门员对比基准，写入血缘）")
    parser.add_argument("--base-model-id", type=str, default=None,
                        help="真正微调父模型的 model_id（写入血缘 parent_model_id 用于展示；与守门员基线区分）")
    parser.add_argument("--library-root-id", type=str, default=None,
                        help="所属算法库根记录 model_id：产出版本写入该字段，库详情据此聚合版本")
    # 2.4 跨代迁移训练（训练模块规范 §3.2.2）：源/目标架构不同时 strict=False 加载
    parser.add_argument("--migration-target", type=str, default=None,
                        help="目标架构（如 yolov8s.pt）：跨代迁移以目标架构建模型，strict=False 复用源 Backbone 层")
    parser.add_argument("--migration-source", type=str, default=None,
                        help="迁移来源描述（如 model_xxx [yolov5s]），写入 [Migration] 日志与模型血缘")
    parser.add_argument("--migration-lr0-degraded", action="store_true",
                        help="lr0 已由后端自动降 10 倍（仅 [Migration] 日志标注）")
    
    args = parser.parse_args()
    
    print(f"[{datetime.now().isoformat()}] Starting training job: {args.job_id}")
    print(f"[{datetime.now().isoformat()}] Dataset: {args.data}")
    print(f"[{datetime.now().isoformat()}] Model: {args.model}")
    print(f"[{datetime.now().isoformat()}] Epochs: {args.epochs}")
    print(f"[{datetime.now().isoformat()}] Image size: {args.imgsz}")
    print(f"[{datetime.now().isoformat()}] Batch size: {args.batch}")
    print(f"[{datetime.now().isoformat()}] Resume: {args.resume}")
    print(f"[{datetime.now().isoformat()}] Advanced: lr0={args.lr0}, optimizer={args.optimizer}, weight_decay={args.weight_decay}, patience={args.patience}")
    
    try:
        # 环境加载提示：torch + ultralytics 首次导入较慢，先给用户可见的进度，避免误以为任务卡死
        print(f"[{datetime.now().isoformat()}] 正在加载深度学习环境（PyTorch / Ultralytics），"
              f"首次运行约需 10~30 秒，请耐心等待...")
        from ultralytics import YOLO
        patch_amp_check_offline()
        stage_internal_weights_to_cwd()
        
        # 选择训练设备：优先使用用户指定的 GPU 索引；否则自动检测
        if args.gpu is not None:
            try:
                import torch
                if torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count():
                    device = f"cuda:{args.gpu}"
                    print(f"[{datetime.now().isoformat()}] 指定训练节点: GPU {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
                else:
                    device = detect_device()
                    print(f"[{datetime.now().isoformat()}] 警告: 指定的 GPU {args.gpu} 不可用，将自动选择设备: {device}")
            except ImportError:
                device = None
                print(f"[{datetime.now().isoformat()}] 警告: 无法导入 torch，将使用默认设备")
        else:
            device = detect_device()
        if device:
            print(f"[{datetime.now().isoformat()}] 选择的训练设备: {device.upper()}")
        # B-16：写 GPU 指纹 sidecar（torch 视角 ↔ nvidia-smi 视角映射），供监控与排障
        _write_gpu_fingerprint(args.job_id, args.job_file, device)

        train_dir = Path(args.project) / args.name
        resume_path = None
        
        # 如果是恢复训练，使用 checkpoint 路径
        if args.resume:
            if args.model.endswith("last.pt") and Path(args.model).exists():
                resume_path = args.model
                print(f"[{datetime.now().isoformat()}] Resuming from checkpoint: {resume_path}")
            else:
                last_pt = train_dir / "weights" / "last.pt"
                if last_pt.exists():
                    resume_path = str(last_pt)
                    print(f"[{datetime.now().isoformat()}] Found checkpoint: {resume_path}")
                else:
                    raise FileNotFoundError(f"No checkpoint found at {last_pt}. Cannot resume training.")
        else:
            print(f"[{datetime.now().isoformat()}] Loading model: {args.model}")
            # 优先使用本地已有权重（models/custom、models/sam、backend 根目录），避免重复联网下载
            resume_path = resolve_model_file(args.model)
            # 标准预训练权重（yolov8n.pt 等）本地未缓存时，从国内镜像下载到 models/custom，
            # 避免 ultralytics 直连 GitHub 下载导致训练长时间卡在启动阶段
            resume_path = ensure_local_base_weights(resume_path)
        
        # 2.4 跨代迁移训练（训练模块规范 §3.2.2）：
        # 以"目标架构"新建模型（如 yolov8s.pt 预训练权重），再 strict=False 按 层名+形状 复用
        # 源模型（源 checkpoint）的权重层 —— Backbone 大部分层可复用，Head/形状不符层保持
        # 目标预训练初始化。lr0 已由后端自动降 10 倍（可 L3 覆盖），全程打 [Migration] 日志。
        migration = None
        if args.migration_target:
            migration = {
                "enabled": True,
                "source": args.migration_source or args.model,
                "target": args.migration_target,
                "lr0_degraded": args.migration_lr0_degraded,
            }
            print(f"[Migration] Starting cross-generation transfer: "
                  f"{migration['source']} → {migration['target']}")
            tgt_path = resolve_model_file(args.migration_target)
            tgt_path = ensure_local_base_weights(tgt_path)
            model = YOLO(tgt_path)
            try:
                import torch as _torch
                src_model = YOLO(resume_path)
                src_sd = src_model.model.state_dict()
                dst_sd = model.model.state_dict()
                loaded, reinit = 0, 0
                for k, v in dst_sd.items():
                    sv = src_sd.get(k)
                    if sv is not None and tuple(sv.shape) == tuple(v.shape):
                        v.copy_(sv.detach().to(v.device))
                        loaded += 1
                    else:
                        reinit += 1
                print(f"[Migration] Loaded {loaded}/{len(dst_sd)} layers from source checkpoint")
                if reinit:
                    print(f"[Migration] {reinit} layers re-initialized "
                          f"(detection head, anchor layers)")
                if args.migration_lr0_degraded:
                    print(f"[Migration] Auto-degraded lr0 to {args.lr0} "
                          f"(10x reduction for transfer learning)")
                del src_model, src_sd
                import gc as _gc
                _gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception as mig_e:
                print(f"[Migration] 源权重复用失败（{mig_e}），"
                      f"使用目标架构预训练权重继续训练")
            print("[Migration] Transfer complete. Proceeding with training.")
        else:
            model = YOLO(resume_path)
        
        # 使用传入的 batch size；-1/0(自动) 会触发 ultralytics 的 auto-batch 校准，
        # 在校准期多进程加载数据，容器环境下易报 [Errno 22] Invalid argument，这里兜底为确定值
        batch_size = args.batch
        if batch_size is None or batch_size < 1:
            batch_size = 32
        
        # 数据加载进程数：容器/Docker/Windows 使用单进程（workers=0）最稳，
        # 避免多进程 DataLoader 在 fork + CUDA + 共享内存环境下的 [Errno 22] / 卡死 / 崩溃
        if args.workers is not None:
            workers = args.workers
            print(f"[{datetime.now().isoformat()}] 数据加载进程数: workers={workers}（由用户指定）")
        elif os.name == 'nt' or os.path.exists('/.dockerenv') or (Path(__file__).resolve().parent.parent.parent / 'app').exists():
            workers = 0
            print(f"[{datetime.now().isoformat()}] 数据加载进程数: workers=0（容器/单进程模式，最稳定）")
        else:
            workers = 4
            print(f"[{datetime.now().isoformat()}] 数据加载进程数: workers=4")
        
        train_kwargs = {
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": batch_size,
            "workers": workers,  # 显式指定，避免默认多进程在容器内崩溃（[Errno 22]）
            "project": args.project,
            "name": args.name,
            "verbose": True,
        }
        
        # 如果检测到设备，显式指定设备
        if device:
            train_kwargs["device"] = device
        
        # 高级训练参数：仅在非 resume 场景下显式传入（resume 使用 checkpoint 中保存的参数）
        if not args.resume:
            advanced = [
                ("lr0", args.lr0),
                ("optimizer", args.optimizer),
                ("weight_decay", args.weight_decay),
                ("patience", args.patience),
            ]
            for key, val in advanced:
                # patience=0 / weight_decay=0 是合法值（禁用早停/无衰减），不能跳过
                if val is not None:
                    train_kwargs[key] = val
        
        # 开始训练：包一层自动降级——AMP(混合精度)检查在容器环境可能失败([Errno 22])，
        # 首次失败自动改用 FP32(amp=False) 重试，保证训练能继续跑而不是直接失败；
        # 失败后写全局标记文件：本环境之后的训练直接 FP32，跳过 AMP 自检与重复重试，
        # 避免每次任务都在启动阶段白白等待几十秒
        AMP_FLAG = "/tmp/ultralytics_amp_failed"

        def _run_training(kwargs):
            if args.resume:
                kwargs["resume"] = True
                print(f"[{datetime.now().isoformat()}] Resuming training from checkpoint...")
            else:
                print(f"[{datetime.now().isoformat()}] Starting new training...")
            return model.train(**kwargs)

        if os.path.exists(AMP_FLAG):
            train_kwargs["amp"] = False
            print(f"[{datetime.now().isoformat()}] 检测到本环境 AMP 检查历史失败，"
                  f"直接以 FP32 精度训练（跳过 AMP 自检，加快启动）")

        try:
            _run_training(dict(train_kwargs))
        except Exception as amp_e:
            err_text = str(amp_e).lower()
            if ("invalid argument" in err_text or "errno 22" in err_text or "amp" in err_text
                    or "autocast" in err_text or "c10::" in err_text):
                # 记录本环境 AMP 检查失败，后续训练直接跳过检查以加快启动
                try:
                    with open(AMP_FLAG, "w") as _f:
                        _f.write(datetime.now().isoformat())
                except Exception:
                    pass
                print(f"[{datetime.now().isoformat()}] AMP 检查未通过（{amp_e}），"
                      f"自动降级为 FP32 精度（amp=False）重新训练，避免任务直接失败...")
                train_kwargs["amp"] = False
                _run_training(train_kwargs)
            else:
                raise amp_e
        
        print(f"[{datetime.now().isoformat()}] Training completed!")
        
        # 训练完成，注册模型
        # N-5：同一秒内两个训练任务先后完成会算出同一个 model_id，再配 mkdir(exist_ok=True)
        # 就会静默覆盖前一个模型的记录（权重与 model.json 被改写）。目录已存在即退让到
        # _2、_3… 后缀，靠 mkdir 的原子性挡住并发（与 model_service._reserve_model_dir、
        # 数据集上传 P1-4 同一套格式）。本脚本是独立子进程，不引 src.* 依赖，故就地实现。
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        n = 1
        while True:
            model_id = f"model_{ts}" if n == 1 else f"model_{ts}_{n}"
            model_dir = Path(args.registry_dir) / model_id
            try:
                model_dir.mkdir(parents=True)
                break
            except FileExistsError:
                n += 1
        print(f"[{datetime.now().isoformat()}] Registered model_id: {model_id}")
        
        # 复制权重文件
        weights_dir = model_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        
        train_dir = Path(args.project) / args.name
        best_pt = train_dir / "weights" / "best.pt"
        
        # 如果默认路径不存在，尝试在 project 目录下查找所有可能的 best.pt
        if not best_pt.exists():
            project_path = Path(args.project)
            # 在 project 目录下递归查找 best.pt
            possible_best_pt = None
            for candidate in project_path.rglob("best.pt"):
                # 优先选择在 weights 目录下的
                if "weights" in str(candidate.parent):
                    possible_best_pt = candidate
                    break
            
            if possible_best_pt:
                print(f"[{datetime.now().isoformat()}] Found best.pt at: {possible_best_pt}")
                best_pt = possible_best_pt
            else:
                raise FileNotFoundError(
                    f"Model weights (best.pt) not found. Expected at: {train_dir / 'weights' / 'best.pt'}. "
                    f"Please check training output in: {args.project}"
                )
        
        if best_pt.exists():
            import shutil
            # 读取训练配置获取类别信息（最优模型选择与守门员共用）
            import yaml as _yaml_mod
            with open(args.data, 'r') as f:
                data_config = _yaml_mod.safe_load(f)
            # M7 最优模型选择（Loss 判据接入）：训练稳定直接用 best.pt；
            # Loss 震荡/发散且 last.pt 跌幅可接受时改选 last.pt（收敛末尾权重更稳）
            best_sel = None
            try:
                try:
                    from gatekeeper import select_best_weights
                except ImportError:
                    from src.yolo.gatekeeper import select_best_weights
                best_sel = select_best_weights(train_dir, args.data,
                                               normalize_class_names(data_config.get("names", [])))
                adopted_pt = Path(best_sel["weights"])
            except Exception as _sel_e:
                print(f"[{datetime.now().isoformat()}] 最优模型选择异常（回退 best.pt）: {_sel_e}")
                best_sel = None
                adopted_pt = best_pt
            shutil.copy(adopted_pt, weights_dir / "best.pt")
            print(f"[{datetime.now().isoformat()}] Model weights saved to {weights_dir / 'best.pt'} "
                  f"(source: {adopted_pt.name}"
                  + (f"，原因: {best_sel['reason']}" if best_sel else "") + ")")
            # 权重选择原因写入后续 gatekeeper 报告（审计溯源）
            weight_selection_note = best_sel["reason"] if best_sel else ""
            weights_path_abs = (weights_dir / "best.pt").resolve()
        else:
            raise FileNotFoundError(f"Model weights file not found: {best_pt}")
        
        # 读取任务元数据（血缘：数据集ID等）
        dataset_id = None
        try:
            with open(args.job_file, "r") as jf:
                _jm = json.load(jf)
            dataset_id = _jm.get("dataset_id")
        except Exception:
            pass

        # ----- 模型守门员：新模型 vs 同业务历史最佳生产模型 -----
        # 决策：promoted（晋升）/ rejected（淘汰/经验失败实验）/ first_version（首版直晋）
        # M7 增强（P5 拍板）：Margin(0.5%) + Loss 稳定性 Unstable 标记 + Data Cleaning Suggestion
        # 无论结果如何，训练本身都视为成功；被拦截模型归档为"失败实验"，不入生产
        gk = None
        try:
            try:
                from gatekeeper import run_gatekeeper, check_loss_stability
            except ImportError:
                from src.yolo.gatekeeper import run_gatekeeper, check_loss_stability
            gk = run_gatekeeper(
                args.baseline,
                str(weights_path_abs),
                args.data,
                normalize_class_names(data_config.get("names", []))
            )
        except Exception as gk_e:
            import traceback as _tb
            print(f"[{datetime.now().isoformat()}] 守门员评估异常: {gk_e}\n{_tb.format_exc()}")
            gk = {
                "result": "rejected",
                "promoted": False,
                "eval_split": "val",
                "new_metrics": None,
                "old_metrics": None,
                "class_ap": {},
                "regressed_classes": [],
                "report": f"守门员评估异常：{gk_e}。模型已产出但未自动进入生产（可人工强制覆盖）。",
            }

        # ----- M7/2.1：Loss 稳定性校验（仅对守门员通过/首版的模型生效，拒绝的不重复判定） -----
        # 命中 Unstable → 降级为 rejected（拒绝自动上线），并给出 Data Cleaning Suggestion
        loss_stability = None
        if gk.get("promoted"):
            try:
                loss_stability = check_loss_stability(train_dir)
            except Exception as _ls_e:
                print(f"[{datetime.now().isoformat()}] Loss 稳定性校验异常（跳过）: {_ls_e}")
                loss_stability = {"stable": True, "check": "no_data",
                                  "detail": "Loss 稳定性校验异常，跳过", "window": 0, "last_values": []}
            gk["loss_stability"] = loss_stability
            if loss_stability and not loss_stability.get("stable"):
                gk["result"] = "rejected"
                gk["promoted"] = False
                gk["loss_unstable"] = True
                gk["data_cleaning_suggestion"] = (
                    "训练 Loss 不稳定（震荡/发散）：建议检查最近标注数据的标签质量"
                    "（类别错标/框偏移/漏标），并用「基于此配置重训」修复后重试。"
                )
                gk["report"] = (gk.get("report", "") + "\n\n"
                                f"[M7] {loss_stability['detail']} "
                                "案件降级为 rejected，需人工介入确认（Loss 异常通常源于数据质量问题）。")

        # 2.4 迁移训练产出标注（训练模块规范 §3.2.3：守门员报告自动标注迁移来源）
        if migration:
            gk["report"] = (gk.get("report", "") + "\n\n"
                            f"此模型由跨代迁移训练产出（{migration['source']} → {migration['target']}），"
                            + ("lr0 已自动降 10 倍（保护 Backbone 既有知识）。" if migration["lr0_degraded"] else "")
                            + "对比基准仍为同业务历史最佳生产模型，满足 Margin 才可晋升。")

        # M7 最优模型选择溯源：记录权重落盘来源（best.pt 或 last.pt）及选择原因（审计用）
        if weight_selection_note:
            gk["weight_selection"] = weight_selection_note

        status = "production_ready" if gk.get("promoted") else "rejected"
        version = next_model_version(args.registry_dir, args.business, args.library_root_id)
        print(f"[{datetime.now().isoformat()}] 守门员决策: {gk.get('result')} → 状态={status}, 版本={version}")
        print(f"[{datetime.now().isoformat()}] 守门员报告: {gk.get('report', '')}")

        # 训练任务类型：ultralytics 按基座权重自动识别（yolov8n-seg.pt → segment）。
        # 这里以前无条件写死 "detect"，导致分割模型档案被误标成检测，
        # 后续按任务展示/筛选/导出全部会串，故改为取模型真实任务，取不到再按文件名兜底。
        train_task = str(getattr(model, "task", "") or "").strip().lower()
        if train_task not in ("detect", "segment", "pose", "classify", "obb"):
            train_task = "segment" if "-seg" in str(args.model).lower() else "detect"
        print(f"[{datetime.now().isoformat()}] 模型任务类型: {train_task}（基座: {args.model}）")

        model_meta = {
            "model_id": model_id,
            "job_id": args.job_id,
            "base_model": args.model,
            "task": train_task,
            "classes": normalize_class_names(data_config.get("names", [])),
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "created_at": datetime.now().isoformat(),
            "weights_path": str(weights_path_abs),
            # M11：architecture 三字段（从 base_model 解析，训练产出即写入）
            **parse_architecture(args.model),
            # 模型仓库 / 守门员字段
            "business": args.business,
            "status": status,
            "version": version,
            # 算法库归属：继承基座模型所在库；无继承（直接用官方权重训练）时产出自身即库根
            "library_root_id": args.library_root_id or model_id,
            "lineage": {
                # 微调时 parent 指向真正的微调父（base_model_id）；从零训练退回守门员基线
                "parent_model_id": args.base_model_id or args.baseline_model_id,
                "base_model": args.model,
                "dataset_id": dataset_id,
                "job_id": args.job_id,
            },
            # 2.4 迁移训练溯源（非迁移产出为 null；前端模型页展示"基于 xxx 迁移"）
            "migration": migration,
            "gatekeeper": gk,
        }
        # P8 半自动发布：守门员通过 → production_ready 但处于"待确认发布"（publish_status=pending + ready_at）
        if status == "production_ready":
            model_meta["publish_status"] = "pending"
            model_meta["ready_at"] = datetime.now().isoformat()
        
        with open(model_dir / "model.json", "w", encoding="utf-8") as f:
            json.dump(model_meta, f, indent=2, ensure_ascii=False)
        
        print(f"[{datetime.now().isoformat()}] Model registered as {model_id}")
        
        # 更新job状态
        with open(args.job_file, "r") as f:
            job_meta = json.load(f)
        
        job_meta["status"] = "completed"
        job_meta["completed_at"] = datetime.now().isoformat()
        job_meta["model_id"] = model_id
        # g6 M5 状态机统一：落库规范化（统一枚举，与查询回填一致）
        job_meta["unified_status"] = "Ready"
        
        # 早停检测：ultralytics 触发早停时训练正常返回，这里将状态标记为“已完成（早停）”
        # 而不是失败，便于 UI 区分展示（底层仅是提前停止训练）
        try:
            log_file = Path(args.job_file).parent / f"{args.job_id}.log"
            if log_file.exists():
                log_text = log_file.read_text(encoding="utf-8", errors="ignore").lower()
                if ("earlystopping" in log_text
                        or "stopped early" in log_text
                        or "no improvement observed" in log_text):
                    job_meta["early_stopped"] = True
                    print(f"[{datetime.now().isoformat()}] 早停触发：模型已收敛，自动停止训练，任务状态=已完成（早停）")
        except Exception:
            pass
        
        with open(args.job_file, "w") as f:
            json.dump(job_meta, f, indent=2)
        
        print(f"[{datetime.now().isoformat()}] Job status updated")

        # 阶段1.3：训练结果回写数据集状态机
        # 守门员合格 → 数据集 completed/已完成训练；守门员不合格 → failed/保持未完成（雪球回收）
        if gk and gk.get("promoted"):
            update_datasets_train_state(
                args.job_file, args.datasets_dir,
                stage="completed", training_status="completed",
                trained_at=datetime.now().isoformat(),
            )
        else:
            update_datasets_train_state(
                args.job_file, args.datasets_dir,
                stage="failed", training_status="incomplete",
            )
        
    except Exception as e:
        import traceback, re as _re
        tb_text = traceback.format_exc()
        err_text = str(e)
        # 打印完整堆栈，便于精确定位失败位置
        print(f"[{datetime.now().isoformat()}] ERROR: {err_text}", file=sys.stderr)
        print(tb_text, file=sys.stderr)
        
        # 更新job状态为失败
        try:
            with open(args.job_file, "r") as f:
                job_meta = json.load(f)
            
            job_meta["status"] = "failed"
            job_meta["error"] = err_text
            job_meta["error_traceback"] = tb_text  # 完整堆栈，供前端弹窗/技术人员排查
            job_meta["failed_at"] = datetime.now().isoformat()
            # g6 M5 状态机统一：失败 → Rejected（reason 细分 oom 时随 OOM 分支落盘）
            job_meta["unified_status"] = "Rejected"
            # g5 GPU 熔断：OOM 显式分类落盘（供前端失败弹窗按 reason 走"显存不足"恢复建议）
            if _re.search(r"out of memory|cuda error|outofmemoryerror|memoryerror", err_text, _re.I):
                job_meta["reason"] = "oom"
                job_meta["oom"] = True
                job_meta["oom_recovery"] = (
                    "本次训练因显存不足（OOM）停止。建议在「重新训练」时降低批次大小"
                    "（如 32 → 16/8）与图片尺寸（如 640 → 512），或改选更小的预训练模型后重试；"
                    "系统会自动沿用本次参数，请在确认调整后再重新训练。"
                )
            
            with open(args.job_file, "w") as f:
                json.dump(job_meta, f, indent=2)
        except:
            pass

        # 阶段1.3：训练失败 → 相关数据集回 failed（保持未完成，参与下一轮雪球）
        try:
            update_datasets_train_state(
                args.job_file, args.datasets_dir,
                stage="failed", training_status="incomplete",
            )
        except Exception:
            pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()

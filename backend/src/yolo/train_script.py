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
    if p.exists() or p.suffix != ".pt" or "/" in weights or "\\" in weights:
        return weights
    import shutil
    import urllib.request
    backend_dir = Path(__file__).resolve().parent.parent.parent
    custom_dir = backend_dir / "models" / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    target = custom_dir / weights
    if target.exists():
        return str(target)
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


def next_model_version(registry_dir: str, business: str) -> str:
    """计算该业务下一个模型版本号：扫描模型仓库中同业务已有版本，最大值 + 0.1；首版 v1.0"""
    best = 0.99  # best + 0.1 = 1.0 → 首版 v1.0
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
                if meta.get("business") != business:
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
    parser.add_argument("--baseline-model-id", type=str, default=None, help="上一代生产模型的 model_id（写入血缘）")
    
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
        
        model = YOLO(resume_path)
        
        # 使用传入的 batch size；-1/0(自动) 会触发 ultralytics 的 auto-batch 校准，
        # 在校准期多进程加载数据，容器环境下易报 [Errno 22] Invalid argument，这里兜底为确定值
        batch_size = args.batch
        if batch_size is None or batch_size < 1:
            batch_size = 16
        
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
            results = _run_training(dict(train_kwargs))
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
                results = _run_training(train_kwargs)
            else:
                raise amp_e
        
        print(f"[{datetime.now().isoformat()}] Training completed!")
        
        # 训练完成，注册模型
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_dir = Path(args.registry_dir) / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
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
            shutil.copy(best_pt, weights_dir / "best.pt")
            print(f"[{datetime.now().isoformat()}] Model weights saved to {weights_dir / 'best.pt'}")
        else:
            raise FileNotFoundError(f"Model weights file not found: {best_pt}")
        
        # 读取训练配置获取类别信息
        import yaml
        with open(args.data, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # 保存模型元数据（使用绝对路径）
        weights_path_abs = (weights_dir / "best.pt").resolve()

        # 读取任务元数据（血缘：数据集ID等）
        dataset_id = None
        try:
            with open(args.job_file, "r") as jf:
                _jm = json.load(jf)
            dataset_id = _jm.get("dataset_id")
        except Exception:
            pass

        # ----- 模型守门员：新模型 vs 同业务上一代生产模型 -----
        # 决策：promoted（晋升）/ rejected（淘汰/经验失败实验）/ first_version（首版直晋）
        # 无论结果如何，训练本身都视为成功；被拦截模型归档为"失败实验"，不入生产
        gk = None
        try:
            try:
                from gatekeeper import run_gatekeeper
            except ImportError:
                from src.yolo.gatekeeper import run_gatekeeper
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

        status = "production_ready" if gk.get("promoted") else "rejected"
        version = next_model_version(args.registry_dir, args.business)
        print(f"[{datetime.now().isoformat()}] 守门员决策: {gk.get('result')} → 状态={status}, 版本={version}")
        print(f"[{datetime.now().isoformat()}] 守门员报告: {gk.get('report', '')}")

        model_meta = {
            "model_id": model_id,
            "job_id": args.job_id,
            "base_model": args.model,
            "task": "detect",
            "classes": normalize_class_names(data_config.get("names", [])),
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "created_at": datetime.now().isoformat(),
            "weights_path": str(weights_path_abs),
            # 模型仓库 / 守门员字段
            "business": args.business,
            "status": status,
            "version": version,
            "lineage": {
                "parent_model_id": args.baseline_model_id,
                "base_model": args.model,
                "dataset_id": dataset_id,
                "job_id": args.job_id,
            },
            "gatekeeper": gk,
        }
        
        with open(model_dir / "model.json", "w", encoding="utf-8") as f:
            json.dump(model_meta, f, indent=2, ensure_ascii=False)
        
        print(f"[{datetime.now().isoformat()}] Model registered as {model_id}")
        
        # 更新job状态
        with open(args.job_file, "r") as f:
            job_meta = json.load(f)
        
        job_meta["status"] = "completed"
        job_meta["completed_at"] = datetime.now().isoformat()
        job_meta["model_id"] = model_id
        
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
        import traceback
        tb_text = traceback.format_exc()
        # 打印完整堆栈，便于精确定位失败位置
        print(f"[{datetime.now().isoformat()}] ERROR: {str(e)}", file=sys.stderr)
        print(tb_text, file=sys.stderr)
        
        # 更新job状态为失败
        try:
            with open(args.job_file, "r") as f:
                job_meta = json.load(f)
            
            job_meta["status"] = "failed"
            job_meta["error"] = str(e)
            job_meta["error_traceback"] = tb_text  # 完整堆栈，供前端弹窗/技术人员排查
            job_meta["failed_at"] = datetime.now().isoformat()
            
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

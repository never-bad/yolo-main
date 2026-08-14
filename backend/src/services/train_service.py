import json
import os
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

        # 训练轮数随数据量反比：数据少多轮、数据多少轮
        if image_count > 0:
            if image_count < 300:
                epochs = 200
            elif image_count < 1200:
                epochs = 120
            elif image_count < 3000:
                epochs = 80
            else:
                epochs = 50
        else:
            epochs = 100

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
        reason = (
            f"检测到 {device_info['name']}"
            f"（{'CUDA 显存 ' + str(device_info['vram_gb']) + 'GB' if cuda_ok else 'CPU'}），"
            f"数据集约 {image_count} 张图 / {class_count} 类，"
            f"已按{mode}场景推荐参数（可手动修改）。"
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

    async def create_job(self, dataset_id: str, version: str, model_name: str, 
                         epochs: int, imgsz: int, batch: int, base_model_id: str = None,
                         lr0: float = None, optimizer: str = None,
                         weight_decay: float = None, patience: int = None,
                         gpu_index: int = None, business: str = None):
        """创建训练任务（支持高级训练参数与训练节点选择、业务/算法类型隔离）

        business 为空时按数据集类别名自动推断业务/算法类型，无需用户手动选择。
        """
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 检查数据集
        dataset_dir = self.datasets_dir / dataset_id / version
        data_yaml = dataset_dir / "data.yaml"
        
        if not await asyncio.to_thread(lambda: data_yaml.exists()):
            raise ValueError(f"Dataset {dataset_id}/{version} not prepared")
        
        # 业务/算法类型：未指定时根据数据集类别名（data.yaml names）自动推断
        if not business:
            try:
                data_cfg = await asyncio.to_thread(self._load_yaml, data_yaml)
                business = infer_business(data_cfg.get("names", []))
                print(f"[create_job] 自动识别业务/算法类型: {business}（数据集 {dataset_id}/{version}）")
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

        # 创建job元数据
        job_meta = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "version": version,
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
        
        # 启动训练进程（在线程中执行）
        await asyncio.to_thread(
            self._start_training, job_id, data_yaml, actual_model_path, epochs, imgsz, batch, False,
            lr0, optimizer, weight_decay, patience, gpu_index,
            business, baseline_path, baseline_model_id
        )
        
        return {"job_id": job_id, "status": "running"}
    
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
        
        # 检查数据集
        dataset_dir = self.datasets_dir / job_meta["dataset_id"] / job_meta["version"]
        data_yaml = dataset_dir / "data.yaml"
        
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

import json
import shutil
import csv
import os
import asyncio
import tempfile
import zipfile
import io
from pathlib import Path
from datetime import datetime
from fastapi import UploadFile
from src.core.settings import settings
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def _load_json(path: Path):
    """同步加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict):
    """同步保存 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _delete_directory(path: Path):
    """同步删除目录"""
    shutil.rmtree(path)


class ModelService:
    def __init__(self):
        self.registry_dir = settings.REGISTRY_DIR
        self.jobs_dir = settings.JOBS_DIR
    
    async def list_models(self):
        """列出所有模型"""
        def _list_models_sync():
            models = []
            
            if not self.registry_dir.exists():
                return models
            
            for model_dir in self.registry_dir.iterdir():
                if model_dir.is_dir():
                    model_file = model_dir / "model.json"
                    if model_file.exists():
                        try:
                            model_meta = _load_json(model_file)
                            # 添加模型文件大小
                            weights_path = model_meta.get("weights_path")
                            if weights_path and Path(weights_path).exists():
                                model_meta["file_size"] = os.path.getsize(weights_path)
                                model_meta["file_size_mb"] = round(model_meta["file_size"] / (1024 * 1024), 2)
                            models.append(model_meta)
                        except Exception:
                            continue
            
            return models
        
        models = await asyncio.to_thread(_list_models_sync)
        return {"models": sorted(models, key=lambda x: x.get("created_at", ""), reverse=True)}
    
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
                
                return info
                
            except Exception as e:
                print(f"Error getting model info: {e}")
                return None
        
        return await asyncio.to_thread(_get_model_info_sync)
    
    async def update_model(self, model_id: str, request):
        """更新模型信息"""
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
            
            model_meta["updated_at"] = datetime.now().isoformat()
            
            _save_json(model_file, model_meta)
            
            return model_meta
        
        return await asyncio.to_thread(_update_model_sync)
    
    async def delete_model(self, model_id: str):
        """删除模型"""
        model_dir = self.registry_dir / model_id
        
        if not await asyncio.to_thread(lambda: model_dir.exists()):
            return None
        
        # 删除模型目录
        await asyncio.to_thread(_delete_directory, model_dir)
        
        return {"ok": True, "message": f"Model {model_id} deleted"}
    
    async def upload_model(self, file: UploadFile):
        """上传已有模型（ZIP格式）"""
        # 验证文件格式
        if not file.filename.endswith('.zip'):
            raise ValueError("Only .zip files are allowed")
        
        # 读取ZIP内容
        content = await file.read()
        filename = file.filename or "model.zip"
        
        # 验证ZIP文件大小（防止解压炸弹，限制10GB）
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
                
                # 解压ZIP文件
                zip_file.extractall(temp_path)
                zip_file.close()
                
                # 查找model.json和weights目录
                model_json_path = None
                weights_dir_path = None
                zip_model_id = None
                
                # 查找model.json（可能在根目录或子目录中）
                for json_file in temp_path.rglob("model.json"):
                    model_json_path = json_file
                    # 如果model.json在子目录中，提取model_id
                    relative_path = json_file.relative_to(temp_path)
                    if len(relative_path.parts) > 1:
                        zip_model_id = relative_path.parts[0]
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
                    # 创建新的model.json
                    model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    model_meta = {
                        "model_id": model_id,
                        "name": zip_filename.replace('.zip', ''),
                        "created_at": datetime.now().isoformat(),
                        "source": "uploaded",
                        "classes": [],
                        "description": f"Uploaded model from {zip_filename}",
                        "tags": ["uploaded"]
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
                
                # 更新model.json
                model_meta["model_id"] = model_id
                model_meta["weights_path"] = str((target_weights_dir / pt_files[0].name).resolve())
                model_meta["file_size"] = total_size
                model_meta["file_size_mb"] = round(total_size / (1024 * 1024), 2)
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
                
                return model_meta
        
        model_meta = await asyncio.to_thread(_process_zip, filename)
        
        # 尝试获取模型信息
        weights_path = model_meta.get("weights_path")
        if weights_path:
            model_info = await self._get_model_info(weights_path)
            if model_info:
                model_meta["model_info"] = model_info
                # 更新model.json
                model_dir = self.registry_dir / model_meta["model_id"]
                model_file = model_dir / "model.json"
                await asyncio.to_thread(_save_json, model_file, model_meta)
        
        return model_meta
    
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
            
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 添加model.json
                if model_file.exists():
                    zipf.write(model_file, f"{model_id}/model.json")
                
                # 添加weights目录
                weights_dir = model_dir / "weights"
                if weights_dir.exists():
                    for weight_file in weights_dir.iterdir():
                        if weight_file.is_file():
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
        return result

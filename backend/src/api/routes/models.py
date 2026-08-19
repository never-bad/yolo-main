from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from src.services.model_service import ModelService
from src.services.dataset_service import DatasetService
from src.services.queue_service import ModelQueueService
from src.core.settings import settings
import os
import shutil

router = APIRouter(prefix="/models", tags=["models"])
model_service = ModelService()
queue_service = ModelQueueService()

class UpdateModelRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    model_code: Optional[str] = None      # 模型唯一标识（小写+下划线，自动规范化并全局校验唯一）
    display_name: Optional[str] = None    # 业务中文名
    status: Optional[str] = None          # active / inactive

class LabelItem(BaseModel):
    index: int = 0
    english_code: str
    chinese_name: Optional[str] = ""
    chinese_desc: Optional[str] = ""

class UpdateLabelsRequest(BaseModel):
    labels: list[LabelItem]

class UploadModelRequest(BaseModel):
    name: Optional[str] = None
    classes: Optional[list[str]] = None

class OverrideModelRequest(BaseModel):
    business: Optional[str] = None      # 业务/算法类型（可选，用于隔离与版本对齐）
    reason: Optional[str] = None        # 强制覆盖原因（高级工程师填写，用于审计）

@router.get("")
async def list_models():
    """列出所有模型"""
    return await model_service.list_models()

@router.get("/custom")
async def list_custom_models():
    """列出用户上传的自定义预训练模型（.pt 文件）"""
    custom_dir = settings.CUSTOM_MODELS_DIR
    models = []
    if custom_dir.exists():
        for pt in sorted(custom_dir.glob("*.pt")):
            models.append({
                "filename": pt.name,
                "path": str(pt),
                "size": pt.stat().st_size if pt.exists() else 0
            })
    return {"models": models}

@router.get("/queues")
async def list_queue_overview():
    """4.2 模型消息队列总览（查看各模型待打包数据包）"""
    return queue_service.list_queues()

# 注意：静态路由必须定义在 /{model_id} 参数路由之前，否则会被参数路由拦截
@router.get("/similar-scan")
async def find_similar_models(min_similarity: float = Query(0.5, ge=0.0, le=1.0)):
    """2.7 相似模型排查：按标签字典类别名相似度扫描全部模型对（含自动选主建议）"""
    return await model_service.find_similar_models(min_similarity)

class MergeRequest(BaseModel):
    main_model_id: str
    merged_model_ids: list[str]
    reason: Optional[str] = None

@router.post("/merge")
async def merge_models(request: MergeRequest):
    """2.7 合并相似模型：差集类别并入主字典、数据集重归属、历史版本保留为分支、日志可回滚"""
    try:
        return await model_service.merge_models(request.main_model_id, request.merged_model_ids, request.reason)
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.get("/merge-log")
async def merge_logs(limit: int = Query(20, ge=1, le=100)):
    """2.7 合并日志（回滚依据）"""
    return await model_service.merge_log(limit)

class RollbackMergeRequest(BaseModel):
    log_index: int = -1   # -1 = 最近一次

@router.post("/rollback-merge")
async def rollback_merge(request: RollbackMergeRequest):
    """2.7 回滚合并：还原数据集归属、撤销 merged_into 标记"""
    try:
        return await model_service.rollback_merge(request.log_index)
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.get("/production")
async def list_production_models(business: Optional[str] = None):
    """列出当前在役的生产模型（status=production_ready），可按业务/算法类型过滤"""
    return await model_service.list_production_models(business)

@router.get("/{model_id}")
async def get_model(model_id: str):
    """获取模型详情"""
    result = await model_service.get_model(model_id)
    if not result:
        raise HTTPException(404, "Model not found")
    return result

@router.put("/{model_id}")
async def update_model(model_id: str, request: UpdateModelRequest):
    """更新模型信息（含阶段0：唯一 code / 中文名 / 启用状态）"""
    try:
        result = await model_service.update_model(model_id, request)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not result:
        raise HTTPException(404, "Model not found")
    return result

@router.get("/{model_id}/labels")
async def get_model_labels(model_id: str):
    """获取模型统一标签字典（四字段：index/english_code/chinese_name/chinese_desc）。

    不存在时自动从 model.json 的 classes 初始化。
    """
    result = await model_service.get_labels_dict(model_id)
    if not result:
        raise HTTPException(404, "Model not found")
    return result

@router.put("/{model_id}/labels")
async def update_model_labels(model_id: str, request: UpdateLabelsRequest):
    """保存模型标签字典（全量覆写；追加禁删保护：在用标签禁止删除/重命名/重排）"""
    try:
        labels = [item.model_dump() for item in request.labels]
        result = await model_service.update_labels_dict(model_id, labels)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not result:
        raise HTTPException(404, "Model not found")
    return result

class SuggestLabelsRequest(BaseModel):
    dataset_id: str
    limit: Optional[int] = 3

@router.post("/{model_id}/labels/suggest")
async def suggest_model_labels(model_id: str, request: SuggestLabelsRequest):
    """AI 识别图片新类别：取该数据集困难/空白样本图发千问 VL，
    返回已知标签之外的新标签候选（四字段），用户采纳后追加进字典"""
    return await model_service.suggest_labels_from_dataset(model_id, request.dataset_id, request.limit)

class AdoptSuggestRequest(BaseModel):
    suggestions: list[dict]  # [{english_code, chinese_name, chinese_desc}]

@router.post("/{model_id}/labels/suggest/adopt")
async def adopt_model_labels(model_id: str, request: AdoptSuggestRequest):
    """采纳 AI 建议的新标签：追加到该模型标签字典末尾（跳过已存在项）"""
    result = await model_service.adopt_suggested_labels(model_id, request.suggestions)
    if result is None:
        raise HTTPException(404, "Model not found")
    return result

@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """删除模型"""
    result = await model_service.delete_model(model_id)
    if not result:
        raise HTTPException(404, "Model not found")
    return result

@router.get("/{model_id}/datasets")
async def get_model_datasets(model_id: str):
    """1.7 模型仓库：返回归属于该模型的所有数据集（含状态机字段）"""
    res = await DatasetService().list_datasets()
    return {"model_id": model_id, "datasets": [d for d in res["datasets"] if d.get("model_id") == model_id]}

@router.get("/{model_id}/queue")
async def get_model_queue(model_id: str):
    """4.2 模型消息队列详情：待打包数据 + 打包历史"""
    return queue_service.get_queue(model_id)

class PackQueueRequest(BaseModel):
    force: Optional[bool] = False

@router.post("/{model_id}/queue/pack")
async def pack_model_queue(model_id: str, request: Optional[PackQueueRequest] = None):
    """4.2 手动触发打包：队列中的数据包立即进入标注页（自动按模型取类别）"""
    return await queue_service.pack_model(model_id)

@router.post("/upload-pt")
async def upload_pretrained_model(file: UploadFile = File(...)):
    """上传自定义预训练模型权重（.pt 文件）

    保存到 custom 模型目录，训练时可通过返回的 path 作为 model_name 使用
    """
    if not file.filename.endswith('.pt'):
        raise HTTPException(400, "Only .pt files are allowed")

    custom_dir = settings.CUSTOM_MODELS_DIR
    custom_dir.mkdir(parents=True, exist_ok=True)

    # 避免文件名冲突
    target = custom_dir / file.filename
    if target.exists():
        base, ext = target.stem, target.suffix
        counter = 1
        while target.exists():
            target = custom_dir / f"{base}_{counter}{ext}"
            counter += 1

    try:
        with open(target, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    return {"filename": target.name, "path": str(target), "size": target.stat().st_size}

@router.post("/upload")
async def upload_model(
    file: UploadFile = File(...)
):
    """上传已有模型（ZIP格式，与导出格式相同）
    
    Args:
        file: ZIP格式的模型压缩包，包含model.json和weights目录
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(400, "Only .zip files are allowed")
    
    try:
        result = await model_service.upload_model(file)
        return result
    except Exception as e:
        raise HTTPException(400, str(e))

@router.post("/{model_id}/override")
async def override_model(model_id: str, request: Optional[OverrideModelRequest] = None):
    """人工强制覆盖（Override）：高级工程师手动将被守门员拦截的模型强制设为生产版本。

    仅限高级工程师在特殊情况下使用（如数据质量已确认、线上紧急修复等）。
    系统会记录覆盖操作与原因，便于审计追溯。
    """
    reason = request.reason if request else None
    business = request.business if request else None
    try:
        return await model_service.override_model(model_id, business=business, reason=reason)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

@router.get("/{model_id}/export")
async def export_model(model_id: str):
    """导出模型为ZIP文件"""
    zip_path = await model_service.export_model(model_id)
    if not zip_path:
        raise HTTPException(404, "Model not found")
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{model_id}.zip",
        headers={"Content-Disposition": f"attachment; filename={model_id}.zip"}
    )

@router.get("/{model_id}/charts")
async def generate_training_charts(
    model_id: str,
    chart_type: str = Query("all", description="图表类型: loss, metrics, all")
):
    """生成训练图表
    
    Args:
        model_id: 模型ID
        chart_type: 图表类型 - loss（损失曲线）, metrics（指标曲线）, all（所有图表）
    
    Returns:
        图表图片文件
    """
    try:
        chart_paths = await model_service.generate_training_charts(model_id, chart_type)
        if not chart_paths:
            raise HTTPException(404, "Model not found")
        
        # 如果只有一个图表，直接返回
        if chart_type == "loss":
            return FileResponse(
                chart_paths["loss_chart"],
                media_type="image/png",
                filename=f"{model_id}_loss.png"
            )
        elif chart_type == "metrics":
            return FileResponse(
                chart_paths["metrics_chart"],
                media_type="image/png",
                filename=f"{model_id}_metrics.png"
            )
        else:
            # 返回损失曲线图（默认），前端可以分别请求
            return FileResponse(
                chart_paths["loss_chart"],
                media_type="image/png",
                filename=f"{model_id}_loss.png"
            )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/{model_id}/promote-to-detector")
async def promote_to_detector(model_id: str):
    """将训练好的模型升级为 AI 预标注的检测模型（热切换）。

    在相同验证集上对比 mAP50-95（核心）、Precision/Recall（辅助），
    仅当新模型显著优于当前检测模型时才自动切换。
    """
    import asyncio
    try:
        result = await asyncio.to_thread(model_service.promote_to_detector, model_id)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

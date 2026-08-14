from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from src.services.model_service import ModelService
from src.core.settings import settings
import os
import shutil

router = APIRouter(prefix="/models", tags=["models"])
model_service = ModelService()

class UpdateModelRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None

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

# 注意：静态路由必须定义在 /{model_id} 参数路由之前，否则会被参数路由拦截
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
    """更新模型信息"""
    result = await model_service.update_model(model_id, request)
    if not result:
        raise HTTPException(404, "Model not found")
    return result

@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """删除模型"""
    result = await model_service.delete_model(model_id)
    if not result:
        raise HTTPException(404, "Model not found")
    return result

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

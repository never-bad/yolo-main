from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict
from src.services.dataset_service import DatasetService
from src.services.model_service import ModelService

router = APIRouter(prefix="/datasets", tags=["datasets"])
dataset_service = DatasetService()
model_service = ModelService()

class PrepareRequest(BaseModel):
    split_ratio: Optional[Dict[str, float]] = {"train": 0.8, "val": 0.2}
    classes: Optional[list[str]] = None

class UpdateDatasetRequest(BaseModel):
    description: Optional[str] = None
    tags: Optional[list[str]] = None

class SealRequest(BaseModel):
    force: bool = False  # 时间窗口兜底：未标注完成时强制封板
    split_ratio: Optional[Dict[str, float]] = None  # 封板划分比例（缺省用 prepare 时配置）

class BindModelRequest(BaseModel):
    model_id: Optional[str] = None  # 归属模型；None 表示解除归属

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    model_id: Optional[str] = None,
    model_code: Optional[str] = None,
):
    """上传数据集压缩包（zip / tar / tar.gz / tgz）

    model_id:  1.7 模型仓库归属（可选），数据直接挂到对应模型下
    model_code: 2.2 动态建模型（可选）：按归一化 code 精确匹配模型；
               命中 → 挂到该模型；未命中 → 自动创建空白模型再挂载（优先于 model_id）
    """
    allowed = (".zip", ".tar", ".tar.gz", ".tgz")
    if not (file.filename or "").lower().endswith(allowed):
        raise HTTPException(400, "Only zip / tar / tar.gz / tgz files are allowed")
    
    target_model_id = model_id
    auto_created = False
    if model_code:
        try:
            target = await model_service.create_empty_model(model_code)
        except ValueError as e:
            raise HTTPException(400, str(e))
        target_model_id = target.get("model_id")
        auto_created = bool(target.get("empty"))
    
    result = await dataset_service.upload_dataset(file, target_model_id)
    result["model_code"] = model_code
    result["model_auto_created"] = auto_created if model_code else False
    return result

@router.put("/{dataset_id}/model")
async def bind_dataset_model(dataset_id: str, request: BindModelRequest):
    """绑定/解绑数据集到模型（1.7 模型仓库归属管理）"""
    result = await dataset_service.bind_model(dataset_id, request.model_id)
    return {"dataset_id": dataset_id, "model_id": result.get("model_id")}

@router.post("/{dataset_id}/prepare")
async def prepare_dataset(dataset_id: str, request: PrepareRequest):
    """准备数据集：解压、校验、生成配置"""
    result = await dataset_service.prepare_dataset(
        dataset_id, 
        request.split_ratio, 
        request.classes
    )
    return result

@router.get("")
async def list_datasets():
    """列出所有数据集"""
    return await dataset_service.list_datasets()

@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    """获取数据集详情"""
    result = await dataset_service.get_dataset(dataset_id)
    if not result:
        raise HTTPException(404, "Dataset not found")
    return result

@router.get("/{dataset_id}/tree")
async def get_dataset_tree(dataset_id: str):
    """获取数据集目录树（文件夹结构）"""
    result = await dataset_service.get_dataset_tree(dataset_id)
    if not result or not result.get("tree"):
        raise HTTPException(404, "Dataset directory not found")
    return result

@router.post("/{dataset_id}/seal")
async def seal_dataset(dataset_id: str, request: SealRequest):
    """封板：数量✓ + 标注✓ + 校验✓ 三条件（先标注后封板，封板时统一划分+转 txt+生成 data.yaml）"""
    try:
        result = await dataset_service.seal_dataset(dataset_id, request.force, request.split_ratio)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result

@router.post("/{dataset_id}/validate")
async def validate_dataset(dataset_id: str):
    """重新执行标注格式校验（2.1 第一级），结果写入 meta 并返回"""
    try:
        return await dataset_service.validate_dataset(dataset_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.put("/{dataset_id}")
async def update_dataset(dataset_id: str, request: UpdateDatasetRequest):
    """更新数据集信息（封板后只读）"""
    try:
        result = await dataset_service.update_dataset(dataset_id, request)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not result:
        raise HTTPException(404, "Dataset not found")
    return result

@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """删除数据集"""
    result = await dataset_service.delete_dataset(dataset_id)
    if not result:
        raise HTTPException(404, "Dataset not found")
    return result

@router.get("/{dataset_id}/export/annotated")
async def export_annotated_dataset(
    dataset_id: str,
    version: str = Query("v1", description="数据集版本")
):
    """导出标注后的数据集（包含图片和标签）"""
    zip_path = await dataset_service.export_annotated_dataset(dataset_id, version)
    if not zip_path:
        raise HTTPException(404, "Dataset not found")
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{dataset_id}_annotated.zip",
        headers={"Content-Disposition": f"attachment; filename={dataset_id}_annotated.zip"}
    )

@router.get("/{dataset_id}/export/original")
async def export_original_dataset(
    dataset_id: str,
    version: str = Query("v1", description="数据集版本")
):
    """导出标注前的数据集（仅包含图片）"""
    zip_path = await dataset_service.export_original_dataset(dataset_id, version)
    if not zip_path:
        raise HTTPException(404, "Dataset not found")
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{dataset_id}_original.zip",
        headers={"Content-Disposition": f"attachment; filename={dataset_id}_original.zip"}
    )

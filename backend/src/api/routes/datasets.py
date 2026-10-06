from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict
from src.services.dataset_service import DatasetService
from src.services.model_service import ModelService
from src.core.settings import settings

router = APIRouter(prefix="/datasets", tags=["datasets"])
dataset_service = DatasetService()
model_service = ModelService()

class PrepareRequest(BaseModel):
    split_ratio: Optional[Dict[str, float]] = {"train": 0.7, "val": 0.15, "test": 0.15}
    classes: Optional[list[str]] = None

class UpdateDatasetRequest(BaseModel):
    description: Optional[str] = None
    tags: Optional[list[str]] = None

class SealRequest(BaseModel):
    force: bool = False  # 时间窗口兜底：未标注完成时强制封板
    split_ratio: Optional[Dict[str, float]] = None  # 封板划分比例（缺省用 prepare 时配置）

class BindModelRequest(BaseModel):
    model_id: Optional[str] = None  # 归属模型；None 表示解除归属
    force: bool = False             # 类别序号对不上时仍强制挂载（前端确认弹窗后才传 True）

@router.get("/{dataset_id}/bind-check")
async def check_dataset_bind(dataset_id: str, model_id: str = Query(...)):
    """挂载前校验：该数据集换绑到 model_id 后，已标注类别的序号会不会错位

    返回 {conflict, blocking[], warnings[], summary}；summary 是可直接展示的大白话。
    """
    root_id = model_service.resolve_library_root_id(model_id)
    if not root_id:
        raise HTTPException(400, f"归属的算法库不存在：{model_id}")
    try:
        return await dataset_service.check_bind_target(dataset_id, root_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.put("/{dataset_id}/model")
async def bind_dataset_model(dataset_id: str, request: BindModelRequest):
    """绑定/解绑数据集到算法库（1.7 模型仓库归属管理）

    传入库内任一版本的模型 id 也会换算成该库的根记录 id 后写入。
    换库时若已标注类别的序号与目标库字典对不上，默认拒绝（400），需 force=True 放行。
    """
    root_id = model_service.resolve_library_root_id(request.model_id) if request.model_id else None
    if request.model_id and not root_id:
        raise HTTPException(400, f"归属的算法库不存在：{request.model_id}")
    try:
        result = await dataset_service.bind_model(dataset_id, root_id, root_id, request.force)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"dataset_id": dataset_id, "model_id": result.get("model_id")}

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    model_id: Optional[str] = None,
):
    """上传数据集压缩包（zip / tar / tar.gz / tgz）

    数据集必须有归属的算法库（导入页强制选择，没有库就先在弹窗里新建）：
    model_id: 归属的算法库/库内任一版本记录 id —— 一律换算成它所属「库根记录」后写入，
              使库详情的挂载数据集列表、准备阶段的类别过滤、训练产出的库归属都按库生效
    """
    allowed = (".zip", ".tar", ".tar.gz", ".tgz")
    if not (file.filename or "").lower().endswith(allowed):
        raise HTTPException(400, "Only zip / tar / tar.gz / tgz files are allowed")

    if not model_id:
        raise HTTPException(400, "请先选择该数据集归属的算法库（没有合适的库请先新建算法库）")
    library_root_id = model_service.resolve_library_root_id(model_id)
    if not library_root_id:
        raise HTTPException(400, f"归属的算法库不存在：{model_id}")

    return await dataset_service.upload_dataset(file, library_root_id, library_root_id)

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

@router.post("/advance-buckets")
async def advance_buckets():
    """按模型滚动灌桶（幂等）：A 包标注完成自动灌桶，桶满 SEAL_MIN_IMAGES 自动封板出成品数据集"""
    return await dataset_service.advance_buckets()


class SealThresholdRequest(BaseModel):
    seal_min_images: int


@router.get("/config/seal-threshold")
async def get_seal_threshold():
    """封板/桶灌数量门槛（当前生效值 + 默认值），供前端桶进度条展示与编辑"""
    from src.core.settings import Settings
    return {
        "seal_min_images": settings.SEAL_MIN_IMAGES,
        "default": Settings.model_fields["SEAL_MIN_IMAGES"].default,
        "seal_max_days": settings.SEAL_MAX_DAYS,
    }


@router.put("/config/seal-threshold")
async def set_seal_threshold(request: SealThresholdRequest):
    """在线修改封板/桶灌数量门槛（持久化到 data/seal_threshold.json，重启后仍生效）。
    0 = 关闭数量门槛（标注完审核通过即可封板/灌桶即满即封）。"""
    import json
    v = request.seal_min_images
    if not isinstance(v, int) or v < 0 or v > 1_000_000:
        raise HTTPException(400, "门槛需为 0 ~ 1000000 的整数（0 表示关闭数量门槛）")
    settings.SEAL_MIN_IMAGES = v
    # 列表接口结果含 seal_min_images，门槛变了要作废签名缓存
    import src.services.dataset_service as _ds
    with _ds._LIST_CACHE_LOCK:
        _ds._LIST_CACHE["sig"] = None
        _ds._LIST_CACHE["result"] = None
    try:
        settings.SEAL_THRESHOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
        settings.SEAL_THRESHOLD_FILE.write_text(
            json.dumps({"SEAL_MIN_IMAGES": v}, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        raise HTTPException(500, f"门槛已生效（本次运行），但持久化失败：{e}")
    return {"seal_min_images": v, "persisted": True}

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

@router.post("/{dataset_id}/approve")
async def approve_review(dataset_id: str):
    """审核通过：人工确认标注无误后标记，之后才允许自动灌桶/封板"""
    try:
        return await dataset_service.approve_review(dataset_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.post("/{dataset_id}/unpour")
async def unpour_bucket(dataset_id: str):
    """撤回：把已灌入接收桶但未封板的图片退回，恢复可编辑"""
    try:
        return await dataset_service.unpour_bucket(dataset_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

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

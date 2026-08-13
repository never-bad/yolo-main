from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from src.services.annotation_service import AnnotationService
from src.services.sam_service import SAMService
import tempfile
import os
from pathlib import Path

router = APIRouter(prefix="/annotations", tags=["annotations"])
annotation_service = AnnotationService()
sam_service = SAMService()

class CreateTaskRequest(BaseModel):
    dataset_id: str
    version: str = "v1"
    classes: Optional[List[str]] = None  # 可选，如果不提供则从 data.yaml 读取

class BBox(BaseModel):
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

class SaveAnnotationRequest(BaseModel):
    boxes: List[BBox]
    ai_annotated: bool = False

class ExportSplitRequest(BaseModel):
    train: float = 0.7
    val: float = 0.2
    test: float = 0.1

@router.post("/tasks")
async def create_annotation_task(request: CreateTaskRequest):
    """创建标注任务"""
    result = await annotation_service.create_task(
        request.dataset_id,
        request.version,
        request.classes
    )
    return result

@router.get("/tasks/{task_id}/items")
async def get_task_items(task_id: str):
    """获取标注任务的图片列表"""
    result = await annotation_service.get_task_items(task_id)
    if not result:
        raise HTTPException(404, "Task not found")
    return result

@router.post("/tasks/{task_id}/items/{image_id}")
async def save_annotation(task_id: str, image_id: str, request: SaveAnnotationRequest):
    """保存图片标注"""
    result = await annotation_service.save_annotation(
        task_id,
        image_id,
        request.boxes,
        request.ai_annotated
    )
    return result

@router.get("/tasks/{task_id}/items/{image_id}")
async def get_image_annotation(task_id: str, image_id: str):
    """获取单张图片的标注"""
    result = await annotation_service.get_image_annotation(task_id, image_id)
    if not result:
        raise HTTPException(404, "Image not found")
    return result

@router.get("/tasks/{task_id}/export")
async def export_annotations(task_id: str, format: str = "yolo"):
    """导出标注为YOLO格式"""
    if format != "yolo":
        raise HTTPException(400, "Only yolo format is supported")
    
    result = await annotation_service.export_to_yolo(task_id)
    if not result.get("ok", False):
        raise HTTPException(500, result.get("error", "Export failed"))
    return result

@router.post("/tasks/{task_id}/export-split")
async def export_annotations_split(task_id: str, request: ExportSplitRequest):
    """导出标注为YOLO格式，并按比例自动划分训练/验证/测试集"""
    result = await annotation_service.export_to_yolo_split(
        task_id, request.train, request.val, request.test
    )
    if not result.get("ok", False):
        raise HTTPException(500, result.get("error", "Export failed"))
    return result

@router.post("/tasks/{task_id}/clear-ai")
async def clear_ai_annotations(task_id: str):
    """清除该任务所有 AI 预标注（批量误标过多时一键清理，便于重新标注）"""
    result = await annotation_service.clear_ai_annotations(task_id)
    if not result.get("ok", False):
        raise HTTPException(404, result.get("error", "Task not found"))
    return result

@router.post("/tasks/{task_id}/clean-overlaps")
async def clean_task_overlaps(task_id: str):
    """清理该任务全部图片的高度重合标注框（跨类别/同类重复），一次性根治历史遗留的重叠标注。

    与检测时使用相同的合并逻辑，对任何检测模型的历史标注统一生效，无需按模型分别处理。
    """
    result = await sam_service.clean_task_annotations(task_id)
    if not result.get("ok", False):
        raise HTTPException(404, result.get("error", "Task not found"))
    return result

@router.post("/tasks/{task_id}/import-yolo")
async def import_yolo(task_id: str, file: UploadFile = File(...)):
    """导入 YOLO 格式标注（zip 包内含 labels/*.txt，归一化 class cx cy w h）。

    按文件名与任务图片匹配，写入 annotations.json；同名图片已有标注会被覆盖。
    用于把 X-AnyLabeling 等外部工具标注的数据接入平台闭环。
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "只支持 .zip 压缩包（内含 labels/*.txt）")

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        with open(tmp.name, "wb") as f:
            import shutil
            shutil.copyfileobj(file.file, f)
        result = await annotation_service.import_yolo_labels(task_id, Path(tmp.name))
    finally:
        await file.close()
        os.unlink(tmp.name)
    return result

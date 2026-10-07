from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
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
    # 矩形四坐标改为可选：分割目标只提交轮廓点（方框是给 SAM 圈范围的中间产物，用完就扔），
    # 只有纯检测目标必须带矩形；由服务端 clean_polygon / ensure_box_bbox 统一校验与补算。
    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    # 分割轮廓（像素坐标 [[x,y],...]）；类型放宽为 list，由服务端 clean_polygon 统一校验，
    # 避免个别坏点导致整张图的标注保存失败
    polygon: Optional[list] = None
    confidence: Optional[float] = None  # AI 置信度（仅 AI 标注携带）
    source: Optional[str] = None        # manual | ai | ai_qwen | import
    audit: Optional[dict] = None        # L2 千问/人工审核结论，随框落盘（供问题图片检索）

class SaveAnnotationRequest(BaseModel):
    boxes: List[BBox]
    ai_annotated: bool = False
    sample_type: Optional[str] = None      # 人工覆盖样本类型：normal / hard / background
    sample_reason: Optional[str] = None    # 覆盖时附带判据（如 manual）
    reviewed: bool = False                 # 人工复核标记：画布上编辑过标注后保存才为 True（粘性）

@router.post("/tasks")
async def create_annotation_task(request: CreateTaskRequest):
    """创建标注任务"""
    try:
        result = await annotation_service.create_task(
            request.dataset_id,
            request.version,
            request.classes
        )
    except ValueError as e:
        # P1-2：数据集不存在/未 prepare 等业务前置错误 → 400（服务层已抛语义化 ValueError）
        raise HTTPException(400, str(e))
    return result

@router.get("/tasks/by-dataset/{dataset_id}")
async def find_annotation_task(dataset_id: str, version: str = "v1"):
    """按数据集查找已存在的标注任务（标注页直达 / 自动建任务去重用）"""
    result = await asyncio.to_thread(annotation_service.find_task_by_dataset, dataset_id, version)
    if not result:
        return {"task": None}
    return {"task": result}

@router.get("/tasks/{task_id}/items")
async def get_task_items(task_id: str):
    """获取标注任务的图片列表"""
    result = await annotation_service.get_task_items(task_id)
    if not result:
        raise HTTPException(404, "Task not found")
    return result

@router.post("/tasks/{task_id}/items/{image_id}")
async def save_annotation(task_id: str, image_id: str, request: SaveAnnotationRequest):
    """保存图片标注（自动补全标签四字段 + 判定样本类型，人工可覆盖）"""
    result = await annotation_service.save_annotation(
        task_id,
        image_id,
        request.boxes,
        request.ai_annotated,
        request.sample_type,
        request.sample_reason,
        request.reviewed,
    )
    return result

@router.get("/tasks/{task_id}/items/{image_id}")
async def get_image_annotation(task_id: str, image_id: str):
    """获取单张图片的标注"""
    result = await annotation_service.get_image_annotation(task_id, image_id)
    if not result:
        raise HTTPException(404, "Image not found")
    return result

@router.delete("/tasks/{task_id}/images/{image_id}")
async def delete_task_image(task_id: str, image_id: str):
    """删除任务中的一张图片（同时清理其标注 JSON 与同名标签文件，并同步数据集计数）"""
    try:
        return await annotation_service.delete_image(task_id, image_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.get("/tasks/{task_id}/export")
async def export_annotations(task_id: str, format: str = "yolo"):
    """导出标注为YOLO格式"""
    if format != "yolo":
        raise HTTPException(400, "Only yolo format is supported")
    
    result = await annotation_service.export_to_yolo(task_id)
    if not result.get("ok", False):
        raise HTTPException(500, result.get("error", "Export failed"))
    return result

@router.get("/tasks/{task_id}/export-reviewed")
async def export_reviewed_annotations(task_id: str):
    """单独提取"人工修过"的图片 + YOLO 标签（含分割轮廓），打包成 zip 直接下载"""
    result = await annotation_service.export_reviewed_zip(task_id)
    if not result.get("ok", False):
        raise HTTPException(400, result.get("error", "Export failed"))
    name = result["filename"]
    return FileResponse(
        result["zip_path"],
        media_type="application/zip",
        filename=name,
        headers={"Content-Disposition": f"attachment; filename={name}"}
    )

@router.post("/tasks/{task_id}/clear-ai")
async def clear_ai_annotations(task_id: str):
    """清除该任务所有 AI 预标注（批量误标过多时一键清理，便于重新标注）"""
    result = await annotation_service.clear_ai_annotations(task_id)
    if not result.get("ok", False):
        raise HTTPException(404, result.get("error", "Task not found"))
    return result

@router.post("/tasks/{task_id}/clear-all")
async def clear_all_annotations(task_id: str):
    """清除该任务数据集的全部标注框（AI 标注 + 人工标注一视同仁）。

    清空前自动打包备份到 annotations/backups/，误清可解压回退；
    已封板 / 已入桶 / 已归档的数据集拒绝执行。
    """
    try:
        return await annotation_service.clear_all_annotations(task_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

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

    按文件名与任务图片匹配，写入每图 JSON 标注文件；同名图片已有标注会被覆盖。
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

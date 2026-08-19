from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import traceback
from src.services.sam_service import SAMService

router = APIRouter(prefix="/sam", tags=["sam"])
sam_service = SAMService()


class ConfigRequest(BaseModel):
    detector: Optional[str] = None            # yolo_world | grounding_dino | qwen_vl | none
    detector_weights: Optional[str] = None
    grounding_dino_model: Optional[str] = None  # Transformers 集成的 GD 模型名（如 IDEA-Research/grounding-dino-tiny）
    sam_enabled: Optional[bool] = None
    sam_weights: Optional[str] = None
    imgsz: Optional[int] = None
    sam_imgsz: Optional[int] = None
    conf: Optional[float] = None
    iou: Optional[float] = None              # NMS IoU 阈值（在线可调）
    half: Optional[bool] = None
    device: Optional[str] = None              # auto | cpu | GPU 索引
    # 千问 VL 大模型预标注（本地预留接口）
    qwen_enabled: Optional[bool] = None
    qwen_backend: Optional[str] = None        # ollama | dashscope
    qwen_endpoint: Optional[str] = None
    qwen_model: Optional[str] = None
    qwen_api_key: Optional[str] = None
    qwen_timeout: Optional[int] = None
    qwen_mock: Optional[bool] = None


class AutoLabelRequest(BaseModel):
    task_id: str
    image_id: str
    classes: List[str]
    conf: Optional[float] = None
    prompts: Optional[List[str]] = None  # 与 classes 一一对应的英文提示词，提升中文识别率


class BatchStartRequest(BaseModel):
    task_id: str
    classes: List[str]
    conf: Optional[float] = None
    prompts: Optional[List[str]] = None  # 与 classes 一一对应的英文提示词


class InteractiveLabelRequest(BaseModel):
    task_id: str
    image_id: str
    classes: List[str]
    conf: Optional[float] = None
    prompts: Optional[List[str]] = None  # 与 classes 一一对应的英文提示词
    region: Optional[dict] = None        # 用户框选的局部区域（图像坐标）{"x1","y1","x2","y2"}，缺省全图


@router.get("/available")
async def sam_available():
    """检查 SAM 预标注可用性（需模型可加载）"""
    return await sam_service.is_available()


@router.get("/config")
async def get_config():
    """读取 SAM 配置"""
    return sam_service.get_config()


@router.post("/config")
async def update_config(request: ConfigRequest):
    """更新 SAM 配置"""
    return sam_service.update_config(request)


@router.get("/models")
async def list_models():
    """列出可用的检测模型（.pt 文件）"""
    return sam_service.list_models()


@router.post("/models")
async def upload_model(file: UploadFile = File(...)):
    """上传一个检测模型（.pt）到统一目录"""
    try:
        filename = file.filename.replace("\\", "/").split("/")[-1]
        if not filename.endswith(".pt"):
            raise HTTPException(400, "仅支持 .pt 模型文件")
        data = await file.read()
        if not data:
            raise HTTPException(400, "文件内容为空")
        sam_service.save_model(filename, data)
        return {"ok": True, "name": filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/validate")
async def validate():
    """验证模型有效性"""
    return await sam_service.validate_model()


@router.post("/auto-label")
async def auto_label(request: AutoLabelRequest):
    """单图预标注：按类别名自动生成目标框"""
    try:
        return await sam_service.auto_label(
            request.task_id, request.image_id, request.classes, request.conf, request.prompts
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@router.post("/interactive-label")
async def interactive_label(request: InteractiveLabelRequest):
    """交互式标注：在用户框选的局部区域内按文本提示检测目标框"""
    try:
        return await sam_service.interactive_label(
            request.task_id, request.image_id, request.classes,
            request.conf, request.prompts, request.region
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@router.post("/batch/start")
async def batch_start(request: BatchStartRequest):
    """启动异步批量预标注"""
    try:
        return await sam_service.batch_auto_label(
            request.task_id, request.classes, request.conf, request.prompts
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/batch/{batch_id}")
async def batch_status(batch_id: str):
    """查询批量预标注进度"""
    result = await sam_service.get_batch_status(batch_id)
    if not result:
        raise HTTPException(404, "Batch task not found")
    return result


@router.post("/batch/{batch_id}/stop")
async def batch_stop(batch_id: str):
    """取消批量预标注"""
    result = await sam_service.stop_batch(batch_id)
    if not result:
        raise HTTPException(404, "Batch task not found")
    return result
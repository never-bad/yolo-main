from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import traceback
from src.services.sam_service import SAMService

router = APIRouter(prefix="/sam", tags=["sam"])
sam_service = SAMService()


class ConfigRequest(BaseModel):
    detector: Optional[str] = None            # yolo_world | grounding_dino | qwen_vl | none
    detectors: Optional[List[str]] = None     # 多引擎同时使用：["grounding_dino","yolo_world","qwen_vl","trained_model"]，空=沿用单引擎
    trained_model_id: Optional[str] = None    # 「本库已训模型」引擎使用的模型版本 id（registry 目录名）
    detector_weights: Optional[str] = None
    grounding_dino_model: Optional[str] = None  # Transformers 集成的 GD 模型名（如 IDEA-Research/grounding-dino-tiny）
    sam_enabled: Optional[bool] = None        # 单图 AI 预标注 / 交互式标注是否用 SAM 描分割轮廓
    batch_sam_enabled: Optional[bool] = None  # 批量预标注是否也描轮廓（默认关：只出方框，速度快一倍以上）
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
    qwen_keep_alive: Optional[str] = None       # ollama：模型驻留内存时长（"5m"/"-1"=常驻）
    qwen_num_parallel: Optional[int] = None     # ollama：并行处理请求数（OLLAMA_NUM_PARALLEL）
    qwen_image_max_side: Optional[int] = None   # 送大模型前的图片最长边（默认 1024）
    qwen_coord_mode: Optional[str] = None       # 坐标约定：auto | norm1000 | pixel
    qwen_audit_enabled: Optional[bool] = None   # 单图预标注后是否自动做 L2 审核（关掉约快 4 倍）
    qwen_audit_batch: Optional[bool] = None     # 批量预标注是否也做 L2 审核（默认关）
    qwen_verify_enabled: Optional[bool] = None  # 框级二次校验（反幻觉：候选框裁剪复问，回答否则丢弃）
    qwen_verify_margin: Optional[float] = None  # 校验裁剪的四周外扩比例（默认 0.35）


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


class PointLabelRequest(BaseModel):
    task_id: str
    image_id: str
    class_id: int = 0                  # 抠出来的轮廓归到哪个类别（前端"绘制类别"）
    points: List[dict]                 # [{"x":..,"y":..,"label":1|0}] 1=点中了 0=点错了


class ErasePolygonRequest(BaseModel):
    polygon: List[list]                # 待擦除的分割轮廓 [[x,y],...]（原图像素）
    erase_rects: List[dict]            # 要挖掉的矩形 [{"x1","y1","x2","y2"}]
    width: float
    height: float
    class_id: int = 0


class AuditRequest(BaseModel):
    task_id: str
    image_id: str
    boxes: Optional[List[dict]] = None   # 前端当前画布上的框（未保存也可审）；缺省时后端读已保存标注


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


@router.get("/trained-versions")
async def trained_versions(task_id: Optional[str] = None):
    """列出可用来预标注的「已训模型版本」（优先当前算法库自己的，权重必须已落盘）"""
    try:
        return sam_service.list_trained_versions(task_id or "")
    except Exception as e:
        traceback.print_exc()
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


@router.post("/point-label")
async def point_label(request: PointLabelRequest):
    """点选抠图：在图上点一下（或几点微调）→ SAM 沿边缘描出轮廓 → 返回分割标注框"""
    try:
        return await sam_service.point_label(
            request.task_id, request.image_id, request.class_id, request.points
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@router.post("/erase-polygon")
async def erase_polygon(request: ErasePolygonRequest):
    """区域擦除：把分割轮廓按矩形挖掉多余部分（纯几何、零推理、毫秒级）。

    用于"抠大了，拖框切掉文字水印/相邻水面"，替代一下一下点右键负点。
    """
    try:
        cfg = sam_service._read_config()
        boxes = sam_service.erase_polygon(
            request.polygon, request.erase_rects,
            request.width, request.height, request.class_id, cfg
        )
        return {"boxes": boxes}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@router.post("/audit")
async def audit_image(request: AuditRequest):
    """千问 L2 审核：对当前图片标注框做中文语义二次审核（漏标/粘连/标签错误/框质量/合规）"""
    try:
        return await sam_service.audit_image(
            request.task_id, request.image_id, request.boxes
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@router.get("/audit/issues")
async def audit_issues(task_id: str):
    """问题图片检索：扫描任务全部已保存标注，返回带审核异议（reject/modify/split）的图片列表"""
    try:
        return await sam_service.list_audit_issues(task_id)
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
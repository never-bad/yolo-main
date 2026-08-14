from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.services.train_service import TrainService

router = APIRouter(prefix="/train", tags=["train"])
train_service = TrainService()

class TrainJobRequest(BaseModel):
    dataset_id: str
    version: str = "v1"
    model_name: str = "yolov8n.pt"  # 可以是预训练模型或已有模型ID
    epochs: int = 10
    imgsz: int = 640
    batch: int = -1
    base_model_id: Optional[str] = None  # 用于微调的已有模型ID
    business: Optional[str] = None       # 业务/算法类型：传空则系统按数据集类别名自动分配（守门员按此隔离对比）
    # 高级训练参数（可选，缺省则由系统默认）
    lr0: Optional[float] = None          # 初始学习率
    optimizer: Optional[str] = None      # auto / SGD / Adam / AdamW
    weight_decay: Optional[float] = None # 权重衰减
    patience: Optional[int] = None       # 早停轮数（0 = 关闭）
    gpu_index: Optional[int] = None      # 训练节点：指定 GPU 索引（多卡环境；缺省自动）

@router.get("/gpus")
async def list_gpus():
    """列出当前服务器可用的 GPU（训练节点），用于创建任务时选择"""
    return await train_service.list_gpus()

@router.get("/suggest")
async def suggest_train_params(dataset_id: str, version: str = "v1", base_model_id: str = None):
    """自动推荐训练参数（基础 + 高级），根据硬件与数据集规模"""
    return await train_service.suggest_params(dataset_id, version, base_model_id)

@router.get("/business")
async def infer_business(dataset_id: str, version: str = "v1"):
    """根据数据集类别名自动推断业务/算法类型（选中数据集后系统自动分配，无需手动选择）"""
    return await train_service.infer_dataset_business(dataset_id, version)

@router.post("/jobs")
async def create_train_job(request: TrainJobRequest):
    """创建训练任务（支持基于已有模型微调）"""
    try:
        result = await train_service.create_job(
            dataset_id=request.dataset_id,
            version=request.version,
            model_name=request.model_name,
            epochs=request.epochs,
            imgsz=request.imgsz,
            batch=request.batch,
            base_model_id=request.base_model_id,
            business=request.business,
            lr0=request.lr0,
            optimizer=request.optimizer,
            weight_decay=request.weight_decay,
            patience=request.patience,
            gpu_index=request.gpu_index
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.get("/jobs")
async def list_train_jobs():
    """列出所有训练任务"""
    return await train_service.list_jobs()

@router.get("/jobs/{job_id}")
async def get_train_job(job_id: str):
    """获取训练任务详情"""
    result = await train_service.get_job(job_id)
    if not result:
        raise HTTPException(404, "Job not found")
    return result

@router.get("/jobs/{job_id}/tree")
async def get_train_job_tree(job_id: str):
    """获取训练任务输出目录树（文件夹结构）"""
    result = await train_service.get_job_tree(job_id)
    if not result:
        raise HTTPException(404, "Job not found")
    return result

@router.post("/jobs/{job_id}/stop")
async def stop_train_job(job_id: str):
    """停止训练任务"""
    result = await train_service.stop_job(job_id)
    return result

@router.post("/jobs/{job_id}/resume")
async def resume_train_job(job_id: str):
    """继续训练中断的任务（支持正常停止和崩溃恢复）"""
    try:
        result = await train_service.resume_job(job_id)
        if not result:
            raise HTTPException(404, "Job not found")
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to resume training: {str(e)}")

@router.delete("/jobs/{job_id}")
async def delete_train_job(job_id: str):
    """删除训练任务"""
    result = await train_service.delete_job(job_id)
    if not result:
        raise HTTPException(404, "Job not found")
    return result

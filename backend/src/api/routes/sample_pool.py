from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from src.services.sample_pool_service import SamplePoolService

router = APIRouter(prefix="/sample-pool", tags=["sample-pool"])

sample_pool_service = SamplePoolService()


class AddHardRequest(BaseModel):
    dataset_id: str
    model_id: str
    image_names: List[str]
    version: Optional[str] = "v1"


class AddBackgroundRequest(BaseModel):
    dataset_id: str
    image_names: List[str]
    version: Optional[str] = "v1"


@router.post("/hard")
async def add_hard_samples(req: AddHardRequest):
    """把数据集指定图片（连同标注）加入某模型的困难样本库（1.6）"""
    try:
        return await sample_pool_service.add_hard_samples(
            req.dataset_id, req.model_id, req.image_names, req.version
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/background")
async def add_background_samples(req: AddBackgroundRequest):
    """把数据集指定图片作为无目标背景（负样本）加入全局空白样本库（1.6）"""
    try:
        return await sample_pool_service.add_background_samples(
            req.dataset_id, req.image_names, req.version
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("")
async def list_pool():
    """样本池状态：困难样本（按模型）+ 空白样本（全局）"""
    return await sample_pool_service.list_pool()


@router.delete("/hard/{model_id}")
async def clear_hard_pool(model_id: str):
    """清空某模型的困难样本库"""
    return await sample_pool_service.clear_hard(model_id)


@router.delete("/background")
async def clear_background_pool():
    """清空全局空白样本库"""
    return await sample_pool_service.clear_background()
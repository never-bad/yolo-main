import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
from src.services.dataset_service import DatasetService, AlignmentConfirmRequired
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
    alignment_decisions: Optional[List[dict]] = None  # 换库触发类别对齐 needs_confirm 时的人工确认决策

class MergeRequest(BaseModel):
    dataset_ids: list[str]          # 至少 2 个待合并的数据集 id
    library_root_id: Optional[str] = None  # 归属算法库（缺省取第一个源的归属库）
    display_name: Optional[str] = None    # 合并后数据集命名（缺省 merged_<时间>）

class ClassAlignmentRequest(BaseModel):
    # [{"src_index": 0, "target_id": 1}]；target_id=null/-1 表示忽略该源类
    decisions: list[dict]

@router.post("/{dataset_id}/confirm-class-alignment")
async def confirm_class_alignment(dataset_id: str, request: ClassAlignmentRequest):
    """B-64 导入闸门：人工确认「包内类名 → 本库标签字典」的对应关系。

    确认后同步重写 labels/*.txt、每图 JSON、data.yaml 与 meta.class_map，
    把报告置为 confirmed；未确认的数据集会在封板时被 label_guard 明确拒绝。
    """
    try:
        return await asyncio.to_thread(
            dataset_service.confirm_class_alignment, dataset_id, request.decisions
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

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
        result = await dataset_service.bind_model(
            dataset_id, root_id, root_id, request.force, request.alignment_decisions
        )
    except AlignmentConfirmRequired as e:
        # 换库触发类别对齐 needs_confirm：409 返回报告，前端弹确认框后重发带 decisions 的请求
        return JSONResponse(status_code=409, content=e.report)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"dataset_id": dataset_id, "model_id": result.get("model_id")}

class CancelUploadRequest(BaseModel):
    upload_id: str


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    model_id: Optional[str] = None,
    upload_id: Optional[str] = None,
    role: Optional[str] = None,
):
    """上传数据集压缩包（zip / tar / tar.gz / tgz）

    数据集必须有归属的算法库（导入页强制选择，没有库就先在弹窗里新建）：
    model_id: 归属的算法库/库内任一版本记录 id —— 一律换算成它所属「库根记录」后写入，
              使库详情的挂载数据集列表、准备阶段的类别过滤、训练产出的库归属都按库生效
    upload_id: 前端生成的本次上传标识；登记在途产物后，取消接口可据此精确清理，
               避免「上传中断 / 请求断开」在服务端留下目录、压缩包或影子库记录。
    role: 数据集角色。缺省 "train"（进入灌桶→封板训练流程）；"test" 表示「直接上传测试集」，
          创建即标记 role=test，跳过灌桶/时间窗自动封板（仍走准备阶段抽取标注），
          由前端在准备完成后调 set-as-test-set 封板为算法库的 held-out 测试集。
    """
    allowed = (".zip", ".tar", ".tar.gz", ".tgz")
    if not (file.filename or "").lower().endswith(allowed):
        raise HTTPException(400, "Only zip / tar / tar.gz / tgz files are allowed")

    if role not in (None, "train", "test"):
        raise HTTPException(400, "role 仅支持 train / test")

    if not model_id:
        raise HTTPException(400, "请先选择该数据集归属的算法库（没有合适的库请先新建算法库）")
    library_root_id = model_service.resolve_library_root_id(model_id)
    if not library_root_id:
        raise HTTPException(400, f"归属的算法库不存在：{model_id}")

    return await dataset_service.upload_dataset(file, library_root_id, library_root_id, upload_id, role)


@router.post("/upload/cancel")
async def cancel_upload(request: CancelUploadRequest):
    """取消上传：按 upload_id 清理服务端在途产物（数据集目录 / 原始压缩包 / PG 影子库）。

    幂等且可与上传并发：即使上传请求已断开、前端拿不到 dataset_id，也能清干净；
    尚未到达服务端的上传返回 no_pending_upload，本身不产生文件，无需清理。
    """
    return await dataset_service.cancel_upload(request.upload_id)

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

@router.post("/merge")
async def merge_datasets_endpoint(request: MergeRequest):
    """多数据集合并为持久新数据集（方案 B）：源数据集只读，产出新数据集可直接训练/封板/导出。

    类目统一：提供 library_root_id 且库有标签字典 → 用库字典顺序（字典外类别剔除）；
    否则各源类别取并集（首次出现顺序）。任一源未生成训练产物（缺 data.yaml / images / labels）则 400。
    """
    try:
        return await dataset_service.merge_datasets(
            request.dataset_ids, request.library_root_id, request.display_name
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


class SealThresholdRequest(BaseModel):
    seal_min_images: int
    model_id: str | None = None   # 传库 id = 单独覆盖该库；不传 = 改全局默认


@router.get("/config/seal-threshold")
async def get_seal_threshold():
    """封板/桶灌数量门槛（全局默认 + 按库覆盖表），供前端桶进度条展示与编辑"""
    from src.core.settings import Settings
    return {
        "seal_min_images": settings.SEAL_MIN_IMAGES,
        "default": Settings.model_fields["SEAL_MIN_IMAGES"].default,
        "seal_max_days": settings.SEAL_MAX_DAYS,
        "per_model": dict(settings.SEAL_THRESHOLDS_PER_MODEL),
    }


@router.put("/config/seal-threshold")
async def set_seal_threshold(request: SealThresholdRequest):
    """在线修改封板/桶灌数量门槛（持久化到 data/seal_threshold.json，重启后仍生效）。
    model_id 为空 = 改全局默认；非空 = 单独覆盖该库（seal_min_images 传 null 语义用 -1 表示恢复跟随全局）。
    0 = 关闭数量门槛（标注完审核通过即可封板/灌桶即满即封）。"""
    import json
    v = request.seal_min_images
    if not isinstance(v, int) or v < -1 or v > 1_000_000:
        raise HTTPException(400, "门槛需为 0 ~ 1000000 的整数（0 关闭数量门槛，-1 恢复跟随全局默认）")
    mid = (request.model_id or "").strip()
    if mid:
        # 按库覆盖：-1 = 删除覆盖，恢复跟随全局
        if v == -1:
            settings.SEAL_THRESHOLDS_PER_MODEL.pop(mid, None)
        else:
            settings.SEAL_THRESHOLDS_PER_MODEL[mid] = v
    else:
        if v == -1:
            raise HTTPException(400, "全局默认不支持 -1")
        settings.SEAL_MIN_IMAGES = v
    # 列表接口结果含门槛/桶进度，变了要作废签名缓存
    import src.services.dataset_service as _ds
    with _ds._LIST_CACHE_LOCK:
        _ds._LIST_CACHE["sig"] = None
        _ds._LIST_CACHE["result"] = None
    try:
        settings.SEAL_THRESHOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
        # 读旧文件合并写回，避免覆盖丢失另一份配置
        data = {}
        if settings.SEAL_THRESHOLD_FILE.exists():
            try:
                data = json.loads(settings.SEAL_THRESHOLD_FILE.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data["SEAL_MIN_IMAGES"] = settings.SEAL_MIN_IMAGES
        data["per_model"] = dict(settings.SEAL_THRESHOLDS_PER_MODEL)
        settings.SEAL_THRESHOLD_FILE.write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        raise HTTPException(500, f"门槛已生效（本次运行），但持久化失败：{e}")
    return {
        "seal_min_images": settings.SEAL_MIN_IMAGES,
        "model_id": mid or None,
        "per_model": dict(settings.SEAL_THRESHOLDS_PER_MODEL),
        "persisted": True,
    }

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

class SetAsTestSetRequest(BaseModel):
    force: bool = False  # 存在泄漏（测试集与同库训练图 md5 重合）时强制放行（写审计）

@router.post("/{dataset_id}/set-as-test-set")
async def set_dataset_as_test_set(dataset_id: str, request: SetAsTestSetRequest):
    """将该数据集封板为所属算法库的独立测试集（held-out）。

    - 校验归属库 / 已标注 / 标签字典一致 / 与同库训练图无 md5 泄漏
    - 封板产物仅 test split（images/test + labels/test + data.yaml 只含 test 键）
    - 写 meta.role=test + 库根指针 test_dataset_id
    - 未通过泄漏检查需 force=True 放行
    """
    try:
        return await dataset_service.set_as_test_set(dataset_id, request.force)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

@router.post("/{dataset_id}/reseal-test-set")
async def reseal_dataset_as_test_set(dataset_id: str):
    """测试集动态增删图片后重新封板（版本号 +1，同库旧指标标 stale）。"""
    try:
        return await dataset_service.reseal_test_set(dataset_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

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

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from src.services.model_service import ModelService
from src.services.dataset_service import DatasetService
from src.services.queue_service import ModelQueueService
from src.db import meta_store
from src.core.settings import settings
import shutil

router = APIRouter(prefix="/models", tags=["models"])
model_service = ModelService()
queue_service = ModelQueueService()

class CreateModelRequest(BaseModel):
    display_name: str = ""
    model_code: str
    business: Optional[str] = ""
    description: Optional[str] = ""

class UpdateModelRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    model_code: Optional[str] = None      # 模型唯一标识（小写+下划线，自动规范化并全局校验唯一）
    display_name: Optional[str] = None    # 业务中文名
    status: Optional[str] = None          # active / inactive
    model_library_id: Optional[str] = None  # 归入算法库（库标识；传空串/None 表示移出库降级为独立模型）

class LabelItem(BaseModel):
    index: int = 0
    english_code: str
    chinese_name: Optional[str] = ""
    chinese_desc: Optional[str] = ""
    # 英文检测提示词：告诉 GroundingDINO / YOLO-World「这个类别在画面里长什么样」
    # （这两个引擎只认英文，填中文或拼音等于无效输入）。留空则回退到英文名。
    ai_prompt_en: Optional[str] = ""

class UpdateLabelsRequest(BaseModel):
    labels: list[LabelItem]

class OverrideModelRequest(BaseModel):
    business: Optional[str] = None      # 业务/算法类型（可选，用于隔离与版本对齐）
    reason: Optional[str] = None        # 强制覆盖原因（高级工程师填写，用于审计）

@router.get("")
async def list_models():
    """列出所有模型"""
    return await model_service.list_models()

@router.post("")
async def create_model(req: CreateModelRequest):
    """首页「+ 新建算法库」：创建空白模型（空标签字典 + 采集库），code 已存在返回 400"""
    try:
        meta = await model_service.create_empty_model(
            req.model_code,
            display_name=req.display_name or "",
            business=req.business or "",
            description=req.description or "",
        )
        return meta
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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

@router.get("/queues")
async def list_queue_overview():
    """4.2 模型消息队列总览（查看各模型待打包数据包）"""
    return queue_service.list_queues()

class LibraryRequest(BaseModel):
    model_library_id: str = ""          # 库标识（小写+下划线，唯一；空则后端按时间戳生成）
    display_name: Optional[str] = ""
    description: Optional[str] = ""
    business_scenario: Optional[str] = ""
    base_algorithm: Optional[str] = ""
    architecture_family: Optional[str] = ""
    label_dict_id: Optional[str] = None

@router.get("/libraries")
async def list_libraries():
    """算法库列表（供模型管理页「算法库」下拉与库统计）。

    库来源有两处，统一输出：
    1. registry 中的库根记录（首页「+ 新建算法库」产生的 empty 模型，平台主流程用的就是它）；
    2. PG model_libraries 表的行（保留兼容）。
    附加每个库挂接的模型数与生产模型名。
    """
    libs = await meta_store.list_model_libraries()
    seen = {l.get("model_library_id") for l in libs}
    res = await model_service.list_models()
    models_list = res.get("models", []) if isinstance(res, dict) else (res or [])
    # registry 库根记录也是一等公民库（否则首页建的库在下拉里看不到）
    for m in models_list:
        rid = m.get("model_id")
        if m.get("empty") and rid and rid not in seen:
            libs.append({
                "model_library_id": rid,
                "display_name": m.get("display_name") or m.get("name") or rid,
                "description": m.get("description") or "",
            })
            seen.add(rid)
    by_lib: dict[str, list] = {}
    prod: dict[str, str] = {}
    for m in models_list:
        if m.get("empty"):
            continue
        # 归属优先取实际库根（平台训练/手动归库写的 library_root_id），兼容老字段
        lib_id = m.get("library_root_id") or m.get("model_library_id")
        if not lib_id or lib_id == m.get("model_id"):
            continue  # 独立模型自视为根，不算挂库
        by_lib.setdefault(lib_id, []).append(m)
        if m.get("status") == "production_ready":
            prod[lib_id] = m.get("display_name") or m.get("model_id", "")
    for lib in libs:
        lib["model_count"] = len(by_lib.get(lib["model_library_id"], []))
        lib["production_model_name"] = prod.get(lib["model_library_id"], "")
    return {"libraries": libs}

@router.post("/libraries")
async def create_library(req: LibraryRequest):
    """新建/覆盖算法库（upsert 语义：同 id 覆盖）"""
    try:
        import re
        from datetime import datetime
        lib_id = req.model_library_id.strip()
        if not lib_id:
            lib_id = f"lib_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        lib_id = re.sub(r"[^a-z0-9_]+", "_", lib_id.lower()).strip("_")
        if not lib_id:
            raise ValueError("model_library_id 非法：需为小写字母/数字/下划线")
        await meta_store.upsert_model_library({
            "model_library_id": lib_id,
            "display_name": req.display_name or lib_id,
            "description": req.description or "",
            "business_scenario": req.business_scenario or "",
            "base_algorithm": req.base_algorithm or "",
            "architecture_family": req.architecture_family or "",
            "label_dict_id": req.label_dict_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        return await meta_store.get_model_library(lib_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.delete("/libraries/{library_id}")
async def delete_library(library_id: str):
    """删除算法库（仅解绑记录，不删除挂接模型；模型 model_library_id 留空降级为独立模型）"""
    existed = await meta_store.get_model_library(library_id)
    if not existed:
        raise HTTPException(404, "Model library not found")
    await meta_store.delete_model_library(library_id)
    return {"ok": True, "model_library_id": library_id}

# 注意：静态路由必须定义在 /{model_id} 参数路由之前，否则会被参数路由拦截
@router.get("/similar-scan")
async def find_similar_models(min_similarity: float = Query(0.5, ge=0.0, le=1.0)):
    """2.7 相似模型排查：按标签字典类别名相似度扫描全部模型对（含自动选主建议）"""
    return await model_service.find_similar_models(min_similarity)

class MergeRequest(BaseModel):
    main_model_id: str
    merged_model_ids: list[str]
    reason: Optional[str] = None

@router.post("/merge")
async def merge_models(request: MergeRequest):
    """2.7 合并相似模型：差集类别并入主字典、数据集重归属、历史版本保留为分支、日志可回滚"""
    try:
        return await model_service.merge_models(request.main_model_id, request.merged_model_ids, request.reason)
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.get("/merge-log")
async def merge_logs(limit: int = Query(20, ge=1, le=100)):
    """2.7 合并日志（回滚依据）"""
    return await model_service.merge_log(limit)

class RollbackMergeRequest(BaseModel):
    log_index: int = -1   # -1 = 最近一次

@router.post("/rollback-merge")
async def rollback_merge(request: RollbackMergeRequest):
    """2.7 回滚合并：还原数据集归属、撤销 merged_into 标记"""
    try:
        return await model_service.rollback_merge(request.log_index)
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.get("/production")
async def list_production_models(business: Optional[str] = None):
    """列出当前在役的生产模型（status=production_ready），可按业务/算法类型过滤"""
    return await model_service.list_production_models(business)

@router.get("/publish/pending")
async def list_pending_publish(business: Optional[str] = None):
    """P8 半自动发布：待确认发布的模型（守门员已通过，生产候选）"""
    return await model_service.list_pending_publish(business)

class ConfirmPublishRequest(BaseModel):
    force: Optional[bool] = False   # 冷却期内强制确认（跳过冷却判定）

@router.post("/publish/{model_id}")
async def confirm_publish(model_id: str, request: Optional[ConfirmPublishRequest] = None):
    """P8 半自动发布：管理员确认发布。冷却期（PUBLISH_CONFIRM_COOLDOWN_HOURS）内默认拒绝，
    传 force=true 跳过冷却判定。确认后该模型为唯一生产版本，同业务其他已发布模型自动 superseded。
    """
    from datetime import datetime, timedelta
    detail = await model_service.get_model(model_id)
    if not detail:
        raise HTTPException(404, "Model not found")
    if detail.get("status") != "production_ready":
        raise HTTPException(400, f"模型 {model_id} 当前状态 {detail.get('status')}，不可确认发布（需先通过守门员）")

    cooldown_h = settings.PUBLISH_CONFIRM_COOLDOWN_HOURS
    ready_at = detail.get("ready_at") or ""
    force_confirm = bool((request or ConfirmPublishRequest()).force)
    if cooldown_h > 0 and ready_at and not force_confirm:
        try:
            ready_dt = datetime.fromisoformat(ready_at.replace("Z", "+00:00"))
            if datetime.now() - ready_dt < timedelta(hours=cooldown_h):
                raise HTTPException(
                    400,
                    f"仍在 {cooldown_h} 小时冷却期内（ready_at={ready_at}），"
                    f"确认发布需等待冷却期结束或传 force=true 强制确认",
                )
        except ValueError:
            pass  # ready_at 解析失败不阻断（历史模型无冷却要求）
    try:
        return await model_service.confirm_publish(model_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

# ⚠️ B-07 路由顺序陷阱：自本行起为通配路由 `/{model_id}`。
# FastAPI 按注册顺序匹配，新增任何 GET 固定路径（如 /xxx）必须写在本注释**之前**，
# 否则会被 `/{model_id}` 吞掉（model_id="xxx"），表现为诡异的 404/错数据。
@router.get("/{model_id}")
async def get_model(model_id: str):
    """获取模型详情"""
    result = await model_service.get_model(model_id)
    if not result:
        raise HTTPException(404, "Model not found")
    return result

@router.put("/{model_id}")
async def update_model(model_id: str, request: UpdateModelRequest):
    """更新模型信息（含阶段0：唯一 code / 中文名 / 启用状态）"""
    try:
        result = await model_service.update_model(model_id, request)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not result:
        raise HTTPException(404, "Model not found")
    return result

@router.get("/{model_id}/labels")
async def get_model_labels(model_id: str):
    """获取模型统一标签字典（五字段：index/english_code/chinese_name/chinese_desc/ai_prompt_en）。

    不存在时自动从 model.json 的 classes 初始化。
    """
    result = await model_service.get_labels_dict(model_id)
    if not result:
        raise HTTPException(404, "Model not found")
    return result

@router.get("/{model_id}/labels/usage")
async def get_model_labels_usage(model_id: str):
    """标签引用统计：每个类别被多少张图标注、被多少个数据集引用，以及整本字典是否已锁死。

    前端用它把"在用"的英文名格子锁住并显示 已用/未用 标识。
    """
    result = await model_service.get_labels_usage(model_id)
    if result is None:
        raise HTTPException(404, "Model not found")
    return result

@router.put("/{model_id}/labels")
async def update_model_labels(model_id: str, request: UpdateLabelsRequest):
    """保存模型标签字典（全量覆写；追加禁删保护：在用标签禁止删除/重命名/重排）"""
    try:
        labels = [item.model_dump() for item in request.labels]
        result = await model_service.update_labels_dict(model_id, labels)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not result:
        raise HTTPException(404, "Model not found")
    return result

class SuggestLabelsRequest(BaseModel):
    dataset_id: str
    limit: Optional[int] = 3

@router.post("/{model_id}/labels/suggest")
async def suggest_model_labels(model_id: str, request: SuggestLabelsRequest):
    """AI 识别图片新类别：取该数据集困难/空白样本图发千问 VL，
    返回已知标签之外的新标签候选（四字段），用户采纳后追加进字典"""
    return await model_service.suggest_labels_from_dataset(model_id, request.dataset_id, request.limit)

class AdoptSuggestRequest(BaseModel):
    suggestions: list[dict]  # [{english_code, chinese_name, chinese_desc}]

@router.post("/{model_id}/labels/suggest/adopt")
async def adopt_model_labels(model_id: str, request: AdoptSuggestRequest):
    """采纳 AI 建议的新标签：追加到该模型标签字典末尾（跳过已存在项）"""
    result = await model_service.adopt_suggested_labels(model_id, request.suggestions)
    if result is None:
        raise HTTPException(404, "Model not found")
    return result

class SyncLabelsRequest(BaseModel):
    dry_run: bool = True

@router.post("/{model_id}/labels/sync-annotations")
async def sync_model_labels_to_annotations(model_id: str, request: SyncLabelsRequest):
    """把标签字典的中文名/描述刷进本库已标注数据（dry_run=true 只预览条数与样例，不改写）。

    只改展示文案，不动 class_id / english_code，因此训练产物与已封板数据不受影响；
    正式同步会先备份被改的原始标注 JSON，可用 /labels/sync-annotations/rollback 回退。
    """
    try:
        result = await model_service.sync_labels_to_annotations(model_id, request.dry_run)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if result is None:
        raise HTTPException(404, "Model not found")
    return result

@router.post("/{model_id}/labels/sync-annotations/rollback")
async def rollback_model_labels_sync(model_id: str):
    """回退最近一次标签文案同步（用备份原件覆盖回写）"""
    try:
        result = await model_service.rollback_labels_sync(model_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if result is None:
        raise HTTPException(404, "Model not found")
    return result

async def _attached_datasets(model_id: str) -> list[dict]:
    """挂接到该算法库的数据集清单（P2-5 删库守卫与「库详情—挂载数据集」共用同一口径）

    按库聚合：先把传入的 id 换算成库根记录（版本 id 也归到整库），
    再逐个把数据集的归属（新数据写 library_root_id，老数据只有 model_id）换算成库根后比对
    —— 未归属、归属其它库、以及指向已删除模型的悬空引用一律不出现。
    """
    root_id = model_service.resolve_library_root_id(model_id) or model_id
    res = await DatasetService().list_datasets()
    owned = []
    for d in res["datasets"]:
        owner = d.get("library_root_id") or d.get("model_id")
        if not owner:
            continue
        # 老数据集可能直接挂在库内某个版本模型上：换算成该版本所属库根再比对
        if owner != root_id:
            owner = model_service.resolve_library_root_id(owner)
        if owner == root_id:
            owned.append(d)
    return owned

@router.delete("/{model_id}")
async def delete_model(model_id: str, action: str = "", target_model_id: str = "", force: bool = False):
    """删除模型（P2-5：先查挂接数据集清单，禁止硬删制造孤儿数据集）

    - 挂接为空：直接删除（与原行为一致）；
    - 挂接非空且未传 action：不删，返回 {empty: false, datasets: [...]} 交前端弹确认；
    - action=unbind：挂接数据集逐个置为「未归属」后再删（标注 JSON 原样保留，可重新归属）；
    - action=transfer：逐个转移挂接到 target_model_id 所在库后再删
      （类别序号冲突时报 400，force=true 表示仍坚持转移）。
    """
    attached = await _attached_datasets(model_id)
    if attached and not action:
        return {
            "ok": False, "deleted": False, "empty": False,
            "message": f"该算法库仍挂接 {len(attached)} 个数据集，请选择转移归属或保留为未归属",
            "datasets": [
                {
                    "dataset_id": d.get("dataset_id"),
                    "name": d.get("filename") or d.get("dataset_id"),
                    "image_count": d.get("image_count", 0),
                }
                for d in attached
            ],
        }
    if attached:
        ds_service = DatasetService()
        if action == "unbind":
            for d in attached:
                await ds_service.bind_model(d["dataset_id"], None, None)
        elif action == "transfer":
            target_root = model_service.resolve_library_root_id(target_model_id) or target_model_id
            if not target_model_id or not (settings.REGISTRY_DIR / target_root).exists():
                raise HTTPException(400, f"转移目标库不存在：{target_model_id or '(未指定)'}")
            if target_root == (model_service.resolve_library_root_id(model_id) or model_id):
                raise HTTPException(400, "转移目标不能是被删除的库自身")
            for d in attached:
                try:
                    await ds_service.bind_model(d["dataset_id"], target_root, target_root, force=force)
                except ValueError as e:
                    raise HTTPException(400, str(e))
        else:
            raise HTTPException(400, f"不支持的 action：{action}（仅支持 unbind / transfer）")
    result = await model_service.delete_model(model_id)
    if not result:
        raise HTTPException(404, "Model not found")
    result.update({
        "empty": not attached, "deleted": True,
        "action": action or None, "handled_datasets": len(attached),
    })
    return result

@router.get("/{model_id}/datasets")
async def get_model_datasets(model_id: str):
    """1.7 模型仓库：返回归属于该算法库的所有数据集（含状态机字段）"""
    owned = await _attached_datasets(model_id)
    return {"model_id": model_id, "datasets": owned}

@router.get("/{model_id}/queue")
async def get_model_queue(model_id: str):
    """4.2 模型消息队列详情：待打包数据 + 打包历史"""
    return queue_service.get_queue(model_id)

class PackQueueRequest(BaseModel):
    force: Optional[bool] = False

@router.post("/{model_id}/queue/pack")
async def pack_model_queue(model_id: str, request: Optional[PackQueueRequest] = None):
    """4.2 手动触发打包：队列中的数据包立即进入标注页（自动按模型取类别）"""
    return await queue_service.pack_model(model_id)

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
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    library_id: Optional[str] = Form(None),
):
    """上传已有模型（ZIP 包或单个 .pt 权重文件）
    
    Args:
        file: ZIP格式的模型压缩包（包含model.json和weights目录），
              或单个 .pt 权重文件（自动命名为 best.pt 并解析任务/类别）
        name: 可选，模型显示名。留空则用上传文件名去扩展名
              （权重一律落盘为 best.pt，故不传时列表里会都显示 "best"）
        library_id: 可选，归入的算法库根记录 id。留空则入库为「独立模型」
    """
    if not (file.filename.endswith('.zip') or file.filename.endswith('.pt')):
        raise HTTPException(400, "Only .zip or .pt files are allowed")
    
    try:
        result = await model_service.upload_model(file, display_name=name or "", library_id=library_id or "")
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

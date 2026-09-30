"""元数据 DAO：datasets / annotation_tasks / task_items / models / buckets 的增删改查。

设计约定：
- Service 层通过本层与 PostgreSQL 交互元数据（标注内容仍存一图一 JSON 文件）
- 键名与现有 JSON 文件字段完全一致，Service 层无需感知数据库差异
- 每个方法自开会话并提交（serverless 式），需要跨方法同事务时传入 session 参数
"""
import logging
from typing import Optional

from sqlalchemy import delete, func, select, update

from src.db.engine import get_session
from src.db.models import (
    BucketRecord,
    DatasetRecord,
    ModelLibraryRecord,
    ModelRecord,
    StatusMappingRecord,
    TaskItemRecord,
    TaskRecord,
)
from src.db.status import dataset_unified, to_unified

logger = logging.getLogger(__name__)

# ============ 行 <-> dict 映射 ============

_DATASET_META_COLS = [
    "dataset_id", "filename", "archive", "size", "uploaded_at", "status",
    "stage", "training_status", "sealed_at", "trained_at", "last_trained_round",
    "model_id", "version", "image_count", "label_count", "has_labels",
    "split_ratio", "classes", "class_filter", "label_validation",
    "txt_to_json", "model_auto_match", "prepared_at", "annotation_task_id",
    "description", "tags", "bucket_pending", "bucket_sealed",
    "bucket_archived", "bucket_archived_at", "unified_status",
]

_TASK_COLS = ["task_id", "dataset_id", "version", "classes", "model_id",
              "created_at", "imported_annotations", "unified_status"]

_TASK_ITEM_COLS = ["task_id", "image_id", "image_path", "width", "height", "annotated", "has_label"]

_MODEL_COLS = ["model_id", "model_code", "name", "display_name", "business",
               "status", "source", "empty", "classes", "labels_file",
               "description", "tags", "labels", "created_at", "updated_at",
               "unified_status", "model_library_id"]


def _dataset_row_to_meta(row) -> dict:
    meta = {c: getattr(row, c) for c in _DATASET_META_COLS}
    meta.update(row.extra or {})
    return meta


def _meta_to_dataset_row(meta: dict) -> dict:
    row = {c: meta.get(c) for c in _DATASET_META_COLS}
    known = set(_DATASET_META_COLS)
    row["extra"] = {k: v for k, v in meta.items() if k not in known}
    return row


def _task_row_to_meta(row) -> dict:
    meta = {c: getattr(row, c) for c in _TASK_COLS}
    meta.update(row.extra or {})
    return meta


def _meta_to_task_row(meta: dict) -> dict:
    row = {c: meta.get(c) for c in _TASK_COLS}
    known = set(_TASK_COLS)
    row["extra"] = {k: v for k, v in meta.items() if k not in known}
    return row


# ============ datasets ============

async def get_dataset(dataset_id: str) -> Optional[dict]:
    async with get_session() as s:
        row = (await s.execute(
            select(DatasetRecord).where(DatasetRecord.dataset_id == dataset_id)
        )).scalar_one_or_none()
        return _dataset_row_to_meta(row) if row else None


async def list_datasets() -> list[dict]:
    async with get_session() as s:
        rows = (await s.execute(
            select(DatasetRecord).order_by(DatasetRecord.uploaded_at.desc())
        )).scalars().all()
        return [_dataset_row_to_meta(r) for r in rows]


async def upsert_dataset(meta: dict) -> None:
    row_data = _meta_to_dataset_row(meta)
    # M5 双写：统一状态列随旧列一起写入（读侧由 USE_UNIFIED_STATUS 开关控制）
    row_data["unified_status"] = dataset_unified(meta) or row_data.get("unified_status")
    async with get_session() as s:
        await s.execute(
            delete(DatasetRecord).where(DatasetRecord.dataset_id == row_data["dataset_id"])
        )
        s.add(DatasetRecord(**row_data))
        await s.commit()


async def delete_dataset_record(dataset_id: str) -> None:
    async with get_session() as s:
        await s.execute(delete(DatasetRecord).where(DatasetRecord.dataset_id == dataset_id))
        await s.commit()


async def count_datasets() -> int:
    async with get_session() as s:
        return (await s.execute(select(func.count()).select_from(DatasetRecord))).scalar() or 0


# ============ annotation_tasks / task_items ============

async def get_task(task_id: str) -> Optional[dict]:
    async with get_session() as s:
        row = (await s.execute(
            select(TaskRecord).where(TaskRecord.task_id == task_id)
        )).scalar_one_or_none()
        return _task_row_to_meta(row) if row else None


async def upsert_task(meta: dict) -> None:
    row_data = _meta_to_task_row(meta)
    # M5 双写：标注任务统一状态（默认 Annotating）
    if not row_data.get("unified_status"):
        row_data["unified_status"] = to_unified("task", row_data.get("status", ""), "Annotating")
    async with get_session() as s:
        await s.execute(delete(TaskRecord).where(TaskRecord.task_id == row_data["task_id"]))
        s.add(TaskRecord(**row_data))
        await s.commit()


async def list_task_items(task_id: str) -> list[dict]:
    async with get_session() as s:
        rows = (await s.execute(
            select(TaskItemRecord).where(TaskItemRecord.task_id == task_id)
        )).scalars().all()
        return [{c: getattr(r, c) for c in _TASK_ITEM_COLS} for r in rows]


async def replace_task_items(task_id: str, items: list[dict]) -> None:
    """整表替换某任务的图片列表（items 中不含 task_id 时自动补）。

    按 image_id 去重（历史任务中存在同一 image_id 重复条目，前端已按此逻辑去重展示）。
    """
    async with get_session() as s:
        await s.execute(delete(TaskItemRecord).where(TaskItemRecord.task_id == task_id))
        seen = set()
        for it in items:
            image_id = it.get("image_id")
            if not image_id or image_id in seen:
                continue
            seen.add(image_id)
            row = {c: it.get(c) for c in _TASK_ITEM_COLS}
            row["task_id"] = task_id
            s.add(TaskItemRecord(**row))
        await s.commit()


async def remove_task_item(task_id: str, image_id: str) -> bool:
    async with get_session() as s:
        res = await s.execute(
            delete(TaskItemRecord).where(
                TaskItemRecord.task_id == task_id,
                TaskItemRecord.image_id == image_id,
            )
        )
        await s.commit()
        return (res.rowcount or 0) > 0


async def update_task_item_has_label(task_id: str, image_id: str, has_label: bool) -> None:
    """P2：单图标注状态回填（行不存在则新建，保持 PG 影子库与标注 JSON 一致）。"""
    async with get_session() as s:
        row = (await s.execute(
            select(TaskItemRecord).where(
                TaskItemRecord.task_id == task_id,
                TaskItemRecord.image_id == image_id,
            )
        )).scalar_one_or_none()
        if row is None:
            s.add(TaskItemRecord(
                task_id=task_id, image_id=image_id,
                image_path="", width=0, height=0,
                annotated=has_label, has_label=has_label,
            ))
        else:
            row.has_label = has_label
            if has_label:
                row.annotated = True
        await s.commit()


async def mark_task_items_labeled(task_id: str, image_ids: list) -> int:
    """批量回填 has_label（YOLO 标签导入一次可能几百张，逐行 update 太慢）。

    只更新已存在的行；task_items 与 task.json 的 items 同源，正常不会缺失。
    """
    ids = list({i for i in (image_ids or []) if i})
    if not ids:
        return 0
    async with get_session() as s:
        res = await s.execute(
            update(TaskItemRecord)
            .where(TaskItemRecord.task_id == task_id,
                   TaskItemRecord.image_id.in_(ids))
            .values(has_label=True, annotated=True)
        )
        await s.commit()
        return res.rowcount or 0


async def delete_task_record(task_id: str) -> None:
    async with get_session() as s:
        await s.execute(delete(TaskItemRecord).where(TaskItemRecord.task_id == task_id))
        await s.execute(delete(TaskRecord).where(TaskRecord.task_id == task_id))
        await s.commit()


# ============ models ============

def _model_row_to_meta(row) -> dict:
    meta = {c: getattr(row, c) for c in _MODEL_COLS}
    meta.update(row.extra or {})
    return meta


async def get_model(model_id: str) -> Optional[dict]:
    async with get_session() as s:
        row = (await s.execute(
            select(ModelRecord).where(ModelRecord.model_id == model_id)
        )).scalar_one_or_none()
        return _model_row_to_meta(row) if row else None


async def list_models() -> list[dict]:
    async with get_session() as s:
        rows = (await s.execute(select(ModelRecord).order_by(ModelRecord.model_id))).scalars().all()
        return [_model_row_to_meta(r) for r in rows]


async def upsert_model(meta: dict) -> None:
    known = set(_MODEL_COLS)
    row_data = {c: meta.get(c) for c in _MODEL_COLS}
    # M5 双写：模型统一状态（active→training 兜底，存量数据由迁移脚本补齐）
    if not row_data.get("unified_status"):
        row_data["unified_status"] = to_unified("model", row_data.get("status", ""), "")
    row_data["extra"] = {k: v for k, v in meta.items() if k not in known}
    async with get_session() as s:
        await s.execute(delete(ModelRecord).where(ModelRecord.model_id == row_data["model_id"]))
        s.add(ModelRecord(**row_data))
        await s.commit()


async def delete_model_record(model_id: str) -> None:
    async with get_session() as s:
        await s.execute(delete(ModelRecord).where(ModelRecord.model_id == model_id))
        await s.commit()


# ============ buckets ============

async def get_buckets_data() -> dict:
    """返回全量 buckets 数据（对齐 buckets.json 结构：{models: {...}}）。"""
    async with get_session() as s:
        rows = (await s.execute(select(BucketRecord))).scalars().all()
        models = {}
        for r in rows:
            models[r.model_id] = {
                "bucket_version_count": r.bucket_version_count,
                "receiving": r.receiving or {},
                "sealed": r.sealed or [],
            }
        return {"models": models}


async def save_buckets_data(data: dict) -> None:
    """整体保存 buckets 数据（对齐 buckets.json 结构）。"""
    models = data.get("models", {}) or {}
    async with get_session() as s:
        await s.execute(delete(BucketRecord))
        for model_id, b in models.items():
            s.add(BucketRecord(
                model_id=model_id,
                bucket_version_count=b.get("bucket_version_count", 0),
                receiving=b.get("receiving") or {},
                sealed=b.get("sealed") or [],
            ))
        await s.commit()


# ============ model_libraries（Phase 0.3） ============

_LIBRARY_COLS = ["model_library_id", "display_name", "description",
                 "business_scenario", "base_algorithm", "architecture_family",
                 "label_dict_id", "current_production_model_id",
                 "created_at", "updated_at"]


def _library_row_to_meta(row) -> dict:
    return {c: getattr(row, c) for c in _LIBRARY_COLS}


async def list_model_libraries() -> list[dict]:
    async with get_session() as s:
        rows = (await s.execute(
            select(ModelLibraryRecord).order_by(ModelLibraryRecord.model_library_id)
        )).scalars().all()
        return [_library_row_to_meta(r) for r in rows]


async def get_model_library(model_library_id: str) -> Optional[dict]:
    async with get_session() as s:
        row = (await s.execute(
            select(ModelLibraryRecord).where(
                ModelLibraryRecord.model_library_id == model_library_id)
        )).scalar_one_or_none()
        return _library_row_to_meta(row) if row else None


async def upsert_model_library(meta: dict) -> None:
    row_data = {c: meta.get(c) for c in _LIBRARY_COLS}
    async with get_session() as s:
        await s.execute(delete(ModelLibraryRecord).where(
            ModelLibraryRecord.model_library_id == row_data["model_library_id"])
        )
        s.add(ModelLibraryRecord(**row_data))
        await s.commit()


async def delete_model_library(model_library_id: str) -> None:
    async with get_session() as s:
        await s.execute(delete(ModelLibraryRecord).where(
            ModelLibraryRecord.model_library_id == model_library_id)
        )
        await s.commit()


# ============ status_mapping（M5 映射表种子/查询） ============

async def seed_status_mapping() -> None:
    """写入/刷新 status_mapping 表种子数据（幂等，startup 调用）。"""
    from src.db.status import STATUS_MAPPING_SEED
    async with get_session() as s:
        for domain, old_value, new_value, rule in STATUS_MAPPING_SEED:
            await s.execute(delete(StatusMappingRecord).where(
                StatusMappingRecord.domain == domain,
                StatusMappingRecord.old_value == old_value,
            ))
            s.add(StatusMappingRecord(
                domain=domain, old_value=old_value,
                new_value=new_value, rule=rule,
            ))
        await s.commit()


async def list_status_mappings(domain: str = "") -> list[dict]:
    async with get_session() as s:
        q = select(StatusMappingRecord)
        if domain:
            q = q.where(StatusMappingRecord.domain == domain)
        rows = (await s.execute(q.order_by(StatusMappingRecord.domain))).scalars().all()
        return [{"domain": r.domain, "old_value": r.old_value,
                 "new_value": r.new_value, "rule": r.rule} for r in rows]
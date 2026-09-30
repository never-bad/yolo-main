"""JSON 主存储 → PostgreSQL 影子库的统一收口层。

为什么收口：
- 平台以 JSON 文件为唯一真值源，PG 只是可重建的查询影子库；影子库写失败绝不能影响业务主流程。
- 各业务写点自行手抄字段调 meta_store.upsert_* 有两个必然后果：漏写点、字段口径与 meta_store
  的列清单漂移（例如 datetime 被扫进 JSONB extra 直接抛序列化异常）。因此本层一律
  "落盘后回读 JSON 真值再写库"，调用点只剩一行 `await shadow.xxx(id)`。
- 回读还顺带消掉了跨进程写：训练子进程 train_script.py 会直接改数据集 meta.json，
  进程内双写钩不到，靠 refresh_dataset 的 mtime 水位在读侧自愈。

约定（三条硬规则）：
1. 所有公开入口经 _guard：PG_ENABLED 关闭时直接返回，异常降级为 warning，永不抛出。
2. 路径规则与 scripts/migrate_to_pg.py 完全一致，保证"双写结果 == 全量重灌结果"。
3. task.json 的 items 从不进 TaskRecord.extra（体量大且已由 task_items 承载），
   整表替换 items 只允许在 create_task 时用 —— 已存在任务的 has_label 是运行时回填的，
   用 task.json 快照整表覆盖会把 PG 里已标注的标记冲掉。
"""
import json
import logging
from pathlib import Path
from typing import Callable, Iterable, Optional

from src.core.settings import settings
from src.db import meta_store

logger = logging.getLogger(__name__)

# "dataset:{id}" / "model:{id}" / "task:{id}" → 上次成功同步时源文件的 mtime
_WATERMARK: dict[str, float] = {}


# ============ 内部工具 ============

def _dataset_meta_path(dataset_id: str) -> Path:
    return Path(settings.DATASETS_DIR) / dataset_id / "meta.json"


def _task_meta_path(task_id: str) -> Path:
    return Path(settings.ANNOTATIONS_DIR) / task_id / "task.json"


def _model_meta_path(model_id: str) -> Path:
    return Path(settings.REGISTRY_DIR) / model_id / "model.json"


def _buckets_path() -> Path:
    return Path(settings.DATA_DIR) / "buckets.json"


def _read_json(path: Path) -> Optional[dict]:
    """读真值文件；缺失或损坏返回 None（读不到真值就不写库，避免脏数据进影子库）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except Exception:  # noqa: BLE001
        return None


async def _guard(op: str, key: str, factory: Callable[[], any]) -> bool:
    """统一兜底：开关判断 + 吞异常。factory 返回待执行的协程，避免未 await 警告。"""
    if not getattr(settings, "PG_ENABLED", False):
        return False
    try:
        await factory()
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[shadow] {op} {key} failed: {e}")
        return False


def _mark_synced(wm_key: str, path: Path) -> None:
    m = _mtime(path)
    if m is not None:
        _WATERMARK[wm_key] = m


# ============ datasets ============

async def sync_dataset(dataset_id: str) -> bool:
    """按真值文件把数据集 meta 同步进 PG（写盘点落盘后调用）。"""
    if not dataset_id:
        return False
    wm = f"dataset:{dataset_id}"
    path = _dataset_meta_path(dataset_id)

    def _factory():
        meta = _read_json(path)
        if meta is None:
            logger.debug(f"[shadow] dataset {dataset_id} meta.json unreadable, skip")
            return _noop()
        if not meta.get("dataset_id"):
            meta["dataset_id"] = dataset_id
        _mark_synced(wm, path)
        return meta_store.upsert_dataset(meta)

    return await _guard("sync_dataset", dataset_id, _factory)


async def _noop() -> None:
    return None


async def drop_dataset(dataset_id: str) -> bool:
    if not dataset_id:
        return False
    _WATERMARK.pop(f"dataset:{dataset_id}", None)
    return await _guard("drop_dataset", dataset_id,
                        lambda: meta_store.delete_dataset_record(dataset_id))


async def refresh_dataset(dataset_id: str) -> bool:
    """读侧自愈：真值文件 mtime 变了才写库（用于跨进程写入与热路径兜底）。"""
    if not dataset_id:
        return False
    path = _dataset_meta_path(dataset_id)
    m = _mtime(path)
    if m is None:
        return False
    if _WATERMARK.get(f"dataset:{dataset_id}") == m:
        return False
    return await sync_dataset(dataset_id)


async def refresh_datasets(dataset_ids: Iterable[str]) -> int:
    """批量 mtime 水位刷新，返回实际写库条数。"""
    done = 0
    for ds_id in {d for d in dataset_ids if d}:
        if await refresh_dataset(ds_id):
            done += 1
    return done


async def refresh_all_datasets() -> int:
    """全量扫描 datasets/*/meta.json 按水位补齐差集。

    用于一次操作波及多个数据集且调用方无法枚举的场景（灌桶/封板会在桶账本之外
    同时新建快照数据集、改写源包 meta）。只 stat 不改动的文件不写库，代价是目录遍历。
    """
    root = Path(settings.DATASETS_DIR)
    if not root.exists():
        return 0
    ids = [p.parent.name for p in root.glob("*/meta.json")]
    return await refresh_datasets(ids)


# ============ annotation_tasks / task_items ============

async def sync_task(task_id: str, with_items: bool = False) -> bool:
    """同步任务元数据。

    with_items=True 仅在 create_task 时使用（整表替换 task_items）。
    对已存在的任务只能同步任务级字段——items 里的 has_label 由标注/导入路径逐行回填，
    task.json 快照并不携带该字段。
    """
    if not task_id:
        return False
    wm = f"task:{task_id}"
    path = _task_meta_path(task_id)

    def _factory():
        meta = _read_json(path)
        if meta is None:
            logger.debug(f"[shadow] task {task_id} task.json unreadable, skip")
            return _noop()
        if not meta.get("task_id"):
            meta["task_id"] = task_id
        items = meta.pop("items", None) or []
        _mark_synced(wm, path)
        return _apply_task(meta, items if with_items else None)

    return await _guard("sync_task", task_id, _factory)


async def _apply_task(meta: dict, items: Optional[list]) -> None:
    await meta_store.upsert_task(meta)
    if items is not None:
        await meta_store.replace_task_items(meta["task_id"], items)


async def drop_task(task_id: str) -> bool:
    if not task_id:
        return False
    _WATERMARK.pop(f"task:{task_id}", None)
    return await _guard("drop_task", task_id,
                        lambda: meta_store.delete_task_record(task_id))


async def remove_task_item(task_id: str, image_id: str) -> bool:
    if not task_id or not image_id:
        return False
    return await _guard("remove_task_item", f"{task_id}/{image_id}",
                        lambda: meta_store.remove_task_item(task_id, image_id))


async def mark_items_labeled(task_id: str, image_ids: list) -> bool:
    """YOLO 标签批量导入后回填 has_label（单条 bulk UPDATE，避免逐图往返）。"""
    ids = [i for i in (image_ids or []) if i]
    if not task_id or not ids:
        return False
    return await _guard("mark_items_labeled", f"{task_id}x{len(ids)}",
                        lambda: meta_store.mark_task_items_labeled(task_id, ids))


# ============ models ============

async def sync_model(model_id: str) -> bool:
    if not model_id:
        return False
    wm = f"model:{model_id}"
    path = _model_meta_path(model_id)

    def _factory():
        meta = _read_json(path)
        if meta is None:
            logger.debug(f"[shadow] model {model_id} model.json unreadable, skip")
            return _noop()
        if not meta.get("model_id"):
            meta["model_id"] = model_id
        # labels_dict.json 是标签真值，与迁移脚本同口径并入 labels 列
        dict_file = path.parent / "labels_dict.json"
        if dict_file.exists():
            labels = _read_json(dict_file)
            if labels is not None:
                meta["labels"] = labels
                try:
                    meta["labels_file"] = str(dict_file.relative_to(Path(settings.REGISTRY_DIR)))
                except Exception:  # noqa: BLE001
                    meta["labels_file"] = str(dict_file)
        _mark_synced(wm, path)
        return meta_store.upsert_model(meta)

    return await _guard("sync_model", model_id, _factory)


async def drop_model(model_id: str) -> bool:
    if not model_id:
        return False
    _WATERMARK.pop(f"model:{model_id}", None)
    return await _guard("drop_model", model_id,
                        lambda: meta_store.delete_model_record(model_id))


# ============ buckets ============

async def sync_buckets() -> bool:
    """桶账本是单文件整体覆盖，同步即读 buckets.json 全量重建 BucketRecord。"""
    path = _buckets_path()

    def _factory():
        data = _read_json(path)
        if data is None:
            return _noop()
        return meta_store.save_buckets_data(data)

    return await _guard("sync_buckets", "buckets", _factory)

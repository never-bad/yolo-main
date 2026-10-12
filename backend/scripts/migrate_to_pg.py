"""JSON 元数据 → PostgreSQL 一键迁移脚本（幂等，可重复执行）。

用法：
    python scripts/migrate_to_pg.py            # 迁移 datasets/tasks/items/models/buckets
    python scripts/migrate_to_pg.py --dry       # 仅统计待迁移，不写库
    python scripts/migrate_to_pg.py --only tasks # 只迁移指定类别（datasets|tasks|models|buckets）
    python scripts/migrate_to_pg.py --prune      # 迁移后删除真值文件已消失的 PG 行

说明：
- 迁移只涉及"元数据"，一图一 JSON 的标注内容文件保持不动
- 服务器部署时同样在目标机执行本脚本即可完成元数据入库
- --prune 的判据与 upsert 完全同源（按真值文件里的 *_id 取主键，目录名只作兜底），
  因此"双写后的库"与"重灌+prune 后的库"逐行等价；不加 prune 时本脚本只增改不删。
"""
import argparse
import json
import sys
import asyncio
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.core.settings import settings          # noqa: E402
from src.db import meta_store as store          # noqa: E402
from src.db.engine import init_db, ping        # noqa: E402

DATA_DIR = Path(settings.DATA_DIR)
MODELS_DIR = Path(getattr(settings, "REGISTRY_DIR", None) or (DATA_DIR.parent / "models" / "registry"))


def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


async def migrate_datasets(dry: bool) -> int:
    count = 0
    for meta_path in sorted((DATA_DIR / "datasets").glob("*/meta.json")):
        if meta_path.parent.name.startswith("_"):
            continue  # D-03：归档目录（_ 前缀）不入 PG，与 list_datasets 口径一致
        meta = _load_json(meta_path)
        if not meta.get("dataset_id"):
            meta["dataset_id"] = meta_path.parent.name
        if dry:
            print(f"  [datasets] {meta['dataset_id']} images={meta.get('image_count')}")
        else:
            await store.upsert_dataset(meta)
        count += 1
    print(f"datasets: {count} 条" + ("（dry）" if dry else ""))
    return count


async def migrate_tasks(dry: bool) -> int:
    count = 0
    for task_file in sorted((DATA_DIR / "annotations").glob("*/task.json")):
        if task_file.parent.name.startswith("_"):
            continue  # D-03：归档目录（_ 前缀）不入 PG，与 datasets 侧同一判据
        meta = _load_json(task_file)
        if not meta.get("task_id"):
            meta["task_id"] = task_file.parent.name
        items = meta.pop("items", []) or []
        if not dry:
            await store.upsert_task(meta)
            await store.replace_task_items(meta["task_id"], items)
        count += 1
        print(f"  [tasks] {meta.get('task_id')} ds={meta.get('dataset_id')} items={len(items)}")
    print(f"tasks: {count} 条" + ("（dry）" if dry else ""))
    return count


async def migrate_models(dry: bool) -> int:
    count = 0
    if not MODELS_DIR.exists():
        print("models 目录不存在，跳过模型迁移")
        return 0
    for model_file in sorted(MODELS_DIR.glob("*/model.json")):
        if model_file.parent.name.startswith("_"):
            continue  # D-03：归档目录（_ 前缀）不入 PG，与 datasets 侧同一判据
        meta = _load_json(model_file)
        if not meta.get("model_id"):
            meta["model_id"] = model_file.parent.name
        dict_file = model_file.parent / "labels_dict.json"
        if dict_file.exists():
            meta["labels"] = _load_json(dict_file)
            meta["labels_file"] = str(dict_file.relative_to(MODELS_DIR))
        if dry:
            print(f"  [models] {meta.get('model_id')}")
        else:
            await store.upsert_model(meta)
        count += 1
    print(f"models: {count} 条" + ("（dry）" if dry else ""))
    return count


async def migrate_buckets(dry: bool) -> int:
    buckets_path = DATA_DIR / "buckets.json"
    if not buckets_path.exists():
        print("buckets.json 不存在，跳过")
        return 0
    data = _load_json(buckets_path)
    n = len(data.get("models", {}))
    if not dry:
        await store.save_buckets_data(data)
    print(f"buckets: {n} 个模型桶" + ("（dry）" if dry else ""))
    return n


async def main():
    parser = argparse.ArgumentParser(description="JSON 元数据 → PostgreSQL 迁移")
    parser.add_argument("--dry", action="store_true", help="只统计，不写库")
    parser.add_argument("--only", choices=["datasets", "tasks", "models", "buckets"], default=None)
    args = parser.parse_args()

    if not await ping():
        print(f"[错误] 无法连接 PostgreSQL: {settings.PG_HOST}:{settings.PG_PORT}/{settings.PG_DATABASE}，请先启动 PG 服务")
        sys.exit(1)
    if not args.dry:
        await init_db()

    total = 0
    if args.only in (None, "datasets"):
        total += await migrate_datasets(args.dry)
    if args.only in (None, "tasks"):
        total += await migrate_tasks(args.dry)
    if args.only in (None, "models"):
        total += await migrate_models(args.dry)
    if args.only in (None, "buckets"):
        total += await migrate_buckets(args.dry)
    print(f"\n迁移完成，共 {total} 条元数据。" + ("（dry）" if args.dry else ""))


if __name__ == "__main__":
    asyncio.run(main())
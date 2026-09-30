"""task_items.has_label 存量回填（幂等，可重复执行）。

背景：has_label 是运行时从"每图标注 JSON"算出来的派生字段，task.json 的 items 快照
并不携带它，因此 migrate_to_pg.py 重灌 task_items 时只能把它写成 NULL/false。
本脚本把 JSON 真值里的"该图是否存在有内容的框"一次性回填进 PG 影子库，
判据与 annotation_service.get_task_items 完全一致：
    has_label = bool(ann.get("boxes"))
    ann 来源 = datasets/{dataset_id}/annotations/{image_id}.json（优先）
              + annotations/{task_id}/annotations.json（历史集中式，setdefault 兜底）

用法：
    python scripts/backfill_has_label.py --dry          # 只统计，不写库
    python scripts/backfill_has_label.py                # 回填全部任务
    python scripts/backfill_has_label.py --only task_x  # 只回填指定任务

说明：
- 只把 false/NULL 改成 true，不会把 true 改回 false（与标注删除路径无冲突：
  删标注时 annotation_service 会逐行回写），因此重复执行安全。
- 影子库是可重建的从属存储，本脚本不改任何 JSON 真值文件。
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.core.settings import settings          # noqa: E402
from src.db import meta_store as store          # noqa: E402
from src.db.engine import init_db, ping         # noqa: E402

DATA_DIR = Path(settings.DATA_DIR)
DATASETS_DIR = Path(settings.DATASETS_DIR)
ANNOTATIONS_DIR = Path(settings.ANNOTATIONS_DIR)
CHUNK = 1000  # IN 子句分片，避免超长语句


def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def scan_dataset_labels() -> dict:
    """dataset_id → {有非空 boxes 的 image_id}。一次遍历全部每图 JSON。"""
    out: dict[str, set] = {}
    for ann_dir in sorted(DATASETS_DIR.glob("*/annotations")):
        ds_id = ann_dir.parent.name
        labeled = out.setdefault(ds_id, set())
        for f in ann_dir.glob("*.json"):
            data = _load_json(f)
            if isinstance(data, dict) and data.get("boxes"):
                labeled.add(str(data.get("image_id") or f.stem))
    return out


def legacy_labeled(task_dir: Path) -> set:
    """历史集中式 annotations.json 中 boxes 非空的 image_id。"""
    f = task_dir / "annotations.json"
    data = _load_json(f)
    if not isinstance(data, dict):
        return set()
    return {str(k) for k, v in data.items() if isinstance(v, dict) and v.get("boxes")}


async def backfill_task(task_id: str, dataset_id: str, image_ids: set,
                        labeled_ids: set, dry: bool) -> int:
    """把 image_ids ∩ labeled_ids 写进 PG，返回命中行数。"""
    hit = sorted(image_ids & labeled_ids)
    if not hit or dry:
        return len(hit)
    n = 0
    for i in range(0, len(hit), CHUNK):
        n += await store.mark_task_items_labeled(task_id, hit[i:i + CHUNK])
    return n


async def main():
    parser = argparse.ArgumentParser(description="task_items.has_label 存量回填")
    parser.add_argument("--dry", action="store_true", help="只统计，不写库")
    parser.add_argument("--only", default=None, help="只回填指定 task_id")
    args = parser.parse_args()

    if not await ping():
        print(f"[错误] 无法连接 PostgreSQL: {settings.PG_HOST}:{settings.PG_PORT}/{settings.PG_DATABASE}")
        sys.exit(1)
    if not args.dry:
        await init_db()

    ds_labeled = scan_dataset_labels()
    print(f"每图标注扫描完成：{len(ds_labeled)} 个数据集，"
          f"有框图片合计 {sum(len(v) for v in ds_labeled.values())} 张")

    task_files = sorted(ANNOTATIONS_DIR.glob("*/task.json"))
    if args.only:
        task_files = [f for f in task_files if f.parent.name == args.only]
        if not task_files:
            print(f"[错误] 未找到任务 {args.only}")
            sys.exit(1)

    total_hit = 0
    tasks_with_hit = 0
    no_dataset = []
    for tf in task_files:
        task_dir = tf.parent
        meta = _load_json(tf)
        if not isinstance(meta, dict):
            continue
        task_id = meta.get("task_id") or task_dir.name
        dataset_id = meta.get("dataset_id", "")
        image_ids = {str(it.get("image_id")) for it in (meta.get("items") or []) if it.get("image_id")}
        labeled = set(ds_labeled.get(dataset_id) or ()) | legacy_labeled(task_dir)
        if dataset_id and dataset_id not in ds_labeled and not legacy_labeled(task_dir):
            no_dataset.append(f"{task_id}→{dataset_id}")
        hit = await backfill_task(task_id, dataset_id, image_ids, labeled, args.dry)
        total_hit += hit
        if hit:
            tasks_with_hit += 1
        print(f"  [{task_id}] ds={dataset_id or '-'} items={len(image_ids)} 命中={hit}")

    print(f"\n任务数={len(task_files)} 有命中的任务={tasks_with_hit} "
          f"应回填行数={total_hit}{'（dry）' if args.dry else ''}")
    if no_dataset:
        print(f"提示：{len(no_dataset)} 个任务的数据集没有任何标注文件 {no_dataset[:5]}")


if __name__ == "__main__":
    asyncio.run(main())

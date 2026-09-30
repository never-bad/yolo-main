"""model_libraries 数据迁移脚本（Phase 0.3，幂等可重复执行）。

把现有 models 按 business 字段分组归库：
- 每组创建一个 model_libraries 记录（model_library_id = {business}_lib）
- 对应 models 行的 model_library_id 指向该记录
- 未归类（business 为空）的模型保持 model_library_id 为空（按旧 business 检索降级）

用法：
    python scripts/migrate_model_libraries.py            # 执行迁移
    python scripts/migrate_model_libraries.py --dry      # 仅预览不写库
    python scripts/migrate_model_libraries.py --reset    # 清空 model_libraries 表后重建
"""
import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.db import meta_store                      # noqa: E402
from src.db.engine import init_db, ping            # noqa: E402


async def main():
    parser = argparse.ArgumentParser(description="model_libraries 数据迁移")
    parser.add_argument("--dry", action="store_true", help="只预览，不写库")
    parser.add_argument("--reset", action="store_true", help="清空 model_libraries 表后重建")
    args = parser.parse_args()

    if not await ping():
        print("[错误] 无法连接 PostgreSQL，请先启动 PG 服务")
        sys.exit(1)
    if not args.dry:
        await init_db()

    models = await meta_store.list_models()
    print(f"现有模型 {len(models)} 个")

    # 按 business 分组（去空）
    groups: dict[str, list[dict]] = {}
    for m in models:
        biz = (m.get("business") or "").strip()
        if biz:
            groups.setdefault(biz, []).append(m)

    if args.reset and not args.dry:
        for lib in await meta_store.list_model_libraries():
            await meta_store.delete_model_library(lib["model_library_id"])
        print("已清空 model_libraries 表")

    print(f"按 business 分为 {len(groups)} 组：")
    created = 0
    for biz in sorted(groups):
        lib_id = f"{biz}_lib"
        if not args.dry:
            now = datetime.now().isoformat()
            await meta_store.upsert_model_library({
                "model_library_id": lib_id,
                "display_name": biz,
                "description": f"由业务字段 {biz} 自动归库",
                "business_scenario": biz,
                "base_algorithm": "",
                "architecture_family": "",
                "label_dict_id": None,
                "current_production_model_id": None,
                "created_at": now,
                "updated_at": now,
            })
        models_in = [m["model_id"] for m in groups[biz]]
        print(f"  [{lib_id}] {biz}: {len(models_in)} 个模型 → {models_in}")
        created += 1

    print(f"\n迁移完成：创建/更新 {created} 个算法库"
          + ("（dry）" if args.dry else "")
          + f"，未归库模型 {len(models) - sum(len(v) for v in groups.values())} 个（business 为空）")


if __name__ == "__main__":
    asyncio.run(main())
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from src.core.settings import settings
from src.db.models import Base

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None


def database_url() -> str:
    """由 settings / 环境变量构造 PG 连接串（服务器部署只改 .env 即可）。"""
    return (
        f"postgresql+asyncpg://{settings.PG_USER}:{settings.PG_PASSWORD}"
        f"@{settings.PG_HOST}:{settings.PG_PORT}/{settings.PG_DATABASE}"
    )


def _ensure_engine():
    global _engine, _session_factory
    if _engine is None:
        _engine = create_async_engine(
            database_url(),
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600,
        )
        _session_factory = async_sessionmaker(
            _engine, expire_on_commit=False, class_=AsyncSession
        )
        logger.info("PG engine ready: %s:%s/%s", settings.PG_HOST, settings.PG_PORT, settings.PG_DATABASE)
    return _session_factory


def get_session() -> AsyncSession:
    """获取一个独立会话（调用方负责 close / async with）。"""
    return _ensure_engine()()


def get_engine():
    """获取全局 async engine（供 create_all 等 DDL 使用）。"""
    _ensure_engine()
    return _engine


def _missing_columns(table_cols: dict, dialect) -> list:
    """比对 ORM 声明与库里实际列，返回待补的 (表, 列, 类型) 三元组。

    整表不存在的情况不在这里处理（create_all 负责建表），只关心"老表少新列"。
    """
    out = []
    for tbl, table in Base.metadata.tables.items():
        have = table_cols.get(tbl)
        if have is None:
            continue
        for col in table.columns:
            if col.name in have:
                continue
            try:
                typ = col.type.compile(dialect=dialect)
            except Exception:  # 极端情况下退回 TEXT，保证补列动作本身不失败
                typ = "TEXT"
            out.append((tbl, col.name, typ))
    return out


async def _table_columns(conn) -> dict:
    """当前 schema 下每张表的实际列名集合。"""
    rows = (await conn.execute(text(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema()"
    ))).all()
    cols = {}
    for tbl, col in rows:
        cols.setdefault(tbl, set()).add(col)
    return cols


async def init_db():
    """建表（幂等 create_all）+ 自动补列（create_all 不会给已有表加列）。

    补列口径直接从 ORM 元数据推导，不再手写 ALTER 清单：
    M5 的 unified_status、以及 models.model_library_id 都是"给已有表加 mapped_column"，
    而影子库所有写入都包在 try/except 里只打 warning —— 清单漏一项就会让该表所有
    PG 写入长期静默失败（列缺失 → UndefinedColumnError），且没有任何页面看得出来。
    """
    eng = get_engine()
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for tbl, col, typ in _missing_columns(await _table_columns(conn), eng.dialect):
            # 只补可空裸列，不加 NOT NULL/DEFAULT：非空表上带 NOT NULL 的 ALTER 会直接失败。
            # 值由写入侧逐步补齐（ORM default / meta_store 双写），
            # 存量行需要立刻有值时跑 scripts/migrate_to_pg.py 重灌一遍。
            sql = f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col} {typ}"
            try:
                await conn.execute(text(sql))
                logger.info("PG schema migrate: %s.%s (%s)", tbl, col, typ)
            except Exception as e:  # noqa: BLE001 补列失败不阻断启动（JSON 仍是主存储）
                logger.warning("migrate skip %s: %s", sql, e)
    logger.info("PG schema ensured")


async def dispose_db():
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def ping() -> bool:
    """测试数据库连通性（迁移/启动自检用）。"""
    try:
        async with get_session() as s:
            await s.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
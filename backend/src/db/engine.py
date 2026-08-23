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


async def init_db():
    """建表（幂等 create_all）+ 轻量列迁移（create_all 不会给已有表加列）。"""
    eng = get_engine()
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # 迁移：task_items 新增列（has_label/annotated 等由 ORM 引入，历史表缺列需幂等补上，
        # 否则保存标注时的 PG 回填会因 UndefinedColumnError 失败）
        migrate_sql = [
            "ALTER TABLE task_items ADD COLUMN IF NOT EXISTS has_label BOOLEAN DEFAULT FALSE",
            "ALTER TABLE task_items ADD COLUMN IF NOT EXISTS annotated BOOLEAN DEFAULT FALSE",
        ]
        for sql in migrate_sql:
            try:
                await conn.execute(text(sql))
            except Exception as e:  # noqa: BLE001 表不存在等场景静默跳过，不阻断启动
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
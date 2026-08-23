import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.core.settings import settings
from src.api.routes import datasets, annotations, train, logs, models, sam, sample_pool

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时挂载后台任务：
    1) PG 影子库初始化（幂等建表，容错：连不上不影响启动，JSON 仍为主存储）
    2) 模型消息队列轮询（定时打包 → 自动进入标注页）
    3) 桶灌自动轮询（B 系列：A 包标完自动灌桶 → 满 1500 自动封板）
    """
    try:
        from src.db.engine import init_db
        from src.db import meta_store

        await init_db()
        # M5：状态机映射表种子（幂等；PG 不可用时静默跳过）
        await meta_store.seed_status_mapping()
        logger.info("PG 影子库 schema 就绪")
    except Exception as e:  # PG 不可用时容错，不阻断服务启动
        logger.warning("PG 影子库初始化跳过（服务继续以 JSON 为主存储运行）: %s", e)
    from src.services.queue_service import ModelQueueService
    from src.services.dataset_service import DatasetService
    from src.api.routes import train as train_routes
    poll_task = asyncio.create_task(ModelQueueService().poll_loop())
    bucket_task = asyncio.create_task(DatasetService().bucket_poll_loop())
    seal_task = asyncio.create_task(DatasetService().seal_poll_loop())
    # P6 GPU 熔断（2.3）：复用路由层唯一 TrainService 实例（共享 running_processes），
    # GPU 利用率持续过低自动停训（配置关停时循环内直接 return）
    gpu_watch_task = asyncio.create_task(train_routes.train_service.gpu_watch_loop())
    try:
        yield
    finally:
        poll_task.cancel()
        bucket_task.cancel()
        seal_task.cancel()
        gpu_watch_task.cancel()


app = FastAPI(title="YOLO Training Platform API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件（用于访问图片等）
app.mount("/static", StaticFiles(directory=str(settings.DATA_DIR)), name="static")

# 路由
app.include_router(datasets.router)
app.include_router(annotations.router)
app.include_router(train.router)
app.include_router(logs.router)
app.include_router(models.router)
app.include_router(sam.router)
app.include_router(sample_pool.router)

@app.get("/")
async def root():
    return {"message": "YOLO Training Platform API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}

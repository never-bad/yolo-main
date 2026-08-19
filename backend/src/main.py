import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.core.settings import settings
from src.api.routes import datasets, annotations, train, logs, models, sam, sample_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时挂载模型消息队列后台轮询（定时打包 → 自动进入标注页）"""
    from src.services.queue_service import ModelQueueService
    poll_task = asyncio.create_task(ModelQueueService().poll_loop())
    try:
        yield
    finally:
        poll_task.cancel()


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

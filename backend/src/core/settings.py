from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 项目根目录
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    
    # 数据目录
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    DATASETS_DIR: Path = DATA_DIR / "datasets"
    ANNOTATIONS_DIR: Path = DATA_DIR / "annotations"
    JOBS_DIR: Path = DATA_DIR / "jobs"

    # 样本池（1.6）：困难样本按模型 1:1，空白样本库全局共享 0:N
    SAMPLE_POOL_DIR: Path = DATA_DIR / "sample_pool"
    HARD_POOL_DIR: Path = SAMPLE_POOL_DIR / "hard"        # hard/{model_id}/{images,labels,meta.json}
    BACKGROUND_POOL_DIR: Path = SAMPLE_POOL_DIR / "background"  # background/images + labels(空txt)
    
    # 模型目录
    MODELS_DIR: Path = BASE_DIR / "models"
    REGISTRY_DIR: Path = MODELS_DIR / "registry"
    CUSTOM_MODELS_DIR: Path = MODELS_DIR / "custom"

    # SAM 智能预标注
    SAM_MODELS_DIR: Path = MODELS_DIR / "sam"
    SAM_CONFIG_FILE: Path = DATA_DIR / "sam_config.json"
    
    # API配置
    API_PREFIX: str = ""
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # 训练并发限制：同时最多运行几个训练任务（可通过环境变量 MAX_CONCURRENT_TRAIN_JOBS 覆盖）
    # A10(24GB) 建议 3；若服务器 CPU 核数多可适当调高
    MAX_CONCURRENT_TRAIN_JOBS: int = 3

    # 标注任务自动创建：数据集 prepare 完成后，图片数达到该阈值自动创建标注任务（0 关闭）
    # 雪球闭环（1.8）：新数据自动建任务，标注页可直接进入，不再需要手动建任务
    AUTO_TASK_MIN_IMAGES: int = 20

    # 模型消息队列（D）：被分配进模型的数据先入队，达阈值或定时自动打包进入标注页面
    # 阈值：入队图片数达到该值立即打包（0 关闭阈值触发）
    MODEL_QUEUE_MIN_IMAGES: int = 20
    # 定时：后台轮询间隔（秒），到达间隔且队列非空即打包（0 关闭定时触发）
    MODEL_QUEUE_POLL_SECONDS: int = 300
    # 队列状态文件：data/queues/<model_id>/queue.json + 打包后的批次记录
    QUEUES_DIR: Path = DATA_DIR / "queues"

    # 封板数量门槛：图片数未达该值默认拒绝封板（force 强制封板可放行；0 关闭）
    # 用户确认暂定 1500，可按业务批量调整
    SEAL_MIN_IMAGES: int = 1500
    
    class Config:
        env_file = ".env"
        
    def init_directories(self):
        """初始化所有必需的目录"""
        for dir_path in [
            self.DATA_DIR,
            self.UPLOADS_DIR,
            self.DATASETS_DIR,
            self.ANNOTATIONS_DIR,
            self.JOBS_DIR,
            self.MODELS_DIR,
            self.REGISTRY_DIR,
            self.SAM_MODELS_DIR,
            self.CUSTOM_MODELS_DIR,
            self.SAMPLE_POOL_DIR,
            self.HARD_POOL_DIR,
            self.BACKGROUND_POOL_DIR,
            self.BACKGROUND_POOL_DIR / "images",
            self.BACKGROUND_POOL_DIR / "labels",
            self.QUEUES_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.init_directories()

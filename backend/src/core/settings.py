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

    # P8 半自动发布冷却期（小时）：守门员通过 → 标记 production_ready（待确认），
    # 冷却期过后管理员才能"确认发布"（0 关闭冷却期，立即确认）
    PUBLISH_CONFIRM_COOLDOWN_HOURS: int = 24

    # 封板数量门槛：图片数未达该值默认拒绝封板（force 强制封板可放行；0 关闭）
    # 用户要求暂定 5000，可按业务批量调整；
    # 运行期可在「数据集列表-桶进度」处在线修改（支持全局默认 + 按算法库单独覆盖），
    # 持久化到 data/seal_threshold.json，启动时由 load_runtime_overrides() 覆盖此默认值。
    SEAL_MIN_IMAGES: int = 5000  # 桶灌阈值（全局默认）：满 N 张自动封板
    SEAL_THRESHOLDS_PER_MODEL: dict = {}  # 按算法库单独门槛 {model_id: N}；未设置的库用全局默认
    SEAL_THRESHOLD_FILE: Path = DATA_DIR / "seal_threshold.json"

    # P1 封板时间窗口兜底：数据集准备完成进入标注阶段后，超过该天数（未达数量门槛）自动封板；
    # 数量门槛与时间窗口"先达者"生效。0 关闭时间窗口
    SEAL_MAX_DAYS: int = 7
    # 时间窗口兜底轮询间隔（秒）：后台定时扫描"超期未封板"数据集自动封板（0 关闭）
    SEAL_POLL_SECONDS: int = 3600

    # 桶灌自动轮询间隔（秒）：后台定时扫描"标注完成的 A 包"自动灌桶/封板（0 关闭）
    BUCKET_POLL_SECONDS: int = 60

    # ===== P6 GPU 熔断（2.3）：训练途中 GPU 利用率持续过低 → 判定 GPU 异常 → 自动停训 =====
    # 利用率低于该百分比视为"异常空闲"（正常训练加载/推理时利用率通常 > 50%）
    GPU_ABNORMAL_UTIL_PCT: int = 10
    # 持续低利用率达到该秒数（默认 5 分钟）触发熔断：自动 stop_training + 状态设为 failed(reason=gpu_abnormal)
    GPU_ABNORMAL_SECONDS: int = 300
    # GPU 监控轮询间隔（秒）：后台定时采样各运行中任务的 GPU 利用率（0 关闭熔断监控）
    GPU_WATCH_POLL_SECONDS: int = 30

    # ===== PostgreSQL 元数据库（元数据入库；连接参数由 .env / 环境变量覆盖，服务器部署只改这里）=====
    PG_HOST: str = "127.0.0.1"
    PG_PORT: int = 5432
    PG_USER: str = "postgres"
    PG_PASSWORD: str = "postgres"
    PG_DATABASE: str = "yolo"
    # 是否启用数据库作为元数据源（false 则完全退回 JSON 文件模式，便于迁移期灰度）
    PG_ENABLED: bool = True

    # M5 状态机统一灰度开关：true 读取统一枚举列 unified_status，false 读取旧列（回滚安全）
    USE_UNIFIED_STATUS: bool = False

    class Config:
        # 绝对路径加载 .env，无论从哪个目录启动都能读到（服务器部署只改 backend/.env）
        env_file = str(Path(__file__).resolve().parent.parent.parent / ".env")
        
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


def load_runtime_overrides():
    """启动时加载运行期在线修改的配置覆盖（data/seal_threshold.json）。
    文件不存在/损坏时静默跳过，保持 .env/默认值。"""
    import json
    try:
        if settings.SEAL_THRESHOLD_FILE.exists():
            data = json.loads(settings.SEAL_THRESHOLD_FILE.read_text(encoding="utf-8"))
            v = data.get("SEAL_MIN_IMAGES")
            if isinstance(v, int) and v >= 0:
                settings.SEAL_MIN_IMAGES = v
            per = data.get("per_model")
            if isinstance(per, dict):
                settings.SEAL_THRESHOLDS_PER_MODEL = {
                    str(k): int(x) for k, x in per.items() if isinstance(x, int) and int(x) >= 0
                }
    except Exception:
        pass


def seal_threshold_for(model_id: str | None = None) -> int:
    """某算法库生效的封板/灌桶门槛：有按库覆盖用覆盖值，否则用全局默认。"""
    if model_id:
        v = settings.SEAL_THRESHOLDS_PER_MODEL.get(str(model_id))
        if v is not None:
            return v
    return settings.SEAL_MIN_IMAGES

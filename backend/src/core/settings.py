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
    INFERENCE_RESULTS_DIR: Path = DATA_DIR / "inference_results"
    
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
            self.INFERENCE_RESULTS_DIR,
            self.SAM_MODELS_DIR,
            self.CUSTOM_MODELS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.init_directories()

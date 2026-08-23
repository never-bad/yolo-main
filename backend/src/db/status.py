"""M5 状态机统一（Phase 0）。

统一枚举（新旧映射表见 status_mapping 表，本模块为内存态定义）：

| 域 | 统一枚举值 |
|---|---|
| 数据集 | Collecting / Annotating / Sealed / Training / Completed / Failed |
| 训练任务 | Queued / Running / Evaluating / Ready / Rejected / Archived |
| 模型 | training / evaluating / production_ready / superseded / rejected / archived |

双写期：写入方先写旧列，再用本模块的 to_unified() 计算新列值。
灰度期：读取方根据 USE_UNIFIED_STATUS 开关决定读 unified_status 还是旧列。
"""
from src.core.settings import settings

# ============ 统一枚举常量 ============

DATASET_STATES = ["Collecting", "Annotating", "Sealed", "Training", "Completed", "Failed"]
TASK_STATES = ["Queued", "Running", "Evaluating", "Ready", "Rejected", "Archived"]
MODEL_STATES = ["training", "evaluating", "production_ready", "superseded", "rejected", "archived"]

# ============ 旧值 → 新值映射（对应 status_mapping 表种子数据） ============

# 数据集：真实代码 status 列（uploaded/prepared/sealed）+ training_status 子状态
DATASET_STATUS_MAP = {
    "uploaded": "Collecting",
    "prepared": "Annotating",
    "sealed": "Sealed",
    "incomplete": "Annotating",     # 兜底：未训练且未封板
    "completed": "Completed",
    "failed": "Failed",
}

# 模型：真实代码 status 列（active/production_ready/rejected/superseded + archived）
MODEL_STATUS_MAP = {
    "active": "training",           # 无 task 状态可推断时按 training 兜底
    "production_ready": "production_ready",
    "rejected": "rejected",
    "superseded": "superseded",
    "archived": "archived",
}

# 训练任务（影子库未建表，直接由训练状态机使用）
TASK_STATUS_MAP = {
    "Queued": "Queued",
    "Running": "Running",
    "Evaluating": "Evaluating",
    "Ready": "Ready",
    "Production Ready": "Ready",
    "Rejected": "Rejected",
    "Archived": "Archived",
}

# g6 M5 规范化：训练任务 JSON 真实 status 值 → 统一枚举（list_jobs/get_job 查询回填）
#   running    → Running（训练中）
#   completed  → Ready（训练+守门员完成，产出模型待发布或已入生产）
#   failed     → Rejected（失败/守门员不通过；reason 细分 oom/gpu_abnormal/其他）
#   crashed    → Archived（进程崩溃中断，可人工恢复）
#   stopped    → Archived（用户手动停止）
TASK_REAL_STATUS_MAP = {
    "running": "Running",
    "completed": "Ready",
    "failed": "Rejected",
    "crashed": "Archived",
    "stopped": "Archived",
}

_DOMAIN_MAP = {
    "dataset": DATASET_STATUS_MAP,
    "task": TASK_STATUS_MAP,
    "model": MODEL_STATUS_MAP,
}

STATUS_MAPPING_SEED = [
    (domain, old, new, "直接映射")
    for domain, m in _DOMAIN_MAP.items() for old, new in m.items()
]
# g6：补充训练任务真实状态映射（统一枚举直通之上，供影子库 status_mapping 表查询）
STATUS_MAPPING_SEED += [
    ("task", old, new, "真实落库值 → 统一枚举")
    for old, new in TASK_REAL_STATUS_MAP.items()
]


def to_unified(domain: str, old_value: str, default: str = "") -> str:
    """旧枚举值 → 统一枚举值（未命中返回 default，便于调用方兜底）。"""
    if not old_value:
        return default
    m = _DOMAIN_MAP.get(domain, {})
    return m.get(old_value, default)


def dataset_unified(meta: dict) -> str:
    """数据集统一状态：training_status=completed 优先于 stage；sealed 兜底 Sealed。"""
    if (meta.get("training_status") or "") == "completed":
        return "Completed"
    return to_unified("dataset", meta.get("status"), "Collecting")


def task_unified(meta: dict) -> str:
    """训练任务统一状态：真实 status 值 → M5 统一枚举。

    失败任务统一归 Rejected（reason 细分 oom/gpu_abnormal；前端失败弹窗依据
    reason 与 error 展示恢复建议）。early_stopped 仍归 Ready（早停=收敛，产出有效模型）。
    """
    return TASK_REAL_STATUS_MAP.get(meta.get("status"), "Queued")


def use_unified_status() -> bool:
    """灰度开关：true 读 unified_status 列，false 读旧列（回滚安全）。"""
    return bool(getattr(settings, "USE_UNIFIED_STATUS", False))
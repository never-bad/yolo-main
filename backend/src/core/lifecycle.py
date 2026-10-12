"""后台循环观测层（K-3 / I-10）：每个 loop 在每轮迭代登记心跳/错误。

/health 端点读取 BG_LOOP_STATE 字典返回。仅观测，绝不改变 loop 行为。

设计约束（批次 K-3 红线）：
- 本模块只提供两个纯写入函数，不持锁、不做 IO、不抛异常给调用方；
- 调用点只允许出现在 loop 的 while 顶部（record_heartbeat）与既有 except 分支内
  （record_error），不得改动 sleep 周期、调用顺序或异常处理路径；
- last_error 登记的是"最近一次"，恢复后不自动清空——排障时需要靠它定位时间窗口，
  是否已恢复由 last_heartbeat_at 是否继续推进来判断。
"""
from datetime import datetime, timezone
from typing import Any

# loop 名 -> 观测状态。进程内单例，随 FastAPI lifespan 启动的 4 个后台任务写入。
BG_LOOP_STATE: dict[str, dict[str, Any]] = {}


def _initial(name: str) -> dict:
    """一个 loop 的观测状态初值（None 字段在 /health 输出时被过滤掉）。"""
    return {
        "name": name,
        "last_heartbeat_at": None,
        "iteration_count": 0,
        "last_error": None,
        "last_error_at": None,
    }


def record_heartbeat(name: str) -> None:
    """登记一次迭代：刷新心跳时间戳并累加迭代计数。"""
    s = BG_LOOP_STATE.setdefault(name, _initial(name))
    s["last_heartbeat_at"] = datetime.now(timezone.utc).isoformat()
    s["iteration_count"] = s.get("iteration_count", 0) + 1


def record_error(name: str, err: Exception) -> None:
    """登记最近一次异常（类型名 + 消息）与其发生时间。"""
    s = BG_LOOP_STATE.setdefault(name, _initial(name))
    s["last_error"] = f"{type(err).__name__}: {err}"
    s["last_error_at"] = datetime.now(timezone.utc).isoformat()

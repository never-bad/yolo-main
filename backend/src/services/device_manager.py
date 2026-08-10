"""GPU 设备探测与推荐工具

基于 nvidia-smi 获取 GPU 信息，提供空闲 GPU 优先的选择逻辑。
参考设计文档 2.3 / 4.2 节，独立从零实现。
"""
import subprocess
from typing import Dict, List


def get_available_gpus() -> List[Dict]:
    """调用 nvidia-smi 获取 GPU 信息列表

    Returns:
        每个 GPU 的 index / name / memory_total_mb / memory_free_mb / utilization。
        无 GPU 或 nvidia-smi 不可用时返回空列表。
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return []

        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                try:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mb": int(parts[2]),
                        "memory_free_mb": int(parts[3]),
                        "utilization": int(parts[4]),
                    })
                except ValueError:
                    continue
        return gpus
    except Exception:
        return []


def get_best_device() -> str:
    """返回最佳设备字符串

    逻辑：显存可用量最大的 GPU 优先；无 GPU 时退回 CPU。
    """
    gpus = get_available_gpus()
    if not gpus:
        return "cpu"
    best = max(gpus, key=lambda g: g["memory_free_mb"])
    return str(best["index"])


def get_device_summary() -> Dict:
    """返回设备概要，供前端展示"""
    gpus = get_available_gpus()
    if not gpus:
        return {"available": False, "device": "cpu", "gpus": []}
    best = max(gpus, key=lambda g: g["memory_free_mb"])
    return {
        "available": True,
        "device": str(best["index"]),
        "gpus": gpus,
    }
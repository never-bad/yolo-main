# -*- coding: utf-8 -*-
"""B-44 测试集评测编排 + 后台循环（P1）。

- run_test_eval：单次评测一个模型在其算法库测试集上的指标，写回 model.json 的 test_eval 块。
- mark_pending：训练注册后（或手动）把模型 test_eval 置为 pending，交后台循环串行评测。
- test_eval_loop：扫描 registry 中 status=pending 的 test_eval，串行执行；训练在跑时避让 GPU。

所有落盘走 JSON 真值源（model.json），不动 ORM / PG 主存储；评测产物不进 jobs_dir
（避免 B-43 那种 glob("*.json") 误读 task 档案）。
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

from src.core.lifecycle import record_error, record_heartbeat
from src.core.settings import settings
from src.utils.class_labels import model_class_names, resolve_library_root_id
from src.utils.json_io import load_json, save_json_atomic
from src.yolo.gatekeeper import evaluate_model

logger = logging.getLogger(__name__)

MAX_CONCURRENT_TEST_EVAL = 1  # 串行，避免与训练抢 GPU
TEST_EVAL_POLL_SECONDS = 60


# --------------------------------------------------------------------------- #
# 读写辅助
# --------------------------------------------------------------------------- #
def _load_model_meta(model_id: str) -> Optional[dict]:
    if not model_id:
        return None
    p = settings.REGISTRY_DIR / model_id / "model.json"
    if not p.exists():
        return None
    try:
        return load_json(p)
    except Exception:
        return None


def _write_test_eval(model_id: str, te: dict) -> None:
    p = settings.REGISTRY_DIR / model_id / "model.json"
    meta = load_json(p)
    meta["test_eval"] = te
    save_json_atomic(p, meta)


def _weights_sha256(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _resolve_imgsz(meta: dict) -> int:
    """各模型沿用训练 imgsz（B-44 已拍板）；找不到回落 640。"""
    for cand in ("imgsz", "train_imgsz", "img_size"):
        v = meta.get(cand)
        if isinstance(v, int) and v > 0:
            return v
    mm = meta.get("model_meta") or {}
    if isinstance(mm, dict):
        for cand in ("imgsz", "train_imgsz", "img_size"):
            v = mm.get(cand)
            if isinstance(v, int) and v > 0:
                return v
    return 640


def _resolve_conf_preference(meta: dict, library_root_id: str, conf_preference) -> str:
    """默认偏好：请求参数 > 库根 test_eval_preference > miss（在意漏检）。"""
    if conf_preference:
        return conf_preference
    root = _load_model_meta(library_root_id) or {}
    return root.get("test_eval_preference") or "miss"


# --------------------------------------------------------------------------- #
# 单次评测
# --------------------------------------------------------------------------- #
def run_test_eval(model_id: str, force: bool = False, focus_metrics=None,
                  extra_artifacts=None, conf=None, conf_preference=None) -> dict:
    """对一个模型在其算法库测试集上跑评测，写回 test_eval 块。

    Returns: {"ok": bool, "test_eval"?: dict, "error"?: str}
    """
    meta = _load_model_meta(model_id)
    if not meta:
        return {"ok": False, "error": f"模型不存在: {model_id}"}
    library_root_id = resolve_library_root_id(model_id) or model_id
    root = _load_model_meta(library_root_id) or {}
    test_dataset_id = root.get("test_dataset_id")
    if not test_dataset_id:
        return {"ok": False, "error": "该算法库未设置测试集"}
    test_meta_path = settings.DATASETS_DIR / test_dataset_id / "meta.json"
    if not test_meta_path.exists():
        return {"ok": False, "error": f"测试集不存在: {test_dataset_id}"}
    test_meta = load_json(test_meta_path)
    sealed_version = test_meta.get("test_sealed_version", 1)
    data_yaml = settings.DATASETS_DIR / test_dataset_id / "data.yaml"
    if not data_yaml.exists():
        return {"ok": False, "error": f"测试集未封板（缺 data.yaml）: {test_dataset_id}"}
    weights = meta.get("weights_path")
    if not weights or not Path(weights).exists():
        return {"ok": False, "error": f"模型权重不存在: {weights}"}
    classes = (model_class_names(library_root_id)
               or test_meta.get("classes") or meta.get("classes") or [])
    if not classes:
        return {"ok": False, "error": "无法解析类别字典（空类别）"}

    conf_preference = _resolve_conf_preference(meta, library_root_id, conf_preference)
    imgsz = _resolve_imgsz(meta)

    # 写 running 状态（防止并发重复触发）
    te = meta.get("test_eval") or {}
    if te.get("status") == "running" and not force:
        return {"ok": False, "error": "评测进行中"}
    te.update({
        "status": "running", "dataset_id": test_dataset_id,
        "dataset_sealed_version": sealed_version, "weights": "weights/best.pt",
        "started_at": time.time(), "error": None,
    })
    _write_test_eval(model_id, te)

    try:
        result = evaluate_model(
            weights, str(data_yaml), classes,
            conf=conf, conf_preference=conf_preference,
            imgsz=imgsz, iou=0.7, device="cuda:0",
        )
        te["metrics"] = result
        te["weights_sha256"] = _weights_sha256(Path(weights))
        te["eval_config"] = {
            "imgsz": imgsz,
            "conf": result.get("conf_used"),
            "conf_preference": conf_preference,
            "iou": 0.7, "device": "cuda:0",
        }
        te["focus_metrics"] = focus_metrics or []
        te["extra_artifacts"] = extra_artifacts or []
        te["evaluated_at"] = time.time()
        te["engine"] = "ultralytics 8.x"
        te["status"] = "done" if not result.get("error") else "failed"
        if result.get("error"):
            te["error"] = result["error"]
        _write_test_eval(model_id, te)
        return {"ok": True, "test_eval": te}
    except Exception as e:  # noqa: BLE001
        te["status"] = "failed"
        te["error"] = str(e)
        _write_test_eval(model_id, te)
        return {"ok": False, "error": str(e)}


def mark_pending(model_id: str, focus_metrics=None, extra_artifacts=None) -> bool:
    """训练注册后调用：该模型所属算法库有测试集则置 pending，否则 skipped。

    返回 True 表示已置 pending（后台循环会评测），False 表示无测试集（写 skipped）。
    """
    meta = _load_model_meta(model_id)
    if not meta:
        return False
    library_root_id = resolve_library_root_id(model_id) or model_id
    root = _load_model_meta(library_root_id) or {}
    test_dataset_id = root.get("test_dataset_id")
    if not test_dataset_id:
        te = {"status": "skipped", "reason": "no_test_set", "dataset_id": None,
              "marked_at": time.time()}
        _write_test_eval(model_id, te)
        return False
    test_meta = _load_model_meta(test_dataset_id)
    sealed_version = (test_meta or {}).get("test_sealed_version", 1)
    te = {
        "status": "pending", "dataset_id": test_dataset_id,
        "dataset_sealed_version": sealed_version,
        "focus_metrics": focus_metrics or [], "extra_artifacts": extra_artifacts or [],
        "requested_at": time.time(), "error": None,
    }
    _write_test_eval(model_id, te)
    return True


def _stale_library_test_evals(library_root_id: str, new_version: int) -> None:
    """模块级封装：测试集换版/解绑后把同库已评测模型标 stale（委托 DatasetService 实例）。

    真正逻辑在 DatasetService._stale_library_test_evals（只读/写 JSON，无需特殊实例状态），
    这里只做一层转发，避免 models.py 直接依赖 dataset_service 实例的私有方法。
    """
    try:
        from src.services.dataset_service import DatasetService
        DatasetService()._stale_library_test_evals(library_root_id, new_version)
    except Exception as e:  # noqa: BLE001
        logger.warning("stale_library_test_evals failed for %s: %s", library_root_id, e)


# --------------------------------------------------------------------------- #
# 后台循环
# --------------------------------------------------------------------------- #
def _iter_pending_models() -> list:
    out = []
    if not settings.REGISTRY_DIR.exists():
        return out
    for d in settings.REGISTRY_DIR.iterdir():
        if not d.is_dir():
            continue
        mf = d / "model.json"
        if not mf.exists():
            continue
        try:
            meta = load_json(mf)
        except Exception:
            continue
        te = meta.get("test_eval") or {}
        if te.get("status") == "pending":
            out.append(d.name)
    return out


def list_library_model_ids(root_id: str) -> list:
    """列出某算法库根下的全部模型 id（含库根自身之外的版本模型）。"""
    out = []
    if not settings.REGISTRY_DIR.exists():
        return out
    for d in settings.REGISTRY_DIR.iterdir():
        if not d.is_dir():
            continue
        mf = d / "model.json"
        if not mf.exists():
            continue
        try:
            meta = load_json(mf)
        except Exception:
            continue
        rid = resolve_library_root_id(d.name) or d.name
        if rid == root_id and d.name != root_id:
            out.append(d.name)
    return out


async def test_eval_loop(poll_seconds: int | None = None):
    """扫描 registry 中 status=pending 的 test_eval，串行评测；训练在跑时避让 GPU。"""
    poll = poll_seconds or TEST_EVAL_POLL_SECONDS
    name = "test_eval"
    while True:
        try:
            record_heartbeat(name)
            # 训练在跑则避让 GPU，下轮再评
            try:
                from src.api.routes import train as train_routes
                running_train = train_routes.train_service._active_training_count()
            except Exception:
                running_train = 0
            if running_train and running_train > 0:
                await asyncio.sleep(poll)
                continue
            for model_id in _iter_pending_models():
                try:
                    # 评测是 CPU/GPU 阻塞任务（ultralytics val 可能耗时数分钟），
                    # 必须丢到线程里跑，否则会卡死整个事件循环（含 /health 与其它请求）。
                    await asyncio.to_thread(run_test_eval, model_id, False)
                except Exception as e:  # noqa: BLE001
                    logger.warning("test_eval failed for %s: %s", model_id, e)
            await asyncio.sleep(poll)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            record_error(name, e)
            await asyncio.sleep(poll)


# --------------------------------------------------------------------------- #
# 多模型比对矩阵
# --------------------------------------------------------------------------- #
def get_test_eval_matrix(root_id: str, model_ids: list) -> dict:
    """同库多模型测试集指标对比矩阵。

    返回：
    {
      "models": [{model_id, display_name, status, metrics, eval_config, weights_sha256, stale, comparable}],
      "comparable_key": "dataset_id+sealed_version+imgsz+conf_preference+names",  # 口径指纹
      "classes": [...],  # 参与比对的类别并集
    }
    口径不一致（imgsz / 封板版本 / conf_preference / 类别字典）的模型 comparable=False，
    不参与「最优」高亮，但仍展示数值。
    """
    root = _load_model_meta(root_id) or {}
    test_dataset_id = root.get("test_dataset_id")
    rows = []
    names_union = []
    for mid in (model_ids or []):
        meta = _load_model_meta(mid)
        if not meta:
            continue
        rid = resolve_library_root_id(mid) or mid
        if rid != root_id:
            continue  # 跨库拒绝
        te = meta.get("test_eval") or {}
        m = te.get("metrics") or {}
        ec = te.get("eval_config") or {}
        # 口径指纹：数据集 + 封板版本 + imgsz + conf_preference + 类别字典顺序
        key = "||".join([
            str(te.get("dataset_id")),
            str(te.get("dataset_sealed_version")),
            str(ec.get("imgsz")),
            str(ec.get("conf_preference")),
            ",".join(str(n) for n in (m.get("class_ap") or {}).keys()),
        ])
        for n in (m.get("class_ap") or {}).keys():
            if n not in names_union:
                names_union.append(n)
        rows.append({
            "model_id": mid,
            "display_name": meta.get("display_name") or meta.get("name") or mid,
            "version": meta.get("version"),
            "status": te.get("status"),
            "stale": te.get("status") == "stale",
            "metrics": m,
            "eval_config": ec,
            "weights_sha256": te.get("weights_sha256"),
            "comparable_key": key,
            "comparable": test_dataset_id is not None,
            "error": te.get("error"),
        })
    # 以首个有效口径为基准，标记一致的 comparable=True
    base_key = next((r["comparable_key"] for r in rows if r["comparable_key"]), None)
    for r in rows:
        r["comparable"] = bool(base_key) and r["comparable_key"] == base_key
    return {
        "test_dataset_id": test_dataset_id,
        "models": rows,
        "classes": names_union,
        "base_key": base_key,
    }

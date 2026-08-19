"""模型消息队列服务（数据链路改造·阶段D）。

被分配进模型的数据流入模型消息队列（data/queues/<model_id>/queue.json），
当队列图片数达到阈值 MODEL_QUEUE_MIN_IMAGES，或定时（MODEL_QUEUE_POLL_SECONDS）到达时，
自动打包匹配的数据集，并创建/复用该数据集的标注任务（任务类别按模型标签字典英文码自动获取），
标注页面即可直接进入，无需手工建任务/选类别。

queue.json 结构：
{
  "model_id": "m1",
  "created_at": "...",
  "pending":   [{"image_id", "dataset_id", "version", "enqueued_at"}],
  "last_enqueue_at": "...",
  "last_packed_at": "...",
  "history":   [{"packed_at", "batches": [{"dataset_id","version","task_id","images"}]}]
}
"""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from src.core.settings import settings

logger = logging.getLogger(__name__)

IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class ModelQueueService:
    def __init__(self):
        self.queues_dir = settings.QUEUES_DIR

    # ------------------------------------------------------------------
    # 队列读写
    # ------------------------------------------------------------------
    def _queue_path(self, model_id: str) -> Path:
        return self.queues_dir / model_id / "queue.json"

    def _load_queue(self, model_id: str) -> dict:
        p = self._queue_path(model_id)
        try:
            return _load_json(p)
        except Exception:
            return None

    def _load_or_create(self, model_id: str) -> dict:
        q = self._load_queue(model_id)
        if q is None:
            q = {
                "model_id": model_id,
                "created_at": datetime.now().isoformat(),
                "pending": [],
                "last_enqueue_at": None,
                "last_packed_at": None,
                "history": [],
            }
        q.setdefault("pending", [])
        q.setdefault("history", [])
        return q

    def _save_queue(self, model_id: str, q: dict):
        p = self._queue_path(model_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        _save_json(p, q)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def _scan_dataset_images(self, version_dir: Path) -> list:
        """扫描数据集版本下所有图片名（stem 去重；兼容平铺 images/ 与划分 images/{train,val,test}）"""
        if not version_dir.exists():
            return []
        images_dir = version_dir / "images"
        if not images_dir.exists():
            return []
        stems = []
        seen = set()
        dirs = [images_dir]
        for s in ("train", "val", "test"):
            d = images_dir / s
            if d.is_dir():
                dirs.append(d)
        for d in dirs:
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in IMG_SUFFIX and p.stem not in seen:
                    seen.add(p.stem)
                    stems.append(p.stem)
        return stems

    def _labels_dict_classes(self, model_id: str):
        """模型标签字典英文码（按 index 升序）；无字典返回 None（任务类别回退数据集文件）"""
        try:
            dic = _load_json(settings.REGISTRY_DIR / model_id / "labels_dict.json")
            labels = dic.get("labels") or []
            names = [
                (l.get("english_code") or "").strip()
                for l in sorted(labels, key=lambda x: x.get("index", 0))
            ]
            return [n for n in names if n] or None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 入队
    # ------------------------------------------------------------------
    async def enqueue_dataset(self, model_id: str, dataset_id: str, version: str = "v1") -> dict:
        """数据集分配进模型时调用：全部图片流入模型消息队列。

        同一 (dataset_id, image_id) 不入队两次；返回是否已达打包阈值。
        """
        version_dir = settings.DATASETS_DIR / dataset_id / version
        image_ids = await asyncio.to_thread(self._scan_dataset_images, version_dir)

        def _sync():
            q = self._load_or_create(model_id)
            known = {(e["dataset_id"], e["image_id"]) for e in q["pending"]}
            now = datetime.now().isoformat()
            added = []
            for im in image_ids:
                if (dataset_id, im) in known:
                    continue
                q["pending"].append({
                    "image_id": im,
                    "dataset_id": dataset_id,
                    "version": version,
                    "enqueued_at": now,
                })
                known.add((dataset_id, im))
                added.append(im)
            if added:
                q["last_enqueue_at"] = now
            self._save_queue(model_id, q)
            threshold = settings.MODEL_QUEUE_MIN_IMAGES
            return {
                "model_id": model_id,
                "enqueued": len(added),
                "pending": len(q["pending"]),
                "threshold_reached": threshold > 0 and len(q["pending"]) >= threshold,
            }

        return await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # 打包（阈值或定时触发）
    # ------------------------------------------------------------------
    async def pack_model(self, model_id: str) -> dict:
        """把当前队列按 (dataset_id, version) 分组打包：创建/复用标注任务后清空 pending。

        任务类别 = 模型标签字典英文码（自动判定模型需要标注的内容）。
        返回 {packed: [{dataset_id, version, task_id, images}]}
        """
        q = await asyncio.to_thread(self._load_queue, model_id)
        if not q or not q.get("pending"):
            return {"model_id": model_id, "packed": [], "message": "队列为空，无需打包"}

        groups = {}
        for e in q["pending"]:
            key = (e["dataset_id"], e.get("version", "v1"))
            groups.setdefault(key, 0)
            groups[key] += 1

        from src.services.annotation_service import AnnotationService
        ann_svc = AnnotationService()
        classes = await asyncio.to_thread(self._labels_dict_classes, model_id) or []

        packed = []
        for (ds_id, ver), n in sorted(groups.items()):
            try:
                existing = await asyncio.to_thread(ann_svc.find_task_by_dataset, ds_id, ver)
                if existing:
                    task_id = existing["task_id"]
                else:
                    task = await ann_svc.create_task(ds_id, ver, list(classes))
                    task_id = task["task_id"]
                packed.append({
                    "dataset_id": ds_id,
                    "version": ver,
                    "task_id": task_id,
                    "images": n,
                    "classes": list(classes),
                })
            except Exception as e:
                logger.warning(f"pack {model_id}/{ds_id}/{ver}: create task failed: {e}")
                packed.append({"dataset_id": ds_id, "version": ver, "task_id": None, "images": n, "error": str(e)})

        def _commit():
            q = self._load_or_create(model_id)
            now = datetime.now().isoformat()
            q["history"] = q.get("history", []) + [{"packed_at": now, "batches": packed}]
            q["pending"] = []
            q["last_packed_at"] = now
            self._save_queue(model_id, q)
            return q

        await asyncio.to_thread(_commit)
        logger.info(f"queue {model_id} packed {len(packed)} batches")
        return {"model_id": model_id, "packed": packed}

    async def pack_timed(self) -> dict:
        """定时触发：扫描所有队列，未空队列且超时（距上次入队 >= MODEL_QUEUE_POLL_SECONDS）即打包。"""
        poll_seconds = settings.MODEL_QUEUE_POLL_SECONDS
        if poll_seconds <= 0 or not self.queues_dir.exists():
            return {"packed_models": []}

        import time
        model_ids = []
        for qdir in self.queues_dir.iterdir():
            if not qdir.is_dir():
                continue
            model_ids.append(qdir.name)

        packed_models = []
        for model_id in model_ids:
            q = await asyncio.to_thread(self._load_queue, model_id)
            if not q or not q.get("pending"):
                continue
            last = q.get("last_enqueue_at")
            if not last:
                continue
            try:
                elapsed = time.time() - datetime.fromisoformat(last).timestamp()
            except Exception:
                continue
            if elapsed >= poll_seconds:
                res = await self.pack_model(model_id)
                packed_models.append({"model_id": model_id, "packed": res.get("packed", [])})
        return {"packed_models": packed_models}

    async def poll_loop(self):
        """后台轮询（随 FastAPI lifespan 启动）：每 5 秒检查一次超时队列"""
        while True:
            try:
                await self.pack_timed()
            except Exception as e:
                logger.warning(f"queue poll error: {e}")
            await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_queue(self, model_id: str) -> dict:
        q = self._load_queue(model_id)
        if q is None:
            return {
                "model_id": model_id,
                "pending": 0,
                "last_enqueue_at": None,
                "last_packed_at": None,
                "history": [],
            }
        return {
            "model_id": model_id,
            "pending": len(q.get("pending", [])),
            "pending_items": q.get("pending", []),
            "last_enqueue_at": q.get("last_enqueue_at"),
            "last_packed_at": q.get("last_packed_at"),
            "history": q.get("history", []),
        }

    def list_queues(self) -> dict:
        queues = []
        if not self.queues_dir.exists():
            return {"queues": []}
        for qdir in sorted(self.queues_dir.iterdir()):
            if not qdir.is_dir():
                continue
            queues.append(self.get_queue(qdir.name))
        return {"queues": queues}
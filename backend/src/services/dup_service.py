"""M3 重复图检测服务：全库感知哈希查重（相同/近似图片）

解决「1 万张里混入了重复图」的目标①。为什么不用千问：查重是"1 万张两两比较"
的规模问题（≈5 千万对），任何 VLM 都跑不动；标准做法是感知哈希——纯 CPU、
单张 <5ms，全量算图 + 聚类约数十秒，无需 GPU。

算法（双层，误判互补）：
  1. MD5 精确哈希：文件字节级完全一致 → 严格重复（同一张图复制/改名）。
  2. dHash 64 位感知哈希：灰度 9x8 缩放后相邻像素比较。Hamming 距离 ≤ threshold
     （默认 6，1~20 可调）判为视觉近重复（缩放/压缩/轻微调色）。裁剪/加水印
     属于"部分重复"，dHash 不保证抓到，本期不做（需 CLIP，另一模块）。
  3. 并查集聚类：互相近重复的图归同组（传递闭包，A≈B、B≈C → A/B/C 一组）。

写回语义（与批量审核一致，绝不碰框坐标）：
  每图 audit_meta.duplicate = {group_id, group_size, representative, similar, ...}
  - 只写已存在的标注 JSON；没有标注 JSON 的图只进结果汇总，不落盘
    （避免凭空造 JSON 把"未标注"图变成"已标注"）。
  - 重跑先清掉不在任何重复组图上的旧 duplicate 字段，防残留假结论。
  - classify_audit 会把 duplicate 作为一个问题类型桶，前端筛选自动出现
    「审核·重复图片」分组。

性能：dHash 用线程池并发（IO+解码为主）；两两 Hamming 用 numpy 分块
popcount（1 万张 → 512x10000 块 × 若干轮，秒级，内存峰值 <100MB）。
"""
import asyncio
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from src.core.settings import settings
from src.utils.json_io import load_json, save_json_atomic

# Pillow 11 移除了模块级重采样常量的部分旧写法，这里做兼容取值
_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

# 8 位查表 popcount（numpy 向量化，避免逐比特循环）
_POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _dhash64(img_abs: Path) -> Optional[int]:
    """dHash 64 位：灰度缩放到 9x8，每行右像素 > 左像素 记 1，得 8x8=64 位"""
    try:
        with Image.open(img_abs) as im:
            g = im.convert("L").resize((9, 8), _RESAMPLE)
        px = np.asarray(g, dtype=np.int16)
        bits = (px[:, 1:] > px[:, :-1]).ravel()
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h
    except Exception:
        return None  # 坏图：跳过并计数，不阻断整批


def _md5_file(img_abs: Path) -> Optional[str]:
    """文件字节级 MD5（分块读，10G 级大图也不会撑内存）"""
    try:
        h = hashlib.md5()
        with open(img_abs, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _pairs_within_threshold(hashes: List[int], thr: int) -> List[tuple]:
    """分块 numpy 求 Hamming ≤ thr 的所有图对 (i, j, dist)，i < j。

    1 万张全量两两 = 5 千万对，纯 Python 循环要分钟级；numpy 分块
    XOR + 查表 popcount 秒级完成，内存峰值约 2x(512x10000x8B) <100MB。
    """
    n = len(hashes)
    if n < 2 or thr < 0:
        return []
    arr = np.array(hashes, dtype=np.uint64)
    pairs = []
    step = 512
    for s in range(0, n, step):
        e = min(s + step, n)
        block = arr[s:e, None] ^ arr[None, :]           # (chunk, n) uint64
        bv = block.view(np.uint8).reshape(e - s, n, 8)  # 拆成 8 字节
        dist = _POPCNT[bv].sum(axis=2)                  # (chunk, n) Hamming 距离
        ii, jj = np.nonzero(dist <= thr)
        for a, b in zip(ii.tolist(), jj.tolist()):
            gi, gj = s + a, b
            if gi < gj:  # 对角线（自比，dist=0）与下三角（重复对）跳过
                pairs.append((gi, gj, int(dist[a, b])))
    return pairs


class _UnionFind:
    """并查集：互相近重复的图传递归组"""

    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]  # 路径压缩
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


class DupService:
    """重复图检测：内存任务状态 + 后台线程扫描 + audit_meta 写回"""

    def __init__(self):
        self.tasks: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    async def start_scan(self, task_id: str, threshold: int = 6) -> dict:
        """启动全库重复图扫描（后台运行，不占 GPU）"""
        task_file = settings.ANNOTATIONS_DIR / task_id / "task.json"
        if not task_file.exists():
            raise ValueError("Task not found")
        task_meta = await asyncio.to_thread(load_json, task_file)
        items = task_meta.get("items", [])
        if not items:
            raise ValueError("任务内没有图片")
        thr = max(1, min(int(threshold or 6), 20))
        batch_id = f"dup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        st = {
            "batch_id": batch_id, "task_id": task_id,
            "status": "running",           # running | done | cancelled | error
            "phase": "hash",               # hash(算哈希) | cluster(聚类) | writeback(写回)
            "total": len(items), "done": 0,
            "failed_images": 0, "last_error": None,
            "error": None, "cancelled": False,
            "threshold": thr,
            "result_summary": None,
        }
        self.tasks[batch_id] = st
        ds_id, ver = task_meta.get("dataset_id", ""), task_meta.get("version", "v1")
        asyncio.create_task(self._run_scan(batch_id, items, ds_id, ver, thr))
        return {"batch_id": batch_id, "total": len(items), "threshold": thr}

    def get_status(self, batch_id: str) -> Optional[dict]:
        st = self.tasks.get(batch_id)
        return dict(st) if st else None

    def stop_scan(self, batch_id: str) -> Optional[dict]:
        st = self.tasks.get(batch_id)
        if not st:
            return None
        if st["status"] == "running":
            st["cancelled"] = True
        return dict(st)

    # ------------------------------------------------------------------
    # 扫描主流程
    # ------------------------------------------------------------------
    async def _run_scan(self, batch_id: str, items: list, ds_id: str, ver: str, thr: int) -> None:
        st = self.tasks[batch_id]
        ann_dir = settings.DATASETS_DIR / ds_id / "annotations"
        try:
            # ---- 阶段 1：并发算 MD5 + dHash（IO/解码为主，线程池有效）----
            st["phase"] = "hash"
            n = len(items)
            dhashes: List[Optional[int]] = [None] * n
            md5s: List[Optional[str]] = [None] * n

            def _hash_one(idx: int):
                img_abs = settings.DATA_DIR / items[idx]["image_path"]
                return idx, _md5_file(img_abs), _dhash64(img_abs)

            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=16) as ex:
                futs = [ex.submit(_hash_one, i) for i in range(n)]
                for fut in as_completed(futs):
                    idx, md5, dh = fut.result()
                    if md5 is None or dh is None:
                        st["failed_images"] += 1
                    if md5 is not None:
                        md5s[idx] = md5
                    if dh is not None:
                        dhashes[idx] = dh
                    st["done"] += 1
                    await asyncio.sleep(0)  # 让出事件循环，status 接口不被饿死

            if st["cancelled"]:
                st["status"] = "cancelled"
                return

            # ---- 阶段 2：MD5 精确合并 + dHash 近重复聚类 ----
            st["phase"] = "cluster"
            st["current_image"] = None
            valid = [i for i in range(n) if dhashes[i] is not None]
            uf = _UnionFind(n)
            # 2a. 文件字节级一致 → 严格重复
            by_md5: Dict[str, int] = {}
            for i in valid:
                m = md5s[i]
                if m in by_md5:
                    uf.union(by_md5[m], i)
                else:
                    by_md5[m] = i
            # 2b. dHash Hamming ≤ thr → 视觉近重复（dist=0 的严格相同对也会被抓到）
            pairs = await loop.run_in_executor(
                None, _pairs_within_threshold, [dhashes[i] for i in valid], thr)
            # 局部下标 → 全局下标；记录 pairwise 距离供 similar 明细用
            pair_dist: Dict[tuple, int] = {}
            for a, b, d in pairs:
                gi, gj = valid[a], valid[b]
                uf.union(gi, gj)
                pair_dist[(min(gi, gj), max(gi, gj))] = d

            groups: Dict[int, List[int]] = {}
            for i in valid:
                groups.setdefault(uf.find(i), []).append(i)
            dup_groups = [sorted(members) for members in groups.values() if len(members) >= 2]
            # 组内按最小 image_id 排序，保证 representative 稳定可复现
            dup_groups.sort(key=lambda g: items[g[0]]["image_id"])

            if st["cancelled"]:
                st["status"] = "cancelled"
                return

            # ---- 阶段 3：写回 audit_meta.duplicate（只改 JSON，不碰框坐标）----
            st["phase"] = "writeback"
            in_group = set()
            skipped_no_ann = 0
            groups_detail = []
            for ordinal, members in enumerate(dup_groups):
                if st["cancelled"]:
                    break
                group_id = f"{batch_id}_{ordinal:04d}"
                ids = [items[i]["image_id"] for i in members]
                groups_detail.append({"group_id": group_id, "image_ids": ids})
                in_group.update(members)
                for i in members:
                    rep = i == members[0]
                    similar = []
                    for j in members:
                        if j == i:
                            continue
                        key = (min(i, j), max(i, j))
                        similar.append({
                            "image_id": items[j]["image_id"],
                            "distance": pair_dist.get(key),  # md5 直接合并的对可能没有距离
                            "exact": md5s[i] is not None and md5s[i] == md5s[j],
                        })
                    similar.sort(key=lambda s: (s["distance"] is None, s["distance"] or 0))
                    ann_path = ann_dir / f"{items[i]['image_id']}.json"
                    # load_json 对缺失文件抛 FileNotFoundError（不是返回 None），
                    # 组内无标注 JSON 的图必须捕获后跳过，否则会炸掉整个写回阶段
                    try:
                        ann = load_json(ann_path)
                    except (FileNotFoundError, OSError):
                        ann = None
                    if not ann:
                        # 没有标注 JSON 的图不凭空造文件（否则"未标注"会被误判成"已标注"）
                        skipped_no_ann += 1
                        continue
                    meta = ann.get("audit_meta") or {}
                    meta["duplicate"] = {
                        "group_id": group_id,
                        "group_size": len(members),
                        "representative": rep,
                        "similar": similar,
                        "threshold": thr,
                        "algorithm": "md5+dhash64",
                        "scanned_at": datetime.now().isoformat(),
                    }
                    ann["audit_meta"] = meta
                    save_json_atomic(ann_path, ann, backup=False)
                    st["done"] += 1  # 写回阶段进度续在哈希阶段之后

            # 清残留：上一轮扫到、这一轮已不在重复组的图，把旧 duplicate 抹掉
            # （成员集合一次性算好，别在循环里反复构建）
            in_group_ids = {items[i]["image_id"] for i in in_group}
            cleaned = 0
            for p in ann_dir.glob("*.json"):
                if st["cancelled"]:
                    break
                ann = load_json(p)
                if not ann:
                    continue
                meta = ann.get("audit_meta") or {}
                if "duplicate" in meta and p.stem not in in_group_ids:
                    meta.pop("duplicate", None)
                    ann["audit_meta"] = meta
                    save_json_atomic(p, ann, backup=False)
                    cleaned += 1

            st["status"] = "done"
            st["result_summary"] = {
                "threshold": thr,
                "total": n,
                "hashed": len(valid),
                "failed_images": st["failed_images"],
                "groups": len(dup_groups),
                "duplicate_images": sum(len(g) for g in dup_groups),
                "skipped_no_ann": skipped_no_ann,
                "cleaned_stale": cleaned,
                "groups_detail": groups_detail[:500],  # 防御上限：组特别多时详情截断
                "groups_detail_truncated": len(groups_detail) > 500,
            }
        except Exception as e:
            st["status"] = "error"
            st["error"] = str(e)
            st["last_error"] = str(e)


dup_service = DupService()

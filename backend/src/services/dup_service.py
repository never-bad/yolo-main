"""M3 重复图检测服务：全库感知哈希查重（相同/近似图片）

解决「1 万张里混入了重复图」的目标①。为什么不用千问：查重是"1 万张两两比较"
的规模问题（≈5 千万对），任何 VLM 都跑不动；标准做法是感知哈希——纯 CPU、
单张 <5ms，全量算图 + 聚类约数十秒，无需 GPU。

算法（双层，误判互补）：
  1. MD5 精确哈希：文件字节级完全一致 → 严格重复（同一张图复制/改名）。
  2. 几何不变感知哈希（D4 八型体 dHash64，标识 md5+dhash64-d4）：
     - 旧的单向水平 dHash 只能抓「缩放/压缩/调色」，对几何变换全部失效
       （实测：水平镜像距离 33、垂直翻转 22、旋转 90/180 为 30/35，均 > 阈值 6）。
     - 做法：对灰度图施加二面体群 D4 的 8 种变换（4 旋转 × 是否镜像），
       每种各算一次原版 dHash，取最小值作为规范哈希。
       D4 是群，所以图无论怎么镜像/翻转/旋转，这 8 个哈希的集合不变，
       最小值严格不变，且与长宽比无关。实测七种几何变换距离全部为 0。
     - 只存 1 个规范值（不是 8 个），两两比较仍是一次 64 位 XOR，性能不变。
     - 阈值语义不变（64 位 Hamming 距离，默认 6，1~20 可调），历史配置继续有效；
       不同内容图实测最小距离 17，误报余量充足。单图约 5.6ms。
     - 已知边界：裁剪/构图差异/局部遮挡属于"部分重复"，本方案不保证抓到。
       实测「镜像 + 约 8% 裁剪」距离 27 仍会漏（对齐后可以降到 4）。
       多尺度中心裁剪方案试过：不同尺度的哈希落在不同数值区间，取全局最小值
       会被数值大小主导而非内容相似度，反而把提亮/对比度检测弄坏，故未采用。
       要覆盖裁剪需另设计（分块哈希或特征点匹配），不在本期范围。
  3. 并查集聚类：互相近重复的图归同组（传递闭包，A≈B、B≈C → A/B/C 一组）。

写回语义（与批量审核一致，绝不碰框坐标）：
  每图 audit_meta.duplicate = {group_id, group_size, representative, similar, ...}
  - 只写已存在的标注 JSON；没有标注 JSON 的图只进结果汇总，不落盘
    （避免凭空造 JSON 把"未标注"图变成"已标注"）。
  - 重跑清残留有两道护栏：①只遍历本任务的图（不碰同数据集其它任务的图）；
    ②只清本算法（md5+dhash64-d4）自己写的结论，历史旧算法结论原样保留。
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

# 算法标识：写进 audit_meta.duplicate.algorithm。
# 换算法必须换标识——清残留时据此只清本算法自己写的结论，
# 历史旧算法（md5+dhash64）的结论原样保留。
_ALGORITHM = "md5+dhash64-d4"


def _pack64(bits: np.ndarray) -> int:
    """8x8 布尔矩阵 → 64 位整数（行优先，与旧 dHash 的位序一致）"""
    return int.from_bytes(np.packbits(bits.ravel()).tobytes(), "big")


# 二面体群 D4：4 个旋转 × 是否镜像 = 8 种几何变换，覆盖镜像/翻转/四向旋转。
# 用 PIL 的 transpose 常量表达（C 层实现，纯内存搬运，不解码不重采样）。
_D4 = (
    None,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.TRANSPOSE,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.TRANSVERSE,
)


def _dhash_one(g: Image.Image) -> int:
    """单个型体的 dHash64：灰度缩到 9x8，右像素 > 左像素 记 1（与旧算法一致）"""
    px = np.asarray(g.resize((9, 8), _RESAMPLE), dtype=np.int16)
    return _pack64(px[:, 1:] > px[:, :-1])


def _variant_hashes(g: Image.Image) -> List[int]:
    """对灰度图施加 D4 的 8 种变换，各算一次 dHash。

    为什么这样一定成立（而不是在位矩阵上凑旋转/取反）：
    D4 是一个群。若图 I 变成 I' = r·I（r 属于 D4），那么
    {g·I' | g 属于 D4} = {g·r·I} = {g·I}，集合完全相同，只是顺序被打乱。
    因此取最小值得到的规范哈希严格不随几何变换改变，与长宽比无关。
    早先试图在 8x8 位矩阵上用 rot90/转置/取反凑这个群，实测旋转 90 度
    距离仍有 23（居中取格子偏移 + 像素相等时的平局都会破坏闭合），故弃用。
    """
    out = []
    for t in _D4:
        out.append(_dhash_one(g if t is None else g.transpose(t)))
    return out


def _dhash64(img_abs: Path) -> Optional[int]:
    """几何不变感知哈希：8 个型体哈希中的最小值（规范值）。

    取最小值而不是存 8 个，是为了让两两比较仍是「一次 64 位 XOR」——
    1 万张图 5 千万对，若逐型体交叉比较要乘 64 倍，跑不动。
    残余差异只来自重采样与 JPEG 噪声，由阈值（默认 6）吸收。
    """
    try:
        with Image.open(img_abs) as im:
            g = im.convert("L")
        return min(_variant_hashes(g))
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
                        "algorithm": _ALGORITHM,
                        "scanned_at": datetime.now().isoformat(),
                    }
                    ann["audit_meta"] = meta
                    save_json_atomic(ann_path, ann, backup=False)
                    st["done"] += 1  # 写回阶段进度续在哈希阶段之后

            # 清残留：上一轮扫到、这一轮已不在重复组的图，把旧 duplicate 抹掉。
            # 两道护栏，缺一不可：
            #   ① 只遍历「本任务的图」。原实现 glob 整个目录，会把不属于本任务
            #      的图（同数据集其它标注任务）的 duplicate 一并抹掉。
            #   ② 只清「本算法上一轮写的」结论。历史用旧算法（md5+dhash64）扫出
            #      的结论一律保留不动——用户明确要求新算法只对新扫描生效。
            in_group_ids = {items[i]["image_id"] for i in in_group}
            task_ids = {it["image_id"] for it in items}
            cleaned = 0
            for image_id in task_ids:
                if st["cancelled"]:
                    break
                if image_id in in_group_ids:
                    continue
                p = ann_dir / f"{image_id}.json"
                try:
                    ann = load_json(p)
                except (FileNotFoundError, OSError):
                    continue
                if not ann:
                    continue
                meta = ann.get("audit_meta") or {}
                old = meta.get("duplicate")
                if not isinstance(old, dict):
                    continue
                if old.get("algorithm") != _ALGORITHM:
                    continue  # 旧算法/其它来源的结论，保留
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

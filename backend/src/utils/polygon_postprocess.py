# -*- coding: utf-8 -*-
"""AI 轮廓后处理管线：把分割模型/SAM 输出的密集锯齿多边形压成稀疏平滑多边形。

设计目标（对应标注页的三个痛点）：
  - 顶点太多 → 前端渲染/编辑卡顿（自适应 DP + 顶点上限）
  - 边界锯齿 → 看起来"糙"（等弧长重采样 + 闭合高斯平滑）
  - 平滑过头把边界"缩掉" → 贴合度下降（法向回投影 + IoU 兜底验证）

管线五步：
  1) 自适应 DP 简化：容差从 eps_px 起步逐步放大，直到顶点数 ≤ max_vertices；
     容差是"绝对像素"而非周长比例，避免大目标被整段削平。
  2) 等弧长重采样：把疏密不均的顶点摊匀，让高斯核对每一段的作用一致。
  3) 闭合高斯平滑：对 x/y 两个坐标序列分别做 1D 高斯滤波（wrap 边界处理闭合环），
     sigma 以"弧长"为单位，保证大小目标平滑力度一致。
  4) 法向回投影：平滑会把尖角"圆化"、把边界向内收缩。对每个顶点沿局部法向
     ±radius 采样，在 prob 图（或 mask 的模糊距离场）上找前景↔背景过渡最强处，
     把顶点拉回真实边缘。没有 prob 时退化为 mask 二值图的梯度。
  5) IoU 兜底：后处理多边形与原 mask 的 IoU < target_iou 时，逐级回退
     （去回投影 → 加大 max_vertices → 原样返回），保证"瘦身后不丢面积"。

坐标约定：输入 polygon 为原图像素坐标 [[x, y], ...]；mask/prob 与原图同尺寸
（若不同尺寸，内部按仿射缩放对齐，与平台 letterbox 反算一致）。
纯 numpy/cv2/scipy 实现，不依赖 GPU，单框耗时毫秒级。
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 是硬依赖，兜底只是让 import 不炸
    cv2 = None

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover
    gaussian_filter1d = None

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------

def _polygon_to_mask(pts: np.ndarray, shape_hw) -> np.ndarray:
    """多边形 → 二值掩码（原图尺寸）。"""
    m = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillPoly(m, [pts.reshape(-1, 1, 2).astype(np.int32)], 1)
    return m.astype(bool)


def _polygon_iou(a: np.ndarray, b: np.ndarray, shape_hw) -> float:
    """两个多边形（原图像素坐标）的 IoU，用掩码栅格化计算。"""
    ma = _polygon_to_mask(a, shape_hw)
    mb = _polygon_to_mask(b, shape_hw)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter) / float(union) if union else 1.0


def _mask_iou(poly: np.ndarray, mask: np.ndarray) -> float:
    """多边形与给定二值 mask 的 IoU。"""
    m = _polygon_to_mask(poly, mask.shape[:2])
    inter = np.logical_and(m, mask).sum()
    union = np.logical_or(m, mask).sum()
    return float(inter) / float(union) if union else 1.0


def _resample_equal_arc(pts: np.ndarray, n: int) -> np.ndarray:
    """闭合折线按弧长等距重采样为 n 个点。"""
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    total = seg.sum()
    if total <= 1e-6 or n < 3:
        return pts.copy()
    s = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = np.zeros((n, 2), dtype=float)
    for k, t in enumerate(targets):
        i = int(np.searchsorted(s, t, side="right") - 1)
        i = min(max(i, 0), len(seg) - 1)
        f = (t - s[i]) / seg[i] if seg[i] > 1e-9 else 0.0
        p0 = pts[i]
        p1 = pts[(i + 1) % len(pts)]
        out[k] = p0 + (p1 - p0) * f
    return out


def _gaussian_smooth_closed(pts: np.ndarray, sigma_px: float) -> np.ndarray:
    """闭合环坐标序列的 1D 高斯平滑（mode="wrap" 保证首尾无缝）。

    sigma_px 是**空间半径（像素）**：按环的平均边长折算成顶点序号 sigma，
    顶点越密、参与平均的邻居越多，保证不同疏密的环平滑力度一致。
    """
    if gaussian_filter1d is None or sigma_px <= 0.3 or len(pts) < 6:
        return pts.copy()
    xy = np.vstack([pts, pts[:1]])
    mean_step = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).mean())
    if mean_step <= 1e-6:
        return pts.copy()
    sigma_idx = float(sigma_px) / mean_step
    sm = np.column_stack([
        gaussian_filter1d(pts[:, 0], sigma_idx, mode="wrap"),
        gaussian_filter1d(pts[:, 1], sigma_idx, mode="wrap"),
    ])
    return sm


def _reproject_to_edge(pts: np.ndarray, edge_score: np.ndarray, radius: int,
                       search: int = 7) -> np.ndarray:
    """法向回投影：每个顶点沿局部法向在 ±radius 像素内找边缘得分最高点。

    edge_score: (H, W) float，越大越"像真实边缘"（prob 图的梯度幅值，或 mask 的距离场梯度）。
    得分图在 ±search 的 3×3 邻域内取过最大值，容忍 1px 量化噪声。
    """
    if radius <= 0:
        return pts.copy()
    h, w = edge_score.shape[:2]
    # 局部法向：取相邻两顶点连线的垂直方向（指向多边形外侧的判定交给打分）
    n = len(pts)
    out = pts.copy()
    k = max(int(search), 1)
    # 预先把得分图放大到搜索带内可整数寻址
    for i in range(n):
        p = pts[i]
        q = pts[(i + 1) % n]
        r = pts[(i - 1) % n]
        t = q - r  # 切向
        tl = np.linalg.norm(t)
        if tl < 1e-6:
            continue
        t = t / tl
        normal = np.array([-t[1], t[0]], dtype=float)
        x0, y0 = p
        best_v, best_d = -1.0, 0.0
        for d in range(-radius, radius + 1):
            if d == 0:
                continue
            xi = int(round(x0 + normal[0] * d))
            yi = int(round(y0 + normal[1] * d))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            v = edge_score[max(0, yi - k):yi + k + 1, max(0, xi - k):xi + k + 1].max()
            # 距离衰减：同等得分优先离原位近的，避免小抖动放大成漂移
            v = v - abs(d) * 0.002
            if v > best_v:
                best_v, best_d = v, d
        if best_v > 0:
            out[i] = p + normal * best_d
    return out


def _edge_score_from(prob: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """构造"边缘得分图"：prob 可用时取其梯度幅值，否则用 mask 距离场梯度。"""
    if prob is not None:
        src = np.clip(prob, 0.0, 1.0).astype(np.float32)
    else:
        # mask 距离场：前景内部到边界的距离，梯度天然在边界最强
        src = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mx = float(mag.max())
    return mag / mx if mx > 1e-6 else mag


def _thin_collinear(pts: np.ndarray, edge_pts: np.ndarray = None, min_dev: float = 1.2,
                    max_edge: float = 25.0, max_boundary_dev: float = 3.0) -> np.ndarray:
    """共线抽稀：删掉平直段上不携带形状信息的冗余顶点（密点拖不动的根源）。

    关键守卫（缺了就会退化成直边伪影，IoU 掩盖不住）：
      a) 删点后新边长度 ≤ max_edge —— 绝不允许出现横跨几十像素的长边；
      b) 新边中点到真实边界距离 ≤ max_boundary_dev —— 保证"长边"其实是贴着
         边界的短弦，而不是切过弯弧的割线；
      c) 三点近似共线（中间点偏离两端连线 < min_dev）才考虑删。
    有 edge_pts 时用 KD 树精确判 (b)；没有则退化为只看 (a)(c)。
    逐轮删除，每轮只删一个点并立即重算，避免批量删除累积出超长边。
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 5:
        return pts
    tree = None
    if edge_pts is not None and len(edge_pts) >= 3 and cKDTree is not None:
        tree = cKDTree(edge_pts)
    changed = True
    while changed and len(pts) >= 5:
        changed = False
        n = len(pts)
        for i in range(n):
            p, q, r = pts[i - 1], pts[i], pts[(i + 1) % n]
            v = r - p
            vl = float(np.linalg.norm(v))
            if vl > max_edge:
                continue                      # (a) 删完边太长
            if vl < 1e-6:
                continue
            if float(abs(np.cross(v, q - p)) / vl) > min_dev:
                continue                      # (c) 曲率明显，必须留
            if tree is not None:
                if float(tree.query((p + r) / 2.0, k=1)[0]) > max_boundary_dev:
                    continue                  # (b) 新弦切过弯，会成直边
            pts = np.delete(pts, i, axis=0)
            changed = True
            break
    return pts


def _subdivide_long_edges(pts: np.ndarray, mb: np.ndarray, edge_pts: np.ndarray,
                          max_len: float = 12.0, max_dev: float = 1.5,
                          max_iter: int = 4, max_vertices: int = 240) -> np.ndarray:
    """长边细分：DP 简化留下的"两个远端顶点直连"是直边伪影的根源——平滑和回投影
    只会挪动已有顶点，不会在长边中间补点，弯边界永远贴不上。

    对每条边取中点，若中点到真实 mask 边界（edge_pts）的距离 > max_dev，
    说明这条直边"切掉了"一段弯曲边界：在中点处（拉到真实边界最近点上）插入顶点，
    迭代直到所有边都"短且贴合"。顶点数会回升，但每个都是贴合曲率的短边——
    多几十个顶点远比手修几条直边省时间。
    """
    if edge_pts is None or len(edge_pts) == 0 or cKDTree is None or len(pts) < 3:
        return pts
    tree = cKDTree(edge_pts)
    for _ in range(max_iter):
        n = len(pts)
        if n >= max_vertices:
            break
        q = np.roll(pts, -1, axis=0)
        mid = (pts + q) / 2.0
        seg_len = np.linalg.norm(q - pts, axis=1)
        d, idx = tree.query(mid, k=1)
        # 需要细分的边：够长 且 中点偏离真实边界
        bad = (seg_len > max_len) & (d > max_dev)
        if not bad.any():
            break
        new = []
        for i in range(n):
            new.append(pts[i])
            if bad[i]:
                # 插到真实边界上（mid 的最近边界点），而不是几何中点
                new.append(edge_pts[idx[i]])
        pts = np.asarray(new, dtype=float)
        if len(pts) > max_vertices:
            break
    return pts


# ----------------------------------------------------------------------
# 主管线
# ----------------------------------------------------------------------

def mask_to_polygon(mask: np.ndarray, prob: np.ndarray = None,
                    eps_px: float = 0.5, target_iou: float = 0.97,
                    max_vertices: int = 200, smooth_sigma_px: float = 0.0,
                    reproject_radius: int = 0, subdivide: bool = True,
                    spacing_px: float = 10.0) -> np.ndarray:
    """二值掩码 → 贴合边界的多边形（Nx2, float, 原图像素坐标）。

    默认走"等弧长重采样 + 吸附真实边界 + 共线抽稀"（spacing_px>0）：
    顶点沿边界按 ~spacing_px 等距落点、每个点踩在 mask 边界上，再删掉
    平直段上不携带形状信息的冗余点。曲率大处点密、平直处点疏，
    既没有 DP 简化压出的长直边，也没有原始轮廓的密集拖不动的点。
    （实测：DP 系方案是"直边 vs 密点"二选一，重采样系两解。）
    平滑/回投影默认关闭：它们会把贴好的顶点重新抹离边界（dev 0.7→2-3px）。

    参数：
      mask             : (H, W) 二值掩码（>0 前景），与输出坐标系同尺寸（原图）
      prob             : (H, W) float 概率图（可选）。提供时回投影用它找边缘，
                         比二值 mask 的梯度更贴合真实过渡带；None 时退化为距离场。
      eps_px           : DP 初始容差（原图像素）；仅在 spacing_px<=0 的旧路径生效
      target_iou       : IoU 兜底阈值（后处理多边形 vs 原 mask），低于则逐级回退
      max_vertices     : 顶点数软上限（自适应 DP 的目标）
      smooth_sigma_px  : 高斯平滑空间半径（像素）；0 关闭
      reproject_radius : 法向回投影搜索半径（像素）；0 关闭
      subdivide        : 长边细分开关（旧 DP 路径修直边用；新路径不需要）
      spacing_px       : 重采样间距（原图像素）；>0 走新路径，<=0 退回旧 DP 路径

    返回 (N, 2) float64；失败（无轮廓/退化）返回空数组 (0, 2)。
    """
    if cv2 is None or mask is None:
        return np.zeros((0, 2), dtype=float)
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.shape[0] < 4 or m.shape[1] < 4:
        return np.zeros((0, 2), dtype=float)
    found = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in (found[0] if len(found) == 2 else found[1])
                if c is not None and len(c) >= 3]
    if not contours:
        return np.zeros((0, 2), dtype=float)
    cnt = max(contours, key=cv2.contourArea)  # 只处理最大块；多块拆分在上游 _mask_components
    if cv2.contourArea(cnt) <= 0:
        return np.zeros((0, 2), dtype=float)

    def _simplify(cnt, tol_px: float) -> np.ndarray:
        approx = cv2.approxPolyDP(cnt, max(float(tol_px), 0.5), True)
        return approx.reshape(-1, 2).astype(float)

    mb = m.astype(bool)
    tiou = float(target_iou)
    # 真实边界点集（重采样吸附/细分时把新顶点落到这里）
    edge_pts = None
    if subdivide or (spacing_px and spacing_px > 0):
        efound = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        ecnt = [c.reshape(-1, 2).astype(float) for c in
                (efound[0] if len(efound) == 2 else efound[1]) if c is not None and len(c) >= 3]
        if ecnt:
            edge_pts = np.vstack(ecnt)

    # ---- 0) 新默认路径：等弧长重采样 + 共线抽稀（点少、贴边、无直边） ----
    # DP 的问题本质：它只"删点"不"挪点"，删完远端两点直连必然出直边。
    # 这里反过来：沿真实轮廓按 spacing_px 等距落点（点永远在边界上，
    # 弯的地方自然密），再删平直段上不携带形状信息的冗余点（抽稀后边变长
    # 但贴着边界，不是直边伪影）。IoU 不达标才退回旧 DP 路径。
    if spacing_px and float(spacing_px) > 0:
        raw = cnt.reshape(-1, 2).astype(float)
        seg = np.linalg.norm(np.diff(np.vstack([raw, raw[:1]]), axis=0), axis=1)
        perim = float(seg.sum())
        n = int(perim / float(spacing_px))
        if n >= 6:
            rs = _resample_equal_arc(raw, min(n, 400))
            thinned = _thin_collinear(rs, edge_pts=edge_pts)
            if len(thinned) >= 3 and _mask_iou(thinned, mb) >= tiou:
                return np.round(thinned, 2)

    # ---- 1) 自适应 DP：沿容差阶梯放大，只接受"IoU 仍达标"的最简版本 ----
    # 先按初始容差抽一遍；顶点还多就逐级放大容差试更简的版本，
    # 一旦 IoU 掉破 target_iou 立即停 —— 宁可用顶点多的版本，不丢面积。
    tol = max(float(eps_px), 0.5)
    pts = _simplify(cnt, tol)
    if len(pts) < 3:
        return np.zeros((0, 2), dtype=float)
    for _ in range(10):
        if len(pts) <= max_vertices:
            break
        cand = _simplify(cnt, tol * 1.5)
        if len(cand) < 3 or _mask_iou(cand, mb) < tiou:
            break
        tol *= 1.5
        pts = cand

    # ---- 1.5) 长边细分：DP 留下的长直边是"直边伪影"的根源 ----
    # 平滑/回投影只挪已有顶点、不在边中间补点，弯边界永远贴不上；
    # 这里在中点偏离真实边界处插点并吸附到边界上，迭代到所有边"短且贴合"。
    if subdivide and edge_pts is not None:
        pts = _subdivide_long_edges(pts, mb, edge_pts, max_len=12.0, max_dev=1.5)

    # ---- 2)+3) 等弧长重采样 + 闭合高斯平滑（IoU 掉破阈值就跳过） ----
    if smooth_sigma_px > 0 and len(pts) >= 6:
        n_resample = int(min(max(len(pts), 24), 240))
        sm = _gaussian_smooth_closed(_resample_equal_arc(pts, n_resample), float(smooth_sigma_px))
        if len(sm) >= 3 and _mask_iou(sm, mb) >= min(tiou, _mask_iou(pts, mb)):
            pts = sm

    # ---- 4) 法向回投影（同样只在 IoU 不劣化时保留） ----
    if reproject_radius > 0:
        edge = _edge_score_from(prob, mb)
        rp = _reproject_to_edge(pts, edge, int(reproject_radius))
        if len(rp) >= 3 and _mask_iou(rp, mb) >= min(tiou, _mask_iou(pts, mb)):
            pts = rp

    # ---- 5) 兜底：最终版本仍不达标（细长目标平滑必掉面积等）→ 退回最小容差 DP ----
    if _mask_iou(pts, mb) < tiou:
        dp = _simplify(cnt, max(float(eps_px), 0.5))
        if len(dp) >= 3 and _mask_iou(dp, mb) > _mask_iou(pts, mb):
            pts = dp
    return np.round(pts, 2)


def refine_polygon(polygon, mask: np.ndarray, prob: np.ndarray = None,
                   target_iou: float = 0.97, max_vertices: int = 240,
                   smooth_sigma_px: float = 1.5, reproject_radius: int = 3,
                   subdivide: bool = True):
    """对已有稀疏多边形做"长边细分 + 平滑 + 回投影"（不重走 mask→轮廓）。

    用于 trained_model 已经给出 polygon（由 _mask_components 产出）但希望
    进一步"贴边瘦身"的场景：直接喂回原 mask 做 IoU 兜底。
    细分步骤会修掉 DP 留下的长直边（直边伪影），返回与入参同构的
    [[x, y], ...]；失败返回原 polygon。
    """
    pts = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if cv2 is None or len(pts) < 3 or mask is None:
        return polygon
    m = (np.asarray(mask) > 0)
    before = _mask_iou(pts, m)

    # 长边细分（直边修复的主步骤）
    if subdivide and cKDTree is not None:
        efound = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        ecnt = [c.reshape(-1, 2).astype(float) for c in
                (efound[0] if len(efound) == 2 else efound[1]) if c is not None and len(c) >= 3]
        if ecnt:
            pts = _subdivide_long_edges(pts, m, np.vstack(ecnt), max_len=12.0,
                                        max_dev=1.5, max_vertices=int(max_vertices))

    # 平滑
    sm = pts
    if smooth_sigma_px > 0 and len(pts) >= 6:
        sm = _resample_equal_arc(pts, int(min(max(len(pts), 24), 240)))
        sm = _gaussian_smooth_closed(sm, float(smooth_sigma_px))
    # 回投影
    out = sm
    if reproject_radius > 0:
        edge = _edge_score_from(prob, m)
        out = _reproject_to_edge(sm, edge, int(reproject_radius))

    # IoU 兜底：只接受"面积不丢"的版本（比 before 差就逐级退回）
    for cand in (out, sm, pts):
        if len(cand) >= 3 and _mask_iou(cand, m) >= min(float(target_iou), before):
            return [[round(float(x), 2), round(float(y), 2)] for x, y in cand]
    return polygon

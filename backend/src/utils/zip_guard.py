# -*- coding: utf-8 -*-
"""ZIP 解压安全收口（S-01 / S-02）。

两类风险，统一在此防护，供三处解压入口共用：
  1. **路径遍历**：成员名含绝对路径或 `..`，解压时写到目标目录之外（Zip Slip）。
  2. **解压炸弹**：极小压缩包解压出巨量数据撑爆磁盘。

收口范围：
  - `dataset_service._extract_zip`：保留其 Windows 长路径解压循环，仅把成员名判定换成本模块 `is_safe_member`
  - `model_service.upload_model`：内存流解压 → `guarded_extractall`
  - `annotation_service.import_yolo_labels`：文件解压 → `guarded_extractall`

注意：本模块只做"拒绝/跳过不安全成员 + 体积预检"，不改写正常数据的解压行为。
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

# 解压炸弹阈值（可用环境变量覆盖）
# MAX_UNCOMPRESSED_BYTES：解压后总字节上限，默认 50 GB
# MAX_RATIO：解压后/压缩前 体积比上限。图片/权重本身已压缩，正常包比值 ≈1~10；
#            取 100 留足余量，仍能拦住经典 1GB→数百GB 的炸弹包。
MAX_UNCOMPRESSED_BYTES = int(os.getenv("ZIP_MAX_UNCOMPRESSED_BYTES", 50 * 1024 ** 3))
MAX_RATIO = float(os.getenv("ZIP_MAX_RATIO", "100"))


class UnsafeZipMember(Exception):
    """zip 含不安全成员名（路径遍历）"""


class ZipBombRisk(Exception):
    """zip 解压后体积/压缩比超阈值（解压炸弹）"""


def is_safe_member(member: str) -> bool:
    """成员名是否可安全解压到目标目录内。

    拒绝：绝对路径（`/etc/x`、`C:\\x`、UNC `\\\\srv\\x`）、含 `..` 路径段。
    注意：Windows 上 `Path("/etc/x").is_absolute()` 为 False（视作盘符相对路径），
    因此必须显式判前导 `/` 或 `\\`——dataset_service 原判定在 Windows 下漏掉这一类。
    """
    if not member:
        return False
    if member.startswith(("/", "\\")):
        return False
    p = Path(member)
    if p.is_absolute() or (len(member) >= 2 and member[1] == ":"):
        return False
    if ".." in p.parts or ".." in str(p):
        return False
    return True


def check_zip_bomb(zf: zipfile.ZipFile) -> None:
    """按中央目录声明的 file_size 预检解压后总体积与压缩比。

    声明值不可信也无妨——它只用于"明显异常即拒"，真实写入量另有上限由调用方磁盘余量兜底。
    """
    total_uncompressed = 0
    total_compressed = 0
    for info in zf.infolist():
        total_uncompressed += info.file_size
        total_compressed += max(info.compress_size, 1)
    if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
        raise ZipBombRisk(
            f"ZIP 解压后体积 {total_uncompressed / 1024 ** 3:.1f} GB 超上限 "
            f"{MAX_UNCOMPRESSED_BYTES / 1024 ** 3:.0f} GB，已拒绝"
        )
    if total_uncompressed > 10 * 1024 * 1024 and total_uncompressed / total_compressed > MAX_RATIO:
        raise ZipBombRisk(
            f"ZIP 压缩比 {total_uncompressed / total_compressed:.0f}x 超上限 {MAX_RATIO:.0f}x，"
            f"疑似解压炸弹，已拒绝"
        )


def safe_members(zf: zipfile.ZipFile) -> list[str]:
    """返回可安全解压的成员名列表（不安全者由调用方记日志跳过）。"""
    return [m for m in zf.namelist() if is_safe_member(m)]


def guarded_extractall(zf: zipfile.ZipFile, dest: Path, *, strict: bool = True) -> list[str]:
    """带校验的 extractall。

    strict=True：发现不安全成员直接抛 UnsafeZipMember（上传入口用，宁可拒不可猜）。
    strict=False：跳过不安全成员继续解压其余（兼容历史宽松导入）。
    两种模式都会先做解压炸弹预检。
    """
    check_zip_bomb(zf)
    unsafe = [m for m in zf.namelist() if not is_safe_member(m)]
    if unsafe and strict:
        raise UnsafeZipMember(
            f"ZIP 含 {len(unsafe)} 个不安全路径成员（绝对路径/含 ..），已拒绝。示例：{unsafe[:3]}"
        )
    if unsafe:  # 宽松模式：逐成员解压，跳过不安全者
        for member in safe_members(zf):
            zf.extract(member, dest)
        return unsafe
    zf.extractall(dest)
    return []

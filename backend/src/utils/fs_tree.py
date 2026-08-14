"""文件系统目录树构建工具（用于前端展示数据集/任务文件夹结构）"""
from pathlib import Path
from typing import Optional


def build_tree(
    root: Path,
    url_root: Optional[Path] = None,
    max_depth: int = 6,
    max_children: int = 300,
    skip_hidden: bool = True,
) -> Optional[dict]:
    """递归构建目录树

    - 目录节点: {"name", "type": "dir", "path", "children": [...]}
    - 文件节点: {"name", "type": "file", "path", "ext", "size"}
    - path 为相对 url_root 的 posix 路径（前端可直接拼 /static/{path} 访问）；
      若未提供 url_root，则使用绝对路径。
    - 单个目录下条目过多时截断并附加 "truncated": True 标记。
    """
    if not root.exists() or not root.is_dir():
        return None

    def _rel(target: Path) -> str:
        if url_root is None:
            return str(target)
        try:
            return target.relative_to(url_root).as_posix()
        except ValueError:
            return str(target)

    def _walk(current: Path, depth: int) -> dict:
        node = {"name": current.name, "type": "dir", "path": _rel(current), "children": []}
        if depth >= max_depth:
            return node
        try:
            entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return node
        # 真实条目总数（跳过隐藏文件）：children 可能被 max_children 截断，
        # 但 file_count 必须完整，避免前端"按目录数出的图片数"偏小
        node["file_count"] = sum(
            1 for e in entries if not (skip_hidden and e.name.startswith("."))
        )
        count = 0
        for entry in entries:
            if skip_hidden and entry.name.startswith("."):
                continue
            if count >= max_children:
                node["truncated"] = True
                break
            if entry.is_dir():
                node["children"].append(_walk(entry, depth + 1))
            else:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                node["children"].append({
                    "name": entry.name,
                    "type": "file",
                    "path": _rel(entry),
                    "ext": entry.suffix.lower(),
                    "size": size,
                })
            count += 1
        return node

    return _walk(root, 0)
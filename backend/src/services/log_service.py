import asyncio
import json
from pathlib import Path

from src.core.settings import settings


# ANSI 转义码处理函数（保留 ANSI 码，供前端解析）
# 前端会解析这些转义码并渲染为彩色


def _read_log_lines(log_file: Path, offset: int, max_lines: int = None):
    """同步读取日志（B-17 重写：二进制模式，offset 恒为纯字节偏移）

    旧实现用文本模式 open(..., encoding=...) + seek/tell：TextIOWrapper.tell()
    返回的是 opaque cookie 而非字节位置，一旦 seek 到非法位置（前端曾把
    /logs/lines 返回的 total「行数」当作 /logs/tail 的「字节 offset」传入），
    后续 tell() 会返回 3.4e38 量级的污染值；该值经 JSON 传到浏览器后超出
    Number.MAX_SAFE_INTEGER，前端增量轮询全部 422 静默死亡 —— 训练卡片进度
    永远停在打开页面那一刻的快照（"最后更新 X 分钟前"持续增长）。

    二进制模式下 seek/tell 即纯字节偏移，恒可靠；解码统一 utf-8 + replace
    （日志由平台自身以 utf-8 写出）。行切分与旧文本模式（universal newlines）
    保持一致：\\r / \\r\\n / \\n 都分行（tqdm 以 \\r 刷新进度行）。

    Args:
        log_file: 日志文件路径
        offset: 读取起始字节位置（负数由调用方 tail_logs 处理为"从末尾开始"）
        max_lines: 最多读取的行数，None表示读取所有
    """
    lines = []
    new_offset = offset

    try:
        with open(log_file, "rb") as f:
            f.seek(offset)

            if max_lines is not None and max_lines > 0:
                read_rows = 0
                while read_rows < max_lines:
                    raw = f.readline()
                    if not raw:  # 到达文件末尾
                        break
                    new_offset = f.tell()  # 二进制模式：纯字节偏移，恒可靠
                    # tqdm 单个进度行内含多个 \r 刷新帧，拆分成多行与旧行为一致
                    for seg in raw.decode("utf-8", errors="replace").splitlines():
                        lines.append(seg)
                        read_rows += 1
            else:
                data = f.read()
                new_offset = f.tell()
                lines = data.decode("utf-8", errors="replace").splitlines()
    except Exception:
        # 文件不存在 / offset 非法（负数、超界）等：返回已读内容与当前 offset，
        # 调用方按"本轮无新行"处理，绝不抛出污染值
        return lines, max(0, new_offset)

    return lines, max(0, new_offset)


def _get_file_tail_offset(log_file: Path, n_lines: int = 50):
    """获取文件最后N行的起始字节偏移（B-17：二进制实现，与 _read_log_lines 的
    字节 offset 语义严格一致；旧文本模式实现返回 TextIOWrapper cookie，不可靠）

    Returns:
        int: 最后N行的起始字节偏移；文件总行数少于N行或读取失败时返回0
    """
    try:
        with open(log_file, "rb") as f:
            data = f.read()
    except Exception:
        # 文件不存在等：从0开始（下游把"读不到"当本轮无新行处理）
        return 0

    # keepends 保留行尾分隔符（\r / \r\n / \n），字节长度精确
    rows = data.splitlines(keepends=True)
    if len(rows) <= n_lines:
        return 0
    return len(data) - len(b"".join(rows[-n_lines:]))


class LogService:
    def __init__(self):
        self.jobs_dir = settings.JOBS_DIR
    
    async def stream_logs(self, job_id: str):
        """SSE流式日志 - 只推送增量日志（从末尾50条之后开始）"""
        log_file = self.jobs_dir / f"{job_id}.log"
        
        # 如果日志文件不存在，等待创建
        for _ in range(10):
            if log_file.exists():
                break
            await asyncio.sleep(0.5)
        
        if not log_file.exists():
            yield "data: Log file not found\n\n"
            return
        
        # 获取最后50行的起始位置作为初始offset，只推送增量日志
        offset = await asyncio.to_thread(_get_file_tail_offset, log_file, 50)
        
        while True:
            try:
                # 在线程中读取日志（支持多种编码），每次最多读取50条
                lines, new_offset = await asyncio.to_thread(_read_log_lines, log_file, offset, 50)
                
                # 只推送新增的日志行（增量）
                for line in lines:
                    # 清理行内容：只移除空字符和替换字符，保留 ANSI 转义码
                    cleaned_line = line.rstrip().replace('\x00', '').replace('\ufffd', '')
                    if cleaned_line:
                        yield f"data: {cleaned_line}\n\n"
                
                offset = new_offset
                
                # 检查任务是否完成
                job_file = self.jobs_dir / f"{job_id}.json"
                if await asyncio.to_thread(lambda: job_file.exists()):
                    def _check_job_status():
                        if job_file.exists():
                            with open(job_file, "r", encoding="utf-8") as f:
                                return json.load(f)
                        return None
                    
                    job_meta = await asyncio.to_thread(_check_job_status)
                    
                    if job_meta and job_meta.get("status") in ["completed", "failed", "stopped"]:
                        # 任务结束，再等待一小段时间确保日志全部输出
                        await asyncio.sleep(1)
                        # 任务结束时，读取剩余所有日志（不限制行数）
                        lines, new_offset = await asyncio.to_thread(_read_log_lines, log_file, offset, None)
                        for line in lines:
                            # 清理行内容：只移除空字符和替换字符，保留 ANSI 转义码
                            cleaned_line = line.rstrip().replace('\x00', '').replace('\ufffd', '')
                            if cleaned_line:
                                yield f"data: {cleaned_line}\n\n"
                        break
                
                await asyncio.sleep(1)
                
            except Exception as e:
                yield f"data: Error reading log: {str(e)}\n\n"
                break
    
    async def tail_logs(self, job_id: str, offset: int):
        """轮询获取增量日志，每次最多返回50条

        offset 语义（B-17）：纯字节偏移。负数约定为"从文件末尾锚定"——
        返回当前文件大小作为下一次请求的起点、不返回行；前端用 tail(-1)
        完成初始锚定，避免任何行数/字节数混用。
        """
        log_file = self.jobs_dir / f"{job_id}.log"

        if not await asyncio.to_thread(lambda: log_file.exists()):
            return {"offset": max(0, offset), "lines": []}

        if offset < 0:
            def _file_size():
                try:
                    return log_file.stat().st_size
                except Exception:
                    return 0
            size = await asyncio.to_thread(_file_size)
            return {"offset": size, "lines": []}

        # 防御（B-17）：offset 超出文件大小（历史污染值/文件被轮转）→ 钳制到文件
        # 末尾，返回空行；前端下一轮从文件尾继续，避免拿天文数字反复空转
        def _clamp():
            try:
                return min(offset, log_file.stat().st_size)
            except Exception:
                return offset
        offset = await asyncio.to_thread(_clamp)

        try:
            # 在线程中读取日志（支持多种编码），限制最多50条
            lines, new_offset = await asyncio.to_thread(_read_log_lines, log_file, offset, 50)
            
            # 清理每一行：只移除空字符和替换字符，保留 ANSI 转义码
            cleaned_lines = [
                line.rstrip().replace('\x00', '').replace('\ufffd', '')
                for line in lines
                if line.rstrip()
            ]
            
            return {"offset": new_offset, "lines": cleaned_lines}
        
        except Exception as e:
            return {"offset": offset, "lines": [], "error": str(e)}
    
    async def get_log_lines(self, job_id: str, n: int = 100):
        """获取日志文件的最后N行，n=0表示获取所有日志"""
        log_file = self.jobs_dir / f"{job_id}.log"
        
        if not await asyncio.to_thread(lambda: log_file.exists()):
            return {"lines": [], "total": 0, "error": "Log file not found"}
        
        try:
            def _read_last_n_lines():
                # 尝试多种编码方式
                encodings = ['utf-8', 'gbk', 'gb2312', 'cp936', 'latin-1']
                all_lines = []
                
                for encoding in encodings:
                    try:
                        with open(log_file, "r", encoding=encoding, errors="replace") as f:
                            all_lines = f.readlines()
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                    except Exception:
                        return [], 0
                
                # 清理每一行
                cleaned_lines = [
                    line.rstrip().replace('\x00', '').replace('\ufffd', '')
                    for line in all_lines
                    if line.rstrip()
                ]
                
                total = len(cleaned_lines)
                
                # n=0 表示返回所有日志
                if n == 0 or n >= total:
                    return cleaned_lines, total
                else:
                    # 返回最后 n 行
                    return cleaned_lines[-n:], total
            
            lines, total = await asyncio.to_thread(_read_last_n_lines)
            
            return {"lines": lines, "total": total, "returned": len(lines)}
        
        except Exception as e:
            return {"lines": [], "total": 0, "error": str(e)}

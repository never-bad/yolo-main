#!/usr/bin/env bash
# =============================================================================
# YOLO 训练可视化平台 —— 一键发布脚本（本机 → 服务器）
#
# 用法（在 Git Bash / Trae 终端里运行）:
#   bash docker/publish.sh            # 构建 + 推送 + SSH 远程部署 + 部署后验证
#   bash docker/publish.sh --dry-run  # 只打印将执行的步骤，不实际执行（先试跑看流程）
#   bash docker/publish.sh --reset   # 清除已保存的服务器配置，重新询问
#
# 首次运行会依次问三个问题（服务器 IP / SSH 用户名 / 项目路径），答案保存在
# docker/.deploy.env（已加入 .gitignore，不会提交到仓库）。之后每次全自动，
# 最多输一次服务器密码（没配免密登录的话）。
#
# 前置条件:
#   1. Docker Desktop —— 脚本检测不到 daemon 会自动尝试拉起，最多等 2 分钟
#   2. docker login 阿里云仓库 —— 若推送时报 unauthorized，去阿里云「容器镜像服务」
#      控制台取新的临时令牌，执行:
#      docker login crpi-ujjfs42qeoqwaisn.cn-hangzhou.personal.cr.aliyuncs.com
#      再重跑本脚本
#
# 本脚本做什么（顺序）:
#   1. 检查本机 Docker daemon（未运行则自动启动 Docker Desktop 并等待就绪）
#   2. 调 docker/build_push.sh --app-only 构建并推送代码层镜像
#      （环境层镜像沿用仓库里已有的，依赖没变不重建 —— 复用首次部署的成果）
#   3. SSH 到服务器执行 docker/deploy.sh（装依赖/拉镜像/启动/健康检查，幂等可重复）
#   4. 部署后验证（本脚本追加，deploy.sh 未覆盖的部分）:
#      a. 镜像内 /app/scripts 是否存在（G-2 镜像脚本补验，docker 内直接查）
#      b. 后端健康接口 —— 从容器内部探测 :8000/health，
#         不依赖宿主端口映射（deploy.sh 探测宿主 :8000，而该端口被同事服务占用，
#         属已知误报，待批次 I 修正 deploy.sh）
#
# 不做的事（红线）:
#   - 不执行 migrate_to_pg.py / DROP TABLE 等任何数据库操作（批次 I 的 I-05 负责）
#   - 不重建环境层镜像（依赖变更时手动: bash docker/build_push.sh --env-only 或全量）
# =============================================================================
set -euo pipefail

DRY_RUN=0
RESET=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --reset)   RESET=1 ;;
        *) echo "未知参数: $arg（支持 --dry-run / --reset）"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF="$SCRIPT_DIR/.deploy.env"

RED='\033[1;31m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[发布]${NC} $*"; }
warn() { echo -e "${YELLOW}[警告]${NC} $*"; }
fail() { echo -e "${RED}[错误]${NC} $*"; exit 1; }
# 统一执行入口: dry-run 模式只打印不执行
run()  { if [ "$DRY_RUN" -eq 1 ]; then echo -e "  ${YELLOW}[dry-run]${NC} $*"; else "$@"; fi; }

# ---------- 0. 服务器配置（首次交互式询问，之后读 .deploy.env / 环境变量） ----------
[ "$RESET" -eq 1 ] && rm -f "$CONF"
if [ -f "$CONF" ]; then
    # shellcheck disable=SC1090
    . "$CONF"
fi

ask() {  # ask <变量名> <提示> [默认值]
    local __var="$1" __hint="$2" __def="${3:-}" __ans
    if [ -n "$__def" ]; then
        read -r -p "$__hint [回车默认: $__def]: " __ans
        __ans="${__ans:-$__def}"
    else
        read -r -p "$__hint: " __ans
    fi
    [ -n "$__ans" ] || fail "该项不能为空"
    printf -v "$__var" '%s' "$__ans"
}

NEED_ASK=0
[ -z "${SERVER:-}" ]     && NEED_ASK=1
[ -z "${SSH_USER:-}" ]   && NEED_ASK=1
[ -z "${REMOTE_DIR:-}" ] && NEED_ASK=1
if [ "$NEED_ASK" -eq 1 ]; then
    log "首次运行，配置服务器信息（保存在 docker/.deploy.env；以后可用 --reset 重设）"
    [ -z "${SERVER:-}" ]     && ask SERVER     "服务器 IP（XShell 会话里能看到）" "192.168.1.114"
    [ -z "${SSH_USER:-}" ]   && ask SSH_USER   "SSH 用户名（XShell 登录用的那个）" "root"
    [ -z "${REMOTE_DIR:-}" ] && ask REMOTE_DIR "项目在服务器上的路径（其下须有 docker 子目录，如 /opt/yolo-main）" ""
    cat > "$CONF" <<EOF
# publish.sh 服务器配置（勿提交，已被 .gitignore 排除）
SERVER="${SERVER}"
SSH_USER="${SSH_USER}"
REMOTE_DIR="${REMOTE_DIR}"
SSH_PORT="${SSH_PORT:-22}"
EOF
    log "配置已保存: $CONF"
fi
SSH_PORT="${SSH_PORT:-22}"

# ---------- 1. 本机 Docker daemon ----------
if docker info >/dev/null 2>&1; then
    log "Docker daemon 正常（$(docker --version 2>/dev/null)）"
elif [ "$DRY_RUN" -eq 1 ]; then
    warn "[dry-run] Docker daemon 未运行（正式运行时会自动尝试启动 Docker Desktop）"
else
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            DD="/c/Program Files/Docker/Docker/Docker Desktop.exe"
            if [ -f "$DD" ]; then
                warn "Docker daemon 未运行，自动启动 Docker Desktop（最多等 2 分钟）..."
                powershell.exe -NoProfile -Command "Start-Process -FilePath 'C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe'"
                ok=0
                for _ in $(seq 1 24); do
                    if docker info >/dev/null 2>&1; then ok=1; break; fi
                    sleep 5
                done
                [ "$ok" -eq 1 ] || fail "Docker daemon 2 分钟内未就绪，请手动启动 Docker Desktop 后重跑本脚本"
                log "Docker daemon 已就绪"
            else
                fail "Docker daemon 未运行且未找到 Docker Desktop，请先启动后重跑本脚本"
            fi
            ;;
        *)
            fail "Docker daemon 未运行，请先启动 Docker 再重跑本脚本"
            ;;
    esac
fi

# ---------- 2. 构建并推送代码层镜像（环境层复用仓库已有，不重建） ----------
log "构建并推送代码层镜像（环境层沿用首次部署成果，依赖未变不重建）"
run bash "$SCRIPT_DIR/build_push.sh" --app-only \
    || fail "镜像构建/推送失败。若是 unauthorized: 到阿里云容器镜像服务控制台取临时令牌，执行 docker login crpi-ujjfs42qeoqwaisn.cn-hangzhou.personal.cr.aliyuncs.com 后重跑"

# ---------- 3. SSH 远程部署 + 部署后验证（一次连接，最多输一次密码） ----------
log "SSH 到 ${SSH_USER}@${SERVER}:${SSH_PORT}，在 ${REMOTE_DIR} 执行部署..."
run ssh -t -p "$SSH_PORT" "${SSH_USER}@${SERVER}" "\
cd '${REMOTE_DIR}/docker' && sudo bash deploy.sh && \
echo '' && \
echo '========== 部署后验证（publish.sh 追加） ==========' && \
echo '--- a. 镜像内 /app/scripts（G-2 补验）---' && \
sudo docker exec yolo-platform ls /app/scripts && \
echo '--- b. 后端健康检查（容器内 :8000/health）---' && \
sudo docker exec yolo-platform python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=10); print('health OK')\" && \
echo '========== 验证通过，部署完成 =========='" \
    || fail "远程部署或验证失败。请把上方完整输出发给 AI 助手排查（常见: docker login 过期 / 路径不对 / 服务器磁盘满）"

log "全部完成。前端 http://${SERVER}:3000（局域网），数据卷未动。"

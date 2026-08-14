#!/usr/bin/env bash
# =============================================================================
# YOLO 训练可视化平台 —— 服务器端零配置部署脚本
#
# 用法:
#   cd /path/to/yolo-main/docker
#   sudo bash deploy.sh
#   （镜像名与 compose 默认不一致时才需指定：YOLO_IMAGE=<镜像>:<标签> sudo bash deploy.sh）
# 注意: 阿里云/腾讯云仓库是私有的，第一次部署前请先执行 docker login（每次临时令牌也需在此登录）
#
# 功能顺序:
#   1. 检查 Docker（缺失则自动安装）
#   2. 检查 NVIDIA Container Toolkit（缺失则自动安装）
#   3. 验证 GPU 对容器可见（仅提示，不影响启动）
#   4. 从镜像仓库拉取统一镜像并启动（零配置，无需上传代码/模型）
#   5. 健康检查并输出访问地址
# =============================================================================
set -euo pipefail

RED='\033[1;31m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[部署]${NC} $*"; }
warn() { echo -e "${YELLOW}[警告]${NC} $*"; }
fail() { echo -e "${RED}[错误]${NC} $*"; exit 1; }

# ---------- 1. Docker ----------
if command -v docker >/dev/null 2>&1; then
    log "Docker 已安装：$(docker --version 2>/dev/null)"
else
    warn "未检测到 Docker，开始自动安装..."
    (curl -fsSL https://get.docker.com | sh) || fail "Docker 自动安装失败，请手动安装后重新执行本脚本"
    systemctl enable --now docker
    log "Docker 安装完成"
fi

# ---------- 2. NVIDIA Container Toolkit ----------
if docker info 2>/dev/null | grep -qi "nvidia"; then
    log "NVIDIA Container Toolkit 已配置"
else
    warn "未检测到 NVIDIA Container Toolkit，开始自动安装..."
    if curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null && \
       curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list; then
        apt-get update >/dev/null && apt-get install -y nvidia-container-toolkit >/dev/null
        nvidia-ctk runtime configure --runtime=docker
        systemctl restart docker
        log "NVIDIA Container Toolkit 安装完成"
    else
        warn "Toolkit 源下载失败，请手动安装（不影响服务启动，但容器将无法使用 GPU）"
    fi
fi

# ---------- 3. GPU 可见性检查（仅提示，不阻断） ----------
if docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    log "GPU 已对容器可见："
    docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi | head -n 12 || true
else
    warn "GPU 对容器不可见（NVIDIA 驱动版本过低或 Toolkit 未生效），训练将退化到 CPU"
fi

# ---------- 4. 拉取镜像并启动 ----------
cd "$(dirname "$0")"
if ! docker compose pull >/dev/null 2>&1; then
    warn "镜像拉取失败。若是私有仓库，请先执行: docker login crpi-ujjfs42qeoqwaisn.cn-hangzhou.personal.cr.aliyuncs.com"
    docker compose pull
fi
log "启动服务（模型已内置，无需上传代码/模型）..."
docker compose up -d
log "服务已启动："
docker compose ps

# ---------- 5. 健康检查 ----------
log "等待后端就绪..."
backend_ready=0
for i in $(seq 1 90); do
    if (echo > "/dev/tcp/127.0.0.1/8000") 2>/dev/null; then backend_ready=1; break; fi
    sleep 2
done
if [ "$backend_ready" -eq 1 ]; then
    log "后端健康检查通过（端口 8000）"
else
    warn "90 秒内未探测到后端端口，请检查日志: docker logs yolo-platform"
fi

# ---------- 6. 输出访问信息 ----------
SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "=========================================================="
echo "  部署完成！访问地址:"
echo "    本机访问 : http://localhost:3000"
[ -n "$SERVER_IP" ] && echo "    局域网/公网: http://$SERVER_IP:3000"
echo ""
echo "  常用命令:"
echo "    查看状态 : cd $(pwd) && docker compose ps"
echo "    查看日志 : docker logs -f yolo-platform"
echo "    停止平台 : cd $(pwd) && docker compose down"
echo "    更新部署 : 更新镜像后重新执行本脚本（docker compose pull && docker compose up -d）"
echo "=========================================================="
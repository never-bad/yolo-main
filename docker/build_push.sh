#!/usr/bin/env bash
# =============================================================================
# YOLO 训练可视化平台 - 构建 & 推送镜像脚本
#
# 在【本地有模型的机器】上执行：自动构建统一镜像（模型已打进镜像）并推送到仓库。
#
# 用法:
#   bash docker/build_push.sh [镜像仓库地址] [标签]
#
# 示例:
#   bash docker/build_push.sh   # 默认推送到下方阿里云仓库
#   bash docker/build_push.sh crpi-ujjfs42qeoqwaisn.cn-hangzhou.personal.cr.aliyuncs.com/yolo-main/yolo-platform latest
#   bash docker/build_push.sh registry.cn-hangzhou.aliyuncs.com/your-ns/yolo-platform v1.0
#   bash docker/build_push.sh ccr.ccs.tencentyun.com/your-ns/yolo-platform latest
#
# 前置:
#   - 已安装 Docker 并启动
#   - 已登录目标仓库（docker login <仓库域名>）
#   - backend/models/ 下已放入全部模型（GD / sam_b.pt / YOLO 权重）
# =============================================================================
set -euo pipefail

REGISTRY="${1:-crpi-ujjfs42qeoqwaisn.cn-hangzhou.personal.cr.aliyuncs.com/yolo-main/yolo-platform}"
TAG="${2:-latest}"
IMAGE="${REGISTRY}:${TAG}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "  构建统一镜像: ${IMAGE}"
echo "  项目目录    : ${PROJECT_ROOT}"
echo "=========================================================="

# 前置检查：模型目录非空（镜像会把模型打进去，空目录说明漏了文件）
if [ ! -d backend/models ] || [ -z "$(ls -A backend/models 2>/dev/null)" ]; then
    echo "[警告] backend/models/ 为空！请确认已放入模型文件（sam_b.pt、GD 模型、YOLO 权重等）。" >&2
    read -r -p "继续构建？(y/N) " ans
    [[ "${ans:-N}" =~ ^[Yy]$ ]] || exit 1
fi

echo "[1/3] 构建镜像（首次需数分钟，含 torch cu128 下载）..."
# --network=host: 复用宿主机网络/DNS，规避容器桥接网络 DNS 解析失败（内网服务器常见）
# --provenance=false --sbom=false: 关闭 BuildKit 附加的 OCI 元数据（provenance/sbom），
#   否则推送到阿里云个人版仓库报 "unknown manifest class for application/vnd.oci.empty.v1+json"
#   有的 Docker 版本若提示未知 flag，可改用: DOCKER_BUILDKIT=0 docker build --network=host ...
docker build --network=host --provenance=false --sbom=false -f docker/Dockerfile -t "$IMAGE" .

echo "[2/3] 推送镜像..."
docker push "$IMAGE"

echo "[3/3] 完成！"
echo ""
echo "  镜像: ${IMAGE}"
echo ""
echo "  服务器端零配置部署："
echo "    docker pull ${IMAGE}"
echo "    docker compose up -d   # 需 IMAGE 环境变量一致或改 compose 中的 image"
echo "  或直接运行："
echo "    docker run -d --name yolo -p 3000:80 -p 8000:8000 --gpus all --shm-size=24gb ${IMAGE}"
echo "=========================================================="
#!/usr/bin/env bash
# =============================================================================
# YOLO 训练可视化平台 - 分层镜像构建 & 推送脚本（环境层/代码层分开上传）
#
# 用法:
#   bash docker/build_push.sh                    # 全量：构建并推送 环境层 + 代码层（首次/依赖变更时）
#   bash docker/build_push.sh --env-only         # 仅环境层（依赖变更时）
#   bash docker/build_push.sh --app-only         # 仅代码层（日常代码/模型更新，推荐）
#
# 仓库地址可自定义（与 docker-compose.yml 中 YOLO_IMAGE 前缀保持一致）:
#   bash docker/build_push.sh --registry crpi-ujjfs42qeoqwaisn.cn-hangzhou.personal.cr.aliyuncs.com/yolo-main
#
# 分层说明:
#   yolo-app-env (docker/Dockerfile.env)  系统依赖 + torch cu128 + Python 依赖 + nginx/supervisor
#   yolo-app     (docker/Dockerfile)      FROM 环境层 + 前端产物 + 后端代码 + 模型文件
#
# 前置:
#   - 已安装 Docker 并启动
#   - 已登录目标仓库（docker login <仓库域名>）
#   - backend/models/ 下已放入全部模型（GD / sam_b.pt / YOLO 权重）
# =============================================================================
set -euo pipefail

DEFAULT_REGISTRY="crpi-ujjfs42qeoqwaisn.cn-hangzhou.personal.cr.aliyuncs.com/yolo-main"
REGISTRY="$DEFAULT_REGISTRY"
MODE="all"
TAG="latest"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-only) MODE="env"; shift ;;
        --app-only) MODE="app"; shift ;;
        --registry) REGISTRY="$2"; shift 2 ;;
        *) TAG="$1"; shift ;;  # 兼容旧用法：位置参数视为 tag
    esac
done

ENV_IMAGE="$REGISTRY/yolo-app-env:$TAG"
APP_IMAGE="$REGISTRY/yolo-app:$TAG"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "  仓库   : ${REGISTRY}"
echo "  模式   : ${MODE}（all=环境+代码 / env=仅环境 / app=仅代码）"
echo "  标签   : ${TAG}"
echo "=========================================================="

if [[ "$MODE" != "app" ]]; then
    echo "[1/3] 构建环境层镜像: ${ENV_IMAGE}"
    # --network=host: 复用宿主机网络/DNS，规避容器桥接网络 DNS 解析失败（内网服务器常见）
    # --provenance=false --sbom=false: 关闭 BuildKit 附加的 OCI 元数据（provenance/sbom），
    #   否则推送到阿里云个人版仓库报 "unknown manifest class for application/vnd.oci.empty.v1+json"
    docker build --network=host --provenance=false --sbom=false \
        -f docker/Dockerfile.env -t "$ENV_IMAGE" .
    echo "[2/3] 推送环境层镜像..."
    docker push "$ENV_IMAGE"
fi

if [[ "$MODE" != "env" ]]; then
    # 前置检查：模型目录非空（代码层会把模型打进去，空目录说明漏了文件）
    if [ ! -d backend/models ] || [ -z "$(ls -A backend/models 2>/dev/null)" ]; then
        echo "[警告] backend/models/ 为空！请确认已放入模型文件（sam_b.pt、GD 模型、YOLO 权重等）。" >&2
        read -r -p "继续构建？(y/N) " ans
        [[ "${ans:-N}" =~ ^[Yy]$ ]] || exit 1
    fi

    echo "构建代码层镜像: ${APP_IMAGE}"
    echo "  基础环境层 : ${ENV_IMAGE}"
    docker build --network=host --provenance=false --sbom=false \
        --build-arg ENV_IMAGE="$ENV_IMAGE" \
        -f docker/Dockerfile -t "$APP_IMAGE" .
    echo "推送代码层镜像..."
    docker push "$APP_IMAGE"
fi

echo ""
echo "  完成！镜像:"
[[ "$MODE" != "app" ]] && echo "    环境层 ${ENV_IMAGE}"
[[ "$MODE" != "env" ]] && echo "    代码层 ${APP_IMAGE}"
echo ""
echo "  服务器端零配置部署（compose 默认指向代码层）:"
echo "    docker pull ${APP_IMAGE}"
echo "    docker compose up -d   # 需 YOLO_IMAGE=${APP_IMAGE} 或保持默认"
echo "=========================================================="
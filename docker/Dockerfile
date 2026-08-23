# =============================================================================
# YOLO 训练可视化平台 - 代码层镜像（backend + frontend + 模型，单容器）
#
# 分层设计（环境与代码分开更新）：
#   基础镜像：yolo-app-env（docker/Dockerfile.env）已含系统依赖 + torch cu128 + Python 依赖 + nginx/supervisor
#   本镜像  ：叠加前端构建产物 + 后端代码 + 模型文件
#
# 日常更新流程（代码/模型变化时环境层无需重建）:
#   docker build -f docker/Dockerfile -t <REG>/yolo-main/yolo-app:latest .
#   仅 frontend-builder 阶段与代码/模型 COPY 层变化，耗时远小于环境层重建。
#
# 基础镜像名可用 --build-arg ENV_IMAGE=... 覆盖（默认使用环境层 latest）
#
# 构建（在有模型的机器上执行）:
#   docker build --build-arg ENV_IMAGE=crpi-ujjfs42qeoqwaisn.cn-hangzhou.personal.cr.aliyuncs.com/yolo-main/yolo-app-env:latest \
#       -f docker/Dockerfile -t crpi-ujjfs42qeoqwaisn.cn-hangzhou.personal.cr.aliyuncs.com/yolo-main/yolo-app:latest .
# =============================================================================

# 环境层镜像（默认阿里云个人版仓库；本地调试可指到已构建的本地环境镜像）
ARG ENV_IMAGE=crpi-ujjfs42qeoqwaisn.cn-hangzhou.personal.cr.aliyuncs.com/yolo-main/yolo-app-env:latest

# ---------------- 阶段 A：构建前端产物（独立阶段，npm 依赖不进入运行时） ----------------
FROM node:22-alpine AS frontend-builder

# npm 国内镜像加速
RUN npm config set registry https://registry.npmmirror.com

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/src ./src
COPY frontend/index.html ./
COPY frontend/vite.config.ts ./
COPY frontend/tsconfig.json ./
COPY frontend/tsconfig.node.json ./
COPY frontend/.env.production ./
RUN npm run build:prod

# ---------------- 运行时（基于环境层镜像） ----------------
FROM ${ENV_IMAGE}

# 后端代码
COPY backend/src /app/src
COPY backend/run.py /app/run.py

# 模型文件直接打进代码层（Grounding-DINO / SAM sam_b.pt / YOLO 训练权重）
# 新增训练权重：放入 backend/models/custom 后重新构建代码层镜像
COPY backend/models /app/models

# 数据目录：dataset 持久化在 /app/data（compose 用命名卷挂载）；训练输出写入 /app/runs
RUN mkdir -p /app/data /app/runs

# 前端产物（阶段 A）与单容器 nginx 配置
COPY --from=frontend-builder /app/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY docker/supervisord.conf /etc/supervisor/conf.d/yolo.conf
RUN mkdir -p /var/log/supervisor /var/log/nginx

EXPOSE 8000 80

# 镜像级健康检查（python:3.11-slim 无 curl，用 urllib 探测后端 /health）
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# 统一入口：supervisord 同时拉起 uvicorn(8000) 与 nginx(80)
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/yolo.conf"]
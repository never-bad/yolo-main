# =============================================================================
# YOLO 训练可视化平台 - 统一生产镜像（backend + frontend 单容器）
#
# 本镜像把前后端、Nginx、训练模型全部打进一个镜像：
#   1. 前端阶段：Node 22 构建 Vue3 产物
#   2. 运行时阶段：Python 3.11 + torch(CUDA 12.8) + nginx + supervisor
#      模型文件 backend/models/ 直接 COPY 进 /app/models，容器内训练/推理零联网
#
# 说明（基础镜像取舍）：
#   - RTX 5070 (Blackwell sm_120) 需要 CUDA 12.8，nvidia/cuda:11.8 无法驱动；
#   - torch cu128 的 pip wheel 自带 CUDA 运行库，无需 nvidia/cuda 基础镜像，
#     使用 python:3.11-slim 体积更小、构建更快。
#   - 如确需 nvidia/cuda 基础镜像，请改用 nvidia/cuda:12.8.0-runtime-ubuntu22.04。
#
# 构建（在有模型的机器上执行）:
#   docker build -f docker/Dockerfile -t crpi-ujjfs42qeoqwaisn.cn-hangzhou.personal.cr.aliyuncs.com/yolo-main/yolo-platform:latest .
# =============================================================================

# ---------------- 阶段 1：构建前端产物 ----------------
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

# ---------------- 阶段 2：运行时（Python + torch + nginx + supervisor） ----------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# APT 换中科大镜像（同时兼容 Debian12 传统 sources.list 与 Debian13+ trixie 的 deb822 debian.sources）
RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's@https\?://deb.debian.org@http://mirrors.ustc.edu.cn@g; s@https\?://security.debian.org@http://mirrors.ustc.edu.cn/debian-security@g' /etc/apt/sources.list.d/debian.sources; \
    else \
        sed -i 's@https\?://deb.debian.org@http://mirrors.ustc.edu.cn@g; s@https\?://security.debian.org@http://mirrors.ustc.edu.cn/debian-security@g' /etc/apt/sources.list; \
    fi; \
    cat /etc/apt/sources.list.d/debian.sources 2>/dev/null | grep -E 'URIs|deb ' | head -n 4 || true

# 系统依赖：OpenCV/Pillow/ffmpeg 运行库 + nginx + supervisor
# （supervisor 统一托管 backend(uvicorn) 与 frontend(nginx) 两个进程）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    # Pillow 构建依赖
    libjpeg-dev \
    zlib1g-dev \
    libtiff-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 依赖：先装 torch/torchvision stable cu128（支持 RTX 50 系 Blackwell）再装其余（清华源）
# 踩坑记录（为何用 wheel 直链）：
#   1. 官方 download.pytorch.org 在境外，2.5GB 大文件极易 BrokenPipe 断流；
#   2. mirrors.aliyun.com/pytorch-wheels 目录能看到 wheel 文件，但缺少 PEP 503
#      简易索引页（/cu128/torch/ 返回 404），pip 报 "from versions: none"；
#   3. 因此改为直接下载 wheel 文件（文件名已从镜像目录列表核对），
#      其依赖（nvidia-* / sympy / networkx 等）自动从清华 pypi 安装。
#      配套版本对：torch 2.11.0+cu128 与 torchvision 0.26.0+cu128（同批发布，依赖 torch==2.11.0）。
# 注意：文件名中的 "+" 必须 URL 编码为 %2B，否则阿里 OSS 会 403。
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --default-timeout=300 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl" \
    "https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.26.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl" \
 && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=300 -r /app/requirements.txt --no-cache-dir

# 后端代码
COPY backend/src /app/src
COPY backend/run.py /app/run.py

# 模型文件直接打进镜像（Grounding-DINO / SAM sam_b.pt / YOLO 训练权重）
# 新增训练权重：放入 backend/models/custom 后重新构建镜像
COPY backend/models /app/models

# 数据目录：dataset 持久化在 /app/data（compose 用命名卷挂载）；训练输出写入 /app/runs
RUN mkdir -p /app/data /app/runs

# 前端产物（阶段 1）与单容器 nginx 配置
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
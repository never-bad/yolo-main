# 大模型智能预标注功能 · 设计文档

> 版本：v2.0  
> 日期：2026-08-10  
> 关联参考：`xiaocLoveMoney-融合开发参考.md` 第 3.2 节（SAM3 大模型智能预标注）

---

## 1. 背景与目标

现有标注工具（`frontend/src/pages/DatasetAnnotate.vue`）依赖纯手工拖框标注，效率低。本功能引入 AI 预标注，让用户基于**类别列表**一键对当前图片或整个数据集自动生成目标框，再人工微调，目标将标注效率提升 5-10 倍。

### 1.1 两种使用模式

| 模式 | 说明 | 触发方式 |
|------|------|---------|
| 单图预标注（交互式） | 对当前图片执行一次 AI 检测/分割，结果填入画布 | 画布「AI 预标注」按钮 |
| 整集批量预标注（异步后台） | 对任务下所有**未标注**图片批量标注，完成后通知 | 任务页「批量预标注」按钮 |

---

## 2. 技术选型

### 2.1 选型约束

> ⚠️ 参考文档约束：Myolotrain 为 **AGPL-3.0**，仅参考其设计思路，**禁止复制其源代码**。底层全部使用 MIT/开源库从零实现。

### 2.2 组件选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 分割基座 | `ultralytics` 内置 `SAM`（`from ultralytics import SAM`） | requirements 已有 `ultralytics==8.3.251`，无需新增重依赖 |
| 文本驱动检测（主路线） | `YOLO-World`（`yolov8s-world.pt`） | 用类别名自动生成目标框，实现「零样本预标注」 |
| GPU 调度 | `torch.cuda` + `nvidia-smi` 探测 | 参考文档 2.3 / 3.2 方案 |
| 配置存储 | `sam_config.json` 文件 | 与现有 `settings.py` 文件系统风格一致 |

### 2.3 主流程（推荐）

由于最终目标是 BBox 标注（非分割 mask），YOLO-World 输出的检测框在多数场景下已足够使用。SAM 精修作为**可选增强**，仅在需要精细边界时启用。

```
┌─ 默认模式（快速，推荐首次预标注）──────────────────┐
│ YOLO-World 按类别检测 → BBox 直接返回              │
│ 耗时：~0.5s/张 | 显存：~2GB                        │
└────────────────────────────────────────────────────┘

┌─ 增强模式（精修，适合边缘模糊/重叠严重的目标）──────┐
│ YOLO-World 检测 → SAM 按 box 批量分割 → 外接矩形    │
│ 耗时：~2-3s/张 | 显存：~5GB                        │
└────────────────────────────────────────────────────┘
```

**调用方通过 `use_sam: bool = False` 参数切换。** 默认走快速模式，用户可在预设中启用 SAM 精修。

### 2.4 SAM 批量推理优化

**不使用逐框推理**（for box in boxes → sam.predict(image, box)），而是一次性将所有 box prompt 批量送入 SAM：

```python
# ❌ 逐框（慢）
for box in yolo_boxes:
    mask = sam.predict(image, box)

# ✅ 批量（快 3-5 倍）
masks = sam.predict(image, all_boxes)  # SAM 原生支持多 prompt 批量
```

---

## 3. 目录结构变更

```
yolo-main/
├── backend/
│   ├── src/
│   │   ├── core/
│   │   │   └── settings.py                  # 改：新增 SAM_MODELS_DIR / SAM_CONFIG_FILE / PRESETS_DIR
│   │   ├── services/
│   │   │   ├── sam_service.py               # 新：SAM 模型缓存/推理/预标注核心 + 质量报告 + 预设管理
│   │   │   └── device_manager.py            # 新：GPU 探测与推荐
│   │   └── api/routes/
│   │       ├── sam.py                       # 新：SAM 路由
│   │       └── annotations.py               # 改：注册 sam 路由
│   └── main.py                              # 改：include sam.router + resume_batch_tasks()
├── models/
│   └── sam/                                 # SAM 权重目录（sam_b.pt 等）
├── data/
│   ├── sam_config.json                      # 模型路径/imgsz/device 配置
│   ├── presets/                             # 预标注预设文件（按任务保存参数组合）
│   │   └── {task_name}.json
│   └── batch_tasks/                         # 批量任务状态落盘（防重启丢失）
│       └── {batch_id}.json
└── docs/
    └── design/
        └── SAM预标注功能设计文档.md           # 本文档
```

---

## 4. 后端设计

### 4.1 配置（`backend/src/core/settings.py`）

```python
# 新增
SAM_MODELS_DIR: Path = MODELS_DIR / "sam"       # backend/models/sam
SAM_MODEL_PATH: Path = SAM_MODELS_DIR / "sam_b.pt"
SAM_IMGSZ: int = 1024
SAM_HALF: bool = False                          # 半精度，省显存
SAM_CONFIG_FILE: Path = DATA_DIR / "sam_config.json"
PRESETS_DIR: Path = DATA_DIR / "presets"        # 预标注预设目录
BATCH_TASKS_DIR: Path = DATA_DIR / "batch_tasks" # 批量任务落盘目录
```

*说明：`settings.py` 已有 `init_directories()` 机制，将 `PRESETS_DIR` 和 `BATCH_TASKS_DIR` 加入自动创建列表。`SAM_MODELS_DIR` 需用户手动放置权重文件。*

### 4.2 `sam_service.py`（核心服务）

```python
class SAMService:
    def __init__(self):
        self._model = None          # 全局单例缓存
        self._model_name = None     # 记录已加载的权重名，避免重复加载
        self.lock = asyncio.Lock()  # 并发加载保护
        self.batch_tasks = {}       # batch_id -> BatchTask（内存缓存，落盘为真源）

    # --- 模型生命周期（单例缓存，参考文档 MODEL_CACHE 模式）---
    async def get_model(self):          # 未加载才实例化，否则返回缓存
    async def is_available(self) -> dict
    async def validate_model(self) -> dict
    async def download_model(self) -> dict  # 【P3 新增】权重下载与进度

    # --- 单图预标注 ---
    async def auto_label(
        self,
        task_id: str,
        image_id: str,
        classes: list[str],
        conf: float | dict[str, float] = 0.25,  # 【优化】支持按类别细粒度阈值
        prompts: dict[str, str] | None = None,   # 【新增】YOLO-World 文本提示词映射
        use_sam: bool = False,                   # 【P0】SAM 从必选改为可选
        only_unannotated: bool = True,           # 【P1】默认跳过已标注图片
    ) -> list[dict]:
        # 1) 从 task.json 的 items 定位真实图片绝对路径
        # 2) YOLO-World 按每类的 prompts（或 fallback 到 class_name）检测 → boxes
        # 3) 可选：SAM 批量分割 → 最小外接矩形（仅 use_sam=True）
        # 4) 后处理过滤：尺寸/宽高比/位置/重叠去重
        # 5) 返回 [{class_id, x1, y1, x2, y2, score}]

    # --- 批量预标注（异步后台任务）---
    async def batch_auto_label(
        self,
        task_id: str,
        classes: list[str],
        conf: float | dict = 0.25,
        prompts: dict | None = None,
        use_sam: bool = False,
        only_unannotated: bool = True,   # 【P1】跳过已有标注的图片
    ) -> str:
        # 生成 batch_id，创建落盘文件 data/batch_tasks/{batch_id}.json
        # 遍历 items（跳过已标注的），逐张 auto_label
        # 每张完成后：写 annotations.json → 更新进度到落盘文件
        # 完成后：生成质量报告 → 标记 status="done"

    async def resume_batch_tasks(self):   # 【P0】启动时恢复未完成的任务
        # 扫描 data/batch_tasks/*.json
        # status="running" 的标记为 "crashed"，供前端展示
        # status="pending" 的恢复执行

    # --- 质量报告（【P2】新增）---
    async def generate_quality_report(
        self, task_id: str, batch_id: str
    ) -> dict:
        # 返回：
        # {
        #   "total_images": 100,
        #   "total_boxes": 487,
        #   "avg_boxes_per_image": 4.9,
        #   "empty_images": 3,
        #   "empty_image_list": ["img_042.jpg", "img_088.jpg", "img_123.jpg"],
        #   "low_confidence_boxes": 42,
        #   "class_distribution": {"helmet": 201, "person": 186},
        #   "class_imbalance_warning": true,
        #   "per_image_flags": [
        #     {"image": "001.jpg", "boxes": 23, "flag": "too_many"},
        #     {"image": "042.jpg", "boxes": 0,  "flag": "empty"}
        #   ]
        # }

    # --- 后处理过滤（【新增】）---
    def _post_filter(self, boxes: list, filters: dict) -> list:
        # 支持规则：
        #   min_area: 最小框面积（px²）
        #   max_area: 最大框面积
        #   min_aspect: 最小宽高比
        #   max_aspect: 最大宽高比
        #   roi: 有效区域 [(x1,y1,x2,y2), ...]（排除天花板等无效区域）
        #   iou_threshold: NMS 去重阈值
        #   class_region: {"class_name": [(x1,y1,x2,y2)]} 类别位置限制
```

**关键点：**
- 类别映射：`classes` 即 `data.yaml` 的 `names`。`prompts` 参数可覆盖默认的 YOLO-World 文本提示。
- `conf` 支持 `float`（全局）或 `dict[str, float]`（按类别），如 `{"helmet": 0.3, "person": 0.15}`。
- 输出格式零改动：返回 `{class_id, x1, y1, x2, y2}`，与现有 `BBox`、`save_annotation`、`export_to_yolo` 完全兼容。
- 用 `run_in_executor` 跑同步推理，避免阻塞事件循环。

### 4.3 `api/routes/sam.py`（新路由）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/sam/available` | 检查模型可用性（权重存在 + 试推理） |
| GET | `/sam/download/status` | **【P3】** 权重下载进度 |
| POST | `/sam/download/start` | **【P3】** 发起权重下载 |
| GET | `/sam/config` | 读取配置 |
| POST | `/sam/config` | 更新配置（模型路径/imgsz/device/half） |
| POST | `/sam/validate` | 验证模型有效性 |
| POST | `/sam/auto-label` | 单图预标注（返回 boxes） |
| POST | `/sam/batch/start` | 启动异步批量任务，返回 `batch_id` |
| GET | `/sam/batch/{batch_id}` | 查询进度/结果/质量报告 |
| POST | `/sam/batch/{batch_id}/stop` | 取消任务 |
| GET | `/sam/presets` | **【P2】** 列出所有预设 |
| GET | `/sam/presets/{task_id}` | **【P2】** 读取某任务的所有预设 |
| POST | `/sam/presets/{task_id}` | **【P2】** 保存一个预设 |
| DELETE | `/sam/presets/{task_id}/{preset_name}` | **【P2】** 删除一个预设 |

在 `backend/src/main.py` 中注册 `sam.router`，并在 `on_startup` 中调用 `sam_service.resume_batch_tasks()`。

### 4.4 批量任务管理（【P0 优化】落盘防重启丢失）

```python
# ⚠️ 不再仅用内存 dict
# 真源为 data/batch_tasks/{batch_id}.json
# 内存 self.batch_tasks 仅为运行时缓存

# data/batch_tasks/{batch_id}.json 结构：
{
    "batch_id": "uuid",
    "task_id": "task_001",
    "total": 500,
    "done": 127,
    "failed": 3,
    "failed_items": ["img_042.jpg", "img_088.jpg", "img_123.jpg"],
    "current_image": "img_128.jpg",
    "boxes_written": 531,
    "cancelled": false,
    "status": "running",         // pending | running | done | crashed | cancelled
    "params": {
        "classes": ["helmet", "person"],
        "conf": {"helmet": 0.3, "person": 0.15},
        "use_sam": false,
        "only_unannotated": true
    },
    "quality_report": null,      // 完成时填充
    "created_at": "2026-08-10T16:00:00",
    "updated_at": "2026-08-10T16:05:30"
}
```

**恢复逻辑（`resume_batch_tasks`）：**
- 启动时扫描 `data/batch_tasks/*.json`
- `status="running"` → 标记为 `"crashed"`，用户可选择重新执行或忽略
- `status="pending"` → 恢复执行（`asyncio.create_task`）

- 每张完成后写 `annotations.json` 并标记 `annotated`，前端可刷新列表看到 ✓。
- 进度上报：`GET /sam/batch/{id}` 轮询，或复用 `logs.py` 的 SSE 模式。
- 完成后自动生成质量报告，挂载到 `batch_tasks/{batch_id}.json` 的 `quality_report` 字段。

### 4.5 预设系统（【P2】新增）

用户针对不同场景反复调优的参数组合，一键保存和加载，避免每次重新调整。

```json
// data/presets/helmet-detection.json
[
    {
        "name": "默认快速",
        "use_sam": false,
        "conf": {"helmet": 0.25, "person": 0.25, "vest": 0.25},
        "prompts": {
            "helmet": "safety helmet",
            "person": "construction worker",
            "vest": "reflective safety vest"
        },
        "filters": {}
    },
    {
        "name": "高精度头盔",
        "use_sam": true,
        "conf": {"helmet": 0.35, "person": 0.20, "vest": 0.30},
        "prompts": {
            "helmet": "yellow safety helmet on head, worn by construction worker",
            "person": "person wearing reflective vest and helmet",
            "vest": "orange reflective safety vest on construction worker torso"
        },
        "filters": {
            "min_area": 200,
            "min_aspect": 0.3,
            "max_aspect": 3.0,
            "iou_threshold": 0.85
        }
    }
]
```

### 4.6 质量报告（【P2】新增）

每次批量预标注完成后，自动生成一份质量报告，帮助用户快速判断这批预标注是否可信：

```json
{
    "overview": {
        "total_images": 100,
        "annotated_images": 97,
        "empty_images": 3,
        "total_boxes": 487,
        "avg_boxes_per_image": 5.0,
        "low_confidence_count": 42
    },
    "class_distribution": {
        "helmet": {"count": 201, "avg_per_image": 2.1, "avg_score": 0.72},
        "person": {"count": 186, "avg_per_image": 1.9, "avg_score": 0.65},
        "vest":   {"count": 100, "avg_per_image": 1.0, "avg_score": 0.81}
    },
    "warnings": [
        {"type": "empty_images", "count": 3, "images": ["img_042.jpg", "img_088.jpg", "img_123.jpg"], 
         "suggestion": "这些图片未检出任何目标，建议手动标注或降低 conf 阈值"},
        {"type": "too_many_boxes", "count": 2, "images": ["img_015.jpg(23 boxes)", "img_067.jpg(31 boxes)"],
         "suggestion": "检测框过多，可能存在误检，建议提高 conf 阈值或添加负向提示词"},
        {"type": "class_imbalance", "class": "vest", "count": 100,
         "suggestion": "vest 类检出数量远少于 helmet(201)，检查提示词或降低该类 conf"}
    ],
    "box_size_distribution": {
        "small(<32x32)": 45, "medium(32-128)": 302, "large(>128x128)": 140
    }
}
```

### 4.7 权重下载（【P3】新增）

`/sam/available` 返回中增加 `weights_installed: bool` 字段。若为 `false`，前端展示「下载模型权重」入口：

```
POST /sam/download/start
  → 后台从 ultralytics 官方源下载 SAM 权重到 models/sam/
  → 显示下载进度（bytes/total）

GET /sam/download/status
  → { downloading: true, progress: 0.67, speed: "12.3 MB/s", eta: "45s" }
```

---

## 5. 前端设计

### 5.1 API 封装（`frontend/src/api/annotations.ts`）

```ts
export interface AIBox extends BBox { score?: number }

export const getSamAvailable = () => api.get('/sam/available')
export const downloadSamModel = () => api.post('/sam/download/start')
export const getDownloadProgress = () => api.get('/sam/download/status')
export const autoLabelImage = (taskId, imageId, classes, conf, prompts?, useSam?, onlyUnannotated?) =>
  api.post('/sam/auto-label', { task_id: taskId, image_id: imageId, classes, conf, prompts, use_sam: useSam, only_unannotated: onlyUnannotated })
export const startBatchLabel = (taskId, classes, conf, prompts?, useSam?, onlyUnannotated?) =>
  api.post('/sam/batch/start', { task_id: taskId, classes, conf, prompts, use_sam: useSam, only_unannotated: onlyUnannotated })
export const getBatchProgress = (batchId) => api.get(`/sam/batch/${batchId}`)
export const getPresets = (taskId?) => api.get(taskId ? `/sam/presets/${taskId}` : '/sam/presets')
export const savePreset = (taskId, preset) => api.post(`/sam/presets/${taskId}`, preset)
export const deletePreset = (taskId, presetName) => api.delete(`/sam/presets/${taskId}/${presetName}`)
```

### 5.2 界面交互（`frontend/src/pages/DatasetAnnotate.vue`）

在画布 `controls` 区新增「AI 预标注」按钮（含下拉菜单），在 `annotations-panel` 新增「批量预标注」入口：

```
┌─ AI 预标注 ─────────────────────────────────────┐
│  ┌──────────────────────────────────────┐        │
│  │ 预设：[默认快速 ▼]  [保存当前参数]    │        │
│  ├──────────────────────────────────────┤        │
│  │ ☐ 使用 SAM 精修（更慢但更精确）      │        │
│  │ ☐ 仅标注未标注的图片                 │        │
│  ├──────────────────────────────────────┤        │
│  │ helmet  阈值 [0.25 ──●── 1.0]       │        │
│  │ 提示词: [safety helmet         ✏️]  │        │
│  │ person  阈值 [0.15 ──●── 1.0]       │        │
│  │ 提示词: [construction worker   ✏️]  │        │
│  ├──────────────────────────────────────┤        │
│  │ 后处理：                             │        │
│  │ ☐ 过滤过小框（< 200px²）            │        │
│  │ ☐ 过滤异常宽高比                     │        │
│  │ ☐ NMS 去重（IoU > 0.85）            │        │
│  └──────────────────────────────────────┘        │
│                                                  │
│      [预览 3 张]    [当前图片预标注]               │
│      [整集批量预标注]                              │
└──────────────────────────────────────────────────┘
```

**交互要点：**

- **预设系统**：顶部下拉选择已保存的参数组合，一键加载。调好参数后点「保存当前参数」存为新预设。
- **按类别阈值**：每类独立滑动条，默认值来自 `data.yaml` 或预设。
- **提示词编辑**：每类旁边有编辑图标，展开可配置 YOLO-World 文本提示词。支持正向/负向描述。
- **后处理开关**：三项快速过滤规则（尺寸/宽高比/NMS），打勾即生效。
- **3 张预览**：先对前 3 张未标注图片跑预标注并展示结果，满意再批量。
- **预标注结果**用橙色框显示，人工标注为绿色，便于区分审查。

### 5.3 批量预标注进度页

```
┌─ 批量预标注进度 ─────────────────────────────────┐
│  进度：████████████░░░░░░  67% (335/500)          │
│  已写入框数：1,423   |   失败：2 张               │
│  当前处理：img_336.jpg                            │
│  ETA：约 2 分 30 秒                               │
│                                                   │
│  [取消任务]                                       │
└──────────────────────────────────────────────────┘
```

完成后自动跳转质量报告：

```
┌─ 预标注质量报告 ─────────────────────────────────┐
│  ✅ 预标注完成！共标注 497/500 张图片            │
│                                                   │
│  📊 检出统计                                     │
│  helmet:  201 框 (avg 2.1/图, score 0.72)        │
│  person:  186 框 (avg 1.9/图, score 0.65)        │
│  vest:    100 框 (avg 1.0/图, score 0.81)        │
│                                                   │
│  ⚠️ 3 张图片未检出任何目标                       │
│     img_042.jpg, img_088.jpg, img_123.jpg         │
│     [跳转手动标注]                                │
│                                                   │
│  💡 建议：                                       │
│  · vest 类检出较少，尝试降低阈值或调整提示词      │
│  · 2 张图片框数异常多，可能存在误检               │
│                                                   │
│  [导出 YOLO 格式]    [返回标注页继续人工审核]     │
└──────────────────────────────────────────────────┘
```

### 5.4 预标注反馈与调节面板（【新增】）

当用户对预标注结果不满意时，提供快速调节入口：

```
┌─ 预标注反馈 ────────────────────────────────────┐
│  本次预标注效果如何？                            │
│  ○ 满意    ○ 一般    ● 需要调整                  │
│                                                  │
│  常见问题（可多选）：                            │
│  ☑ 漏检太多（目标没被框到）                      │
│  □ 误检太多（不该框的框了）                      │
│  □ 框的位置不准                                  │
│  □ 类别分错了                                    │
│  □ 框太大/太小                                   │
│                                                  │
│  当前参数：conf=0.25, prompt="safety helmet"     │
│                                                  │
│  系统建议：                                      │
│  💡 降低 helmet 的 conf 至 0.15 可减少漏检       │
│  💡 尝试提示词 "safety helmet on construction    │
│     worker head" 可缩小语义范围减少误检           │
│                                                  │
│  [应用建议参数重新生成]  [手动调整参数]           │
└──────────────────────────────────────────────────┘
```

---

## 6. 数据流

```
前端点「AI 预标注」
   │  POST /sam/auto-label {task_id, image_id, classes, conf, prompts, use_sam}
   ▼
backend: sam_service.auto_label
   │  定位图片绝对路径
   │  YOLO-World(prompts) → boxes                       （文本→检测）
   │  可选：SAM(all_boxes) → masks → 最小外接矩形        （批量精修）
   │  _post_filter(boxes, filters)                       （后处理过滤）
   ▼
返回 [{class_id, x1, y1, x2, y2, score}]
   │
前端合并进 currentBoxes → 画布橙色框 → 人工微调
   │                 ↓ 修改后点保存
   └── save_annotation → annotations.json（结构不变）→ export_to_yolo
```

### 6.1 与现有代码的集成点（零破坏）

| 现有文件 | 集成方式 |
|---------|---------|
| `backend/src/api/routes/annotations.py` | 输出 `BBox` 结构完全一致，新增 `sam.py` 路由 |
| `backend/src/services/annotation_service.py` | 复用 `save_annotation` 写 `annotations.json` |
| `backend/src/main.py` | 注册 `sam.router` + `on_startup` 恢复批量任务 |
| `frontend/src/api/annotations.ts` | 追加 SAM 相关 API 函数 |
| `frontend/src/pages/DatasetAnnotate.vue` | 追加按钮、预设面板、反馈面板与状态逻辑 |

---

## 7. 预标注质量闭环（【新增】长期迭代策略）

预标注不是一次性操作，而是持续改进的正循环。YOLO-World 作为通用模型无法针对特定场景优化，真正的质量提升来自**用人工修正数据训练自定义模型**。

### 7.1 三阶段迭代飞轮

```
第一阶段（冷启动）
  YOLO-World 预标注 → 人工修正 → 存入训练集
         │
         ▼
第二阶段（有基础数据后）  
  用修正后的数据训练自定义 YOLOv8 → 自定义模型预标注 → 修正量大幅减少
         │
         ▼
第三阶段（数据积累后）
  自定义模型替代 YOLO-World → 预标注效果远超通用模型 → 几乎不用修正
```

### 7.2 短期调节手段（不改模型权重）

| 调节方式 | 耗时 | 效果 | 适用场景 |
|:--|:--:|------|------|
| 调整置信度阈值（按类别） | 10 秒 | 减少误检/漏检 | 某类置信度明显异常 |
| 修改文本提示词 | 1 分钟 | 提升检测准确性 | 目标被漏检或分类错误 |
| 启用/禁用 SAM 精修 | 一键 | 改善框的贴合度 | 边界模糊或重叠目标 |
| 调整后处理过滤规则 | 3 分钟 | 清理系统性噪声 | 大量假框或位置异常 |

### 7.3 提示词调优策略

YOLO-World 以文本驱动检测，提示词是调节检测效果的最强手段：

```json
// 示例：安全帽检测的提示词调优
{
    "helmet": {
        "prompt": "yellow safety helmet on construction worker head",
        "negative_prompt": "motorcycle helmet, bicycle helmet, hard hat on shelf",
        "notes": "加上场景+颜色缩小语义范围，排除摩托车头盔和货架上的帽子"
    },
    "no_helmet": {
        "prompt": "person head without helmet, bare head of worker",
        "notes": "负向描述——没有戴安全帽的人头"
    }
}
```

---

## 8. 分阶段实施

| 阶段 | 内容 | 验收标准 | 预估工时 |
|:--:|------|------|:--:|
| S1 · 基座接入 | `sam_service.py` 加载/缓存 + `device_manager` + `/sam/available`、`/sam/config`、`/sam/validate` + 权重下载接口 | 权重可加载、GPU 探测正常、可用性接口返回正确 | 2天 |
| S2 · 单图预标注（默认模式） | YOLO-World 检测 + 后处理过滤 + `/sam/auto-label` + 前端「AI 预标注」按钮（含按类别 conf + 提示词） | 单图跑通，橙色框可编辑可保存 | 2天 |
| S3 · 批量预标注 | 异步落盘 `/sam/batch/*` + `only_unannotated` + 前端进度条 + 恢复机制 + 质量报告 | 整集标注完成，重启不丢进度，质量报告正确 | 2天 |
| S4 · SAM 精修增强 | `use_sam=True` 批量推理 + 前端开关 + 3 张预览对比 | 默认/SAM 模式可切换，效果对比可见 | 1天 |
| S5 · 预设系统 | 预设 CRUD + 前端保存/加载/切换 | 可一键切换参数组合 | 1天 |
| S6 · 反馈调节面板 | 满意度反馈 + 常见问题标记 + 系统建议 + 快速重新生成 | 用户可快速反馈并调节参数 | 1天 |
| **总计** | | | **约 9 天** |

每阶段独立可上线，S1-S3 为最小闭环（约 6 天）。

---

## 9. 风险与对策

| 风险 | 对策 |
|------|------|
| SAM 权重 ~2.4GB，首次加载慢 | 单例缓存常驻；`/sam/available` 预加载；加载期间显示进度；`/sam/download` 提供下载入口 |
| 推理吃显存 | `device_manager` 选空闲 GPU；`SAM_HALF=True` 半精度；`imgsz` 可调；默认不启用 SAM 精修 |
| 批量任务阻塞事件循环 | `run_in_executor` 后台执行 + 每张写盘，可随时 `stop` |
| **后端重启导致批量任务丢失** | **【P0 已解决】任务状态落盘至 `data/batch_tasks/`，启动时 `resume_batch_tasks()` 恢复或标记 crashed** |
| 预标注误报/漏检 | 按类别 conf 阈值；提示词调优；后处理过滤规则；结果用独立颜色仅作辅助，人工确认后保存；长期用修正数据训练自定义模型替代 YOLO-World |
| YOLO-World 类别名与 `data.yaml` 不一致 | 以 `task.json["classes"]` 为唯一来源，`prompts` 参数提供自定义映射 |
| **预标注覆盖人工标注成果** | **【P1 已解决】`only_unannotated=True` 默认跳过已有标注的图片** |

---

## 10. 待确认事项

- [ ] SAM / YOLO-World 权重的下载与放置目录（`backend/models/sam`）— **已设计 download 接口**
- [ ] 批量任务进度上报方式：轮询（`GET /sam/batch/{id}`）还是 SSE
- [ ] 是否同时支持纯 SAM 交互模式（点/框提示）作为增强
- [ ] 预标注结果「追加」与「全覆盖」的默认策略 — **默认追加，可选全覆盖**
- [ ] 新：质量报告中"低置信度"的阈值定义（建议 < 0.3）
- [ ] 新：反馈面板中"系统建议"规则的完善程度（初版可先不做自动建议，只收集反馈）

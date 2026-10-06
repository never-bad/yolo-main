# 问题与 Bug 跟踪

> 定位：集中登记开发过程中发现的所有问题与 Bug，一条一号，**只增不删**；修复后移入 §5 归档，编号保留（其他文档可能引用）。
> 编号规则（按分类前缀）：
> - `B-xx` 功能缺陷（Bug）｜`P-xx` 性能/健壮性（Performance）｜`C-xx` 兼容性（Compatibility）｜`S-xx` 安全（Security）｜`D-xx` 文档与口径漂移（Doc drift）｜`R-xx` 已修复归档（Resolved）
> 紧急程度：🔴 紧急（近期部署/交付前必须处理）｜🟡 中（本迭代内处理）｜🟢 低（待处理，后期统一修正）
> 状态：`待拍板` = 需用户决策后才能动工；`待处理` = 已确认但排期靠后；`已缓解` = 有临时对策；`已修复` = 进 §5
> 交叉引用：改进项 `I-xx` 查《功能实现与改进点跟踪》；文件位置查《项目文件功能说明》；页面/接口行为查《平台功能与页面关联》。
> 更新时间：2026-09-30（B-05 部分修复登记）

---

## 1. 总览（按紧急程度排序）

| 编号 | 分类 | 紧急 | 一句话描述 | 状态 | 关联 |
|---|---|---|---|---|---|
| I-01* | 功能缺陷 | 🔴 | shadow.py 等 4 个功能必需文件未入 git，clone 后跑不起来 | 待处理 | 文件说明 §5.4 |
| S-01 | 安全 | 🔴 | 模型上传 extractall 无路径遍历防护 | 待处理 | I-02 |
| S-02 | 安全 | 🔴 | YOLO 标签导入 extractall 无 member 校验 | 待处理 | I-02 |
| B-05 | 功能缺陷 | 🔴 | CI/bat 引用不存在的 docker/{backend,frontend}/Dockerfile | 待处理 | 文件说明 §4 |
| S-03 | 安全 | 🟡 | 全平台无鉴权 + CORS `*` + 20 GB 上传上限 | **待拍板** | X-04 |
| B-02 | 功能缺陷 | 🟡 | 清除标注后 PG `has_label` 虚高（有意不回填） | **待拍板** | I-04 |
| B-03 | 功能缺陷 | 🟡 | 全量重灌会冲掉 has_label，必须二步入库流程 | 已缓解 | 改进跟踪 §2.3 |
| B-04 | 功能缺陷 | 🟡 | weight_mappings 有表无 DAO，恒为空 | **待拍板** | I-06 |
| P-01 | 性能/体验 | 🟡 | goAnnotate 用 window.location.href，整页刷新丢 SPA 状态 | 待处理 | — |
| P-02 | 性能 | 🟡 | 超大 task.json（13k+ items）预标注 O(N²) 重复全量读 | 需验证 | — |
| D-01 | 文档漂移 | 🟡 | 平台细节说明 §2.4「无多边形」已过期（polygon 已实现） | 待处理 | I-12 |
| D-02 | 口径漂移 | 🟡 | `/datasets?focus=` 传了没人消费 | 待处理 | — |
| D-03 | 口径漂移 | 🟡 | `_` 前缀归档目录：PG 有行、列表页不可见 | 待处理 | — |
| D-04 | 文档漂移 | 🟡 | API_DOCUMENTATION「推理服务」章节无对应路由 | 待处理 | X-08 |
| D-05 | 文档漂移 | 🟢 | 平台细节说明「满 1500 封板」≠ 代码 `SEAL_MIN_IMAGES=5000`；**09-30 复审：代码注释亦有 4 处过期写 1500**（dataset_service L2601/L2828/L2928 + main.py L18 docstring，后者还漏列 GPU 熔断任务） | 待处理 | I-12 |
| D-06 | 文档漂移 | 🟢 | 文档称状态"一套口径"，实际 `USE_UNIFIED_STATUS=False` | 待处理 | I-03 |
| P-03 | 健壮性 | 🟢 | 损坏 zip（如 1.15 MB 的 person.zip）可入库，训练时才炸 | 待处理 | — |
| P-04 | 体验 | 🟢 | axios 无统一错误拦截，各调用点各自 try/catch | 待处理 | I-08 |
| B-06 | 死代码 | 🟢 | components/Logs/* 预留未接入任何页面 | 待处理 | I-07 |
| B-07 | 风险约定 | 🟢 | models.py 通配路由 `/{model_id}` 顺序陷阱（新增路由易被吞） | 约定防护 | 页面关联 §6 |
| I-11* | 悬空目录 | 🟢 | `data/inference_results/` 无对应路由/服务 | 待确认 | I-11 |

> `*` 两项以改进项为主编号（`I-01`、`I-11`），此处仅登记存在性，避免双号漂移。

---

## 2. 🔴 紧急

### I-01 / B-01　功能必需文件未入版本控制 —— 功能缺陷

- **现象**：以下文件在磁盘上被大量 import，但 `git ls-files` 不跟踪（`git clone` 后平台直接起不来）：
  - `backend/src/db/shadow.py` —— 44 处写点 `await shadow.*` 依赖
  - `backend/src/utils/class_labels.py` —— `live_classes` / `init_labels_from_classes`
  - `backend/scripts/backfill_has_label.py` —— 部署固定第二步
  - `frontend/src/components/LibraryCreateModal.vue` —— Dashboard / DatasetUpload / Models 三页 import
- **根因**：`.gitignore` 未排除它们，纯粹是新增后未 `git add`。
- **处置**：`git add` 上述 4 文件并提交。**这是四份文档所述一切工作的交付前提**。
- 详见《项目文件功能说明》§5.4。

### S-01　模型上传解压无路径遍历防护 —— 安全

- **位置**：[model_service.py](file:///c:/Users/41653/Desktop/yolo-main/yolo-main/backend/src/services/model_service.py) L1516-L1540：`zip_file.testzip()` 后直接 `extractall(temp_path)`。
- **风险**：恶意 zip 成员名含 `../../` 或绝对路径时可写出目标目录之外。
- **对照**：`dataset_service._extract_zip`（L119）**有**防护（`member_path.is_absolute() or '..' in str(...)` 跳过 + `_ensure_long_path`），两处标准不一致。
- **处置**：复用 dataset_service 的 member 校验写法（改进项 I-02）。上传源若全部可信可降为 🟡，但嵌入他人平台后上传方不可控，按 🔴 处理。

### S-02　YOLO 标签导入解压无 member 校验 —— 安全

- **位置**：[annotation_service.py](file:///c:/Users/41653/Desktop/yolo-main/yolo-main/backend/src/services/annotation_service.py) L1461：`zf.extractall(tmp_dir)`，无任何成员名校验。
- **风险 / 处置**：同 S-01，一并修（I-02）。

### B-05　CI 与构建脚本引用不存在的 Dockerfile —— 功能缺陷

- **现象**：`.cnb.yml` L11/L19 引用 `docker/backend/Dockerfile`、`docker/frontend/Dockerfile`；`build-unified.bat` 同样引用。这两个目录实际为**空**。
- **影响**：CNB 流水线一旦触发必失败。现行有效路径是 `docker/Dockerfile` + `docker/Dockerfile.env`（见 `docker/build_push.sh` 头部注释）。
- **处置**：`.cnb.yml` 改指向新路径，或整份停用；`build-unified.bat` 删除、统一用 `build_push.sh`（《项目文件功能说明》§5.3）。
- **进展（09-30 核对）**：`build-unified.bat` 与 `docker/{backend,frontend}` 空目录已删除；**`.cnb.yml` 仍引用旧路径，待改**。

---

## 3. 🟡 中

### S-03　无鉴权 + CORS 全开 —— 安全（待拍板）

- **现状**：后端所有端点无 token/session 校验；`main.py` CORS `allow_origins=["*"]`；nginx `client_max_body_size 20480m`。
- **判断**：若平台**嵌入他人平台**（既定方向），鉴权与来源限制应由宿主网关承担，当前实现可接受；若**独立公网部署**则必须补。
- **待拍板**：确认部署形态后再决定加 API 前缀鉴权中间件还是维持现状（黑名单 X-04 当前禁止自建用户体系）。

### B-02　清除标注不回填 PG，`has_label` 虚高 —— 功能缺陷（待拍板）

- **现象**：`clear_ai_annotations`（L1013）与 `clear_all_annotations`（L1054）的 docstring 明写"不回填 PG 影子库（一图一 JSON 才是唯一权威）"。清标后 JSON 真值已空，但 PG `task_items.has_label` 仍为 true，按 PG 查询的计数/筛选会偏大。
- **两个选项**：A 接受虚高（真值唯一原则，PG 只做加速）；B 在 `shadow.py` 加反向 `mark_items_unlabeled` 并接线（需处理"清标后重标"的顺序覆盖）。
- 关联改进项 I-04。

### B-03　全量重灌冲掉 `has_label` —— 功能缺陷（流程风险，已缓解）

- **机理**：task.json items 只带 `annotated` 不带 `has_label`；`migrate_to_pg.py` 全量重灌 tasks 域时整表替换 → `has_label` 变 NULL。
- **缓解**：`migrate_to_pg.py` 文档头已写警告；部署流程固定第二步跑 `backfill_has_label.py`；回归 T4 用例（合成任务 + `with_items=True` 反向对照）守住收口层不再误用。
- **残留风险**：人工只跑重灌忘了回填 → 列表页"有标注"筛选失效。处置：把二步并入 `deploy.sh`（I-05）。

### B-04　`weight_mappings` 有表无 DAO —— 功能缺陷（待拍板）

- **现状**：`db/models.py` L150 定义了表，`meta_store` 无对应读写函数，`shadow.py` 无接线，`migrate_to_pg.py` 不灌它 → 恒为空。
- **待拍板**：A 留给外部平台的接口位（补一行注释说明用途）；B 漏实现（补 DAO）；C 删表。关联 I-06。

### P-01　`goAnnotate` 整页刷新 —— 性能/体验

- **位置**：DatasetUpload.vue L560-L562：`window.location.href = '/annotate?dataset_id=' + id`，而非 `router.push`。
- **影响**：离开 SPA 路由，白屏重载、前端状态全丢；与站内其他跳转（全部 `router.push`）风格不一致。
- **处置**：改为 `router.push('/annotate?dataset_id='+id)`。一行改动、低风险，属"不需要复杂代码"的改进，允许做。

### P-02　超大 task.json 预标注 O(N²) —— 性能（需验证）

- **背景**：历史教训（见项目记忆）：13000+ items 的任务在预标注时反复全量加载/写回 task.json，耗时随任务规模平方增长。
- **现状**：收口层已避免 items 进 PG extra，但**服务侧逐图保存是否仍全量重写 task.json 未实测**。
- **处置**：用一个 ≥10k items 的任务实测预标注耗时；确认瓶颈后再决定是否做增量写。在确认前不要按"已修复"对待。

### D-01　`平台细节说明.md` §2.4 标注实体描述过期 —— 文档漂移

- **文档说法**：「只有矩形框 BBox，无多边形/掩码」。**代码实际**：`polygon` 轮廓已实现（`annotation_service`、`DatasetAnnotate.vue`、`annotation_importers` 等 8 个文件命中）。
- **处置**：改文档（I-12）。

### D-02　`/datasets?focus=` 参数无人消费 —— 口径漂移

- **现象**：ModelDetail.vue L185 跳 `/datasets?focus={id}`，但 DatasetUpload.vue **没有 `useRoute()`**（只有 `onMounted` L975），`focus` 参数被完全忽略——用户跳过去后看不到"被聚焦的数据集"。
- **处置**：DatasetUpload 加 `useRoute` 读取 `query.focus` 并滚动/高亮对应卡片。小改动，允许做。

### D-03　归档数据集在 PG 可见、列表不可见 —— 口径漂移

- **机理**：`list_datasets`（dataset_service L1693）跳过 `_` 前缀目录；但 `migrate_to_pg.py` 与 `shadow.py` 的 glob **不跳过** → PG 里存在 `_archived_ds_v1` 等行（含 `ds_v1` 这种文件内 id 兜底出来的行，属口径而非 bug）。
- **影响**：直接查 PG 的下游（含未来嵌入方）会看到列表页不存在的"幽灵数据集"。
- **处置**：统一口径——migrate/shadow 同样跳过 `_` 前缀，或给 datasets 表加 `archived` 列并在文档声明。倾向后者（改动小、信息不丢）。

### D-04　API_DOCUMENTATION「推理服务」章节无对应实现 —— 文档漂移

- **处置**：在章节头标注"未实现/由宿主平台提供"（当前阶段禁止补推理代码，见 X-08）。

---

## 4. 🟢 低（待处理，后期统一修正）

| 编号 | 分类 | 描述 | 处置备忘 |
|---|---|---|---|
| D-05 | 文档 | 平台细节说明多处「满 1500 自动封板」，实际 `SEAL_MIN_IMAGES=5000`（README 为 5000；09-30 复审发现代码注释另有 4 处过期写 1500：dataset_service L2601/L2828/L2928 + main.py L18） | 改文档 + 注释改为引用 `settings.SEAL_MIN_IMAGES`，随 I-12 一起做 |
| D-06 | 文档 | 文档称前后端状态"一套口径"，实际 `USE_UNIFIED_STATUS=False`，读侧仍走旧列 | 等 I-03 灰度决策后一并修订 |
| P-03 | 健壮性 | 损坏/超小 zip（教训：1.15 MB 的 person.zip）上传即入库，训练阶段才暴露无效图片 | 上传/prepare 时加"可解码抽样 + 包体大小下限"校验 |
| P-04 | 体验 | `axios.ts` 无响应拦截器，错误处理散落在 6 个页面各自 try/catch + alert | 加统一拦截 + toast（I-08） |
| B-06 | 死代码 | `components/Logs/LogPanel.vue`、`LogToolbar.vue` 不被任何页面引用（Train 自绘日志区） | 接入或删除（I-07） |
| B-07 | 风险约定 | `/models` 路由：`/{model_id}` 通配注册在前会吞静态段。现有顺序正确，但**新增静态路径必须插在通配之前** | 已在《平台功能与页面关联》§6 留警示，代码加注释即可 |
| I-11 | 悬空 | `backend/data/inference_results/` 目录存在但无路由/服务写入方 | 确认历史用途后清理或补文档 |
| — | 悬空 | `docker/backend/`、`docker/frontend/` 空目录（B-05 的诱因） | 随 B-05 一并删 |

---

## 5. 已修复归档（`R-xx`，2026-09 本轮）

| 编号 | 问题 | 修复 | 验证 |
|---|---|---|---|
| R-01 | Postgres 缺 `unified_status` 列 → **38 天影子库静默写失败** | `engine.py` 改为按 ORM 模型自动推导补列（SQLAlchemy `create_all` 不会给既有表加列） | S1~S9 共 50 项 PASS |
| R-02 | #3 封板开关不生效、#6 标签不同步 | `dataset_service` / `class_labels` 对应修复 | `test_fix_3_6.py` 34/34 PASS |
| R-03 | 影子库写点散落、跨进程写（train_script 子进程直改 meta.json）不同步 | 新增 `src/db/shadow.py` 收口层（`_guard` 吞异常降级 + 落盘后回读真值 + mtime 水位自愈），五域 44 处接线（dataset 23 / model 16 / annotation 3 / train 2） | `test_shadow_writethrough.py` 8 组 36 项 FAILS=0 |
| R-04 | `task_items.has_label` 存量全为 false/NULL（从未回填） | 新增 `scripts/backfill_has_label.py`（判据与 `get_task_items` 逐字一致：`bool(ann.get("boxes"))`，逐图 JSON 优先、annotations.json 兜底；幂等，只 false→true） | 回填 95,934 / 174,944 行；抽样 6 任务双向比对零差异；`annotated − has_label = 3214` 为背景样本，语义正确 |
| R-05 | `migrate_to_pg.py` 的 `--prune` 实现被外部回退（文档头留着说明、函数丢失） | 重新补入 `_truth_ids()` + `prune_orphans(dry)` + main 参数接线 | `py_compile` + `--dry --prune-only` EXIT=0；prune 清掉 6 条游离（2 datasets + 2 models + 2 游离 task_items），纠偏后三域与真值等集（23/12/179） |
| R-06 | 38 个任务 `has_label` 零命中疑似异常 | 归因：其 `dataset_id` 指向的目录已不存在（17 个目录）或无 annotations 子目录 → 真值无标注，PG 为 false 正确 | **非 bug**，结案 |
| R-07 | 回归 T4 依赖真实数据抽样，prune 后无样本导致 FAIL | T4 重写为确定性用例：tempfile 假真值 + `zz_` 合成任务 + `with_items=True` 反向对照 | FAILS=0 |

---

## 6. 分类统计与处理原则

| 分类 | 🔴 | 🟡 | 🟢 | 已归档 |
|---|---|---|---|---|
| B 功能缺陷 | 2（B-01/B-05） | 3 | 2 | 3 |
| P 性能/健壮性 | 0 | 2 | 2 | 0 |
| S 安全 | 2 | 1（待拍板） | 0 | 0 |
| D 文档/口径漂移 | 0 | 4 | 2 | 0 |
| C 兼容性 | 0 | 0 | 0 | 1（C-01，见下） |

**C-01（归档）**：`data.yaml` 曾用绝对路径导致跨平台（Windows 开发 / Linux 容器）训练失败 → 已定硬约束：**一律相对路径 `.`**。后续任何改 `seal_dataset`/`prepare` 的任务不得回退。

处理原则（与《功能实现与改进点跟踪》§3 黑名单一致）：
1. 🔴 全部属于"补防护 / 入 git / 修脚本引用"，**不需要新框架**，当前阶段就该做；
2. 待拍板三项（S-03、B-02、B-04）**未决策前不动代码**；
3. 🟢 项攒批处理，修一个带一个，不单独排期；
4. 任何修复必须附回归脚本且**保留脚本**（§5 的 test_* 是资产不是垃圾）。

# 问题与 Bug 跟踪

> 定位：集中登记开发过程中发现的所有问题与 Bug，一条一号，**只增不删**；修复后移入 §5 归档，编号保留（其他文档可能引用）。
> 编号规则（按分类前缀）：
> - `B-xx` 功能缺陷（Bug）｜`P-xx` 性能/健壮性（Performance）｜`C-xx` 兼容性（Compatibility）｜`S-xx` 安全（Security）｜`D-xx` 文档与口径漂移（Doc drift）｜`R-xx` 已修复归档（Resolved）
> 紧急程度：🔴 紧急（近期部署/交付前必须处理）｜🟡 中（本迭代内处理）｜🟢 低（待处理，后期统一修正）
> 状态：`待拍板` = 需用户决策后才能动工；`待处理` = 已确认但排期靠后；`已缓解` = 有临时对策；`已修复` = 进 §5
> 交叉引用：改进项 `I-xx` 查《功能实现与改进点跟踪》；文件位置查《项目文件功能说明》；页面/接口行为查《平台功能与页面关联》。
> 更新时间：2026-10-14（**首次实际部署验证**：真机抓到 N 批次回归 **B-08**——JSON IO 收口漏改 29 处函数对象引用致 `seal_poll` NameError，`a48a6fc` 已修复 + 重新部署验证通过；本批次新增 B-08 一条）
> 上次更新：2026-10-13（批次 L 后置：P-02 实测确认 O(N²) 风险，预留改进项 I-13；I-10 同步关闭——`7f757a8` 已落地 `lifecycle.py`，文档漏更本批次补齐）
> 更早更新：2026-10-10（批次 I 前置：P-01/D-01/D-02/D-04/D-05/D-06/B-07 经核实已于 10-06 由 905a535/305756e 修复，状态更正；I-11 结案）

---

## 1. 总览（按紧急程度排序）

| 编号 | 分类 | 紧急 | 一句话描述 | 状态 | 关联 |
|---|---|---|---|---|---|
| I-01* | 功能缺陷 | 🔴 | shadow.py 等 4 个功能必需文件未入 git，clone 后跑不起来 | 已解决（10-10 核实：4 文件全部 TRACKED，未跟踪文件归零） | 文件说明 §5.4 |
| S-01 | 安全 | 🔴 | 模型上传 extractall 无路径遍历防护 | 已修复（d0a81fb，zip_guard 收口） | I-02 |
| S-02 | 安全 | 🔴 | YOLO 标签导入 extractall 无 member 校验 | 已修复（ffe2c79，zip_guard 收口） | I-02 |
| B-05 | 功能缺陷 | 🔴 | CI/bat 引用不存在的 docker/{backend,frontend}/Dockerfile | 已解决（10-10，de6850e 批次 G-1 删 .cnb.yml） | 文件说明 §4 |
| B-08 | 功能缺陷 | 🔴 | N 批次 JSON IO 收口漏改 29 处**函数对象引用**（`_load_json`/`_save_json`），运行时 NameError，封板兜底循环每轮失败 | 已修复（2026-10-14 `a48a6fc`，首次实际部署验证真机抓到） | 批次 N · I-13 |
| S-03 | 安全 | 🟡 | 全平台无鉴权 + CORS `*` + 20 GB 上传上限 | 已拍板（10-10）：嵌入宿主，网关鉴权 | X-04 |
| B-02 | 功能缺陷 | 🟡 | 清除标注后 PG `has_label` 虚高（有意不回填） | 已拍板（10-10）：接受虚高，重标覆盖 | I-04 |
| B-03 | 功能缺陷 | 🟡 | 全量重灌会冲掉 has_label，必须二步入库流程 | 已缓解 | 改进跟踪 §2.3 |
| B-04 | 功能缺陷 | 🟡 | weight_mappings 有表无 DAO，恒为空 | 已解决（10-10，bbd3d6d 批次 G-3 删表） | I-06 |
| P-01 | 性能/体验 | 🟡 | goAnnotate 用 window.location.href，整页刷新丢 SPA 状态 | 已解决（10-06 305756e：goAnnotate 改 router.push，实测 DatasetUpload L765） | — |
| P-02 | 性能 | 🟡 | 超大 task.json（13k+ items）预标注 O(N²) 重复全量读 | 已确认（预留） | I-13 |
| D-01 | 文档漂移 | 🟡 | 平台细节说明 §2.4「无多边形」已过期（polygon 已实现） | 已解决（10-06 905a535：平台细节说明 polygon 口径同步；该文件未入 git，见 §4 风险登记） | I-12 |
| D-02 | 口径漂移 | 🟡 | `/datasets?focus=` 传了没人消费 | 已解决（10-06 305756e：applyFocus/consumeFocus 消费 ?focus=，含 keep-alive onActivated 场景） | — |
| D-03 | 口径漂移 | 🟡 | `_` 前缀归档目录：PG 有行、列表页不可见 | 待处理 | — |
| D-04 | 文档漂移 | 🟡 | API_DOCUMENTATION「推理服务」章节无对应路由 | 已解决（10-06 905a535：API_DOCUMENTATION L1028 推理章节加未实现警示） | X-08 |
| D-05 | 文档漂移 | 🟢 | 平台细节说明「满 1500 封板」≠ 代码 `SEAL_MIN_IMAGES=5000`；**09-30 复审：代码注释亦有 4 处过期写 1500**（dataset_service L2601/L2828/L2928 + main.py L18 docstring，后者还漏列 GPU 熔断任务） | 已解决（10-06 905a535：后端注释改引用 settings.SEAL_MIN_IMAGES，全后端零 1500） | I-12 |
| D-06 | 文档漂移 | 🟢 | 文档称状态"一套口径"，实际 `USE_UNIFIED_STATUS=False` | 已解决（10-06 905a535：平台细节说明 L54 声明口径现状；I-03 10-13 拍板=选项 B PG 影子库优先+1-2min 窗口容差，正式收口） | I-03 |
| P-03 | 健壮性 | 🟢 | 损坏 zip（如 1.15 MB 的 person.zip）可入库，训练时才炸 | 待处理 | — |
| P-04 | 体验 | 🟢 | axios 无统一错误拦截，各调用点各自 try/catch | 已解决（10-11 批次 J：全局 toast + axios 拦截器 + unhandledrejection 兜底 + 143 alert/11 confirm 全量迁移） | I-08 |
| B-06 | 死代码 | 🟢 | components/Logs/* 预留未接入任何页面 | 已解决（10-10 批次 I：删除未接入的预留组件） | I-07 |
| B-07 | 风险约定 | 🟢 | models.py 通配路由 `/{model_id}` 顺序陷阱（新增路由易被吞） | 已解决（10-06 905a535：routes/models.py 通配路由前加顺序警示注释） | 页面关联 §6 |
| B-09 | 功能缺陷 | 🔴 | 封板 data.yaml 写死源目录绝对 path，改名/删源/换机器后训练报 images not found，需手动改 yaml | 已修复（10-06 b365934） | 安全升级文档 |
| B-10 | 体验 | 🟡 | 每次切页（首页/数据集/模型）组件销毁重建，整页重新加载 | 已修复（10-06 d99ff5b） | — |
| B-11 | 功能缺陷 | 🔴 | GroundingDINO 恒报 `Invalid device string: '0'`，多引擎检测里 GD 一路从未生效（auto 分支返回裸索引） | 已修复（2026-10-27，含 `get_best_device` 返回 `cuda:N` + 坏实例不缓存） | B-08 · I-15 |
| B-12 | 功能缺陷 | 🔴 | 批量预标注一开就整批失败「批量预标注出错: Invalid device string: '0'」，同一配置下单图标注却正常出框（预加载不隔离单引擎失败） | 已修复（2026-10-27，逐引擎隔离预加载 + 换引擎兜底真正生效 + 全失败不再假报完成） | B-11 · I-16 |
| B-13 | 模型幻觉 | 🔴 | 千问把红安全帽/人头等近似物误标成「自救器」（画面中根本没有目标也出框）——VLM 整图检测固有幻觉，score 还是硬编码 0.9 无真置信度 | 已缓解（2026-10-27：提示词收紧 + 框级裁剪复问校验默认开，见 I-17；根治靠标签字典写视觉描述） | I-15 · I-17 |
| B-14 | 功能缺陷 | 🔴 | **训练启动后一个 epoch 都没跑就被熔断中断**：ultralytics AMP 自检会联网下载 yolo11n.pt（5.4MB），弱网（10~30KB/s + TLS 反复重置）卡住十几分钟，GPU 全程 0% 被 `gpu_watch` 判为"GPU 卡死"误杀（**不是 OOM**，见 B-15 排障记录）。真机 job_20261028_011142 / 012750 均因此中断，续训后原样再卡一次 | 已修复（2026-10-28：AMP 自检改离线判定 + 熔断启动宽限期 900s + 续训清理失败残留） | B-15 · I-18 |
| I-11* | 悬空目录 | 🟢 | `data/inference_results/` 无对应路由/服务 | 已结案（10-10：目录空且无代码引用，随批次 I 删除） | I-11 |

> `*` 两项以改进项为主编号（`I-01`、`I-11`），此处仅登记存在性，避免双号漂移。

---

## 2. 🔴 紧急

### I-01 / B-01　功能必需文件未入版本控制 —— 功能缺陷（已解决 ✅ 2026-10-10 核实）

- **现象**：以下文件在磁盘上被大量 import，但 `git ls-files` 不跟踪（`git clone` 后平台直接起不来）：
  - `backend/src/db/shadow.py` —— 44 处写点 `await shadow.*` 依赖
  - `backend/src/utils/class_labels.py` —— `live_classes` / `init_labels_from_classes`
  - `backend/scripts/backfill_has_label.py` —— 部署固定第二步
  - `frontend/src/components/LibraryCreateModal.vue` —— Dashboard / DatasetUpload / Models 三页 import
- **根因**：`.gitignore` 未排除它们，纯粹是新增后未 `git add`。
- **处置**：`git add` 上述 4 文件并提交。**这是四份文档所述一切工作的交付前提**。
- **结案（2026-10-10 复核）**：4 文件全部 `git ls-files --error-unmatch` 命中（TRACKED）；`git ls-files --others --exclude-standard` = **0**（工作区已无未跟踪文件）。本项关闭。
- 详见《项目文件功能说明》§5.4。

### S-01　模型上传解压无路径遍历防护 —— 安全（已修复 ✅）

- **原位置**：model_service 上传解压处（原 L1516-1540）。
- **修复**：`d0a81fb fix(S-01)` 引入 `backend/src/utils/zip_guard.py`（路径遍历 + 解压炸弹双防护，`MAX_UNCOMPRESSED_BYTES` 默认 50 GB / `MAX_RATIO` 100，可用环境变量覆盖），model_service 现调用 `zip_guard.guarded_extractall(zip_file, temp_path, strict=True)`。
- 2026-10-10 复核确认在当前分支祖先链上，三处解压入口（dataset/model/annotation）统一收口。

### S-02　YOLO 标签导入解压无 member 校验 —— 安全（已修复 ✅）

- **修复**：`ffe2c79 fix(S-02)`，annotation_service 现调用 `zip_guard.guarded_extractall(zf, tmp_dir, strict=True)`，与 S-01 同一收口模块。

### B-05　CI 与构建脚本引用不存在的 Dockerfile —— 功能缺陷（已解决 ✅ 2026-10-10，批次 G-1）

- **现象**：`.cnb.yml` 三阶段流水线属分层镜像改造前旧架构：前两阶段用 `docker/Dockerfile`（统一镜像）重复构建 backend/frontend 两个 tag，第三阶段动态生成已废弃的 `Dockerfile.unified`；`build-unified.bat` 已先行删除。
- **拍板结论**：**删除 `.cnb.yml` 整份**（实际部署一直是本地 `build_push.sh --app-only` + 服务器 `deploy.sh`，CI 从未启用）。
- **实施**：`de6850e chore(G-1)` 仓库根 `git rm .cnb.yml`（89 行整份删除），`docker/` 其余 7 个 tracked 文件一字未动。
- **验证（验收方独立复核）**：`git diff bf5454c..HEAD -- .cnb.yml` = 89 deletions；`grep -rn cnb docker/` 零命中；`bash -n docker/build_push.sh` EXIT=0；`.cnb.yml` 已不存在于工作区。**B-05 终结**。

### B-08　N 批次 JSON IO 收口漏改 29 处**函数对象引用** → 运行时 NameError —— 功能缺陷（已修复 ✅ 2026-10-14，`a48a6fc`）

- **发现场景**：**首次实际部署验证**（服务器本地 build + `compose up -d --force-recreate`）后，容器启动日志每轮出现 `seal poll: scan failed: name '_load_json' is not defined`；`/health` 的 `background_loops.seal_poll.last_error` 同步暴露同一 NameError（前三个循环无 `last_error` 字段）。
- **根因**：批次 N（`0ec2a10`）把各 service 的私有 `_load_json` / `_save_json` 收敛为共享的 `utils/json_io.py` 的 `load_json` / `save_json_atomic`，并**删除了各文件的本地 `def`**。但改名**只覆盖了 `_load_json(` 左括号直接调用形式，漏掉「函数对象传递」形式**——典型写法 `await asyncio.to_thread(_load_json, path)`（函数名后跟逗号）。定义已删、引用未改 → 运行时 `NameError`。
- **清单（29 处）**：`dataset_service.py` **15**（含 `_collect_due_seal_datasets` 内 → seal_poll 每轮必踩，故启动约 8 秒即稳定复现）、`model_service.py` **6**、`train_service.py` **7**、`sam_service.py` **1**。除 seal_poll 这条「稳定复现」外，其余 28 处是随数据集/模型/训练元数据读写随时引爆的**暗雷**（如逐图标注保存、模型元数据读写、训练任务状态读写）。
- **为何此前未发现**：后端无 pyflakes/ruff 等静态检查，`vue-tsc` 只管前端；此类「未定义名」只在运行时暴露。本次全靠 seal_poll 定时循环 + `/health.last_error` 观测抓到（**M-1 双重探测 / K-3 lifecycle 心跳的投入直接见效**）。
- **修复**：`a48a6fc fix(json_io): 补漏 N 批次 29 处 _load_json/_save_json 对象引用改名（seal_poll NameError）` —— 改名 `_load_json`→`load_json`、`_save_json`→`save_json_atomic`（两名字本就在各文件顶部 `from ...json_io import` 中存在，仅改引用）。**4 files changed, 29 insertions(+), 29 deletions(-)**；`ast.parse` 语法全过；`grep -rn "_load_json\|_save_json" backend/src/services/` **0 残留**。
- **验证（真机闭环，全绿）**：本机修复 → push `backup/v2-dataflow-20261013`（`c5793a7..a48a6fc`）→ 服务器 `git pull --ff-only`（快进 4 files +29/-29）→ 重新 build（新镜像 `ded9b02e2431`）→ `cd /opt/yolo-main/docker && docker compose up -d --force-recreate yolo-platform` → 日志 `(无 seal poll / Traceback ✓)`、`/health` 的 `seal_poll` **不再出现 `last_error` 字段**、四循环心跳齐全。
- **有意未改**：`utils/class_labels.py`（自带 `def _load_json`，自洽）；`scripts/migrate_to_pg.py` / `scripts/backfill_has_label.py`（各自 `def`；其中 `migrate_to_pg.py` 属红线 B-03，不动）。
- **教训沉淀**：批量改名必须**同时覆盖 `f(` 与 `f,`**（`asyncio.to_thread` 等把函数当对象传递）两种形式；建议后续把 **pyflakes 纳入提交前检查**（可捕获全类别未定义名，成本极低）——关联改进项待登《功能实现与改进点跟踪》。

### B-11　GroundingDINO 每张图都推理失败（`Invalid device string: '0'`）—— 功能缺陷（已修复 ✅ 2026-10-27）

- **现象**：标注页每次 AI 预标注，日志固定出现两行：
  `multi-detect: 引擎 grounding_dino 推理失败: Invalid device string: '0'`、
  `multi-detect: 引擎 grounding_dino 推理失败: GroundingDINO 检测器不可用：Invalid device string: '0'`。
  界面上千问仍在出框，所以**看起来只是"刷日志"**，实际是 GD 引擎从第一次调用起就**从未真正工作过**（多引擎合并里永远只有千问一路结果）。
- **根因**：`device_manager.get_best_device()` 返回的是裸索引 `str(best["index"])` = `"0"`，而 `sam_service._resolve_device()` 只在「配置文件里手写 device」这条路径上做了 `"0" → "cuda:0"` 归一化，**`device="auto"` 分支直接 return 了 `get_best_device()` 的裸值**。于是 torch 收到 `"0"` 抛 `Invalid device string: '0'`。上一轮 `b9ae200` 只修了配置侧（用户手写 `device: "0"` 的场景），auto 分支漏改 —— **同一个 bug 的第二个入口**。
- **加重因素**：`_load_grounding_dino()` 先把模型赋给 `self._gd_model` 再 `.to(device)`；`.to` 抛异常后**半初始化的模型仍留在缓存里**（非 None），后续每张图都"复用"这个躺在 CPU 上的模型或继续抛同样的错，把「加载失败」伪装成「已加载」。
- **修复**：
  1. `device_manager.get_best_device()` 直接返回 `cuda:N`（注释写明为何不能返回裸索引）；
  2. 抽 `_normalize_device()`，`_resolve_device()` 的 **auto 分支与显式分支都过它**（双保险，任何来源的设备字符串都不会裸奔到 torch）；
  3. `_load_grounding_dino()` 的 `.to()` 失败时清空 `_gd_model/_gd_processor/_gd_name` 再 raise，坏实例不进缓存。
- **验证**：`py_compile` 通过。**真机待部署后确认**：日志不再出现 `Invalid device string`。
- **行为提醒**：修好之后 GD 会**真的开始出框**（此前等于关闭）。若当前只靠千问的结果已够用，建议在「AI 标注模型设置」里取消勾选 GroundingDINO —— 少一路引擎更快，也不会引入 GD 的误检。

### B-12　批量预标注整批失败（`批量预标注出错: Invalid device string: '0'`）—— 功能缺陷（已修复 ✅ 2026-10-27）

- **现象**：同一套配置下，单图「AI 预标注」能出框（日志只刷两行 GD 失败），但一点「批量预标注」，进度条没动就弹 `批量预标注出错: Invalid device string: '0'`，一张都标不了。用户视角像是"新问题"，其实是 B-11 的**第二个出口**。
- **根因（错误传播，不是设备本身）**：`_run_batch` 在进入主循环前会**预加载所有启用的引擎**（`for det in self._active_detectors(cfg): self._load_engine(det, cfg)`），这一步没有任何 per-engine 的 try/except。GD 因 B-11 的裸 `'0'` 抛异常 → 直接冒泡到 `_run_batch` 最外层 `except` → `bt.status="error"`。而单图链路 `_detect()` 是**逐引擎 try/except**（失败只打日志、继续下一个引擎），所以千问照常工作。**同一个坏引擎，一条链路容错、另一条链路不容错**，才出现"单图行、批量不行"的错觉。
- **顺带发现的两个隐藏问题**：
  1. **换引擎兜底从未生效**：`alt_cfg["detector"] = "grounding_dino"` 只改了 `detector` 字段，而 `_active_detectors()` 优先读 `detectors` 列表 → 多引擎配置下"换引擎"实际还是跑原来那批引擎；且它会**无条件尝试 GD**，即使 GD 本次已确认不可用（每张图都白等一次模型加载）。
  2. **全失败假报完成**：单图异常被 `批量预标注跳过图片` 吞掉并 `bt.done += 1`，若所有图都失败，任务以 `status=done, 标注 0 张` 收场，用户以为"图里没目标"而不是"服务坏了"。
- **修复**（`sam_service.py`）：
  1. 预加载**逐个引擎 try/except**，失败引擎记入 `bt.preload_warnings` 并从本次批量的 `detectors` 里剔除（不再每张图重试一次）；只有**一个引擎都没加载成功**才 `raise`，错误信息带上 `device=` 与「设备改成 auto/cpu」的指引；
  2. SAM 精修加载失败降级为「只出方框」，不再中断整批；
  3. 换引擎兜底改为**只用本次确认可用的引擎**（`usable`），并同时覆盖 `detectors` + `detector` 两个字段，换引擎才真的换；
  4. 单图异常累计 `bt.failed_images` / `bt.last_error`，若「全部图片都失败且一个框都没写入」则以 `error` 结束并回传最后一条错误，杜绝假 `done`；
  5. 批量开始打一行 `qwen: 批量开始 引擎=[...] 设备=cuda:0 已跳过=...`，一眼看清实际跑的是哪几路引擎、设备解析成了什么。
- **验证**：`py_compile` 通过；`/batch/{id}` 状态新增 `preload_warnings`、`failed_images` 字段（前端不读也不受影响）。**真机待部署后确认**：批量能正常开跑，日志出现「批量开始」那行且 `已跳过` 为空。
- **部署提醒**：本次修复与 B-11 的设备归一化都在**代码层**，`docker compose up -d` 不会自动生效 —— 必须重新 build 镜像（`bash docker/publish.sh` 或服务器本地 build），否则容器里仍是旧代码，症状完全一样。

### B-13　千问把近似物误标成目标（幻觉框）—— 模型幻觉（已缓解 ✅ 2026-10-27，见 I-17）

- **现象（真机截图）**：画面（矿上候车室，两名工人）中**根本没有自救器**，AI 预标注却出了 2 个框——分别套在「头顶矿灯/发髻」和「红色安全帽」上。用户反馈"有时候画面中并没有自救器，就会乱标"。
- **定性**：VLM 整图检测的**固有幻觉**，不是链路 bug。qwen3-vl:4b 对煤矿专有设备无领域知识，看到"头上戴的东西"就近似脑补；且千问不返回真实置信度（`score` 硬编码 0.9），无法用阈值过滤；已有的 L2 审核（I-15）默认关、且审核结论只标注不删框。
- **缓解（三层，`28d1f4e` 后续提交）**：
  1. **检测提示词收紧**：明确「只框目标本体、宁缺毋滥；人/头部/安全帽/矿灯/衣物不是目标；看不清不输出」——能压一部分，压不干净；
  2. **框级二次校验（核心，默认开）**：`_detect_qwen` 对每个候选框按原图坐标裁剪外扩小图（`qwen_verify_margin=0.35`），复问一次「图里是否出现至少一个完整的『X』？相似但不同的物体不算」，回答「否」即丢弃——小图 token 少，单框约 +0.3~0.6s；校验调用失败 fail-open 保留原框（网络抖动不丢真目标）。配置 `qwen_verify_enabled`（前端「AI 标注模型设置 → 准确性」可关）；
  3. **数据侧（用户可立即做，效果最直接）**：标签字典里把「自救器」的 `chinese_desc` 写成**视觉描述**——外观（颜色/形状/大小/佩戴位置，如"悬挂在腰间的橙黄色扁盒，带背带"）+ 易混淆物否定（"不是安全帽、不是矿灯、不在头上"）。整图检测与校验复问都吃这段描述。
- **验证**：`py_compile` + `vue-tsc` 通过。真机待部署后确认：对截图同款场景重标，日志出现 `qwen verify: 拦截 N 个疑似幻觉框` 或不再出框。
- **遗留**：根治仍建议训练自有模型（本库已训模型引擎 `_detect_trained`），零样本 VLM 只能缓解不能消灭幻觉。

---

### B-14　训练启动即被熔断中断（AMP 自检联网下载卡死）—— 功能缺陷（已修复 ✅ 2026-10-28）

- **现象**：任务 `job_20261028_011142`（数据集 `zijiuqi_20261028_010001_v1`，模型 `/app/models/custom/yolov8m.pt`，batch=12 / imgsz=640 / epochs=300）启动约 15 分钟后被中断；用户怀疑**服务器显存 OOM**。
- **排查结论：不是 OOM。** 三条硬证据：
  1. 日志 `Transferred 475/475 items from pretrained weights` —— yolov8m 权重**加载成功**，模型文件正常；
  2. GPU 为 `NVIDIA A10 (24114MiB)`，batch=12 的 yolov8m 训练远用不满 22GB，且日志全程**无** `CUDA out of memory` / `RuntimeError` 字样；
  3. 过滤掉下载进度行后，日志从 `AMP: running Automatic Mixed Precision (AMP) checks...` **直接跳到熔断**——中间约 180 行全是 `Downloading ... yolo11n.pt` 进度条，**一个 epoch 都没开始**。
- **根因**：ultralytics 8.3.251 的 `check_amp()` 内部会执行 `YOLO("yolo11n.pt")`，从 `github.com/ultralytics/assets` 拉约 5.4MB 权重做 FP16/FP32 前向比对。服务器访问 GitHub 实测 **10~30 KB/s** 且 TLS 连接被反复重置（日志有 `Download failure, retrying 1/3 2/3 3/3 ... SSL: UNEXPECTED_EOF_WHILE_READING`）。训练进程卡在下载上十几分钟，期间 GPU 利用率恒为 0%。
- **致命连锁**：`gpu_watch` 后台循环只看 GPU 利用率——利用率 <10% 持续 300s 即判"GPU 异常/训练卡死"并 SIGTERM 杀掉训练。**"启动阶段的网络等待"被误判为"GPU 卡死"**。续训（resume）后会原样再走一遍 AMP 下载，再卡再熔断，形成死循环（该任务熔断两次：01:26:35、01:33:37）。另一个任务 `job_20261028_012750` 同一死法（01:33:07）。
- **修复（三处）**：
  1. **`train_script.py` 新增 `patch_amp_check_offline()`**：在 `from ultralytics import YOLO` 后打补丁，把 `check_amp` 换成本地判定（CUDA/MPS → 启用 AMP，CPU → 关闭），彻底跳过联网比对。注意 `engine/trainer.py` 是 `from ..utils.checks import check_amp` 的模块级绑定，**必须同时替换 `checks` 与 `trainer` 两处引用**，只改一处无效。打补丁失败不阻塞训练（退回原生逻辑）；
  2. **`settings.GPU_ABNORMAL_GRACE_SECONDS = 900`**：本轮训练启动后 15 分钟内不做熔断判定（覆盖环境加载/权重加载/数据集扫描等天然 0% 利用率阶段）。取 `resumed_at or created_at` 计时，续训会重新计时；
  3. **续训清理失败残留**：`resume_training` 把 status 改回 running 时，一并清掉 `error / reason / gpu_abnormal / oom / oom_recovery / failed_at / crashed_at / crashed`——否则前端显示"训练中"却带着失败原因，状态自相矛盾。
  4. **顺带加固** `_resolve_job_gpu`：缓存的 GPU 索引若已不在 `get_available_gpus()` 列表里（掉卡/驱动重置）则重新推断，避免拿着失效索引去采样。
- **即时绕行方案（不等重建镜像）**：把 `yolo11n.pt` 放到训练进程工作目录 `BASE_DIR`（容器内 `/app/backend/`）即可——`YOLO("yolo11n.pt")` 会优先加载本地同名文件，不再联网。可从能联网的机器下载后 scp 进去（本机实测 91 秒，5,613,764 字节）。
- **验证**：`py_compile` 通过；部署后训练日志应出现 `AMP 自检已改为离线判定（跳过 yolo11n.pt 联网下载）`，且启动到第一个 epoch 应在 1~2 分钟内完成。

---

### B-15　熔断信息里的 GPU 索引与实际 GPU 数不符 —— 待确认（🟡 可能是掉卡信号）

- **现象**：`job_20261028_011142` 的 `error` 字段写「训练所在 **GPU1** 利用率持续低于 10%」，但训练日志首行是「检测到 NVIDIA GPU: NVIDIA A10 (**共 1 个设备**)」，且 `/train/gpus` 当前只返回 `index:0` 一块卡。同一任务的日志文件里，第二次熔断写的是 **GPU0**。
- **疑点**：`get_gpu_utilization(1)` 用 `nvidia-smi -i 1`，卡不存在时 `returncode != 0` → 返回 `None` → 调用方 `continue` 跳过，**不会**触发熔断。因此"GPU1"只能是 `_resolve_job_gpu` 真的从 `get_available_gpus()` 里枚举到了 index=1，即**熔断那一刻 nvidia-smi 列出了 2 块卡，之后少了一块**。
- **影响**：若属实，服务器存在**掉卡**风险（第二块 GPU 掉线），训练若被调度到该卡会直接失败或长期空转。
- **待办（需上机确认，暂无免密 SSH）**：在服务器执行 `nvidia-smi -L`（看列出几块卡）、`nvidia-smi`（看是否有 `ERR!`/掉卡）、`dmesg | grep -i -E "xid|pcie|nvidia"`（Xid 错误常伴掉卡），并与 `lspci | grep -i nvidia`（看物理卡数）对比。若 PCI 层有 2 块而 nvidia-smi 只列 1 块，即为掉卡，需检查供电/散热/驱动。
- **已做的防御**：`_resolve_job_gpu` 增加索引有效性校验（见 B-14 修复第 4 点），即使掉卡也会重新推断到存活的卡，不会拿着失效索引误判。

---

## 3. 🟡 中

### S-03　无鉴权 + CORS 全开 —— 安全（已拍板 ✅ 2026-10-10）

- **现状**：后端所有端点无 token/session 校验；`main.py` CORS `allow_origins=["*"]`；nginx `client_max_body_size 20480m`。
- **拍板结论**：**确认嵌入宿主平台**。鉴权与来源限制由宿主网关承担（黑名单 X-04 禁止自建用户体系），本平台不加鉴权中间件，S-03 关闭。
- **残留约束**：部署文档须写死前提——`:8000`/`:3000` 不得直接暴露公网，必须经宿主网关（待并入部署评估文档的部署前提节）。

### B-02　清除标注不回填 PG，`has_label` 虚高 —— 功能缺陷（已拍板 ✅ 2026-10-10）

- **现象**：`clear_ai_annotations`（L1013）与 `clear_all_annotations`（L1054）的 docstring 明写"不回填 PG 影子库（一图一 JSON 才是唯一权威）"。清标后 JSON 真值已空，但 PG `task_items.has_label` 仍为 true，按 PG 查询的计数/筛选会偏大。
- **拍板结论**：**接受虚高（A 方案）**。清标不动 PG；重新标注保存时由既有写通链路覆盖对齐——`save_annotation`（annotation_service L1018-1023）每次保存按当前框数调 `meta_store.update_task_item_has_label(..., bool(norm_boxes))`，有框写 true、0 框写 false，天然实现"标注完覆盖 PG 原记录"，无需新增反向回填链路。
- 关联改进项 I-04 随之关闭。

### B-03　全量重灌冲掉 `has_label` —— 功能缺陷（流程风险，已缓解）

- **机理**：task.json items 只带 `annotated` 不带 `has_label`；`migrate_to_pg.py` 全量重灌 tasks 域时整表替换 → `has_label` 变 NULL。
- **缓解**：`migrate_to_pg.py` 文档头已写警告；部署流程固定第二步跑 `backfill_has_label.py`；回归 T4 用例（合成任务 + `with_items=True` 反向对照）守住收口层不再误用。
- **残留风险**：人工只跑重灌忘了回填 → 列表页"有标注"筛选失效。处置：把二步并入 `deploy.sh`（I-05）。

### B-04　`weight_mappings` 有表无 DAO —— 功能缺陷（已解决 ✅ 2026-10-10，批次 G-3）

- **现状**：`db/models.py` L150 定义了表，`meta_store` 无对应读写函数，`shadow.py` 无接线，`migrate_to_pg.py` 不灌它 → 恒为空。实际跨代迁移由 ultralytics `strict=False` 自动匹配承担（train_service L1107 注释），从未用到过该表。
- **拍板结论**：**选 C 删表**——跨代映射功能暂缓，已移入下版本需求。
- **实施**：`bbd3d6d fix(G-3)` 三处——① `db/models.py` 删 `WeightMappingRecord` 类（文件尾块，实测 L148-162，删后 147 行、以 `rule:` 行结尾、恰一个尾 LF、无 BOM）；② `train_service.py` L1107 注释去「 + weight_mappings 溯源」字样；③ 本地 PG 执行 `DROP TABLE IF EXISTS weight_mappings`（服务器存量库待部署时执行同一条）。
- **验证（验收方独立复核）**：`git grep WeightMappingRecord|weight_mappings -- backend/ docker/` 零命中（exit 1）；`from src.db import models` 无报错、`hasattr(WeightMappingRecord)=False`、`Base.metadata.tables` 恰 7 张且不含 `weight_mappings`；本地 PG 实测 `to_regclass('public.weight_mappings')=None`、`pg_tables` 7 张。
- 关联改进项 I-06 随之关闭。

### P-01　`goAnnotate` 整页刷新 —— 性能/体验（已解决 ✅ 2026-10-06，`305756e`）

- **位置**：DatasetUpload.vue L560-L562：`window.location.href = '/annotate?dataset_id=' + id`，而非 `router.push`。
- **影响**：离开 SPA 路由，白屏重载、前端状态全丢；与站内其他跳转（全部 `router.push`）风格不一致。
- **处置**：改为 `router.push('/annotate?dataset_id='+id)`。一行改动、低风险，属"不需要复杂代码"的改进，允许做。
- **结案（2026-10-10 复核）**：`305756e fix(P-01,D-02)` 已落地。实测 `frontend/src/pages/DatasetUpload.vue` L764-767：L764 注释自标「P-01：改用 router.push 走 SPA 跳转，避免整页刷新丢失前端状态」，L766 为 `router.push('/annotate?dataset_id=' + dataset.dataset_id)`；`git grep window.location.href -- frontend/src/pages/DatasetUpload.vue` **零命中**（ModelDetail.vue L1169/L1304 另有 2 处，不属 P-01 范围，未在本项内处理）。本文档状态此前漏改，本批次更正。

### P-02　超大 task.json 预标注 O(N²) —— 性能（已确认 ✅ 2026-10-13 实测）

- **根因（实测复现）**：`annotation_service.save_annotation`(L917-1021) 每次保存都 `load_json(task_file)` **全量读 task.json**（L933），但只写一图 JSON（`_save_image_ann` L1009，不动 task.json）。`sam_service._run_batch`(L2519-2707) 对 N 张图循环预标注 → N 次 save_annotation → N × O(N) = **O(N²) 总 IO**。
- **实测数据**（独立基准脚本 `C:/Users/41653/.workbuddy/tmp/p02/bench.py`，模板对齐 `task_20260810_193124/task.json`，结构 1:1）：

  | N items | task.json MB | T1 单次 load ms | T2 连续 N 次 load 总 s | T2/张 ms |
  |---:|---:|---:|---:|---:|
  | 100 | 0.013 | 0.33 | 0.03 | 0.25 |
  | 1,000 | 0.125 | 1.45 | 0.76 | 0.76 |
  | 5,000 | 0.625 | 4.42 | 26.0 | 5.20 |
  | 10,000 | 1.249 | 10.10 | 95.9 | 9.59 |
  | **13,000** | **1.624** | **8.79** | **121.8** | **9.37** |

  T2 总耗时拟合：N 增长 130× → T2 增长 **4872×** → 拟合指数 b = log(4872)/log(130) = **2.00**（完美 O(N²)）。

- **影响量化**：
  - 13k items 任务批量预标注总浪费 = **122 s**（纯重复读 task.json IO），约占整批量 50%
  - 10k items = 96 s；1k items = 0.76 s —— **任务规模越大浪费越严重**
  - OS page cache 命中下单次 load 仍要 9 ms；若 cache miss（其他写操作弄脏，如 `create_task`/`add_images`/`shadow.sync_task`），退化到磁盘 IO，13k 单次 ≈ 25-50 ms → 13k 总浪费 ≈ 5-10 分钟
- **判定**：**O(N²) 确认存在**。当前量级「温和」（13k 任务 2 分钟），不是「慢到不能用」，但属于**可避免的浪费**——save_annotation 在保存阶段本不需要 task.json。
- **处置（预留，不在本批范围）**：改进项 I-13「save_annotation L933 去除全量读」——save 时不再依赖 task.json（调用方传 info 入参，或每个 task 维护内存级 `_task_meta_cache`，只在增删图片时刷新）。预估收益：13k 任务预标注节省 ~120 s，10k 节省 96 s。**风险点**：PG `has_label` 回填路径（L1015-1020）依赖 task 上下文，影响面需评审后再动。
- **未在本批做修复的理由**：① 改善为体验优化而非阻塞功能（🟡 中）；② 改动需要 review L938/N-context/PG 同步三方状态机；③ 与 I-10（`/health` 观测）联动更稳——先把「何时退化」观测到位再谈优化。

### D-01　`平台细节说明.md` §2.4 标注实体描述过期 —— 文档漂移（已解决 ✅ 2026-10-06，`905a535`）

- **文档说法**：「只有矩形框 BBox，无多边形/掩码」。**代码实际**：`polygon` 轮廓已实现（`annotation_service`、`DatasetAnnotate.vue`、`annotation_importers` 等 8 个文件命中）。
- **处置**：改文档（I-12）。
- **结案（2026-10-10 复核）**：`905a535` 已同步。实测仓库根 `平台细节说明.md` L69 已写「框可带可选 `polygon:[[x,y],…]`（像素坐标 ≥3 点）= 分割轮廓（SAM 精修/手工描点），无 polygon 者仍为普通检测框，两类可共存」，"无多边形"旧说法已消失。⚠️ 该文件被 `.gitignore` L28 排除且未被 git 跟踪，修复仅存于本机磁盘，见 §4 风险登记行。改进项 I-12 随之关闭。

### D-02　`/datasets?focus=` 参数无人消费 —— 口径漂移（已解决 ✅ 2026-10-06，`305756e`）

- **现象**：ModelDetail.vue L185 跳 `/datasets?focus={id}`，但 DatasetUpload.vue **没有 `useRoute()`**（只有 `onMounted` L975），`focus` 参数被完全忽略——用户跳过去后看不到"被聚焦的数据集"。
- **处置**：DatasetUpload 加 `useRoute` 读取 `query.focus` 并滚动/高亮对应卡片。小改动，允许做。
- **结案（2026-10-10 复核）**：`305756e` 已落地。实测 DatasetUpload.vue：L454 `import { useRouter, useRoute } from 'vue-router'`、L466 `const route = useRoute()`；L1282-1300 `applyFocus`（`[data-ds]` 定位 + 最多 10 次 100ms 重试 + `scrollIntoView({block:'center'})` + 内联 outline/box-shadow 高亮 4s 后淡出）；L1302-1306 `consumeFocus`（读 `route.query.focus`，连带消费 `route.query.stage`）；调用点两处——L1313（`onMounted`，注释自标「D-02」）与 L1326（`onActivated`，覆盖 B-10 keep-alive 场景：带 focus 跳回时组件不重新挂载，首次激活跳过）。ModelDetail.vue L185 的 `/datasets?focus=${d.dataset_id}` 现已有消费方。

### D-03　归档数据集在 PG 可见、列表不可见 —— 口径漂移

- **机理**：`list_datasets`（dataset_service L1693）跳过 `_` 前缀目录；但 `migrate_to_pg.py` 与 `shadow.py` 的 glob **不跳过** → PG 里存在 `_archived_ds_v1` 等行（含 `ds_v1` 这种文件内 id 兜底出来的行，属口径而非 bug）。
- **影响**：直接查 PG 的下游（含未来嵌入方）会看到列表页不存在的"幽灵数据集"。
- **处置**：统一口径——migrate/shadow 同样跳过 `_` 前缀，或给 datasets 表加 `archived` 列并在文档声明。倾向后者（改动小、信息不丢）。

### D-04　API_DOCUMENTATION「推理服务」章节无对应实现 —— 文档漂移（已解决 ✅ 2026-10-06，`905a535`）

- **处置**：在章节头标注"未实现/由宿主平台提供"（当前阶段禁止补推理代码，见 X-08）。
- **结案（2026-10-10 复核）**：`905a535` 已落地。实测 `API_DOCUMENTATION.md`（仓库根）L1026 为章节标题 `## 推理服务`，紧随其后的 **L1028** 已插入警示块：「⚠️ **本节接口本平台未实现（D-04 / 红线 X-08）**：平台定位是"标注 + 训练"，推理由**宿主系统**承担（模型产出 `best.pt` 后交由宿主加载推理）。下列 `/infer/*` 路由在本仓库 `backend/src` 中**不存在对应实现**，仅作历史接口约定保留，勿按此对接。」；`git grep -n "infer" -- backend/src/main.py` 零命中（exit 1），与警示文字一致。未补任何推理代码，符合 X-08。改进项 I-12 随之关闭。

---

## 4. 🟢 低（待处理，后期统一修正）

| 编号 | 分类 | 描述 | 处置备忘 |
|---|---|---|---|
| D-05 | 文档 | 平台细节说明多处「满 1500 自动封板」，实际 `SEAL_MIN_IMAGES=5000`（README 为 5000；09-30 复审发现代码注释另有 4 处过期写 1500：dataset_service L2601/L2828/L2928 + main.py L18） | 改文档 + 注释改为引用 `settings.SEAL_MIN_IMAGES`，随 I-12 一起做 → ✅ **结案（2026-10-10 复核）**：`905a535` 已修。`git grep -n "1500" -- backend/src/**/*.py` **零命中（exit 1）**；注释已改引用配置项（实测 `main.py` L20 lifespan docstring 写「满 `settings.SEAL_MIN_IMAGES` 自动封板」，`dataset_service.py` L2722/L2727/L2896/L2954 同口径，原行号随代码演进漂移到 2722+/2954+）；`平台细节说明.md` L38/L132/L135/L272/L291 五处均写「默认 5000」。⚠️ 该说明文件未入 git，见下方风险登记行。I-12 关闭 |
| D-06 | 文档 | 文档称前后端状态"一套口径"，实际 `USE_UNIFIED_STATUS=False`，读侧仍走旧列 | 等 I-03 灰度决策后一并修订 → ✅ **结案（2026-10-10 复核，文档侧）**：`905a535` 已在 `平台细节说明.md` **L54** 声明「状态口径现状（D-06）：影子库已建 `unified_status` 统一列，但 `USE_UNIFIED_STATUS=False`——**读侧仍走旧列**（stage/status/training_status 多套），统一状态机仅影子库写入侧生效，灰度切换待 I-03 决策」。文档不再谎称"一套口径"，本项关闭；**代码侧灰度切换仍开放，归 I-03 需用户拍板** |
| P-03 | 健壮性 | 损坏/超小 zip（教训：1.15 MB 的 person.zip）上传即入库，训练阶段才暴露无效图片 | 上传/prepare 时加"可解码抽样 + 包体大小下限"校验 |
| P-04 | 体验 | `axios.ts` 无响应拦截器，错误处理散落在 6 个页面各自 try/catch + alert | 加统一拦截 + toast（I-08）→ ✅ **已解决（2026-10-11 批次 J）**：① `api/axios.ts` 加响应拦截器（`d8e3dac`）——网络异常/超时（无 `err.response`）与 5xx 统一弹全局 toast 并打 `err.__toastShown = true`，**只标记不吞错**，仍原样 `Promise.reject(err)` 交由各页 catch；`baseURL`/`timeout` 未动。4xx（含 404）不在拦截器分支，仍由各页自行提示，避免与既有 catch 重复弹两次。② 新建 `utils/toast.ts`（纯 DOM 单实例懒挂载、`success/error/info` 三态、可配时长、连续提示复用同一 timer），`style.css` 追加 `.app-toast`（`z-index:10000`、`white-space:pre-line`、配色逐值沿用原 `.delete-toast`），**未引入任何 UI 组件库**（守红线 X-02）。③ `main.ts` 加 `unhandledrejection` 兜底（漏 catch 的 axios 错误弹 5000ms toast，已有 `__toastShown` 则跳过）；原 `window.alert` 全局替换逻辑（L11-13）**未动**。④ 6 个页面 143 处 alert + 11 处原生 confirm 全量迁移（`83ccb89`/`4498e45`/`0a85f09`）：Dashboard 3+1、Models 15+0、ModelDetail 18+8、DatasetUpload 28+0、Train 23+0、DatasetAnnotate 56+2。实测 `git grep -rn "alert(" frontend/src/pages/` **零命中**、原生 `confirm(` **零命中**。→ I-08 关闭。<br>**机制更正说明**：`window.alert` 早已在本批次之前被 `composables/useDialog.ts` 的 `showAlert` 全局替换为**居中模态弹框**（`main.ts` L11-13），P-04 原始描述「各自 try/catch + alert」中的 alert 实为该弹框、**非浏览器原生 alert**——所以本项的真实痛点是「阻塞式弹框 + 错误处理散落」，而非原生弹窗被 iframe 拦截。<br>**三套提示机制收敛结果**：① `window.alert` 全局替换成的 useDialog 弹框 → 保留 `showAlert`/`showConfirm`/`showPrompt` 能力，但替换逻辑**仅保留给历史路径**（本批次后 pages 目录已无 `alert(` 调用点）；② DatasetAnnotate 页内自绘的本地 toast（`toastMsg`/`toastType`/`toastTimer`/`showToast`/`.delete-toast`）→ **删除并收敛到全局 `utils/toast.ts`**，签名一致故 36 处既有调用点零改动；③ 原生 `confirm(` → **全部改 `await showConfirm`，清零**。至此提示机制统一为「轻提示用全局 toast、需确认用 showConfirm」两套 |
| B-06 | 死代码 | ~~`components/Logs/LogPanel.vue`、`LogToolbar.vue` 不被任何页面引用（Train 自绘日志区）~~ | 接入或删除（I-07）→ ✅ **已解决（2026-10-10 批次 I：删除未接入的预留组件）**：选择「删除」而非「接入」——Train 页自绘日志区已稳定运行且直连 `useLogStore`（实测 `Train.vue` L517 import、L520 取 store、L1285/L1404 `clearLogs`、L1458 `setStreaming`、L1465/L1486 `addLog`、L1911 `setStreaming(false)`），替换成预留件属无收益重构。删除前复核引用面：`git grep -n "LogPanel\|LogToolbar\|components/Logs"` 全仓命中仅 `LogPanel.vue` L3 `<LogToolbar`、L32 `import LogToolbar` 两处**自引用**，无任何页面/路由/测试引用 → 判定死代码无疑，`git rm` 两文件（`Logs/` 目录随之消失）。`useLogStore`（`store/logs.ts`）**保留**，Train 页仍在用，未产生孤儿 store。残留文档引用（本批次按「diff = 2 deleted + 2 docs」约束未动，登记待后续清理）：`docs/平台功能与页面关联.md` L295/L531/L532/L534、`docs/项目文件功能说明.md` L248/L319 仍列这两个组件 |
| B-07 | 风险约定 | `/models` 路由：`/{model_id}` 通配注册在前会吞静态段。现有顺序正确，但**新增静态路径必须插在通配之前** | 已在《平台功能与页面关联》§6 留警示，代码加注释即可 → ✅ **结案（2026-10-10 复核）**：`905a535` 已把警示写进代码。实测 `backend/src/api/routes/models.py` **L249-251**（紧邻 L252 `@router.get("/{model_id}")`）：「⚠️ B-07 路由顺序陷阱：自本行起为通配路由 `/{model_id}`。FastAPI 按注册顺序匹配，新增任何 GET 固定路径（如 /xxx）必须写在本注释**之前**，否则会被 `/{model_id}` 吞掉（model_id="xxx"），表现为诡异的 404/错数据。」文档 + 代码双侧留痕，本项关闭 |
| I-11 | 悬空 | `backend/data/inference_results/` 目录存在但无路由/服务写入方 | 确认历史用途后清理或补文档 → ✅ **已结案（2026-10-10 批次 I）**：核实为**空目录**（`Get-ChildItem -Force` 计数 = 0，仅 `.`/`..`），且 `git grep -n "inference_results" -- backend/src/**/*.py` **零命中（exit 1）**——无任何路由/服务读写方，历史遗留无疑。已删除该空目录（git 不跟踪空目录 → **无 diff**，动作记在 commit body）。残留（均为文档层，本批次按「diff 恰 2 docs」约束未动）：`docs/项目文件功能说明.md` L171 仍列该目录为「⚠ 无对应路由，历史遗留」，`API_DOCUMENTATION.md` L1336/L1385/L3392 在**未实现**的推理章节与 CLI 示例内引用同名路径（已由 D-04 警示覆盖） |
| — | 悬空 | ~~`docker/backend/`、`docker/frontend/` 空目录（B-05 的诱因）~~ | ✅ 已删除（09-30），B-05 已于 10-10 随批次 G-1 终结 |
| — | 风险登记 | `平台细节说明.md`（仓库根）被 `.gitignore` 排除且未被跟踪（实测 `git check-ignore -v` 命中 `.gitignore:28`、`git ls-files --error-unmatch` exit 1）→ **D-01/D-05/D-06 的文档修复仅存于本机磁盘**，clone/换机器即丢失，且审阅者无法从仓库看到口径已更正 | 处置待用户确认仓库分发范围（是否解除 `.gitignore` 对 `docs/`、`平台细节说明.md` 的排除，或明确标注为内部文档，见《功能实现与改进点跟踪》§4 各模块优化方向「工程」行）；**本批次不动 `.gitignore`** |

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
| R-08 | 镜像内缺 `/app/scripts`，部署后 `docker compose exec` 跑 `migrate_to_pg.py` / `backfill_has_label.py` 报 No such file（文档外发现，内部代号 BLOCKER-1） | `b062ba0 fix(G-2)`：`docker/Dockerfile` 在 `COPY backend/models`（L49）后插入 `COPY backend/scripts /app/scripts`（现 L53），`.dockerignore` 已排除 `__pycache__` | 验收方独立复跑：`python backend/scripts/migrate_to_pg.py --help` 与 `backfill_has_label.py --help` 均打印 usage、EXIT=0（证明 `parent.parent` 自定位 + `import src.*` 链通）。真实镜像构建顺延至服务器 `build_push.sh --app-only` 后补验 |

---

## 6. 分类统计与处理原则

| 分类 | 🔴 | 🟡 | 🟢 | 已归档 |
|---|---|---|---|---|
| B 功能缺陷 | 0 | 1（B-03 已缓解） | 0 | 3 + 当日关闭 2（B-04/B-05）+ I-01/B-01 + 批次 I 前置更正 1（B-07）+ 批次 I 删除 1（B-06）+ 批次 N 补漏 1（B-08） |
| P 性能/健壮性 | 0 | 1（P-02 已确认·预留） | 1（P-03） | 2（批次 I 前置更正：P-01 + 批次 J：P-04） |
| S 安全 | 0（S-01/S-02 已修复） | 1（S-03 已拍板关闭） | 0 | 0 |
| D 文档/口径漂移 | 0 | 1（D-03） | 0 | 5（批次 I 前置更正：D-01/D-02/D-04/D-05/D-06） |
| C 兼容性 | 0 | 0 | 0 | 1（C-01，见下） |

> **2026-10-10 更新说明**：上表各列为当日复核后的实际在册数。S-01/S-02 由 `d0a81fb`/`ffe2c79` 修复（zip_guard 收口），B-05 由 `de6850e` 关闭，B-04 由 `bbd3d6d` 关闭，I-01/B-01 核实 4 文件全部 TRACKED、未跟踪文件归零；S-03/B-02 拍板关闭（见 §3 详情节）。
>
> **2026-10-10 批次 I 前置更正说明**：本表按更正后口径重算——此前 7 项（P-01/D-01/D-02/D-04/D-05/D-06/B-07）实际已由 10-06 的 `305756e`（P-01/D-02）与 `905a535`（D-01/D-04/D-05/D-06/B-07）修复，但文档状态漏改仍标「待处理/约定防护」，现全部移入「已归档」列（B 🟢 2→0、P 🟡 2→1 且已归档 0→1、D 🟡 4→1 且 🟢 2→0 已归档 0→5）。另：**I-11**（悬空目录 `backend/data/inference_results/`）本批次结案并删除空目录，该项在 §1 总览单列为「悬空目录」类、原表无对应分类行，故不计入上表五行，仅在此登记；git 不跟踪空目录，删除不产生 diff。**B-06**（死代码 `components/Logs/*`）已于批次 I 随 `git rm` 两文件关闭，B 类 🟢 归 0。当前全表在册**未关闭项仅剩 5 条**：B-03（已缓解，靠 I-05 落地——本批次已并入 `deploy.sh`）、P-02（需实测）、P-03、P-04、D-03（待批次 K）。
>
> **2026-10-11 批次 J 更新说明**：**P-04** 于批次 J 结案（详见 §1 总览 L35 与 §4 低优先表），P 类 🟢 由 2 归 1（仅剩 P-03）、已归档由 1 增 2。连带关闭《功能实现与改进点跟踪》的 **I-08**（全局轻提示 + 统一错误拦截）与 **I-09**（SAM 接口从 `api/annotations.ts` 拆出为 `api/sam.ts`）。批次 J 全部为前端 + 文档改动，**零 backend 改动**。当前全表在册**未关闭项仅剩 4 条**：B-03（已缓解）、P-02（需实测）、P-03、D-03（待批次 K）。批次 J 遗留待拍板一项：`api/sam.ts` 内 3 个疑似死导出 `getSamAvailable` / `uploadSamModel` / `AIBox`（全仓仅定义处或仅文档提及、无代码调用方），按「拆分为纯搬迁、零逻辑变更」原则随迁保留未删，已在《功能实现与改进点跟踪》I-09 行登记。
>
> **2026-10-14 首次实际部署验证更新说明**：本批为「部署验证」性质，**新增 B-08**（N 批次 JSON IO 收口漏改 29 处函数对象引用 → 运行时 `NameError`，🔴），已在同日 `a48a6fc` 修复并**真机重部署验证通过**（详见 §2 B-08 详情节），B 类「已归档」+1、🔴 在册仍为 0。本次部署验证同时完成：服务器本地 build（复用本地 env 层 `yolo-app-env:latest`，免推阿里云 ~10GB）、`compose up -d --force-recreate` 换用新镜像、容器内代码哈希与本机一致、`/health` 的 `background_loops` 四循环心跳齐全（M-1 双重探测 / K-3 lifecycle 心跳**落地验证通过**）、前端 200。**在册未关闭项仍为 4 条**：B-03（已缓解）、P-02（已确认·预留）、P-03、D-03（待批次 K）。

**C-01（归档）**：`data.yaml` 曾用绝对路径导致跨平台（Windows 开发 / Linux 容器）训练失败 → 已定硬约束：**一律相对路径 `.`**。后续任何改 `seal_dataset`/`prepare` 的任务不得回退。

处理原则（与《功能实现与改进点跟踪》§3 黑名单一致）：
1. 🔴 全部属于"补防护 / 入 git / 修脚本引用"，**不需要新框架**，当前阶段就该做；
2. ~~待拍板三项（S-03、B-02、B-04）**未决策前不动代码**~~ → **已于 10-10 全部拍板**：S-03 嵌入宿主/网关鉴权（关闭）；B-02 接受虚高+重标覆盖（关闭）；B-04 删表、功能移入下版本需求（并入批次 G）；
3. 🟢 项攒批处理，修一个带一个，不单独排期；
4. 任何修复必须附回归脚本；§5 的 `test_*` 属资产。
   - ⚠️ **2026-10-10 更正**：经全盘核实，`_tmp_scripts/` 回归脚本（`test_fix_3_6.py` 34 项、`test_shadow_writethrough.py` 36 项、S1~S9 50 项）**在本地与 git 中均不存在**（`.gitignore` 仅含 `_tmp/`），此前"脚本已保留"的记录不成立。后续任务书中的"跑回归脚本"验证项按"脚本不在则如实报跳过、不许自造替代"处理。

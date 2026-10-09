<template>
  <div class="train">
    <div class="card">
      <h2>创建训练任务</h2>
      <div class="form-group">
        <label>选择数据集</label>
        <div class="ds-selectbar">
          <button type="button" @click="loadDatasets" class="refresh-btn" :disabled="loadingDatasets">
            <span v-if="loadingDatasets" class="loading-spinner"></span>
            {{ loadingDatasets ? '加载中...' : '刷新' }}
          </button>
        </div>
        <div class="ds-grid">
          <div
            v-for="ds in datasets"
            :key="ds.dataset_id"
            :class="['ds-card', { selected: isDsSelected(ds.dataset_id), disabled: !isDatasetPrepared(ds) }]"
            @click="selectDataset(ds)"
          >
            <div class="ds-card-header">
              <span class="ds-name">{{ ds.display_name || ds.dataset_id }} <span v-if="trainForm.dataset_id === ds.dataset_id" class="ds-main-tag">主</span></span>
              <span :class="'status-badge status-' + ds.status">{{ getDatasetStatus(ds) }}</span>
            </div>
            <div class="ds-card-info">
              <span v-if="ds.filename">文件: {{ ds.filename }}</span>
              <span v-if="ds.image_count">{{ ds.image_count }} 张图片</span>
              <span v-if="ds.classes && ds.classes.length">{{ ds.classes.length }} 类</span>
            </div>
            <div v-if="!isDatasetPrepared(ds)" class="ds-card-warn">需先"准备"</div>
          </div>
          <div v-if="!loadingDatasets && datasets.length === 0" class="empty-state">暂无已标注导出的数据集，请先在标注页完成标注并导出</div>
        </div>
        <small v-if="selectedDatasetIds.length > 1">✓ 已选 {{ selectedDatasetIds.length }} 个数据集，将聚合为同一训练任务（统一类别命名空间，自动按已标注导出训练）</small>
        <small v-else>提示：仅显示已标注的数据集；点击卡片可多选聚合训练（单数据集训练保持不变）</small>
        <small v-if="!presetLibraryRootId && primaryLibraryId" class="lib-pick-info">
          主数据集归属算法库「{{ libraryNameOf(primaryLibraryId) }}」——训练产出会作为该库的新版本
        </small>
        <small v-else-if="!presetLibraryRootId && selectedDatasetIds.length && !primaryLibraryId" class="lib-pick-info warn">
          ⚠ 主数据集尚未归属算法库，产出库将由基座模型推断；建议先到「数据集」页点「去归属」补选
        </small>
        <small v-if="crossLibraryDatasets.length" class="lib-pick-info warn">
          ⚠ 所选数据集中有 {{ crossLibraryDatasets.length }} 个属于其它算法库（{{ crossLibraryDatasets.map((d: any) => libraryNameOf(datasetLibraryId(d)) || '未归属').join('、') }}），
          本次产出只归入主数据集所在的库
        </small>
        <label class="aggregate-opt">
          <input type="checkbox" v-model="aggregateIncomplete" :disabled="training" />
          自动带入同业务「未完成训练」数据集（阶段1.4 雪球：上一轮不合格的一并合训）
        </label>
      </div>
      <div class="form-group">
        <label>优化目标</label>
        <select v-model="trainForm.business" :disabled="training">
          <option v-for="b in businessScenes" :key="b.value" :value="b.value">{{ b.label }}</option>
        </select>
        <small v-if="autoAssignedBiz">
          ✓ 已按数据类别自动分配为「{{ autoAssignedBiz }}」—— 新模型只会与同优化目标上一代生产模型对比；识别不准可手动调整
        </small>
        <small v-else>选择优化目标：以通用检测为默认；选择「高精度质检 / 实时视频流」等目标后系统自动锁定对应黄金超参模板</small>
      </div>
      <div class="form-group pretrain-block">
        <label class="check-line">
          <input type="checkbox" v-model="pretrainEnabled" :disabled="training" />
          启用预训练权重（推荐）
        </label>
        <div v-if="pretrainEnabled" class="pretrain-source">
          <div class="source-radios">
            <label class="source-radio">
              <input type="radio" value="universal" v-model="pretrainSource" :disabled="training" />
              通用预训练
            </label>
            <label class="source-radio">
              <input type="radio" value="business" v-model="pretrainSource" :disabled="training" />
              指定业务模型（历史最佳）
            </label>
          </div>
          <!-- 通用预训练：内置 YOLO / 自定义导入 -->
          <div v-if="pretrainSource === 'universal'" class="form-group">
            <label class="model-label">预训练模型</label>
            <div class="model-selector">
              <div class="version-select">
                <label>YOLO版本</label>
                <select v-model="selectedVersion" :disabled="training">
                  <option v-for="(config, key) in yoloVersions" :key="key" :value="key">
                    {{ config.name }}
                  </option>
                </select>
              </div>
              <div class="size-select">
                <label>模型大小</label>
                <select v-model="selectedModelSize" :disabled="training">
                  <option v-for="model in currentVersionModels" :key="model.size" :value="model.size">
                    {{ model.size.toUpperCase() }} - {{ model.name }}
                  </option>
                </select>
              </div>
            </div>
            <div class="custom-model-block">
              <label class="custom-model-label">
                <button type="button" class="add-model-btn" @click="ptFile?.click()" :disabled="uploadingPt">
                  <span v-if="uploadingPt" class="loading-spinner"></span>
                  {{ uploadingPt ? '导入中...' : '添加' }}
                </button>
                自定义导入的模型
              </label>
              <select v-if="customModels.length" v-model="customSelectedPath" :disabled="training" @change="onSelectCustomModel">
                <option value="">-- 使用上方 YOLO 预训练模型 --</option>
                <option v-for="cm in customModels" :key="cm.path" :value="cm.path">
                  {{ cm.filename }} ({{ (cm.size / 1024 / 1024).toFixed(1) }} MB)
                </option>
              </select>
              <small v-else>尚未导入自定义模型，可点「添加」上传 .pt 权重文件</small>
              <input ref="ptFile" type="file" accept=".pt" class="pt-file-input" @change="handlePtFile" />
            </div>
            <small>当前选择: {{ trainForm.model_name }}</small>
          </div>
          <!-- 指定业务模型：算法库选择 + 跨代迁移入口二（§3.2.2） -->
          <div v-else class="form-group">
            <label>选择已有模型继续训练</label>
            <div class="select-with-refresh">
              <select v-model="trainForm.base_model_id" :disabled="loadingModels || training">
                <option value="">-- 请选择模型 --</option>
                <option v-for="model in baseModelOptions" :key="model.model_id" :value="model.model_id">
                  {{ model.model_id }} ({{ model.classes?.length || 0 }} 类)
                </option>
              </select>
              <button type="button" @click="loadModels" class="refresh-btn" :disabled="loadingModels">
                <span v-if="loadingModels" class="loading-spinner"></span>
                {{ loadingModels ? '加载中...' : '刷新' }}
              </button>
            </div>
            <small v-if="autoPickInfo" class="auto-pick-info">{{ autoPickInfo }}</small>
            <small v-else>将所选模型作为预训练权重继续训练；若与数据集优化目标匹配到已训练模型会自动默认选中</small>
            <!-- 2.4 跨代迁移训练（高级）：位于"启用预训练权重（指定业务模型）"内 -->
            <div class="migration-block">
              <label class="migration-toggle">
                <input type="checkbox" v-model="migrationEnabled" :disabled="training || !trainForm.base_model_id" />
                跨代迁移训练（高级）：训练到不同的 YOLO 架构
              </label>
              <small>从所选源模型迁移到下方目标架构（如 yolov5s → yolov8s）；系统自动 strict=False 复用 Backbone 层、lr0 降 10 倍</small>
              <div v-if="migrationEnabled" class="model-selector migration-selector">
                <div class="version-select">
                  <label>目标 YOLO 版本</label>
                  <select v-model="migrationTargetVersion" :disabled="training">
                    <option v-for="(config, key) in yoloVersions" :key="key" :value="key">
                      {{ config.name }}
                    </option>
                  </select>
                </div>
                <div class="size-select">
                  <label>目标模型大小</label>
                  <select v-model="migrationTargetSize" :disabled="training">
                    <option v-for="model in yoloVersions[migrationTargetVersion as keyof typeof yoloVersions]?.models || []" :key="model.size" :value="model.size">
                      {{ model.size.toUpperCase() }} - {{ model.name }}
                    </option>
                  </select>
                </div>
                <small v-if="selectedBaseArch && selectedBaseArch + '.pt' !== migrationTargetName" class="migration-hint">
  ⟶ 将执行跨代迁移：{{ selectedBaseArch }} → {{ migrationTargetName }}
</small>
<small v-else-if="selectedBaseArch && selectedBaseArch + '.pt' === migrationTargetName" class="migration-hint warn">
  ⚠ {{ migrationTargetName }} 与源模型架构一致，将按普通微调处理（不会执行迁移）
</small>
              </div>
            </div>
          </div>
        </div>
        <div v-else class="from-scratch-note">
          ⚠️ 已按「强制从零初始化」配置：<b>不使用任何预训练权重</b>，模型权重将随机初始化后开始训练。
        </div>
      </div>
      <p v-if="effectiveLibraryName" class="auto-pick-info">
        本次训练产出将归入算法库「{{ effectiveLibraryName }}」，训练完成后作为该库的新版本出现在库详情的「模型版本」中
      </p>
      <div class="form-group">
        <label>训练节点（GPU）</label>
        <select v-model="trainForm.gpu_index" :disabled="training || !cudaAvailable">
          <option :value="null">自动选择（推荐）</option>
          <option v-for="g in gpus" :key="g.index" :value="g.index">
            GPU {{ g.index }} — {{ g.name }}（共 {{ g.total_gb }}G，空闲 {{ g.free_gb }}G）
          </option>
        </select>
        <small v-if="!cudaAvailable">当前服务器未检测到 GPU，将使用 CPU 训练</small>
        <small v-else>选择用于训练的显卡；不确定时保持"自动选择"</small>
      </div>
      <!-- L2 智能推荐区（三层分档：基准/白银/黄金，点击应用） -->
      <div v-if="suggestGolden || suggestReason" class="l2-panel">
        <div class="l2-head">
          <span class="l2-title">L2 智能推荐 —— 系统按硬件与数据量自动计算，点击档位一键应用</span>
        </div>
        <!-- 三层分档卡片：基准（稳定优先）/ 白银（均衡）/ 黄金（效率最优） -->
        <div v-if="suggestTiers.length" class="tier-row">
          <button
            v-for="t in suggestTiers"
            :key="t.key"
            class="tier-card"
            :class="{ 'tier-active': activeTier === t.key }"
            @click="applyTier(t)"
            type="button"
          >
            <span class="tier-label">{{ t.label }}</span>
            <span class="tier-vals">Batch <b>{{ t.batch }}</b> · LR <b>{{ t.lr0 }}</b> · <b>{{ t.epochs }}</b>轮</span>
            <span class="tier-desc">{{ t.desc }}</span>
          </button>
        </div>
        <div v-if="suggestGolden" class="l2-body">
          <div class="l2-values">
            <div class="l2-value"><span>Batch</span><b>{{ suggestGolden.batch }}</b></div>
            <div class="l2-value"><span>LR</span><b>{{ suggestGolden.lr0 }}</b></div>
            <div class="l2-value"><span>Epochs</span><b>{{ suggestGolden.epochs }}</b></div>
            <div class="l2-value"><span>Img</span><b>{{ suggestParamsRes?.imgsz ?? 640 }}</b></div>
            <div class="l2-value"><span>Workers</span><b>{{ workersValue }}</b></div>
          </div>
          <div v-if="resourceEstimate.imageCount > 0" class="l2-resource">
            <span class="res-item">⏱ 预计耗时：约 {{ resourceEstimate.time }}</span>
            <span class="res-item">🖥 显存占用：约 {{ resourceEstimate.mem }}GB<template v-if="resourceEstimate.totalGpu"> / {{ resourceEstimate.totalGpu }}GB 可用</template></span>
          </div>
          <div class="l2-formulas">
            <div v-for="(f, k) in suggestGolden.formulas" :key="k" class="l2-formula">{{ f }}</div>
          </div>
        </div>
        <div v-if="suggestReason" class="l2-reason">{{ suggestReason }}</div>
      </div>
      <div class="form-group adv-block">
        <label class="adv-toggle">
          <input type="checkbox" v-model="showAdvanced" :disabled="training" />
          自定义参数（L3 专家微调：批次 / 轮数 / 学习率 / 优化器 / 训练增强）
        </label>
        <span v-if="suggesting" class="suggest-status">
          <span class="loading-spinner"></span> 分析环境并推荐参数...
        </span>
        <span v-else-if="suggestReason" class="suggest-status">✓ 已自动推荐（展开可手动微调）</span>
      </div>
      <div v-if="showAdvanced">
        <div class="grid" style="grid-template-columns: repeat(3, 1fr);">
          <div class="form-group">
            <label>训练轮数</label>
            <input v-model.number="trainForm.epochs" type="number" :disabled="training" />
          </div>
          <div class="form-group">
            <label>图片尺寸</label>
            <input v-model.number="trainForm.imgsz" type="number" :disabled="training" />
          </div>
          <div class="form-group">
            <label>批次大小</label>
            <input v-model.number="trainForm.batch" type="number" placeholder="-1 表示自动" :disabled="training" />
            <small>提示：-1 表示根据显存自动计算最佳值（推荐），或手动设置如 24、32、48</small>
          </div>
        </div>
        <div class="grid adv-grid" style="grid-template-columns: repeat(4, 1fr);">
        <div class="form-group">
          <label>学习率 lr0</label>
          <input v-model.number="trainForm.lr0" type="number" step="0.0001" :disabled="training" />
          <small>初始学习率，越大收敛越快但易震荡</small>
        </div>
        <div class="form-group">
          <label>优化器</label>
          <select v-model="trainForm.optimizer" :disabled="training">
            <option value="auto">auto（自动）</option>
            <option value="SGD">SGD</option>
            <option value="Adam">Adam</option>
            <option value="AdamW">AdamW</option>
          </select>
          <small>auto 按模型规格自动选择</small>
        </div>
        <div class="form-group">
          <label>权重衰减</label>
          <input v-model.number="trainForm.weight_decay" type="number" step="0.0001" :disabled="training" />
          <small>正则化，防止过拟合</small>
        </div>
        <div class="form-group">
          <label>早停轮数 patience</label>
          <input v-model.number="trainForm.patience" type="number" min="0" :disabled="training" />
          <small>验证指标连续 N 轮不涨则停止（0 = 关闭）</small>
        </div>
      </div>
      <div class="adv-enhance">
        <div class="form-group">
          <label class="check-line">
            <input type="checkbox" v-model="trainForm.recall_enabled" :disabled="training" />
            回忆集混训（增量训练防遗忘）
          </label>
          <small>从历史难例库抽取 {{ Math.round(recallRatio * 100) }}% 数据，防止模型遗忘旧场景。</small>
          <input id="recall_ratio" v-model.number="recallRatio" type="range" min="0.05" max="0.3" step="0.05" :disabled="training || !trainForm.recall_enabled" />
        </div>
        <div class="form-group">
          <label for="hard_ratio">困难样本比例 {{ trainForm.hard_sample_ratio }}</label>
          <input id="hard_ratio" v-model.number="trainForm.hard_sample_ratio" type="range" min="0" max="0.5" step="0.01" :disabled="training" />
          <small>从该模型困难样本库按比例抽样并入训练（0 = 关闭，默认）</small>
        </div>
        <div class="form-group">
          <label for="bg_ratio">空白样本抽样比例 {{ trainForm.background_sample_ratio }}</label>
          <input id="bg_ratio" v-model.number="trainForm.background_sample_ratio" type="range" min="0" max="0.5" step="0.01" :disabled="training" />
          <small>空白样本库（负样本）抽样比例（0 = 关闭，默认 5%）</small>
        </div>
      </div>
      <!-- §3.2.1 从零训练入口：仅 L3 高级模式底部提供 -->
      <div class="l3-block">
        <a class="scratch-link" href="javascript:void(0)" @click="confirmScratchInit">
          强制从零初始化
        </a>
      </div>
      </div>
      <div v-if="suggestReason" class="suggest-reason">{{ suggestReason }}</div>
      <button @click="startTraining" :disabled="training || preparingWeight || !trainForm.dataset_id">
        <span v-if="training || preparingWeight" class="loading-spinner"></span>
        {{ training ? '训练中...' : (preparingWeight ? '准备权重中（仅首次下载）...' : '开始训练') }}
      </button>
    </div>

    <div class="card">
      <h2>训练任务列表</h2>
      <button @click="loadJobs" :disabled="loadingJobs" class="secondary">
        <span v-if="loadingJobs" class="loading-spinner"></span>
        {{ loadingJobs ? '加载中...' : '刷新' }}
      </button>
      <div v-if="loadingJobs" class="loading-state">
        <span class="loading-spinner large"></span>
        <span>加载任务列表...</span>
      </div>
      <div v-else class="jobs-list">
        <div v-for="job in jobs" :key="job.job_id" class="job-item">
          <div class="job-head">
            <h3 class="job-name" @click="openJobFiles(job.job_id)" title="点击查看任务文件">{{ job.job_id }}</h3>
            <span class="job-pill" :class="'jp-' + jobPillKey(job.status)">{{ jobStatusText(job.status) }}</span>
          </div>
          <p>数据集: {{ job.dataset_id }}&nbsp;|&nbsp;模型: {{ job.model_name }}&nbsp;|&nbsp;{{ job.epochs }} 轮&nbsp;|&nbsp;Img {{ job.imgsz }}&nbsp;|&nbsp;Batch {{ job.batch }}</p>
          <p>节点: {{ job.gpu_index != null && job.gpu_index !== undefined ? 'GPU ' + job.gpu_index : '自动选择' }}&nbsp;|&nbsp;启动: {{ new Date(job.created_at).toLocaleString() }}</p>
          <p class="job-badges">
            <span v-if="job.gpu_abnormal" class="gpu-abnormal-badge" title="GPU 利用率持续过低，系统自动停止训练">⚠️ GPU 异常熔断</span>
            <span v-if="job.early_stopped" class="early-stop-badge" title="模型已收敛，系统自动提前停止训练以节省资源">✓ 已完成（早停）</span>
            <span v-if="job.migration?.enabled" class="migration-badge"
                  :title="`跨代迁移训练：${job.migration.source_architecture} → ${job.migration.target_architecture}`">
              🔄 迁移 {{ job.migration.source_architecture }} → {{ job.migration.target_architecture }}
            </span>
          </p>
          <div v-if="job.status === 'running'" class="job-progress">
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: jobProgress[job.job_id]?.percent + '%' }">
                <span v-if="jobProgress[job.job_id]?.has && jobProgress[job.job_id].percent > 0" class="progress-fill-text">
                  {{ jobProgress[job.job_id].percent }}%
                </span>
              </div>
            </div>
            <div class="progress-meta">
              <span v-if="jobProgress[job.job_id]?.has">
                Epoch {{ jobProgress[job.job_id].current }} / {{ jobProgress[job.job_id].total }}
              </span>
              <span v-else>等待训练启动...</span>
              <span class="progress-percent" v-if="jobProgress[job.job_id]?.has">{{ jobProgress[job.job_id].percent }}%</span>
            </div>
            <div class="progress-meta">
              <span class="progress-last" :class="{ warn: isStale(job.job_id) }">
                {{ lastActivityText(job.job_id) }}
              </span>
            </div>
            <!-- 迷你曲线（Sparkline）：实时渲染 mAP 走势，无需进入详情页 -->
            <div v-if="sparkPts(job)" class="spark-wrap">
              <span class="spark-label">mAP 走势</span>
              <svg class="spark" :viewBox="'0 0 100 26'" preserveAspectRatio="none">
                <polyline :points="sparkPts(job)" fill="none" stroke="#8e44ad" stroke-width="1.4" />
                <circle v-for="(pt, i) in sparkDots(job)" :key="i" :cx="pt.x" :cy="pt.y" r="1.5" fill="#6c2d82" />
              </svg>
            </div>
            <div v-if="jobProgress[job.job_id]?.lossLine" class="loss-line">{{ jobProgress[job.job_id].lossLine }}</div>
          </div>
          <p v-if="job.model_id">模型ID: {{ job.model_id }}
            <span v-if="modelMetaOf(job.model_id)">
              <span :class="'gk-badge gk-' + modelMetaOf(job.model_id).status">{{ gkStatusText(modelMetaOf(job.model_id)) }}</span>
              <span v-if="modelMetaOf(job.model_id).business" class="gk-version">业务: {{ bizLabel(modelMetaOf(job.model_id).business) }}</span>
              <span v-if="modelMetaOf(job.model_id).version" class="gk-version">版本: {{ modelMetaOf(job.model_id).version }}</span>
            </span>
          </p>
          <details v-if="gkReportOf(job.model_id)" class="gk-report">
            <summary>🛡️ 模型守门员评估报告</summary>
            <div class="gk-report-body">
              <div class="gk-report-text">{{ gkReportOf(job.model_id)!.report }}</div>
              <div v-if="gkReportOf(job.model_id)!.regressed_classes?.length" class="gk-regressed">
                退化类别: <span v-for="c in gkReportOf(job.model_id)!.regressed_classes" :key="c" class="gk-regressed-item">{{ c }}</span>
              </div>
            </div>
          </details>
          <p v-if="job.base_model_id">基础模型: {{ job.base_model_id }}</p>
          <p v-if="job.resume_count">续训次数: {{ job.resume_count }}</p>
          <p v-if="job.stopped_at">中断时间: {{ new Date(job.stopped_at).toLocaleString() }}</p>
          <p v-if="job.failed_at">失败时间: {{ new Date(job.failed_at).toLocaleString() }}</p>
          <p v-if="job.crashed_at">崩溃时间: {{ new Date(job.crashed_at).toLocaleString() }}</p>
          <p v-if="job.status === 'stopped'" style="color: #2ecc71; font-weight: bold;">✓ 可恢复训练</p>
          <p v-else-if="job.resume_count > 0" style="color: #2ecc71; font-weight: bold;">✓ 已恢复训练</p>
          <div class="job-actions">
            <!-- 主按钮（每行唯一，最高频操作） -->
            <button
              v-if="job.status === 'running'"
              @click="stopJob(job.job_id)" type="button" class="secondary"
              :disabled="stoppingJob === job.job_id"
            >
              <span v-if="stoppingJob === job.job_id" class="loading-spinner"></span>
              {{ stoppingJob === job.job_id ? '停止中...' : '停止' }}
            </button>
            <button
              v-else-if="job.status !== 'running' && (job.status === 'stopped' || job.status === 'failed' || job.status === 'crashed' || job.can_resume)"
              @click="resumeJob(job.job_id)" type="button" class="secondary"
              :disabled="resumingJob === job.job_id"
            >
              <span v-if="resumingJob === job.job_id" class="loading-spinner"></span>
              {{ resumingJob === job.job_id ? '恢复中...' : '继续训练' }}
            </button>
            <button
              v-else-if="job.status === 'failed' || job.status === 'crashed'"
              @click="retrainJob(job)" type="button" class="primary"
              :disabled="retrainingJob === job.job_id"
            >
              <span v-if="retrainingJob === job.job_id" class="loading-spinner"></span>
              {{ retrainingJob === job.job_id ? '重新训练中...' : '重新训练' }}
            </button>
            <!-- 次要操作收纳至 ⋯ 菜单（§2.2 操作收敛） -->
            <div class="job-more">
              <button type="button" class="ghost job-more-btn" @click="toggleJobMore(job.job_id)">⋯</button>
              <div v-if="jobMoreOpen === job.job_id" class="job-more-menu" @click.self="jobMoreOpen = ''">
                <button v-if="job.status === 'failed' || job.status === 'crashed'" type="button" class="job-more-item" @click="showFailureDialog(job)">失败原因</button>
                <button type="button" class="job-more-item" @click="openJobFiles(job.job_id)">查看文件</button>
                <router-link v-if="job.model_id" :to="`/models/${job.model_id}`" class="job-more-item link">查看模型</router-link>
                <button type="button" class="job-more-item danger" @click="deleteJobItem(job.job_id)" :disabled="deletingJob === job.job_id">
                  {{ deletingJob === job.job_id ? '删除中...' : '删除' }}
                </button>
              </div>
            </div>
          </div>
        </div>
        <div v-if="jobs.length === 0" class="empty-state">
          暂无训练任务
        </div>
      </div>
    </div>

    <!-- 任务文件浏览器弹窗 -->
    <div v-if="jobFiles" class="job-files-mask" @click.self="closeJobFiles">
      <div class="job-files-dialog">
        <div class="viewer-header">
          <h3>任务文件 - {{ jobFiles.job_id }}</h3>
          <button class="secondary viewer-close" @click="closeJobFiles">关闭</button>
        </div>
        <div class="viewer-section-title">文件夹结构</div>
        <div v-if="jobFilesLoading" class="loading-state">
          <span class="loading-spinner large"></span>
          <span>加载任务文件...</span>
        </div>
        <div v-else-if="jobTree" class="viewer-tree">
          <FolderTree
            :nodes="jobTree.children"
            :depth="0"
            :expanded="jobTreeExpanded"
            @toggle-dir="toggleJobDir"
            @open-file="openJobFile"
          />
        </div>
        <div v-else class="empty-state">暂无输出文件</div>
        <div v-if="jobFiles?.log_file" class="job-log-link">
          <button class="secondary" @click="openJobFile({ name: '训练日志', path: jobFiles.log_file, ext: '.log', type: 'file' })">
            📄 查看训练日志
          </button>
        </div>
      </div>
    </div>

    <!-- 文件内容预览弹窗 -->
    <div v-if="jobFilePreview" class="file-preview-mask" @click.self="closeJobFilePreview">
      <div class="file-preview-dialog">
        <div class="viewer-header">
          <h3>{{ jobFilePreview.name }}</h3>
          <button class="secondary viewer-close" @click="closeJobFilePreview">关闭</button>
        </div>
        <div v-if="jobFilePreviewIsImage" class="file-preview-img-wrap">
          <img :src="'/static/' + jobFilePreview.path" @error="handleFileImgError" />
        </div>
        <pre v-else class="file-preview-text">{{ jobFilePreviewContent || '加载中...' }}</pre>
      </div>
    </div>

    <!-- 训练失败原因弹窗（面向客户的可读说明 + 操作步骤） -->
    <div v-if="failureModal" class="job-files-mask" @click.self="closeFailureDialog">
      <div class="job-files-dialog failure-dialog">
        <div class="viewer-header">
          <h3>
            <span class="fail-emoji">🛑</span> 训练未成功
          </h3>
          <button class="secondary viewer-close" @click="closeFailureDialog">关闭</button>
        </div>
        <div class="failure-body">
          <div class="fail-title">{{ failureModal.title }}</div>
          <div class="fail-reason">{{ failureModal.reason }}</div>
          <div v-if="failureModal.failedAt" class="fail-time">失败时间：{{ failureModal.failedAt }}</div>
          <div v-if="failureModal.steps && failureModal.steps.length" class="fail-steps">
            <div class="fail-steps-title">建议这样操作：</div>
            <ol>
              <li v-for="(s, i) in failureModal.steps" :key="i">{{ s }}</li>
            </ol>
          </div>
          <details class="fail-log">
            <summary>查看原始错误信息（技术人员用）</summary>
            <pre class="fail-log-text">{{ failureModal.rawError || '（无详细日志）' }}</pre>
          </details>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { createTrainJob, listTrainJobs, deleteTrainJob, resumeTrainJob, getTrainJobTree, suggestTrainParams, inferBusiness, listGpus, prepareWeights, type TrainJobRequest } from '@/api/train'
import { listModels, uploadPretrainedPt, listCustomModels } from '@/api/models'
import { listDatasets } from '@/api/datasets'
import { subscribeLogsSSE, pollLogsTail, getLogLines } from '@/api/logs'
import FolderTree from '@/components/FolderTree.vue'
import { useLogStore } from '@/store/logs'
import { showAlert, showConfirm } from '@/composables/useDialog'

const logStore = useLogStore()

const trainForm = ref<TrainJobRequest>({
  dataset_id: '',
  version: 'v1',
  model_name: 'yolov8n.pt',
  epochs: 10,
  imgsz: 640,
  batch: 32,  // 默认批次 32；-1 表示自动（后端归一化也为 32）
  base_model_id: undefined,
  business: 'general',
  gpu_index: null,  // null = 自动选择 GPU
  // 1.5/1.6 训练增强：样本池抽样 + 回忆集混训
  hard_sample_ratio: 0,   // 困难样本抽样默认关闭（0），用户按需开启
  background_sample_ratio: 0.05,
  // 2.4 跨代迁移训练：目标架构（如 yolov8s.pt），与源模型架构不同时自动迁移
  migration_target_model: undefined,
  recall_enabled: false  // 回忆集混训默认关闭，需在 L3 专家微调中手动开启
})

// 回忆集混训抽取比例（L3 滑块 5%~30%，默认 10%）
const recallRatio = ref(0.1)

// 阶段1.3（训练任务聚合）：多选数据集 → 聚合为同一训练任务（第一个选中者为主数据集）
const selectedDatasetIds = ref<string[]>([])
// 阶段1.4（雪球）：自动带入同业务「未完成训练」数据集
const aggregateIncomplete = ref(false)
const isDsSelected = (id: string) => selectedDatasetIds.value.includes(id)

// 业务/算法场景：模型按业务隔离存储与守门员对比
const businessScenes = [
  { value: 'general', label: '通用目标检测' },
  { value: 'pedestrian', label: '行人检测' },
  { value: 'vehicle', label: '车辆 / 车牌' },
  { value: 'defect', label: '工业缺陷检测' },
  { value: 'package', label: '包裹 / 物流' }
]
const bizLabel = (v: string) => businessScenes.find(b => b.value === v)?.label || v

// 任务状态中文文案（§2.2 任务列表状态色块）
const JOB_STATUS_TEXT: Record<string, string> = {
  running: '训练中', queued: '排队中', completed: '已完成',
  failed: '失败', crashed: '崩溃', stopped: '已停止',
  ready: '就绪', evaluating: '评估中'
}
const jobStatusText = (s: string) => JOB_STATUS_TEXT[s] || s
// 四色状态映射：done(绿 完成) / run(蓝 训练·排队) / fail(红 失败·崩溃) / stop(橙 已停止)
const jobPillKey = (status: string): string => {
  if (status === 'completed' || status === 'ready') return 'done'
  if (status === 'failed' || status === 'crashed') return 'fail'
  if (status === 'stopped') return 'stop'
  return 'run'
}

// 任务卡守门员状态：通过 job.model_id 在模型列表中找到对应模型入库状态
const modelMetaOf = (modelId: string) => {
  return models.value.find((m: any) => m.model_id === modelId)
}

// ===== 任务卡 Sparkline（§2.2：mAP 走势迷你曲线，无需进详情页） =====
// 数据源：任务/关联模型的 training_metrics.training_history.metrics_mAP50_95
const sparkHistory = (job: any): number[] => {
  const th = job?.training_metrics?.training_history
    || job?.training_history
    || modelMetaOf(job.model_id)?.training_metrics?.training_history
  const arr = th?.metrics_mAP50_95 || th?.metrics_mAP50 || []
  return (Array.isArray(arr) ? arr : []).filter((v: any) => typeof v === 'number' && isFinite(v))
}
const sparkRange = (vals: number[]) => {
  const min = Math.min(...vals)
  const max = Math.max(...vals)
  return { min, max, span: (max - min) || 1 }
}
const sparkPts = (job: any): string => {
  const vals = sparkHistory(job)
  if (vals.length < 2) return ''
  const { min, span } = sparkRange(vals)
  return vals.map((v, i) => {
    const x = (i / (vals.length - 1)) * 100
    const y = 24 - ((v - min) / span) * 22
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}
const sparkDots = (job: any) => {
  const vals = sparkHistory(job)
  if (vals.length < 2) return []
  const { min, span } = sparkRange(vals)
  return vals.map((v, i) => ({
    x: (i / (vals.length - 1)) * 100,
    y: 24 - ((v - min) / span) * 22
  }))
}

// ⋯ 更多操作菜单（§2.2 操作收敛）：每次仅展开一个任务
const jobMoreOpen = ref('')
const toggleJobMore = (jobId: string) => {
  jobMoreOpen.value = jobMoreOpen.value === jobId ? '' : jobId
}
const gkReportOf = (modelId: string) => {
  return modelMetaOf(modelId)?.gatekeeper
}
const gkStatusText = (meta: any) => {
  switch (meta?.status) {
    case 'production_ready': return meta.override ? '生产就绪（人工覆盖）' : '✅ 生产就绪'
    case 'rejected': return '❌ 已淘汰（未入生产）'
    case 'superseded': return '⚠️ 已退役（被新版本替代）'
    case 'training': return '⏳ 训练中'
    case 'evaluating': return '🔍 评估中'
    default: return meta?.status || '未知状态'
  }
}

// 训练节点：服务器 GPU 列表
const gpus = ref<{ index: number; name: string; total_gb: number; free_gb: number }[]>([])
const cudaAvailable = ref(true)

const loadGpus = async () => {
  try {
    const res = await listGpus()
    cudaAvailable.value = res.cuda_available
    gpus.value = res.gpus || []
  } catch (e) {
    // 旧后端无此接口时静默降级：保持"自动选择"
    cudaAvailable.value = true
    gpus.value = []
  }
}

// 高级参数：L3 专家微调默认折叠；选中数据集+模型后 L2 会自动推荐并回填，展开可手动微调
const showAdvanced = ref(false)
// §3.2.1 强制从零初始化：仅 L3 高级模式底部提供入口，需二次确认；
// 确认后关闭预训练权重开关（不提交新字段，后端仍以 model_name 作为架构参考）
const confirmScratchInit = async () => {
  if (training.value) return
  const ok = await showConfirm(
    '强制从零初始化后，将不使用任何预训练权重，模型权重随机初始化开始训练。\n收敛更慢、最终精度可能偏低，仅当迁移/微调出现问题时使用。确定继续？',
    '从零初始化'
  )
  if (ok) {
    pretrainEnabled.value = false
    pretrainSource.value = 'universal'
    migrationEnabled.value = false
    trainForm.value.base_model_id = undefined
    trainForm.value.migration_target_model = undefined
  }
}
const suggesting = ref(false)
const suggestReason = ref('')
// 2.2 黄金参数（L2 只读展示）：公式结果 + 推荐参数快照
const suggestGolden = ref<any | null>(null)
const suggestParamsRes = ref<any | null>(null)
// 三层分档（基准/白银/黄金）：点击某档即应用其 批次/学习率/轮数 到表单
const suggestTiers = ref<any[]>([])
const activeTier = ref('golden')
const applyTier = (tier: any) => {
  if (!tier) return
  activeTier.value = tier.key
  trainForm.value.batch = tier.batch
  trainForm.value.epochs = tier.epochs
  if (trainForm.value.lr0 !== undefined) trainForm.value.lr0 = tier.lr0
  alert('已应用「' + tier.label + '」档参数（可继续手动微调）')
}
let suggestTimer: any = null

// ===== L2 资源预估（§2.1 智能推荐区：Workers / 预计耗时 / 显存占用） =====
// 显存锚点（yolov8n@640@batch32 ≈ 4GB，按规格/批次/尺寸线性外推）
const SCALE_MEM: Record<string, number> = { t: 0.7, n: 1, s: 1.5, m: 2.5, b: 2, c: 3, l: 4, e: 8, x: 6 }
const resourceEstimate = computed(() => {
  const res = suggestParamsRes.value
  const cnt: number = res?.dataset?.image_count ?? 0
  const epochs = trainForm.value.epochs || 10
  const batch = trainForm.value.batch && trainForm.value.batch > 0 ? trainForm.value.batch : 32
  const imgsz = trainForm.value.imgsz || 640
  const m = /(?:yolo)(\d+)([a-z]*?)([tnslmxbce])\.pt/i.exec(trainForm.value.model_name || '') || []
  const scale = (m[3] || 'n').toLowerCase()
  const isGpu = res?.device?.type === 'cuda'
  // 显存 ≈ 锚点 × 规格系数 × (batch/32) × (imgsz/640)²
  const mem = Math.max(1, Math.round((4 * (SCALE_MEM[scale] ?? 1) * (batch / 32) * Math.pow(imgsz / 640, 2)) * 10) / 10)
  // 耗时 ≈ 轮数 × 每轮步数 × 单步耗时（GPU≈0.18s / CPU≈1.2s）
  const steps = Math.max(1, Math.ceil(cnt / batch))
  const secPerStep = isGpu ? 0.18 : 1.2
  return {
    imageCount: cnt,
    time: fmtDuration(steps * epochs * secPerStep),
    mem,
    totalGpu: isGpu ? (res?.device?.vram_gb ?? 0) : 0
  }
})
function fmtDuration(sec: number): string {
  const s = Math.max(1, Math.round(sec))
  if (s >= 3600) return `约 ${Math.floor(s / 3600)} 小时 ${Math.round((s % 3600) / 60)} 分`
  if (s >= 60) return `约 ${Math.round(s / 60)} 分钟`
  return `约 ${s} 秒`
}
// Workers：YOLO 数据加载工作进程数（后端未直接返回，按图片规模推导）
const workersValue = computed(() => {
  const cnt = resourceEstimate.value.imageCount
  return Math.min(8, Math.max(2, Math.floor(cnt / 256) || 2))
})

// ===== 预训练权重开关（§3.2.1 从零训练 / §3.2.2 跨代迁移入口二） =====
// 勾选「启用预训练权重」→ 二选一来源：
//   universal（通用预训练）= 内置 YOLO 权重 / 自定义导入 .pt
//   business（指定业务模型）= 从算法库历史最佳继续训练（含跨代迁移入口）
// 关闭开关 = 强制从零初始化（随机权重，无任何预训练）
const pretrainEnabled = ref(true)
const pretrainSource = ref<'universal' | 'business'>('universal')
// 从算法库详情页「发起训练」跳转时带入的库 id（?library_root_id=xxx）：
// 优先级最高；未带入时由主数据集的归属推导（见 effectiveLibraryRootId）
const presetLibraryRootId = ref('')
const isBusinessPretrain = computed(
  () => pretrainEnabled.value && pretrainSource.value === 'business'
)

const models = ref<any[]>([])
// 数据集归属的算法库（库根记录 id）：导入页必选；老数据没写过该字段时退化到它挂的模型 id。
// 悬空 id（模型记录已删除）不认，交给后端按基座模型推导，避免凭空生成一个新库
const datasetLibraryId = (ds: any): string => {
  const id = ds?.library_root_id || ds?.model_id || ''
  if (!id) return ''
  return models.value.some((m: any) => (m.library_root_id || m.model_id) === m.model_id && m.model_id === id)
    ? id : ''
}
const libraryNameOf = (id: string): string => {
  if (!id) return ''
  const m = models.value.find((x: any) => x.model_id === id)
  return m?.display_name || m?.model_code || id
}
// 主数据集（trainForm.dataset_id）归属的库
const primaryLibraryId = computed(() => {
  const main = datasets.value.find((d: any) => d.dataset_id === trainForm.value.dataset_id)
  return main ? datasetLibraryId(main) : ''
})
// 本次产出的归属库：库详情页带入的库优先，其次按主数据集归属推导
const effectiveLibraryRootId = computed(() => presetLibraryRootId.value || primaryLibraryId.value)
const effectiveLibraryName = computed(() => libraryNameOf(effectiveLibraryRootId.value))
// 多数据集聚合训练时，归属以主数据集为准；列出属于其它库的数据集用于页面提示
const crossLibraryDatasets = computed(() => {
  const root = effectiveLibraryRootId.value
  if (!root || selectedDatasetIds.value.length <= 1) return []
  return datasets.value.filter(
    (d: any) => selectedDatasetIds.value.includes(d.dataset_id) && datasetLibraryId(d) !== root
  )
})
// 可作为"继续训练的已有模型"的选项：排除算法库的空白占位记录（它没有权重文件，选了必然报错）
const baseModelOptions = computed(() => models.value.filter((m: any) => !m.empty))
const datasets = ref<any[]>([])
const loadingDatasets = ref(false)
const loadingModels = ref(false)
const loadingJobs = ref(false)
const stoppingJob = ref<string | null>(null)
const resumingJob = ref<string | null>(null)
const deletingJob = ref<string | null>(null)
const retrainingJob = ref<string | null>(null)

// YOLO 版本配置
const yoloVersions = {
  v5: { 
    name: 'YOLOv5', 
    models: [
      { size: 'n', name: 'Nano (最快)' },
      { size: 's', name: 'Small' },
      { size: 'm', name: 'Medium' },
      { size: 'l', name: 'Large' },
      { size: 'x', name: 'XLarge (最准)' }
    ]
  },
  v8: { 
    name: 'YOLOv8', 
    models: [
      { size: 'n', name: 'Nano (最快)' },
      { size: 's', name: 'Small' },
      { size: 'm', name: 'Medium' },
      { size: 'l', name: 'Large' },
      { size: 'x', name: 'XLarge (最准)' }
    ]
  },
  v9: { 
    name: 'YOLOv9', 
    models: [
      { size: 't', name: 'Tiny (最快)' },
      { size: 's', name: 'Small' },
      { size: 'm', name: 'Medium' },
      { size: 'c', name: 'Compact' },
      { size: 'e', name: 'Extended (最准)' }
    ]
  },
  v10: { 
    name: 'YOLOv10', 
    models: [
      { size: 'n', name: 'Nano (最快)' },
      { size: 's', name: 'Small' },
      { size: 'm', name: 'Medium' },
      { size: 'b', name: 'Balanced' },
      { size: 'l', name: 'Large' },
      { size: 'x', name: 'XLarge (最准)' }
    ]
  },
  v11: { 
    name: 'YOLO11', 
    models: [
      { size: 'n', name: 'Nano (最快)' },
      { size: 's', name: 'Small' },
      { size: 'm', name: 'Medium' },
      { size: 'l', name: 'Large' },
      { size: 'x', name: 'XLarge (最准)' }
    ]
  }
}

const selectedVersion = ref<keyof typeof yoloVersions>('v8')
const selectedModelSize = ref('n')
// 2.4 跨代迁移训练：目标架构选择（仅微调模式展示）+ 源架构提示
const migrationEnabled = ref(false)
const migrationTargetVersion = ref<keyof typeof yoloVersions>('v8')
const migrationTargetSize = ref('n')
const migrationTargetName = computed(() => {
  const v = migrationTargetVersion.value === 'v11' ? 'yolo11' : `yolo${migrationTargetVersion.value}`
  return `${v}${migrationTargetSize.value}.pt`
})

// 自定义导入的 .pt 预训练模型
const customModels = ref<any[]>([])
const customSelectedPath = ref('')
const uploadingPt = ref(false)
const ptFile = ref<HTMLInputElement | null>(null)

// 当前版本的模型列表
const currentVersionModels = computed(() => {
  return yoloVersions[selectedVersion.value]?.models || []
})

// 生成模型名称
const generateModelName = () => {
  const version = selectedVersion.value
  const size = selectedModelSize.value
  if (version === 'v11') {
    return `yolo11${size}.pt`
  }
  return `yolo${version}${size}.pt`
}

// 监听版本和大小变化，更新 model_name（或迁移目标架构）
watch([selectedVersion, selectedModelSize], () => {
  if (pretrainSource.value === 'universal') {
    trainForm.value.model_name = generateModelName()
  }
})
watch([migrationTargetVersion, migrationTargetSize], () => {
  if (migrationEnabled.value) {
    trainForm.value.migration_target_model = migrationTargetName.value
  }
})
// 指定业务模型：已选源模型的架构（如 yolov5s），用于迁移提示；computed 保证自动选中也能生效
const selectedBaseArch = computed(() => {
  if (!isBusinessPretrain.value || !trainForm.value.base_model_id) return ''
  const m = models.value.find((x: any) => x.model_id === trainForm.value.base_model_id)
  if (m?.architecture_name) return `${m.architecture_name}${m.model_scale || ''}`
  // 兜底：从血缘/基座文件名提取架构（如 yolov8n.pt → yolov8n），与后端 _model_architecture 一致
  const src = m?.lineage?.base_model || m?.model_name || m?.filename || ''
  return (/(yolov?\d+[a-z]*?)([nslmxbce])\.pt/i.exec(src) || []).slice(1, 3).join('') || ''
})
// 退出"指定业务模型"来源（切回通用预训练 / 关闭预训练）时，清空源模型与迁移设置（§3.2.2）
watch(isBusinessPretrain, (on) => {
  if (!on) {
    migrationEnabled.value = false
    trainForm.value.migration_target_model = undefined
    trainForm.value.base_model_id = undefined
  }
})

// 监听版本变化，重置模型大小为默认值
watch(selectedVersion, (newVersion) => {
  const models = yoloVersions[newVersion]?.models
  if (models && models.length > 0) {
    selectedModelSize.value = models[0].size
  }
})

// ===== 自动推荐训练参数 =====
// 根据硬件（GPU显存）与数据集规模推荐参数并回填表单；支持手动修改
const fetchSuggest = async () => {
  if (!trainForm.value.dataset_id) return
  suggesting.value = true
  try {
    const res = await suggestTrainParams(
      trainForm.value.dataset_id,
      trainForm.value.version || 'v1',
      isBusinessPretrain.value ? trainForm.value.base_model_id : undefined,
      // 2.4 迁移训练：按目标架构（大小）套用黄金参数 Batch/LR 公式
      isBusinessPretrain.value
        ? (migrationEnabled.value ? trainForm.value.migration_target_model : undefined)
        : trainForm.value.model_name
    )
    if (res?.params) {
      const p = res.params
      trainForm.value.epochs = p.epochs
      trainForm.value.imgsz = p.imgsz
      trainForm.value.batch = p.batch
      trainForm.value.lr0 = p.lr0
      trainForm.value.optimizer = p.optimizer
      trainForm.value.weight_decay = p.weight_decay
      trainForm.value.patience = p.patience
      suggestReason.value = res.reason || ''
      // 2.2：黄金参数 L2 只读展示（CPU/不可用时为 null）
      suggestGolden.value = res.golden || null
      // 三层分档（基准/白银/黄金）：后端 golden.tiers 为计算快照
      const tiersObj = res.golden?.tiers || null
      suggestTiers.value = tiersObj ? Object.values(tiersObj) : []
      // 表单当前值对应哪一档（batch 最接近者），用于默认高亮
      if (suggestTiers.value.length) {
        const cur = p.batch
        activeTier.value = suggestTiers.value
          .slice().sort((a: any, b: any) =>
            Math.abs(a.batch - cur) - Math.abs(b.batch - cur))[0]?.key || 'golden'
      }
      suggestParamsRes.value = res.params || null
    }
  } catch (e: any) {
    // 推荐失败不打断用户操作，保留当前值
    console.error('获取推荐参数失败:', e)
    const detail = e?.response?.data?.detail || e?.message || '网络异常'
    suggestReason.value = `自动推荐失败（已保留当前参数）：${detail}`
  } finally {
    suggesting.value = false
  }
}

// 数据集/版本/模型/微调开关变化 → 防抖自动推荐
watch(
  [
    () => trainForm.value.dataset_id,
    () => trainForm.value.version,
    () => trainForm.value.model_name,
    () => trainForm.value.base_model_id,
    // 2.4 迁移目标架构变化 → 重新推荐（黄金参数按目标模型大小计算）
    () => trainForm.value.migration_target_model,
    isBusinessPretrain
  ],
  () => {
    clearTimeout(suggestTimer)
    suggestTimer = setTimeout(fetchSuggest, 400)
  }
)

// ===== 业务场景自动分配 =====
// 选中数据集后，按数据集类别名（data.yaml names）自动推断业务/算法类型并回填，
// 无需手动选择；识别不准时仍可手动切换下拉。切换数据集/数据集版本会重新推断。
const autoAssignedBiz = ref('')
const autoAssignBusiness = async () => {
  if (!trainForm.value.dataset_id) {
    autoAssignedBiz.value = ''
    return
  }
  try {
    const res = await inferBusiness(trainForm.value.dataset_id, trainForm.value.version || 'v1')
    if (res?.business) {
      trainForm.value.business = res.business
      autoAssignedBiz.value = bizLabel(res.business)
    } else {
      autoAssignedBiz.value = ''
    }
  } catch (e: any) {
    console.error('自动分配业务场景失败:', e)
    autoAssignedBiz.value = ''
  }
}
watch(
  [() => trainForm.value.dataset_id, () => trainForm.value.version],
  () => autoAssignBusiness()
)

// ===== 微调模型自动匹配 =====
// 勾选「基于已有模型微调」时，按当前数据集的业务/算法类型（由类别名自动推断，
// 无需数据集命名规则）自动默认选中同业务已训练的生产模型（版本最高者）；
// 若无匹配则保持空，等待手动选择。
const pickScope = ref('')          // 已自动匹配过的 数据集:版本:业务 范围，避免重复请求
const autoPickInfo = ref('')       // 自动匹配结果提示
let autoPickTimer: any = null
const numOfVersion = (ver: any) => {
  const s = String(ver ?? '')
  if (!s.startsWith('v')) return 0
  const n = parseFloat(s.slice(1))
  return isNaN(n) ? 0 : n
}
const autoPickBaseModel = async () => {
  const datasetId = trainForm.value.dataset_id
  const version = trainForm.value.version || 'v1'
  if (!isBusinessPretrain.value || !datasetId) return
  const biz = trainForm.value.business || (await (async () => {
    try {
      const res = await inferBusiness(datasetId, version)
      return res?.business || ''
    } catch (e: any) {
      console.error('推断业务失败:', e)
      return ''
    }
  })())
  if (!biz) return
  if (pickScope.value === `${datasetId}:${version}:${biz}`) return
  pickScope.value = `${datasetId}:${version}:${biz}`
  if (!models.value.length) {
    try { await loadModels() } catch (e) { return }
  }
  // 从算法库详情页进来时，只在本库的版本里挑基座（产出会归入本库，不该借别的库的权重）
  const inScope = (m: any) => presetLibraryRootId.value
    ? (m.library_root_id || m.model_id) === presetLibraryRootId.value
    : m.business === biz
  const candidates = models.value.filter(
    (m: any) => !m.empty && inScope(m) && m.status === 'production_ready'
  ).sort((a: any, b: any) => numOfVersion(b.version) - numOfVersion(a.version))
  if (candidates.length) {
    trainForm.value.base_model_id = candidates[0].model_id
    autoPickInfo.value = `✓ 已自动匹配${presetLibraryRootId.value ? '本算法库' : `同业务「${bizLabel(biz)}」`}的已训练模型 ${candidates[0].model_id}（可改选其它）`
  } else {
    autoPickInfo.value = presetLibraryRootId.value
      ? '本算法库暂无可选为基座的版本，将直接以官方预训练权重训练并归入本库'
      : `同业务「${bizLabel(biz)}」暂无已训练的生产模型，可手动选择任意模型`
  }
}
watch(
  [isBusinessPretrain, () => trainForm.value.dataset_id, () => trainForm.value.version],
  () => {
    clearTimeout(autoPickTimer)
    autoPickTimer = setTimeout(autoPickBaseModel, 300)
  }
)

const loadDatasets = async () => {
  loadingDatasets.value = true
  try {
    const result = await listDatasets()
    // 只显示已标注的数据集
    datasets.value = (result.datasets || []).filter((d: any) => d.annotated)
  } catch (error: any) {
    console.error('加载数据集失败:', error)
    alert('加载数据集失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    loadingDatasets.value = false
  }
}

// 获取数据集准备状态
const getDatasetStatus = (dataset: any) => {
  if (dataset.status === 'prepared') return '已准备'
  if (dataset.status === 'uploaded') return '已上传'
  return dataset.status || '未知'
}

// 检查数据集是否已准备（允许 prepared 和 uploaded 状态）
const isDatasetPrepared = (dataset: any) => {
  return dataset.status === 'prepared' || dataset.status === 'uploaded'
}

// 卡片式选择数据集（支持多选聚合：1.3）
const selectDataset = (ds: any) => {
  if (!isDatasetPrepared(ds)) {
    alert('该数据集尚未准备，请先在"数据集"页面执行"准备"操作')
    return
  }
  const idx = selectedDatasetIds.value.indexOf(ds.dataset_id)
  if (idx >= 0) {
    selectedDatasetIds.value.splice(idx, 1)
    // 若取消的是主数据集，自动顺延首位为新的主数据集
    if (trainForm.value.dataset_id === ds.dataset_id) {
      trainForm.value.dataset_id = selectedDatasetIds.value[0] || ''
    }
  } else {
    selectedDatasetIds.value.push(ds.dataset_id)
    if (!trainForm.value.dataset_id) {
      trainForm.value.dataset_id = selectedDatasetIds.value[0]
    }
  }
}

// 加载已上传的自定义模型；若客户上传过模型（且尚未手动选择），默认优先使用它，
// 未上传任何模型时才保持内置预训练模型（首次使用时镜像下载并缓存）
let autoPickedCustom = false
const loadCustomModels = async () => {
  try {
    const result = await listCustomModels()
    customModels.value = result.models || []
    // 默认用客户上传的模型（仅页面进入第一次生效，后续尊重用户手动选择）
    if (!autoPickedCustom && !isBusinessPretrain.value && customModels.value.length) {
      autoPickedCustom = true
      customSelectedPath.value = customModels.value[0].path
      trainForm.value.model_name = customModels.value[0].path
    }
    // 若当前选中了已存在的自定义模型，保持状态
    if (customSelectedPath.value && !customModels.value.some(cm => cm.path === customSelectedPath.value)) {
      customSelectedPath.value = ''
      trainForm.value.model_name = generateModelName()
    }
  } catch (error: any) {
    console.error('加载自定义模型失败:', error)
  }
}

// 处理用户选择的 .pt 文件并上传
const handlePtFile = async (event: any) => {
  const file = event.target.files?.[0]
  if (!file) return
  if (!file.name.endsWith('.pt')) {
    alert('仅支持 .pt 格式的模型权重文件')
    return
  }
  uploadingPt.value = true
  try {
    const result = await uploadPretrainedPt(file)
    await loadCustomModels()
    // 自动选中刚上传的模型
    customSelectedPath.value = result.path
    trainForm.value.model_name = result.path
    alert('模型导入成功: ' + result.filename)
  } catch (error: any) {
    alert('模型导入失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    uploadingPt.value = false
    event.target.value = ''
  }
}

// 选择自定义模型
const onSelectCustomModel = () => {
  if (customSelectedPath.value) {
    trainForm.value.model_name = customSelectedPath.value
  } else {
    trainForm.value.model_name = generateModelName()
  }
}

const router = useRouter()
const route = useRoute()
const training = ref(false)
// 训练前准备权重（镜像下载缓存）的进行中标记，用于按钮提示
const preparingWeight = ref(false)
const currentJobId = ref<string | null>(null)
const jobs = ref<any[]>([])
let eventSource: EventSource | null = null
let pollInterval: any = null

// 每个任务的独立进度（key: job_id）
const jobProgress = ref<Record<string, { current: number; total: number; percent: number; has: boolean; lossLine: string; lastActivity: number }>>({})
// 每个任务独立的轮询定时器
const jobPollers = new Map<string, any>()
// 每个任务独立的日志偏移量
const jobOffsets = new Map<string, number>()
// 每秒跳动一次，用于实时显示"最后更新 X 秒前"（判断训练是否卡住）
const nowTick = ref(Date.now())
let nowTimer: any = null

// 去除 ANSI 转义码
const stripAnsi = (text: string): string => text.replace(/\x1B\[[0-9;]*[mK]/g, '')

const startTraining = async () => {
  if (!trainForm.value.dataset_id) {
    alert('请选择数据集')
    return
  }
  
  // 检查数据集是否已准备
  const selectedDataset = datasets.value.find(ds => ds.dataset_id === trainForm.value.dataset_id)
  if (selectedDataset && selectedDataset.status === 'uploaded') {
    alert('数据集尚未准备（prepare）。请先在数据集上传页面执行"准备"操作后再开始训练。')
    return
  }
  
  if (isBusinessPretrain.value && !trainForm.value.base_model_id) {
    alert('请选择要用作预训练权重的业务模型')
    return
  }
  
  training.value = true
  try {
    // 训练前先确保预训练权重已本地化（镜像下载缓存），避免启动阶段联网下载卡顿。
    // 自定义导入/微调模型本就是本地路径，命中缓存秒返；下载失败不阻断（训练脚本内仍有镜像/官方兜底）。
    // 2.4 跨代迁移：预下载"目标架构"权重（迁移训练将以目标架构建模）
    const toPrepare =
      (!isBusinessPretrain.value && trainForm.value.model_name?.endsWith('.pt') &&
       !trainForm.value.model_name.includes('/') && !trainForm.value.model_name.includes('\\'))
        ? trainForm.value.model_name
        : (migrationEnabled.value && trainForm.value.migration_target_model?.endsWith('.pt')
            ? trainForm.value.migration_target_model : undefined)
    if (toPrepare) {
      preparingWeight.value = true
      try {
        await prepareWeights(toPrepare)
      } catch (e: any) {
        console.warn('权重预下载失败，将由训练脚本兜底:', e?.message || e)
      } finally {
        preparingWeight.value = false
      }
    }
    const params: TrainJobRequest = {
      ...trainForm.value,
      base_model_id: isBusinessPretrain.value ? trainForm.value.base_model_id : undefined,
      // 2.4 跨代迁移训练：仅指定业务模型且开启迁移时透传目标架构
      migration_target_model: (isBusinessPretrain.value && migrationEnabled.value)
        ? trainForm.value.migration_target_model : undefined,
      // 阶段1.3/1.4：多选聚合 + 未完成回收
      dataset_ids: selectedDatasetIds.value.length > 1 ? [...selectedDatasetIds.value] : undefined,
      aggregate_incomplete: aggregateIncomplete.value ? true : undefined,
      // 算法库归属：库详情页发起训练时带入，否则由主数据集的归属推导，使产出成为该库的新版本
      library_root_id: effectiveLibraryRootId.value || undefined
    }
    
    const result = await createTrainJob(params)
    currentJobId.value = result.job_id
    // 阶段1.3：聚合训练结果提示
    if (result.aggregated && result.aggregation) {
      alert(
        `训练任务已启动！聚合 ${result.aggregation.dataset_count} 个数据集、` +
        `${result.aggregation.unified_names.length} 类统一命名空间、` +
        `${result.aggregation.total_images} 张图片进行训练。`
      )
    } else {
      const lf = result.label_filter
      if (lf && lf.dropped && lf.dropped.length > 0) {
        alert(
          '训练任务已启动！\n\n' +
          '已按微调模型标签字典过滤数据集类别：\n' +
          `剔除（模型字典外）: ${lf.dropped.join('、')}\n` +
          `保留训练: ${lf.kept.join('、')}` +
          (lf.model_code ? `\n模型: ${lf.model_code}` : '')
        )
      } else {
        alert('训练任务已启动!')
      }
    }
    
    // 开始订阅日志
    logStore.clearLogs()
    subscribeToLogs(result.job_id)
    
    loadJobs()
    
    // 重置表单（保留主数据集；聚合/雪球开关复位）
    const keepMain = trainForm.value.dataset_id
    selectedDatasetIds.value = keepMain ? [keepMain] : []
    aggregateIncomplete.value = false
    trainForm.value = {
      dataset_id: keepMain,
      version: 'v1',
      model_name: 'yolov8n.pt',
      epochs: 10,
      imgsz: 640,
      batch: 32,
      base_model_id: undefined,
      business: trainForm.value.business || 'general',
      gpu_index: null,
      hard_sample_ratio: 0,
      background_sample_ratio: 0.05,
      migration_target_model: undefined,
      recall_enabled: false
    }
    pretrainEnabled.value = true
    pretrainSource.value = 'universal'
    recallRatio.value = 0.1
    migrationEnabled.value = false
  } catch (error: any) {
    alert('启动失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    training.value = false
  }
}

// 重新训练：复用失败/中断任务的原参数，直接重新创建训练任务（"-1 自动"会由后端归一化）
const retrainJob = async (job: any) => {
  const prevScrollY = window.scrollY
  if (!(await showConfirm(`用相同参数重新创建训练任务？\n数据集: ${job.dataset_id} / 模型: ${job.model_name} / 轮数: ${job.epochs}`))) return

  retrainingJob.value = job.job_id
  try {
    const params: TrainJobRequest = {
      dataset_id: job.dataset_id,
      version: job.version || 'v1',
      model_name: job.model_name || 'yolov8n.pt',
      epochs: job.epochs ?? 10,
      imgsz: job.imgsz ?? 640,
      batch: job.batch ?? 32,
      base_model_id: job.base_model_id || undefined,
      // 2.4 迁移训练任务重跑保持迁移语义（源模型 + 目标架构）
      ...(job.migration?.enabled && job.migration.target_model
        ? { migration_target_model: job.migration.target_model } : {}),
      business: job.business || 'general',
      lr0: job.lr0 ?? undefined,
      optimizer: job.optimizer || undefined,
      weight_decay: job.weight_decay ?? undefined,
      patience: job.patience ?? undefined,
      gpu_index: job.gpu_index ?? undefined,
      // 阶段1.3：聚合任务重跑时带原始数据集列表
      ...(Array.isArray(job.dataset_ids) && job.dataset_ids.length > 1
        ? { dataset_ids: job.dataset_ids }
        : {}),
      // 重跑保持原任务的算法库归属（产出仍落在同一个库下）
      library_root_id: job.library_root_id || undefined
    }
    const result = await createTrainJob(params)
    restoreScroll(prevScrollY)
    alert('已重新创建训练任务: ' + result.job_id)
    currentJobId.value = result.job_id
    subscribeToLogs(result.job_id)
    loadJobs()
  } catch (error: any) {
    restoreScroll(prevScrollY)
    alert('重新训练失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    retrainingJob.value = null
  }
}

// 恢复页面滚动位置（原生 alert 弹出/关闭可能引起页面跳动）
const restoreScroll = (y: number) => {
  requestAnimationFrame(() => {
    if (window.scrollY !== y) window.scrollTo(0, y)
  })
}

const stopJob = async (jobId: string) => {
  const prevScrollY = window.scrollY
  if (!(await showConfirm('确定要停止此训练任务吗？'))) return
  
  stoppingJob.value = jobId
  try {
    stopJobPolling(jobId)
    const { stopTrainJob } = await import('@/api/train')
    await stopTrainJob(jobId)
    alert('任务已停止')
    restoreScroll(prevScrollY)
    loadJobs()
  } catch (error: any) {
    alert('停止失败: ' + (error.response?.data?.detail || error.message))
    restoreScroll(prevScrollY)
  } finally {
    stoppingJob.value = null
  }
}

const resumeJob = async (jobId: string) => {
  const prevScrollY = window.scrollY
  if (!(await showConfirm('确定要继续训练此任务吗？\n如果有 checkpoint 将从 checkpoint 恢复，否则从原始模型重新开始。'))) return
  
  resumingJob.value = jobId
  try {
    stopJobPolling(jobId)
    await resumeTrainJob(jobId)
    alert('训练已恢复!')
    restoreScroll(prevScrollY)
    
    // 开始订阅日志
    logStore.clearLogs()
    currentJobId.value = jobId
    subscribeToLogs(jobId)
    
    loadJobs()
  } catch (error: any) {
    // 提取更详细的错误信息
    let errorMsg = '恢复训练失败: '
    if (error.response?.data?.detail) {
      errorMsg += error.response.data.detail
    } else if (error.message) {
      errorMsg += error.message
    } else {
      errorMsg += '未知错误'
    }
    alert(errorMsg)
  } finally {
    resumingJob.value = null
  }
}

const deleteJobItem = async (jobId: string) => {
  const prevScrollY = window.scrollY
  if (!(await showConfirm(`确定要删除任务 ${jobId} 吗？此操作不可恢复！`))) return
  
  deletingJob.value = jobId
  try {
    stopJobPolling(jobId)
    await deleteTrainJob(jobId)
    alert('删除成功!')
    restoreScroll(prevScrollY)
    loadJobs()
  } catch (error: any) {
    alert('删除失败: ' + (error.response?.data?.detail || error.message))
    restoreScroll(prevScrollY)
  } finally {
    deletingJob.value = null
  }
}

const loadModels = async () => {
  loadingModels.value = true
  try {
    const result = await listModels()
    models.value = result.models
  } catch (error: any) {
    console.error('加载模型失败:', error)
    alert('加载模型失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    loadingModels.value = false
  }
}

const subscribeToLogs = (jobId: string) => {
  logStore.setStreaming(true)
  
  // 尝试SSE
  try {
    eventSource = subscribeLogsSSE(
      jobId,
      (line: string) => {
        logStore.addLog(line)
      },
      (error: any) => {
        console.error('SSE error, falling back to polling', error)
        logStore.setStreaming(false)
        startPolling(jobId)
      }
    )
  } catch (error) {
    console.error('Failed to start SSE, using polling', error)
    startPolling(jobId)
  }
}

let pollOffset = 0
const startPolling = (jobId: string) => {
  pollOffset = 0
  pollInterval = setInterval(async () => {
    try {
      const result = await pollLogsTail(jobId, pollOffset)
      result.lines.forEach((line: string) => {
        logStore.addLog(line)
      })
      pollOffset = result.offset
    } catch (error) {
      console.error('Polling error', error)
    }
  }, 2000)
}

// 从单条日志解析进度（返回 null 表示非进度行）
// 兼容 YOLO 日志格式：epoch 进度 "1/10 ..." + tqdm 批次百分比 "640: 57% ━ 4/7"
const parseProgressLine = (cleanLine: string): { epoch: number; totalEpochs: number; batchPct: number } | null => {
  const m = cleanLine.match(/^(\d+)\/(\d+)\s+/)
  if (!m) return null
  // 取行内最后一个 tqdm 百分比 NN%（epoch 内批次进度，0~100）
  const pcts = [...cleanLine.matchAll(/(\d+)%/g)].map((x) => parseInt(x[1]))
  const batchPct = pcts.length ? pcts[pcts.length - 1] : 100
  return { epoch: parseInt(m[1]), totalEpochs: parseInt(m[2]), batchPct }
}

// 更新单个任务的进度（细粒度：epoch + 批次百分比）
const updateJobProgress = (jobId: string, lines: string[]) => {
  let cur = jobProgress.value[jobId]?.current || 0
  let total = jobProgress.value[jobId]?.total || 0
  let lossLine = jobProgress.value[jobId]?.lossLine || ''
  for (const raw of lines) {
    const clean = stripAnsi(raw).trim()
    const p = parseProgressLine(clean)
    if (p) {
      // 细粒度进度：epoch 内按批次百分比推进，避免长时间停留在同一百分比
      cur = (p.epoch - 1) + p.batchPct / 100
      total = p.totalEpochs
      if (clean.includes('loss')) {
        lossLine = clean
      }
    }
  }
  jobProgress.value[jobId] = {
    current: cur,
    total,
    percent: total > 0 ? Math.min(100, Math.round((cur / total) * 100)) : 0,
    has: total > 0,
    lossLine,
    lastActivity: Date.now()
  }
}

// 为某个任务启动独立日志轮询（用于任务卡牌进度）
const startJobPolling = async (jobId: string) => {
  if (jobPollers.has(jobId)) return
  if (!jobOffsets.has(jobId)) jobOffsets.set(jobId, 0)

  // 先读取历史日志，初始化进度
  try {
    const result = await getLogLines(jobId, 0)
    if (result.lines && !result.error) {
      updateJobProgress(jobId, result.lines)
      jobOffsets.set(jobId, result.total || result.lines.length)
    }
  } catch (e) {
    // 忽略历史读取失败
  }

  const timer = setInterval(async () => {
    try {
      const off = jobOffsets.get(jobId) || 0
      const result = await pollLogsTail(jobId, off)
      if (result.lines && result.lines.length) {
        updateJobProgress(jobId, result.lines)
        jobOffsets.set(jobId, result.offset)
      }
    } catch (error) {
      console.error('Job polling error', jobId, error)
    }
  }, 2000)
  jobPollers.set(jobId, timer)
}

// 停止某个任务的独立轮询
const stopJobPolling = (jobId: string) => {
  const timer = jobPollers.get(jobId)
  if (timer) {
    clearInterval(timer)
    jobPollers.delete(jobId)
  }
  jobOffsets.delete(jobId)
}

// 判断某任务是否长时间无新日志（可能卡住，而非正在训练）
const isStale = (jobId: string): boolean => {
  const la = jobProgress.value[jobId]?.lastActivity || 0
  return la > 0 && (nowTick.value - la) > 60_000
}

// 显示"最后更新 X 秒/分钟前"
const lastActivityText = (jobId: string): string => {
  const la = jobProgress.value[jobId]?.lastActivity || 0
  if (!la) return '等待训练启动...'
  const sec = Math.max(0, Math.round((nowTick.value - la) / 1000))
  if (sec < 5) return '更新中...'
  if (sec < 60) return `最后更新 ${sec} 秒前`
  return `最后更新 ${Math.floor(sec / 60)} 分钟前`
}

// 为所有运行中的任务启动独立轮询
const startAllJobPolling = () => {
  jobs.value.forEach((job) => {
    if (job.status === 'running') {
      startJobPolling(job.job_id)
    }
  })
}

// 训练失败操作建议：匹配常见错误，返回"客户看得懂"的说明与操作步骤
const getFailureAdvice = (error: string = ''): { title: string; reason: string; steps: string[] } => {
  const e = error.toLowerCase()

  // 1. 显存不足
  if (e.includes('out of memory') || e.includes('cuda out of memory') || e.includes('cublas') || e.includes('cudnn')) {
    return {
      title: '显卡内存不够了',
      reason: '训练需要占用的显存超过了当前显卡的可用空间，最常见原因是一次性塞入的图片太多。',
      steps: [
        '在"训练参数"里把"批次大小"调小：从 32 依次试 16、8、4',
        '同时可把"图片尺寸"从 640 调到 512，显存占用会大幅下降',
        '如果有多个 GPU，可在"训练节点"下拉里换一块空闲的显卡再试'
      ]
    }
  }

  // 2. 系统内存不足（错误1455 / 页面文件）
  if (e.includes('1455') || e.includes('page file') || e.includes('i/o error') || e.includes('memoryerror')) {
    return {
      title: '服务器内存不够了',
      reason: '训练时系统的运行内存（RAM）被占满，程序被系统强制终止。',
      steps: [
        '关闭服务器上其他占用内存的程序（尤其是之前的训练进程）',
        '在"训练参数"里减小"批次大小"（如 16 → 8）',
        '重启服务器释放内存后再重新训练'
      ]
    }
  }

  // 3. 字体缺失/下载失败
  if (e.includes('arial.ttf') || e.includes('font') || e.includes('download failure')) {
    return {
      title: '缺少绘图字体',
      reason: '训练结束画图表时需要 Arial.ttf 字体，但当前环境里没有且无法联网下载。',
      steps: [
        '检查服务器能否访问外网（下载字体/模型文件）',
        '联系管理员把 arial.ttf 放到字体目录后重试'
      ]
    }
  }

  // 4. 权重文件损坏
  if (e.includes('pytorchstreamreader') || e.includes('zip archive') || e.includes('central directory') || e.includes('broken')) {
    return {
      title: '模型文件损坏或下载不完整',
      reason: '预训练权重文件损坏、被截断或格式不对，没法加载。',
      steps: [
        '重新上传/下载对应的 .pt 权重文件，注意文件大小要完整',
        '换一个预训练模型版本再试（如 yolov8n.pt 换 yolov8s.pt）'
      ]
    }
  }

  // 5. 文件缺失
  if (e.includes('filenotfound') || e.includes('no such file')) {
    return {
      title: '缺少文件',
      reason: '训练所需的某个文件（数据集、权重或输出目录）不存在。',
      steps: [
        '确认数据集已上传并执行过"准备数据集"',
        '确认选择的预训练模型/权重文件存在',
        '确认后重新创建训练任务'
      ]
    }
  }

  // 6. 权限不足/文件占用
  if (e.includes('permission') || e.includes('access is denied')) {
    return {
      title: '文件被占用或没有权限',
      reason: '需要写入的输出文件正被其他程序占用，或没有写入权限。',
      steps: [
        '关闭占用训练输出目录的其他程序',
        '检查服务器磁盘权限，或联系管理员处理后重试'
      ]
    }
  }

  // 7. 数据集为空/标注缺失
  if ((e.includes('dataset') || e.includes('labels') || e.includes('images')) && e.includes('empty')) {
    return {
      title: '数据集有问题',
      reason: '数据集里图片或标注缺失，训练无法从空数据开始。',
      steps: [
        '到"数据集"页面确认图片与标注文件完整',
        '重新执行"准备数据集"，然后回到训练页重新创建任务'
      ]
    }
  }

  // 8. GPU 驱动/深度学习依赖不匹配
  if (e.includes('nvidia driver') || e.includes('sm_version') || e.includes('not enough compute') || e.includes('nvcc') || e.includes('undefined symbol') || e.includes('so: cannot open')) {
    return {
      title: 'GPU 驱动或运行环境不兼容',
      reason: '服务器显卡驱动或深度学习组件与当前模型不兼容，编程层面无法自动修复。',
      steps: [
        '在服务器上执行 nvidia-smi 确认显卡驱动正常',
        '联系管理员升级显卡驱动或重启容器环境后再训练'
      ]
    }
  }

  // 9. 数据加载进程相关系统错误（Docker 容器里多进程 DataLoader 的典型故障）
  if (e.includes('errno 22') || e.includes('invalid argument') || e.includes('resource_sharer') || e.includes('sem_open') || e.includes('no space left on device')) {
    return {
      title: '训练环境的数据加载异常',
      reason: '训练启动时系统底层的数据加载进程出问题（容器内多进程读图不稳定），与你的数据集本身无关。',
      steps: [
        '点击"重新训练"重试一次，系统会自动改用单进程加载数据（更稳定）',
        '若仍失败，把"批次大小"改为 16 或 8（不要用 -1 自动）再试',
        '连续失败请联系管理员查看服务器训练日志'
      ]
    }
  }

  // 10. P6 GPU 熔断（2.3）：GPU 利用率持续过低自动停训
  if (e.includes('gpu_abnormal') || (e.includes('gpu') && (e.includes('熔断') || e.includes('利用率')))) {
    return {
      title: 'GPU 异常，已自动停止训练（熔断）',
      reason: '训练所用显卡的利用率长时间过低，系统判定 GPU 异常或训练卡死，已自动停止训练以保护资源。',
      steps: [
        '在服务器上执行 nvidia-smi 查看显卡状态、驱动与显存占用',
        '确认 GPU 恢复正常后，点击"重新训练"或"继续训练"重新发起',
        '若频繁熔断：检查是否多个训练任务挤占同一块 GPU（可在训练页分别指定不同训练节点）'
      ]
    }
  }

  // 默认：未知错误
  return {
    title: '训练没有成功',
    reason: '这次失败的具体原因比较特殊，未能自动分类；原始错误信息已折叠在下方，可直接提供给技术人员。',
    steps: [
      '先按常见问题自查：批次大小是否过大、服务器内存/磁盘是否充足、数据集是否已准备、权重文件是否完整',
      '点击下方"原始错误信息"，复制内容发给管理员或平台技术人员',
      '修复后重新创建训练任务即可'
    ]
  }
}

// ===== 任务文件浏览器 =====
const jobFiles = ref<any | null>(null)  // 正在浏览文件的任务
const jobFilesLoading = ref(false)
const jobTree = ref<any | null>(null)  // 任务目录树
const jobTreeExpanded = reactive(new Set<string>())
const jobFilePreview = ref<any | null>(null)
const jobFilePreviewIsImage = ref(false)
const jobFilePreviewContent = ref('')

// 打开任务文件浏览器
const openJobFiles = async (jobId: string) => {
  jobFiles.value = { job_id: jobId, log_file: null }
  jobFilesLoading.value = true
  jobTree.value = null
  jobTreeExpanded.clear()
  try {
    const result = await getTrainJobTree(jobId)
    jobTree.value = result.tree
    jobFiles.value.log_file = result.log_file
  } catch (e: any) {
    jobTree.value = null
    showAlert('加载任务文件失败: ' + (e.response?.data?.detail || e.message), '错误')
  } finally {
    jobFilesLoading.value = false
  }
}

const closeJobFiles = () => {
  jobFiles.value = null
  jobTree.value = null
  jobTreeExpanded.clear()
  closeJobFilePreview()
}

const toggleJobDir = (node: any) => {
  if (jobTreeExpanded.has(node.path)) jobTreeExpanded.delete(node.path)
  else jobTreeExpanded.add(node.path)
}

const JOB_IMG_EXTS = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']

// 点击文件：图片直接预览，文本读取内容
const openJobFile = async (node: any) => {
  jobFilePreview.value = node
  jobFilePreviewContent.value = ''
  jobFilePreviewIsImage.value = JOB_IMG_EXTS.includes((node.ext || '').toLowerCase())
  if (jobFilePreviewIsImage.value) return
  try {
    const res = await fetch('/static/' + node.path)
    if (!res.ok) throw new Error('HTTP ' + res.status)
    jobFilePreviewContent.value = await res.text()
  } catch (e: any) {
    jobFilePreviewContent.value = '无法读取该文件：' + (e.message || e)
  }
}

const closeJobFilePreview = () => {
  jobFilePreview.value = null
  jobFilePreviewContent.value = ''
}

const handleFileImgError = (e: any) => {
  const el = e.target as HTMLImageElement
  el.style.display = 'none'
}

// 弹窗展示训练失败原因与操作建议（面向客户的结构化说明）
const failureModal = ref<{
  title: string
  reason: string
  steps: string[]
  rawError: string
  failedAt: string
} | null>(null)

const closeFailureDialog = () => {
  failureModal.value = null
}

const showFailureDialog = (job: any) => {
  const { title, reason, steps } = getFailureAdvice(job.error || job.message || '')
  // 优先展示完整堆栈（技术人员排查用），其次展示错误摘要
  const rawError = (job.error_traceback || job.error || job.message || '未知错误').slice(0, 4000)
  failureModal.value = {
    title,
    reason,
    steps,
    rawError,
    failedAt: job.failed_at || job.crashed_at
      ? new Date(job.failed_at || job.crashed_at).toLocaleString()
      : ''
  }
}

// 已自动弹窗提示过的失败任务（避免重复提示），持久化到 localStorage 只提示一次
const FAIL_ALERT_KEY = 'yolo_train_failed_alerted'
const failedJobsAlerted = ref<Set<string>>(
  new Set(JSON.parse(localStorage.getItem(FAIL_ALERT_KEY) || '[]'))
)

// 监听任务状态变化，训练失败时先提醒，再询问是否跳转训练页面
watch(
  jobs,
  (list) => {
    list.forEach((job) => {
      if ((job.status === 'failed' || job.status === 'crashed') && !failedJobsAlerted.value.has(job.job_id)) {
        failedJobsAlerted.value.add(job.job_id)
        localStorage.setItem(FAIL_ALERT_KEY, JSON.stringify([...failedJobsAlerted.value]))
        showConfirm('训练失败，是否跳转到训练页面？', '训练失败').then((ok) => {
          if (ok) router.push({ name: 'Train' })
        })
      }
    })
  },
  { deep: true }
)

const loadJobs = async () => {
  loadingJobs.value = true
  try {
    const result = await listTrainJobs()
    jobs.value = result.jobs
    startAllJobPolling()
    // 同步刷新模型列表，保证任务卡的守门员状态徽章/报告最新
    loadModels()
  } catch (error: any) {
    alert('加载失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    loadingJobs.value = false
  }
}

onMounted(async () => {
  loadJobs()
  loadModels()
  // 算法库详情页「发起训练」跳转：记住归属库，产出自动成为该库的新版本
  const libQ = route.query.library_root_id
  if (typeof libQ === 'string' && libQ) presetLibraryRootId.value = libQ
  // 1.7 模型仓库跳转：/train?dataset_ids=ds1,ds2 自动预选聚合
  const q = route.query.dataset_ids
  if (typeof q === 'string' && q) {
    await loadDatasets()
    const ids = q.split(',').filter(Boolean)
    const selectable = ids.filter((id: string) => datasets.value.some((d: any) => d.dataset_id === id))
    if (selectable.length) {
      selectDataset({ dataset_id: selectable[0] })
      selectable.slice(1).forEach((id: string) => selectDataset({ dataset_id: id }))
    }
  } else {
    loadDatasets()
  }
  loadCustomModels()
  loadGpus()
  // 每秒跳动，刷新"最后更新 X 秒前"提示
  nowTimer = setInterval(() => { nowTick.value = Date.now() }, 1000)
})

onUnmounted(() => {
  if (eventSource) {
    eventSource.close()
  }
  if (pollInterval) {
    clearInterval(pollInterval)
  }
  if (nowTimer) {
    clearInterval(nowTimer)
  }
  // 清理所有任务卡牌的独立轮询
  jobPollers.forEach((timer) => clearInterval(timer))
  jobPollers.clear()
  jobOffsets.clear()
  logStore.setStreaming(false)
})
</script>

<style scoped>
.train-progress {
  margin-bottom: 0.75rem;
}

.progress-bar {
  height: 18px;
  background: #eef0f3;
  border-radius: 9px;
  overflow: hidden;
  border: 1px solid #e0e3e8;
}

.progress-fill {
  position: relative;
  height: 100%;
  background: linear-gradient(90deg, #8e44ad, #6c2d82);
  border-radius: 9px;
  transition: width 0.4s ease;
  min-width: 0%;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 进度条内嵌百分比文字 */
.progress-fill-text {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  color: #fff;
  font-size: 0.75rem;
  font-weight: bold;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
  white-space: nowrap;
  line-height: 1;
}

/* 高级参数面板 */
.adv-block {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.adv-toggle {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  cursor: pointer;
  font-size: 0.95rem;
  color: #2c3e50;
}

.adv-toggle input {
  width: auto;
}

.suggest-status {
  font-size: 0.8rem;
  color: #27ae60;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}

.auto-pick-info {
  font-size: 0.8rem;
  color: #27ae60;
}

/* 数据集归属算法库提示（产出归库 / 跨库聚合 / 未归属） */
.lib-pick-info {
  display: block;
  margin-top: 0.3rem;
  font-size: 0.8rem;
  line-height: 1.4;
  color: #27ae60;
}

.lib-pick-info.warn {
  color: #e67e22;
}

.adv-grid .form-group {
  min-width: 0;
}

.adv-enhance {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-top: 0.75rem;
  padding: 0.8rem 1rem;
  background: #f6f9fc;
  border: 1px solid #e3eaf2;
  border-radius: 6px;
}

.adv-enhance .form-group {
  min-width: 0;
}

.adv-enhance .form-group small {
  display: block;
  margin-top: 0.25rem;
  color: #8593a6;
  line-height: 1.4;
}

.check-line {
  display: flex;
  width: fit-content;
  align-items: center;
  gap: 0.4rem;
  cursor: pointer;
  color: #2c3e50;
  font-weight: 600;
}

.check-line input {
  width: auto;
}

.suggest-reason {
  margin: 0.4rem 0 0.75rem;
  padding: 0.55rem 0.8rem;
  background: #eefbf3;
  border: 1px solid #c8ecd4;
  border-radius: 4px;
  font-size: 0.8rem;
  color: #1e7e46;
  line-height: 1.5;
}

/* ===== 2.2 黄金参数 L2 智能推荐区（只读高亮） ===== */
.l2-panel {
  margin: 0.5rem 0 0.75rem;
  border: 1px solid #c3d4f2;
  border-radius: 6px;
  background: linear-gradient(180deg, #f2f7ff, #eaf1fb);
  overflow: hidden;
}

/* 三层分档卡片：基准（稳定优先）/ 白银（均衡）/ 黄金（效率最优） */
.tier-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.5rem;
  padding: 0.6rem 0.8rem;
}

.tier-card {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  padding: 0.55rem 0.7rem;
  border: 1px solid #d3dff5;
  border-radius: 6px;
  background: #fff;
  cursor: pointer;
  text-align: left;
  transition: all 0.15s ease;
}

.tier-card:hover {
  border-color: #7ba3e8;
  box-shadow: 0 1px 4px rgba(44, 90, 160, 0.12);
}

.tier-card.tier-active {
  border: 2px solid #2c5aa0;
  background: #e8f0ff;
  box-shadow: 0 1px 5px rgba(44, 90, 160, 0.2);
  transform: translateY(-1px);
}

.tier-label {
  font-weight: bold;
  font-size: 0.85rem;
  color: #2c5aa0;
}

.tier-vals {
  font-size: 0.78rem;
  color: #44506a;
}

.tier-vals b {
  color: #1c3a6e;
}

.tier-desc {
  font-size: 0.72rem;
  color: #7d8799;
  line-height: 1.4;
}

.l2-head {
  padding: 0.45rem 0.8rem;
  border-bottom: 1px dashed #c3d4f2;
}

.l2-title {
  font-size: 0.82rem;
  font-weight: bold;
  color: #2c5aa0;
}

.l2-body {
  padding: 0.6rem 0.8rem 0.7rem;
}

.l2-values {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin-bottom: 0.45rem;
}

.l2-value {
  display: inline-flex;
  align-items: baseline;
  gap: 0.35rem;
  background: #fff;
  border: 1px solid #d6e2f5;
  border-radius: 6px;
  padding: 0.3rem 0.7rem;
  font-size: 0.85rem;
  color: #4a6fa5;
}

.l2-value b {
  color: #8e44ad;
  font-size: 1.05rem;
  font-weight: 800;
}

.l2-formulas {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.l2-formula {
  font-size: 0.75rem;
  color: #5a7cb0;
  font-family: 'Courier New', monospace;
  line-height: 1.5;
}

.l2-reason {
  padding: 0.4rem 0.8rem 0.6rem;
  font-size: 0.78rem;
  color: #3d6ca8;
  border-top: 1px dashed #d0dff2;
  line-height: 1.5;
}

.l2-panel + .adv-block .adv-toggle {
  font-weight: bold;
  color: #2c3e50;
}

/* L3 区块分隔（展开专家微调时） */

.progress-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 0.4rem;
  font-size: 0.9rem;
  color: #555;
}

.progress-percent {
  font-weight: bold;
  color: #8e44ad;
}

/* 最后更新提示：正常灰色，长时间无新日志变橙色警告 */
.progress-last {
  font-size: 0.8rem;
  color: #999;
}

.progress-last.warn {
  color: #e67e22;
  font-weight: bold;
}

.loss-line {
  margin-top: 0.4rem;
  padding: 0.5rem 0.75rem;
  background: #f7f0fb;
  border: 1px solid #ead9f2;
  border-radius: 4px;
  font-family: 'Courier New', monospace;
  font-size: 0.8rem;
  color: #6c2d82;
  word-break: break-all;
}

.jobs-list {
  margin-top: 1rem;
  display: grid;
  gap: 1rem;
}

.job-item {
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.job-item h3 {
  margin-bottom: 0.5rem;
  color: #2c3e50;
}

.job-item p {
  margin: 0.25rem 0;
  color: #7f8c8d;
}

.job-item .job-actions {
  margin-top: 0.5rem;
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.job-item .job-actions button {
  margin-top: 0;
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}

/* 失败原因按钮（红色描边） */
.job-item .job-actions button.danger-outline {
  background: #fff5f5;
  border: 1px solid #e74c3c;
  color: #e74c3c;
}

.job-item .job-actions button.danger-outline:hover {
  background: #fdecea;
}

.select-with-refresh {
  display: flex;
  gap: 0.5rem;
}

.select-with-refresh select {
  flex: 1;
}

.ds-selectbar {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 0.5rem;
}

.ds-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 0.6rem;
  margin-bottom: 0.5rem;
}

.ds-card {
  border: 1px solid #d0d3d8;
  border-radius: 6px;
  padding: 0.6rem 0.75rem;
  cursor: pointer;
  background: #fff;
  transition: all 0.15s ease;
}

.ds-card:hover {
  border-color: #8e44ad;
  box-shadow: 0 2px 6px rgba(142, 68, 173, 0.15);
}

.ds-card.selected {
  border-color: #8e44ad;
  background: #f7f0fb;
  box-shadow: 0 0 0 2px rgba(142, 68, 173, 0.25);
}

.ds-card.disabled {
  opacity: 0.55;
  cursor: not-allowed;
  background: #f8f9fa;
}

.ds-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
}

.ds-name {
  font-weight: bold;
  font-size: 0.9rem;
  word-break: break-all;
}

.status-badge {
  font-size: 0.7rem;
  padding: 0.1rem 0.4rem;
  border-radius: 8px;
  white-space: nowrap;
  color: #fff;
}

.status-prepared {
  background: #2ecc71;
}

.status-uploaded {
  background: #f39c12;
}

.status-unknown {
  background: #95a5a6;
}

.ds-card-info {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  font-size: 0.8rem;
  color: #7f8c8d;
}

.ds-card-warn {
  margin-top: 0.35rem;
  font-size: 0.75rem;
  color: #b9770e;
}

.ds-main-tag {
  font-size: 0.65rem;
  background: #8e44ad;
  color: #fff;
  border-radius: 4px;
  padding: 0.05rem 0.3rem;
  margin-left: 0.25rem;
  vertical-align: middle;
}

.aggregate-opt {
  display: flex;
  width: fit-content;
  align-items: center;
  gap: 0.35rem;
  margin-top: 0.4rem;
  font-size: 0.8rem;
  color: #27ae60;
  cursor: pointer;
}

.empty-state {
  grid-column: 1 / -1;
  text-align: center;
  color: #95a5a6;
  padding: 1rem;
  border: 1px dashed #d0d3d8;
  border-radius: 6px;
}

.model-label {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.add-model-btn {
  background: #8e44ad;
  color: white;
  border: none;
  padding: 0.3rem 1rem;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: bold;
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  box-shadow: 0 2px 4px rgba(142, 68, 173, 0.3);
}

.add-model-btn:disabled {
  background: #b9a0c9;
  cursor: not-allowed;
}

.pt-file-input {
  display: none;
}

.custom-model-block {
  margin-top: 0.6rem;
}

.custom-model-label {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: #7f8c8d;
}

.custom-model-block select {
  width: 100%;
  margin-top: 0.25rem;
}

.refresh-btn {
  padding: 0.5rem 1rem;
  white-space: nowrap;
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}

.model-selector {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.model-selector .version-select,
.model-selector .size-select {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.model-selector label {
  font-size: 0.875rem;
  color: #7f8c8d;
}

.model-selector select {
  width: 100%;
}

.loading-state {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 2rem;
  color: #7f8c8d;
}

.empty-state {
  padding: 2rem;
  text-align: center;
  color: #7f8c8d;
  background: #f8f9fa;
  border-radius: 4px;
}

button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.25rem;
}

/* 任务名称可点击 */
.job-name {
  cursor: pointer;
}
.job-name:hover {
  color: #2c6fbb;
  text-decoration: underline;
}

/* 任务文件浏览器弹窗 */
.job-files-mask {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 1.5rem;
}

.job-files-dialog {
  background: white;
  border-radius: 8px;
  width: 90%;
  max-width: 900px;
  max-height: 88vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.viewer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.25rem;
  border-bottom: 1px solid #eee;
}

.viewer-header h3 {
  margin: 0;
  color: #2c3e50;
}

.viewer-close {
  padding: 0.35rem 0.9rem;
  font-size: 0.875rem;
}

.viewer-section-title {
  padding: 0.75rem 1.25rem 0.25rem;
  font-weight: 600;
  color: #2c3e50;
  border-top: 1px solid #f0f0f0;
  margin-top: 0.5rem;
}

.viewer-tree {
  max-height: 50vh;
  overflow-y: auto;
  padding: 0.5rem 1.25rem 0.75rem;
}

.job-log-link {
  padding: 0.5rem 1.25rem 1rem;
  border-top: 1px solid #f0f0f0;
}

/* 文件内容预览弹窗 */
.file-preview-mask {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1100;
  padding: 1.5rem;
}

.file-preview-dialog {
  background: white;
  border-radius: 8px;
  width: 80%;
  max-width: 900px;
  max-height: 88vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.file-preview-img-wrap {
  flex: 1;
  overflow: auto;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f5f5f5;
  padding: 1rem;
}

.file-preview-img-wrap img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.file-preview-text {
  flex: 1;
  overflow: auto;
  margin: 0;
  padding: 1rem 1.25rem;
  font-size: 0.8rem;
  line-height: 1.5;
  background: #fafbfc;
  white-space: pre-wrap;
  word-break: break-all;
  color: #2c3e50;
}

/* 早停完成徽标 */
.early-stop-badge {
  display: inline-block;
  margin-left: 0.4rem;
  padding: 0.1rem 0.5rem;
  background: #eafaf1;
  border: 1px solid #82e0aa;
  border-radius: 10px;
  color: #1e8449;
  font-size: 0.75rem;
  font-weight: bold;
  white-space: nowrap;
}

/* P6 GPU 熔断徽标（2.3） */
.gpu-abnormal-badge {
  display: inline-block;
  margin-left: 0.4rem;
  padding: 0.1rem 0.5rem;
  background: #fdf0ec;
  border: 1px solid #f0a08a;
  border-radius: 10px;
  color: #c0392b;
  font-size: 0.75rem;
  font-weight: bold;
  white-space: nowrap;
}

/* 2.4 跨代迁移徽标（任务卡） */
.migration-badge {
  display: inline-block;
  margin-left: 0.4rem;
  padding: 0.1rem 0.5rem;
  background: #eef3fd;
  border: 1px solid #9dbff2;
  border-radius: 10px;
  color: #1f5fbf;
  font-size: 0.75rem;
  font-weight: bold;
  white-space: nowrap;
}

/* ===== 2.4 跨代迁移训练区（微调模式） ===== */
.migration-block {
  margin-top: 0.75rem;
  padding: 0.6rem 0.75rem;
  background: #f6f9ff;
  border: 1px dashed #9dbff2;
  border-radius: 8px;
}
.migration-toggle {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.85rem;
  cursor: pointer;
  color: #1f5fbf;
}
.migration-block > small {
  display: block;
  margin: 0.3rem 0 0.4rem;
  color: #7f8c8d;
}
.migration-selector {
  margin-top: 0.5rem;
  padding-top: 0.5rem;
  border-top: 1px solid #e3ebfa;
}
.migration-hint {
  display: block;
  margin-top: 0.4rem;
  color: #1f5fbf;
  font-weight: bold;
}
.migration-hint.warn {
  color: #e67e22;
}

/* ===== 训练失败原因弹窗（面向客户） ===== */
.failure-dialog {
  max-width: 560px;
}

.failure-body {
  padding: 1rem 1.25rem 1.25rem;
  overflow-y: auto;
}

.fail-emoji {
  margin-right: 0.2rem;
}

.fail-title {
  font-size: 1.15rem;
  font-weight: bold;
  color: #e74c3c;
}

.fail-reason {
  margin-top: 0.5rem;
  line-height: 1.6;
  color: #4a5568;
}

.fail-time {
  margin-top: 0.4rem;
  font-size: 0.8rem;
  color: #999;
}

.fail-steps {
  margin-top: 0.9rem;
  background: #fdf3f1;
  border: 1px solid #f5c6bc;
  border-radius: 6px;
  padding: 0.7rem 1rem 0.9rem;
}

.fail-steps-title {
  font-weight: bold;
  color: #c0392b;
  font-size: 0.9rem;
}

.fail-steps ol {
  margin: 0.5rem 0 0;
  padding-left: 1.25rem;
}

.fail-steps li {
  line-height: 1.7;
  font-size: 0.88rem;
  color: #34495e;
}

.fail-log {
  margin-top: 0.9rem;
}

.fail-log summary {
  cursor: pointer;
  font-size: 0.82rem;
  color: #7f8c8d;
  user-select: none;
}

.fail-log-text {
  margin: 0.5rem 0 0;
  padding: 0.6rem 0.8rem;
  background: #2d3436;
  color: #dfe6e9;
  font-size: 0.75rem;
  line-height: 1.5;
  border-radius: 4px;
  overflow: auto;
  max-height: 12rem;
  white-space: pre-wrap;
  word-break: break-all;
}

/* ===== 模型守门员：任务卡状态徽章与报告 ===== */
.gk-badge {
  display: inline-block;
  margin-left: 0.4rem;
  padding: 0.1rem 0.5rem;
  border-radius: 10px;
  font-size: 0.72rem;
  font-weight: bold;
  white-space: nowrap;
  vertical-align: middle;
}

.gk-production_ready {
  background: #eafaf1;
  border: 1px solid #82e0aa;
  color: #1e8449;
}

.gk-rejected {
  background: #fdecea;
  border: 1px solid #f5b7b1;
  color: #c0392b;
}

.gk-superseded {
  background: #fef9e7;
  border: 1px solid #f9e79f;
  color: #b7950b;
}

.gk-version {
  display: inline-block;
  margin-left: 0.5rem;
  font-size: 0.78rem;
  color: #7f8c8d;
  vertical-align: middle;
}

.gk-report {
  margin-top: 0.45rem;
  border: 1px solid #d7dde4;
  border-radius: 6px;
  background: #fafbfc;
}

.gk-report summary {
  cursor: pointer;
  padding: 0.45rem 0.7rem;
  font-size: 0.82rem;
  color: #34495e;
  user-select: none;
}

.gk-report-body {
  padding: 0.4rem 0.7rem 0.65rem;
  border-top: 1px dashed #dfe4ea;
}

.gk-report-text {
  font-size: 0.8rem;
  line-height: 1.6;
  color: #4a5568;
  word-break: break-word;
}

.gk-regressed {
  margin-top: 0.45rem;
  font-size: 0.78rem;
  color: #c0392b;
}

.gk-regressed-item {
  display: inline-block;
  margin: 0.15rem 0.25rem 0 0;
  padding: 0.05rem 0.45rem;
  background: #fdecea;
  border: 1px solid #f5b7b1;
  border-radius: 8px;
  font-weight: bold;
}

/* ===== 预训练权重开关 / 从零初始化（§3.2.1 / §3.2.2） ===== */
.pretrain-block {
  padding: 0.75rem 1rem;
  background: #f8f6fc;
  border: 1px solid #e8dff2;
  border-radius: 8px;
}
.check-line {
  font-weight: 700;
}
.pretrain-source {
  margin-top: 0.6rem;
  padding-top: 0.6rem;
  border-top: 1px dashed #dccfe8;
}
.source-radios {
  display: flex;
  gap: 1.25rem;
  margin-bottom: 0.5rem;
}
.source-radio {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.85rem;
  color: #4a5568;
  cursor: pointer;
}
.source-radio input {
  width: auto;
}
.from-scratch-note {
  margin-top: 0.6rem;
  padding: 0.55rem 0.75rem;
  background: #fff8e6;
  border: 1px solid #f5dc9a;
  border-radius: 6px;
  font-size: 0.82rem;
  color: #8a6d1a;
  line-height: 1.5;
}
.from-scratch-note b {
  font-weight: bold;
}
/* 强制从零初始化链接（L3 高级模式底部） */
.l3-block {
  margin-top: 0.4rem;
  text-align: right;
}
.scratch-link {
  font-size: 0.78rem;
  color: #c0392b;
  text-decoration: underline dashed;
  cursor: pointer;
}
.scratch-link:hover {
  color: #922b21;
}

/* ===== L2 资源预估（§2.1：预计耗时 / 显存占用） ===== */
.l2-resource {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin-bottom: 0.45rem;
}
.res-item {
  font-size: 0.8rem;
  color: #34495e;
  background: #fff;
  border: 1px solid #d6e2f5;
  border-radius: 6px;
  padding: 0.25rem 0.6rem;
}

/* ===== 任务列表卡片（§2.2：状态色块 / Sparkline） ===== */
.job-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.6rem;
  margin-bottom: 0.5rem;
}
.job-head h3 {
  margin-bottom: 0;
}
.job-pill {
  flex-shrink: 0;
  font-size: 0.72rem;
  font-weight: bold;
  padding: 0.15rem 0.6rem;
  border-radius: 10px;
  color: #fff;
  white-space: nowrap;
}
.jp-done { background: #27ae60; }
.jp-run { background: #3498db; }
.jp-fail { background: #e74c3c; }
.jp-stop { background: #e67e22; }
.job-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
}
.job-badges span {
  margin-left: 0;
}
.spark-wrap {
  margin-top: 0.4rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.spark-label {
  font-size: 0.72rem;
  color: #9b7ab8;
  white-space: nowrap;
}
.spark {
  flex: 1;
  height: 26px;
  border: 1px dashed #e3d8ee;
  border-radius: 4px;
  background: #fbf9fd;
}

/* ⋯ 更多操作菜单（§2.2 操作收敛） */
.job-more {
  position: relative;
}
.job-more-btn {
  padding: 0.3rem 0.6rem;
  font-size: 1rem;
  line-height: 1;
}
.job-more-menu {
  position: absolute;
  right: 0;
  top: calc(100% + 4px);
  z-index: 30;
  min-width: 130px;
  background: #fff;
  border: 1px solid #d7dde4;
  border-radius: 6px;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
  padding: 0.3rem;
  display: flex;
  flex-direction: column;
}
.job-more-item {
  display: block;
  width: 100%;
  text-align: left;
  padding: 0.4rem 0.6rem;
  font-size: 0.82rem;
  color: #34495e;
  background: none;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
.job-more-item:hover {
  background: #f2f5f9;
}
.job-more-item.link {
  color: #2c6fbb;
}
.job-more-item.danger {
  color: #e74c3c;
}
.job-more-item.danger:hover {
  background: #fdf0ee;
}
</style>

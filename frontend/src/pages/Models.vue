<template>
  <div class="models">
    <!-- 模型管理 -->
    <div class="card">
      <h2>模型管理</h2>
      
      <!-- 模型上传 -->
      <div class="upload-section">
        <h3>上传模型</h3>
        <div class="form-group">
          <label>选择 .zip 或 .pt 文件</label>
          <div v-if="selectedModelFile" class="selected-file">
            <span class="selected-file-icon">📦</span>
            <span class="selected-file-name">{{ selectedModelFile.name }}</span>
            <span class="selected-file-size">({{ (selectedModelFile.size / 1024 / 1024).toFixed(2) }} MB)</span>
          </div>
          <div v-else class="selected-file empty">尚未选择文件</div>
          <div class="upload-row">
            <input type="file" accept=".zip,.pt" id="modelZip" @change="handleModelFileSelect" :disabled="uploadingModel" class="file-input" />
            <label for="modelZip" class="file-btn" :disabled="uploadingModel">选择文件</label>
          </div>
        </div>
        <div class="form-group">
          <label for="modelUploadName">模型名称</label>
          <input
            id="modelUploadName"
            v-model="modelUploadName"
            type="text"
            class="name-input"
            maxlength="64"
            :disabled="uploadingModel"
            placeholder="如：积水分割 v2（dfs 机位）；留空则用文件名"
          />
          <p class="form-hint">权重入库后统一存为 best.pt，不填名称时列表里会都显示「best」，建议填写</p>
        </div>
        <div class="form-group">
          <label for="modelUploadLib">归入算法库</label>
          <select
            id="modelUploadLib"
            v-model="modelUploadLib"
            class="lib-select"
            :disabled="uploadingModel"
            @focus="ensureLibraries"
          >
            <option value="">不归库（独立模型）</option>
            <option v-for="lib in libraries" :key="lib.model_library_id" :value="lib.model_library_id">
              {{ lib.display_name || lib.model_library_id }}
            </option>
          </select>
          <p class="form-hint">
            归库后模型只在算法库详情里出现；不归库会作为「独立模型」单独占一张卡片。
            <button type="button" class="link-btn" @click="newLibOpen = true">＋新建算法库</button>
          </p>
        </div>
        <button class="upload-btn" @click="uploadModelFile" :disabled="!selectedModelFile || uploadingModel">
          <span v-if="uploadingModel" class="loading-spinner"></span>
          {{ uploadingModel ? '上传中...' : '上传' }}
        </button>
        <p class="form-hint">直接传 .pt 权重即可（自动入库并可作预标注）；ZIP 文件应包含 model.json 和 weights 目录</p>
      </div>
      <LibraryCreateModal v-model:open="newLibOpen" confirm-text="创建并选中" @created="onNewLibrary" />

      <div v-if="loadingModels" class="loading-state">
        <span class="loading-spinner large"></span>
        <span>加载模型列表...</span>
      </div>
      <div v-else-if="models.length > 0" class="models-list">
        <!-- 规范 §2.1：模型列表表格化，只展示决策所需核心信息 -->
        <table class="model-table">
          <thead>
            <tr>
              <th>模型</th>
              <th>算法架构</th>
              <th>数据集 / 类别</th>
              <th>核心指标</th>
              <th>超参摘要</th>
              <th>创建时间</th>
              <th>算法库</th>
              <th class="th-ops">操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="model in models" :key="model.model_id" class="model-table-row">
              <td class="cell-name">
                <div class="m-title">
                  <span v-if="model.status" :class="'m-status m-' + model.status">{{ statusText(model) }}</span>
                  <b class="m-name">{{ model.display_name || model.model_id }}</b>
                  <span v-if="model.version" class="m-version">版本 {{ model.version }}</span>
                  <span v-if="model.gatekeeper?.report" class="gk-dot" title="守门员评估报告（详情页可查看全文）">🛡️</span>
                </div>
                <div class="m-sub">
                  <code class="m-id">{{ model.model_id }}</code>
                  <span v-if="model.model_code" class="m-code">{{ model.model_code }}</span>
                  <span v-else class="m-code m-code-unset" title="尚未配置模型唯一 code（模型流转/自动关联依赖它）">未配置code</span>
                </div>
              </td>
              <td class="cell-arch">{{ modelArch(model) }}</td>
              <td class="cell-data">
                <span class="d-data">{{ modelDatasetText(model) }}</span>
                <span v-if="model.classes?.length" class="d-classes">{{ model.classes.length }} 类</span>
              </td>
              <td class="cell-map">
                <template v-if="model.gatekeeper?.new_metrics?.mAP50_95 != null">
                  <b>{{ modelMapText(model) }}</b><small class="map-lab"> mAP50-95</small>
                </template>
                <span v-else class="dim">—</span>
              </td>
              <td class="cell-hps">{{ modelHpsText(model) }}</td>
              <td class="cell-time">{{ fmtDateTime(model.created_at) }}</td>
              <td class="cell-lib">
                <select
                  class="lib-select"
                  :value="modelLibId(model)"
                  :disabled="assigningLibId === model.model_id"
                  @focus="ensureLibraries"
                  @change="assignLibrary(model, ($event.target as HTMLSelectElement).value)"
                >
                  <option value="">未归库（独立模型）</option>
                  <option v-for="lib in libraries" :key="lib.model_library_id" :value="lib.model_library_id">
                    {{ lib.display_name || lib.model_library_id }}
                  </option>
                </select>
              </td>
              <td class="cell-ops">
                <button @click="viewModelDetails(model)" class="primary small">详情</button>
                <div
                  class="more-menu-wrap"
                  :class="{ 'menu-open': moreOpenId === model.model_id }"
                  @mouseenter="cancelMenuClose"
                  @mouseleave="scheduleMenuClose"
                >
                  <button class="secondary small" @click.stop="toggleMenu(model.model_id)" title="更多操作">⋯</button>
                  <div v-if="moreOpenId === model.model_id" class="more-menu">
                    <button class="menu-item" @click="renameModelItem(model); closeMenu()" :disabled="renamingModel === model.model_id">
                      {{ renamingModel === model.model_id ? '保存中...' : '重命名' }}
                    </button>
                    <button class="menu-item" @click="exportModelFile(model.model_id); closeMenu()" :disabled="exportingModel === model.model_id">
                      <span v-if="exportingModel === model.model_id" class="loading-spinner"></span>
                      {{ exportingModel === model.model_id ? '导出中...' : '导出 PT' }}
                    </button>
                    <button class="menu-item menu-danger" @click="deleteModelItem(model.model_id); closeMenu()" :disabled="deletingModel === model.model_id">
                      {{ deletingModel === model.model_id ? '删除中...' : '删除' }}
                    </button>
                  </div>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-else class="empty-state">
        暂无模型，请先训练模型
      </div>
      <button @click="loadModels()" class="secondary" :disabled="loadingModels">
        <span v-if="loadingModels" class="loading-spinner"></span>
        {{ loadingModels ? '加载中...' : '刷新模型列表' }}
      </button>
    </div>

    <!-- ===== P8 半自动发布：待确认发布（生产候选） ===== -->
    <div class="card">
      <div class="publish-head">
        <h2>待确认发布（P8 半自动发布）</h2>
        <button class="secondary" @click="loadPendingPublish" :disabled="pendingLoading || publishingModel !== null">
          <span v-if="pendingLoading" class="loading-spinner"></span>
          {{ pendingLoading ? '加载中...' : '刷新' }}
        </button>
      </div>
      <p class="form-hint">
        守门员通过后进入"待确认发布"（冷却期 {{ COOLDOWN_HOURS }} 小时），管理员确认后成为该业务唯一生产版本，同业务其他已发布模型自动退役（superseded）
      </p>
      <div v-if="pendingLoading" class="loading-state">
        <span class="loading-spinner large"></span>
        <span>加载待确认发布列表...</span>
      </div>
      <div v-else-if="pendingError" class="warning-state">{{ pendingError }}</div>
      <div v-else-if="pendingModels.length === 0" class="empty-state">
        暂无待确认发布的模型（守门员通过后自动出现在这里）
      </div>
      <div v-else class="pending-list">
        <div v-for="m in pendingModels" :key="m.model_id" class="pending-item">
          <div class="pending-head">
            <div class="pending-title">
              <span class="publish-badge">待确认发布</span>
              <b>{{ m.display_name || m.model_id }}</b>
              <span v-if="m.version" class="m-version">版本 {{ m.version }}</span>
              <span v-if="m.migration?.enabled" class="migration-badge"
                    :title="`跨代迁移训练产出：${m.migration.source} → ${m.migration.target}`">
                🔄 迁移
              </span>
            </div>
            <div class="pending-actions">
              <span v-if="cooldownRemaining(m.ready_at) > 0" class="cooldown-hint"
                    :title="`ready_at=${m.ready_at}，冷却结束后自动可确认`">
                ⏳ 冷却中，剩 {{ cooldownRemaining(m.ready_at).toFixed(1) }}h
              </span>
              <button
                @click="confirmPublishItem(m)"
                class="primary"
                :disabled="publishingModel === m.model_id"
                :title="cooldownRemaining(m.ready_at) > 0
                  ? '仍在冷却期内，点击将使用 force=true 强制确认发布'
                  : '确认后该模型成为本业务唯一生产版本，同业务其他已发布模型自动退役'"
              >
                <span v-if="publishingModel === m.model_id" class="loading-spinner"></span>
                {{ publishingModel === m.model_id ? '确认中...' : (cooldownRemaining(m.ready_at) > 0 ? '强制确认' : '确认发布') }}
              </button>
            </div>
          </div>
          <div class="pending-meta">
            <span>业务: {{ bizLabel(m.business) }}</span>
            <span>就绪时间: {{ fmtTime(m.ready_at) }}</span>
            <span v-if="m.gatekeeper?.new_metrics?.mAP50_95 != null">
              mAP50-95: <b>{{ (m.gatekeeper.new_metrics.mAP50_95 * 100).toFixed(1) }}%</b>
            </span>
            <span v-if="m.lineage?.base_model">基座: {{ m.lineage.base_model }}</span>
            <span v-if="m.gatekeeper?.report" :title="'点击查看守门员报告'"
                  class="publish-report-link" @click="viewModelDetails(m)">🛡️ 守门员报告</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 2.7 相似模型排查与合并（人工工具） -->
    <div class="card">
      <div class="sim-header">
        <h2>相似模型排查与合并</h2>
        <div class="sim-actions">
          <label class="sim-threshold">
            相似度阈值
            <input type="number" v-model.number="simThreshold" min="0" max="1" step="0.05" :disabled="simScanning" />
          </label>
          <button class="secondary" @click="runSimilarScan" :disabled="simScanning">
            <span v-if="simScanning" class="loading-spinner"></span>
            {{ simScanning ? '扫描中...' : '扫描相似模型' }}
          </button>
          <button class="secondary" @click="loadMergeLogs" :disabled="loadingMergeLogs" title="查看合并历史记录（回滚依据）">合并日志</button>
          <button class="danger-outline" @click="doRollbackMerge(-1)" :disabled="rollingBackMerge" title="回滚最近一次合并（还原数据集归属）">
            <span v-if="rollingBackMerge" class="loading-spinner"></span>
            回滚最近合并
          </button>
        </div>
      </div>
      <p class="form-hint">
        按标签字典类别名（english_code）相似度扫描模型对；命中组推荐「数据集更多者」为主模型。
        勾选后合并：差集类别自动并入主模型标签字典、数据集重归属主模型、被合并模型保留为历史分支（不删除），全程写入日志可回滚。
      </p>

      <div v-if="simScanning" class="loading-state"><span class="loading-spinner"></span>扫描中...</div>
      <div v-else-if="simPairs.length" class="sim-list">
        <div v-for="(pair, i) in simPairs" :key="i" class="sim-pair" :class="{ checked: pair.checked }">
          <div class="sim-pair-main">
            <input type="checkbox" v-model="pair.checked" class="sim-check" />
            <span class="sim-score">{{ (pair.similarity * 100).toFixed(0) }}% 相似</span>
            <span class="sim-models">
              <span class="sim-model" :class="{ 'is-main': pair.suggested_main === pair.a.model_id }">
                {{ pair.a.name }} <code>{{ pair.a.code }}</code>（{{ pair.a.dataset_count }} DS）
                <span v-if="pair.suggested_main === pair.a.model_id" class="sim-main-tag">主</span>
              </span>
              <span class="sim-vs">⇄</span>
              <span class="sim-model" :class="{ 'is-main': pair.suggested_main === pair.b.model_id }">
                {{ pair.b.name }} <code>{{ pair.b.code }}</code>（{{ pair.b.dataset_count }} DS）
                <span v-if="pair.suggested_main === pair.b.model_id" class="sim-main-tag">主</span>
              </span>
            </span>
          </div>
          <div class="sim-common" v-if="pair.common_classes.length">共同类别: {{ pair.common_classes.join('、') }}</div>
        </div>
        <button class="primary" @click="confirmMergeSelected" :disabled="!selectedPairCount || mergingSim">
          <span v-if="mergingSim" class="loading-spinner"></span>
          {{ mergingSim ? '合并中...' : `合并选中的 ${selectedPairCount} 组（按推荐主模型吸收）` }}
        </button>
      </div>
      <div v-else-if="simScanned" class="empty-state">
        未发现相似度 ≥ {{ simThreshold }} 的模型对（可补充模型类别英文名后再扫）
      </div>
    </div>

    <!-- 合并日志弹窗 -->
    <div v-if="showMergeLogs" class="modal-overlay" @click.self="closeMergeLogs">
      <div class="modal-content modal-large">
        <div class="modal-header">
          <h3>合并日志（2.7 · 可回滚）</h3>
          <button @click="closeMergeLogs" class="close-btn">×</button>
        </div>
        <div class="modal-body">
          <div v-if="loadingMergeLogs" class="loading-state"><span class="loading-spinner"></span>加载中...</div>
          <div v-else-if="!mergeLogs.length" class="empty-state">暂无合并记录</div>
          <div v-else class="merge-logs">
            <div v-for="(log, i) in mergeLogs" :key="i" class="merge-log-item">
              <div class="merge-log-head">
                <b>{{ log.at }}</b>
                <span>主模型: <code>{{ log.main_model_id }}</code></span>
                <span>被合并: <code>{{ (log.merged_model_ids || []).join(', ') }}</code></span>
                <button class="danger-outline small" @click="doRollbackMerge(i)" :disabled="rollingBackMerge">回滚此条</button>
              </div>
              <div class="merge-log-detail">
                <span>{{ (log.classes_added || []).reduce((s: number, c: any) => s + (c.added_classes || []).length, 0) }} 个类别并入主字典</span>
                <span>{{ Object.keys(log.datasets_rebound || {}).reduce((s: number, k: string) => s + (log.datasets_rebound[k] || []).length, 0) }} 个数据集重归属</span>
                <small v-if="log.reason">原因: {{ log.reason }}</small>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 模型详情对话框 -->
    <div v-if="showModelDetails" class="modal-overlay" @click.self="closeModelDetails">
      <div class="modal-content modal-large">
        <div class="modal-header">
          <h3>模型详情: {{ detailModel?.model_id }}</h3>
          <button @click="closeModelDetails" class="close-btn">×</button>
        </div>
        <div class="modal-body">
          <div v-if="loadingDetails" class="loading-state">
            <span class="loading-spinner large"></span>
            <span>加载模型详情...</span>
          </div>
          <template v-else>
            <!-- 基本信息 -->
            <div class="detail-section">
              <h4>基本信息</h4>
              <table class="detail-table">
                <tr><td>模型ID</td><td>{{ detailModel?.model_id }}</td></tr>
                <tr v-if="detailModel?.status"><td>状态</td><td><span :class="'m-status m-' + detailModel.status">{{ statusText(detailModel) }}</span></td></tr>
                <tr v-if="detailModel?.version"><td>版本</td><td>{{ detailModel.version }}（同业务内按 0.1 递增）</td></tr>
                <tr v-if="detailModel?.business"><td>业务场景</td><td>{{ bizLabel(detailModel.business) }}</td></tr>
                <tr v-if="detailModel?.override"><td>人工覆盖</td><td>{{ statusText(null, detailModel) }} <small v-if="detailModel.override.reason">（{{ detailModel.override.reason }}）</small></td></tr>
                <tr><td>基础模型</td><td>{{ detailModel?.base_model || '-' }}</td></tr>
                <tr v-if="detailModel?.lineage?.dataset_id || detailModel?.lineage?.parent_model_id">
                  <td>血缘</td>
                  <td>
                    <span v-if="detailModel.lineage.parent_model_id">上一代: {{ detailModel.lineage.parent_model_id }}<br/></span>
                    <span v-if="detailModel.lineage.dataset_id">数据集: {{ detailModel.lineage.dataset_id }}</span>
                  </td>
                </tr>
                <tr><td>创建时间</td><td>{{ detailModel?.created_at || '-' }}</td></tr>
                <tr><td>类别数量</td><td>{{ detailModel?.classes?.length || 0 }}</td></tr>
                <tr><td>图片尺寸</td><td>{{ detailModel?.imgsz || '-' }}</td></tr>
                <tr><td>训练轮数</td><td>{{ detailModel?.epochs || '-' }}</td></tr>
                <tr v-if="detailModel?.file_size_mb"><td>模型大小</td><td>{{ detailModel.file_size_mb }} MB</td></tr>
              </table>
            </div>

            <!-- 守门员评估报告 -->
            <div v-if="detailModel?.gatekeeper" class="detail-section">
              <h4>🛡️ 模型守门员评估</h4>
              <div class="gk-panel" :class="'gk-panel-' + detailModel.gatekeeper.result">
                <div class="gk-verdict">
                  {{ gkVerdictText(detailModel.gatekeeper) }}
                </div>
                <div v-if="detailModel.gatekeeper.report" class="gk-report-text">{{ detailModel.gatekeeper.report }}</div>
                <div v-if="detailModel.gatekeeper.new_metrics?.mAP50_95 != null" class="gk-metrics">
                  <span v-if="detailModel.gatekeeper.old_metrics?.mAP50_95 != null">
                    新模型 mAP50-95: <b>{{ (detailModel.gatekeeper.new_metrics.mAP50_95 * 100).toFixed(1) }}%</b>
                    vs 旧模型: <b>{{ (detailModel.gatekeeper.old_metrics.mAP50_95 * 100).toFixed(1) }}%</b>
                  </span>
                  <span v-else>mAP50-95: <b>{{ (detailModel.gatekeeper.new_metrics.mAP50_95 * 100).toFixed(1) }}%</b></span>
                  <span v-if="detailModel.gatekeeper.eval_split"> · 评估集: {{ detailModel.gatekeeper.eval_split === 'test' ? '独立测试集 test' : '验证集 val' }}</span>
                </div>
                <details v-if="detailModel.gatekeeper.class_ap && Object.keys(detailModel.gatekeeper.class_ap).length">
                   <summary>各类别 AP 对比（防偏科校验）</summary>
                   <table class="detail-table class-ap-table">
                     <tr><td>类别</td><td>旧模型 AP</td><td>新模型 AP</td><td>相对变化</td></tr>
                     <tr v-for="(row, cls) in detailModel.gatekeeper.class_ap" :key="cls">
                       <td>{{ cls }}</td>
                       <td>{{ row.old_ap == null ? '—' : (row.old_ap * 100).toFixed(1) + '%' }}</td>
                       <td>{{ row.new_ap == null ? '—' : (row.new_ap * 100).toFixed(1) + '%' }}</td>
                       <td :class="{ 'ap-regressed': row.regressed }">
                         {{ row.delta_pct == null ? '—' : (row.delta_pct > 0 ? '+' : '') + row.delta_pct + '%' }}
                         <span v-if="row.regressed" class="ap-warn">⚠️ 退化</span>
                       </td>
                     </tr>
                   </table>
                 </details>
               </div>
             </div>

            <!-- 关键参数表 -->
            <div v-if="modelKeyParams.length" class="detail-section">
              <h4>关键参数</h4>
              <table class="detail-table">
                <tr v-for="row in modelKeyParams" :key="row.label">
                  <td>{{ row.label }}</td>
                  <td>{{ row.value }}</td>
                </tr>
              </table>
            </div>

            <!-- 模型报告（点击打开弹窗） -->
            <div class="detail-section">
              <div class="report-header">
                <h4>模型报告</h4>
                <button class="primary" @click="openModelReport">查看模型报告</button>
              </div>
              <p class="form-hint">包含性能总结、诊断报告、训练配置与全部训练图表</p>
            </div>
          </template>
        </div>
      </div>
    </div>

    <!-- 模型报告弹窗 -->
    <div v-if="showModelReport" class="modal-overlay" @click.self="closeModelReport">
      <div class="modal-content modal-large report-modal">
        <div class="modal-header">
          <h3>模型报告: {{ detailModel?.model_id }}</h3>
          <button @click="closeModelReport" class="close-btn">×</button>
        </div>
        <!-- 顶部固定下载栏（滚动时不移动） -->
        <div class="report-download-bar">
          <span class="report-bar-title">📄 模型诊断报告</span>
          <button class="primary" @click="downloadModelReport" :disabled="!modelReportText">
            <span v-if="downloadingChart === 'report'" class="loading-spinner"></span>
            ⬇ 下载完整报告（.html）
          </button>
        </div>
        <div class="modal-body">
          <!-- 性能总结 -->
          <div v-if="modelSummary" class="detail-section">
            <h4>性能总结</h4>
            <div class="summary-box">
              <div class="summary-verdict" :class="'verdict-' + modelSummary.level">
                <span class="verdict-badge">{{ modelSummary.verdict }}</span>
                <span class="verdict-desc">{{ modelSummary.overall }}</span>
              </div>
              <ul class="summary-list">
                <li v-for="(item, i) in modelSummary.items" :key="i" :class="'summary-'+item.type">
                  <span class="summary-icon">{{ item.icon }}</span>
                  <span>{{ item.text }}</span>
                </li>
              </ul>
            </div>
          </div>

          <!-- 诊断报告 -->
          <div class="detail-section">
            <h4>诊断报告</h4>
            <pre v-if="modelReportText" class="report-pre">{{ modelReportText }}</pre>
            <p v-else class="empty-state">暂无训练指标，无法生成报告</p>
            <div class="report-exec-bar">
              <button class="primary" @click="onOneClickExecute">一键执行</button>
            </div>
          </div>

          <!-- 训练配置 -->
          <div v-if="detailModel?.training_metrics?.job_config" class="detail-section">
            <h4>训练配置</h4>
            <table class="detail-table">
              <tr><td>数据集</td><td>{{ detailModel.training_metrics.job_config.dataset_id || '-' }}</td></tr>
              <tr><td>训练轮数</td><td>{{ detailModel.training_metrics.job_config.epochs || '-' }}</td></tr>
              <tr><td>批次大小</td><td>{{ detailModel.training_metrics.job_config.batch || '-' }}</td></tr>
              <tr><td>训练状态</td><td>{{ detailModel.training_metrics.job_config.status || '-' }}</td></tr>
              <tr v-if="detailModel.training_metrics.job_config.completed_at">
                <td>完成时间</td>
                <td>{{ detailModel.training_metrics.job_config.completed_at }}</td>
              </tr>
            </table>
          </div>

          <!-- 训练图表 -->
          <div v-if="finalMetricsOption" class="detail-section">
            <h4>最终训练指标</h4>
            <div class="chart-container">
              <v-chart :option="finalMetricsOption" autoresize />
            </div>
          </div>
          
          <div v-if="lossChartOption" class="detail-section">
            <h4>训练损失曲线</h4>
            <div class="chart-container">
              <v-chart :option="lossChartOption" autoresize />
            </div>
          </div>
          
          <div v-if="mapChartOption" class="detail-section">
            <h4>训练指标曲线</h4>
            <div class="chart-container">
              <v-chart :option="mapChartOption" autoresize />
            </div>
          </div>
          
          <div v-if="radarChartOption" class="detail-section">
            <h4>指标雷达图</h4>
            <div class="chart-container">
              <v-chart :option="radarChartOption" autoresize />
            </div>
          </div>
          
          <div v-if="gaugeChartOption" class="detail-section">
            <h4>mAP50 评分仪表</h4>
            <div class="chart-container chart-gauge">
              <v-chart :option="gaugeChartOption" autoresize />
            </div>
          </div>
          
          <div v-if="lossAreaOption" class="detail-section">
            <h4>训练损失面积图</h4>
            <div class="chart-container">
              <v-chart :option="lossAreaOption" autoresize />
            </div>
          </div>
          
          <div v-if="lossScatterOption" class="detail-section">
            <h4>损失分布散点图</h4>
            <div class="chart-container">
              <v-chart :option="lossScatterOption" autoresize />
            </div>
          </div>
          
          <div v-if="heatmapChartOption" class="detail-section">
            <h4>训练指标热力图</h4>
            <div class="chart-container chart-large">
              <v-chart :option="heatmapChartOption" autoresize />
            </div>
          </div>

          <!-- 类别列表 -->
          <div v-if="detailModel?.classes" class="detail-section">
            <h4>类别列表 ({{ detailModel.classes.length }})</h4>
            <div class="class-tags">
              <span v-for="(cls, idx) in detailModel.classes" :key="idx" class="class-tag">
                {{ idx }}: {{ cls }}
              </span>
            </div>
          </div>

          <!-- 描述 -->
          <div v-if="detailModel?.description" class="detail-section">
            <h4>描述</h4>
            <p>{{ detailModel.description }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onActivated, computed } from 'vue'
import { listModels, deleteModel, getModel, updateModel, uploadModel, exportModel, generateTrainingCharts, findSimilarModels, mergeModels, getMergeLogs, rollbackMerge, listPendingPublish, confirmPublish, listLibraries, type ModelDetails } from '@/api/models'
import { downloadFile } from '@/utils/download'
import { showAlert, showConfirm, showPrompt } from '@/composables/useDialog'
import LibraryCreateModal from '@/components/LibraryCreateModal.vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, BarChart, ScatterChart, RadarChart, HeatmapChart, GaugeChart } from 'echarts/charts'
import { 
  GridComponent, 
  TooltipComponent, 
  LegendComponent, 
  TitleComponent,
  RadarComponent,
  VisualMapComponent,
  PolarComponent
} from 'echarts/components'

// 注册 ECharts 组件
use([
  CanvasRenderer, 
  LineChart, 
  BarChart, 
  ScatterChart, 
  RadarChart, 
  HeatmapChart,
  GaugeChart,
  GridComponent, 
  TooltipComponent, 
  LegendComponent, 
  TitleComponent,
  RadarComponent,
  VisualMapComponent,
  PolarComponent
])

const models = ref<any[]>([])

// ===== 规范 §2.1 模型列表展示辅助 =====
const moreOpenId = ref<string | null>(null)
// 「⋯」悬停菜单的关闭缓冲：指针从按钮移向选项时，哪怕只穿过几个像素的空隙
// 都会触发一次 mouseleave，立即置空会让菜单在人还没点到选项前就消失。
// 这里延后关闭，并在指针回到按钮/菜单范围内时取消（见 wrapper 上的 mouseenter）。
let menuCloseTimer: ReturnType<typeof setTimeout> | null = null
const cancelMenuClose = () => {
  if (menuCloseTimer) {
    clearTimeout(menuCloseTimer)
    menuCloseTimer = null
  }
}
const scheduleMenuClose = () => {
  const id = moreOpenId.value
  if (!id) return
  cancelMenuClose()
  menuCloseTimer = setTimeout(() => {
    menuCloseTimer = null
    // 只对当时那一项生效：期间若用户已切到别的行的菜单，不能把新开的关掉
    if (moreOpenId.value === id) moreOpenId.value = null
  }, 160)
}
const closeMenu = () => {
  cancelMenuClose()
  moreOpenId.value = null
}
const toggleMenu = (modelId: string) => {
  cancelMenuClose()
  moreOpenId.value = moreOpenId.value === modelId ? null : modelId
}
// 算法架构：优先使用 M11 解析出的架构字段；存量模型按血缘/基座文件名兜底（如 yolov8n.pt → yolov8n）
const modelArch = (m: any) => {
  if (m?.architecture_name) return m.model_scale ? `${m.architecture_name}${m.model_scale}` : m.architecture_name
  const src = m?.lineage?.base_model || m?.base_model || m?.model_name || ''
  const mm = /(yolov?\d+[a-z]*?)([nslmxbce])\.pt/i.exec(src)
  return mm ? mm[0].replace(/\.pt$/i, '') : '—'
}
// 核心指标：守门员评估的 mAP50-95
const modelMapText = (m: any) => {
  const v = m?.gatekeeper?.new_metrics?.mAP50_95
  return v == null ? '—' : (v * 100).toFixed(1) + '%'
}
const modelDatasetText = (m: any) => m?.lineage?.dataset_id || '—'
// 超参摘要：Epochs / Batch / Size（列表字段缺失时兜底 training_metrics.job_config）
const modelHpsText = (m: any) => {
  const jc = m?.training_metrics?.job_config || {}
  const parts: string[] = []
  const e = m?.epochs ?? jc.epochs
  const b = jc.batch
  const s = m?.imgsz ?? jc.imgsz
  if (e != null) parts.push(`Epochs: ${e}`)
  if (b != null) parts.push(`Batch: ${b}`)
  if (s != null) parts.push(`Size: ${s}`)
  return parts.length ? parts.join(' / ') : '—'
}
const fmtDateTime = (s?: string) => (s ? s.replace('T', ' ').slice(0, 19).replace('Z', '') : '—')

// ===== 模型仓库 / 守门员辅助 =====
const businessScenes: { value: string; label: string }[] = [
  { value: 'general', label: '通用目标检测' },
  { value: 'pedestrian', label: '行人检测' },
  { value: 'vehicle', label: '车辆 / 车牌' },
  { value: 'defect', label: '工业缺陷检测' },
  { value: 'package', label: '包裹 / 物流' }
]
const bizLabel = (v?: string) => businessScenes.find(b => b.value === v)?.label || v || '通用目标检测'

const statusText = (meta?: any, overrideInfo?: any) => {
  const s = overrideInfo?.to_status || overrideInfo?.from_status || meta?.status
  switch (s) {
    case 'production_ready': return meta?.override || overrideInfo?.to_status ? '生产就绪（人工覆盖）' : '✅ 生产就绪'
    case 'production': return '🏭 生产中'
    case 'active': return '已发布'
    case 'archived': return '已归档'
    case 'rejected': return '❌ 已淘汰'
    case 'superseded': return '⚠️ 已退役'
    case 'training': return '⏳ 训练中'
    case 'evaluating': return '🔍 评估中'
    default: return s || '未知状态'
  }
}

const gkVerdictText = (gk: any) => {
  switch (gk?.result) {
    case 'promoted': return '✅ 判定结果：晋升（Promote）—— 新模型已通过守门员校验，正式成为生产版本'
    case 'first_version': return '✅ 判定结果：首版直晋 —— 该业务首个模型，无旧基准可对比，直接设为生产版本 v1.0'
    case 'rejected': return '❌ 判定结果：淘汰（Reject）—— 未通过守门员校验，已归档为失败实验，旧模型继续服役'
    default: return '🔍 判定结果：' + (gk?.result || '未知')
  }
}

// 模型详情
const showModelDetails = ref(false)
const detailModel = ref<ModelDetails | null>(null)
const loadingDetails = ref(false)

// Loading 状态
const loadingModels = ref(false)
const deletingModel = ref<string | null>(null)
const exportingModel = ref<string | null>(null)
const downloadingChart = ref<string | null>(null)

// 模型上传
const selectedModelFile = ref<File | null>(null)
const modelUploadName = ref('')
const uploadingModel = ref(false)
// 上传时直接归入的算法库（留空 = 独立模型）；新建库弹窗开关
const modelUploadLib = ref('')
const newLibOpen = ref(false)

const loadModels = async (silent = false) => {
  if (!silent) loadingModels.value = true
  try {
    const data = await listModels()
    // 过滤首页「新建算法库」产生的空白占位记录（empty），它们属于算法库层，不与训练产物并排展示
    models.value = (data.models || []).filter((m: any) => !m.empty)
    // 仅当列表里确有已归库模型时，才取库名回显下拉选中项（否则不发这次请求）
    if (models.value.some((m: any) => modelLibId(m))) ensureLibraries()
  } catch (error: any) {
    alert('加载模型失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    loadingModels.value = false
  }
}

// ===== 算法库（仅作模型表「算法库」列下拉数据源；创建/管理在首页） =====
const libraries = ref<any[]>([])
const librariesLoaded = ref(false)
let librariesLoading: Promise<void> | null = null
const assigningLibId = ref<string | null>(null)

// 按需加载：只有点开「算法库」下拉时才请求一次（本页主流程是模型列表，
// 算法库取不到不应打断浏览，故失败时静默记录、不弹窗）
const ensureLibraries = () => {
  if (librariesLoaded.value || librariesLoading) return
  librariesLoading = listLibraries()
    .then((res) => {
      libraries.value = res.libraries || []
      librariesLoaded.value = true
    })
    .catch((e: any) => {
      console.warn('加载算法库失败:', e.response?.data?.detail || e.message)
    })
    .finally(() => {
      librariesLoading = null
    })
}

// 当前归库标识：手动归库写的 model_library_id 优先，其次平台训练自带的库根
const modelLibId = (model: any) => {
  if (model.model_library_id) return model.model_library_id
  const root = model.library_root_id
  return root && root !== model.model_id ? root : ''
}

// 模型归入/移出算法库（仅写归库标记，不触发其他连锁操作）
const assignLibrary = async (model: any, libId: string) => {
  const target = libId || ''
  const libName = target ? (libraries.value.find(l => l.model_library_id === target)?.display_name || target) : '独立模型'
  if (!(await showConfirm(`将模型「${model.display_name || model.model_id}」归入算法库「${libName}」？\n仅影响算法库统计展示，可随时改回。`))) return
  assigningLibId.value = model.model_id
  try {
    await updateModel(model.model_id, { model_library_id: target })
    // 同步更新本地行，避免整表重载（模型列表数据量大）
    if (target) {
      model.model_library_id = target
      model.library_root_id = target
    } else {
      delete model.model_library_id
      delete model.library_root_id
    }
  } catch (e: any) {
    alert('归库失败: ' + (e.response?.data?.detail || e.message))
    await loadModels()
  } finally {
    assigningLibId.value = null
  }
}

// ===== P8 半自动发布：待确认发布（守门员通过 → 管理员确认 → 唯一生产版本） =====
const COOLDOWN_HOURS = 24                           // 与后端 PUBLISH_CONFIRM_COOLDOWN_HOURS 对齐
const pendingModels = ref<any[]>([])
const pendingLoading = ref(false)
const pendingError = ref('')
const publishingModel = ref<string | null>(null)

const fmtTime = (s?: string) => (s ? s.replace('T', ' ').slice(0, 19).replace('Z', '') : '-')

// 冷却剩余小时（ready_at + 24h vs 现在）；无 ready_at（历史模型）视为冷却完成
const cooldownRemaining = (readyAt?: string) => {
  if (!readyAt) return 0
  const t = new Date(readyAt).getTime()
  if (isNaN(t)) return 0
  const remain = (t + COOLDOWN_HOURS * 3_600_000 - Date.now()) / 3_600_000
  return remain > 0 ? remain : 0
}

const loadPendingPublish = async () => {
  pendingLoading.value = true
  pendingError.value = ''
  try {
    const res = await listPendingPublish()
    pendingModels.value = res.models || []
  } catch (e: any) {
    pendingError.value = '加载待确认发布失败: ' + (e.response?.data?.detail || e.message)
  } finally {
    pendingLoading.value = false
  }
}

const confirmPublishItem = async (m: any) => {
  const force = cooldownRemaining(m.ready_at) > 0
  const ok = await showConfirm(
    `确认发布模型 ${m.model_id}（版本 ${m.version || '-'}）？\n\n` +
    `发布后将成为该业务（${bizLabel(m.business)}）唯一生产版本，` +
    `同业务其他已发布生产模型将自动退役（superseded）。` +
    (force ? `\n\n⚠ 模型仍在 ${COOLDOWN_HOURS} 小时冷却期内，本次将使用"强制确认"（跳过冷却判定）。` : ''),
    '确认发布'
  )
  if (!ok) return
  publishingModel.value = m.model_id
  try {
    const res = await confirmPublish(m.model_id, { force })
    const superseded = (res.superseded || []).length
    showAlert(
      `✅ 已确认发布 ${m.model_id}！\n` +
      `publish_status=published（${res.meta?.published_at || ''}）` +
      (superseded ? `\n同时退役了 ${superseded} 个原生产模型（已标记为"已退役"）。` : '')
    )
    await loadPendingPublish()
    await loadModels()
  } catch (e: any) {
    alert('确认发布失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    publishingModel.value = null
  }
}

const viewModelDetails = async (model: any) => {
  showModelDetails.value = true
  loadingDetails.value = true
  
  try {
    // 获取完整的模型详情（包含训练指标）
    const details = await getModel(model.model_id)
    detailModel.value = details
  } catch (error: any) {
    console.error('获取模型详情失败:', error)
    detailModel.value = model  // 降级使用基本信息
  } finally {
    loadingDetails.value = false
  }
}

const closeModelDetails = () => {
  showModelDetails.value = false
  detailModel.value = null
}

// 打开/关闭模型报告弹窗
const openModelReport = () => {
  showModelReport.value = true
}
const closeModelReport = () => {
  showModelReport.value = false
}

// 一键执行（暂只添加按钮，动作待后续实现）
const onOneClickExecute = () => {
  showAlert('"一键执行"功能开发中，敬请期待。', '一键执行')
}

// 性能总结：根据最终指标生成自然语言评估（性能 + 可用度）
const modelSummary = computed(() => {
  const metrics = detailModel.value?.training_metrics?.training_history?.final_metrics
  if (!metrics) return null
  const mAP50 = metrics.mAP50
  const mAP50_95 = metrics.mAP50_95
  const precision = metrics.precision
  const recall = metrics.recall
  if (mAP50 == null || mAP50_95 == null) return null

  const items: { type: 'good' | 'warn' | 'bad'; icon: string; text: string }[] = []
  let level: 'good' | 'warn' | 'bad' = 'good'
  let verdict = '性能良好'

  // 规则1: mAP50 整体水平
  if (mAP50 < 0.3) {
    level = 'bad'
    verdict = '性能较差'
    items.push({ type: 'bad', icon: '⚠️', text: `mAP50 仅 ${(mAP50 * 100).toFixed(1)}%，整体偏低，建议增加数据量或数据增强，或延长训练。` })
  } else if (mAP50 < 0.5) {
    level = 'warn'
    verdict = '性能一般'
    items.push({ type: 'warn', icon: '⚠️', text: `mAP50 为 ${(mAP50 * 100).toFixed(1)}%，中等偏低，可考虑更多训练或数据增强。` })
  } else {
    items.push({ type: 'good', icon: '✅', text: `mAP50 为 ${(mAP50 * 100).toFixed(1)}%，目标定位能力良好。` })
  }

  // 规则2/3: 精确-召回平衡
  if (precision != null && recall != null) {
    if (precision < 0.5 && recall > 0.8) {
      level = 'warn'
      items.push({ type: 'warn', icon: '⚖️', text: `高召回(${(recall * 100).toFixed(1)}%)低精确(${(precision * 100).toFixed(1)}%)，模型过于激进，误报频繁。` })
    } else if (precision > 0.8 && recall < 0.5) {
      level = 'warn'
      items.push({ type: 'warn', icon: '⚖️', text: `高精确(${(precision * 100).toFixed(1)}%)低召回(${(recall * 100).toFixed(1)}%)，模型过于保守，漏检严重。` })
    } else {
      items.push({ type: 'good', icon: '⚖️', text: `Precision ${(precision * 100).toFixed(1)}% / Recall ${(recall * 100).toFixed(1)}%，精确与召回较为均衡。` })
    }
  }

  // mAP50-95 严格阈值表现
  items.push({
    type: mAP50_95 >= 0.5 ? 'good' : 'warn',
    icon: '📈',
    text: `mAP50-95 为 ${(mAP50_95 * 100).toFixed(1)}%，${mAP50_95 >= 0.5 ? '在严格 IoU 阈值下表现稳定。' : '在严格 IoU 阈值下表现一般，定位精度有待提升。'}`
  })

  // 可用度评估
  let overall = ''
  if (level === 'good') {
    overall = '该模型性能良好，可用度高，可直接用于 AI 预标注或生产环境。'
  } else if (level === 'warn') {
    overall = '该模型性能一般，可用度中等，建议优化数据或继续训练后再投入使用。'
  } else {
    overall = '该模型性能较差，可用度低，不建议直接用于预标注，建议扩充数据并重新训练。'
  }

  return { level, verdict, items, overall }
})

// 关键参数表格数据
const modelKeyParams = computed(() => {
  const metrics = detailModel.value?.training_metrics?.training_history?.final_metrics
  const info = detailModel.value?.model_info
  const rows: { label: string; value: string }[] = []

  if (metrics?.mAP50 != null) rows.push({ label: 'mAP50', value: (metrics.mAP50 * 100).toFixed(1) + '%' })
  if (metrics?.mAP50_95 != null) rows.push({ label: 'mAP50-95', value: (metrics.mAP50_95 * 100).toFixed(1) + '%' })
  if (metrics?.precision != null) rows.push({ label: 'Precision', value: (metrics.precision * 100).toFixed(1) + '%' })
  if (metrics?.recall != null) rows.push({ label: 'Recall', value: (metrics.recall * 100).toFixed(1) + '%' })
  if (metrics?.precision != null && metrics?.recall != null) {
    const p = metrics.precision, r = metrics.recall
    const f1 = (p + r) === 0 ? 0 : (2 * p * r) / (p + r)
    rows.push({ label: 'F1', value: (f1 * 100).toFixed(1) + '%' })
  }
  if (info?.total_params_m) rows.push({ label: '参数数量', value: info.total_params_m + 'M' })
  if (info?.task) rows.push({ label: '任务类型', value: info.task })
  return rows
})

// 模型报告：参照 ModelDiagnosticAdvisor 逻辑生成可读的诊断报告
const showModelReport = ref(false)
const modelReportText = computed(() => {
  const history = detailModel.value?.training_metrics?.training_history
  const metrics = history?.final_metrics
  if (!metrics) return ''

  // 统一转为百分比（0-1 自动放大为 0-100）
  const toPct = (v: any) => (v == null ? 0 : (v <= 1 && v >= -1 ? v * 100 : v))
  const map50 = toPct(metrics.mAP50)
  const precision = toPct(metrics.precision)
  const recall = toPct(metrics.recall)
  const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0

  // 损失取最后一个 epoch
  const train_loss = history.train_box_loss?.[history.train_box_loss.length - 1] ?? 0
  const val_loss = history.val_box_loss?.[history.val_box_loss.length - 1] ?? 0

  const diagnosis: string[] = []
  const suggestions: string[] = []

  // 1. 总体评价
  let status_level = '🔴 较差'
  if (map50 >= 80) status_level = '🟢 优秀'
  else if (map50 >= 60) status_level = '🔵 良好'
  else if (map50 >= 40) status_level = '🟠 一般'

  // 2. P-R 平衡诊断
  if (precision > 85 && recall < 30) {
    diagnosis.push('模型过于保守（高精准，低召回）。')
    diagnosis.push('模型非常有把握时才敢画框，导致大量目标被漏检。')
    suggestions.push('📉 降低推理阈值：尝试将部署时的 Confidence Threshold 从 0.5 降至 0.25-0.3。')
    suggestions.push('🔄 数据增强：增加 Mosaic 或 Mixup 增强比例，强迫模型学习局部特征。')
  } else if (precision < 40 && recall > 70) {
    diagnosis.push('模型过于敏感（低精准，高召回）。')
    diagnosis.push('模型存在大量误检（把背景当目标），或者框画得不准。')
    suggestions.push('📈 提高推理阈值：尝试提高 Confidence Threshold 至 0.6-0.7。')
    suggestions.push('🚫 增加负样本：在训练集中加入一些不包含目标的背景图片（Empty images）。')
    suggestions.push('🏷️ 检查标注：检查是否有标注框过大或包含过多背景的情况。')
  } else if (precision < 40 && recall < 40) {
    diagnosis.push('模型尚未收敛或学习能力不足。')
    diagnosis.push('精确率和召回率双低，说明模型还没学会特征。')
    suggestions.push('📚 增加数据量：当前数据可能不足以支撑模型学习。')
    suggestions.push('⏳ 增加训练轮数：目前的 Epochs 可能不够，建议继续训练。')
    suggestions.push('🔍 检查标注质量：排查是否存在大量标注错误或标签混淆。')
  } else {
    diagnosis.push('模型的精确率与召回率较为均衡，无明显偏差。')
    suggestions.push('✔️ 若追求更高精度，可在更高 IoU 阈值下评估并微调。')
  }

  // 3. 过拟合/欠拟合检测（基于 Loss）
  if (train_loss > 0 && val_loss > 0) {
    const loss_gap = val_loss - train_loss
    if (loss_gap > train_loss * 0.5) {
      diagnosis.push('⚠️ 检测到过拟合风险。')
      suggestions.push('💊 正则化：增加 Weight Decay 或 Dropout。')
      suggestions.push('🛑 早停机制：建议在验证集 Loss 开始上升时停止训练。')
    }
  }

  const map50_95 = toPct(metrics.mAP50_95)

  let report = `### 📊 模型诊断报告\n\n`
  report += `**综合评分**: ${status_level} (mAP50: ${map50.toFixed(1)}%, F1-Score: ${f1.toFixed(1)}%)\n\n`
  if (map50_95 > 0) {
    report += `**⚡ 综合性能**: mAP50-95 为 ${map50_95.toFixed(1)}%\n\n`
  }
  report += `**🩺 现象解读**:\n`
  if (diagnosis.length === 0) {
    report += `- 模型各项指标表现均衡，未发现明显问题。\n`
  } else {
    diagnosis.forEach((d) => { report += `- ${d}\n` })
  }
  report += `\n**💡 推荐操作方案**:\n`
  if (suggestions.length === 0) {
    report += `1. 保持当前训练配置；若有更高精度需求，可增加训练轮数或数据量后继续训练。\n`
  } else {
    suggestions.forEach((s, i) => { report += `${i + 1}. ${s}\n` })
  }
  return report
})

// 将 Blob 转为 base64 字符串（用于内嵌到 HTML 报告）
const blobToBase64 = (blob: Blob): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      const idx = result.indexOf(',')
      resolve(idx >= 0 ? result.slice(idx + 1) : result)
    }
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })

// 简单 Markdown → HTML（覆盖报告用到的标题、加粗、列表、段落）
const mdToHtml = (md: string): string => {
  const esc = (s: string) =>
    s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/\n/g, '<br>')
  const inline = (s: string) =>
    esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  const lines = md.split('\n')
  let html = ''
  let inList = false
  const closeList = () => {
    if (inList) { html += '</ul>\n'; inList = false }
  }
  for (const raw of lines) {
    const line = raw.trimEnd()
    if (line.startsWith('### ')) { closeList(); html += `<h3>${inline(line.slice(4))}</h3>\n` }
    else if (line.startsWith('## ')) { closeList(); html += `<h2>${inline(line.slice(3))}</h2>\n` }
    else if (line.match(/^[-*]\s/)) {
      if (!inList) { html += '<ul>\n'; inList = true }
      html += `<li>${inline(line.replace(/^[-*]\s/, ''))}</li>\n`
    } else if (line.match(/^\d+\.\s/)) {
      if (!inList) { html += '<ul>\n'; inList = true }
      html += `<li>${inline(line.replace(/^\d+\.\s/, ''))}</li>\n`
    } else if (line.trim() === '') { closeList(); html += '<br>\n' }
    else { closeList(); html += `<p>${inline(line)}</p>\n` }
  }
  closeList()
  return html
}

// 组装完整的 HTML 模型报告（诊断文字 + 训练图表，图表内嵌 base64）
const buildReportHtml = (md: string, lossBase64: string, metricsBase64: string, model: any): string => {
  const title = `模型报告: ${model?.model_id || 'model'}`
  const lossImg = lossBase64
    ? `<div class="chart"><h3>训练损失曲线</h3><img src="data:image/png;base64,${lossBase64}" alt="损失曲线"></div>`
    : ''
  const metricsImg = metricsBase64
    ? `<div class="chart"><h3>训练指标曲线</h3><img src="data:image/png;base64,${metricsBase64}" alt="指标曲线"></div>`
    : ''
  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>${title}</title>
<style>
  body{font-family:"Microsoft YaHei","PingFang SC",sans-serif;color:#24292f;margin:32px auto;max-width:900px;padding:0 20px;line-height:1.8}
  h1{font-size:24px;border-bottom:2px solid #0969da;padding-bottom:10px}
  h2{font-size:20px;margin-top:28px;color:#0969da}
  h3{font-size:17px;margin-top:20px}
  p{margin:8px 0}
  ul{margin:8px 0 12px;padding-left:22px}
  li{margin:4px 0}
  .chart{margin:24px 0}
  .chart img{width:100%;max-width:860px;border:1px solid #e0e3e8;border-radius:6px}
  strong{color:#111}
</style>
</head>
<body>
<h1>${title}</h1>
${mdToHtml(md)}
${lossImg}
${metricsImg}
</body>
</html>`
}

// 下载模型报告为完整 HTML 文件（含诊断文字与训练图表）
const downloadModelReport = async () => {
  if (!modelReportText.value) return
  const modelId = detailModel.value?.model_id || 'model'
  downloadingChart.value = 'report'
  try {
    // 获取训练图表并内嵌到报告中，保证下载内容完整
    let lossBase64 = '', metricsBase64 = ''
    try {
      const [lossBlob, metricsBlob] = await Promise.all([
        generateTrainingCharts(modelId, 'loss'),
        generateTrainingCharts(modelId, 'metrics')
      ])
      ;[lossBase64, metricsBase64] = [await blobToBase64(lossBlob), await blobToBase64(metricsBlob)]
    } catch (e) {
      console.warn('获取训练图表失败，报告将仅包含诊断文字:', e)
    }
    const html = buildReportHtml(modelReportText.value, lossBase64, metricsBase64, detailModel.value)
    const blob = new Blob([html], { type: 'text/html;charset=utf-8' })
    await downloadFile(blob, `${modelId}_report.html`)
  } finally {
    downloadingChart.value = null
  }
}

// 训练损失图表配置
const lossChartOption = computed(() => {
  const history = detailModel.value?.training_metrics?.training_history
  if (!history || !history.epochs) return null
  
  const series: any[] = []
  
  if (history.train_box_loss) {
    series.push({
      name: 'Train Box Loss',
      type: 'line',
      data: history.train_box_loss,
      smooth: true
    })
  }
  if (history.train_cls_loss) {
    series.push({
      name: 'Train Cls Loss',
      type: 'line',
      data: history.train_cls_loss,
      smooth: true
    })
  }
  if (history.val_box_loss) {
    series.push({
      name: 'Val Box Loss',
      type: 'line',
      data: history.val_box_loss,
      smooth: true,
      lineStyle: { type: 'dashed' }
    })
  }
  if (history.val_cls_loss) {
    series.push({
      name: 'Val Cls Loss',
      type: 'line',
      data: history.val_cls_loss,
      smooth: true,
      lineStyle: { type: 'dashed' }
    })
  }
  
  if (series.length === 0) return null
  
  return {
    title: { text: '训练损失曲线', left: 'center' },
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0 },
    xAxis: {
      type: 'category',
      data: history.epochs,
      name: 'Epoch'
    },
    yAxis: { type: 'value', name: 'Loss' },
    series
  }
})

// mAP 图表配置
const mapChartOption = computed(() => {
  const history = detailModel.value?.training_metrics?.training_history
  if (!history || !history.epochs) return null
  
  const series: any[] = []
  
  if (history.metrics_mAP50) {
    series.push({
      name: 'mAP50',
      type: 'line',
      data: history.metrics_mAP50,
      smooth: true
    })
  }
  if (history.metrics_mAP50_95) {
    series.push({
      name: 'mAP50-95',
      type: 'line',
      data: history.metrics_mAP50_95,
      smooth: true
    })
  }
  if (history.metrics_precision) {
    series.push({
      name: 'Precision',
      type: 'line',
      data: history.metrics_precision,
      smooth: true
    })
  }
  if (history.metrics_recall) {
    series.push({
      name: 'Recall',
      type: 'line',
      data: history.metrics_recall,
      smooth: true
    })
  }
  
  if (series.length === 0) return null
  
  return {
    title: { text: '训练指标曲线', left: 'center' },
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0 },
    xAxis: {
      type: 'category',
      data: history.epochs,
      name: 'Epoch'
    },
    yAxis: { type: 'value', name: 'Value', max: 1 },
    series
  }
})

// 最终指标柱状图
const finalMetricsOption = computed(() => {
  const metrics = detailModel.value?.training_metrics?.training_history?.final_metrics
  if (!metrics) return null
  
  const data = []
  const labels = []
  
  if (metrics.mAP50 !== null && metrics.mAP50 !== undefined) {
    labels.push('mAP50')
    data.push((metrics.mAP50 * 100).toFixed(1))
  }
  if (metrics.mAP50_95 !== null && metrics.mAP50_95 !== undefined) {
    labels.push('mAP50-95')
    data.push((metrics.mAP50_95 * 100).toFixed(1))
  }
  if (metrics.precision !== null && metrics.precision !== undefined) {
    labels.push('Precision')
    data.push((metrics.precision * 100).toFixed(1))
  }
  if (metrics.recall !== null && metrics.recall !== undefined) {
    labels.push('Recall')
    data.push((metrics.recall * 100).toFixed(1))
  }
  
  if (data.length === 0) return null
  
  return {
    title: { text: '最终训练指标 (%)', left: 'center' },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: labels },
    yAxis: { type: 'value', max: 100 },
    series: [{
      type: 'bar',
      data: data,
      itemStyle: {
        color: function(params: any) {
          const colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666']
          return colors[params.dataIndex % colors.length]
        }
      },
      label: {
        show: true,
        position: 'top',
        formatter: '{c}%'
      }
    }]
  }
})

// 损失散点图 - 显示训练和验证损失的分布
const lossScatterOption = computed(() => {
  const history = detailModel.value?.training_metrics?.training_history
  if (!history || !history.epochs) return null
  
  const trainData: number[][] = []
  const valData: number[][] = []
  
  if (history.train_box_loss && history.train_cls_loss) {
    history.epochs.forEach((epoch: number, idx: number) => {
      const boxLoss = history.train_box_loss![idx]
      const clsLoss = history.train_cls_loss![idx]
      if (boxLoss !== undefined && clsLoss !== undefined) {
        trainData.push([boxLoss, clsLoss, epoch])
      }
    })
  }
  
  if (history.val_box_loss && history.val_cls_loss) {
    history.epochs.forEach((epoch: number, idx: number) => {
      const boxLoss = history.val_box_loss![idx]
      const clsLoss = history.val_cls_loss![idx]
      if (boxLoss !== undefined && clsLoss !== undefined) {
        valData.push([boxLoss, clsLoss, epoch])
      }
    })
  }
  
  if (trainData.length === 0 && valData.length === 0) return null
  
  return {
    title: { text: '损失分布散点图', left: 'center' },
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        return `${params.seriesName}<br/>Box Loss: ${params.data[0].toFixed(4)}<br/>Cls Loss: ${params.data[1].toFixed(4)}<br/>Epoch: ${params.data[2]}`
      }
    },
    legend: { bottom: 0 },
    xAxis: { type: 'value', name: 'Box Loss', scale: true },
    yAxis: { type: 'value', name: 'Cls Loss', scale: true },
    series: [
      {
        name: 'Train Loss',
        type: 'scatter',
        data: trainData,
        symbolSize: 10,
        itemStyle: { color: '#5470c6' }
      },
      {
        name: 'Val Loss',
        type: 'scatter',
        data: valData,
        symbolSize: 10,
        itemStyle: { color: '#ee6666' }
      }
    ]
  }
})

// 雷达图 - 多维指标对比
const radarChartOption = computed(() => {
  const metrics = detailModel.value?.training_metrics?.training_history?.final_metrics
  if (!metrics) return null
  
  const indicator = []
  const values = []
  
  if (metrics.mAP50 !== null && metrics.mAP50 !== undefined) {
    indicator.push({ name: 'mAP50', max: 1 })
    values.push(metrics.mAP50)
  }
  if (metrics.mAP50_95 !== null && metrics.mAP50_95 !== undefined) {
    indicator.push({ name: 'mAP50-95', max: 1 })
    values.push(metrics.mAP50_95)
  }
  if (metrics.precision !== null && metrics.precision !== undefined) {
    indicator.push({ name: 'Precision', max: 1 })
    values.push(metrics.precision)
  }
  if (metrics.recall !== null && metrics.recall !== undefined) {
    indicator.push({ name: 'Recall', max: 1 })
    values.push(metrics.recall)
  }
  
  if (indicator.length < 3) return null  // 雷达图至少需要3个维度
  
  return {
    title: { text: '指标雷达图', left: 'center' },
    tooltip: {
      trigger: 'item'
    },
    radar: {
      indicator: indicator,
      shape: 'polygon',
      splitNumber: 5,
      axisName: {
        color: '#666'
      },
      splitLine: {
        lineStyle: { color: '#ddd' }
      },
      splitArea: {
        areaStyle: { color: ['rgba(114, 172, 209, 0.1)', 'rgba(114, 172, 209, 0.2)'] }
      }
    },
    series: [{
      type: 'radar',
      data: [{
        value: values,
        name: '模型性能',
        areaStyle: {
          color: 'rgba(84, 112, 198, 0.4)'
        },
        lineStyle: {
          color: '#5470c6',
          width: 2
        },
        itemStyle: {
          color: '#5470c6'
        }
      }]
    }]
  }
})

// 面积图 - 训练损失趋势（填充区域）
const lossAreaOption = computed(() => {
  const history = detailModel.value?.training_metrics?.training_history
  if (!history || !history.epochs) return null
  
  const series: any[] = []
  
  if (history.train_box_loss) {
    series.push({
      name: 'Train Box Loss',
      type: 'line',
      data: history.train_box_loss,
      smooth: true,
      areaStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(84, 112, 198, 0.5)' },
            { offset: 1, color: 'rgba(84, 112, 198, 0.1)' }
          ]
        }
      },
      lineStyle: { color: '#5470c6' }
    })
  }
  if (history.train_cls_loss) {
    series.push({
      name: 'Train Cls Loss',
      type: 'line',
      data: history.train_cls_loss,
      smooth: true,
      areaStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(145, 204, 117, 0.5)' },
            { offset: 1, color: 'rgba(145, 204, 117, 0.1)' }
          ]
        }
      },
      lineStyle: { color: '#91cc75' }
    })
  }
  if (history.train_dfl_loss) {
    series.push({
      name: 'Train DFL Loss',
      type: 'line',
      data: history.train_dfl_loss,
      smooth: true,
      areaStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(250, 200, 88, 0.5)' },
            { offset: 1, color: 'rgba(250, 200, 88, 0.1)' }
          ]
        }
      },
      lineStyle: { color: '#fac858' }
    })
  }
  
  if (series.length === 0) return null
  
  return {
    title: { text: '训练损失面积图', left: 'center' },
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0 },
    xAxis: {
      type: 'category',
      data: history.epochs,
      name: 'Epoch',
      boundaryGap: false
    },
    yAxis: { type: 'value', name: 'Loss' },
    series
  }
})

// 仪表盘图 - 显示最终 mAP 分数
const gaugeChartOption = computed(() => {
  const metrics = detailModel.value?.training_metrics?.training_history?.final_metrics
  if (!metrics || metrics.mAP50 === null || metrics.mAP50 === undefined) return null
  
  const mAP50Value = (metrics.mAP50 * 100)
  
  return {
    title: { text: 'mAP50 评分仪表', left: 'center' },
    series: [{
      type: 'gauge',
      startAngle: 180,
      endAngle: 0,
      min: 0,
      max: 100,
      splitNumber: 10,
      itemStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 1, y2: 0,
          colorStops: [
            { offset: 0, color: '#ee6666' },
            { offset: 0.5, color: '#fac858' },
            { offset: 1, color: '#91cc75' }
          ]
        }
      },
      progress: {
        show: true,
        width: 20
      },
      pointer: {
        show: true,
        length: '60%',
        width: 8
      },
      axisLine: {
        lineStyle: {
          width: 20,
          color: [[1, '#e0e0e0']]
        }
      },
      axisTick: {
        distance: -30,
        splitNumber: 5,
        lineStyle: { width: 2, color: '#999' }
      },
      splitLine: {
        distance: -35,
        length: 10,
        lineStyle: { width: 3, color: '#999' }
      },
      axisLabel: {
        distance: -20,
        color: '#666',
        fontSize: 12
      },
      detail: {
        valueAnimation: true,
        formatter: '{value}%',
        color: '#333',
        fontSize: 24,
        offsetCenter: [0, '70%']
      },
      data: [{ value: parseFloat(mAP50Value.toFixed(1)), name: 'mAP50' }]
    }]
  }
})

// 热力图 - 显示各epoch各指标的表现（归一化）
const heatmapChartOption = computed(() => {
  const history = detailModel.value?.training_metrics?.training_history
  if (!history || !history.epochs) return null
  
  const metrics: string[] = []
  const data: number[][] = []
  
  // 收集所有指标数据
  const metricData: { [key: string]: number[] } = {}
  
  if (history.metrics_mAP50) {
    metricData['mAP50'] = history.metrics_mAP50
    metrics.push('mAP50')
  }
  if (history.metrics_mAP50_95) {
    metricData['mAP50-95'] = history.metrics_mAP50_95
    metrics.push('mAP50-95')
  }
  if (history.metrics_precision) {
    metricData['Precision'] = history.metrics_precision
    metrics.push('Precision')
  }
  if (history.metrics_recall) {
    metricData['Recall'] = history.metrics_recall
    metrics.push('Recall')
  }
  
  if (metrics.length === 0) return null
  
  // 构建热力图数据 [epochIdx, metricIdx, value]
  metrics.forEach((metric, metricIdx) => {
    const values = metricData[metric]
    values.forEach((value, epochIdx) => {
      data.push([epochIdx, metricIdx, value])
    })
  })
  
  return {
    title: { text: '训练指标热力图', left: 'center' },
    tooltip: {
      position: 'top',
      formatter: (params: any) => {
        return `Epoch ${history.epochs![params.data[0]]}<br/>${metrics[params.data[1]]}: ${(params.data[2] * 100).toFixed(1)}%`
      }
    },
    grid: {
      top: 60,
      bottom: 60,
      left: 80
    },
    xAxis: {
      type: 'category',
      data: history.epochs,
      name: 'Epoch',
      splitArea: { show: true }
    },
    yAxis: {
      type: 'category',
      data: metrics,
      splitArea: { show: true }
    },
    visualMap: {
      min: 0,
      max: 1,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 10,
      inRange: {
        color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
      }
    },
    series: [{
      name: '指标值',
      type: 'heatmap',
      data: data,
      label: { show: false },
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    }]
  }
})

// 重命名模型显示名：名称全平台唯一，判重统一交给后端（见 update_model 的 _display_name_owner）
const renamingModel = ref<string | null>(null)
const renameModelItem = async (model: any) => {
  const old = String(model.display_name || '')
  const input = await showPrompt(
    `给模型起个新名字（留空则显示模型 ID）\n名字不能与已有模型或算法库重名：`,
    old,
    '重命名模型',
  )
  if (input === null) return
  const name = input.trim()
  if (name === old) return
  // 重名（含与算法库同名）由后端权威校验并返回 400 提示：本地 models 列表已过滤掉
  // 库占位记录，在前端判会漏，且列表可能是旧的。
  renamingModel.value = model.model_id
  try {
    await updateModel(model.model_id, { display_name: name })
    // 与「归库」一致：成功就静默更新本地行（名称列会立刻变），失败才弹窗并重载
    model.display_name = name
  } catch (e: any) {
    alert('重命名失败: ' + (e.response?.data?.detail || e.message))
    await loadModels()
  } finally {
    renamingModel.value = null
  }
}

const deleteModelItem = async (modelId: string) => {
  if (!(await showConfirm(`确定要删除模型 ${modelId} 吗？此操作不可恢复！`))) return
  
  deletingModel.value = modelId
  try {
    await deleteModel(modelId)
    alert('删除成功!')
    loadModels()
  } catch (error: any) {
    alert('删除失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    deletingModel.value = null
  }
}

// 模型上传功能
// 记录上一次由文件名自动预填的名称，用户手改过就不再覆盖
let autoFilledName = ''

const handleModelFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files && target.files[0]) {
    const file = target.files[0]
    selectedModelFile.value = file
    const stem = file.name.replace(/\.(zip|pt)$/i, '')
    const cur = modelUploadName.value.trim()
    if (!cur || cur === autoFilledName) {
      modelUploadName.value = stem
      autoFilledName = stem
    }
  }
}

const uploadModelFile = async () => {
  if (!selectedModelFile.value) return
  
  uploadingModel.value = true
  const libId = modelUploadLib.value
  try {
    const res: any = await uploadModel(selectedModelFile.value, modelUploadName.value.trim(), libId)
    const libName = libId
      ? (libraries.value.find(l => l.model_library_id === libId)?.display_name || libId)
      : ''
    // 以服务端返回为准：旧后端会静默丢掉 library_id 字段，只看本地选项会误报"已归库"
    const bound = !libId || (res && (res.library_root_id || res.model_library_id) === libId)
    if (!libId) {
      showAlert('模型上传成功（独立模型，未归库）')
    } else if (bound) {
      showAlert(`模型上传成功，已归入算法库「${libName}」`)
    } else {
      showAlert(`模型上传成功，但归入「${libName}」未生效，请在下方「算法库」列手动补归（后端需重启以启用上传即归库）`)
    }
    // 清空表单
    selectedModelFile.value = null
    modelUploadName.value = ''
    autoFilledName = ''
    // 重新加载模型列表
    loadModels()
  } catch (error: any) {
    alert('上传失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    uploadingModel.value = false
  }
}

// 上传框里顺手新建算法库：建好即选中，避免出现"建了库却还要回去归库"
const onNewLibrary = (meta: any) => {
  const id = meta?.model_id
  if (!id) return
  if (!libraries.value.some(l => l.model_library_id === id)) {
    libraries.value.push({
      model_library_id: id,
      display_name: meta.display_name || meta.name || id,
    })
  }
  modelUploadLib.value = id
}

// 模型导出功能
const exportModelFile = async (modelId: string) => {
  exportingModel.value = modelId
  try {
    const blob = await exportModel(modelId)
    downloadFile(blob, `${modelId}.zip`)
  } catch (error: any) {
    alert('导出失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    exportingModel.value = null
  }
}

onMounted(() => {
  loadModels()
  loadPendingPublish()
})

// B-10：keep-alive 缓存后切回本页时静默刷新（首次挂载跳过，避免重复请求）
let _modelsFirstActivate = true
onActivated(() => {
  if (_modelsFirstActivate) { _modelsFirstActivate = false; return }
  loadModels(true)
  loadPendingPublish()
})

// ===== 2.7 相似模型排查与合并 =====
const simThreshold = ref(0.5)
const simPairs = ref<any[]>([])
const simScanning = ref(false)
const simScanned = ref(false)
const mergingSim = ref(false)
const selectedPairCount = computed(() => simPairs.value.filter(p => p.checked).length)

const runSimilarScan = async () => {
  simScanning.value = true
  try {
    const res = await findSimilarModels(simThreshold.value)
    simPairs.value = (res.pairs || []).map((p: any) => ({ ...p, checked: false }))
    simScanned.value = true
    if (simPairs.value.length) {
      alert(`发现 ${simPairs.value.length} 组相似模型对，请勾选（默认按推荐主模型吸收）后点击合并。`)
    }
  } catch (e: any) {
    alert('扫描失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    simScanning.value = false
  }
}

const confirmMergeSelected = async () => {
  const selected = simPairs.value.filter(p => p.checked)
  if (!selected.length) return
  const reason = await showPrompt('合并原因（写入日志用于审计，可选）:', '')
  if (reason === null) return
  const detail = selected.map((p: any) => {
    const mainId = p.suggested_main
    const sub = p.a.model_id === mainId ? p.b : p.a
    return `· 将「${sub.name || sub.model_id}」并入主模型「${mainId === p.a.model_id ? p.a.name : p.b.name}」`
  }).join('\n')
  if (!(await showConfirm(`确认执行以下合并吗？\n\n${detail}\n\n合并后：差集类别并入主模型标签字典、数据集重归属主模型、被合并模型保留为历史分支。`))) return
  mergingSim.value = true
  try {
    let totalDs = 0, totalCls = 0
    for (const p of selected) {
      const mainId = p.suggested_main
      const subId = p.a.model_id === mainId ? p.b.model_id : p.a.model_id
      const res = await mergeModels(mainId, [subId], reason || undefined)
      totalDs += res.datasets_rebound || 0
      totalCls += res.classes_added_count || 0
    }
    alert(`合并完成：${selected.length} 组，共并入 ${totalCls} 个类别、重归属 ${totalDs} 个数据集。\n可在「合并日志」中查看记录或回滚。`)
    simPairs.value = []
    simScanned.value = false
    loadModels()
  } catch (e: any) {
    alert('合并失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    mergingSim.value = false
  }
}

// 合并日志弹窗
const showMergeLogs = ref(false)
const mergeLogs = ref<any[]>([])
const loadingMergeLogs = ref(false)
const rollingBackMerge = ref(false)
const loadMergeLogs = async () => {
  showMergeLogs.value = true
  loadingMergeLogs.value = true
  try {
    const res = await getMergeLogs(20)
    mergeLogs.value = res.logs || []
  } catch (e: any) {
    alert('加载合并日志失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    loadingMergeLogs.value = false
  }
}
const closeMergeLogs = () => {
  showMergeLogs.value = false
}

const doRollbackMerge = async (logIndex: number) => {
  if (!(await showConfirm(
    logIndex === -1
      ? '确定回滚最近一次合并吗？\n\n将还原数据集归属、撤销被合并模型的标记（并入主模型的标签类别保留）。'
      : '确定回滚这条合并吗？\n\n将还原数据集归属、撤销被合并模型的标记（并入主模型的标签类别保留）。'
  ))) return
  rollingBackMerge.value = true
  try {
    const res = await rollbackMerge(logIndex)
    alert(`回滚完成：还原 ${res.restored_datasets?.length || 0} 个数据集、撤销 ${res.unmarked_models?.length || 0} 个模型标记。`)
    closeMergeLogs()
    loadModels()
  } catch (e: any) {
    alert('回滚失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    rollingBackMerge.value = false
  }
}
</script>

<style scoped>
.empty-state {
  padding: 2rem;
  text-align: center;
  color: #7f8c8d;
  background: #f8f9fa;
  border-radius: 4px;
  margin: 1rem 0;
}

/* 模型列表样式 */
.models-list {
  margin: 1rem 0;
  display: grid;
  gap: 1rem;
}

.model-item {
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.model-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.model-header h3 {
  margin: 0;
  color: #2c3e50;
}

.model-actions {
  display: flex;
  gap: 0.5rem;
}

.model-actions button {
  padding: 0.25rem 0.75rem;
  font-size: 0.875rem;
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}

.model-item p {
  margin: 0.25rem 0;
  color: #7f8c8d;
}

/* 模态框样式 */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  border-radius: 8px;
  width: 90%;
  max-width: 600px;
  max-height: 80vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border-bottom: 1px solid #ddd;
}

.modal-header h3 {
  margin: 0;
}

.close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: #7f8c8d;
  padding: 0;
  line-height: 1;
}

.close-btn:hover {
  color: #2c3e50;
}

.modal-body {
  padding: 1rem;
  overflow-y: auto;
}

.detail-section {
  margin-bottom: 1.5rem;
}

.detail-section h4 {
  margin: 0 0 0.5rem 0;
  color: #2c3e50;
  border-bottom: 1px solid #eee;
  padding-bottom: 0.25rem;
}

.detail-table {
  width: 100%;
  border-collapse: collapse;
}

.detail-table td {
  padding: 0.5rem;
  border-bottom: 1px solid #eee;
}

.detail-table td:first-child {
  color: #7f8c8d;
  width: 120px;
}

.class-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.class-tag {
  background: #e3f2fd;
  color: #1976d2;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.875rem;
}

.modal-large {
  max-width: 900px;
  max-height: 90vh;
}

/* 模型报告弹窗 */
.report-modal .modal-body {
  max-height: calc(90vh - 120px);
}

.report-download-bar {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.6rem 1rem;
  background: #f0f4ff;
  border-bottom: 1px solid #d8e0f0;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
}

.report-bar-title {
  font-weight: bold;
  color: #2c3e50;
  font-size: 0.95rem;
}

/* 诊断报告右下角的一键执行按钮 */
.report-exec-bar {
  display: flex;
  justify-content: flex-end;
  margin-top: 0.75rem;
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 2rem;
  color: #7f8c8d;
}

/* 性能总结 */
.summary-box {
  border: 1px solid #e0e3e8;
  border-radius: 6px;
  padding: 0.75rem 1rem;
  background: #fafbfc;
}

.summary-verdict {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  margin-bottom: 0.6rem;
  font-size: 0.95rem;
}

.summary-verdict.verdict-good {
  background: #d5f4e6;
  color: #1e8449;
}

.summary-verdict.verdict-warn {
  background: #fdf2d9;
  color: #b9770e;
}

.summary-verdict.verdict-bad {
  background: #fdeaea;
  color: #c0392b;
}

.verdict-badge {
  font-weight: bold;
  padding: 0.15rem 0.6rem;
  border-radius: 12px;
  background: rgba(0, 0, 0, 0.08);
  white-space: nowrap;
}

.verdict-desc {
  line-height: 1.5;
}

.summary-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.summary-list li {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  font-size: 0.9rem;
  line-height: 1.5;
  color: #2c3e50;
}

.summary-icon {
  flex-shrink: 0;
}

.summary-list li.summary-bad {
  color: #c0392b;
}

.summary-list li.summary-warn {
  color: #b9770e;
}

/* 模型报告 */
.report-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.report-header h4 {
  margin: 0;
  border-bottom: none;
  padding-bottom: 0;
}

.report-actions {
  display: flex;
  gap: 0.5rem;
}

.report-pre {
  background: #f6f8fa;
  border: 1px solid #e0e3e8;
  border-radius: 6px;
  padding: 1rem 1.25rem;
  font-family: ui-monospace, 'Courier New', monospace;
  font-size: 1rem;
  line-height: 1.9;
  color: #24292f;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 480px;
  overflow-y: auto;
}

.chart-container {
  width: 100%;
  height: 300px;
  margin: 1rem 0;
}

.chart-container.chart-gauge {
  height: 250px;
}

.chart-container.chart-large {
  height: 400px;
}

button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.25rem;
}

/* 模型上传部分 */
.upload-section {
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 4px;
  margin-bottom: 1.5rem;
  border: 1px dashed #ddd;
}

.upload-section h3 {
  margin-top: 0;
  margin-bottom: 1rem;
  color: #2c3e50;
  font-size: 1rem;
}

.upload-section .form-group {
  margin-bottom: 0.75rem;
}

.upload-section .form-group:last-of-type {
  margin-bottom: 1rem;
}

.form-hint {
  font-size: 0.875rem;
  color: #7f8c8d;
  margin-top: 0.25rem;
}

/* 提示行里的「＋新建算法库」：文字链样式，避免和主上传按钮抢视觉 */
.link-btn {
  background: none;
  border: none;
  padding: 0;
  color: #2c6ec5;
  font-size: inherit;
  font-family: inherit;
  cursor: pointer;
  text-decoration: underline;
}

.link-btn:hover {
  color: #2459a5;
}

/* 上传行（选择文件 + 上传按钮 齐平） */
.upload-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-top: 0.25rem;
}

.file-input {
  display: none;
}

.name-input {
  max-width: 420px;
}

.file-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 38px;
  padding: 0 1rem;
  background: #eef0f3;
  border: 1px solid #d0d3d8;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.95rem;
  color: #333;
  white-space: nowrap;
}

.file-btn:hover {
  background: #e2e5ea;
}

.upload-btn {
  background: #8e44ad;
  color: white;
  border: none;
  height: 38px;
  padding: 0 1.2rem;
  border-radius: 4px;
  cursor: pointer;
  font-weight: bold;
  font-size: 0.95rem;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  box-shadow: 0 2px 4px rgba(142, 68, 173, 0.3);
}

.upload-btn:disabled {
  background: #b9a0c9;
  cursor: not-allowed;
}

.selected-file {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
  padding: 0.75rem 1rem;
  background: #eaf4fb;
  border: 1px solid #3498db;
  border-radius: 6px;
  font-size: 1rem;
  color: #1a5276;
}

.selected-file.empty {
  background: #f8f9fa;
  border: 1px dashed #c0c4cc;
  color: #9aa0a6;
}

.selected-file-icon {
  font-size: 1.25rem;
}

.selected-file-name {
  font-weight: bold;
  color: #2c3e50;
  word-break: break-all;
}

.selected-file-size {
  color: #7f8c8d;
  white-space: nowrap;
}

/* 图表下载按钮 */
.chart-download-buttons {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

/* ===== 模型仓库 / 守门员样式 ===== */
.m-status {
  display: inline-block;
  margin-right: 0.4rem;
  padding: 0.1rem 0.5rem;
  border-radius: 10px;
  font-size: 0.72rem;
  font-weight: bold;
  white-space: nowrap;
  vertical-align: middle;
}

.m-production_ready {
  background: #eafaf1;
  border: 1px solid #82e0aa;
  color: #1e8449;
}

.m-rejected {
  background: #fdecea;
  border: 1px solid #f5b7b1;
  color: #c0392b;
}

.m-superseded {
  background: #fef9e7;
  border: 1px solid #f9e79f;
  color: #b7950b;
}

/* 补充状态配色：训练中（蓝）、评估中（蓝紫）、待确认发布（橙）、旧态 active（灰） */
.m-training {
  background: #eaf2f8;
  border: 1px solid #aed6f1;
  color: #2471a3;
}

.m-evaluating {
  background: #f4ecf7;
  border: 1px solid #d2b4de;
  color: #7d3c98;
}

.m-pending_production {
  background: #fef5e7;
  border: 1px solid #f5cba7;
  color: #ca6f1e;
}

.m-active,
.m-archived {
  background: #f2f3f4;
  border: 1px solid #d5d8dc;
  color: #616a6b;
}

.m-version {
  display: inline-block;
  margin-left: 0.4rem;
  font-size: 0.78rem;
  color: #7f8c8d;
  vertical-align: middle;
}

/* 模型唯一 code 标记（阶段0） */
.m-code {
  display: inline-block;
  margin-left: 0.4rem;
  padding: 1px 8px;
  font-size: 0.75rem;
  font-family: Consolas, Monaco, monospace;
  color: #2471a3;
  background: #eaf2f8;
  border: 1px solid #aed6f1;
  border-radius: 4px;
  vertical-align: middle;
}

.m-code-unset {
  color: #c0392b;
  background: #fdedec;
  border-color: #f5b7b1;
}

/* ===== 规范 §2.1 模型列表表格 ===== */
.model-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

.model-table th {
  text-align: left;
  padding: 0.5rem 0.6rem;
  border-bottom: 2px solid #ddd;
  color: #7f8c8d;
  font-size: 0.78rem;
  white-space: nowrap;
}

.model-table td {
  padding: 0.55rem 0.6rem;
  border-bottom: 1px solid #eee;
  vertical-align: top;
}

.model-table .th-ops {
  width: 150px;
}

.model-table .m-title {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.1rem;
  margin-bottom: 0.2rem;
}

.model-table .m-name {
  font-size: 0.95rem;
}

.model-table .m-sub {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.2rem;
}

.model-table .m-id {
  font-family: Consolas, Monaco, monospace;
  font-size: 0.72rem;
  color: #95a5a6;
  background: #f8f9f9;
  padding: 0 0.3rem;
  border-radius: 3px;
}

.model-table .gk-dot {
  cursor: help;
  font-size: 0.85rem;
  margin-left: 0.2rem;
}

.model-table .cell-map b {
  font-size: 0.95rem;
  color: #1e8449;
}

.model-table .map-lab,
.model-table .dim {
  color: #95a5a6;
}

.model-table .cell-data .d-data {
  word-break: break-all;
}

.model-table .cell-data .d-classes {
  display: block;
  margin-top: 0.15rem;
  color: #7f8c8d;
  font-size: 0.78rem;
}

.model-table .cell-ops {
  white-space: nowrap;
  position: relative;
}

.model-table .cell-ops .small {
  padding: 0.15rem 0.55rem;
  font-size: 0.78rem;
}

/* ⋯ 更多操作下拉菜单 */
.more-menu-wrap {
  position: relative;
  display: inline-block;
  margin-left: 0.35rem;
  vertical-align: middle;
}

/* 菜单打开时，在按钮正下方铺一条透明的“桥”，把按钮与菜单之间那 4px 视觉缝隙填上。
   没有它，指针一离开按钮就落在 wrapper 之外，mouseleave 立即触发、菜单来不及点。 */
.more-menu-wrap.menu-open::before {
  content: '';
  position: absolute;
  left: -8px;
  right: -8px;
  top: 100%;
  height: 12px;
}

.more-menu {
  position: absolute;
  right: 0;
  top: calc(100% + 4px);
  z-index: 30;
  min-width: 120px;
  background: #fff;
  border: 1px solid #d5d8dc;
  border-radius: 8px;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
  padding: 0.25rem 0;
  display: flex;
  flex-direction: column;
}

.more-menu .menu-item {
  background: none;
  border: none;
  width: 100%;
  text-align: left;
  padding: 0.45rem 0.8rem;
  cursor: pointer;
  font-size: 0.82rem;
  color: #2c3e50;
}

.more-menu .menu-item:hover {
  background: #f4f6f7;
}

.more-menu .menu-item:disabled {
  color: #bdc3c7;
  cursor: not-allowed;
}

.more-menu .menu-danger {
  color: #c0392b;
}

.more-menu .menu-danger:hover {
  background: #fdecea;
}

/* ===== P8 半自动发布：待确认发布卡片 ===== */
.publish-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}
.warning-state {
  padding: 1rem;
  color: #c0392b;
  background: #fdf0ec;
  border: 1px solid #f0a08a;
  border-radius: 8px;
}
.publish-badge {
  display: inline-block;
  padding: 0.1rem 0.5rem;
  background: #fef3e2;
  border: 1px solid #f5c97b;
  border-radius: 10px;
  color: #b9770e;
  font-size: 0.75rem;
  font-weight: bold;
}
/* 2.4 跨代迁移徽标（待发布列表） */
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
}
.pending-list {
  margin-top: 0.5rem;
}
.pending-item {
  padding: 0.8rem 1rem;
  margin-bottom: 0.6rem;
  background: #fff;
  border: 1px solid #e0e6ec;
  border-left: 4px solid #f5c97b;
  border-radius: 8px;
}
.pending-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  flex-wrap: wrap;
}
.pending-title {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
}
.pending-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.cooldown-hint {
  font-size: 0.78rem;
  color: #b9770e;
  background: #fef9e7;
  border: 1px solid #f9e79f;
  padding: 0.15rem 0.5rem;
  border-radius: 10px;
  white-space: nowrap;
}
.pending-meta {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  margin-top: 0.45rem;
  font-size: 0.82rem;
  color: #566573;
}
.publish-report-link {
  cursor: pointer;
  color: #2471a3;
  text-decoration: underline dotted;
}

.dict-input {
  width: 100%;
  padding: 4px 8px;
  border: 1px solid #d7dde4;
  border-radius: 4px;
  font-size: 0.85rem;
  box-sizing: border-box;
}

.dict-input:focus {
  outline: none;
  border-color: #3498db;
}

.labels-index {
  color: #7f8c8d;
  font-family: Consolas, Monaco, monospace;
  text-align: center;
}

.suggest-box {
  margin-top: 0.9rem;
  padding: 0.8rem;
  border: 1px dashed #3498db;
  border-radius: 6px;
  background: rgba(52, 152, 219, 0.04);
}

.suggest-toolbar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.suggest-msg {
  font-size: 0.8rem;
  color: #7f8c8d;
}

.suggest-imgs {
  font-size: 0.75rem;
  color: #95a5a6;
  font-family: Consolas, Monaco, monospace;
}

.small {
  padding: 2px 8px;
  font-size: 0.75rem;
}

.form-actions {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.8rem;
  justify-content: flex-end;
}

.danger-outline {
  background: #fff5f5;
  border: 1px solid #e74c3c;
  color: #e74c3c;
}

.danger-outline:hover {
  background: #fdecea;
}

/* 守门员报告折叠 */
.gk-report {
  margin-top: 0.5rem;
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

.gk-report-text {
  padding: 0.4rem 0.7rem 0.65rem;
  border-top: 1px dashed #dfe4ea;
  font-size: 0.8rem;
  line-height: 1.6;
  color: #4a5568;
  word-break: break-word;
}

.gk-regressed {
  padding: 0.4rem 0.7rem 0.65rem;
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

/* 详情弹窗守门员面板 */
.gk-panel {
  border: 1px solid #d7dde4;
  border-radius: 6px;
  padding: 0.8rem 1rem;
  background: #fafbfc;
}

.gk-panel-promoted,
.gk-panel-first_version {
  border-color: #82e0aa;
  background: #f0fdf5;
}

.gk-panel-rejected {
  border-color: #f5b7b1;
  background: #fef6f5;
}

.gk-verdict {
  font-weight: bold;
  margin-bottom: 0.4rem;
  color: #2c3e50;
  font-size: 0.9rem;
  line-height: 1.5;
}

.gk-panel .gk-report-text {
  padding: 0.3rem 0 0.5rem;
  border-top: none;
}

.gk-metrics {
  font-size: 0.85rem;
  color: #4a5568;
  margin: 0.3rem 0 0.5rem;
  line-height: 1.6;
}

.gk-metrics b {
  color: #2c3e50;
}

.gk-panel details summary {
  cursor: pointer;
  font-size: 0.8rem;
  color: #34495e;
  user-select: none;
  margin-top: 0.35rem;
}

.class-ap-table {
  margin-top: 0.5rem;
  font-size: 0.8rem;
}

.ap-regressed {
  color: #c0392b;
  font-weight: bold;
}

.ap-warn {
  margin-left: 0.35rem;
  font-size: 0.72rem;
  padding: 0.05rem 0.35rem;
  background: #fdecea;
  border: 1px solid #f5b7b1;
  border-radius: 8px;
}

/* ===== 2.7 相似模型排查与合并 ===== */
.sim-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 0.5rem;
}

.sim-header h2 {
  margin: 0;
}

.sim-actions {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
}

.sim-threshold {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.85rem;
  color: #4a5568;
}

.sim-list {
  margin-top: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.sim-pair {
  border: 1px solid #e0e3e8;
  border-radius: 6px;
  padding: 0.6rem 0.8rem;
  background: #fafbfc;
}

.sim-pair.checked {
  border-color: #3498db;
  background: #f0f7ff;
}

.sim-pair-main {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.sim-check { width: auto; }

.sim-score {
  font-weight: bold;
  color: #2c6ec5;
  background: #e8f0fb;
  padding: 0.15rem 0.6rem;
  border-radius: 10px;
  white-space: nowrap;
  font-size: 0.82rem;
}

.sim-models {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.sim-model {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.88rem;
  color: #34495e;
}

.sim-model code {
  font-size: 0.72rem;
  color: #7a8aa0;
  background: #f3f6fa;
  padding: 1px 6px;
  border-radius: 4px;
}

.sim-model.is-main {
  color: #1e8449;
  font-weight: bold;
}

.sim-main-tag {
  background: #1e8449;
  color: #fff;
  font-size: 0.68rem;
  padding: 0 5px;
  border-radius: 8px;
  font-weight: bold;
}

.sim-vs { color: #b0b8c4; }

.sim-common {
  margin-top: 0.35rem;
  font-size: 0.8rem;
  color: #7f8c8d;
}

.merge-logs {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.merge-log-item {
  border: 1px solid #e0e3e8;
  border-radius: 6px;
  padding: 0.6rem 0.8rem;
  background: #fafbfc;
}

.merge-log-head {
  display: flex;
  align-items: center;
  gap: 0.9rem;
  flex-wrap: wrap;
  font-size: 0.85rem;
  color: #34495e;
}

.merge-log-head code {
  font-size: 0.72rem;
  color: #7a8aa0;
  background: #f3f6fa;
  padding: 1px 6px;
  border-radius: 4px;
}

.merge-log-detail {
  margin-top: 0.4rem;
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  font-size: 0.8rem;
  color: #7f8c8d;
}
</style>

<template>
  <div class="models">
    <!-- 模型管理 -->
    <div class="card">
      <h2>模型管理</h2>
      
      <!-- 模型上传 -->
      <div class="upload-section">
        <h3>上传模型</h3>
        <div class="form-group">
          <label>选择 .zip 文件（格式与导出格式相同）</label>
          <div v-if="selectedModelFile" class="selected-file">
            <span class="selected-file-icon">📦</span>
            <span class="selected-file-name">{{ selectedModelFile.name }}</span>
            <span class="selected-file-size">({{ (selectedModelFile.size / 1024 / 1024).toFixed(2) }} MB)</span>
          </div>
          <div v-else class="selected-file empty">尚未选择文件</div>
          <div class="upload-row">
            <input type="file" accept=".zip" id="modelZip" @change="handleModelFileSelect" :disabled="uploadingModel" class="file-input" />
            <label for="modelZip" class="file-btn" :disabled="uploadingModel">选择文件</label>
            <button class="upload-btn" @click="uploadModelFile" :disabled="!selectedModelFile || uploadingModel">
              <span v-if="uploadingModel" class="loading-spinner"></span>
              {{ uploadingModel ? '上传中...' : '上传' }}
            </button>
          </div>
          <p class="form-hint">ZIP文件应包含 model.json 和 weights 目录</p>
        </div>
      </div>

      <div v-if="loadingModels" class="loading-state">
        <span class="loading-spinner large"></span>
        <span>加载模型列表...</span>
      </div>
      <div v-else-if="models.length > 0" class="models-list">
        <div v-for="model in models" :key="model.model_id" class="model-item">
          <div class="model-header">
            <h3>
              <span v-if="model.status" :class="'m-status m-' + model.status">{{ statusText(model) }}</span>
              {{ model.display_name || model.model_id }}
              <span v-if="model.model_code" class="m-code">{{ model.model_code }}</span>
              <span v-else class="m-code m-code-unset" title="尚未配置模型唯一 code，可在「编辑」中配置（模型流转/自动关联依赖它）">未配置code</span>
              <span v-if="model.version" class="m-version">版本 {{ model.version }}</span>
            </h3>
            <div class="model-actions">
              <button @click="viewModelDetails(model)" class="secondary">详情</button>
              <button @click="openLabelsDict(model)" class="secondary" title="维护该模型统一标签字典（index/english_code/中文名/中文描述）">标签字典</button>
              <button 
                @click="editModel(model)" 
                class="secondary"
                :disabled="editingModel === model.model_id"
              >
                <span v-if="editingModel === model.model_id" class="loading-spinner"></span>
                {{ editingModel === model.model_id ? '保存中...' : '编辑' }}
              </button>
              <button 
                @click="exportModelFile(model.model_id)" 
                class="secondary"
                :disabled="exportingModel === model.model_id"
              >
                <span v-if="exportingModel === model.model_id" class="loading-spinner"></span>
                {{ exportingModel === model.model_id ? '导出中...' : '导出' }}
              </button>
              <button 
                v-if="model.status !== 'production_ready'"
                @click="overrideModelItem(model)" 
                class="danger-outline"
                :disabled="overridingModel === model.model_id"
                title="人工强制覆盖：将被守门员拦截的模型手动设为生产版本（高级工程师操作，将记录原因）"
              >
                <span v-if="overridingModel === model.model_id" class="loading-spinner"></span>
                {{ overridingModel === model.model_id ? '强制设置中...' : '强制设为生产' }}
              </button>
              <button 
                @click="promoteModel(model.model_id)" 
                class="primary"
                :disabled="promotingModel === model.model_id"
                title="在相同验证集上对比 mAP50-95 与标注速度，确认更优后自动切换为 AI 预标注检测模型"
              >
                <span v-if="promotingModel === model.model_id" class="loading-spinner"></span>
                {{ promotingModel === model.model_id ? '评估对比中...' : '升级为预标注模型' }}
              </button>
              <button 
                @click="deleteModelItem(model.model_id)" 
                class="danger"
                :disabled="deletingModel === model.model_id"
              >
                <span v-if="deletingModel === model.model_id" class="loading-spinner"></span>
                {{ deletingModel === model.model_id ? '删除中...' : '删除' }}
              </button>
            </div>
          </div>
          <p v-if="model.business">业务场景: {{ bizLabel(model.business) }}</p>
          <p v-if="model.classes">类别数: {{ model.classes.length }}</p>
          <p v-if="model.base_model">基础模型: {{ model.base_model }}</p>
          <p v-if="model.lineage && (model.lineage.dataset_id || model.lineage.parent_model_id || model.lineage.base_model)">
            血缘:
            <span v-if="model.lineage.base_model">基座 {{ model.lineage.base_model }}</span>
            <template v-if="model.lineage.dataset_id"> · 数据集 {{ model.lineage.dataset_id }}</template>
            <template v-if="model.lineage.parent_model_id"> · 上一代 {{ model.lineage.parent_model_id }}</template>
          </p>
          <p v-if="model.created_at">创建时间: {{ model.created_at }}</p>
          <p v-if="model.description">描述: {{ model.description }}</p>
          <details v-if="model.gatekeeper && model.gatekeeper.report" class="gk-report">
            <summary>🛡️ 模型守门员评估报告</summary>
            <div class="gk-report-text">{{ model.gatekeeper.report }}</div>
            <div v-if="model.gatekeeper.regressed_classes?.length" class="gk-regressed">
              退化类别: <span v-for="c in model.gatekeeper.regressed_classes" :key="c" class="gk-regressed-item">{{ c }}</span>
            </div>
          </details>
        </div>
      </div>
      <div v-else class="empty-state">
        暂无模型，请先训练模型
      </div>
      <button @click="loadModels" class="secondary" :disabled="loadingModels">
        <span v-if="loadingModels" class="loading-spinner"></span>
        {{ loadingModels ? '加载中...' : '刷新模型列表' }}
      </button>
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
        未发现相似度 ≥ {{ simThreshold }} 的模型对（可在「标签字典」中完善 english_code / 中文名后再扫）
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
                 <button
                   v-if="detailModel?.status !== 'production_ready'"
                   @click="overrideModelItem(detailModel)"
                   class="danger-outline gk-override-btn"
                   :disabled="overridingModel === detailModel.model_id"
                 >
                   <span v-if="overridingModel === detailModel.model_id" class="loading-spinner"></span>
                   {{ overridingModel === detailModel.model_id ? '强制设置中...' : '强制设为生产（Override）' }}
                 </button>
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
<!-- 标签字典弹窗（阶段0：模型统一标签字典四字段） -->
<div v-if="showLabelsDict" class="modal-overlay" @click.self="closeLabelsDict">
  <div class="modal-content modal-large">
    <div class="modal-header">
      <h3>标签字典：{{ labelsDictModel?.model_code || labelsDictModel?.model_id }}</h3>
      <button @click="closeLabelsDict" class="close-btn">×</button>
    </div>
    <div class="modal-body">
      <p class="form-hint">
        四字段含义：<b>index</b>（YOLO 训练序号，保存后自动重排连续）· <b>english_code</b>（检测与 GD 预标注用，必填且唯一）·
        <b>中文名</b>（界面显示）· <b>中文描述</b>（大模型预标注的语义描述）。训练时只会保留本字典内的标签。
      </p>
      <div v-if="labelsDictLoading" class="loading-state"><span class="loading-spinner"></span>加载中...</div>
      <template v-else>
        <table class="detail-table">
          <thead>
            <tr><th style="width:56px">index</th><th>english_code *</th><th>中文名</th><th>中文描述</th><th style="width:44px"></th></tr>
          </thead>
          <tbody>
            <tr v-for="(lab, i) in labelsDictLabels" :key="i">
              <td class="labels-index">{{ i }}</td>
              <td><input v-model="lab.english_code" class="dict-input" placeholder="如 person" /></td>
              <td><input v-model="lab.chinese_name" class="dict-input" placeholder="如 行人" /></td>
              <td><input v-model="lab.chinese_desc" class="dict-input" placeholder="大模型预标注语义描述，可空" /></td>
              <td><button class="danger-outline small" @click="removeLabelRow(i)" title="删除该标签（已被图片标注的标签保存时会被拒绝）">✕</button></td>
            </tr>
            <tr v-if="!labelsDictLabels.length"><td colspan="5" class="empty-state">暂无标签，请添加</td></tr>
          </tbody>
        </table>
        <div class="form-actions">
          <button class="secondary" @click="addLabelRow">+ 添加标签</button>
          <button class="secondary" @click="prepareSuggest" title="用千问 VL 识别数据集中已知标签之外的新目标，给出候选标签（四字段），勾选后一键采纳追加">🤖 AI 识别新标签</button>
          <button class="primary" @click="saveLabelsDict" :disabled="labelsDictSaving">
            <span v-if="labelsDictSaving" class="loading-spinner"></span>
            {{ labelsDictSaving ? '保存中...' : '保存字典' }}
          </button>
        </div>
        <div v-if="suggestReady" class="suggest-box">
          <div class="suggest-toolbar">
            <select v-model="suggestDatasetId" class="dict-input" style="width:auto">
              <option v-for="ds in suggestDatasets" :key="ds.dataset_id" :value="ds.dataset_id">{{ ds.dataset_id }}</option>
            </select>
            <button class="secondary" @click="runSuggest" :disabled="suggestRunning">
              <span v-if="suggestRunning" class="loading-spinner"></span>
              {{ suggestRunning ? '识别中...' : '识别' }}
            </button>
            <span class="suggest-msg">{{ suggestMsg }}</span>
          </div>
          <table v-if="suggestCandidates.length" class="detail-table">
            <thead>
              <tr><th style="width:32px"></th><th>english_code</th><th>中文名</th><th>中文描述</th><th>命中图</th></tr>
            </thead>
            <tbody>
              <tr v-for="(c, i) in suggestCandidates" :key="i">
                <td><input type="checkbox" v-model="suggestSelected" :value="c" /></td>
                <td>{{ c.english_code }}</td>
                <td>{{ c.chinese_name }}</td>
                <td>{{ c.chinese_desc }}</td>
                <td class="suggest-imgs">{{ (c.images || []).join(', ') }}</td>
              </tr>
            </tbody>
          </table>
          <div v-if="suggestCandidates.length" class="form-actions">
            <button class="primary" @click="adoptSuggest" :disabled="!suggestSelected.length">采纳选中（{{ suggestSelected.length }}）</button>
          </div>
        </div>
      </template>
    </div>
  </div>
</div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { listModels, updateModel, deleteModel, getModel, uploadModel, exportModel, generateTrainingCharts, promoteToDetector, overrideModel, getLabelsDict, updateLabelsDict, getModelDatasets, suggestModelLabels, findSimilarModels, mergeModels, getMergeLogs, rollbackMerge, type ModelDetails } from '@/api/models'
import { downloadFile } from '@/utils/download'
import { showAlert, showConfirm, showPrompt } from '@/composables/useDialog'
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

// ===== 阶段0：标签字典弹窗 =====
const showLabelsDict = ref(false)
const labelsDictModel = ref<any>(null)
const labelsDictLoading = ref(false)
const labelsDictSaving = ref(false)
const labelsDictLabels = ref<any[]>([])

const openLabelsDict = async (model: any) => {
  showLabelsDict.value = true
  labelsDictModel.value = model
  labelsDictLoading.value = true
  labelsDictLabels.value = []
  try {
    const dic = await getLabelsDict(model.model_id)
    labelsDictLabels.value = (dic.labels || []).map((l: any) => ({ ...l }))
  } catch (e: any) {
    alert('加载标签字典失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    labelsDictLoading.value = false
  }
}
const closeLabelsDict = () => {
  showLabelsDict.value = false
  labelsDictModel.value = null
}
const addLabelRow = () => {
  labelsDictLabels.value.push({ index: labelsDictLabels.value.length, english_code: '', chinese_name: '', chinese_desc: '' })
}
const removeLabelRow = async (i: number) => {
  const ok = await showConfirm('确认删除该标签？\n已被图片标注使用的标签在保存时会被平台拒绝（追加禁删保护）。')
  if (ok) labelsDictLabels.value.splice(i, 1)
}

// ---- AI 识别新类别（千问 VL 看图 → 候选标签 → 一键采纳追加）----
const suggestReady = ref(false)
const suggestDatasets = ref<any[]>([])
const suggestDatasetId = ref('')
const suggestRunning = ref(false)
const suggestMsg = ref('')
const suggestCandidates = ref<any[]>([])
const suggestSelected = ref<any[]>([])

const prepareSuggest = async () => {
  suggestReady.value = true
  suggestMsg.value = ''
  suggestCandidates.value = []
  suggestSelected.value = []
  try {
    const res = await getModelDatasets(labelsDictModel.value.model_id)
    suggestDatasets.value = res.datasets || []
    suggestDatasetId.value = suggestDatasets.value[0]?.dataset_id || ''
    if (!suggestDatasets.value.length) suggestMsg.value = '该模型暂无绑定数据集，请先绑定后再 AI 识别'
  } catch (e: any) {
    suggestMsg.value = '加载模型数据集失败：' + (e.response?.data?.detail || e.message)
  }
}
const runSuggest = async () => {
  if (!suggestDatasetId.value) { suggestMsg.value = '请先选择数据集'; return }
  suggestRunning.value = true
  suggestMsg.value = ''
  try {
    const res = await suggestModelLabels(labelsDictModel.value.model_id, suggestDatasetId.value, 3)
    if (!res.ok) { suggestMsg.value = res.message || '识别失败'; suggestCandidates.value = []; return }
    suggestCandidates.value = res.candidates || []
    suggestMsg.value = res.message || ''
  } catch (e: any) {
    suggestMsg.value = 'AI 识别失败：' + (e.response?.data?.detail || e.message)
  } finally {
    suggestRunning.value = false
  }
}
const adoptSuggest = () => {
  const exist = new Set(labelsDictLabels.value.map((l: any) => (l.english_code || '').toLowerCase()))
  let added = 0
  for (const c of suggestSelected.value) {
    const code = (c.english_code || '').trim()
    if (!code || exist.has(code.toLowerCase())) continue
    labelsDictLabels.value.push({ index: labelsDictLabels.value.length, english_code: code, chinese_name: c.chinese_name || '', chinese_desc: c.chinese_desc || '' })
    exist.add(code.toLowerCase())
    added++
  }
  suggestSelected.value = []
  showAlert(added ? `已追加 ${added} 个标签到字典末尾（index 递增，已标注数据不受影响），检查后点「保存字典」即可生效` : '选中的标签已存在于字典或为空，未追加')
}
const saveLabelsDict = async () => {
  if (!labelsDictLabels.value.length) { alert('标签字典不能为空'); return }
  for (const lab of labelsDictLabels.value) {
    if (!lab.english_code || !lab.english_code.trim()) { alert('english_code 不能为空'); return }
  }
  labelsDictSaving.value = true
  try {
    await updateLabelsDict(labelsDictModel.value.model_id, labelsDictLabels.value)
    showAlert('标签字典已保存!')
    closeLabelsDict()
  } catch (e: any) {
    alert('保存失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    labelsDictSaving.value = false
  }
}

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
    case 'rejected': return '❌ 判定结果：淘汰（Reject）—— 未通过守门员校验，已归档为失败实验，旧模型继续服役（可强制覆盖）'
    default: return '🔍 判定结果：' + (gk?.result || '未知')
  }
}

const overridingModel = ref<string | null>(null)
// 人工强制覆盖（Override）：高级工程师手动将被守门员拦截的模型强制设为生产版本
const overrideModelItem = async (model: any) => {
  if (!(await showConfirm(
    `确定将模型 ${model.model_id} 强制设为生产版本吗？\n\n` +
    `该操作会绕过模型守门员的自动评估（仅建议在数据质量已确认、线上紧急修复等特殊情况下使用），\n` +
    `系统会记录操作记录与原因，便于审计追溯。`
  ))) return

  const reason = await showPrompt('请填写强制覆盖原因（用于审计，可选）:', '')
  if (reason === null) return  // 用户取消

  overridingModel.value = model.model_id
  try {
    const res = await overrideModel(model.model_id, {
      business: model.business || undefined,
      reason: reason || undefined
    })
    const superseded = (res.superseded || []).length
    alert(
      `✅ 已强制设为生产就绪！\n${res.meta?.version || ''}` +
      (superseded ? `\n同时退役了 ${superseded} 个原生产模型（已标记为"已退役"）。` : '')
    )
    loadModels()
  } catch (error: any) {
    alert('强制覆盖失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    overridingModel.value = null
  }
}

// 模型详情
const showModelDetails = ref(false)
const detailModel = ref<ModelDetails | null>(null)
const loadingDetails = ref(false)

// Loading 状态
const loadingModels = ref(false)
const editingModel = ref<string | null>(null)
const deletingModel = ref<string | null>(null)
const exportingModel = ref<string | null>(null)
const downloadingChart = ref<string | null>(null)
const promotingModel = ref<string | null>(null)

// 模型上传
const selectedModelFile = ref<File | null>(null)
const uploadingModel = ref(false)

const loadModels = async () => {
  loadingModels.value = true
  try {
    const data = await listModels()
    models.value = data.models
  } catch (error: any) {
    alert('加载模型失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    loadingModels.value = false
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

const editModel = async (model: any) => {
  const code = await showPrompt('模型唯一 code（小写字母+下划线，如 safety_helmet；模型流转/自动关联依赖它，留空则不修改）:', model.model_code || '')
  if (code === null) return

  const displayName = await showPrompt('模型中文名（界面显示用，可选）:', model.display_name || '')
  if (displayName === null) return

  const name = await showPrompt('请输入模型名称（可选）:', model.name || model.model_id)
  if (name === null) return
  
  const description = await showPrompt('请输入模型描述（可选）:', model.description || '')
  if (description === null) return
  
  const tagsInput = await showPrompt('请输入标签（逗号分隔，可选）:', model.tags ? model.tags.join(',') : '')
  const tags = tagsInput ? tagsInput.split(',').map((t: string) => t.trim()).filter((t: string) => t) : undefined
  
  editingModel.value = model.model_id
  try {
    await updateModel(model.model_id, {
      model_code: code.trim() || undefined,
      display_name: displayName.trim() || undefined,
      name: name || undefined,
      description: description || undefined,
      tags: tags
    })
    alert('更新成功!')
    loadModels()
  } catch (error: any) {
    alert('更新失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    editingModel.value = null
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

// 升级为预标注检测模型（热切换）
const promoteModel = async (modelId: string) => {
  if (!(await showConfirm(`确定将模型 ${modelId} 升级为 AI 预标注检测模型吗？\n\n将在相同验证集上对比 mAP50-95 与标注速度，确认更优后才会自动切换。`))) return

  promotingModel.value = modelId
  try {
    const res = await promoteToDetector(modelId)
    const nm = res.new_model || {}
    const om = res.old_model || {}
    const fmt = (v: any) => (v == null ? '—' : (v * 100).toFixed(1) + '%')
    const fmtSpeed = (v: any) => (v == null ? '—' : v + ' ms/张')
    const speedNote = nm.speed_ms != null && om.speed_ms != null
      ? (nm.speed_ms < om.speed_ms ? '（新模型标注速度更快）' : '（新模型标注速度更慢）')
      : ''
    if (res.switched) {
      alert(
        `✅ 已自动切换为预标注模型！${speedNote}\n\n` +
        `对比结果（同一验证集）：\n` +
        `  新模型  mAP50-95: ${fmt(nm.mAP50_95)}  P: ${fmt(nm.precision)}  R: ${fmt(nm.recall)}\n` +
        `  新模型  标注速度: ${fmtSpeed(nm.speed_ms)}\n` +
        `  旧模型  mAP50-95: ${fmt(om.mAP50_95)}  P: ${fmt(om.precision)}  R: ${fmt(om.recall)}\n` +
        `  旧模型  标注速度: ${fmtSpeed(om.speed_ms)}\n` +
        `AI 预标注将使用新模型。`
      )
    } else {
      const reason = (nm.mAP50_95 != null && om.mAP50_95 != null && nm.mAP50_95 <= om.mAP50_95 + 0.01 &&
        !(nm.speed_ms != null && om.speed_ms != null && nm.speed_ms < om.speed_ms))
        ? '新模型在 mAP50-95 与标注速度上均未显著优于当前检测模型'
        : '新模型 mAP50-95 未显著超过当前检测模型'
      alert(
        `未切换。${reason}。\n\n` +
        `  新模型  mAP50-95: ${fmt(nm.mAP50_95)}  P: ${fmt(nm.precision)}  R: ${fmt(nm.recall)}\n` +
        `  新模型  标注速度: ${fmtSpeed(nm.speed_ms)}\n` +
        `  旧模型  mAP50-95: ${fmt(om.mAP50_95)}  P: ${fmt(om.precision)}  R: ${fmt(om.recall)}\n` +
        `  旧模型  标注速度: ${fmtSpeed(om.speed_ms)}\n` +
        (om.error ? `\n旧模型评估提示：${om.error}` : '')
      )
    }
  } catch (error: any) {
    alert('升级失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    promotingModel.value = null
  }
}

// 模型上传功能
const handleModelFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files && target.files[0]) {
    selectedModelFile.value = target.files[0]
  }
}

const uploadModelFile = async () => {
  if (!selectedModelFile.value) return
  
  uploadingModel.value = true
  try {
    await uploadModel(selectedModelFile.value)
    alert('模型上传成功!')
    // 清空表单
    selectedModelFile.value = null
    // 重新加载模型列表
    loadModels()
  } catch (error: any) {
    alert('上传失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    uploadingModel.value = false
  }
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

.gk-override-btn {
  margin-top: 0.6rem;
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

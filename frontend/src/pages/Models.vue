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
            <h3>{{ model.model_id }}</h3>
            <div class="model-actions">
              <button @click="viewModelDetails(model)" class="secondary">详情</button>
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
          <p v-if="model.classes">类别数: {{ model.classes.length }}</p>
          <p v-if="model.base_model">基础模型: {{ model.base_model }}</p>
          <p v-if="model.created_at">创建时间: {{ model.created_at }}</p>
          <p v-if="model.description">描述: {{ model.description }}</p>
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
                <tr><td>基础模型</td><td>{{ detailModel?.base_model || '-' }}</td></tr>
                <tr><td>创建时间</td><td>{{ detailModel?.created_at || '-' }}</td></tr>
                <tr><td>类别数量</td><td>{{ detailModel?.classes?.length || 0 }}</td></tr>
                <tr><td>图片尺寸</td><td>{{ detailModel?.imgsz || '-' }}</td></tr>
                <tr><td>训练轮数</td><td>{{ detailModel?.epochs || '-' }}</td></tr>
                <tr v-if="detailModel?.file_size_mb"><td>模型大小</td><td>{{ detailModel.file_size_mb }} MB</td></tr>
              </table>
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
import { ref, onMounted, computed } from 'vue'
import { listModels, updateModel, deleteModel, getModel, uploadModel, exportModel, generateTrainingCharts, promoteToDetector, type ModelDetails } from '@/api/models'
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
  const name = await showPrompt('请输入模型名称（可选）:', model.name || model.model_id)
  if (name === null) return
  
  const description = await showPrompt('请输入模型描述（可选）:', model.description || '')
  if (description === null) return
  
  const tagsInput = await showPrompt('请输入标签（逗号分隔，可选）:', model.tags ? model.tags.join(',') : '')
  const tags = tagsInput ? tagsInput.split(',').map((t: string) => t.trim()).filter((t: string) => t) : undefined
  
  editingModel.value = model.model_id
  try {
    await updateModel(model.model_id, {
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

// 下载训练图表
const downloadTrainingChart = async (modelId: string, chartType: 'loss' | 'metrics') => {
  downloadingChart.value = chartType
  try {
    const blob = await generateTrainingCharts(modelId, chartType)
    downloadFile(blob, `${modelId}_${chartType}_chart.png`)
  } catch (error: any) {
    alert('下载图表失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    downloadingChart.value = null
  }
}

onMounted(() => {
  loadModels()
})
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
</style>

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
            :class="['ds-card', { selected: trainForm.dataset_id === ds.dataset_id, disabled: !isDatasetPrepared(ds) }]"
            @click="selectDataset(ds)"
          >
            <div class="ds-card-header">
              <span class="ds-name">{{ ds.dataset_id }}</span>
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
        <small>提示：仅显示在标注页执行过"导出YOLO(自动划分)"的数据集；点击卡片选择</small>
      </div>
      <div class="form-group">
        <label>业务场景</label>
        <select v-model="trainForm.business" :disabled="training">
          <option v-for="b in businessScenes" :key="b.value" :value="b.value">{{ b.label }}</option>
        </select>
        <small v-if="autoAssignedBiz">
          ✓ 已按数据类别自动分配为「{{ autoAssignedBiz }}」—— 新模型只会与同业务上一代生产模型对比；识别不准可手动调整
        </small>
        <small v-else>选择数据集后，系统将按类别自动分配业务场景（可分业务隔离存储与对比）</small>
      </div>
      <div class="form-group">
        <label>
          <input type="checkbox" v-model="useFineTune" :disabled="training" />
          基于已有模型微调
        </label>
      </div>
      <div v-if="!useFineTune" class="form-group">
        <label class="model-label">
          预训练模型
          <button type="button" class="add-model-btn" @click="ptFile?.click()" :disabled="uploadingPt">
            <span v-if="uploadingPt" class="loading-spinner"></span>
            {{ uploadingPt ? '导入中...' : '添加' }}
          </button>
          <input ref="ptFile" type="file" accept=".pt" class="pt-file-input" @change="handlePtFile" />
        </label>
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
        <div v-if="customModels.length" class="custom-model-block">
          <label class="custom-model-label">自定义导入的模型</label>
          <select v-model="customSelectedPath" :disabled="training" @change="onSelectCustomModel">
            <option value="">-- 使用上方 YOLO 预训练模型 --</option>
            <option v-for="cm in customModels" :key="cm.path" :value="cm.path">
              {{ cm.filename }} ({{ (cm.size / 1024 / 1024).toFixed(1) }} MB)
            </option>
          </select>
        </div>
        <small>当前选择: {{ trainForm.model_name }}</small>
      </div>
      <div v-else class="form-group">
        <label>选择已有模型进行微调</label>
        <div class="select-with-refresh">
          <select v-model="trainForm.base_model_id" :disabled="loadingModels || training">
            <option value="">-- 请选择模型 --</option>
            <option v-for="model in models" :key="model.model_id" :value="model.model_id">
              {{ model.model_id }} ({{ model.classes?.length || 0 }} 类)
            </option>
          </select>
          <button type="button" @click="loadModels" class="refresh-btn" :disabled="loadingModels">
            <span v-if="loadingModels" class="loading-spinner"></span>
            {{ loadingModels ? '加载中...' : '刷新' }}
          </button>
        </div>
        <small>将从所选模型开始继续训练</small>
      </div>
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
      <div class="form-group adv-block">
        <label class="adv-toggle">
          <input type="checkbox" v-model="showAdvanced" :disabled="training" />
          高级参数（学习率 / 优化器 / 权重衰减 / 早停）
        </label>
        <span v-if="suggesting" class="suggest-status">
          <span class="loading-spinner"></span> 分析环境并推荐参数...
        </span>
        <span v-else-if="suggestReason" class="suggest-status">✓ 已自动推荐（可手动修改）</span>
      </div>
      <div v-if="showAdvanced" class="grid adv-grid" style="grid-template-columns: repeat(4, 1fr);">
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
      <div v-if="suggestReason" class="suggest-reason">{{ suggestReason }}</div>
      <button @click="startTraining" :disabled="training || !trainForm.dataset_id">
        <span v-if="training" class="loading-spinner"></span>
        {{ training ? '训练中...' : '开始训练' }}
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
          <h3 class="job-name" @click="openJobFiles(job.job_id)" title="点击查看任务文件">{{ job.job_id }}</h3>
          <p>数据集: {{ job.dataset_id }}</p>
          <p>模型: {{ job.model_name }}</p>
          <p>轮数: {{ job.epochs }} | 图片尺寸: {{ job.imgsz }} | 批次: {{ job.batch }}</p>
          <p>节点: {{ job.gpu_index != null && job.gpu_index !== undefined ? 'GPU ' + job.gpu_index : '自动选择' }}</p>
          <p>状态: <span :class="'status-badge status-' + job.status">{{ job.status }}</span> <span v-if="job.early_stopped" class="early-stop-badge" title="模型已收敛，系统自动提前停止训练以节省资源">✓ 已完成（早停）</span></p>
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
            <!-- 失败原因（任务卡左边） -->
            <button 
              v-if="job.status === 'failed' || job.status === 'crashed'" 
              @click="showFailureDialog(job)" 
              type="button"
              class="danger-outline"
            >
              失败原因
            </button>
            <button 
              v-if="job.status === 'running'" 
              @click="stopJob(job.job_id)" 
              type="button"
              class="secondary"
              :disabled="stoppingJob === job.job_id"
            >
              <span v-if="stoppingJob === job.job_id" class="loading-spinner"></span>
              {{ stoppingJob === job.job_id ? '停止中...' : '停止' }}
            </button>
            <button 
              v-if="job.status !== 'running' && (job.status === 'stopped' || job.status === 'failed' || job.status === 'crashed' || job.can_resume)" 
              @click="resumeJob(job.job_id)" 
              type="button"
              class="secondary"
              :disabled="resumingJob === job.job_id"
            >
              <span v-if="resumingJob === job.job_id" class="loading-spinner"></span>
              {{ resumingJob === job.job_id ? '恢复中...' : '继续训练' }}
            </button>
            <button 
              v-if="job.status === 'failed' || job.status === 'crashed' || job.status === 'stopped'" 
              @click="retrainJob(job)" 
              type="button"
              class="primary"
              :disabled="retrainingJob === job.job_id"
            >
              <span v-if="retrainingJob === job.job_id" class="loading-spinner"></span>
              {{ retrainingJob === job.job_id ? '重新训练中...' : '重新训练' }}
            </button>
            <button 
              @click="deleteJobItem(job.job_id)" 
              type="button"
              class="danger"
              :disabled="deletingJob === job.job_id"
            >
              <span v-if="deletingJob === job.job_id" class="loading-spinner"></span>
              {{ deletingJob === job.job_id ? '删除中...' : '删除' }}
            </button>
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
import { useRouter } from 'vue-router'
import { createTrainJob, listTrainJobs, deleteTrainJob, resumeTrainJob, getTrainJobTree, suggestTrainParams, inferBusiness, listGpus, type TrainJobRequest } from '@/api/train'
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
  batch: -1,  // -1 表示根据显存自动计算最佳 batch size
  base_model_id: undefined,
  business: 'general',
  gpu_index: null  // null = 自动选择 GPU
})

// 业务/算法场景：模型按业务隔离存储与守门员对比
const businessScenes = [
  { value: 'general', label: '通用目标检测' },
  { value: 'pedestrian', label: '行人检测' },
  { value: 'vehicle', label: '车辆 / 车牌' },
  { value: 'defect', label: '工业缺陷检测' },
  { value: 'package', label: '包裹 / 物流' }
]
const bizLabel = (v: string) => businessScenes.find(b => b.value === v)?.label || v

// 任务卡守门员状态：通过 job.model_id 在模型列表中找到对应模型入库状态
const modelMetaOf = (modelId: string) => {
  return models.value.find((m: any) => m.model_id === modelId)
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

// 高级参数：默认展开（勾选）；选中数据集+模型后会自动推荐并回填，专家可手动微调
const showAdvanced = ref(true)
const suggesting = ref(false)
const suggestReason = ref('')
let suggestTimer: any = null

const useFineTune = ref(false)
const models = ref<any[]>([])
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

// 监听版本和大小变化，更新 model_name
watch([selectedVersion, selectedModelSize], () => {
  if (!useFineTune.value) {
    trainForm.value.model_name = generateModelName()
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
      useFineTune.value ? trainForm.value.base_model_id : undefined
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
    }
  } catch (e: any) {
    // 推荐失败不打断用户操作，保留当前值
    console.error('获取推荐参数失败:', e)
    suggestReason.value = ''
  } finally {
    suggesting.value = false
  }
}

// 数据集/模型/微调开关变化 → 防抖自动推荐
watch(
  [
    () => trainForm.value.dataset_id,
    () => trainForm.value.model_name,
    () => trainForm.value.base_model_id,
    useFineTune
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

const loadDatasets = async () => {
  loadingDatasets.value = true
  try {
    const result = await listDatasets()
    // 只显示已标注导出的数据集（在标注页执行过"导出YOLO(自动划分)"）
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

// 卡片式选择数据集
const selectDataset = (ds: any) => {
  if (!isDatasetPrepared(ds)) {
    alert('该数据集尚未准备，请先在"数据集"页面执行"准备"操作')
    return
  }
  trainForm.value.dataset_id = ds.dataset_id
}

// 加载已上传的自定义模型
const loadCustomModels = async () => {
  try {
    const result = await listCustomModels()
    customModels.value = result.models || []
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
const training = ref(false)
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
  
  if (useFineTune.value && !trainForm.value.base_model_id) {
    alert('请选择要微调的模型')
    return
  }
  
  training.value = true
  try {
    const params: TrainJobRequest = {
      ...trainForm.value,
      base_model_id: useFineTune.value ? trainForm.value.base_model_id : undefined
    }
    
    const result = await createTrainJob(params)
    currentJobId.value = result.job_id
    alert('训练任务已启动!')
    
    // 开始订阅日志
    logStore.clearLogs()
    subscribeToLogs(result.job_id)
    
    loadJobs()
    
    // 重置表单
    trainForm.value = {
      dataset_id: trainForm.value.dataset_id,
      version: 'v1',
      model_name: 'yolov8n.pt',
      epochs: 10,
      imgsz: 640,
      batch: 16,
      base_model_id: undefined,
      business: trainForm.value.business || 'general',
      gpu_index: null
    }
    useFineTune.value = false
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
      batch: job.batch ?? 16,
      base_model_id: job.base_model_id || undefined,
      business: job.business || 'general',
      lr0: job.lr0 ?? undefined,
      optimizer: job.optimizer || undefined,
      weight_decay: job.weight_decay ?? undefined,
      patience: job.patience ?? undefined,
      gpu_index: job.gpu_index ?? undefined
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

onMounted(() => {
  loadJobs()
  loadModels()
  loadDatasets()
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

.adv-grid .form-group {
  min-width: 0;
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
</style>

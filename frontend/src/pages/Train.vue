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
          <p>状态: <span :class="'status-badge status-' + job.status">{{ job.status }}</span></p>
          <div v-if="job.status === 'running'" class="job-progress">
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: jobProgress[job.job_id]?.percent + '%' }"></div>
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
          <p v-if="job.model_id">模型ID: {{ job.model_id }}</p>
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
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { createTrainJob, listTrainJobs, deleteTrainJob, resumeTrainJob, getTrainJobTree, type TrainJobRequest } from '@/api/train'
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
  base_model_id: undefined
})

const useFineTune = ref(false)
const models = ref<any[]>([])
const datasets = ref<any[]>([])
const loadingDatasets = ref(false)
const loadingModels = ref(false)
const loadingJobs = ref(false)
const stoppingJob = ref<string | null>(null)
const resumingJob = ref<string | null>(null)
const deletingJob = ref<string | null>(null)

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

const loadDatasets = async () => {
  loadingDatasets.value = true
  try {
    const result = await listDatasets()
    // 只显示已标注导出的数据集（在标注页执行过"导出YOLO(自动划分)"）
    datasets.value = (result.datasets || []).filter(d => d.annotated)
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
      base_model_id: undefined
    }
    useFineTune.value = false
  } catch (error: any) {
    alert('启动失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    training.value = false
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

// 训练失败操作建议：根据错误信息匹配常见原因并给出建议
const getFailureAdvice = (error: string = ''): { reason: string; advice: string } => {
  const e = error.toLowerCase()
  if (e.includes('out of memory') || e.includes('cuda out of memory') || e.includes('cublas')) {
    return {
      reason: '显存不足（CUDA out of memory）',
      advice: '建议减小"批次大小 batch"（如 16→8→4），或降低"图片尺寸 imgsz"，或关闭其他占用显存的程序。'
    }
  }
  if (e.includes('1455') || e.includes('page file') || e.includes('i/o error')) {
    return {
      reason: '系统内存/页面文件不足（错误 1455）',
      advice: '建议减小批次大小，关闭其他程序释放内存，或增大系统的虚拟内存（页面文件）。'
    }
  }
  if (e.includes('arial.ttf') || e.includes('font') || e.includes('download failure')) {
    return {
      reason: '绘制图表所需字体（Arial.ttf）缺失或下载失败',
      advice: '请将系统字体 arial.ttf 复制到用户目录下的 Ultralytics 字体目录，或检查网络后重试。'
    }
  }
  if (e.includes('pytorchstreamreader') || e.includes('zip archive') || e.includes('central directory')) {
    return {
      reason: '预训练权重文件（.pt）损坏',
      advice: '请删除损坏的 .pt 文件后重新下载一个完整的预训练权重，再开始训练。'
    }
  }
  if (e.includes('filenotfound') || e.includes('no such file')) {
    return {
      reason: '文件不存在（数据集、权重或输出路径有误）',
      advice: '请检查数据集是否已"准备"、预训练权重路径与输出目录是否存在。'
    }
  }
  if (e.includes('permission') || e.includes('access is denied')) {
    return {
      reason: '文件被占用或权限不足',
      advice: '请关闭占用相关文件的程序（尤其是训练输出目录），以管理员身份运行后重试。'
    }
  }
  if (e.includes('dataset') || e.includes('labels') || e.includes('images') && e.includes('empty')) {
    return {
      reason: '数据集为空或标注文件缺失',
      advice: '请检查数据集的图片与标注文件是否完整，必要时重新上传并"准备"数据集。'
    }
  }
  return {
    reason: error || '未知错误',
    advice: '请查看"训练日志"中的详细报错信息，排查数据集、权重与环境配置后重试。'
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

// 弹窗展示训练失败原因与操作建议
const showFailureDialog = (job: any) => {
  const { reason, advice } = getFailureAdvice(job.error || job.message || '')
  const failedAt = job.failed_at || job.crashed_at
  showAlert(
    `【训练失败】${reason}\n\n` +
    (failedAt ? `失败时间: ${new Date(failedAt).toLocaleString()}\n\n` : '') +
    `💡 操作建议:\n${advice}\n\n` +
    `（可点击"删除"移除该任务，或修复后重新训练）`,
    '训练失败'
  )
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
  height: 100%;
  background: linear-gradient(90deg, #8e44ad, #6c2d82);
  border-radius: 9px;
  transition: width 0.4s ease;
  min-width: 0;
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
</style>

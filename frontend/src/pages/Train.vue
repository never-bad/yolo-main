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
          <div v-if="!loadingDatasets && datasets.length === 0" class="empty-state">暂无数据集，请先上传</div>
        </div>
        <small>提示：点击卡片选择数据集，需为"已准备"状态</small>
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
          <button type="button" class="add-model-btn" @click="$refs.ptFile.click()" :disabled="uploadingPt">
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

    <div v-if="currentJobId" class="card">
      <h2>训练进度</h2>
      <div class="train-progress">
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: trainProgress.percent + '%' }"></div>
        </div>
        <div class="progress-meta">
          <span v-if="trainProgress.has">Epoch {{ trainProgress.current }} / {{ trainProgress.total }}</span>
          <span v-else>等待训练启动...</span>
          <span class="progress-percent" v-if="trainProgress.has">{{ trainProgress.percent }}%</span>
        </div>
        <div v-if="latestLossLine" class="loss-line">{{ latestLossLine }}</div>
      </div>
      <button class="secondary" @click="showTrainDetail = !showTrainDetail">
        {{ showTrainDetail ? '收起详细日志' : '展开详细日志' }}
      </button>
      <LogPanel 
        v-show="showTrainDetail"
        :has-job-id="!!currentJobId"
        :log-info="logInfo"
        :loading-history="loadingHistory"
        @load-history="loadHistoryLogs"
        @lines-change="onLinesChange"
        style="margin-top: 0.75rem;"
      />
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
          <h3>{{ job.job_id }}</h3>
          <p>数据集: {{ job.dataset_id }}</p>
          <p>模型: {{ job.model_name }}</p>
          <p>轮数: {{ job.epochs }} | 图片尺寸: {{ job.imgsz }} | 批次: {{ job.batch }}</p>
          <p>状态: <span :class="'status-badge status-' + job.status">{{ job.status }}</span></p>
          <p v-if="job.model_id">模型ID: {{ job.model_id }}</p>
          <p v-if="job.base_model_id">基础模型: {{ job.base_model_id }}</p>
          <p v-if="job.resume_count">续训次数: {{ job.resume_count }}</p>
          <p v-if="job.stopped_at">中断时间: {{ new Date(job.stopped_at).toLocaleString() }}</p>
          <p v-if="job.failed_at">失败时间: {{ new Date(job.failed_at).toLocaleString() }}</p>
          <p v-if="job.crashed_at">崩溃时间: {{ new Date(job.crashed_at).toLocaleString() }}</p>
          <p v-if="job.can_resume" style="color: #2ecc71; font-weight: bold;">✓ 可恢复训练</p>
          <div class="job-actions">
            <button v-if="job.status === 'running'" @click="viewLogs(job.job_id)">查看日志</button>
            <button 
              v-if="job.status === 'running'" 
              @click="stopJob(job.job_id)" 
              class="secondary"
              :disabled="stoppingJob === job.job_id"
            >
              <span v-if="stoppingJob === job.job_id" class="loading-spinner"></span>
              {{ stoppingJob === job.job_id ? '停止中...' : '停止' }}
            </button>
            <button 
              v-if="job.status === 'stopped' || job.status === 'failed' || job.status === 'crashed' || job.can_resume" 
              @click="resumeJob(job.job_id)" 
              class="secondary"
              :disabled="resumingJob === job.job_id"
            >
              <span v-if="resumingJob === job.job_id" class="loading-spinner"></span>
              {{ resumingJob === job.job_id ? '恢复中...' : '继续训练' }}
            </button>
            <button 
              @click="deleteJobItem(job.job_id)" 
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
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { createTrainJob, listTrainJobs, deleteTrainJob, resumeTrainJob, type TrainJobRequest } from '@/api/train'
import { listModels, uploadPretrainedPt, listCustomModels } from '@/api/models'
import { listDatasets } from '@/api/datasets'
import { subscribeLogsSSE, pollLogsTail, getLogLines } from '@/api/logs'
import { useLogStore } from '@/store/logs'
import LogPanel from '@/components/Logs/LogPanel.vue'
import { showConfirm } from '@/composables/useDialog'

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
    datasets.value = result.datasets || []
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

const training = ref(false)
const currentJobId = ref<string | null>(null)
const jobs = ref<any[]>([])
let eventSource: EventSource | null = null
let pollInterval: any = null

// 日志历史加载相关
const loadingHistory = ref(false)
const logInfo = ref('')
const selectedLogLines = ref(100)
const showTrainDetail = ref(false)

// 去除 ANSI 转义码
const stripAnsi = (text: string): string => text.replace(/\x1B\[[0-9;]*[mK]/g, '')

// 从日志中解析训练进度（epoch 当前/总数）
const trainProgress = computed(() => {
  const logs = logStore.logs
  let current = 0
  let total = 0
  for (let i = logs.length - 1; i >= 0; i--) {
    const clean = stripAnsi(logs[i]).trim()
    const m = clean.match(/^(\d+)\/(\d+)\s/)
    if (m) {
      current = parseInt(m[1])
      total = parseInt(m[2])
      break
    }
  }
  return {
    current,
    total,
    percent: total > 0 ? Math.round((current / total) * 100) : 0,
    has: total > 0
  }
})

// 从日志中提取最新的 epoch 损失指标行
const latestLossLine = computed(() => {
  const logs = logStore.logs
  for (let i = logs.length - 1; i >= 0; i--) {
    const clean = stripAnsi(logs[i]).trim()
    if (clean.match(/^(\d+)\/(\d+)\s/) && clean.includes('loss')) {
      return clean
    }
  }
  return ''
})

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

const stopJob = async (jobId: string) => {
  if (!(await showConfirm('确定要停止此训练任务吗？'))) return
  
  stoppingJob.value = jobId
  try {
    const { stopTrainJob } = await import('@/api/train')
    await stopTrainJob(jobId)
    alert('任务已停止')
    loadJobs()
  } catch (error: any) {
    alert('停止失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    stoppingJob.value = null
  }
}

const resumeJob = async (jobId: string) => {
  if (!(await showConfirm('确定要继续训练此任务吗？\n如果有 checkpoint 将从 checkpoint 恢复，否则从原始模型重新开始。'))) return
  
  resumingJob.value = jobId
  try {
    await resumeTrainJob(jobId)
    alert('训练已恢复!')
    
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
  if (!(await showConfirm(`确定要删除任务 ${jobId} 吗？此操作不可恢复！`))) return
  
  deletingJob.value = jobId
  try {
    await deleteTrainJob(jobId)
    alert('删除成功!')
    loadJobs()
  } catch (error: any) {
    alert('删除失败: ' + (error.response?.data?.detail || error.message))
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

const viewLogs = async (jobId: string) => {
  currentJobId.value = jobId
  logStore.clearLogs()
  
  // 先加载历史日志（默认最后100条）
  await loadHistoryLogs(selectedLogLines.value)
  
  // 然后开始订阅实时日志
  subscribeToLogs(jobId)
}

const loadHistoryLogs = async (n: number) => {
  if (!currentJobId.value) return
  
  loadingHistory.value = true
  logInfo.value = ''
  
  try {
    const result = await getLogLines(currentJobId.value, n)
    
    if (result.error) {
      logInfo.value = `错误: ${result.error}`
      return
    }
    
    // 清空当前日志并加载历史日志
    logStore.clearLogs()
    result.lines.forEach((line: string) => {
      logStore.addLog(line)
    })
    
    logInfo.value = `显示 ${result.returned}/${result.total} 条`
  } catch (error: any) {
    console.error('加载历史日志失败:', error)
    logInfo.value = '加载失败'
  } finally {
    loadingHistory.value = false
  }
}

const onLinesChange = (n: number) => {
  selectedLogLines.value = n
}

const loadJobs = async () => {
  loadingJobs.value = true
  try {
    const result = await listTrainJobs()
    jobs.value = result.jobs
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
})

onUnmounted(() => {
  if (eventSource) {
    eventSource.close()
  }
  if (pollInterval) {
    clearInterval(pollInterval)
  }
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
</style>

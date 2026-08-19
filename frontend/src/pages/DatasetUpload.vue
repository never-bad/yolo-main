<template>
  <div class="dataset-upload">
    <div class="card">
      <h2>上传数据集</h2>
      <div class="form-group">
        <label>选择压缩包文件（支持 .zip / .tar / .tar.gz / .tgz）</label>
        <div v-if="selectedFile" class="selected-file">
          <span class="selected-file-icon">📦</span>
          <span class="selected-file-name">{{ selectedFile.name }}</span>
          <span class="selected-file-size">({{ (selectedFile.size / 1024 / 1024).toFixed(2) }} MB)</span>
        </div>
        <div v-else class="selected-file empty">尚未选择文件</div>
        <div class="upload-row">
          <input type="file" accept=".zip,.tar,.tgz,.tar.gz" id="datasetZip" @change="handleFileSelect" :disabled="uploading" class="file-input" />
          <label for="datasetZip" class="file-btn" :disabled="uploading">选择文件</label>
          <button class="upload-btn" @click="uploadFile" :disabled="!selectedFile || uploading">
            <span v-if="uploading" class="loading-spinner"></span>
            {{ uploading ? '上传中...' : '上传' }}
          </button>
        </div>
      </div>
      <div class="form-group">
        <label>归属已有模型（可选，#3 多模型上传）</label>
        <select v-model="uploadModelId" class="model-code-input" @change="uploadModelCode = ''">
          <option value="">不归属（txt 包自动按类名匹配；其余进入「未归属数据集」，可在模型详情页绑定）</option>
          <option v-for="m in allModels" :key="m.model_id" :value="m.model_id">
            {{ m.display_name || m.name || m.model_code || m.model_id }}{{ m.dataset_count ? `（已挂 ${m.dataset_count} 个数据集）` : '' }}
          </option>
        </select>
        <small class="form-hint">与下方 model code 二选一：选这里直接挂入该模型；留空则交给自动匹配 / 手动绑定</small>
      </div>
      <div class="form-group">
        <label>归属模型 code（可选，2.2 动态建模型）</label>
        <input
          v-model="uploadModelCode"
          class="model-code-input"
          placeholder="如 traffic_scene；留空则不归属。填了后：已有该 code → 挂入对应模型；没有 → 自动创建空白模型"
          @input="uploadModelId = ''"
        />
        <small class="form-hint">将按归一化精确匹配（traffic-scene == trafficscene），不会模糊合并相似模型</small>
      </div>
      <div v-if="uploadResult" class="result">
        <p>✓ 上传成功: {{ uploadResult.dataset_id }}<span v-if="uploadResult.model_auto_created" class="auto-create-tag">（已自动创建模型 {{ uploadResult.model_code }}）</span></p>
        <p v-if="preparingNew" class="auto-prepare-tip">
          <span class="loading-spinner"></span>正在自动准备数据集（解压/重组/统计），请稍候...
        </p>
      </div>
    </div>

    <div class="card">
      <h2>数据集列表</h2>
      <button @click="loadDatasets" :disabled="loading" class="secondary">
        <span v-if="loading" class="loading-spinner"></span>
        {{ loading ? '加载中...' : '刷新' }}
      </button>
      <div v-if="loading" class="loading-state">
        <span class="loading-spinner large"></span>
        <span>加载数据集列表...</span>
      </div>
      <div v-else class="dataset-list">
        <div v-for="dataset in datasets" :key="dataset.dataset_id" class="dataset-item">
          <div class="dataset-header">
            <h3 class="dataset-name" @click="viewDataset(dataset)" title="点击查看数据集文件夹结构">{{ dataset.dataset_id }}</h3>
            <div class="dataset-actions">
              <button 
                @click="viewDataset(dataset)" 
                class="secondary"
                title="查看数据集图片"
              >查看</button>
              <button 
                v-if="dataset.status === 'prepared'"
                @click="goAnnotate(dataset)"
                class="secondary"
                title="打开该数据集的标注任务（无任务自动创建，直达标注页）"
              >去标注</button>
              <button 
                v-if="canSeal(dataset)"
                @click="sealDatasetItem(dataset)" 
                class="primary"
                :disabled="sealingDataset === dataset.dataset_id"
                title="封板后数据集只读，等待进入训练队列"
              >
                <span v-if="sealingDataset === dataset.dataset_id" class="loading-spinner"></span>
                {{ sealingDataset === dataset.dataset_id ? '封板中...' : '封板' }}
              </button>
              <button 
                v-else-if="dataset.stage === 'sealed'"
                class="secondary"
                disabled
                title="已封板（只读，等待训练）"
              >已封板</button>
              <button 
                @click="editDataset(dataset)" 
                class="secondary"
                :disabled="editingDataset === dataset.dataset_id || dataset.stage === 'sealed'"
              >
                <span v-if="editingDataset === dataset.dataset_id" class="loading-spinner"></span>
                {{ editingDataset === dataset.dataset_id ? '保存中...' : '编辑' }}
              </button>
              <button 
                v-if="dataset.status === 'prepared'"
                @click="exportAnnotated(dataset.dataset_id)" 
                class="secondary"
                :disabled="exportingDataset === dataset.dataset_id + '_annotated'"
              >
                <span v-if="exportingDataset === dataset.dataset_id + '_annotated'" class="loading-spinner"></span>
                {{ exportingDataset === dataset.dataset_id + '_annotated' ? '导出中...' : '导出（标注后）' }}
              </button>
              <button 
                v-if="dataset.status === 'prepared'"
                @click="exportOriginal(dataset.dataset_id)" 
                class="secondary"
                :disabled="exportingDataset === dataset.dataset_id + '_original'"
              >
                <span v-if="exportingDataset === dataset.dataset_id + '_original'" class="loading-spinner"></span>
                {{ exportingDataset === dataset.dataset_id + '_original' ? '导出中...' : '导出（标注前）' }}
              </button>
              <button 
                @click="deleteDatasetItem(dataset.dataset_id)" 
                class="danger"
                :disabled="deletingDataset === dataset.dataset_id"
              >
                <span v-if="deletingDataset === dataset.dataset_id" class="loading-spinner"></span>
                {{ deletingDataset === dataset.dataset_id ? '删除中...' : '删除' }}
              </button>
            </div>
          </div>
          <p>文件名: {{ dataset.filename }}</p>
          <p>大小: {{ (dataset.size / 1024 / 1024).toFixed(2) }} MB</p>
          <p>阶段:
            <span :class="'status-badge stage-' + (dataset.stage || 'annotating')">{{ stageLabel(dataset.stage) }}</span>
            <span v-if="dataset.training_status" :class="'status-badge train-' + dataset.training_status" :title="'training_status'">{{ trainingLabel(dataset.training_status) }}</span>
          </p>
          <p v-if="dataset.sealed_at">封板时间: {{ formatTime(dataset.sealed_at) }}</p>
          <p v-if="dataset.image_count">图片数: {{ dataset.image_count }}<span v-if="dataset.stage === 'annotating' && sealMinImages > 0" style="color:#b26a00;font-size:0.8rem">（待封板 {{ dataset.image_count }} / {{ sealMinImages }} 张{{ dataset.image_count < sealMinImages ? `，还差 ${sealMinImages - dataset.image_count} 张` : '' }}）</span></p>
          <p v-if="dataset.label_count">标签数: {{ dataset.label_count }}</p>
          <p v-if="dataset.classes">类别: {{ dataset.classes.join(', ') }}</p>
          <p v-if="dataset.annotation_task_id" class="auto-task-hint">
              ✓ 已自动创建标注任务（{{ dataset.annotation_task_id }}）→ 点「去标注」直接标注
            </p>
          <p v-if="dataset.description">描述: {{ dataset.description }}</p>
          <p v-if="dataset.tags && dataset.tags.length">标签: {{ dataset.tags.join(', ') }}</p>
        </div>
        <div v-if="datasets.length === 0" class="empty-state">
          暂无数据集，请上传数据集
        </div>
      </div>
    </div>

    <!-- 数据集查看弹窗（统计信息 + 图片预览） -->
    <div v-if="viewingDataset" class="viewer-mask" @click.self="closeViewer">
      <div class="viewer-dialog">
        <div class="viewer-header">
          <h3>{{ viewingDataset.dataset_id }}</h3>
          <button class="secondary viewer-close" @click="closeViewer">关闭</button>
        </div>
        <div v-if="viewerLoading" class="loading-state">
          <span class="loading-spinner large"></span>
          <span>加载数据集详情...</span>
        </div>
        <template v-else>
          <!-- 统计信息 -->
          <div class="viewer-stats">
            <div class="stat-item">
              <span class="stat-label">状态</span>
              <span class="stat-value">{{ viewerDetail?.status || viewingDataset.status }}</span>
            </div>
            <div class="stat-item" v-if="viewerDetail?.image_count || viewingDataset.image_count">
              <span class="stat-label">图片数</span>
              <span class="stat-value">{{ viewerDetail?.image_count || viewingDataset.image_count }}</span>
            </div>
            <div class="stat-item" v-if="viewerDetail?.label_count || viewingDataset.label_count">
              <span class="stat-label">标注数</span>
              <span class="stat-value">{{ viewerDetail?.label_count || viewingDataset.label_count }}</span>
            </div>
            <div class="stat-item" v-if="viewerDetail?.images">
              <span class="stat-label">预览图</span>
              <span class="stat-value">{{ viewerDetail.images.length }} 张</span>
            </div>
            <div class="stat-item" v-if="viewerDetail?.prepared_at">
              <span class="stat-label">准备时间</span>
              <span class="stat-value">{{ formatTime(viewerDetail.prepared_at) }}</span>
            </div>
          </div>
          <div class="viewer-classes" v-if="viewerDetail?.classes?.length || viewingDataset.classes?.length">
            <span class="stat-label">类别:</span>
            <span class="viewer-class-tags">
              <span v-for="(c, i) in (viewerDetail?.classes || viewingDataset.classes)" :key="i" class="class-tag">{{ c }}</span>
            </span>
          </div>
          <div class="viewer-description" v-if="viewerDetail?.description || viewingDataset.description">
            {{ viewerDetail?.description || viewingDataset.description }}
          </div>

          <!-- 文件夹结构 -->
          <div class="viewer-section-title">文件夹结构</div>
          <div v-if="treeLoading" class="loading-state">
            <span class="loading-spinner large"></span>
            <span>加载文件夹结构...</span>
          </div>
          <div v-else-if="treeData" class="viewer-tree">
            <FolderTree
              :nodes="treeData.children"
              :depth="0"
              :expanded="treeExpanded"
              @toggle-dir="toggleDir"
              @open-file="openFile"
            />
          </div>
          <div v-else class="empty-state">暂无文件夹信息</div>

          <!-- 图片预览 + 样本池入库（1.6） -->
          <div class="viewer-section-title">
            图片预览
            <span class="section-hint">勾选图片可加入样本池（困难样本/背景负样本），训练时自动抽样并入</span>
          </div>
          <div v-if="viewerImages.length === 0" class="empty-state">
            该数据集尚无图片（或未执行"准备"操作）
          </div>
          <template v-else>
            <div class="pool-toolbar">
              <label class="pool-model-label">困难样本归属模型:</label>
              <select v-model="poolModelId" class="pool-model-select" title="困难样本必须归属模型，训练该模型时才会抽样并入">
                <option value="">-- 选择模型（困难样本必填）--</option>
                <option v-for="m in poolModels" :key="m.model_id" :value="m.model_id">
                  {{ m.display_name || m.name || m.model_code || m.model_id }}
                </option>
              </select>
              <button class="secondary pool-btn" @click="addSelectionToHardPool" :disabled="poolImagesSelected.length === 0 || !poolModelId || !!poolBusy">
                <span v-if="poolBusy === 'hard'" class="loading-spinner"></span>
                {{ poolBusy === 'hard' ? '入库中...' : `加入困难样本库（${poolImagesSelected.length}张）` }}
              </button>
              <button class="secondary pool-btn" @click="addSelectionToBackground" :disabled="poolImagesSelected.length === 0 || !!poolBusy">
                <span v-if="poolBusy === 'bg'" class="loading-spinner"></span>
                {{ poolBusy === 'bg' ? '入库中...' : `加入空白样本库（${poolImagesSelected.length}张）` }}
              </button>
              <button class="secondary pool-btn" @click="poolSelected = {}" :disabled="poolImagesSelected.length === 0">取消选择</button>
            </div>
            <div class="viewer-grid">
              <div v-for="img in viewerImages" :key="img" class="viewer-thumb" :class="{ 'thumb-selected': !!poolSelected[img] }" @click="togglePoolSelect(img)">
                <img :src="'/static/' + img" loading="lazy" @error="handleImgError" />
                <span class="thumb-check" :class="{ checked: !!poolSelected[img] }">{{ poolSelected[img] ? '✓' : '' }}</span>
              </div>
            </div>
          </template>
        </template>
      </div>
    </div>

    <!-- 文件内容预览弹窗 -->
    <div v-if="filePreview" class="file-preview-mask" @click.self="closeFilePreview">
      <div class="file-preview-dialog">
        <div class="viewer-header">
          <h3>{{ filePreview.name }}</h3>
          <button class="secondary viewer-close" @click="closeFilePreview">关闭</button>
        </div>
        <div v-if="filePreviewIsImage" class="file-preview-img-wrap">
          <img :src="'/static/' + filePreview.path" @error="handleImgError" />
        </div>
        <pre v-else class="file-preview-text">{{ filePreviewContent || '加载中...' }}</pre>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { uploadDataset, prepareDataset as prepareDst, listDatasets, getDataset, getDatasetTree, updateDataset, deleteDataset, sealDataset as sealDst, exportAnnotatedDataset, exportOriginalDataset } from '@/api/datasets'
import { listModels } from '@/api/models'
import { addHardSamples, addBackgroundSamples } from '@/api/samplePool'
import { downloadFile } from '@/utils/download'
import { showConfirm, showPrompt } from '@/composables/useDialog'
import FolderTree from '@/components/FolderTree.vue'

const selectedFile = ref<File | null>(null)
const uploadModelCode = ref('')  // 2.2 动态建模型：上传时归属的模型 code
const uploadModelId = ref('')    // #3 多模型上传：勾选已有模型归属（与 model code 二选一）
const allModels = ref<any[]>([]) // 已有模型列表（归属下拉数据源）
const uploading = ref(false)
const uploadResult = ref<any>(null)
const sealMinImages = ref(0)  // #6 封板数量门槛（来自后端），列表展示待封板进度
const datasets = ref<any[]>([])
const loading = ref(false)
const preparingNew = ref(false)  // 正在准备新上传的数据集
const editingDataset = ref<string | null>(null)  // 正在编辑的数据集ID
const deletingDataset = ref<string | null>(null)  // 正在删除的数据集ID
const exportingDataset = ref<string | null>(null)  // 正在导出的数据集ID
const sealingDataset = ref<string | null>(null)  // 正在封板的数据集ID
const viewingDataset = ref<any | null>(null)  // 正在预览的数据集
const viewerImages = ref<string[]>([])  // 预览图片列表
const viewerLoading = ref(false)  // 预览加载状态
const viewerDetail = ref<any | null>(null)  // 数据集详情（统计信息）
const treeData = ref<any | null>(null)  // 文件夹结构树
const treeLoading = ref(false)  // 文件夹结构加载状态
const treeExpanded = reactive(new Set<string>())  // 已展开的目录节点
const filePreview = ref<any | null>(null)  // 正在预览的文件
const filePreviewIsImage = ref(false)  // 预览的是否为图片
const filePreviewContent = ref('')  // 文本文件内容

// ===== 样本池入库（1.6）=====
const poolModels = ref<any[]>([])      // 模型列表（困难样本归属选择）
const poolModelId = ref('')            // 选中的模型 id
const poolSelected = ref<Record<string, boolean>>({})  // 勾选的图片路径
const poolBusy = ref<'' | 'hard' | 'bg'>('')

/** 勾选图片路径集合（返回相对路径数组，入库时后端按文件名 stem 匹配） */
const poolImagesSelected = computed(() => Object.keys(poolSelected.value).filter(k => poolSelected.value[k]))

const togglePoolSelect = (imgPath: string) => {
  poolSelected.value[imgPath] = !poolSelected.value[imgPath]
}

// 加载模型列表（困难样本入库时选择归属模型）
const loadPoolModels = async () => {
  try {
    const result = await listModels()
    poolModels.value = (result.models || []).filter((m: any) => m.status !== 'deleted')
  } catch (e) {
    poolModels.value = []
  }
}

/** 加载已有模型列表，供上传时勾选归属（#3 多模型上传） */
const loadUploadModels = async () => {
  try {
    const result = await listModels()
    allModels.value = (result.models || []).filter((m: any) => m.status !== 'deleted')
  } catch (e) {
    allModels.value = []
  }
}

// 从图片相对路径提取 stem（文件名去扩展名），供后端匹配图片与标注
const imageStems = (paths: string[]) =>
  paths.map(p => {
    const name = p.split('/').pop() || p
    const extIdx = name.lastIndexOf('.')
    return extIdx > 0 ? name.slice(0, extIdx) : name
  })

const addSelectionToHardPool = async () => {
  if (!viewingDataset.value || poolImagesSelected.value.length === 0) return
  if (!poolModelId.value) {
    alert('请先选择困难样本归属模型（困难样本库按模型 1:1 隔离，训练该模型时才会抽样并入）')
    return
  }
  const ok = await showConfirm(`将 ${poolImagesSelected.value.length} 张图片（连同标注）加入模型 ${poolModelId.value} 的困难样本库？\n标注缺失的图片会跳过。`)
  if (!ok) return
  poolBusy.value = 'hard'
  try {
    const res = await addHardSamples({
      dataset_id: viewingDataset.value.dataset_id,
      model_id: poolModelId.value,
      image_names: imageStems(poolImagesSelected.value),
      version: 'v1'
    })
    alert(`入库完成：新增 ${res.added} 张，跳过 ${res.skipped} 张（未找到/已存在）。\n该模型困难样本库现有 ${res.pool_image_count} 张。`)
  } catch (e: any) {
    alert('加入困难样本库失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    poolBusy.value = ''
  }
}

const addSelectionToBackground = async () => {
  if (!viewingDataset.value || poolImagesSelected.value.length === 0) return
  const ok = await showConfirm(`将 ${poolImagesSelected.value.length} 张图片作为无目标背景（负样本，配空标注）加入全局空白样本库？\n训练任意模型时可按比例抽样并入。`)
  if (!ok) return
  poolBusy.value = 'bg'
  try {
    const res = await addBackgroundSamples({
      dataset_id: viewingDataset.value.dataset_id,
      image_names: imageStems(poolImagesSelected.value),
      version: 'v1'
    })
    alert(`入库完成：新增 ${res.added} 张，跳过 ${res.skipped} 张。`)
  } catch (e: any) {
    alert('加入空白样本库失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    poolBusy.value = ''
  }
}

// 打开数据集查看（统计信息 + 图片预览）
const viewDataset = async (dataset: any) => {
  if (dataset.status !== 'prepared') {
    alert('该数据集尚未"准备"，无法查看。请先执行"准备"操作。')
    return
  }
  viewingDataset.value = dataset
  viewerLoading.value = true
  viewerImages.value = []
  viewerDetail.value = null
  poolSelected.value = {}  // 打开新数据集时清空勾选
  try {
    const result = await getDataset(dataset.dataset_id)
    viewerDetail.value = result
    viewerImages.value = result.images || []
  } catch (e: any) {
    alert('加载数据集详情失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    viewerLoading.value = false
  }
  loadTree(dataset.dataset_id)
}

// 雪球闭环（1.8）：直达标注页 —— ?dataset_id=xxx，标注页优先复用已有任务/自动创建
const goAnnotate = (dataset: any) => {
  window.location.href = '/annotate?dataset_id=' + dataset.dataset_id
}

// 关闭预览
const closeViewer = () => {
  viewingDataset.value = null
  viewerImages.value = []
  viewerDetail.value = null
  treeData.value = null
  treeExpanded.clear()
  closeFilePreview()
}

// 加载数据集文件夹结构
const loadTree = async (datasetId: string) => {
  treeLoading.value = true
  treeData.value = null
  treeExpanded.clear()
  try {
    const result = await getDatasetTree(datasetId)
    treeData.value = result.tree
  } catch (e) {
    treeData.value = null
  } finally {
    treeLoading.value = false
  }
}

// 展开/折叠目录
const toggleDir = (node: any) => {
  if (treeExpanded.has(node.path)) treeExpanded.delete(node.path)
  else treeExpanded.add(node.path)
}

const FILE_IMG_EXTS = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']

// 点击文件：图片直接预览，文本读取内容
const openFile = async (node: any) => {
  filePreview.value = node
  filePreviewContent.value = ''
  filePreviewIsImage.value = FILE_IMG_EXTS.includes((node.ext || '').toLowerCase())
  if (filePreviewIsImage.value) return
  try {
    const res = await fetch('/static/' + node.path)
    if (!res.ok) throw new Error('HTTP ' + res.status)
    filePreviewContent.value = await res.text()
  } catch (e: any) {
    filePreviewContent.value = '无法读取该文件：' + (e.message || e)
  }
}

// 关闭文件预览
const closeFilePreview = () => {
  filePreview.value = null
  filePreviewContent.value = ''
}

// 图片加载失败时的兜底（隐藏占位）
const handleImgError = (e: any) => {
  const el = e.target as HTMLImageElement
  el.style.display = 'none'
}

// 时间格式化
const formatTime = (t: any) => {
  if (!t) return ''
  const d = new Date(t)
  if (isNaN(d.getTime())) return String(t).slice(0, 19).replace('T', ' ')
  return d.toLocaleString('zh-CN')
}

// ===== 数据集生命周期状态（MLOps 1.1）=====
const STAGE_LABELS: Record<string, string> = {
  collecting: '采集中',
  annotating: '标注中',
  sealed: '已封板',
  training: '训练中',
  completed: '已完成训练',
  failed: '训练失败',
}
const TRAIN_STATUS_LABELS: Record<string, string> = {
  incomplete: '未完成训练',
  completed: '已完成训练',
}
const stageLabel = (stage?: string) => STAGE_LABELS[stage || 'annotating'] || stage || '标注中'
const trainingLabel = (s?: string) => TRAIN_STATUS_LABELS[s || ''] || s || ''

// 是否可封板：已准备 且 未进入 sealed/training/completed/failed 阶段
const canSeal = (d: any) => {
  if (!d) return false
  const s = d.stage
  if (['sealed', 'training', 'completed', 'failed'].includes(s)) return false
  return d.status === 'prepared'
}

// 封板（标注不完整时后端拒绝，可走强制封板 = 时间窗口兜底）
const sealDatasetItem = async (d: any) => {
  // 数量门槛提示（#6）：标注中且数量未达门槛 → 明确提示差额，避免封板被后端拒绝
  const imgN = Number(d.image_count || 0)
  const minN = sealMinImages.value
  const sealNote = minN > 0 && imgN > 0 && imgN < minN
    ? `\n⚠ 当前 ${imgN} 张 < 封板门槛 ${minN} 张：直接封板会被拒绝，需走「强制封板」兜底`
    : ''
  const ratio = d.split_ratio || { train: 0.8, val: 0.2 }
  const input = window.prompt(
    '请输入数据集划分比例（训练,验证,测试，逗号分隔，如 0.8,0.2,0）：' + sealNote,
    `${ratio.train},${ratio.val},${ratio.test ?? 0}`
  )
  if (input === null) return
  const parts = input.split(/[,，、\s]+/).map(Number)
  if (parts.length < 2 || parts.some(n => Number.isNaN(n) || n < 0) || parts.reduce((a, b) => a + b, 0) <= 0) {
    alert('输入无效，请用逗号分隔三个非负数字，如：0.7,0.2,0.1')
    return
  }
  const t = parts[0], v = parts[1], te = parts[2] ?? 0
  const ok = await showConfirm(
    `确定封板数据集 ${d.dataset_id} 吗？\n` +
    `划分比例：训练:验证:测试 = ${t} : ${v} : ${te}\n` +
    `封板后数据集变为只读，等待进入训练队列。`
  )
  if (!ok) return
  sealingDataset.value = d.dataset_id
  try {
    await sealDst(d.dataset_id, false, { train: t, val: v, test: te })
    alert('封板成功')
  } catch (e: any) {
    const msg = e.response?.data?.detail || e.message
    const force = await showConfirm(`${msg}\n\n是否需要强制封板（时间窗口兜底，跳过标注完成条件）？`)
    if (force) {
      try {
        await sealDst(d.dataset_id, true, { train: t, val: v, test: te })
        alert('强制封板成功')
      } catch (e2: any) {
        alert('封板失败: ' + (e2.response?.data?.detail || e2.message))
      }
    }
  } finally {
    sealingDataset.value = null
    loadDatasets()
  }
}

const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files && target.files[0]) {
    selectedFile.value = target.files[0]
    uploadResult.value = null
  }
}

const uploadFile = async () => {
  if (!selectedFile.value) return
  
  uploading.value = true
  try {
    const result = await uploadDataset(
      selectedFile.value,
      uploadModelId.value || undefined,
      uploadModelCode.value.trim() || undefined
    )
    uploadResult.value = result
    loadDatasets()
    // 上传成功后自动准备（无需用户额外操作）
    preparingNew.value = true
    try {
      const prep = await prepareDst(result.dataset_id)
      if (prep?.model_auto_match?.model_code) {
        alert(`上传并准备成功！已按包内类别自动匹配模型「${prep.model_auto_match.display_name || prep.model_auto_match.model_code}」`)
      } else {
        alert('上传并准备成功!')
      }
      uploadResult.value = null
      loadDatasets()
    } catch (error: any) {
      alert('准备失败: ' + (error.response?.data?.detail || error.message))
    } finally {
      preparingNew.value = false
    }
  } catch (error: any) {
    alert('上传失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    uploading.value = false
  }
}

const loadDatasets = async () => {
  loading.value = true
  try {
    const result = await listDatasets()
    datasets.value = result.datasets
    sealMinImages.value = result.seal_min_images || 0
  } catch (error: any) {
    alert('加载失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    loading.value = false
  }
}

const editDataset = async (dataset: any) => {
  const description = await showPrompt('请输入数据集描述（可选）:', dataset.description || '')
  if (description === null) return
  
  const tagsInput = await showPrompt('请输入标签（逗号分隔，可选）:', dataset.tags ? dataset.tags.join(',') : '')
  const tags = tagsInput ? tagsInput.split(',').map(t => t.trim()).filter(t => t) : undefined
  
  editingDataset.value = dataset.dataset_id
  try {
    await updateDataset(dataset.dataset_id, {
      description: description || undefined,
      tags: tags
    })
    alert('更新成功!')
    loadDatasets()
  } catch (error: any) {
    alert('更新失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    editingDataset.value = null
  }
}

const deleteDatasetItem = async (datasetId: string) => {
  if (!(await showConfirm(`确定要删除数据集 ${datasetId} 吗？此操作不可恢复！`))) return
  
  deletingDataset.value = datasetId
  try {
    await deleteDataset(datasetId)
    alert('删除成功!')
    loadDatasets()
  } catch (error: any) {
    alert('删除失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    deletingDataset.value = null
  }
}

// 导出标注后的数据集
const exportAnnotated = async (datasetId: string) => {
  exportingDataset.value = datasetId + '_annotated'
  try {
    const blob = await exportAnnotatedDataset(datasetId)
    downloadFile(blob, `${datasetId}_annotated.zip`)
    alert('导出成功!')
  } catch (error: any) {
    alert('导出失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    exportingDataset.value = null
  }
}

// 导出标注前的数据集
const exportOriginal = async (datasetId: string) => {
  exportingDataset.value = datasetId + '_original'
  try {
    const blob = await exportOriginalDataset(datasetId)
    downloadFile(blob, `${datasetId}_original.zip`)
    alert('导出成功!')
  } catch (error: any) {
    alert('导出失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    exportingDataset.value = null
  }
}

onMounted(() => {
  loadDatasets()
  loadPoolModels()
  loadUploadModels()
})
</script>

<style scoped>
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

.result {
  margin-top: 1rem;
  padding: 1rem;
  background: #d5f4e6;
  border-radius: 4px;
}

.auto-prepare-tip {
  margin-top: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #1f6f4a;
}

.model-code-input {
  width: 100%;
  padding: 8px 10px;
  border: 1px solid #d5dee8;
  border-radius: 8px;
  font-size: 0.9rem;
}

.model-code-input:focus {
  border-color: #2c6ec5;
  outline: none;
}

.form-hint {
  color: #9aa7b6;
  font-size: 0.78rem;
}

.auto-create-tag {
  color: #2c6ec5;
  font-weight: 600;
}

.dataset-list {
  margin-top: 1rem;
  display: grid;
  gap: 1rem;
}

.dataset-item {
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.dataset-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.dataset-header h3 {
  margin: 0;
  color: #2c3e50;
}

.dataset-actions {
  display: flex;
  gap: 0.5rem;
}

.dataset-actions button {
  padding: 0.25rem 0.75rem;
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.dataset-item p {
  margin: 0.25rem 0;
  color: #7f8c8d;
}

.auto-task-hint {
  color: #2e7d32 !important;
  background: #e8f5e9;
  border: 1px solid #c8e6c9;
  border-radius: 4px;
  padding: 0.15rem 0.5rem;
  display: inline-block;
  font-size: 0.8rem;
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
  gap: 0.25rem;
}

/* 图片预览弹窗 */
.viewer-mask {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 1.5rem;
}

.viewer-dialog {
  background: white;
  border-radius: 8px;
  width: 90%;
  max-width: 1100px;
  max-height: 90vh;
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

.viewer-stats {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 0.75rem 1.25rem;
  padding: 1rem 1.25rem 0.5rem;
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}

.stat-label {
  font-size: 0.75rem;
  color: #95a5a6;
}

.stat-value {
  font-size: 0.95rem;
  color: #2c3e50;
  font-weight: 600;
}

.viewer-classes {
  padding: 0.5rem 1.25rem;
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.viewer-class-tags {
  display: flex;
  gap: 0.35rem;
  flex-wrap: wrap;
}

.class-tag {
  background: #eef4fb;
  color: #2c6fbb;
  border-radius: 12px;
  padding: 0.15rem 0.6rem;
  font-size: 0.8rem;
}

.viewer-description {
  padding: 0.25rem 1.25rem;
  color: #7f8c8d;
  font-size: 0.85rem;
}

.viewer-section-title {
  padding: 0.75rem 1.25rem 0.25rem;
  font-weight: 600;
  color: #2c3e50;
  border-top: 1px solid #f0f0f0;
  margin-top: 0.5rem;
}

.viewer-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 0.75rem;
  padding: 0.75rem 1.25rem 1.25rem;
  overflow-y: auto;
}

.viewer-thumb {
  aspect-ratio: 1;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid #e0e0e0;
  background: #f5f5f5;
}

.viewer-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

/* 样本池入库（1.6）*/
.section-hint {
  font-size: 0.75rem;
  color: #95a5a6;
  font-weight: normal;
  margin-left: 0.5rem;
}

.pool-toolbar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  padding: 0.5rem 1.25rem;
  border-bottom: 1px solid #f0f0f0;
}

.pool-model-label {
  font-size: 0.8rem;
  color: #555;
}

.pool-model-select {
  max-width: 240px;
  padding: 0.35rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
  font-size: 0.85rem;
}

.pool-btn {
  padding: 0.3rem 0.7rem;
  font-size: 0.8rem;
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
}

.viewer-thumb {
  position: relative;
  cursor: pointer;
  transition: border-color 0.15s, box-shadow 0.15s;
}

.viewer-thumb:hover {
  border-color: #8e44ad;
}

.viewer-thumb.thumb-selected {
  border: 2px solid #8e44ad;
  box-shadow: 0 0 0 2px rgba(142, 68, 173, 0.25);
}

.thumb-check {
  position: absolute;
  top: 6px;
  right: 6px;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.45);
  color: white;
  font-size: 14px;
  font-weight: bold;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid rgba(255, 255, 255, 0.7);
}

.thumb-check.checked {
  background: #8e44ad;
  border-color: #8e44ad;
}

/* 数据集名称可点击 */
.dataset-name {
  cursor: pointer;
}
.dataset-name:hover {
  color: #2c6fbb;
  text-decoration: underline;
}

/* 文件夹结构树 */
.viewer-tree {
  max-height: 400px;
  overflow-y: auto;
  padding: 0.5rem 1.25rem 0.75rem;
  border-bottom: 1px solid #f0f0f0;
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

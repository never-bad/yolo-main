<template>
  <div class="annotate">
    <div class="card">
      <h2>创建标注任务</h2>
      <div class="info-box">
        ⚠ 请确保数据集已完成准备（prepare）操作，否则无法创建标注任务
      </div>
      <div class="form-group">
        <label>选择数据集</label>
        <div class="select-with-refresh">
          <select v-model="newTask.datasetId" :disabled="loadingDatasets || creatingTask">
            <option value="">-- 请选择数据集 --</option>
            <option 
              v-for="ds in datasets" 
              :key="ds.dataset_id" 
              :value="ds.dataset_id"
              :disabled="!isDatasetPrepared(ds)"
            >
              {{ ds.dataset_id }} ({{ getDatasetStatus(ds) }}) 
              {{ ds.image_count ? `- ${ds.image_count}张图片` : '' }}
            </option>
          </select>
          <button type="button" @click="loadDatasets" class="refresh-btn" :disabled="loadingDatasets">
            <span v-if="loadingDatasets" class="loading-spinner"></span>
            {{ loadingDatasets ? '加载中...' : '刷新' }}
          </button>
        </div>
        <div class="ds-grid">
          <div
            v-for="ds in datasets"
            :key="ds.dataset_id"
            :class="['ds-card', { selected: newTask.datasetId === ds.dataset_id, disabled: !isDatasetPrepared(ds) }]"
            @click="selectDataset(ds)"
          >
            <div class="ds-card-header">
              <span class="ds-name">{{ ds.dataset_id }}</span>
              <span :class="'status-badge status-' + ds.status">{{ getDatasetStatus(ds) }}</span>
            </div>
            <div class="ds-card-info">
              <span v-if="ds.filename">文件: {{ ds.filename }}</span>
              <span v-if="ds.image_count">{{ ds.image_count }} 张图片</span>
              <span v-if="ds.classes && ds.classes.length" class="ds-classes" :title="ds.classes.join(', ')">
                {{ ds.classes.length }} 类: {{ ds.classes.slice(0, 3).join(', ') }}<template v-if="ds.classes.length > 3"> 等</template>
              </span>
            </div>
            <div v-if="!isDatasetPrepared(ds)" class="ds-card-warn">需先"准备"</div>
          </div>
          <div v-if="!loadingDatasets && datasets.length === 0" class="empty-state">暂无数据集，请先上传</div>
        </div>
        <small>提示：点击卡片选择数据集。状态为"已上传"需先在数据集页执行"准备"；已准备的可直接创建任务</small>
      </div>
      <div class="form-group">
        <label>类别（逗号分隔，可选）</label>
        <input v-model="classesInput" type="text" placeholder="留空则自动读取 data.yaml / classes.txt / coco.names" :disabled="creatingTask" />
        <small>提示：留空自动读取类别。支持 data.yaml（names 字段）、classes.txt、coco.names 等纯文本（按行读取）</small>
      </div>
      <button @click="createTask" :disabled="creatingTask || !newTask.datasetId">
        <span v-if="creatingTask" class="loading-spinner"></span>
        {{ creatingTask ? '创建中...' : '创建任务' }}
      </button>
    </div>

    <div v-if="currentTask" class="annotate-workspace">
      <div class="sidebar">
        <h3>图片列表 ({{ items.length }})</h3>
        <div v-if="importedCount > 0" class="import-info">
          已导入 {{ importedCount }} 个标注
        </div>
        <div class="image-filter">
          <div class="filter-dropdown" @click.stop>
            <button class="filter-trigger" @click="filterOpen = !filterOpen">
              <span>{{ currentFilterLabel }}</span>
              <span class="filter-caret">{{ filterOpen ? '▲' : '▼' }}</span>
            </button>
            <div v-if="filterOpen" class="filter-menu">
              <div
                class="filter-menu-item"
                :class="{ active: imageFilter === 'all' }"
                @click="setFilter('all')"
              >全部（{{ counts.total }}）</div>
              <div class="filter-menu-group">
                <div
                  class="filter-menu-item group-header"
                  :class="{ active: imageFilter === 'annotated' }"
                  @click="setFilter('annotated')"
                >已标注（{{ counts.annotated }}）</div>
                <div
                  class="filter-menu-item sub"
                  :class="{ active: imageFilter === 'manual' }"
                  @click="setFilter('manual')"
                >↳ 人工标注（{{ counts.manual }}）</div>
                <div
                  class="filter-menu-item sub"
                  :class="{ active: imageFilter === 'ai' }"
                  @click="setFilter('ai')"
                >↳ AI标注（{{ counts.ai }}）</div>
              </div>
              <div
                class="filter-menu-item"
                :class="{ active: imageFilter === 'unannotated' }"
                @click="setFilter('unannotated')"
              >未标注（{{ counts.unannotated }}）</div>
            </div>
          </div>
        </div>
        <div v-if="loadingItems" class="loading-state">
          <span class="loading-spinner"></span>
          <span>加载图片列表...</span>
        </div>
        <div v-else class="image-list">
          <div
            v-for="item in filteredItems"
            :key="item.image_id"
            :class="['image-item', { active: currentImage?.image_id === item.image_id, annotated: item.annotated }]"
            @click="selectImageByItem(item)"
          >
            <span class="image-item-name">{{ item.image_id }}</span>
            <span class="image-item-badges">
              <span v-if="item.annotated" class="img-badge ok" title="已有标注">✓</span>
              <span v-if="item.ai_annotated" class="img-badge ai" title="AI 预标注，建议人工审核">AI</span>
            </span>
          </div>
          <div v-if="filteredItems.length === 0" class="empty-filter">
            该类目下没有图片
          </div>
        </div>
        <div class="nav-buttons">
          <button @click="prevImage" :disabled="currentIndex === 0 || loadingAnnotation">上一张</button>
          <button @click="nextImage" :disabled="currentIndex === items.length - 1 || loadingAnnotation">下一张</button>
        </div>
      </div>

      <div class="canvas-area">
        <h3>
          {{ currentImage?.image_id }}
          <span v-if="loadingAnnotation" class="loading-indicator">
            <span class="loading-spinner"></span> 加载标注中...
          </span>
        </h3>
        <div class="ai-toolbar">
          <button class="ai-btn" @click="runAutoLabel" :disabled="!currentImage">
            <span v-if="aiLabeling" class="loading-spinner"></span>
            {{ aiLabeling ? 'AI标注中(点击取消)...' : 'AI 预标注' }}
          </button>
          <label class="ai-conf-label">检测模型:</label>
          <select v-model="selectedDetector" class="model-select" @change="applyDetectorModel">
            <option value="__grounding_dino__">grounding-dino-tiny (GroundingDINO)</option>
            <option v-for="m in samModels" :key="m.name" :value="m.name">{{ m.name }}</option>
          </select>
          <label class="ai-conf-label model-add-label">
            <input type="file" accept=".pt" class="model-add-input" @change="onSamModelFile" />
            添加
          </label>
          <label class="ai-conf-label">置信度:</label>
          <input type="number" v-model.number="aiConf" min="0" max="1" step="0.05" class="conf-input" />
          <label class="ai-conf-label">IoU:</label>
          <input type="number" v-model.number="aiIou" min="0" max="1" step="0.05" class="conf-input" />
          <button class="ai-btn small" @click="onThreshChange" title="保存置信度/IoU 阈值，AI 标注即用">保存阈值</button>
          <div v-if="batchId" class="batch-progress">
            <span>批量预标注: {{ batchProgress?.done || 0 }}/{{ batchProgress?.total || 0 }}</span>
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: batchPercent + '%' }"></div>
            </div>
            <button class="danger small" @click="stopBatch" :disabled="!batchRunning">停止</button>
          </div>
        </div>
        <div class="canvas-container" @mousedown="startDrawing" @mousemove="drawing" @mouseup="endDrawing" @wheel="onWheel" @mouseleave="onCanvasLeave">
          <button
            class="img-nav-btn prev"
            @click.stop="prevImage"
            @mousedown.stop
            :disabled="currentIndex === 0 || loadingAnnotation"
            title="上一张（← / A）"
          >‹</button>
          <canvas ref="canvasRef" @click="onCanvasClick" @dblclick="fitView"></canvas>
          <button
            class="img-nav-btn next"
            @click.stop="nextImage"
            @mousedown.stop
            :disabled="currentIndex === items.length - 1 || loadingAnnotation"
            title="下一张（→ / D）"
          >›</button>
          <div class="view-toolbar">
            <span class="view-hint">滚轮缩放 · 空格+拖拽平移 · 双击复位 · 拖框移动/角点缩放</span>
            <span>缩放 {{ Math.round(viewScale * 100) }}%</span>
            <button class="small" @click.stop="zoomBy(1.2)" title="放大">+</button>
            <button class="small" @click.stop="zoomBy(1 / 1.2)" title="缩小">−</button>
            <button class="small" @click.stop="fitView" title="适应窗口">适应窗口</button>
          </div>
        </div>
        <div class="controls">
          <label>当前类别:</label>
          <select v-model="currentClass">
            <option :value="-1">全部标签</option>
            <option v-for="(cls, idx) in classes" :key="idx" :value="idx">{{ cls }}</option>
          </select>
        </div>
      </div>

      <div class="annotations-panel">
        <h3>当前标注</h3>
        <div class="box-list">
          <div
            v-for="(box, idx) in currentBoxes"
            :key="idx"
            class="box-item"
            :class="{ 'box-item-selected': idx === selectedBoxIndex }"
            @click="selectBoxFromPanel(idx)"
          >
            <span class="box-color" :style="{ background: classColors[box.class_id] || '#999' }"></span>
            <span class="box-class-name" :title="classes[box.class_id] || ('类别' + box.class_id)">{{ classes[box.class_id] || ('类别' + box.class_id) }}</span>
            <select :value="box.class_id" class="box-class-select" :title="classes[box.class_id] || ''" @click.stop @change="onBoxClassChange($event, idx)">
              <option v-for="(cls, cIdx) in classes" :key="cIdx" :value="cIdx">{{ cls }}</option>
            </select>
            <button class="danger small" @click.stop="removeBox(idx)">删除</button>
          </div>
          <div v-if="!currentBoxes.length" class="box-empty">暂无标注框，可手动绘制或使用 AI 预标注</div>
        </div>
        <button class="danger" @click="clearCurrentBoxes" :disabled="!currentBoxes.length">
          清空本图标注
        </button>
        <button @click="runBatchLabel" :disabled="batchRunning || !items.length">
          <span v-if="batchRunning" class="loading-spinner"></span>
          {{ batchRunning ? '批量预标注中...' : '批量预标注' }}
        </button>
        <button
          @click="toggleInteractiveMode"
          :disabled="interactiveBusy || !items.length"
          :class="{ active: interactiveMode }"
          title="框选局部区域 + 文本提示（当前类别）检测，避免全图盲扫误标"
        >
          {{ interactiveMode ? '退出交互式标注' : '交互式标注' }}
        </button>
        <div v-if="interactiveMode && !interactiveBusy" class="interactive-hint">
          在图上拖拽框选要标注的区域（建议先选好"当前类别"），Esc 退出
        </div>
        <div v-if="interactiveBusy" class="interactive-hint">
          <span class="loading-spinner"></span> 正在解析所选区域...
        </div>
        <button class="danger" @click="clearAllAiLabels" :disabled="counts.ai === 0" title="误标过多时一键清除，重新标注">
          清除全部AI标注（{{ counts.ai }}）
        </button>
        <button class="danger" @click="cleanOverlaps" :disabled="counts.annotated === 0" title="清理历史遗留的高度重合标注框（跨类别/同类重复）">
          清理重叠框
        </button>
        <button @click="saveAnnotation" :disabled="savingAnnotation">
          <span v-if="savingAnnotation" class="loading-spinner"></span>
          {{ savingAnnotation ? '保存中...' : '保存' }}
        </button>
        <div class="split-config">
          <span class="split-label">划分比例</span>
          <label>训练 <input type="number" v-model.number="ratioTrain" min="0" max="1" step="0.05" class="ratio-input" /></label>
          <label>验证 <input type="number" v-model.number="ratioVal" min="0" max="1" step="0.05" class="ratio-input" /></label>
          <label>测试 <input type="number" v-model.number="ratioTest" min="0" max="1" step="0.05" class="ratio-input" /></label>
        </div>
        <button @click="exportTask" :disabled="exporting">
          <span v-if="exporting" class="loading-spinner"></span>
          {{ exporting ? '导出中...' : '导出YOLO(自动划分)' }}
        </button>
        <label class="import-label" title="导入 X-AnyLabeling 等外部工具导出的 YOLO 标注（zip 内含 labels/*.txt）">
          <input type="file" accept=".zip" class="import-input" @change="onImportYolo" />
          <span v-if="importing" class="loading-spinner"></span>
          {{ importing ? '导入中...' : '导入YOLO标注' }}
        </label>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import type { BBox } from '@/api/annotations'
import { createAnnotationTask, getTaskItems, getImageAnnotation, saveAnnotation as saveAnn, exportAnnotationsSplit, autoLabelImage, interactiveLabelImage, startBatchLabel, getBatchProgress, stopBatchLabel, getSamConfig, updateSamConfig, getSamModels, uploadSamModel, clearAiAnnotations, cleanTaskOverlaps, importYoloLabels } from '@/api/annotations'
import { listDatasets } from '@/api/datasets'
import { showConfirm } from '@/composables/useDialog'

const newTask = ref({ datasetId: '', version: 'v1' })
const datasets = ref<any[]>([])
const loadingDatasets = ref(false)
const creatingTask = ref(false)
const loadingItems = ref(false)
const savingAnnotation = ref(false)
// 自动保存（防抖）
const autoSaveTimer = ref<any>(null)
const dirty = ref(false)
// 导入 YOLO 标注
const importing = ref(false)
const exporting = ref(false)
// 标注任务持久化 key，用于中途离开页面后返回时恢复
const ANNOTATE_STORE_KEY = 'annotate_task_resume'
// 数据集划分比例（默认 7:2:1）
const ratioTrain = ref(0.7)
const ratioVal = ref(0.2)
const ratioTest = ref(0.1)

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
  newTask.value.datasetId = ds.dataset_id
}
const classesInput = ref('')  // 默认为空，从数据集自动读取
const classes = ref<string[]>([])
const currentTask = ref<string | null>(null)
const items = ref<any[]>([])
const currentIndex = ref(0)
const currentImage = ref<any>(null)
const currentClass = ref(-1)  // -1 表示"全部标签"，显示所有类别的框
const currentBoxes = ref<BBox[]>([])
const selectedBoxIndex = ref(-1)  // 当前选中的标注框索引（点击画布或左侧列表时更新）
const importedCount = ref(0)  // 导入的标注数量
const loadingAnnotation = ref(false)  // 加载标注状态

const canvasRef = ref<HTMLCanvasElement | null>(null)
const ctx = ref<CanvasRenderingContext2D | null>(null)
const img = ref<HTMLImageElement | null>(null)
const isDrawing = ref(false)
const startPos = ref({ x: 0, y: 0 })
const currentPos = ref({ x: 0, y: 0 })

// 画布视图变换（缩放/平移）
const viewScale = ref(1)
const viewOffset = ref({ x: 0, y: 0 })
const isPanning = ref(false)
const panStart = ref({ x: 0, y: 0 })
const spaceDown = ref(false)

// 交互/编辑状态
const editMode = ref<'idle' | 'draw' | 'move' | 'resize'>('idle')
const resizeHandle = ref(-1)  // 控制点索引 0-3 角 / 4-7 边
const moveStartPt = ref<{ x: number; y: number } | null>(null)
const origBox = ref<BBox | null>(null)

// 撤销/重做
const undoStack = ref<BBox[][]>([])
const redoStack = ref<BBox[][]>([])

// 类别颜色（渲染时自动分配）
const classColors = ref<string[]>([])

// SAM 大模型预标注状态
const aiLabeling = ref(false)
const aiAbortCtrl = ref<AbortController | null>(null)
const aiConf = ref(0.25)
const aiIou = ref(0.4)
// 交互式标注状态（框选局部区域 + 文本提示，避免全图盲扫误标）
const interactiveMode = ref(false)
const interactiveBusy = ref(false)
const interactiveRegion = ref<{ x1: number; y1: number; x2: number; y2: number } | null>(null)
const currentImageHasAi = ref(false)  // 当前图片是否由 AI 生成了标注框（用于保存时标记 AI 来源）
const samConfig = ref<any>({ detector_weights: 'yolov8s-world.pt', conf: 0.15 })
const samModels = ref<any[]>([])
// 检测模型下拉当前选中项：__grounding_dino__ 表示 GroundingDINO，其余为 YOLO-World .pt 文件名
const selectedDetector = ref('__grounding_dino__')
const batchId = ref<string | null>(null)
const batchProgress = ref<any>(null)
const batchTimer = ref<any>(null)

// 常见中文类别 → 英文提示词映射，用于提升 YOLO-World 对中文类别的识别率
const CLASS_EN_MAP: Record<string, string> = {
  '人': 'person', '行人': 'person', '人体': 'person',
  '车': 'car', '汽车': 'car', '轿车': 'car', '小汽车': 'car',
  '卡车': 'truck', '货车': 'truck', '公交车': 'bus', '大巴': 'bus',
  '自行车': 'bicycle', '摩托车': 'motorcycle', '电动车': 'motorcycle',
  '狗': 'dog', '猫': 'cat', '马': 'horse', '牛': 'cow', '羊': 'sheep',
  '猪': 'pig', '鸟': 'bird', '飞机': 'airplane', '轮船': 'boat', '船': 'boat',
  '火车': 'train', '瓶子': 'bottle', '杯子': 'cup', '碗': 'bowl',
  '椅子': 'chair', '桌子': 'table', '餐桌': 'dining table', '沙发': 'sofa',
  '床': 'bed', '电视': 'tv', '显示器': 'tv monitor', '电脑': 'laptop',
  '键盘': 'keyboard', '鼠标': 'mouse', '手机': 'cell phone', '手机架': 'cell phone',
  '遥控器': 'remote', '书': 'book', '钟': 'clock', '花瓶': 'vase',
  '剪刀': 'scissors', '香蕉': 'banana', '苹果': 'apple', '橙子': 'orange',
  '胡萝卜': 'carrot', '披萨': 'pizza', '蛋糕': 'cake', '面包': 'bread',
  '安全帽': 'helmet', '头盔': 'helmet', '帽子': 'hat',
  '背心': 'vest', '手套': 'glove', '口罩': 'mask',
  '消防栓': 'fire hydrant', '斑马线': 'zebra crossing', '交通灯': 'traffic light',
  '红绿灯': 'traffic light', '停靠牌': 'stop sign', '停车标志': 'stop sign',
  '限速牌': 'speed limit sign', '路标': 'traffic sign', '指示牌': 'signboard',
  '树木': 'tree', '树': 'tree', '花': 'flower', '草': 'grass',
  '建筑': 'building', '楼房': 'building', '房子': 'house', '烟囱': 'chimney',
  '路灯': 'street light', '电线杆': 'pole', '电线': 'wire',
  '摄像头': 'camera', '监控': 'cctv camera',
  '箱子': 'box', '纸箱': 'cardboard box', '包裹': 'package', '包裹箱': 'package',
  '垃圾桶': 'trash can', '垃圾箱': 'trash can', '饮料瓶': 'bottle',
  '雨伞': 'umbrella', '球': 'sports ball', '棒球棒': 'baseball bat',
  '网球拍': 'tennis racket', '滑雪板': 'skis', '滑板': 'skateboard',
  '冲浪板': 'surfboard', '风筝': 'kite', '棒球手套': 'baseball glove',
  '方向盘': 'steering wheel', '轮胎': 'tire', '车轮': 'wheel',
  '停车场': 'parking meter', '水杯': 'cup', '水壶': 'kettle',
  '书包': 'backpack', '行李箱': 'suitcase', '手提包': 'handbag',
  '领带': 'tie', '围巾': 'scarf', '外套': 'coat', '裤子': 'pants',
  '裙子': 'skirt', '鞋': 'shoe', '衬衫': 'shirt', '短裤': 'shorts',
  '袜子': 'socks', '蝴蝶结': 'tie', '领结': 'tie',
  '火车头': 'train', '地铁': 'train',
  '车牌': 'license plate', '车门': 'car door', '车窗': 'car window',
  '运动员': 'person', '工人': 'person',
  '油桶': 'bucket', '桶': 'bucket', '灭火器': 'fire extinguisher',
}

// 将类别列表转换为与 classes 一一对应的英文提示词（用于 YOLO-World 文本检测）
const buildPrompts = (classesList: string[]): string[] => {
  return classesList.map(c => CLASS_EN_MAP[c] || c)
}

const batchRunning = computed(() =>
  !!batchProgress.value && ['running', 'pending'].includes(batchProgress.value.status)
)
const batchPercent = computed(() => {
  if (!batchProgress.value || !batchProgress.value.total) return 0
  return Math.round((batchProgress.value.done / batchProgress.value.total) * 100)
})

// ===== 图片列表分类过滤（级联下拉菜单） =====
type FilterKey = 'all' | 'annotated' | 'ai' | 'manual' | 'unannotated'
const imageFilter = ref<FilterKey>('all')
const filterOpen = ref(false)
const counts = computed(() => {
  const total = items.value.length
  const annotated = items.value.filter(i => i.annotated).length
  const ai = items.value.filter(i => i.ai_annotated).length
  const manual = items.value.filter(i => i.annotated && !i.ai_annotated).length
  const unannotated = items.value.filter(i => !i.annotated).length
  return { total, annotated, ai, manual, unannotated }
})
const filterLabels: Record<string, string> = {
  all: '全部',
  annotated: '已标注',
  ai: 'AI标注',
  manual: '人工标注',
  unannotated: '未标注',
}
const currentFilterLabel = computed(() => {
  const label = filterLabels[imageFilter.value] || '全部'
  const count = counts.value[imageFilter.value === 'all' ? 'total' : imageFilter.value]
  return `${label}（${count}）`
})
const setFilter = (val: FilterKey) => {
  imageFilter.value = val
  filterOpen.value = false
}
const filteredItems = computed(() => {
  switch (imageFilter.value) {
    case 'annotated': return items.value.filter(i => i.annotated)
    case 'ai': return items.value.filter(i => i.ai_annotated)
    case 'manual': return items.value.filter(i => i.annotated && !i.ai_annotated)
    case 'unannotated': return items.value.filter(i => !i.annotated)
    default: return items.value
  }
})
// 点击过滤后的图片，定位到原始 items 索引
const selectImageByItem = (item: any) => {
  const idx = items.value.findIndex(i => i.image_id === item.image_id)
  if (idx >= 0) selectImage(idx)
}

const createTask = async () => {
  if (!newTask.value.datasetId) {
    alert('请选择数据集')
    return
  }
  
  // 检查数据集是否已准备
  const selectedDataset = datasets.value.find(ds => ds.dataset_id === newTask.value.datasetId)
  if (selectedDataset && selectedDataset.status === 'uploaded') {
    alert('数据集尚未准备（prepare）。请先在数据集上传页面执行"准备"操作后再创建标注任务。')
    return
  }
  
  // 始终让后端读取所选数据集的类别文件（覆盖输入框已有内容），确保使用真实类别名
  const inputClasses = undefined
  
  creatingTask.value = true
  try {
    const result = await createAnnotationTask(newTask.value.datasetId, newTask.value.version, inputClasses)
    currentTask.value = result.task_id
    
    // 使用返回的类别（从数据集类别文件读取，覆盖输入框内容）
    if (result.classes && result.classes.length > 0) {
      classes.value = result.classes
      classesInput.value = result.classes.join(', ')
    }
    
    // 显示导入信息
    if (result.imported_annotations > 0) {
      importedCount.value = result.imported_annotations
      alert(`任务创建成功！\n共 ${result.total_images} 张图片\n已导入 ${result.imported_annotations} 个已有标注`)
    }
    
    await loadTaskItems()
    saveAnnotateState()
  } catch (error: any) {
    alert('创建任务失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    creatingTask.value = false
  }
}

const loadTaskItems = async () => {
  if (!currentTask.value) return
  
  loadingItems.value = true
  try {
    const result = await getTaskItems(currentTask.value)
    items.value = result.items
    
    // 更新类别（如果后端返回了）
    if (result.classes && result.classes.length > 0) {
      classes.value = result.classes
      if (!classesInput.value) {
        classesInput.value = result.classes.join(', ')
      }
    }
    
    // 记录导入的标注数量
    if (result.imported_annotations) {
      importedCount.value = result.imported_annotations
    }
    
    if (items.value.length > 0) {
      selectImage(0)
    }
  } catch (error: any) {
    alert('加载失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    loadingItems.value = false
  }
}

// 持久化当前标注任务，便于中途离开页面后返回继续标注
const saveAnnotateState = () => {
  try {
    localStorage.setItem(ANNOTATE_STORE_KEY, JSON.stringify({
      taskId: currentTask.value,
      imageId: currentImage.value?.image_id || null,
      index: currentIndex.value,
      batchId: batchId.value,  // 恢复批量预标注轮询
    }))
  } catch (e) {
    console.error('保存标注状态失败:', e)
  }
}

// 清空持久化的标注任务（任务删除或完成后）
const clearAnnotateState = () => {
  try {
    localStorage.removeItem(ANNOTATE_STORE_KEY)
  } catch (e) {
    console.error('清空标注状态失败:', e)
  }
}

// 恢复上次未完成的标注任务
const restoreAnnotateState = async () => {
  let saved: any = null
  try {
    const raw = localStorage.getItem(ANNOTATE_STORE_KEY)
    if (raw) saved = JSON.parse(raw)
  } catch (e) {
    return
  }
  if (!saved?.taskId) return
  // 优先恢复批量预标注轮询：不因下面的任务/图片加载失败而中断（批量任务在后端后台仍在运行）
  if (saved.batchId) {
    batchId.value = saved.batchId
    batchProgress.value = { status: 'running', total: 1, done: 0, boxes_written: 0, current_image: '' }
    pollBatch()
  }
  currentTask.value = saved.taskId
  if (!currentTask.value) return
  try {
    const result = await getTaskItems(currentTask.value)
    items.value = result.items
    if (result.classes && result.classes.length > 0) {
      classes.value = result.classes
      classesInput.value = result.classes.join(', ')
    }
    if (result.imported_annotations) {
      importedCount.value = result.imported_annotations
    }
    if (items.value.length > 0) {
      const targetIdx = (saved.index && saved.index < items.value.length) ? saved.index : 0
      await selectImage(targetIdx)
    }
  } catch (e: any) {
    // 任务已不存在或加载失败：清空本地恢复记录，避免每次刷新都重复失败而清空图片列表
    currentTask.value = null
    items.value = []
    clearAnnotateState()
    alert('恢复标注任务失败，已清空记录，请重新创建任务: ' + (e.response?.data?.detail || e.message))
  }
}

const selectImage = async (index: number) => {
  currentIndex.value = index
  currentImage.value = items.value[index]
  currentBoxes.value = []
  selectedBoxIndex.value = -1
  currentImageHasAi.value = false
  // 切换图片：取消上一张的待保存状态（未保存的改动在离页时统一处理）
  dirty.value = false
  clearTimeout(autoSaveTimer.value)
  // 切换图片时重置编辑/撤销状态与视图
  editMode.value = 'idle'
  isDrawing.value = false
  isPanning.value = false
  moveStartPt.value = null
  origBox.value = null
  resizeHandle.value = -1
  undoStack.value = []
  redoStack.value = []

  await nextTick()
  
  // 如果图片已有标注，先加载标注
  if (currentImage.value.annotated && currentTask.value) {
    await loadExistingAnnotation()
  }
  
  loadImageToCanvas()
  saveAnnotateState()
}

// 加载已有标注
const loadExistingAnnotation = async () => {
  if (!currentTask.value || !currentImage.value) return
  
  loadingAnnotation.value = true
  try {
    const result = await getImageAnnotation(currentTask.value, currentImage.value.image_id)
    if (result && result.boxes && result.boxes.length > 0) {
      currentBoxes.value = result.boxes.map((box: any) => ({
        class_id: box.class_id,
        x1: box.x1,
        y1: box.y1,
        x2: box.x2,
        y2: box.y2
      }))
      console.log('Loaded existing annotations:', currentBoxes.value.length)
    }
  } catch (error: any) {
    console.error('加载标注失败:', error)
  } finally {
    loadingAnnotation.value = false
  }
}

const loadImageToCanvas = () => {
  if (!currentImage.value || !canvasRef.value) return
  
  const canvas = canvasRef.value
  ctx.value = canvas.getContext('2d')
  
  // 处理图片路径
  // 后端返回的路径应该是相对于 DATA_DIR 的（例如：datasets/ds_xxx/v1/images/xxx.jpg）
  // 静态文件服务挂载在 /static，目录是 backend/data
  // 所以需要：/static/datasets/...
  let imagePath = currentImage.value.image_path
  // 统一转换为正斜杠
  imagePath = imagePath.replace(/\\/g, '/')
  // 去掉可能的 backend/data/ 或 data/ 前缀（兼容旧格式）
  imagePath = imagePath.replace(/^(backend\/)?data\//, '')
  // 确保路径不以 / 开头（因为会加上 /static/ 前缀）
  imagePath = imagePath.replace(/^\//, '')
  // 静态文件路径不使用 API 基础路径，直接使用 /static/
  const imageUrl = '/static/' + imagePath
  
  console.log('Loading image:', {
    original: currentImage.value.image_path,
    processed: imageUrl
  })
  
  img.value = new Image()
  img.value.crossOrigin = 'anonymous'
  
  img.value.onload = () => {
    if (!img.value || !ctx.value) return
    canvas.width = img.value.width
    canvas.height = img.value.height
    console.log('Image loaded:', img.value.width, 'x', img.value.height)

    // 新图片加载后重置视图并居中
    fitView()

    // 绘制已有标注
    if (currentBoxes.value.length > 0) {
      redrawCanvas()
    }
  }
  
  img.value.onerror = (error) => {
    console.error('Image load error:', {
      url: imageUrl,
      originalPath: currentImage.value.image_path,
      error
    })
    // 显示更友好的错误提示
    const errorMsg = `图片加载失败\n路径: ${imageUrl}\n\n请检查：\n1. 后端服务是否运行\n2. 静态文件服务是否正常\n3. 图片文件是否存在`
    alert(errorMsg)
  }
  
  img.value.src = imageUrl
}

const screenToImg = (e: MouseEvent) => {
  const rect = canvasRef.value!.getBoundingClientRect()
  return {
    x: (e.clientX - rect.left - viewOffset.value.x) / viewScale.value,
    y: (e.clientY - rect.top - viewOffset.value.y) / viewScale.value
  }
}

const getContainerSize = () => {
  const el = canvasRef.value?.parentElement
  return el ? { w: el.clientWidth, h: el.clientHeight } : { w: 800, h: 600 }
}

// 视图适应窗口（图片居中显示）
const fitView = () => {
  if (!img.value || !canvasRef.value) return
  const { w, h } = getContainerSize()
  canvasRef.value.width = Math.max(w, 1)
  canvasRef.value.height = Math.max(h, 1)
  if (!img.value.width) return
  const s = Math.min(w / img.value.width, h / img.value.height, 1)
  viewScale.value = s
  viewOffset.value = {
    x: Math.max((w - img.value.width * s) / 2, 0),
    y: Math.max((h - img.value.height * s) / 2, 0)
  }
  redrawCanvas()
}

const zoomBy = (factor: number) => {
  if (!img.value || !canvasRef.value) return
  const { w, h } = getContainerSize()
  const cx = w / 2
  const cy = h / 2
  const ns = Math.min(Math.max(viewScale.value * factor, 0.05), 20)
  viewOffset.value = {
    x: cx - (cx - viewOffset.value.x) * (ns / viewScale.value),
    y: cy - (cy - viewOffset.value.y) * (ns / viewScale.value)
  }
  viewScale.value = ns
  redrawCanvas()
}

// 滚轮缩放（以鼠标为中心）
const onWheel = (e: WheelEvent) => {
  e.preventDefault()
  if (!img.value || !canvasRef.value) return
  const rect = canvasRef.value.getBoundingClientRect()
  const mx = e.clientX - rect.left
  const my = e.clientY - rect.top
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
  const ns = Math.min(Math.max(viewScale.value * factor, 0.05), 20)
  viewOffset.value = {
    x: mx - (mx - viewOffset.value.x) * (ns / viewScale.value),
    y: my - (my - viewOffset.value.y) * (ns / viewScale.value)
  }
  viewScale.value = ns
  redrawCanvas()
}

const onCanvasLeave = () => {
  if (isDrawing.value) {
    isDrawing.value = false
    editMode.value = 'idle'
  }
  isPanning.value = false
}

// 命中点 p 是否在某个可见框内（从后往前，后画的在上层）
const hitBox = (p: { x: number; y: number }) => {
  const cur = currentClass.value
  for (let i = currentBoxes.value.length - 1; i >= 0; i--) {
    const b = currentBoxes.value[i]
    if (String(cur) !== '-1' && b.class_id !== Number(cur)) continue
    if (p.x >= b.x1 && p.x <= b.x2 && p.y >= b.y1 && p.y <= b.y2) return i
  }
  return -1
}

// 命中选中框的控制点（0-3 角，4-7 边中点），未命中返回 -1
const hitHandle = (idx: number, p: { x: number; y: number }) => {
  const b = currentBoxes.value[idx]
  const hw = 10 / viewScale.value
  const pts = [
    { x: b.x1, y: b.y1 }, { x: (b.x1 + b.x2) / 2, y: b.y1 }, { x: b.x2, y: b.y1 },
    { x: b.x2, y: (b.y1 + b.y2) / 2 }, { x: b.x2, y: b.y2 }, { x: (b.x1 + b.x2) / 2, y: b.y2 },
    { x: b.x1, y: b.y2 }, { x: b.x1, y: (b.y1 + b.y2) / 2 }
  ]
  for (let i = 0; i < 8; i++) {
    if (Math.abs(p.x - pts[i].x) <= hw && Math.abs(p.y - pts[i].y) <= hw) return i
  }
  return -1
}

// 根据控制点缩放选中框
const applyResize = (p: { x: number; y: number }) => {
  const b = origBox.value
  const idx = selectedBoxIndex.value
  if (!b || idx < 0) return
  const h = resizeHandle.value
  let x1 = b.x1, y1 = b.y1, x2 = b.x2, y2 = b.y2
  if (h === 0 || h === 6 || h === 7) x1 = p.x
  if (h === 2 || h === 4 || h === 3) x2 = p.x
  if (h === 0 || h === 1 || h === 2) y1 = p.y
  if (h === 4 || h === 5 || h === 6) y2 = p.y
  if (x1 > x2) [x1, x2] = [x2, x1]
  if (y1 > y2) [y1, y2] = [y2, y1]
  const box = currentBoxes.value[idx]
  if (box) { box.x1 = x1; box.y1 = y1; box.x2 = x2; box.y2 = y2 }
}

// 画框/移动/缩放交互入口
const startDrawing = (e: MouseEvent) => {
  if (!canvasRef.value || !img.value) return
  // 按住空格 + 左键 = 平移
  if (spaceDown.value) {
    isPanning.value = true
    panStart.value = { x: e.clientX, y: e.clientY }
    return
  }
  const p = screenToImg(e)
  // 交互式标注模式：拖拽框选局部提示区域（不生成类别框）
  if (interactiveMode.value) {
    if (interactiveBusy.value) return
    selectedBoxIndex.value = -1
    editMode.value = 'draw'
    startPos.value = p
    currentPos.value = p
    isDrawing.value = true
    redrawCanvas()
    return
  }
  // 1) 命中选中框的控制点 → 缩放
  if (selectedBoxIndex.value >= 0) {
    const h = hitHandle(selectedBoxIndex.value, p)
    if (h >= 0) {
      editMode.value = 'resize'
      resizeHandle.value = h
      origBox.value = { ...currentBoxes.value[selectedBoxIndex.value] }
      pushUndo()
      return
    }
  }
  // 2) 命中某个框内部 → 选中并移动
  const hit = hitBox(p)
  if (hit >= 0) {
    selectedBoxIndex.value = hit
    editMode.value = 'move'
    moveStartPt.value = p
    origBox.value = { ...currentBoxes.value[hit] }
    pushUndo()
    redrawCanvas()
    return
  }
  // 3) 空白处 → 画新框
  selectedBoxIndex.value = -1
  editMode.value = 'draw'
  startPos.value = p
  currentPos.value = p
  isDrawing.value = true
  redrawCanvas()
}

const drawing = (e: MouseEvent) => {
  if (!canvasRef.value) return
  if (isPanning.value) {
    viewOffset.value = {
      x: viewOffset.value.x + (e.clientX - panStart.value.x),
      y: viewOffset.value.y + (e.clientY - panStart.value.y)
    }
    panStart.value = { x: e.clientX, y: e.clientY }
    redrawCanvas()
    return
  }
  if (editMode.value === 'draw' && isDrawing.value) {
    currentPos.value = screenToImg(e)
    redrawCanvas()
    return
  }
  if (editMode.value === 'move' && selectedBoxIndex.value >= 0 && moveStartPt.value && origBox.value) {
    const p = screenToImg(e)
    const dx = p.x - moveStartPt.value.x
    const dy = p.y - moveStartPt.value.y
    const box = currentBoxes.value[selectedBoxIndex.value]
    box.x1 = origBox.value.x1 + dx
    box.y1 = origBox.value.y1 + dy
    box.x2 = origBox.value.x2 + dx
    box.y2 = origBox.value.y2 + dy
    redrawCanvas()
    return
  }
  if (editMode.value === 'resize' && selectedBoxIndex.value >= 0) {
    applyResize(screenToImg(e))
    redrawCanvas()
  }
}

const endDrawing = (e: MouseEvent) => {
  if (isPanning.value) {
    isPanning.value = false
    return
  }
  if (editMode.value === 'draw' && isDrawing.value) {
    isDrawing.value = false
    currentPos.value = screenToImg(e)
    const x1 = Math.min(startPos.value.x, currentPos.value.x)
    const y1 = Math.min(startPos.value.y, currentPos.value.y)
    const x2 = Math.max(startPos.value.x, currentPos.value.x)
    const y2 = Math.max(startPos.value.y, currentPos.value.y)
    editMode.value = 'idle'
    // 交互式标注：框选完成 → 调接口在该区域内检测
    if (interactiveMode.value) {
      if (x2 - x1 > 5 && y2 - y1 > 5) {
        interactiveRegion.value = { x1, y1, x2, y2 }
        runInteractiveLabel()
      }
      redrawCanvas()
      return
    }
    if (x2 - x1 > 5 && y2 - y1 > 5) {
      let drawClass = Number(currentClass.value)
      if (Number.isNaN(drawClass) || drawClass < 0) drawClass = 0
      pushUndo()
      currentBoxes.value.push({ class_id: drawClass, x1, y1, x2, y2 })
      selectedBoxIndex.value = currentBoxes.value.length - 1
    }
    redrawCanvas()
    return
  }
  if (editMode.value === 'move' || editMode.value === 'resize') {
    editMode.value = 'idle'
    moveStartPt.value = null
    origBox.value = null
    resizeHandle.value = -1
  }
}

const onCanvasClick = (e: MouseEvent) => {
  if (!canvasRef.value || !img.value) return
  if (editMode.value !== 'idle') return
  if (interactiveMode.value) return
  const p = screenToImg(e)
  selectedBoxIndex.value = hitBox(p)
  redrawCanvas()
}

// 交互式标注：在用户框选的局部区域内按文本提示检测目标，追加到当前标注
const runInteractiveLabel = async () => {
  if (!interactiveRegion.value || !currentTask.value || !currentImage.value) return
  if (!classes.value.length) {
    alert('当前任务没有类别，无法进行交互式标注')
    return
  }
  interactiveBusy.value = true
  try {
    // 当前类别 >=0 时只标该类别（文本提示），返回的 class_id 是子集索引需映射回全局
    const single = currentClass.value >= 0 && currentClass.value < classes.value.length
    const cls = single ? [classes.value[currentClass.value]] : classes.value
    const res = await interactiveLabelImage(
      currentTask.value,
      currentImage.value.image_id,
      cls,
      aiConf.value,
      buildPrompts(cls),
      interactiveRegion.value
    )
    const base = single ? currentClass.value : 0
    const newBoxes: BBox[] = (res.boxes || []).map((b: any) => ({
      class_id: base + (b.class_id ?? 0),
      x1: b.x1,
      y1: b.y1,
      x2: b.x2,
      y2: b.y2
    }))
    if (newBoxes.length) {
      pushUndo()
      currentBoxes.value.push(...newBoxes)
      alert(`已在所选区域标注 ${newBoxes.length} 个目标`)
    } else {
      alert('该区域未检测到目标，可调整区域范围或类别后重试')
    }
    redrawCanvas()
  } catch (e: any) {
    alert('交互式标注失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    interactiveBusy.value = false
    interactiveRegion.value = null
  }
}

// 启用/退出交互式标注模式
const toggleInteractiveMode = () => {
  interactiveMode.value = !interactiveMode.value
  if (!interactiveMode.value) {
    interactiveRegion.value = null
  }
  redrawCanvas()
}

// 点击左侧列表：选中该标注框并在画布高亮
const selectBoxFromPanel = (index: number) => {
  selectedBoxIndex.value = index
  redrawCanvas()
}

const drawBox = (box: BBox, index = -1) => {
  if (!ctx.value) return
  const selected = index === selectedBoxIndex.value
  const color = classColors.value[box.class_id] || 'lime'
  ctx.value.strokeStyle = selected ? 'orange' : color
  ctx.value.lineWidth = selected ? 3 : 2
  ctx.value.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1)
  ctx.value.fillStyle = selected ? 'orange' : color
  ctx.value.font = '14px Arial'
  ctx.value.fillText(classes.value[box.class_id] || '', box.x1, Math.max(box.y1 - 5, 12))
  if (selected) {
    const s = 8 / viewScale.value
    const pts = [
      { x: box.x1, y: box.y1 }, { x: (box.x1 + box.x2) / 2, y: box.y1 }, { x: box.x2, y: box.y1 },
      { x: box.x2, y: (box.y1 + box.y2) / 2 }, { x: box.x2, y: box.y2 }, { x: (box.x1 + box.x2) / 2, y: box.y2 },
      { x: box.x1, y: box.y2 }, { x: box.x1, y: (box.y1 + box.y2) / 2 }
    ]
    ctx.value.fillStyle = '#fff'
    ctx.value.strokeStyle = 'orange'
    ctx.value.lineWidth = 1.5
    pts.forEach(pt => {
      ctx.value!.fillRect(pt.x - s / 2, pt.y - s / 2, s, s)
      ctx.value!.strokeRect(pt.x - s / 2, pt.y - s / 2, s, s)
    })
  }
}

const redrawCanvas = () => {
  if (!ctx.value || !canvasRef.value || !img.value) return
  const c = canvasRef.value
  ctx.value.clearRect(0, 0, c.width, c.height)
  ctx.value.save()
  ctx.value.translate(viewOffset.value.x, viewOffset.value.y)
  ctx.value.scale(viewScale.value, viewScale.value)
  ctx.value.drawImage(img.value, 0, 0)
  const cur = currentClass.value
  currentBoxes.value.forEach((box, idx) => {
    if (String(cur) === '-1' || box.class_id === Number(cur)) {
      drawBox(box, idx)
    }
  })
  // 正在画的新框
  if (isDrawing.value && editMode.value === 'draw') {
    ctx.value.strokeStyle = 'red'
    ctx.value.lineWidth = 2
    ctx.value.strokeRect(
      startPos.value.x, startPos.value.y,
      currentPos.value.x - startPos.value.x, currentPos.value.y - startPos.value.y
    )
  }
  ctx.value.restore()
}

const removeBox = (index: number) => {
  pushUndo()
  currentBoxes.value.splice(index, 1)
  if (selectedBoxIndex.value === index) selectedBoxIndex.value = -1
  redrawCanvas()
}

// 撤销/重做
const pushUndo = () => {
  undoStack.value.push(JSON.parse(JSON.stringify(currentBoxes.value)))
  if (undoStack.value.length > 50) undoStack.value.shift()
  redoStack.value = []
  markDirty()
}

const undo = () => {
  if (!undoStack.value.length) return
  redoStack.value.push(JSON.parse(JSON.stringify(currentBoxes.value)))
  currentBoxes.value = undoStack.value.pop()!
  selectedBoxIndex.value = -1
  redrawCanvas()
}

const redo = () => {
  if (!redoStack.value.length) return
  undoStack.value.push(JSON.parse(JSON.stringify(currentBoxes.value)))
  currentBoxes.value = redoStack.value.pop()!
  selectedBoxIndex.value = -1
  redrawCanvas()
}

// 修改标注框类别（支持撤销）
const onBoxClassChange = (e: Event, idx: number) => {
  const old = currentBoxes.value[idx].class_id
  const val = Number((e.target as HTMLSelectElement).value)
  if (val === old) return
  pushUndo()
  currentBoxes.value[idx].class_id = val
  redrawCanvas()
}

// 类别颜色：按黄金角在 HSL 色环上均匀分配，肉眼可区分
const assignClassColors = (count: number) => {
  const colors: string[] = []
  for (let i = 0; i < count; i++) {
    colors.push(`hsl(${(i * 137.508) % 360}, 70%, 50%)`)
  }
  return colors
}

// 切换"当前类别"时重绘画布，实现按类别过滤显示框
watch(currentClass, () => {
  redrawCanvas()
})

// 类别列表变化时重新生成颜色
watch(classes, (val) => {
  classColors.value = assignClassColors(val.length)
}, { immediate: true })

// 一键清空当前图片的所有标注（针对误把置信度调成0导致标注过多的情况）
const clearCurrentBoxes = async () => {
  if (!currentBoxes.value.length) return
  if (!(await showConfirm(`确定清空当前图片的 ${currentBoxes.value.length} 个标注框吗？\n\n清空后需点击"保存"才会写入文件。`))) return
  pushUndo()
  currentBoxes.value = []
  redrawCanvas()
}

// ===== SAM 大模型预标注 =====
const loadSamConfigAndModels = async () => {
  try {
    const [cfg, models] = await Promise.all([getSamConfig(), getSamModels()])
    samConfig.value = cfg
    samModels.value = models
    if (typeof cfg.conf === 'number') aiConf.value = cfg.conf
    if (typeof cfg.iou === 'number') aiIou.value = cfg.iou
    // 初始化检测模型下拉选中项
    if (cfg.detector === 'grounding_dino') {
      selectedDetector.value = '__grounding_dino__'
    } else {
      selectedDetector.value = cfg.detector_weights ||
        ((models as any[])?.find?.((m: any) => m.name)?.name ?? 'yolov8s-world.pt')
    }
  } catch (e: any) {
    console.error('加载 SAM 配置/模型失败:', e)
  }
}

const applyDetectorModel = async () => {
  const name = selectedDetector.value
  if (!name) return
  try {
    if (name === '__grounding_dino__') {
      await updateSamConfig({ detector: 'grounding_dino', detector_weights: 'IDEA-Research/grounding-dino-tiny' })
      alert('已切换到 GroundingDINO（grounding-dino-tiny）\n（对单张图生效，批量任务会重新加载）')
    } else {
      await updateSamConfig({ detector: 'yolo_world', detector_weights: name })
      alert(`已切换到检测模型: ${name}\n（对单张图生效，批量任务会重新加载）`)
    }
  } catch (e: any) {
    alert('切换模型失败: ' + (e.response?.data?.detail || e.message))
  }
}

const onSamModelFile = async (e: any) => {
  const file = e.target.files?.[0]
  e.target.value = ''
  if (!file) return
  try {
    alert(`正在上传模型 ${file.name}（${(file.size / 1e6).toFixed(1)} MB）...`)
    const res = await uploadSamModel(file)
    await loadSamConfigAndModels()
    samConfig.value.detector = 'yolo_world'
    samConfig.value.detector_weights = res.name
    selectedDetector.value = res.name
    await updateSamConfig({ detector: 'yolo_world', detector_weights: res.name })
    alert(`模型 ${res.name} 上传成功并已启用`)
  } catch (err: any) {
    alert('上传模型失败: ' + (err.response?.data?.detail || err.message))
  }
}

// 计算两个框的 IoU（交并比），用于去重
const iou = (a: any, b: any) => {
  const x1 = Math.max(a.x1, b.x1)
  const y1 = Math.max(a.y1, b.y1)
  const x2 = Math.min(a.x2, b.x2)
  const y2 = Math.min(a.y2, b.y2)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
  const union = areaA + areaB - inter
  return union > 0 ? inter / union : 0
}

const runAutoLabel = async () => {
  if (!currentTask.value || !currentImage.value) return
  // 若正在标注中，再次点击则询问是否取消
  if (aiLabeling.value) {
    if (!(await showConfirm('是否取消当前 AI 标注？'))) return
    aiAbortCtrl.value?.abort()
    aiLabeling.value = false
    aiAbortCtrl.value = null
    return
  }
  if (!classes.value.length) {
    alert('请先设置类别')
    return
  }
  aiLabeling.value = true
  aiAbortCtrl.value = new AbortController()
  try {
    const result = await autoLabelImage(
      currentTask.value,
      currentImage.value.image_id,
      classes.value,
      aiConf.value,
      buildPrompts(classes.value.map((c: any) => (typeof c === 'string' ? c : c.name))),
      aiAbortCtrl.value.signal
    )
    if (result.error) {
      alert('AI 预标注失败: ' + result.error)
      return
    }
    const aiBoxes = (result.boxes || []).map((b: any) => ({
      class_id: b.class_id,
      x1: b.x1,
      y1: b.y1,
      x2: b.x2,
      y2: b.y2
    }))
    // 按 IoU 去重：与已有任意类别框重叠 >=0.5 的视为同一目标，避免重复标注（含跨类别）
    const newAiBoxes = aiBoxes.filter(
      (ab: any) => !currentBoxes.value.some((cb: any) => iou(cb, ab) >= 0.5)
    )
    currentBoxes.value.push(...newAiBoxes)
    if (newAiBoxes.length > 0) {
      currentImageHasAi.value = true
      markDirty()
    }
    redrawCanvas()
    if (!aiBoxes.length) {
      alert('未检测到目标，可调低置信度重试')
    } else if (!newAiBoxes.length) {
      alert(`检测到 ${aiBoxes.length} 个目标，但均与已有标注重复，未新增`)
    }
  } catch (e: any) {
    if (e?.code === 'ERR_CANCELED' || e?.message === 'canceled') {
      alert('已取消 AI 标注')
    } else {
      alert('AI 预标注失败: ' + (e.response?.data?.detail || e.message))
    }
  } finally {
    aiLabeling.value = false
    aiAbortCtrl.value = null
  }
}

const runBatchLabel = async () => {
  if (!currentTask.value) return
  if (!classes.value.length) {
    alert('请先设置类别')
    return
  }
  if (!(await showConfirm(`确认对 ${items.value.length} 张图片执行批量 AI 预标注？已标注的类别不会被覆盖，仅会为尚未标注的新类别追加检测框。`))) return
  try {
    const result = await startBatchLabel(
      currentTask.value,
      classes.value,
      aiConf.value,
      buildPrompts(classes.value.map((c: any) => (typeof c === 'string' ? c : c.name)))
    )
    batchId.value = result.batch_id
    batchProgress.value = { status: 'running', total: result.total, done: 0, boxes_written: 0, current_image: '' }
    pollBatch()
  } catch (e: any) {
    alert('启动批量预标注失败: ' + (e.response?.data?.detail || e.message))
  }
}

const pollBatch = async () => {
  if (!batchId.value) return
  let res: any
  try {
    res = await getBatchProgress(batchId.value)
  } catch {
    batchTimer.value = setTimeout(pollBatch, 2000)
    return
  }
  if (!res) {
    // 批量任务不存在（例如后端重启导致内存任务丢失），已标注的图片已保存，不丢数据
    batchId.value = null
    batchProgress.value = null
    saveAnnotateState()
    alert('批量预标注任务已中断（可能因后端重启），之前已完成图片的标注已保存。')
    await loadTaskItems()
    return
  }
  batchProgress.value = res
  if (res.status === 'done') {
    batchId.value = null
    saveAnnotateState()
    alert(`批量预标注完成！共标注 ${res.summary?.annotated ?? res.boxes_written} 张图片`)
    await loadTaskItems()
  } else if (res.status === 'cancelled') {
    batchId.value = null
    saveAnnotateState()
    const done = res.done || 0
    const annotated = res.annotated_images ?? done
    alert(`批量预标注已取消，已标注 ${annotated} 张图片`)
    await loadTaskItems()
  } else if (res.status === 'error') {
    batchId.value = null
    saveAnnotateState()
    alert('批量预标注出错: ' + (res.error || '未知错误'))
  } else {
    saveAnnotateState()
    batchTimer.value = setTimeout(pollBatch, 1500)
  }
}

// 一键清除本任务所有 AI 预标注（误标过多时清理重标）
const clearAllAiLabels = async () => {
  if (!currentTask.value) return
  if (!confirm(`确定清除本任务全部 ${counts.value.ai} 张 AI 预标注图片吗？\n将删除这些图片的标注（含之后人工修改），不可恢复。`)) return
  try {
    const res = await clearAiAnnotations(currentTask.value)
    alert(`已清除 ${res.removed ?? 0} 张图片的 AI 标注，可重新标注`)
    await loadTaskItems()
    if (currentImage.value?.annotated) {
      await loadExistingAnnotation()
    } else {
      currentBoxes.value = []
      currentImageHasAi.value = false
      loadImageToCanvas()
    }
  } catch (e: any) {
    alert('清除失败: ' + (e.response?.data?.detail || e.message))
  }
}

// 一键清理本任务全部图片的高度重合标注框（历史遗留重叠框统一根治，对所有检测模型生效）
const cleanOverlaps = async () => {
  if (!currentTask.value) return
  if (!confirm(
    '将对本任务全部图片执行重叠框清理：\n' +
    '1. 跨类别高度重合（IoU≥0.5）只保留一个\n' +
    '2. 同类高度重叠/紧邻的框合并\n\n' +
    '此操作会覆盖图片标注，不可恢复，是否继续？'
  )) return
  try {
    const res = await cleanTaskOverlaps(currentTask.value)
    alert(`清理完成：${res.images_cleaned ?? 0} 张图片、${res.boxes_removed ?? 0} 个重叠框被清理`)
    await loadTaskItems()
    if (currentImage.value?.annotated) {
      await loadExistingAnnotation()
    } else {
      currentBoxes.value = []
      currentImageHasAi.value = false
      loadImageToCanvas()
    }
  } catch (e: any) {
    alert('清理失败: ' + (e.response?.data?.detail || e.message))
  }
}

const stopBatch = async () => {
  if (!batchId.value) return
  try {
    await stopBatchLabel(batchId.value)
    // 提示已标注的图片数量后清理本地状态
    const done = batchProgress.value?.done || 0
    const annotated = batchProgress.value?.annotated_images ?? done
    batchId.value = null
    batchProgress.value = null
    saveAnnotateState()
    alert(`批量预标注已取消，已标注 ${annotated} 张图片`)
    await loadTaskItems()
  } catch (e: any) {
    // 任务不存在（例如后端重启后内存任务清空），清理本地状态并友好提示
    if (e.response?.status === 404) {
      batchId.value = null
      batchProgress.value = null
      saveAnnotateState()
      alert('预标注任务已不存在（可能已重启），已停止')
      return
    }
    alert('停止失败: ' + (e.response?.data?.detail || e.message))
  }
}

onUnmounted(() => {
  if (batchTimer.value) {
    clearTimeout(batchTimer.value)
  }
})

// 保存标注（silent=true 用于自动保存，不弹提示打扰）
const persistAnnotation = async (silent = false): Promise<boolean> => {
  if (!currentTask.value || !currentImage.value) return false
  if (savingAnnotation.value) return false
  savingAnnotation.value = true
  try {
    const aiFlag = currentImageHasAi.value || !!currentImage.value.ai_annotated
    await saveAnn(currentTask.value, currentImage.value.image_id, currentBoxes.value, aiFlag)
    items.value[currentIndex.value].annotated = true
    if (aiFlag) {
      items.value[currentIndex.value].ai_annotated = true
    }
    dirty.value = false
    if (!silent) alert('保存成功!')
    return true
  } catch (error: any) {
    if (!silent) alert('保存失败: ' + (error.response?.data?.detail || error.message))
    return false
  } finally {
    savingAnnotation.value = false
  }
}

const saveAnnotation = () => persistAnnotation(false)

// 标记脏状态并防抖自动保存（所有编辑操作经 pushUndo 触发）
const markDirty = () => {
  dirty.value = true
  clearTimeout(autoSaveTimer.value)
  autoSaveTimer.value = setTimeout(() => persistAnnotation(true), 1500)
}

// 在线调置信度/IoU 阈值：保存到后端配置，AI 标注与交互式标注立即生效
const onThreshChange = async () => {
  try {
    await updateSamConfig({ conf: aiConf.value, iou: aiIou.value })
    alert(`已保存阈值：置信度 ${aiConf.value}，IoU ${aiIou.value}`)
  } catch (e: any) {
    alert('保存阈值失败: ' + (e.response?.data?.detail || e.message))
  }
}

// 导入 YOLO 标注（zip 内含 labels/*.txt）
const onImportYolo = async (e: any) => {
  const file = e.target.files?.[0]
  e.target.value = ''
  if (!file || !currentTask.value) return
  if (!(await showConfirm(
    `导入 ${file.name}？\n\n` +
    `将按文件名匹配任务图片，写入 YOLO 标注。\n` +
    `同名图片的现有标注会被导入内容覆盖，是否继续？`
  ))) return
  importing.value = true
  try {
    const res = await importYoloLabels(currentTask.value, file)
    const msg = `导入完成：${res.imported ?? 0} 张图片`
    const skipped = (res.skipped || []).slice(0, 5).join('、')
    alert(skipped ? `${msg}\n跳过未匹配: ${skipped}${res.skipped.length > 5 ? ' ...' : ''}` : msg)
    await loadTaskItems()
    if (currentImage.value?.annotated) {
      await loadExistingAnnotation()
    }
  } catch (err: any) {
    alert('导入失败: ' + (err.response?.data?.detail || err.message))
  } finally {
    importing.value = false
  }
}

const exportTask = async () => {
  if (!currentTask.value) return
  
  // 校验比例
  const sum = ratioTrain.value + ratioVal.value + ratioTest.value
  if (sum <= 0) {
    alert('划分比例之和必须大于0')
    return
  }
  if (!(await showConfirm(
    `确认导出并自动划分数据集？\n\n` +
    `训练:验证:测试 = ${ratioTrain.value} : ${ratioVal.value} : ${ratioTest.value}\n\n` +
    `导出结构：\n` +
    `  train/images + train/labels\n` +
    `  val/images + val/labels\n` +
    `  test/images + test/labels\n` +
    `  data.yaml（根目录）\n\n` +
    `点击确定后将写入数据集目录。`
  ))) return

  exporting.value = true
  try {
    const result = await exportAnnotationsSplit(
      currentTask.value,
      ratioTrain.value,
      ratioVal.value,
      ratioTest.value
    )
    const c = result.counts || {}
    alert(
      `导出并划分成功！共 ${result.total} 张已标注图片\n\n` +
      `  训练集: ${c.train} 张\n` +
      `  验证集: ${c.val} 张\n` +
      `  测试集: ${c.test} 张\n` +
      `data.yaml 已更新，可直接用于训练。`
    )
  } catch (error: any) {
    alert('导出失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    exporting.value = false
  }
}

const prevImage = () => {
  if (currentIndex.value > 0) {
    selectImage(currentIndex.value - 1)
  }
}

const nextImage = () => {
  if (currentIndex.value < items.value.length - 1) {
    selectImage(currentIndex.value + 1)
  }
}

const handleKeydown = (e: KeyboardEvent) => {
  const tag = (e.target as HTMLElement)?.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
  if (!currentTask.value || loadingAnnotation.value) return

  // Ctrl/Cmd 组合键
  if (e.ctrlKey || e.metaKey) {
    const k = e.key.toLowerCase()
    if (k === 's') {
      e.preventDefault()
      saveAnnotation()
    } else if (k === 'z') {
      e.preventDefault()
      if (e.shiftKey) redo()
      else undo()
    } else if (k === 'y') {
      e.preventDefault()
      redo()
    }
    return
  }

  // 空格：按住平移（只在画布未被聚焦交互时启用）
  if (e.code === 'Space') {
    e.preventDefault()
    spaceDown.value = true
    return
  }

  switch (e.key) {
    case 'Escape':
      if (interactiveMode.value) {
        interactiveMode.value = false
        interactiveRegion.value = null
        redrawCanvas()
      }
      break
    case 'ArrowRight':
    case 'd':
    case 'D':
      e.preventDefault()
      nextImage()
      break
    case 'ArrowLeft':
    case 'a':
    case 'A':
      e.preventDefault()
      prevImage()
      break
    case 'Delete':
    case 'Backspace':
      if (selectedBoxIndex.value >= 0) {
        e.preventDefault()
        removeBox(selectedBoxIndex.value)
      }
      break
  }
}

// 松开空格等按键时复位状态
const handleKeyup = (e: KeyboardEvent) => {
  if (e.code === 'Space') {
    spaceDown.value = false
    if (isPanning.value) {
      isPanning.value = false
      redrawCanvas()
    }
  }
}

// 点击页面其他区域时关闭筛选下拉菜单
const handleDocClick = () => {
  filterOpen.value = false
}

onMounted(() => {
  loadDatasets()
  loadSamConfigAndModels()
  window.addEventListener('keydown', handleKeydown)
  window.addEventListener('keyup', handleKeyup)
  window.addEventListener('resize', fitView)
  document.addEventListener('click', handleDocClick)
  restoreAnnotateState()
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
  window.removeEventListener('keyup', handleKeyup)
  window.removeEventListener('resize', fitView)
  document.removeEventListener('click', handleDocClick)
  // 离页时若有未保存改动，尝试立即保存
  if (dirty.value) persistAnnotation(true)
  saveAnnotateState()
})
</script>

<style scoped>
.info-box {
  background: #fff3cd;
  border: 1px solid #ffc107;
  padding: 0.75rem;
  border-radius: 4px;
  margin-bottom: 1rem;
  color: #856404;
}

.form-group small {
  display: block;
  margin-top: 0.25rem;
  color: #7f8c8d;
  font-size: 0.875rem;
}

.annotate-workspace {
  display: grid;
  grid-template-columns: 200px 1fr 250px;
  gap: 1rem;
  margin-top: 1rem;
}

.sidebar, .canvas-area, .annotations-panel {
  background: white;
  border-radius: 8px;
  padding: 1rem;
}

.split-config {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  padding: 0.4rem 0;
  font-size: 0.85rem;
  color: #555;
}

.split-label {
  font-weight: 600;
  color: #333;
}

.ratio-input {
  width: 60px;
  padding: 0.3rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-left: 0.2rem;
}

.image-list {
  max-height: 400px;
  overflow-y: auto;
  margin: 1rem 0;
}

.image-filter {
  position: relative;
  margin: 0.75rem 0 0.25rem;
}

.filter-dropdown {
  position: relative;
}

.filter-trigger {
  width: 100%;
  padding: 0.4rem 0.6rem;
  font-size: 0.85rem;
  border-radius: 6px;
  border: 1px solid #ddd;
  background: #fff;
  color: #333;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.filter-trigger:hover {
  border-color: #3498db;
}

.filter-caret {
  font-size: 0.7rem;
  color: #95a5a6;
}

.filter-menu {
  position: absolute;
  top: calc(100% + 4px);
  left: 0;
  right: 0;
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
  z-index: 20;
  padding: 0.25rem 0;
}

.filter-menu-item {
  padding: 0.4rem 0.75rem;
  font-size: 0.85rem;
  color: #333;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.filter-menu-item:hover {
  background: #f0f6ff;
}

.filter-menu-item.active {
  background: #3498db;
  color: #fff;
}

.filter-menu-item.sub {
  padding-left: 1.5rem;
  color: #555;
}

.filter-menu-group {
  border-top: 1px solid #f0f0f0;
  border-bottom: 1px solid #f0f0f0;
  margin: 0.15rem 0;
}

.filter-menu-item.group-header {
  font-weight: 600;
}

.filter-count {
  font-size: 0.7rem;
  background: rgba(0, 0, 0, 0.08);
  border-radius: 8px;
  padding: 0 0.35rem;
  line-height: 1.3;
  flex-shrink: 0;
}

.filter-menu-item.active .filter-count {
  background: rgba(255, 255, 255, 0.25);
}

.empty-filter {
  padding: 1rem;
  color: #95a5a6;
  text-align: center;
  font-size: 0.85rem;
}

.image-item {
  padding: 0.5rem;
  cursor: pointer;
  border-radius: 4px;
  margin-bottom: 0.25rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.4rem;
}

.image-item-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.image-item-badges {
  display: flex;
  gap: 0.25rem;
  flex-shrink: 0;
}

.img-badge {
  font-size: 0.7rem;
  line-height: 1;
  padding: 0.2rem 0.35rem;
  border-radius: 4px;
  font-weight: 600;
}

.img-badge.ok {
  color: #2ecc71;
  background: rgba(46, 204, 113, 0.12);
}

.img-badge.ai {
  color: #fff;
  background: #e67e22;
}

.image-item:hover {
  background: #ecf0f1;
}

.image-item.active {
  background: #3498db;
  color: white;
}

.image-item.active .img-badge.ok {
  color: #fff;
  background: transparent;
}

.image-item.active .img-badge.ai {
  color: #fff;
  background: rgba(255, 255, 255, 0.35);
}

.image-item.annotated {
  border-left: 3px solid #2ecc71;
}

.nav-buttons {
  display: flex;
  gap: 0.5rem;
}

.nav-buttons button {
  flex: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.25rem;
}

.canvas-container {
  position: relative;
  border: 2px solid #ddd;
  overflow: hidden;
  max-height: 600px;
  cursor: crosshair;
  /* 图片小于显示框时居中；图片过大溢出时 margin 自动归零，从左上滚动不受影响 */
  display: flex;
}

/* 画布缩放/平移工具条 */
.view-toolbar {
  position: absolute;
  bottom: 8px;
  right: 8px;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 0.8rem;
  z-index: 6;
}

.view-hint {
  color: #ccc;
  font-size: 0.75rem;
  margin-right: 0.5rem;
}

.view-toolbar .small {
  min-width: 26px;
  height: 26px;
  padding: 0;
  line-height: 1;
}

/* 左侧标注框条目标签前的颜色块 */
.box-color {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 3px;
  margin-right: 0.5rem;
  flex: none;
}

/* 交互式标注提示条与激活按钮 */
.interactive-hint {
  font-size: 0.8rem;
  color: #e67e22;
  background: #fdf3e7;
  border: 1px solid #f5c6a0;
  border-radius: 4px;
  padding: 0.4rem 0.6rem;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.annotations-panel button.active {
  background: #e67e22;
  color: #fff;
  border-color: #e67e22;
}

/* 导入 YOLO 标注按钮（label 包裹 file input） */
.import-label {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.45rem 0.8rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: #fff;
  cursor: pointer;
  font-size: 0.9rem;
}

.import-label:hover {
  border-color: #42b983;
  color: #42b983;
}

.import-input {
  display: none;
}

.ai-btn.small {
  padding: 0.3rem 0.6rem;
  font-size: 0.8rem;
}

.img-nav-btn {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 44px;
  height: 44px;
  padding: 0;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.45);
  color: #fff;
  font-size: 28px;
  font-weight: bold;
  line-height: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  z-index: 5;
  border: none;
  transition: background 0.2s;
}

.img-nav-btn.prev {
  left: 8px;
}

.img-nav-btn.next {
  right: 8px;
}

.img-nav-btn:hover:not(:disabled) {
  background: rgba(52, 152, 219, 0.9);
}

.img-nav-btn:disabled {
  background: rgba(0, 0, 0, 0.25);
  opacity: 0.4;
  cursor: not-allowed;
}

canvas {
  display: block;
  margin: auto;
}

.controls {
  margin-top: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.controls select {
  flex: 1;
}

.box-list {
  margin: 1rem 0;
  max-height: 400px;
  overflow-y: auto;
}

.box-item {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-bottom: 0.5rem;
  cursor: pointer;
  transition: border-color 0.15s, background-color 0.15s;
}

.box-item:hover {
  border-color: #aaa;
}

.box-item-selected {
  border-color: orange;
  background: rgba(255, 165, 0, 0.12);
}

.box-class-select {
  flex: 1;
  margin-right: 0.5rem;
  padding: 0.25rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  min-width: 0;
}

.box-class-name {
  font-size: 0.85rem;
  color: #333;
  margin-right: 0.5rem;
  white-space: normal;
  word-break: break-all;
  flex: 1 1 100%;
  min-width: 0;
}

.box-empty {
  color: #999;
  font-size: 0.85rem;
  padding: 0.5rem 0;
  text-align: center;
}

.annotations-panel button {
  width: 100%;
  margin-bottom: 0.5rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
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

.refresh-btn {
  padding: 0.5rem 1rem;
  white-space: nowrap;
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}

.import-info {
  background: #d4edda;
  border: 1px solid #c3e6cb;
  color: #155724;
  padding: 0.5rem;
  border-radius: 4px;
  font-size: 0.875rem;
  margin-bottom: 0.5rem;
}

.loading-state {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 2rem;
  color: #7f8c8d;
}

.loading-indicator {
  font-size: 0.875rem;
  color: #7f8c8d;
  font-weight: normal;
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}

button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.25rem;
}

button.small {
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
}

.ai-btn {
  background: #8e44ad;
  color: white;
  border: none;
  padding: 0.4rem 0.75rem;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap;
}
.ai-btn:hover:not(:disabled) {
  background: #732d91;
}

.ai-toolbar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.6rem 0.75rem;
  margin-bottom: 0.75rem;
  background: linear-gradient(90deg, #f3e8f9, #eef3fb);
  border: 1px solid #d9c4e8;
  border-radius: 6px;
  flex-wrap: wrap;
}

.ai-btn {
  background: #8e44ad;
  color: white;
  border: none;
  padding: 0.5rem 1.1rem;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap;
  font-weight: bold;
  font-size: 0.95rem;
  box-shadow: 0 2px 4px rgba(142, 68, 173, 0.3);
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}

.ai-conf-label {
  margin-left: 0.25rem;
  color: #555;
}

.conf-input {
  width: 70px;
  padding: 0.35rem;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.model-select {
  padding: 0.35rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  max-width: 180px;
  background: white;
}

.model-add-label {
  cursor: pointer;
  background: #eef0f3;
  border: 1px solid #d0d3d8;
  border-radius: 4px;
  padding: 0.35rem 0.7rem;
  color: #333;
  font-size: 0.9rem;
  white-space: nowrap;
}

.model-add-label:hover {
  background: #e2e5ea;
}

.model-add-input {
  display: none;
}

.batch-progress {
  margin: 0.75rem 0;
  padding: 0.5rem;
  background: #eef3fb;
  border-radius: 4px;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  font-size: 0.875rem;
}

.progress-bar {
  height: 8px;
  background: #dfe6ed;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #3498db;
  transition: width 0.3s ease;
}
</style>

<template>
  <div class="model-detail">
    <button class="back-btn" @click="$router.push('/')">← 返回仓库</button>

    <div v-if="loading" class="loading-state"><span class="loading-spinner"></span>加载模型库详情...</div>

    <div v-else-if="!model" class="empty-state">模型库不存在或已被删除</div>

    <template v-else>
      <!-- ===== A. 顶部 Header 区域（模型库工作台） ===== -->
      <div class="card wk-header">
        <div class="wk-header-top">
          <div class="wk-title">
            <h1>{{ model.display_name || model.name || model.model_code || model.model_id }}</h1>
            <code class="model-code">{{ model.model_code || model.model_id }}</code>
            <span class="status-tag" :class="model.status">{{ statusLabel(model.status) }}</span>
          </div>
          <button class="ghost small" @click="goSettings" title="进入模型库设置">
            ⚙ 设置
          </button>
        </div>
        <div class="wk-desc">
          {{ model.description || '暂无业务场景描述，点击「编辑简介」补充该模型库的用途说明。' }}
        </div>
        <div class="wk-meta">
          <span>算法架构：{{ archLabel }}</span>
          <span class="meta-sep">|</span>
          <span>标签版本：{{ labelVersionLabel }}</span>
        </div>
        <div class="wk-actions">
          <button class="ghost small" @click="editDescOpen = !editDescOpen">
            {{ editDescOpen ? '收起' : '编辑简介' }}
          </button>
          <button class="ghost small" @click="openLabelModal">管理标签</button>
        </div>
        <div v-if="editDescOpen" class="wk-edit-desc">
          <textarea v-model="descDraft" rows="3" placeholder="描述该模型库的业务场景、识别要求等"></textarea>
          <div class="wk-edit-desc-actions">
            <button class="primary small" :disabled="savingDesc" @click="saveDescription">
              {{ savingDesc ? '保存中...' : '保存' }}
            </button>
            <button class="secondary small" @click="editDescOpen = false">取消</button>
          </div>
          <p v-if="descSaveMsg" class="wk-save-msg" :class="{ err: descSaveErr }">{{ descSaveMsg }}</p>
        </div>
      </div>

      <!-- ===== B. 主操作栏 ===== -->
      <div class="wk-ops">
        <button class="primary" @click="goTrain">
          ▶ 发起训练
        </button>
        <button class="secondary" @click="importFileInput?.click()">
          ⬆ 导入外部模型
        </button>
        <input ref="importFileInput" type="file" accept=".pt,.onnx" class="pt-file-input" @change="handleImportModel" />
      </div>

      <!-- ===== C. Tab 切换区域 ===== -->
      <div class="wk-tabs">
        <button
          v-for="t in wkTabs"
          :key="t.key"
          class="wk-tab"
          :class="{ on: activeTab === t.key }"
          @click="activeTab = t.key"
        >
          {{ t.label }}
        </button>
      </div>

      <!-- ---------- C1. 概览 ---------- -->
      <div v-if="activeTab === 'overview'" class="card wk-pane">
        <div class="wk-pane-head"><h2>模型库概览</h2></div>
        <div class="ov-grid">
          <div class="ov-card">
            <div class="ov-label">业务场景</div>
            <div class="ov-value">{{ model.description || '—' }}</div>
          </div>
          <div class="ov-card">
            <div class="ov-label">算法架构</div>
            <div class="ov-value">{{ archLabel || '—' }}</div>
          </div>
          <div class="ov-card">
            <div class="ov-label">标签体系（{{ labelsDict.labels?.length || 0 }} 类）</div>
            <div class="ov-tags" v-if="labelsDict.labels?.length">
              <span v-for="l in labelsDict.labels" :key="l.index" class="ov-tag">
                {{ l.index }}·{{ l.english_code }}{{ l.chinese_name ? ' ' + l.chinese_name : '' }}
              </span>
            </div>
            <div class="ov-value" v-else>尚未定义标签体系，可在「设置」中创建</div>
          </div>
          <div class="ov-card">
            <div class="ov-label">最新模型权重</div>
            <div v-if="model.version">
              <div class="ov-value">
                版本 {{ normVer(model.version) }}<span v-if="latestMAP">（mAP {{ latestMAP }}）</span>
              </div>
              <button class="secondary small mt4" :disabled="downloading" @click="downloadWeights">
                {{ downloading ? '下载中...' : '⬇ 下载 .pt' }}
              </button>
            </div>
            <div class="ov-value" v-else>该模型库尚无训练产出，发起训练后即可下载</div>
          </div>
        </div>
      </div>

      <!-- ---------- C2. 数据集管理（核心改造区） ---------- -->
      <div v-else-if="activeTab === 'datasets'" class="card wk-pane">
        <div class="wk-pane-head">
          <h2>数据集管理（{{ datasetsView.length }}）</h2>
        </div>

        <!-- 搜索 + 筛选 + 导入新数据 -->
        <div class="ds-toolbar">
          <input v-model="dsKeyword" class="ds-search" placeholder="搜索数据集..." />
          <select v-model="dsFilter" class="ds-filter">
            <option value="">全部状态</option>
            <option value="todo">待标注</option>
            <option value="annotating">标注中</option>
            <option value="pending">待封板</option>
            <option value="sealed">已封板</option>
            <option value="unassigned">未分配</option>
            <option value="failed">训练失败</option>
          </select>
          <router-link to="/datasets"><button class="primary small">+ 导入新数据</button></router-link>
        </div>

        <!-- 空状态：新手引导 -->
        <div v-if="filteredDatasets.length === 0" class="wk-empty">
          <div v-if="datasetsView.length === 0" class="empty-state">
            <h3>该模型库还没有数据集</h3>
            <p>完成以下三步即可训练出第一个模型：</p>
            <ol class="wk-steps">
              <li>① 上传数据：到「数据集」页导入图片（或点上方「+ 导入新数据」）</li>
              <li>② 标注：在标注工具中完成类别标注并导出</li>
              <li>③ 训练：点击「发起训练」开始训练，产出模型版本</li>
            </ol>
          </div>
          <div v-else class="empty-state">没有符合当前筛选条件的数据集</div>
        </div>

        <table class="ds-table" v-else>
          <thead>
            <tr>
              <th>数据集名称</th>
              <th>阶段</th>
              <th>图片数</th>
              <th>标注框</th>
              <th>状态</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in filteredDatasets" :key="d.dataset_id">
              <td>
                <div class="ds-name-main">{{ d.display_name || d.filename || d.dataset_id }}</div>
              </td>
              <td><span class="life-pill" :class="`life-${lifecycleOf(d).key}`">{{ lifecycleOf(d).text }}</span></td>
              <td>{{ d.image_count ?? '—' }}</td>
              <td>{{ d.label_count ?? '—' }}</td>
              <td>
                <span class="train-mark" :class="d.training_status === 'completed' ? 'ok' : 'todo'">
                  {{ d.training_status === 'completed' ? '已完成训练' : (d.training_status && d.training_status !== 'incomplete' ? d.training_status : (d.status === 'prepared' ? '已就绪' : '待处理')) }}
                </span>
              </td>
              <td>
                <div class="row-actions">
                  <!-- 行主按钮（实心，仅一个） -->
                  <router-link v-if="mainActionOf(d).to" :to="mainActionOf(d).to">
                    <button class="primary small">{{ mainActionOf(d).label }}</button>
                  </router-link>
                  <button v-else class="primary small" @click="onMainAction(d)">{{ mainActionOf(d).label }}</button>
                  <!-- 次要操作折叠 -->
                  <div class="more-menu" v-if="d.model_id">
                    <button class="ghost small more-btn" @click="toggleMore(d)">⋯</button>
                    <div v-if="moreOpen === d.dataset_id" class="more-dropdown" @click="moreOpen = ''">
                      <button class="dropdown-item" @click="revalidate(d)">校验</button>
                      <button class="dropdown-item" @click="unbind(d)">解绑</button>
                      <router-link :to="`/datasets?focus=${d.dataset_id}`" class="dropdown-item link">查看详情</router-link>
                      <button class="dropdown-item danger" @click="removeDs(d)">删除</button>
                    </div>
                  </div>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
        <p class="form-hint">说明：未分配数据集不在任何模型库，可点「挂载」加入本库；待封板数据集数量达到门槛后可直接封板。</p>
      </div>

      <!-- ---------- C3. 模型版本 ---------- -->
      <div v-else-if="activeTab === 'versions'" class="card wk-pane">
        <div class="wk-pane-head">
          <h2>模型版本（{{ versions.length }}）</h2>
          <button class="secondary small" :disabled="charting" @click="downloadCharts">
            {{ charting ? '生成中...' : '📈 生成训练图表' }}
          </button>
        </div>

        <div v-if="versions.length === 0" class="empty-state">该模型库还没有训练产出的模型版本</div>

        <div v-for="(v, i) in versions" :key="v.model_id || i" class="ver-card">
          <div class="ver-head">
            <span class="ver-name">{{ normVer(v.version) }}<span v-if="i === 0" class="ver-latest">最新</span></span>
            <span class="status-tag" :class="v.status">{{ statusLabel(v.status) }}</span>
          </div>
          <div class="ver-metrics">
            <div class="ver-metric">
              <span class="vm-label">mAP50-95</span>
              <b>{{ fmtPct(metricOf(v, 'mAP50_95')) }}</b>
            </div>
            <div class="ver-metric">
              <span class="vm-label">Precision</span>
              <b>{{ fmtPct(metricOf(v, 'precision')) }}</b>
            </div>
            <div class="ver-metric">
              <span class="vm-label">Recall</span>
              <b>{{ fmtPct(metricOf(v, 'recall')) }}</b>
            </div>
            <div class="ver-metric">
              <span class="vm-label">参数量</span>
              <b>{{ v.model_info?.total_params_m ? v.model_info.total_params_m + 'M' : '—' }}</b>
            </div>
          </div>
          <div class="ver-meta">
            <span>产出时间：{{ fmtTime(v.created_at) }}</span>
            <span v-if="v.lineage?.parent_model_id">基于 {{ v.lineage.parent_model_id }} 迭代</span>
            <span v-if="v.lineage?.base_model">基座 {{ v.lineage.base_model }}</span>
          </div>
          <!-- 守门员对比（当前 vs 上一代） -->
          <div v-if="v.gatekeeper" class="gk-compare">
            <div class="gk-box">
              <div class="gk-title">守门员评估</div>
              <div class="gk-report-text">{{ v.gatekeeper.report }}</div>
              <div v-if="v.gatekeeper.regressed_classes?.length" class="gk-regressed">
                退化类别：
                <span v-for="c in v.gatekeeper.regressed_classes" :key="c" class="gk-regressed-item">{{ c }}</span>
              </div>
              <div v-if="metricsDelta" class="gk-delta">
                对比上一代：mAP50-95 {{ metricsDelta.deltaText }}
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ---------- C4. 训练任务历史 ---------- -->
      <div v-else-if="activeTab === 'history'" class="card wk-pane">
        <div class="wk-pane-head">
          <h2>训练任务历史（{{ historyJobs.length }}）</h2>
          <button class="secondary small" :disabled="loadingJobs" @click="loadJobs">刷新</button>
        </div>
        <div v-if="historyJobs.length === 0" class="empty-state">该模型库还没有训练任务</div>
        <table class="ds-table" v-else>
          <thead>
            <tr>
              <th>任务</th>
              <th>状态</th>
              <th>数据集</th>
              <th>模型 / 轮数</th>
              <th>启动时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="j in historyJobs" :key="j.job_id">
              <td><div class="ds-name-main">{{ j.job_id }}</div></td>
              <td>
                <span class="life-pill" :class="`job-${j.status}`">{{ jobStatusText(j.status) }}</span>
              </td>
              <td>{{ j.dataset_id }}</td>
              <td>{{ j.model_name }} · {{ j.epochs }} 轮</td>
              <td>{{ fmtTime(j.created_at) }}</td>
              <td>
                <div class="row-actions">
                  <router-link :to="`/train`"><button class="ghost small">查看</button></router-link>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- ---------- C5. 设置（简介 / 标签字典管理 / 静态信息） ---------- -->
      <div v-else-if="activeTab === 'settings'" ref="settingsPane" class="card wk-pane">
        <div class="wk-pane-head"><h2>模型库设置</h2></div>

        <div class="set-group">
          <div class="set-label">显示名称</div>
          <div class="set-row">
            <input v-model="nameDraft" class="ds-search" style="max-width: 340px;" />
            <button class="secondary small" :disabled="savingDesc" @click="saveName">保存</button>
          </div>
        </div>
        <div class="set-group">
          <div class="set-label">静态信息</div>
          <div class="static-info">
            <div><span>模型库 ID</span><code>{{ model.model_id }}</code></div>
            <div v-if="model.model_code"><span>唯一标识</span><code>{{ model.model_code }}</code></div>
            <div><span>业务标签</span><span>{{ bizLabel }}</span></div>
            <div v-if="model.weights_path"><span>权重路径</span><code class="path">{{ model.weights_path }}</code></div>
            <div><span>创建时间</span><span>{{ fmtTime(model.created_at) }}</span></div>
          </div>
        </div>

        <!-- 危险区 -->
        <div class="set-group danger-zone">
          <div class="set-label">危险操作</div>
          <button class="danger" :disabled="deleting" @click="removeModel">
            {{ deleting ? '删除中...' : '删除模型库' }}
          </button>
          <p class="form-hint">删除模型库将同时清除其模型版本与权重文件，且不可恢复！</p>
        </div>
      </div>

      <!-- ---------- D. 管理标签弹窗（标签字典编辑） ---------- -->
      <div v-if="labelModalOpen" class="modal-mask" @click.self="closeLabelModal">
        <div class="modal-panel">
          <div class="modal-head">
            <h3>管理标签字典（{{ labelsDict.labels?.length || 0 }} 类）</h3>
            <button class="ghost small" @click="closeLabelModal">✕</button>
          </div>
          <p class="form-hint">Class ID 由系统自动分配不可修改；英文名影响标注工具与训练；中文别名/描述仅用于展示。</p>
          <table class="ds-table label-table">
            <thead>
              <tr>
                <th style="width: 80px;">Class ID</th>
                <th>英文名</th>
                <th>中文名</th>
                <th>描述</th>
                <th style="width: 60px;"></th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(l, idx) in labelRows" :key="l._tmp || l.index">
                <td>{{ l.index }}</td>
                <td><input v-model="l.english_code" class="cell-input" placeholder="如 person" /></td>
                <td><input v-model="l.chinese_name" class="cell-input" placeholder="如 行人" /></td>
                <td><input v-model="l.chinese_desc" class="cell-input" placeholder="可选" /></td>
                <td><button class="ghost small danger" @click="labelRows.splice(idx, 1)">删</button></td>
              </tr>
            </tbody>
          </table>
          <div class="set-row modal-actions">
            <button class="ghost small" @click="addLabelRow">+ 添加类别</button>
            <button class="primary small" :disabled="savingLabels" @click="saveLabels">
              {{ savingLabels ? '保存中...' : '保存标签字典' }}
            </button>
            <button class="secondary small" @click="closeLabelModal">关闭</button>
            <span v-if="labelsMsg" class="labels-msg">{{ labelsMsg }}</span>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import {
  getModel, getModelDatasets, updateModel, deleteModel, exportModel,
  getLabelsDict, updateLabelsDict, generateTrainingCharts, listModels,
  uploadModel,
  type ModelDataset, type LabelsDict
} from '@/api/models'
import { listDatasets, bindDatasetModel, validateDataset, sealDataset, deleteDataset } from '@/api/datasets'
import { listTrainJobs } from '@/api/train'

const route = useRoute()
const modelId = String(route.params.modelId)

const loading = ref(true)
const model = ref<any>(null)
const allModels = ref<any[]>([])
const mountedDatasets = ref<ModelDataset[]>([])
const unboundDatasets = ref<ModelDataset[]>([])
const labelsDict = ref<LabelsDict>({ labels: [] })
const historyJobs = ref<any[]>([])
const loadingJobs = ref(false)
const sealMinImages = ref(0)

// ===== Header / 概览 =====
const editDescOpen = ref(false)
const descDraft = ref('')
const savingDesc = ref(false)
const descSaveMsg = ref('')
const descSaveErr = ref(false)
const nameDraft = ref('')
const downloading = ref(false)
const charting = ref(false)
const importFileInput = ref<HTMLInputElement | null>(null)

// ===== Tab =====
const activeTab = ref('overview')

// 设置定位与标签弹窗（二取一：⚙ 设置 → 定位设置区；管理标签 → 弹窗就地编辑）
const settingsPane = ref<HTMLElement | null>(null)
const labelModalOpen = ref(false)

function goSettings() {
  activeTab.value = 'settings'
  nextTick(() => settingsPane.value?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
}

function openLabelModal() {
  syncLabelRows()
  labelModalOpen.value = true
}

function closeLabelModal() {
  labelModalOpen.value = false
  syncLabelRows() // 关闭即回滚未保存的编辑
}

const normVer = (v?: string) => (v && !String(v).toLowerCase().startsWith('v') ? 'v' + v : v || '—')

const archOf = (m: any) => {
  return m?.architecture_name || (/(yolov?\d+[a-z]*?)([nslmxbce])?/i.exec(m?.model_name || '') || []).join('')
}
const archLabel = computed(() => {
  const a = archOf(model.value)
  const s = model.value?.model_scale
  return a ? `${a}${s ? '-' + s : ''}` : '未指定'
})
const labelVersionLabel = computed(() => {
  const n = labelsDict.value.labels?.length ?? 0
  const ver = model.value?.version
  return `${ver ? 'v' + String(ver).replace(/^v/i, '') : '—'} (${n}类)`
})
const latestMAP = computed(() => {
  const g = model.value?.gatekeeper
  const m = g?.new_metrics?.mAP50_95 ?? model.value?.training_metrics?.training_history?.final_metrics?.mAP50_95
  return m != null ? fmtPct(m) : ''
})
const bizLabel = computed(() => {
  const v = model.value?.business
  const map: Record<string, string> = {
    general: '通用检测', pedestrian: '行人检测', vehicle: '车辆识别', defect: '高精度质检', package: '包裹分拣'
  }
  return (v && (map[v as string] || v)) || '—'
})

function statusLabel(status?: string): string {
  if (!status) return '未知'
  const map: Record<string, string> = {
    production_ready: '生产就绪', rejected: '守门员拦截', superseded: '已换代',
    training: '训练中', evaluating: '评估中', active: '启用中', inactive: '停用'
  }
  return map[status] || status
}

function fmtTime(t?: string | null): string {
  if (!t) return '—'
  const d = new Date(t)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

function fmtPct(v: number | null | undefined): string {
  if (v == null) return '—'
  return (v >= 0 && v <= 1.5 ? (v * 100).toFixed(1) + '%' : Number(v).toFixed(2))
}

function metricOf(v: any, key: 'mAP50_95' | 'precision' | 'recall') {
  return v?.gatekeeper?.new_metrics?.[key] ??
    v?.training_metrics?.training_history?.final_metrics?.[key] ?? null
}

const metricsDelta = computed(() => {
  const g = model.value?.gatekeeper
  const old = g?.old_metrics?.mAP50_95
  const neu = g?.new_metrics?.mAP50_95
  if (old == null || neu == null) return null
  const d = (neu - old) * 100
  return { deltaText: `${d >= 0 ? '+' : ''}${d.toFixed(1)}%` }
})

// ===== 数据集管理 =====
const dsKeyword = ref('')
const dsFilter = ref('')

type Lifecycle = { key: string; text: string }

function lifecycleOf(d: any): Lifecycle {
  if (!d.model_id) return { key: 'unassigned', text: '未分配' }
  if (d.stage === 'failed') return { key: 'failed', text: '训练失败' }
  if (['sealed', 'training', 'completed'].includes(d.stage)) return { key: 'sealed', text: '已封板' }
  if (d.stage === 'annotating') {
    const hasData = d.annotated || Number(d.label_count || 0) > 0 || Number(d.image_count || 0) > 0
    return hasData && d.status === 'prepared'
      ? { key: 'pending', text: '待封板' }
      : { key: 'annotating', text: '标注中' }
  }
  if (d.annotated) return { key: 'pending', text: '待封板' }
  return { key: 'todo', text: '待标注' }
}

function mainActionOf(d: any): { label: string; to: string } {
  const lc = lifecycleOf(d)
  switch (lc.key) {
    case 'todo': case 'annotating': case 'failed':
      return { label: '去标注', to: `/annotate?dataset_id=${d.dataset_id}` }
    case 'pending': return { label: '封板', to: '' }
    case 'sealed': return { label: '训练', to: `/train?dataset_ids=${d.dataset_id}` }
    case 'unassigned': return { label: '挂载', to: '' }
    default: return { label: '查看', to: '' }
  }
}

const datasetsView = computed<ModelDataset[]>(() => {
  return [...unboundDatasets.value, ...mountedDatasets.value]
})

const filteredDatasets = computed(() => {
  const kw = dsKeyword.value.trim().toLowerCase()
  return datasetsView.value.filter((d) => {
    if (dsFilter.value && lifecycleOf(d).key !== dsFilter.value) return false
    if (!kw) return true
    const hay = `${d.dataset_id} ${d.display_name || ''} ${d.filename || ''}`.toLowerCase()
    return hay.includes(kw)
  })
})

const moreOpen = ref('')
function toggleMore(d: any) {
  moreOpen.value = moreOpen.value === d.dataset_id ? '' : d.dataset_id
}

async function onMainAction(d: any) {
  const lc = lifecycleOf(d)
  if (lc.key === 'pending') await sealAction(d)
  if (lc.key === 'unassigned') await bindCurrent(d)
}

async function sealAction(d: any) {
  const cnt = Number(d.label_count || 0)
  const within = sealMinImages.value > 0 && cnt >= sealMinImages.value
  const msg = `数据集 ${d.dataset_id} 共 ${cnt} 张标注，${within ? '达到封板门槛，确认封板？' : `未达到封板门槛（${sealMinImages.value} 张），确认强制封板？`}`
  if (!confirm(msg)) return
  try {
    await sealDataset(d.dataset_id, !within)
    alert('已封板：数据集将作为该模型的训练数据（只读）。')
    await refresh()
  } catch (e: any) {
    alert('封板失败：' + (e?.response?.data?.detail || e?.message))
  }
}

async function bindCurrent(d: any) {
  if (!confirm(`将数据集 ${d.dataset_id} 挂载到模型库 ${model.value?.display_name || model.value?.name || modelId}？`)) return
  try {
    await bindDatasetModel(d.dataset_id, modelId)
    await refresh()
  } catch (e: any) {
    alert('挂载失败：' + (e?.response?.data?.detail || e?.message))
  }
}

async function revalidate(d: ModelDataset) {
  try {
    const res = await validateDataset(d.dataset_id)
    const ok = res?.status === 'ok'
    if (ok) {
      alert(`校验通过：${res.checked_files} 个标注文件 / ${res.lines_checked} 行标注均符合 YOLO 格式`)
    } else {
      const errs = (res?.errors || []).slice(0, 5).join('\n')
      alert(`校验未通过：${res.invalid_lines} 行异常（${res.invalid_files} 个文件）\n示例：\n${errs || '(无详情)'}`)
    }
    await refresh()
  } catch (e: any) {
    alert('校验失败：' + (e?.response?.data?.detail || e?.message || '未知错误'))
  }
}

async function unbind(d: ModelDataset) {
  if (!confirm(`解绑数据集 ${d.dataset_id}？它将不再归属当前模型库。`)) return
  try {
    await bindDatasetModel(d.dataset_id, null)
    await refresh()
  } catch (e: any) {
    alert('解绑失败：' + (e?.response?.data?.detail || e?.message))
  }
}

async function removeDs(d: ModelDataset) {
  if (!confirm(`删除数据集 ${d.dataset_id}？此操作不可恢复！`)) return
  try {
    await deleteDataset(d.dataset_id)
    await refresh()
  } catch (e: any) {
    alert('删除失败：' + (e?.response?.data?.detail || e?.message))
  }
}

// ===== 模型版本 =====
const versions = computed(() => {
  const list: any[] = []
  if (model.value) list.push(model.value)
  const parent = model.value?.lineage?.parent_model_id
  if (parent) {
    const p = allModels.value.find((m: any) => m.model_id === parent)
    if (p) list.push(p)
  }
  return list
})

// ===== Tab（依赖引用于此处定义之后） =====
const wkTabs = computed(() => [
  { key: 'overview', label: '概览' },
  { key: 'datasets', label: `数据集管理 (${datasetsView.value.length})` },
  { key: 'versions', label: `模型版本 (${versions.value.length})` },
  { key: 'history', label: `训练任务历史 (${historyJobs.value.length})` },
  { key: 'settings', label: '设置' }
])

// ===== 训练任务历史 =====
async function loadJobs() {
  loadingJobs.value = true
  try {
    const res = await listTrainJobs()
    const jobs = res.jobs || []
    const rela = model.value?.lineage?.job_id
    historyJobs.value = jobs.filter((j: any) =>
      j.model_id === modelId || j.base_model_id === modelId || j.job_id === rela
    ).sort((a: any, b: any) => String(b.created_at).localeCompare(String(a.created_at)))
  } catch (e) {
    console.error('加载训练任务失败:', e)
  } finally {
    loadingJobs.value = false
  }
}

function jobStatusText(s: string): string {
  const map: Record<string, string> = {
    running: '训练中', queued: '排队中', completed: '已完成',
    failed: '失败', crashed: '崩溃', stopped: '已停止',
    ready: '就绪', evaluating: '评估中'
  }
  return map[s] || s
}

// ===== 设置：标签字典编辑 =====
const labelRows = ref<any[]>([])
let labelTmpSeq = 0
function addLabelRow() {
  labelRows.value.push({ _tmp: `new_${labelTmpSeq++}`, index: labelRows.value.length, english_code: '', chinese_name: '', chinese_desc: '' })
}
function syncLabelRows() {
  labelRows.value = (labelsDict.value.labels || []).map((l) => ({ ...l }))
}

async function saveLabels() {
  savingLabels.value = true
  labelsMsg.value = ''
  try {
    const rows = labelRows.value.map((r, i) => ({
      index: i,
      english_code: (r.english_code || '').trim(),
      chinese_name: (r.chinese_name || '').trim() || undefined,
      chinese_desc: (r.chinese_desc || '').trim() || undefined
    })).filter((r) => r.english_code)
    await updateLabelsDict(modelId, rows)
    labelsMsg.value = '✓ 标签字典已保存'
    labelModalOpen.value = false
    await refresh()
  } catch (e: any) {
    labelsMsg.value = '保存失败：' + (e?.response?.data?.detail || e?.message)
  } finally {
    savingLabels.value = false
  }
}

const savingLabels = ref(false)
const labelsMsg = ref('')

// ===== 主操作 =====
function goTrain() {
  const ids = mountedDatasets.value.map((d) => d.dataset_id)
  if (ids.length) window.location.href = `/train?dataset_ids=${ids.join(',')}`
  else window.location.href = '/train'
}

async function handleImportModel(e: any) {
  const file = e.target.files?.[0]
  if (!file) return
  try {
    const res = await uploadModel(file)
    alert('导入成功：' + (res?.message || res?.file_name || file.name))
    await refresh()
  } catch (err: any) {
    alert('导入失败：' + (err?.response?.data?.detail || err?.message))
  } finally {
    e.target.value = ''
  }
}

async function downloadWeights() {
  downloading.value = true
  try {
    const blob = await exportModel(modelId)
    const fn = `${model.value?.model_code || modelId}_v${String(model.value?.version || '').replace(/^v/i, '')}.zip`
    saveBlob(blob, fn)
  } catch (e: any) {
    alert('下载失败：' + (e?.response?.data?.detail || e?.message))
  } finally {
    downloading.value = false
  }
}

async function downloadCharts() {
  charting.value = true
  try {
    const blob = await generateTrainingCharts(modelId, 'all')
    window.open(URL.createObjectURL(blob), '_blank')
  } catch (e: any) {
    alert('图表生成失败（可能该模型尚无训练结果）：' + (e?.response?.data?.detail || e?.message))
  } finally {
    charting.value = false
  }
}

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

// ===== 简介 / 名称保存 =====
function saveDescription() {
  savingDesc.value = true
  descSaveMsg.value = ''
  descSaveErr.value = false
  updateModel(modelId, { description: descDraft.value })
    .then(async () => {
      descSaveMsg.value = '✓ 简介已保存'
      await refresh()
      editDescOpen.value = false
    })
    .catch((e: any) => {
      descSaveErr.value = true
      descSaveMsg.value = '保存失败：' + (e?.response?.data?.detail || e?.message)
    })
    .finally(() => { savingDesc.value = false })
}

function saveName() {
  savingDesc.value = true
  descSaveMsg.value = ''
  descSaveErr.value = false
  updateModel(modelId, { display_name: nameDraft.value, name: nameDraft.value })
    .then(async () => {
      descSaveMsg.value = '✓ 名称已保存'
      await refresh()
    })
    .catch((e: any) => {
      descSaveErr.value = true
      descSaveMsg.value = '保存失败：' + (e?.response?.data?.detail || e?.message)
    })
    .finally(() => { savingDesc.value = false })
}

const deleting = ref(false)
async function removeModel() {
  if (!confirm('确定删除该模型库及其全部模型版本与权重文件？此操作不可恢复！')) return
  deleting.value = true
  try {
    await deleteModel(modelId)
    alert('模型库已删除')
    window.location.href = '/'
  } catch (e: any) {
    alert('删除失败：' + (e?.response?.data?.detail || e?.message))
  } finally {
    deleting.value = false
  }
}

// ===== 刷新 =====
async function refresh() {
  loading.value = true
  try {
    const m = await getModel(modelId)
    model.value = m?.model_id ? m : null
    if (!model.value) return
    const dsRes = await getModelDatasets(modelId)
    mountedDatasets.value = dsRes.datasets || []
    const allRes = await listDatasets()
    const all: any[] = allRes.datasets || []
    unboundDatasets.value = all.filter((d: any) => !d.model_id)
    sealMinImages.value = allRes.seal_min_images || 0
    try {
      const ld = await getLabelsDict(modelId)
      labelsDict.value = { labels: ld?.labels || [], model_code: ld?.model_code }
      syncLabelRows()
    } catch { /* 标签字典尚未初始化 */ }
    try {
      const lm = await listModels()
      allModels.value = lm.models || []
    } catch { /* 忽略 */ }
    descDraft.value = m?.description || ''
    nameDraft.value = m?.name || m?.display_name || ''
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  await refresh()
  await loadJobs()
})
</script>

<style scoped>
.model-detail {
  max-width: 1100px;
}

.back-btn {
  background: none;
  border: none;
  color: #2c6ec5;
  cursor: pointer;
  padding: 0;
  font-size: 0.9rem;
  margin-bottom: 12px;
}
.back-btn:hover { text-decoration: underline; }

/* ===== Header（工作台） ===== */
.wk-header { margin-bottom: 12px; }
.wk-header-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }
.wk-title { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.wk-title h1 { margin: 0; color: #1f2d3d; font-size: 1.5rem; }
.model-code { font-size: 0.75rem; color: #7a8aa0; background: #f3f6fa; padding: 2px 8px; border-radius: 6px; }
.status-tag { font-size: 0.75rem; padding: 2px 10px; border-radius: 10px; background: #f1f3f5; color: #6b7c93; }
.status-tag.production_ready, .status-tag.active { background: #e8f5e9; color: #2e7d32; }
.status-tag.rejected { background: #ffebee; color: #c62828; }
.status-tag.evaluating { background: #e3f2fd; color: #1565c0; }
.status-tag.superseded { background: #fff8e1; color: #b26a00; }
.status-tag.training { background: #e8eaf6; color: #3949ab; }

.wk-desc { margin-top: 10px; color: #7f8c8d; font-size: 0.9rem; line-height: 1.6; }
.wk-meta { margin-top: 8px; font-size: 0.85rem; color: #34495e; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.meta-sep { color: #cdd6e0; }
.wk-actions { margin-top: 10px; display: flex; gap: 6px; }
.wk-edit-desc { margin-top: 10px; padding-top: 10px; border-top: 1px dashed #e0e6ee; }
.wk-edit-desc textarea { width: 100%; border: 1px solid #d5dee8; border-radius: 8px; padding: 8px 10px; font-family: inherit; }
.wk-edit-desc-actions { margin-top: 8px; display: flex; gap: 8px; }
.wk-save-msg { margin: 8px 0 0; font-size: 0.8rem; color: #27ae60; }
.wk-save-msg.err { color: #e74c3c; }

/* ===== 主操作栏 ===== */
.wk-ops { display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
.pt-file-input { display: none; }

/* ===== Tab 栏 ===== */
.wk-tabs { display: flex; gap: 4px; border-bottom: 2px solid #eef2f6; margin-bottom: 14px; flex-wrap: wrap; }
.wk-tab {
  padding: 9px 16px;
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
  color: #6b7c93;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 600;
}
.wk-tab:hover { color: #2c6ec5; }
.wk-tab.on { color: #2c6ec5; border-bottom-color: #2c6ec5; }

.wk-pane { padding-bottom: 20px; }
.wk-pane-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
.wk-pane-head h2 { margin: 0; font-size: 1.1rem; color: #2c3e50; }

/* ===== 概览 ===== */
.ov-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
.ov-card { border: 1px solid #e3e9f1; border-radius: 8px; padding: 12px 14px; background: #fbfcfe; }
.ov-label { font-size: 0.75rem; color: #94a3b3; margin-bottom: 6px; font-weight: 600; }
.ov-value { font-size: 0.9rem; color: #2c3e50; line-height: 1.5; }
.ov-tags { display: flex; flex-wrap: wrap; gap: 5px; }
.ov-tag { font-size: 0.72rem; background: #eef4ff; color: #2c6ec5; padding: 2px 8px; border-radius: 10px; }
.mt4 { margin-top: 8px; }

/* ===== 数据集管理 ===== */
.ds-toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }
.ds-search { flex: 1; min-width: 200px; max-width: 360px; padding: 8px 12px; border: 1px solid #d5dee8; border-radius: 8px; font-size: 0.85rem; }
.ds-filter { padding: 8px 10px; border: 1px solid #d5dee8; border-radius: 8px; font-size: 0.85rem; background: #fff; }

.ds-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.ds-table th { text-align: left; padding: 8px 10px; border-bottom: 2px solid #eef2f6; color: #7f8c8d; font-weight: 600; }
.ds-table td { padding: 10px; border-bottom: 1px solid #f0f3f7; vertical-align: middle; }
.ds-name-main { font-weight: 600; color: #1f2d3d; max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* 生命周期状态胶囊（3.3：灰/蓝/橙/绿/白） */
.life-pill { font-size: 0.75rem; padding: 3px 10px; border-radius: 10px; border: 1px solid transparent; white-space: nowrap; display: inline-block; }
.life-todo { background: #eef0f3; color: #6b7c93; }
.life-annotating { background: #e3f2fd; color: #1565c0; }
.life-pending { background: #fff3e0; color: #b26a00; border-color: #ffd9a0; font-weight: 700; }
.life-sealed { background: #e8f5e9; color: #2e7d32; border-color: #c8e6c9; }
.life-unassigned { background: #fff; color: #9aa7b6; border-color: #eaedf1; }
.life-failed { background: #ffebee; color: #c62828; border-color: #ffcdd2; }

.job-running, .job-queued { background: #e3f2fd; color: #1565c0; }
.job-completed, .job-ready { background: #e8f5e9; color: #2e7d32; }
.job-failed, .job-crashed { background: #ffebee; color: #c62828; }
.job-stopped { background: #fff3e0; color: #b26a00; }

.train-mark { font-size: 0.72rem; padding: 2px 8px; border-radius: 10px; white-space: nowrap; }
.train-mark.ok { background: #e0f7fa; color: #00695c; }
.train-mark.todo { background: #f3f6fa; color: #8a94a6; }

.row-actions { display: flex; gap: 6px; align-items: center; position: relative; }
.more-menu { position: relative; }
.more-btn { min-width: 30px; padding: 0 8px; }
.more-dropdown {
  position: absolute; right: 0; top: 30px; z-index: 20;
  background: #fff; border: 1px solid #e3e8f0; border-radius: 8px;
  box-shadow: 0 4px 14px rgba(31, 45, 61, 0.14); min-width: 120px; padding: 4px;
  display: flex; flex-direction: column;
}
.dropdown-item {
  padding: 7px 12px; text-align: left; background: none; border: none;
  font-size: 0.82rem; color: #34495e; cursor: pointer; border-radius: 6px;
}
.dropdown-item:hover { background: #f0f5fb; }
.dropdown-item.danger { color: #e74c3c; }
.dropdown-item.link { text-decoration: none; }

/* 空状态引导 */
.wk-empty .empty-state { text-align: left; }
.wk-steps { margin: 8px 0 0; padding-left: 4px; list-style: none; line-height: 2; color: #5a6b7d; font-size: 0.9rem; }

/* ===== 模型版本 ===== */
.ver-card { border: 1px solid #e3e9f1; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; background: #fff; }
.ver-head { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }
.ver-name { font-size: 1.05rem; font-weight: 700; color: #1f2d3d; }
.ver-latest { font-size: 0.7rem; background: #27ae60; color: #fff; padding: 1px 8px; border-radius: 10px; margin-left: 6px; }
.ver-metrics { display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 8px; }
.ver-metric { display: flex; flex-direction: column; gap: 2px; }
.ver-metric b { font-size: 1.2rem; color: #8e44ad; }
.vm-label { font-size: 0.72rem; color: #94a3b3; }
.ver-meta { display: flex; gap: 14px; flex-wrap: wrap; font-size: 0.78rem; color: #7f8c8d; }

.gk-compare { margin-top: 10px; padding: 10px 12px; background: #fafbfc; border: 1px solid #dde3ea; border-radius: 8px; }
.gk-title { font-size: 0.78rem; font-weight: 700; color: #34495e; margin-bottom: 6px; }
.gk-report-text { font-size: 0.8rem; line-height: 1.6; color: #4a5568; word-break: break-word; }
.gk-regressed { margin-top: 6px; font-size: 0.76rem; color: #c0392b; }
.gk-regressed-item { display: inline-block; margin: 2px 4px 0 0; padding: 1px 8px; background: #fdecea; border: 1px solid #f5b7b1; border-radius: 8px; font-weight: bold; }
.gk-delta { margin-top: 6px; font-size: 0.8rem; color: #1e8449; font-weight: 600; }

/* ===== 设置 ===== */
.set-group { padding: 14px 0; border-bottom: 1px dashed #e3e9f1; }
.set-group:last-child { border-bottom: none; }
.set-label { font-size: 0.88rem; font-weight: 700; color: #2c3e50; margin-bottom: 8px; }
.set-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.cell-input { border: 1px solid #d5dee8; border-radius: 6px; padding: 6px 8px; font-size: 0.82rem; width: 100%; min-width: 90px; }
.label-table input { width: 100%; }
.labels-msg { font-size: 0.8rem; color: #27ae60; }
.static-info { display: flex; flex-direction: column; gap: 6px; font-size: 0.83rem; color: #4a5568; }
.static-info div { display: flex; gap: 12px; align-items: baseline; }
.static-info span:first-child { color: #94a3b3; min-width: 110px; }
.static-info code { background: #f3f6fa; padding: 2px 6px; border-radius: 4px; font-size: 0.78rem; }
.static-info .path { word-break: break-all; }

.form-hint { color: #9aa7b6; font-size: 0.8rem; }
.danger-zone .danger { background: #fff; border: 1px solid #e57373; color: #e53935; border-radius: 8px; padding: 8px 16px; font-size: 0.85rem; cursor: pointer; font-weight: 600; }
.danger-zone .danger:hover { background: #e53935; color: #fff; }
.danger-zone .danger:disabled { opacity: 0.6; cursor: not-allowed; }
.empty-state { text-align: center; color: #7f8c8d; padding: 40px 24px; background: #fff; border: 1px dashed #cfd8e3; border-radius: 12px; }

/* ===== 管理标签弹窗 ===== */
.modal-mask {
  position: fixed; inset: 0; z-index: 100;
  background: rgba(31, 45, 61, 0.45);
  display: flex; align-items: flex-start; justify-content: center;
  padding: 48px 16px; overflow-y: auto;
}
.modal-panel {
  background: #fff; border-radius: 12px; box-shadow: 0 8px 30px rgba(31, 45, 61, 0.2);
  padding: 18px 20px; width: 100%; max-width: 760px;
}
.modal-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.modal-head h3 { margin: 0; font-size: 1.05rem; color: #1f2d3d; }
.modal-actions { margin-top: 12px; }
.modal-panel .label-table input { width: 100%; }
.modal-panel .cell-input { border: 1px solid #d5dee8; border-radius: 6px; padding: 6px 8px; font-size: 0.82rem; min-width: 90px; box-sizing: border-box; }
</style>
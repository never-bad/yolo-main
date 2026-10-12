<template>
  <div class="model-repo">
    <!-- 顶部：标题 + 搜索 + 新建入口 -->
    <div class="repo-toolbar">
      <div class="repo-title">
        <h1>AI 模型仓库</h1>
        <p class="repo-sub">以模型为中心：一个模型一个卡片，采集/上传的数据挂靠到对应模型下，点击卡片查看详情</p>
      </div>
      <div class="toolbar-right">
        <input v-model="searchText" class="search-input" placeholder="搜索算法库..." @input="menuId = null" />
        <button class="primary" @click="createDialogOpen = true">+ 新建算法库</button>
      </div>
    </div>

    <!-- A. 顶部统计栏三格（含待办提醒） -->
    <div class="stats-bar">
      <div class="stat-box">
        <span class="stat-label">总算法库</span>
        <b>{{ filteredModels.length }}</b>
      </div>
      <div class="stat-box">
        <span class="stat-label">总数据集</span>
        <b>{{ totalDatasets }}</b>
      </div>
      <div class="stat-box warn">
        <span class="stat-label">待办事项</span>
        <b>{{ todoPills.length }}</b>
        <div class="todo-pills">
          <router-link v-for="p in todoPills" :key="p" :to="{ path: '/datasets', query: { stage: todoQuery(p) } }" class="todo-pill">{{ p }}</router-link>
        </div>
      </div>
    </div>

    <!-- D. 快捷筛选器：算法 / 业务线 / 状态 -->
    <!-- N-4：archOptions/bizOptions 由 libraryModels 动态推导，库少于 3 个时下拉只剩 1-2 项，
         小白看到的是「筛不出东西」的负价值 → 整栏隐藏，降级为一行说明。filter-row 原样保留。 -->
    <div v-if="libraryModels.length >= 3" class="filter-row">
      <select v-model="filterArch" class="filter-select">
        <option value="">全部算法</option>
        <option v-for="a in archOptions" :key="a" :value="a">{{ a }}</option>
      </select>
      <select v-model="filterBiz" class="filter-select">
        <option value="">全部业务线</option>
        <option v-for="b in bizOptions" :key="b" :value="b">{{ b }}</option>
      </select>
      <select v-model="filterStatus" class="filter-select">
        <option value="">全部状态</option>
        <option value="production_ready">生产就绪</option>
        <option value="active">启用</option>
        <option value="training">训练中</option>
        <option value="evaluating">评估中</option>
        <option value="rejected">守门员拦截</option>
        <option value="superseded">已换代</option>
        <option value="archived">已归档</option>
      </select>
      <span v-if="filterActive" class="filter-reset" @click="resetFilters">清除筛选</span>
    </div>
    <div v-if="libraryModels.length < 3" class="filter-row filter-hint">数据较少，筛选功能在 ≥3 个算法库时启用（当前 {{ libraryModels.length }} 个）</div>

    <!-- 模型对比（勾选 2~3 个后出现） -->
    <div v-if="compareIds.length >= 2" class="compare-bar">
      <span>已选 <b>{{ compareIds.length }}</b> 个算法库（最多 3 个）</span>
      <button class="primary small" @click="compareOpen = true">对比所选模型</button>
      <span class="compare-close" @click="compareIds = []">✕ 取消选择</span>
    </div>

    <div v-if="loading" class="loading-state"><span class="loading-spinner"></span>加载模型仓库...</div>

    <div v-else-if="!libraryModels.length" class="empty-state">
      <h3>欢迎使用 AI 模型管理平台</h3>
      <p>仓库还没有算法库。按下面三步开始你的第一个检测模型：</p>
      <div class="guide-steps">
        <div class="guide-step">
          <span class="step-no">1</span>
          <b>上传数据</b>
          <small>上传图片/标注压缩包，归属到算法库下的数据集</small>
        </div>
        <div class="guide-step">
          <span class="step-no">2</span>
          <b>标注图片</b>
          <small>在标注工作台画框、走 AI 预标注流程</small>
        </div>
        <div class="guide-step">
          <span class="step-no">3</span>
          <b>训练模型</b>
          <small>选择数据集与超参，训练并经守门员评估上线</small>
        </div>
      </div>
      <div class="empty-actions">
        <button class="primary" @click="createDialogOpen = true">+ 新建算法库</button>
        <router-link to="/models"><button class="secondary">去模型管理</button></router-link>
      </div>
    </div>

    <div v-else class="model-grid">
      <div
        v-for="m in filteredModels"
        :key="m.model_id"
        class="model-card"
        :class="{ comparing: isComparing(m.model_id) }"
        @click="gotoDetail(m)"
      >
        <label class="compare-check" @click.stop>
          <input
            type="checkbox"
            :checked="isComparing(m.model_id)"
            @change="toggleCompare(m.model_id)"
          />
          <span class="compare-hint">对比</span>
        </label>

        <div class="mc-head">
          <div class="mc-title">
            <span class="mc-icon">🤖</span>
            <span class="mc-name">{{ m.display_name || m.name || m.model_id }}</span>
            <code class="mc-code">{{ m.model_code || m.model_id }}</code>
          </div>
          <div class="more-wrap" @mouseleave="menuId = null">
            <button class="more-btn" @click.stop="menuId = menuId === m.model_id ? null : m.model_id">⋯</button>
            <div v-if="menuId === m.model_id" class="more-menu" @click.stop>
              <button class="menu-item" @click="openEditDialog(m)">编辑信息</button>
              <button class="menu-item" @click="exportModelItem(m)" :disabled="exporting">
                {{ exporting ? '导出中...' : '导出' }}
              </button>
              <button class="menu-item menu-danger" @click="deleteModelItem(m)" :disabled="deleting">
                {{ deleting ? '删除中...' : '删除' }}
              </button>
            </div>
          </div>
        </div>

        <div v-if="m.description || m.business" class="mc-scene" :title="fullScene(m)">
          <span class="scene-key">场景</span> {{ sceneShort(m) }}
        </div>

        <div class="mc-rows">
          <div class="mc-row">
            挂载 <b>{{ m.dataset_count || 0 }}</b> 个数据集
            <span v-if="pendingCount(m)" class="row-pending">（{{ pendingCount(m) }} 个待处理）</span>
          </div>
          <div class="mc-row">
            标签 <b>{{ m.classes?.length || 0 }}</b> 类
            <span class="row-meta" v-if="m.classes?.length">· {{ classPreview(m) }}</span>
          </div>
          <div class="mc-row">
            最新模型 <b v-if="latestModelText(m)" class="row-model">{{ latestModelText(m) }}</b>
            <span v-else class="dim">暂无模型产出</span>
          </div>
        </div>

        <div class="mc-foot">
          <span class="status-tag" :class="m.status">{{ statusLabel(m.status) }}</span>
          <span v-if="m.version" class="version-badge" title="最新模型版本">最新 {{ normVer(m.version) }}</span>
          <span class="enter">进入详情 →</span>
        </div>
      </div>
    </div>

    <!-- 新建算法库弹窗（组件与数据集导入页共用同一套逻辑） -->
    <LibraryCreateModal v-model:open="createDialogOpen" @created="onLibraryCreated" />

    <!-- 编辑信息弹窗 -->
    <div v-if="editDialogOpen" class="modal-mask" @click.self="editDialogOpen = false">
      <div class="modal edit-modal">
        <h3>编辑算法库信息</h3>
        <div class="form-field">
          <label>算法库名称</label>
          <input v-model="editForm.display_name" />
        </div>
        <div class="form-field">
          <label>业务场景描述</label>
          <textarea v-model="editForm.description" rows="2"></textarea>
        </div>
        <div v-if="editError" class="form-error">{{ editError }}</div>
        <div class="modal-actions">
          <button class="secondary" @click="editDialogOpen = false">取消</button>
          <button class="primary" @click="submitEdit" :disabled="editing">
            {{ editing ? '保存中...' : '保存' }}
          </button>
        </div>
      </div>
    </div>

    <!-- 模型对比弹窗 -->
    <div v-if="compareOpen" class="modal-mask" @click.self="compareOpen = false">
      <div class="modal compare-modal">
        <h3>算法库对比（{{ compareModels.length }} 个）</h3>
        <table class="compare-table">
          <thead>
            <tr>
              <th>算法库</th>
              <th>mAP50-95</th>
              <th>参数量</th>
              <th>推理速度</th>
              <th>标签类数</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in compareModels" :key="m.model_id">
              <td>
                <b>{{ m.display_name || m.name || m.model_id }}</b>
                <small class="tb-code">{{ m.model_code || m.model_id }}</small>
              </td>
              <td :class="{ best: isBest(m, 'map') }">{{ mapText(m) }}</td>
              <td>{{ paramsText(m) }}</td>
              <td :class="{ best: isBest(m, 'speed') }">{{ speedText(m) }}</td>
              <td>{{ m.classes?.length || 0 }}</td>
            </tr>
          </tbody>
        </table>
        <p class="modal-sub">绿色高亮为同列最优；参数量来自模型产出元数据，无产出模型显示 —</p>
        <div class="modal-actions">
          <button class="secondary" @click="compareOpen = false">关闭</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onActivated } from 'vue'
import { useRouter } from 'vue-router'
import { listModels, updateModel, exportModel } from '@/api/models'
import { deleteModelWithGuard, showConfirm } from '@/composables/useDialog'
import { listDatasets } from '@/api/datasets'
import { showToast } from '@/utils/toast'
import LibraryCreateModal from '@/components/LibraryCreateModal.vue'

const router = useRouter()
const loading = ref(true)
const models = ref<any[]>([])
const totalDatasets = ref(0)
const datasets = ref<any[]>([])

// ---- 搜索与筛选 ----
const searchText = ref('')
const filterArch = ref('')
const filterBiz = ref('')
const filterStatus = ref('')
const filterActive = computed(() => searchText.value || filterArch.value || filterBiz.value || filterStatus.value)

const BIZ_OPTIONS: Record<string, string> = {
  general: '通用目标检测',
  pedestrian: '行人检测',
  vehicle: '车辆 / 车牌',
  defect: '工业缺陷检测',
  package: '包裹 / 物流'
}
function bizLabel(b: string): string {
  return BIZ_OPTIONS[b] || b
}

function modelArch(m: any): string {
  if (m?.architecture_name) return m.model_scale ? `${m.architecture_name}${m.model_scale}` : m.architecture_name
  const src = m?.lineage?.base_model || m?.base_model || m?.name || ''
  const mm = /(yolov?\d+[a-z]*?)([nslmxbce])\.pt/i.exec(src)
  return mm ? mm[0].replace(/\.pt$/i, '') : ''
}

// ---- 首页只列「算法库」这一层 ----
// 后端 listModels 返回的是模型仓库里全部记录（库根 + 库内训练版本 + 上传的模型），
// 不加这层判据就会把一条模型记录画成一张算法库卡片，顶部「总算法库」也会虚高。
// 判据：empty=新建的库根；没有任何库归属的（历史）记录按独立模型保留可见；
// 已归库的上传模型与库内训练产出版本不进首页，去库详情/模型管理看。
function isLibraryCard(m: any): boolean {
  if (!m) return false
  if (m.empty === true) return true
  if (m.model_library_id) return false
  const root = m.library_root_id
  return !(root && root !== m.model_id)
}

const libraryModels = computed(() => models.value.filter(isLibraryCard))

const archOptions = computed(() => Array.from(new Set(libraryModels.value.map(modelArch).filter(Boolean))))
const bizOptions = computed(() => Array.from(new Set(libraryModels.value.map((m) => bizLabel(m.business || '')).filter(Boolean))))

const filteredModels = computed(() =>
  libraryModels.value.filter((m) => {
    const kw = searchText.value.trim().toLowerCase()
    if (kw) {
      const hay = `${m.display_name || ''} ${m.name || ''} ${m.model_code || ''} ${m.model_id || ''}`.toLowerCase()
      if (!hay.includes(kw)) return false
    }
    if (filterArch.value && modelArch(m) !== filterArch.value) return false
    if (filterBiz.value && bizLabel(m.business || '') !== filterBiz.value) return false
    if (filterStatus.value && (m.status || '') !== filterStatus.value) return false
    return true
  })
)

function resetFilters() {
  searchText.value = ''
  filterArch.value = ''
  filterBiz.value = ''
  filterStatus.value = ''
}

// ---- 待办事项 ----
const todoPills = computed(() => {
  const pills: string[] = []
  const sealable = datasets.value.filter((d: any) => (d.stage || '').toLowerCase() === 'sealed').length
  const failed = datasets.value.filter((d: any) => (d.stage || '').toLowerCase() === 'failed').length
  if (sealable) pills.push(`${sealable} 个数据集已封板`)
  if (failed) pills.push(`${failed} 个数据集训练失败`)
  return pills
})
const todoQuery = (p: string) => (p.includes('已封板') ? 'sealed' : 'failed')

// ---- 卡片信息 ----
function fullScene(m: any): string {
  return [m.description, m.business ? `业务线：${bizLabel(m.business)}` : ''].filter(Boolean).join(' | ')
}
function sceneShort(m: any): string {
  const s = fullScene(m).replace(/\s+/g, ' ').trim()
  return s.length > 20 ? s.slice(0, 20) + '…' : s
}
function pendingCount(m: any): number {
  const st = m.dataset_stats || {}
  return (st.annotating || 0) + (st.sealed || 0)
}
function classPreview(m: any): string {
  return (m.classes || []).slice(0, 5).join(', ')
}
function latestModelText(m: any): string {
  const v = m.gatekeeper?.new_metrics?.mAP50_95
  const base = normVer(m.version) || '—'
  return v != null ? `${base} · mAP ${(v * 100).toFixed(0)}%` : base
}
function normVer(v?: string): string {
  return v ? (String(v).startsWith('v') ? v : 'v' + v) : ''
}
function statusLabel(status?: string): string {
  if (!status) return '未知'
  const map: Record<string, string> = {
    production_ready: '✅ 生产就绪',
    production: '🏭 生产中',
    active: '已发布',
    archived: '已归档',
    rejected: '❌ 已拦截',
    superseded: '⚠️ 已换代',
    training: '⏳ 训练中',
    evaluating: '🔍 评估中',
    inactive: '停用'
  }
  return map[status] || status
}

// ---- 卡片操作（⋯ 菜单） ----
const menuId = ref<string | null>(null)
const exporting = ref(false)
const deleting = ref(false)

function gotoDetail(m: any) {
  router.push(`/models/${m.model_id}`)
}

const editDialogOpen = ref(false)
const editForm = ref({ model_id: '', display_name: '', description: '' })
const editError = ref('')
const editing = ref(false)
function openEditDialog(m: any) {
  editForm.value = { model_id: m.model_id, display_name: m.display_name || m.name || '', description: m.description || '' }
  editError.value = ''
  editDialogOpen.value = true
  menuId.value = null
}
async function submitEdit() {
  if (!editForm.value.display_name.trim()) {
    editError.value = '名称不能为空'
    return
  }
  editing.value = true
  try {
    await updateModel(editForm.value.model_id, {
      display_name: editForm.value.display_name.trim(),
      description: editForm.value.description,
      name: editForm.value.display_name.trim()
    })
    editDialogOpen.value = false
    await load()
  } catch (e: any) {
    editError.value = e?.response?.data?.detail || '保存失败'
  } finally {
    editing.value = false
  }
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
async function exportModelItem(m: any) {
  exporting.value = true
  try {
    const blob = await exportModel(m.model_id)
    downloadBlob(blob, `${m.model_code || m.model_id}.zip`)
  } catch (e: any) {
    if (!e?.__toastShown) showToast(e?.response?.data?.detail || '导出失败', 'error', 4000)
  } finally {
    exporting.value = false
    menuId.value = null
  }
}
async function deleteModelItem(m: any) {
  const name = m.display_name || m.name || m.model_id
  if (!(await showConfirm(`确认删除算法库「${name}」？该操作不可恢复。`))) return
  deleting.value = true
  try {
    // P2-5：挂接非空时后端不删，守卫流程弹清单让用户选「转移归属 / 保留未归属」
    if (await deleteModelWithGuard(m.model_id, name) !== 'deleted') return
    menuId.value = null
    await load()
  } catch (e: any) {
    if (!e?.__toastShown) showToast(e?.response?.data?.detail || '删除失败', 'error', 4000)
  } finally {
    deleting.value = false
  }
}

// ---- 新建算法库（表单逻辑在 LibraryCreateModal 组件内，导入页复用同一组件）----
const createDialogOpen = ref(false)
async function onLibraryCreated(meta: any) {
  await load()
  if (meta?.model_id) router.push(`/models/${meta.model_id}`)
}

// ---- 模型对比 ----
const compareIds = ref<string[]>([])
const compareOpen = ref(false)
function isComparing(id: string) {
  return compareIds.value.includes(id)
}
function toggleCompare(id: string) {
  const idx = compareIds.value.indexOf(id)
  if (idx >= 0) {
    compareIds.value.splice(idx, 1)
  } else if (compareIds.value.length < 3) {
    compareIds.value.push(id)
  } else {
    showToast('最多同时对比 3 个算法库', 'info', 4000)
  }
}
const compareModels = computed(() => compareIds.value.map((id) => models.value.find((m) => m.model_id === id)).filter(Boolean))
function mapText(m: any): string {
  const v = m?.gatekeeper?.new_metrics?.mAP50_95
  return v == null ? '—' : (v * 100).toFixed(1) + '%'
}
function paramsText(m: any): string {
  const p = m?.model_info?.total_params_m ?? m?.model_info?.total_params
  if (p == null) return '—'
  return m?.model_info?.total_params_m != null ? `${p}M` : String(p)
}
function speedText(m: any): string {
  const s = m?.gatekeeper?.new_metrics?.speed_ms
  return s == null ? '—' : s + 'ms'
}
function isBest(m: any, key: 'map' | 'speed') {
  const list = compareModels.value
  const val = key === 'map' ? m?.gatekeeper?.new_metrics?.mAP50_95 : m?.gatekeeper?.new_metrics?.speed_ms
  if (val == null || !list.length) return false
  if (key === 'map') return val === Math.max(...list.map((x) => x?.gatekeeper?.new_metrics?.mAP50_95 ?? -1))
  return val === Math.min(...list.map((x) => x?.gatekeeper?.new_metrics?.speed_ms ?? Infinity))
}

// ---- 数据加载 ----
async function load() {
  try {
    const modelRes = await listModels()
    models.value = modelRes.models || []
    const dsRes = await listDatasets()
    const all = dsRes.datasets || []
    datasets.value = all
    totalDatasets.value = all.length
  } finally {
    loading.value = false
  }
}
onMounted(load)
// B-10：keep-alive 缓存后，切回本页时静默刷新（不清空已有数据，不闪骨架屏）。
// 首次挂载 activated 也会触发，用 flag 跳过，避免与 onMounted 重复请求。
let _dashFirstActivate = true
onActivated(() => {
  if (_dashFirstActivate) { _dashFirstActivate = false; return }
  load()
})
</script>

<style scoped>
.model-repo {
  max-width: 1200px;
}

.repo-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 14px;
}

.repo-toolbar h1 {
  color: #2c3e50;
  margin-bottom: 6px;
}

.repo-sub {
  color: #7f8c8d;
  margin: 0;
}

.toolbar-right {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.search-input {
  width: 240px;
  padding: 8px 12px;
  border: 1px solid #dce5ee;
  border-radius: 8px;
  font-size: 0.9rem;
  outline: none;
  transition: border-color 0.2s;
}

.search-input:focus {
  border-color: #2c6ec5;
}

.stats-bar {
  display: flex;
  gap: 12px;
  margin-bottom: 14px;
  flex-wrap: wrap;
}

.stat-box {
  flex: 1;
  min-width: 180px;
  background: #f7fafc;
  border: 1px solid #e4e9f0;
  border-radius: 10px;
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.stat-box b {
  font-size: 1.6rem;
  color: #1f2d3d;
}

.stat-box.warn {
  background: #fff9f0;
  border-color: #ffe3bd;
}

.stat-box.warn b {
  color: #b26a00;
}

.stat-label {
  color: #7a8aa0;
  font-size: 0.8rem;
}

.todo-pills {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-top: 4px;
}

.todo-pills .todo-pill {
  text-decoration: none;
  background: #ffedd5;
  color: #b26a00;
  font-size: 0.75rem;
  padding: 2px 8px;
  border-radius: 8px;
}

.todo-pills .todo-pill:hover { background: #fde3c0; text-decoration: underline; }

.filter-row {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 14px;
  flex-wrap: wrap;
}

.filter-hint { color: #6b7280; font-size: 0.85rem; padding-top: 0.5rem; }

.filter-select {
  padding: 6px 10px;
  border: 1px solid #dce5ee;
  border-radius: 8px;
  font-size: 0.85rem;
  background: #fff;
  color: #33475b;
}

.filter-reset {
  color: #2c6ec5;
  font-size: 0.8rem;
  cursor: pointer;
}

.filter-reset:hover {
  text-decoration: underline;
}

.compare-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  background: #eef4ff;
  border: 1px solid #cfe0ff;
  border-radius: 10px;
  padding: 10px 14px;
  margin-bottom: 14px;
  font-size: 0.9rem;
  color: #1f3a5f;
}

.compare-close {
  margin-left: auto;
  color: #7a8aa0;
  cursor: pointer;
}

.compare-close:hover {
  color: #c62828;
}

.primary {
  background: #2c6ec5;
  color: #fff;
  border: 1px solid #2c6ec5;
  border-radius: 8px;
  padding: 8px 16px;
  cursor: pointer;
  font-size: 0.9rem;
}

.primary:hover {
  background: #2459a5;
}

.primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.primary.small {
  padding: 5px 12px;
  font-size: 0.82rem;
}

.secondary {
  background: #fff;
  color: #2c6ec5;
  border: 1px solid #b8cbe4;
  border-radius: 8px;
  padding: 8px 16px;
  cursor: pointer;
  font-size: 0.9rem;
}

.secondary:hover {
  background: #f2f7ff;
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
}

.model-card {
  position: relative;
  background: #fff;
  border: 1px solid #e4e9f0;
  border-radius: 12px;
  padding: 16px;
  cursor: pointer;
  transition: box-shadow 0.2s, transform 0.15s, border-color 0.15s;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.model-card:hover {
  box-shadow: 0 6px 18px rgba(30, 60, 110, 0.12);
  transform: translateY(-2px);
}

.model-card.comparing {
  border-color: #2c6ec5;
  box-shadow: 0 0 0 2px rgba(44, 110, 197, 0.15);
}

.compare-check {
  position: absolute;
  top: 10px;
  right: 10px;
  display: flex;
  align-items: center;
  gap: 4px;
  z-index: 2;
}

.compare-hint {
  font-size: 0.7rem;
  color: #7a8aa0;
}

.mc-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 8px;
  padding-right: 64px;
}

.mc-title {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  min-width: 0;
}

.mc-icon {
  font-size: 1.2rem;
}

.mc-name {
  font-size: 1.12rem;
  font-weight: 700;
  color: #1f2d3d;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 220px;
}

.mc-code {
  font-size: 0.72rem;
  color: #7a8aa0;
  background: #f3f6fa;
  padding: 2px 8px;
  border-radius: 6px;
}

.more-wrap {
  position: relative;
  z-index: 3;
}

.more-btn {
  border: none;
  background: #f3f6fa;
  color: #556;
  width: 28px;
  height: 28px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1.05rem;
  line-height: 1;
}

.more-btn:hover {
  background: #e4ebf3;
}

.more-menu {
  position: absolute;
  top: 32px;
  right: 0;
  background: #fff;
  border: 1px solid #e4e9f0;
  border-radius: 10px;
  box-shadow: 0 8px 22px rgba(30, 60, 110, 0.14);
  min-width: 130px;
  overflow: hidden;
  z-index: 20;
}

.menu-item {
  display: block;
  width: 100%;
  text-align: left;
  border: none;
  background: none;
  padding: 9px 14px;
  font-size: 0.85rem;
  color: #33475b;
  cursor: pointer;
}

.menu-item:hover {
  background: #f2f7ff;
}

.menu-item:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.menu-item.menu-danger {
  color: #c62828;
}

.menu-item.menu-danger:hover {
  background: #ffebee;
}

.mc-scene {
  font-size: 0.82rem;
  color: #556;
  background: #f8fafc;
  border-radius: 6px;
  padding: 6px 10px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.scene-key {
  color: #9e7362;
  font-weight: 600;
}

.mc-rows {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.mc-row {
  font-size: 0.86rem;
  color: #445;
}

.mc-row b {
  color: #2c6ec5;
}

.row-pending {
  color: #b26a00;
  font-size: 0.8rem;
}

.row-meta {
  color: #7a8aa0;
  font-size: 0.78rem;
}

.row-model {
  color: #2e7d32;
}

.dim {
  color: #aab5c2;
  font-size: 0.8rem;
}

.mc-foot {
  display: flex;
  align-items: center;
  gap: 8px;
  border-top: 1px solid #eef2f6;
  padding-top: 10px;
  flex-wrap: wrap;
}

.status-tag {
  font-size: 0.75rem;
  padding: 2px 10px;
  border-radius: 10px;
  background: #f1f3f5;
  color: #6b7c93;
}

.status-tag.production_ready,
.status-tag.production,
.status-tag.active { background: #e8f5e9; color: #2e7d32; }
.status-tag.rejected { background: #ffebee; color: #c62828; }
.status-tag.superseded { background: #eceff1; color: #78909c; }
.status-tag.training { background: #ede7f6; color: #5e35b1; }
.status-tag.evaluating { background: #e8eaf6; color: #3949ab; }

.version-badge {
  font-size: 0.72rem;
  color: #2c6ec5;
  background: #eef4ff;
  padding: 2px 8px;
  border-radius: 10px;
  border: 1px solid #cfe0ff;
}

.enter {
  margin-left: auto;
  color: #2c6ec5;
  font-size: 0.82rem;
  font-weight: 600;
}

/* 空状态引导 */
.empty-state {
  text-align: center;
  color: #7f8c8d;
  padding: 40px 20px;
  background: #fff;
  border: 1px dashed #cfd8e3;
  border-radius: 12px;
}

.empty-state h3 {
  color: #2c3e50;
  margin-bottom: 6px;
}

.guide-steps {
  display: flex;
  gap: 16px;
  justify-content: center;
  margin: 24px 0;
  flex-wrap: wrap;
}

.guide-step {
  width: 200px;
  background: #f7fafc;
  border: 1px solid #e4e9f0;
  border-radius: 10px;
  padding: 16px;
  text-align: left;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.step-no {
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: #2c6ec5;
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 0.85rem;
}

.guide-step b {
  color: #1f2d3d;
}

.guide-step small {
  color: #7a8aa0;
  line-height: 1.5;
}

.empty-actions {
  display: flex;
  gap: 10px;
  justify-content: center;
  margin-top: 12px;
}

/* 弹窗 */
.modal-mask {
  position: fixed;
  inset: 0;
  background: rgba(20, 30, 45, 0.35);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.modal {
  background: #fff;
  border-radius: 14px;
  padding: 22px;
  width: 480px;
  max-width: 90vw;
  max-height: 88vh;
  overflow: auto;
  box-shadow: 0 18px 48px rgba(20, 40, 70, 0.28);
}

.modal h3 {
  color: #1f2d3d;
  margin: 0 0 4px;
}

.modal-sub {
  color: #7a8aa0;
  font-size: 0.82rem;
  margin: 0 0 14px;
}

.form-field {
  margin-bottom: 12px;
  text-align: left;
}

.form-field label {
  display: block;
  font-size: 0.85rem;
  color: #445;
  margin-bottom: 5px;
}

.form-field label i {
  color: #c62828;
  font-style: normal;
}

.form-field input,
.form-field select,
.form-field textarea {
  width: 100%;
  padding: 8px 10px;
  border: 1px solid #d5dee8;
  border-radius: 8px;
  font-size: 0.88rem;
  box-sizing: border-box;
  outline: none;
  font-family: inherit;
}

.form-field input:focus,
.form-field select:focus,
.form-field textarea:focus {
  border-color: #2c6ec5;
}

.form-error {
  color: #c62828;
  font-size: 0.82rem;
  margin-bottom: 10px;
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 8px;
}

.compare-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}

.compare-table th,
.compare-table td {
  border: 1px solid #e9eef4;
  padding: 9px 12px;
  text-align: left;
}

.compare-table th {
  background: #f7fafc;
  color: #445;
  font-weight: 600;
}

.compare-table td.best {
  color: #2e7d32;
  font-weight: 700;
}

.tb-code {
  display: block;
  color: #7a8aa0;
  font-size: 0.72rem;
  margin-top: 2px;
}

.loading-state {
  text-align: center;
  color: #7f8c8d;
  padding: 40px;
}

.loading-spinner {
  display: inline-block;
  width: 18px;
  height: 18px;
  border: 2px solid #cfe0ff;
  border-top-color: #2c6ec5;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-right: 8px;
  vertical-align: middle;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
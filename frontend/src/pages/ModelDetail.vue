<template>
  <div class="model-detail">
    <button class="back-btn" @click="$router.push('/')">← 返回模型仓库</button>

    <div v-if="loading" class="loading-state"><span class="loading-spinner"></span>加载模型详情...</div>

    <div v-else-if="!model" class="empty-state">模型不存在或已被删除</div>

    <template v-else>
      <!-- 模型信息头 -->
      <div class="card model-head">
        <div class="head-main">
          <h1>{{ model.display_name || model.name || model.model_code || model.model_id }}</h1>
          <code class="model-code">{{ model.model_code || model.model_id }}</code>
          <span class="status-tag" :class="model.status">{{ statusLabel(model.status) }}</span>
        </div>
        <div class="head-meta">
          <span v-if="model.version">生产版本：v{{ model.version }}</span>
          <span v-if="model.business">业务：{{ model.business }}</span>
          <span>{{ model.dataset_count || 0 }} 个数据集挂载</span>
        </div>
      </div>

      <!-- 待封板分区（#5 采集库分区：已标注待封板的优先处理） -->
      <div v-if="pendingSeal.length" class="card section">
        <div class="section-head">
          <h2>待封板（{{ pendingSeal.length }}）</h2>
          <router-link :to="'/datasets'"><button class="primary small">去数据集页处理</button></router-link>
        </div>
        <p class="form-hint">已标注待封板：数量达到封板数量门槛（{{ sealMinImages }} 张）后可直接封板，数量不足可走强制封板兜底。封板后数据集只读并进入训练队列。</p>
        <table class="ds-table">
          <thead>
            <tr>
              <th>数据集</th>
              <th>图片 / 门槛</th>
              <th>标注数</th>
              <th>状态</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in pendingSeal" :key="d.dataset_id">
              <td>
                <div class="ds-id">{{ d.dataset_id }}</div>
                <div class="ds-name">{{ d.filename || d.name || '' }}</div>
              </td>
              <td>
                {{ d.image_count ?? '—' }}<span v-if="sealMinImages > 0" style="color:#b26a00"> / {{ sealMinImages }}</span>
              </td>
              <td>{{ d.label_count ?? '—' }}</td>
              <td><span class="stage-pill stage-annotating">待封板</span></td>
              <td>
                <div class="row-actions">
                  <router-link :to="'/annotate?dataset_id=' + d.dataset_id"><button class="ghost small">继续标注</button></router-link>
                  <router-link :to="'/datasets'"><button class="ghost small">封板</button></router-link>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- 挂载数据集 -->
      <div class="card section">
        <div class="section-head">
          <h2>挂载的数据集（{{ datasets.length }}）</h2>
          <div class="section-actions">
            <router-link :to="`/train?dataset_ids=${datasets.map(d => d.dataset_id).join(',')}`">
              <button class="primary small">用这些数据集去训练</button>
            </router-link>
          </div>
        </div>

        <div v-if="!datasets.length" class="inline-empty">
          该模型还没有数据集。可以在下方把「未归属数据集」挂进来，或直接去数据集页上传。
        </div>

        <table class="ds-table" v-else>
          <thead>
            <tr>
              <th>数据集</th>
              <th>阶段</th>
              <th>训练标记</th>
              <th>标注</th>
              <th>标注框数</th>
              <th>入料校验</th>
              <th>封板/训练时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in datasets" :key="d.dataset_id">
              <td>
                <div class="ds-id">{{ d.dataset_id }}</div>
                <div class="ds-name">{{ d.filename || d.name || '' }}</div>
              </td>
              <td><span class="stage-pill" :class="`stage-${d.stage}`">{{ stageLabel(d.stage) }}</span></td>
              <td>
                <span class="train-mark" :class="d.training_status === 'completed' ? 'ok' : 'todo'">
                  {{ d.training_status === 'completed' ? '已完成训练' : '未完成训练' }}
                </span>
              </td>
              <td>{{ d.annotated ? '✓ 已导出' : '—' }}</td>
              <td>{{ d.label_count ?? '—' }}</td>
              <td>
                <div v-if="d.label_validation" class="validation" :class="d.label_validation.status === 'ok' ? 'ok' : 'failed'">
                  <template v-if="d.label_validation.status === 'ok'">
                    ✓ 通过（{{ d.label_validation.lines_checked ?? 0 }} 行）
                  </template>
                  <template v-else>
                    ✗ {{ d.label_validation.invalid_lines ?? 0 }} 行异常 / {{ d.label_validation.invalid_files ?? 0 }} 文件
                  </template>
                </div>
                <div v-else class="validation none">—</div>
              </td>
              <td>
                <div v-if="d.sealed_at" class="time-line">封板 {{ shortTime(d.sealed_at) }}</div>
                <div v-if="d.trained_at" class="time-line">训练 {{ shortTime(d.trained_at) }}</div>
                <span v-if="!d.sealed_at && !d.trained_at">—</span>
              </td>
              <td>
                <div class="row-actions">
                  <router-link :to="`/annotate?dataset_id=${d.dataset_id}`">
                    <button class="ghost small">标注</button>
                  </router-link>
                  <button class="ghost small" @click="revalidate(d)">校验</button>
                  <button class="ghost small" @click="unbind(d)">解绑</button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- 绑定未归属数据集 -->
      <div class="card section">
        <div class="section-head"><h2>挂载未归属数据集</h2></div>
        <div class="bind-row">
          <select v-model="bindTarget" class="bind-select">
            <option value="">选择未归属的数据集...</option>
            <option v-for="u in unboundDatasets" :key="u.dataset_id" :value="u.dataset_id">
              {{ u.dataset_id }}（{{ stageLabel(u.stage) }}）
            </option>
          </select>
          <button class="primary small" :disabled="!bindTarget || binding" @click="bind">
            {{ binding ? '挂载中...' : '挂载到本模型' }}
          </button>
        </div>
        <p class="form-hint" v-if="unboundDatasets.length">从「数据集」上传的旧数据尚未归属任何模型，可在此挂入。</p>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { getModel, getModelDatasets, type ModelDataset } from '@/api/models'
import { listDatasets, bindDatasetModel, validateDataset } from '@/api/datasets'

const route = useRoute()
const modelId = String(route.params.modelId)

const loading = ref(true)
const model = ref<any>(null)
const datasets = ref<ModelDataset[]>([])
const unboundDatasets = ref<any[]>([])
const bindTarget = ref('')
const binding = ref(false)
const sealMinImages = ref(0)  // #6 封板数量门槛（来自后端 settings，前端只读展示）
const pendingSeal = computed(() => datasets.value.filter((d: any) =>
  d.stage === 'annotating' && d.status === 'prepared'
  && (d.annotated || Number(d.label_count || 0) > 0 || Number(d.image_count || 0) > 0)
))

const STAGE_LABEL: Record<string, string> = {
  collecting: '采集中',
  annotating: '标注中',
  sealed: '已封板',
  training: '训练中',
  completed: '已完成训练',
  failed: '训练失败'
}

function stageLabel(s?: string) {
  return STAGE_LABEL[s || 'collecting'] || s
}

function statusLabel(status?: string): string {
  if (!status) return '未知'
  const map: Record<string, string> = {
    production_ready: '生产就绪',
    rejected: '守门员拦截',
    superseded: '已换代',
    training: '训练中',
    evaluating: '评估中',
    active: '启用',
    inactive: '停用'
  }
  return map[status] || status
}

function shortTime(t?: string | null): string {
  if (!t) return ''
  const d = new Date(t)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getMonth() + 1}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

async function refresh() {
  loading.value = true
  try {
    const m = await getModel(modelId)
    model.value = m?.model_id ? m : null
    if (!model.value) return
    const dsRes = await getModelDatasets(modelId)
    datasets.value = dsRes.datasets || []
    const allRes = await listDatasets()
    const all: any[] = allRes.datasets || []
    unboundDatasets.value = all.filter((d: any) => !d.model_id)
    sealMinImages.value = allRes.seal_min_images || 0
  } finally {
    loading.value = false
  }
}

async function bind() {
  if (!bindTarget.value) return
  binding.value = true
  try {
    await bindDatasetModel(bindTarget.value, modelId)
    bindTarget.value = ''
    await refresh()
  } finally {
    binding.value = false
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
  if (!confirm(`解绑数据集 ${d.dataset_id}？它不再归属 ${model.value?.display_name || model.value?.name || modelId}`)) return
  await bindDatasetModel(d.dataset_id, null)
  await refresh()
}

onMounted(refresh)
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

.model-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 16px;
}

.head-main {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.head-main h1 { margin: 0; color: #1f2d3d; font-size: 1.5rem; }

.model-code {
  font-size: 0.75rem;
  color: #7a8aa0;
  background: #f3f6fa;
  padding: 2px 8px;
  border-radius: 6px;
}

.status-tag {
  font-size: 0.75rem;
  padding: 2px 10px;
  border-radius: 10px;
  background: #f1f3f5;
  color: #6b7c93;
}
.status-tag.production_ready, .status-tag.active { background: #e8f5e9; color: #2e7d32; }
.status-tag.rejected { background: #ffebee; color: #c62828; }

.head-meta {
  display: flex;
  gap: 14px;
  color: #7f8c8d;
  font-size: 0.85rem;
}

.section { margin-bottom: 16px; }

.section-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.section-head h2 { margin: 0; font-size: 1.1rem; color: #2c3e50; }

.ds-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

.ds-table th {
  text-align: left;
  padding: 8px 10px;
  border-bottom: 2px solid #eef2f6;
  color: #7f8c8d;
  font-weight: 600;
}

.ds-table td {
  padding: 10px;
  border-bottom: 1px solid #f0f3f7;
  vertical-align: middle;
}

.ds-id { font-weight: 600; color: #1f2d3d; }
.ds-name { color: #7f8c8d; font-size: 0.78rem; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.stage-pill {
  font-size: 0.75rem;
  padding: 3px 8px;
  border-radius: 10px;
  border: 1px solid transparent;
  white-space: nowrap;
}
.stage-collecting { background: #eef4ff; color: #2c6ec5; border-color: #cfe0ff; }
.stage-annotating { background: #fff3e0; color: #b26a00; border-color: #ffe0b2; }
.stage-sealed { background: #e8f5e9; color: #2e7d32; border-color: #c8e6c9; }
.stage-training { background: #ede7f6; color: #5e35b1; border-color: #d1c4e9; }
.stage-completed { background: #e0f7fa; color: #00695c; border-color: #b2ebf2; }
.stage-failed { background: #ffebee; color: #c62828; border-color: #ffcdd2; }

.train-mark {
  font-size: 0.75rem;
  padding: 2px 8px;
  border-radius: 10px;
  white-space: nowrap;
}
.train-mark.ok { background: #e0f7fa; color: #00695c; }
.train-mark.todo { background: #f3f6fa; color: #8a94a6; }

.time-line { color: #7f8c8d; font-size: 0.78rem; }

.validation {
  font-size: 0.75rem;
  padding: 2px 8px;
  border-radius: 10px;
  white-space: nowrap;
  display: inline-block;
}
.validation.ok { background: #e8f5e9; color: #2e7d32; }
.validation.failed { background: #ffebee; color: #c62828; }
.validation.none { color: #b0b8c4; }

.row-actions { display: flex; gap: 6px; }

.inline-empty {
  color: #7f8c8d;
  padding: 24px 0;
}

.bind-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }

.bind-select {
  min-width: 300px;
  padding: 8px 10px;
  border: 1px solid #d5dee8;
  border-radius: 8px;
  background: #fff;
}

.form-hint { color: #9aa7b6; font-size: 0.8rem; }

.empty-state { text-align: center; color: #7f8c8d; padding: 60px; background: #fff; border: 1px dashed #cfd8e3; border-radius: 12px; }
</style>
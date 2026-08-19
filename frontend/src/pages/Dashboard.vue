<template>
  <div class="model-repo">
    <div class="repo-header">
      <div>
        <h1>模型仓库</h1>
        <p class="repo-sub">以模型为中心：一个模型一个卡片，采集/上传的数据挂靠到对应模型下，点击卡片查看其数据集与状态</p>
      </div>
      <div class="repo-stats">
        <span class="stat-pill">共 {{ models.length }} 个模型</span>
        <span class="stat-pill" v-if="totalDatasets">{{ totalDatasets }} 个数据集</span>
        <span class="stat-pill warn" v-if="unboundCount">⚠ {{ unboundCount }} 个数据集未归属</span>
      </div>
    </div>

    <div v-if="loading" class="loading-state"><span class="loading-spinner"></span>加载模型仓库...</div>

    <div v-else-if="!models.length" class="empty-state">
      <p>仓库还没有模型。模型可以暂时为空（待采集），注册第一个模型后挂载数据集。</p>
      <router-link to="/models"><button class="primary">去模型管理</button></router-link>
    </div>

    <div v-else class="model-grid">
      <div
        v-for="m in models"
        :key="m.model_id"
        class="model-card"
        @click="$router.push(`/models/${m.model_id}`)"
      >
        <div class="model-card-head">
          <span class="model-name">{{ m.display_name || m.name || m.model_code || m.model_id }}</span>
          <code class="model-code">{{ m.model_code || m.model_id }}</code>
        </div>

        <div class="model-ds-count">
          挂载 <b>{{ m.dataset_count || 0 }}</b> 个数据集
        </div>

        <div class="stage-bars">
          <span v-for="(c, st) in m.dataset_stats" :key="st" class="stage-pill" :class="`stage-${st}`">
            {{ stageLabel(String(st)) }} × {{ c }}
          </span>
          <span v-if="!m.dataset_count" class="stage-pill stage-empty">空模型 · 待采集</span>
        </div>

        <div class="model-card-foot">
          <span class="status-tag" :class="m.status">{{ statusLabel(m.status) }}</span>
          <span class="enter-btn">查看数据集 →</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { listModels } from '@/api/models'
import { listDatasets } from '@/api/datasets'

const loading = ref(true)
const models = ref<any[]>([])
const totalDatasets = ref(0)
const unboundCount = ref(0)

const STAGE_LABEL: Record<string, string> = {
  collecting: '采集中',
  annotating: '标注中',
  sealed: '已封板',
  training: '训练中',
  completed: '已完成训练',
  failed: '训练失败'
}

function stageLabel(s: string) {
  return STAGE_LABEL[s] || s
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

onMounted(async () => {
  try {
    const modelRes = await listModels()
    models.value = modelRes.models || []

    // 统计未归属数据集（尚未挂到任何模型，提示去模型详情页绑定）
    const dsRes = await listDatasets()
    const all = dsRes.datasets || []
    totalDatasets.value = all.length
    unboundCount.value = all.filter((d: any) => !d.model_id).length
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.model-repo {
  max-width: 1200px;
}

.repo-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 20px;
}

.repo-header h1 {
  color: #2c3e50;
  margin-bottom: 6px;
}

.repo-sub {
  color: #7f8c8d;
  margin: 0;
}

.repo-stats {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.stat-pill {
  background: #f0f4f8;
  color: #4a6785;
  border-radius: 12px;
  padding: 4px 12px;
  font-size: 0.8rem;
  border: 1px solid #dce5ee;
}

.stat-pill.warn {
  background: #fff7e6;
  color: #b26a00;
  border-color: #ffd591;
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}

.model-card {
  background: #fff;
  border: 1px solid #e4e9f0;
  border-radius: 12px;
  padding: 16px;
  cursor: pointer;
  transition: box-shadow 0.2s, transform 0.15s;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.model-card:hover {
  box-shadow: 0 6px 18px rgba(30, 60, 110, 0.12);
  transform: translateY(-2px);
}

.model-card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}

.model-name {
  font-size: 1.15rem;
  font-weight: 700;
  color: #1f2d3d;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-code {
  font-size: 0.72rem;
  color: #7a8aa0;
  background: #f3f6fa;
  padding: 2px 8px;
  border-radius: 6px;
  flex-shrink: 0;
}

.model-ds-count {
  color: #556;
  font-size: 0.9rem;
}

.model-ds-count b {
  color: #2c6ec5;
  font-size: 1.2rem;
}

.stage-bars {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.stage-pill {
  font-size: 0.75rem;
  padding: 3px 8px;
  border-radius: 10px;
  border: 1px solid transparent;
}

.stage-collecting { background: #eef4ff; color: #2c6ec5; border-color: #cfe0ff; }
.stage-annotating { background: #fff3e0; color: #b26a00; border-color: #ffe0b2; }
.stage-sealed { background: #e8f5e9; color: #2e7d32; border-color: #c8e6c9; }
.stage-training { background: #ede7f6; color: #5e35b1; border-color: #d1c4e9; }
.stage-completed { background: #e0f7fa; color: #00695c; border-color: #b2ebf2; }
.stage-failed { background: #ffebee; color: #c62828; border-color: #ffcdd2; }
.stage-empty { background: #fafafa; color: #9e9e9e; border-color: #e0e0e0; font-style: italic; }

.model-card-foot {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-top: 1px solid #eef2f6;
  padding-top: 10px;
}

.status-tag {
  font-size: 0.75rem;
  padding: 2px 10px;
  border-radius: 10px;
  background: #f1f3f5;
  color: #6b7c93;
}

.status-tag.production_ready,
.status-tag.active { background: #e8f5e9; color: #2e7d32; }
.status-tag.rejected { background: #ffebee; color: #c62828; }
.status-tag.superseded { background: #eceff1; color: #78909c; }

.enter-btn {
  color: #2c6ec5;
  font-size: 0.85rem;
  font-weight: 600;
}

.empty-state {
  text-align: center;
  color: #7f8c8d;
  padding: 60px 20px;
  background: #fff;
  border: 1px dashed #cfd8e3;
  border-radius: 12px;
}

.empty-state button { margin-top: 12px; }
</style>
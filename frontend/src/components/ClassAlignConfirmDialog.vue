<template>
  <div v-if="visible" class="modal-mask" @click.self="onCancel">
    <div class="modal align-confirm-modal">
      <h3>更换算法库 · 类别对齐确认</h3>
      <p class="modal-sub">
        该数据集的类别与新库标签字典无法自动唯一对齐，换库前需逐类确认映射关系。<br />
        确认后会按新库重写标签、每图标注与 data.yaml，再完成归属切换；未确认不会换库。
      </p>

      <div class="align-card align-needs_confirm">
        <div class="align-head">
          <span class="align-title">类别对齐</span>
          <span class="align-status needs_confirm">待确认</span>
        </div>
        <p class="align-summary">{{ report?.summary }}</p>
        <div class="align-rows">
          <div class="align-row" v-for="p in (report?.pairs || [])" :key="p.src_index">
            <span class="align-src">{{ p.src_name }} <small>第 {{ p.src_index }} 类</small></span>
            <span class="align-arrow">→</span>
            <select v-model="choices[p.src_index]" class="align-select">
              <option value="">请选择…</option>
              <option v-for="(nm, i) in (report?.canonical_names || [])" :key="i" :value="i">
                {{ nm }}（第 {{ i }} 类）
              </option>
              <option value="__ignore__">忽略该类（丢弃这些框）</option>
            </select>
            <span v-if="p.status === 'ambiguous'" class="align-warn">
              有歧义，候选 {{ (p.candidates || []).join(', ') }}
            </span>
          </div>
        </div>
      </div>

      <div v-if="error" class="form-error">{{ error }}</div>

      <div class="modal-actions">
        <button class="secondary" @click="onCancel">取消</button>
        <button class="primary" :disabled="submitting" @click="submit">
          {{ submitting ? '处理中…' : '确认映射并换库' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { bindDatasetModel } from '@/api/datasets'
import { showToast } from '@/utils/toast'

const props = defineProps<{
  visible: boolean
  datasetId: string
  report: any
  targetModelId: string | null
}>()

const emit = defineEmits<{
  (e: 'update:visible', v: boolean): void
  (e: 'confirmed'): void
  (e: 'cancel'): void
}>()

const choices = ref<Record<number, any>>({})
const submitting = ref(false)
const error = ref('')

// 打开时按报告预填：已有唯一归属的默认选中，歧义/认不出留空待选
watch(
  () => [props.visible, props.report],
  () => {
    const c: Record<number, any> = {}
    for (const p of (props.report?.pairs || []) as any[]) {
      if (p.status === 'exact' || p.status === 'alias') {
        c[p.src_index] = p.class_id
      } else {
        c[p.src_index] = '' // ambiguous / unknown 必须人工选
      }
    }
    choices.value = c
    error.value = ''
  },
  { immediate: true }
)

function onCancel() {
  emit('update:visible', false)
  emit('cancel')
}

async function submit() {
  if (!props.report?.pairs) return
  const decisions: Array<{ src_index: number; target_id: number | null }> = []
  for (const p of props.report.pairs as any[]) {
    const v = choices.value[p.src_index]
    // 数字下标 0 必须用显式判空（undefined/null/''），不能 !v
    if (v === undefined || v === null || v === '') {
      error.value = `第 ${p.src_index} 类（${p.src_name}）尚未选择映射目标`
      return
    }
    if (v === '__ignore__') {
      decisions.push({ src_index: p.src_index, target_id: null })
    } else {
      decisions.push({ src_index: p.src_index, target_id: Number(v) })
    }
  }
  if (!props.targetModelId) {
    error.value = '缺少目标算法库标识，无法换库'
    return
  }
  submitting.value = true
  error.value = ''
  try {
    await bindDatasetModel(props.datasetId, props.targetModelId, false, decisions)
    emit('update:visible', false)
    emit('confirmed')
    showToast('类别映射已确认，已切换归属算法库', 'info', 3500)
  } catch (e: any) {
    const status = e?.response?.status
    const detail = e?.response?.data?.detail || e?.message || ''
    if (status === 409) {
      // 决策不完整被后端再次拦截：刷新报告让用户继续
      error.value = '仍有类别需确认：' + detail
    } else {
      error.value = '换库失败: ' + detail
    }
  } finally {
    submitting.value = false
  }
}
</script>

<style scoped>
.align-confirm-modal {
  max-width: 640px;
  width: 92%;
}
.modal-sub {
  color: #666;
  font-size: 13px;
  line-height: 1.6;
  margin: 6px 0 14px;
}
.align-card {
  border: 1px solid #e3e3e3;
  border-radius: 8px;
  padding: 12px 14px;
  background: #fafafa;
}
.align-card.align-needs_confirm {
  border-color: #f0c36d;
  background: #fffbf2;
}
.align-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.align-title {
  font-weight: 600;
}
.align-status {
  font-size: 12px;
  padding: 2px 10px;
  border-radius: 10px;
}
.align-status.needs_confirm {
  background: #fde8c4;
  color: #8a5a00;
}
.align-summary {
  margin: 8px 0 12px;
  color: #555;
  font-size: 13px;
}
.align-rows {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.align-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.align-src {
  min-width: 130px;
  font-weight: 500;
}
.align-src small {
  color: #999;
  font-weight: 400;
}
.align-arrow {
  color: #999;
}
.align-select {
  flex: 1;
  min-width: 200px;
  padding: 6px 8px;
  border: 1px solid #ccc;
  border-radius: 6px;
}
.align-warn {
  color: #b00;
  font-size: 12px;
}
.form-error {
  color: #c0392b;
  font-size: 13px;
  margin: 10px 0 0;
}
.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 16px;
}
</style>

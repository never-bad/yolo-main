<template>
  <!-- 点遮罩不关闭：表单填了一半时误点外面会丢内容，只能通过「取消」或创建成功关闭 -->
  <div v-if="open" class="modal-mask">
    <div class="modal create-modal">
      <h3>新建算法库</h3>
      <p class="modal-sub">唯一标识用于生成底层路径，创建后写死不可修改</p>
      <div class="form-field">
        <label>算法库名称 <i>*</i></label>
        <input v-model="form.display_name" placeholder="如：安全帽检测" />
      </div>
      <div class="form-field">
        <label>唯一标识 <i>*</i></label>
        <input v-model="form.model_code" placeholder="如：helmet_detect（小写+下划线）" />
      </div>
      <div class="form-field">
        <label>业务线</label>
        <select v-model="form.business">
          <option value="">未指定</option>
          <option v-for="(label, key) in BIZ_OPTIONS" :key="key" :value="key">{{ label }}</option>
        </select>
      </div>
      <div class="form-field">
        <label>业务场景描述</label>
        <textarea v-model="form.description" rows="2" placeholder="告诉团队成员这个模型的用途"></textarea>
      </div>
      <div v-if="error" class="form-error">{{ error }}</div>
      <div class="modal-actions">
        <button class="secondary" @click="close">取消</button>
        <button class="primary" @click="submit" :disabled="creating">
          {{ creating ? '创建中...' : confirmText }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { createModel } from '@/api/models'

const BIZ_OPTIONS: Record<string, string> = {
  general: '通用目标检测',
  pedestrian: '行人检测',
  vehicle: '车辆 / 车牌',
  defect: '工业缺陷检测',
  package: '包裹 / 物流'
}

const props = withDefaults(
  defineProps<{
    open: boolean
    confirmText?: string
  }>(),
  { confirmText: '创建并进入详情' }
)

const emit = defineEmits<{
  (e: 'update:open', value: boolean): void
  /** 创建成功：返回新建算法库的元数据（含 model_id），由调用方决定后续动作 */
  (e: 'created', meta: any): void
}>()

const form = ref({ display_name: '', model_code: '', business: '', description: '' })
const error = ref('')
const creating = ref(false)

// 每次打开都重置成空白，避免上次填写残留
watch(
  () => props.open,
  (v) => {
    if (v) {
      form.value = { display_name: '', model_code: '', business: '', description: '' }
      error.value = ''
    }
  }
)

function close() {
  emit('update:open', false)
}

async function submit() {
  error.value = ''
  const name = form.value.display_name.trim()
  const code = form.value.model_code.trim()
  if (!name) {
    error.value = '请填写算法库名称'
    return
  }
  if (!code) {
    error.value = '请填写唯一标识（英文小写+下划线）'
    return
  }
  creating.value = true
  try {
    const meta = await createModel({
      display_name: name,
      model_code: code,
      business: form.value.business,
      description: form.value.description.trim()
    })
    emit('update:open', false)
    emit('created', meta)
  } catch (e: any) {
    error.value = e?.response?.data?.detail || '创建失败'
  } finally {
    creating.value = false
  }
}
</script>

<style scoped>
/* 弹窗样式与首页保持一致（scoped 样式不外溢，故组件内自带一份） */
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
</style>

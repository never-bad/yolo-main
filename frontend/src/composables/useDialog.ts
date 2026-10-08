import { reactive } from 'vue'
import { deleteModel, listModels } from '@/api/models'

export type DialogType = 'alert' | 'confirm' | 'prompt'

const state = reactive({
  visible: false,
  type: 'alert' as DialogType,
  title: '提示',
  message: '',
  value: '',
})

let resolver: ((val: string | boolean | null) => void) | null = null

const TITLE_MAP: Record<DialogType, string> = {
  alert: '提示',
  confirm: '确认',
  prompt: '输入',
}

function show(type: DialogType, message: string, value?: string, title?: string): Promise<string | boolean | null> {
  state.type = type
  state.title = title || TITLE_MAP[type]
  state.message = message
  state.value = value ?? ''
  state.visible = true
  return new Promise((res) => {
    resolver = res
  })
}

/** 显示居中提示弹框 */
export function showAlert(message: string, title?: string): Promise<string | boolean | null> {
  return show('alert', message, undefined, title)
}

/** 显示居中确认弹框，返回 true 表示确认 */
export function showConfirm(message: string, title?: string): Promise<boolean> {
  return show('confirm', message, undefined, title) as Promise<boolean>
}

/** 显示居中输入弹框，返回输入内容；取消返回 null */
export function showPrompt(message: string, value?: string, title?: string): Promise<string | null> {
  return show('prompt', message, value, title) as Promise<string | null>
}

/** 确定 */
export function resolveDialog(value: string | boolean | null) {
  state.visible = false
  resolver?.(value)
  resolver = null
}

export function useDialog() {
  return { state, resolveDialog }
}

// ===== P2-5 删库守卫流程（三处删库入口共用，禁级联硬删制造孤儿数据集） =====
/**
 * 删除算法库（含挂接数据集守卫）：
 * - 挂接为空：后端直删；
 * - 挂接非空：弹清单让用户选「转移归属 / 保留为未归属」，数据集与标注 JSON 均不删除；
 * - 转移遇类别序号冲突（400）时再确认是否强制转移。
 * 返回 'deleted'（已删除）或 'cancelled'（用户放弃，库未删）。网络/服务异常向上抛，由调用方 catch。
 */
export async function deleteModelWithGuard(modelId: string, displayName: string): Promise<'deleted' | 'cancelled'> {
  const res = await deleteModel(modelId)
  if (res.deleted) return 'deleted'

  const datasets = res.datasets || []
  const list = datasets.map((d, i) => `${i + 1}. ${d.name}（${d.image_count} 张图）`).join('\n')
  const toTransfer = await showConfirm(
    `算法库「${displayName}」仍挂接 ${datasets.length} 个数据集：\n${list}\n\n` +
    '删除该库前请选择数据集去向：\n' +
    '「确定」= 转移到其他算法库\n' +
    '「取消」= 保留为未归属（数据集与标注不会被删除）',
    '该库仍有挂接数据集',
  )

  if (!toTransfer) {
    const r = await deleteModel(modelId, { action: 'unbind' })
    return r.deleted ? 'deleted' : 'cancelled'
  }

  const all = (await listModels()).models || []
  const targets = all.filter((m: any) => (m.library_root_id || m.model_id) === m.model_id && m.model_id !== modelId)
  if (!targets.length) {
    await showAlert('没有其他算法库可供转移，数据集将保留为未归属。')
    const r = await deleteModel(modelId, { action: 'unbind' })
    return r.deleted ? 'deleted' : 'cancelled'
  }
  const menu = targets.map((m: any, i: number) => `${i + 1}. ${m.display_name || m.name || m.model_id}`).join('\n')
  const input = await showPrompt(`输入序号选择目标算法库：\n${menu}`, '1', '转移归属')
  if (input === null) return 'cancelled'
  const idx = Number(input) - 1
  const target = targets[idx]
  if (!Number.isInteger(idx) || !target) {
    await showAlert(`序号无效：${input}`)
    return 'cancelled'
  }
  const targetName = target.display_name || target.name || target.model_id
  try {
    const r = await deleteModel(modelId, { action: 'transfer', target_model_id: target.model_id })
    return r.deleted ? 'deleted' : 'cancelled'
  } catch (e: any) {
    const detail = e?.response?.data?.detail || ''
    if (e?.response?.status === 400 && detail.includes('冲突')) {
      const forced = await showConfirm(`${detail}\n\n仍要强制转移到「${targetName}」吗？`, '类别序号冲突')
      if (!forced) return 'cancelled'
      const r = await deleteModel(modelId, { action: 'transfer', target_model_id: target.model_id, force: true })
      return r.deleted ? 'deleted' : 'cancelled'
    }
    throw e
  }
}
import { reactive } from 'vue'

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
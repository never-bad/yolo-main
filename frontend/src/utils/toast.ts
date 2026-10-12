// 全局轻提示：纯 DOM 单实例节点，非阻塞、自动消失，替代散落各页的 alert 弹框。
// 不依赖任何 UI 组件库（红线 X-02），样式见 src/style.css 的 .app-toast。
// 签名与标注页原本地 showToast 完全一致，收敛时调用点零改动。
export type ToastType = 'success' | 'error' | 'info'

let toastEl: HTMLDivElement | null = null
let toastTimer: any = null

// 单例容器：首次调用时动态挂到 body，之后复用同一节点（新提示直接覆盖旧提示）
const ensureToastEl = (): HTMLDivElement => {
  if (toastEl && toastEl.isConnected) return toastEl
  const el = document.createElement('div')
  el.id = 'app-toast'
  document.body.appendChild(el)
  toastEl = el
  return el
}

export const showToast = (msg: string, type: ToastType = 'info', durationMs = 2500) => {
  const el = ensureToastEl()
  el.className = 'app-toast ' + type
  el.textContent = msg
  el.style.display = 'block'
  // 复用同一 timer：连续提示时先清掉上一次的隐藏动作，避免提前消失
  if (toastTimer) clearTimeout(toastTimer)
  toastTimer = setTimeout(() => {
    el.style.display = 'none'
    el.textContent = ''
  }, durationMs)
}

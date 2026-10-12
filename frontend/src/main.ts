import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'
import App from './App.vue'
import './style.css'
import { showAlert } from './composables/useDialog'
import { showToast } from './utils/toast'
import { normalizeDetail } from './api/axios'

// 将原生 alert 替换为居中弹框，使整个平台的提示都显示在屏幕正中间
window.alert = (message?: string) => {
  showAlert(message ?? '')
}

// 未捕获的 Promise 拒绝兜底：只处理 axios 错误、且拦截器尚未弹过提示的那些，
// 避免与 axios 拦截器、页面自身 catch 的提示叠加成多条。
window.addEventListener('unhandledrejection', (e) => {
  const r: any = e.reason
  if (!r || !r.isAxiosError || r.__toastShown) return
  e.preventDefault()
  const detail = normalizeDetail(r.response?.data?.detail)
  showToast('请求失败：' + (detail || r.message || '未知错误'), 'error', 5000)
})

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.mount('#app')

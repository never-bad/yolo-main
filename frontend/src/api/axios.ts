import axios from 'axios'
import { showToast } from '../utils/toast'

// 使用环境变量配置 API 基础路径
// 开发环境: /dev-api, 生产环境: /prod-api
const apiBaseURL = import.meta.env.VITE_API_BASE || '/dev-api'

const api = axios.create({
  baseURL: apiBaseURL,
  timeout: 300000
})

// 把后端返回的 detail 归一化成单行可读文案：
// 字符串直用；FastAPI 422 校验数组取首项拼成「字段路径: 说明」；其余 JSON 序列化兜底。
export const normalizeDetail = (detail: any): string => {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0]
    const loc = Array.isArray(first?.loc) ? first.loc : []
    return `${loc.join('.')}: ${first?.msg ?? ''}`
  }
  if (detail === undefined || detail === null) return ''
  try {
    return JSON.stringify(detail)
  } catch {
    return String(detail)
  }
}

// 全局错误兜底拦截器：
// - 无 response（网络断开 / 超时 ECONNABORTED）→ 弹「网络异常」轻提示
// - 5xx（服务端错误）→ 弹「服务端错误：detail」轻提示
// - 4xx 一律不弹不标记，仍由各页面自己的 catch 按业务语义提示
// 弹过的错误打 __toastShown 标记，供页面 catch 与 main.ts 的 unhandledrejection 兜底去重。
// 只标记不吞错：原样 Promise.reject 上抛。
api.interceptors.response.use(
  (response) => response,
  (err: any) => {
    if (!err.response) {
      showToast('网络异常或请求超时，请检查服务状态', 'error', 4000)
      err.__toastShown = true
    } else if (err.response.status >= 500) {
      const detail = normalizeDetail(err.response.data?.detail)
      showToast('服务端错误：' + (detail || err.response.status), 'error', 4000)
      err.__toastShown = true
    }
    return Promise.reject(err)
  }
)

export default api

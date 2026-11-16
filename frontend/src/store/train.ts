import { defineStore } from 'pinia'
import { ref } from 'vue'

// 单个任务的进度快照（由日志解析得出，数据源在服务器，不在浏览器）
export interface JobProgress {
  current: number
  total: number
  percent: number
  has: boolean
  lossLine: string
  lastActivity: number
}

/**
 * 训练任务的跨页面缓存。
 *
 * 定位：这是「服务器真值的一份本地镜像」，不是「本设备的状态」。
 * 训练任务列表和进度都来自后端（jobs/*.json + 训练日志），A 电脑和 B 电脑
 * 各自请求同一个后端，看到的必然一致；这里缓存的只是「上一次已经拿到的结果」，
 * 用于切页回来时立刻有内容可画，避免骨架屏闪一下、进度条从 0 重新爬。
 *
 * 因此在切回训练页时**必须**再向服务器刷新一次（loadJobs），
 * 缓存只负责首屏，不负责长期正确性。
 */
export const useTrainStore = defineStore('train', () => {
  const jobs = ref<any[]>([])
  // key: job_id
  const jobProgress = ref<Record<string, JobProgress>>({})
  // 是否成功拉过一次任务列表：用于区分「首次进入（该显示骨架屏）」
  // 和「从别的页面切回来（直接画旧数据，静默刷新）」
  const loadedOnce = ref(false)
  const lastLoadedAt = ref(0)

  const setJobs = (list: any[]) => {
    jobs.value = list
    loadedOnce.value = true
    lastLoadedAt.value = Date.now()
  }

  const setProgress = (jobId: string, p: JobProgress) => {
    jobProgress.value[jobId] = p
  }

  // 删除任务时清掉它的进度残留，避免残留 key 越积越多
  const dropProgress = (jobId: string) => {
    delete jobProgress.value[jobId]
  }

  return { jobs, jobProgress, loadedOnce, lastLoadedAt, setJobs, setProgress, dropProgress }
})
import api from './axios'

export interface BBox {
  class_id: number
  x1: number
  y1: number
  x2: number
  y2: number
}

export const createAnnotationTask = async (datasetId: string, version: string, classes?: string[]) => {
  const { data } = await api.post('/annotations/tasks', {
    dataset_id: datasetId,
    version,
    classes: classes || null  // 如果不提供则从 data.yaml 读取
  })
  return data
}

export const getTaskItems = async (taskId: string) => {
  const { data } = await api.get(`/annotations/tasks/${taskId}/items`)
  return data
}

export const getImageAnnotation = async (taskId: string, imageId: string) => {
  const { data } = await api.get(`/annotations/tasks/${taskId}/items/${imageId}`)
  return data
}

export const saveAnnotation = async (taskId: string, imageId: string, boxes: BBox[], ai_annotated = false) => {
  const { data } = await api.post(`/annotations/tasks/${taskId}/items/${imageId}`, {
    boxes,
    ai_annotated
  })
  return data
}

export const exportAnnotations = async (taskId: string) => {
  const { data } = await api.get(`/annotations/tasks/${taskId}/export?format=yolo`)
  return data
}

// 导出为YOLO格式，并按比例自动划分训练/验证/测试集
export const exportAnnotationsSplit = async (taskId: string, train: number, val: number, test: number) => {
  const { data } = await api.post(`/annotations/tasks/${taskId}/export-split`, { train, val, test })
  return data
}

// ===== SAM 大模型预标注 =====
export interface AIBox extends BBox {
  score?: number
}

export const getSamAvailable = async () => {
  const { data } = await api.get('/sam/available')
  return data
}

export const getSamConfig = async () => {
  const { data } = await api.get('/sam/config')
  return data
}

export const updateSamConfig = async (payload: any) => {
  const { data } = await api.post('/sam/config', payload)
  return data
}

export const getSamModels = async () => {
  const { data } = await api.get('/sam/models')
  return data
}

export const uploadSamModel = async (file: File) => {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/sam/models', form)
  return data
}

export const autoLabelImage = async (taskId: string, imageId: string, classes: string[], conf?: number, prompts?: string[], signal?: AbortSignal) => {
  const { data } = await api.post('/sam/auto-label', {
    task_id: taskId,
    image_id: imageId,
    classes,
    conf: conf ?? undefined,
    prompts
  }, { signal })
  return data
}

export const startBatchLabel = async (taskId: string, classes: string[], conf?: number, prompts?: string[]) => {
  const { data } = await api.post('/sam/batch/start', {
    task_id: taskId,
    classes,
    conf: conf ?? undefined,
    prompts
  })
  return data
}

export const getBatchProgress = async (batchId: string) => {
  const { data } = await api.get(`/sam/batch/${batchId}`)
  return data
}

export const stopBatchLabel = async (batchId: string) => {
  const { data } = await api.post(`/sam/batch/${batchId}/stop`)
  return data
}

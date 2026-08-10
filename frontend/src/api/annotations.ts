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

export const saveAnnotation = async (taskId: string, imageId: string, boxes: BBox[]) => {
  const { data } = await api.post(`/annotations/tasks/${taskId}/items/${imageId}`, {
    boxes
  })
  return data
}

export const exportAnnotations = async (taskId: string) => {
  const { data } = await api.get(`/annotations/tasks/${taskId}/export?format=yolo`)
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

export const autoLabelImage = async (taskId: string, imageId: string, classes: string[], conf?: number, prompts?: string[]) => {
  const { data } = await api.post('/sam/auto-label', {
    task_id: taskId,
    image_id: imageId,
    classes,
    conf: conf ?? undefined,
    prompts
  })
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

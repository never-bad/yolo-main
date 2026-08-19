import api from './axios'

export interface BBox {
  class_id: number
  x1: number
  y1: number
  x2: number
  y2: number
  confidence?: number
  source?: string
}

export const createAnnotationTask = async (datasetId: string, version: string, classes?: string[]) => {
  const { data } = await api.post('/annotations/tasks', {
    dataset_id: datasetId,
    version,
    classes: classes || null  // 如果不提供则从 data.yaml 读取
  })
  return data
}

// 按数据集查找已存在的标注任务（标注页直达 / 自动建任务去重用）
export const findAnnotationTask = async (datasetId: string, version: string = 'v1') => {
  const { data } = await api.get(`/annotations/tasks/by-dataset/${datasetId}`, {
    params: { version }
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

export const saveAnnotation = async (taskId: string, imageId: string, boxes: BBox[], ai_annotated = false, sampleType?: string, sampleReason?: string) => {
  const { data } = await api.post(`/annotations/tasks/${taskId}/items/${imageId}`, {
    boxes,
    ai_annotated,
    sample_type: sampleType || null,
    sample_reason: sampleReason || null
  })
  return data
}

// 导出为YOLO格式（不划分，数据集划分在封板时进行）
export const exportAnnotations = async (taskId: string) => {
  const { data } = await api.get(`/annotations/tasks/${taskId}/export?format=yolo`)
  return data
}

// 清除该任务所有 AI 预标注（批量误标过多时一键清理）
export const clearAiAnnotations = async (taskId: string) => {
  const { data } = await api.post(`/annotations/tasks/${taskId}/clear-ai`)
  return data
}

// 清理该任务全部图片的高度重合标注框（跨类别/同类重复，历史遗留重叠框统一根治）
export const cleanTaskOverlaps = async (taskId: string) => {
  const { data } = await api.post(`/annotations/tasks/${taskId}/clean-overlaps`)
  return data
}

// 导入 YOLO 格式标注（zip 包内含 labels/*.txt），把外部工具（如 X-AnyLabeling）标注的数据接入平台
export const importYoloLabels = async (taskId: string, file: File) => {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post(`/annotations/tasks/${taskId}/import-yolo`, form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
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

// 交互式标注：在用户框选的局部区域内按文本提示检测目标（避免全图盲扫误标）
export const interactiveLabelImage = async (
  taskId: string,
  imageId: string,
  classes: string[],
  conf?: number,
  prompts?: string[],
  region?: { x1: number; y1: number; x2: number; y2: number }
) => {
  const { data } = await api.post('/sam/interactive-label', {
    task_id: taskId,
    image_id: imageId,
    classes,
    conf: conf ?? undefined,
    prompts,
    region
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

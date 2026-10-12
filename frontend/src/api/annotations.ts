import api from './axios'

export interface BBox {
  class_id: number
  x1: number
  y1: number
  x2: number
  y2: number
  polygon?: number[][]     // 分割轮廓（像素坐标 [[x,y],...]）；有则该框是分割标注
  confidence?: number
  source?: string
  box_id?: number          // AI 预标注返回列表下标，用于映射千问审核结论
  audit?: any              // 千问 L2 审核结论 {action,reason,tags,kind,suggested_class_id}（不落盘）
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

// 删除任务中的一张图片（同时清理其标注 JSON 与同名标签文件，并同步数据集计数）
export const deleteTaskImage = async (taskId: string, imageId: string) => {
  const { data } = await api.delete(`/annotations/tasks/${taskId}/images/${imageId}`)
  return data
}

export const saveAnnotation = async (taskId: string, imageId: string, boxes: BBox[], ai_annotated = false, sampleType?: string, sampleReason?: string, reviewed = false) => {
  const { data } = await api.post(`/annotations/tasks/${taskId}/items/${imageId}`, {
    boxes,
    ai_annotated,
    sample_type: sampleType || null,
    sample_reason: sampleReason || null,
    // 人工复核标记：本次保存前画布上真正编辑过标注才传 true（后端粘性记录，供"轮廓质量"筛选）
    reviewed
  })
  return data
}

// 导出为YOLO格式（不划分，数据集划分在封板时进行）
export const exportAnnotations = async (taskId: string) => {
  const { data } = await api.get(`/annotations/tasks/${taskId}/export?format=yolo`)
  return data
}

// 单独提取"已修过"的图片+YOLO标签，打包成 zip（blob）直接下载
export const exportReviewedZip = async (taskId: string): Promise<Blob> => {
  const { data } = await api.get(`/annotations/tasks/${taskId}/export-reviewed`, { responseType: 'blob' })
  return data
}

// 清除该任务所有 AI 预标注（批量误标过多时一键清理）
export const clearAiAnnotations = async (taskId: string) => {
  const { data } = await api.post(`/annotations/tasks/${taskId}/clear-ai`)
  return data
}

// 清除该任务数据集的全部标注框（AI + 人工都清）；后端清空前会自动打包备份，可回退
export const clearAllAnnotations = async (taskId: string) => {
  const { data } = await api.post(`/annotations/tasks/${taskId}/clear-all`)
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


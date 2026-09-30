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

// 「本库已训模型」预标注引擎可选的版本（优先返回当前标注任务所属算法库训练出的版本）
export const getSamTrainedVersions = async (taskId?: string) => {
  const { data } = await api.get('/sam/trained-versions', { params: taskId ? { task_id: taskId } : {} })
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

// 千问 L2 审核：对当前图片标注框做中文语义二次审核（漏标/粘连/标签错误/框质量/合规）
// boxes 不传时后端读已保存标注；传时按数组下标当作 box_id 映射审核结论
export const auditImage = async (
  taskId: string,
  imageId: string,
  boxes?: { class_id: number; x1: number; y1: number; x2: number; y2: number }[]
) => {
  const { data } = await api.post('/sam/audit', {
    task_id: taskId,
    image_id: imageId,
    boxes: boxes || null
  })
  return data
}

// 问题图片检索：返回任务中带审核异议（reject/modify/split）的图片列表
export const listAuditIssues = async (taskId: string) => {
  const { data } = await api.get('/sam/audit/issues', { params: { task_id: taskId } })
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

// 点选抠图（智能抠图）：在图上点目标 → SAM 沿边缘描出轮廓 → 返回一个分割标注
// points: [{x, y, label}]，label=1 点在目标上（要这块），0 点在多抠的地方（抠掉）
export const pointLabelImage = async (
  taskId: string,
  imageId: string,
  classId: number,
  points: { x: number; y: number; label: number }[]
) => {
  const { data } = await api.post('/sam/point-label', {
    task_id: taskId,
    image_id: imageId,
    class_id: classId,
    points
  })
  return data
}

// 区域擦除：把分割轮廓按矩形挖掉多余部分（纯几何、毫秒级，不走 SAM）
// 返回 {boxes: [...]}，全挖空时 boxes 为空数组
export const erasePolygonRegion = async (
  polygon: number[][],
  eraseRects: { x1: number; y1: number; x2: number; y2: number }[],
  width: number,
  height: number,
  classId: number
) => {
  const { data } = await api.post('/sam/erase-polygon', {
    polygon,
    erase_rects: eraseRects,
    width,
    height,
    class_id: classId
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

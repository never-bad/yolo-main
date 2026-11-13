import api from './axios'

// ===== SAM 大模型预标注 =====
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

// 「本库已训模型」预标注引擎可选的版本（B-60：只返回当前标注任务所属算法库训练出的版本）
export const getSamTrainedVersions = async (taskId?: string) => {
  const { data } = await api.get('/sam/trained-versions', { params: taskId ? { task_id: taskId } : {} })
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

// 批量 L2 审核：对任务内全部已标注图片并发跑千问语义二审，结论写回每图 JSON
// 进度/停止复用 getBatchProgress / stopBatchLabel（同一 batch_id 与 /sam/batch/{id} 端点）
export const auditBatchStart = async (taskId: string, concurrency?: number) => {
  const { data } = await api.post('/sam/audit/batch/start', {
    task_id: taskId,
    concurrency: concurrency ?? 16
  })
  return data
}

// 审核结果分类归集：扫描任务全部图片的审核结论，按问题类型 / 建议动作双维度列出需人工二审的图片
export const listAuditClassify = async (taskId: string) => {
  const { data } = await api.get('/sam/audit/classify', { params: { task_id: taskId } })
  return data
}

// M3 重复图检测：MD5 精确 + dHash 感知哈希全库查重，结论写回 audit_meta.duplicate
// 分类接口会归入 duplicate 桶，筛选下拉自动出现「审核·重复图片」分组
export const dupScanStart = async (taskId: string, threshold = 6) => {
  const { data } = await api.post('/sam/duplicates/scan/start', {
    task_id: taskId,
    threshold
  })
  return data
}

export const getDupScanStatus = async (batchId: string) => {
  const { data } = await api.get('/sam/duplicates/scan/status', { params: { batch_id: batchId } })
  return data
}

export const stopDupScan = async (batchId: string) => {
  const { data } = await api.post('/sam/duplicates/scan/stop', null, { params: { batch_id: batchId } })
  return data
}

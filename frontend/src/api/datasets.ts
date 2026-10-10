import api from './axios'

export interface UpdateDatasetRequest {
  description?: string
  tags?: string[]
}

/** 上传数据集压缩包；modelId 为归属的算法库（必填，后端会换算成库根记录）
 *  onProgress 可选：上传字节进度百分比回调，不传则与旧调用方完全一致 */
export const uploadDataset = async (
  file: File, modelId: string,
  onProgress?: (percent: number) => void
) => {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post('/datasets/upload', formData, {
    params: { model_id: modelId },
    onUploadProgress: (e) => {
      if (!onProgress) return
      const total = e.total || e.loaded || 1
      onProgress(Math.min(100, Math.round((e.loaded / total) * 100)))
    }
  })
  return data
}

export const prepareDataset = async (datasetId: string, splitRatio?: any, classes?: string[]) => {
  const { data } = await api.post(`/datasets/${datasetId}/prepare`, {
    split_ratio: splitRatio,
    classes
  })
  return data
}

export const listDatasets = async () => {
  const { data } = await api.get('/datasets')
  return data
}

export const getDataset = async (datasetId: string) => {
  const { data } = await api.get(`/datasets/${datasetId}`)
  return data
}

export const getDatasetTree = async (datasetId: string) => {
  const { data } = await api.get(`/datasets/${datasetId}/tree`)
  return data
}

export const updateDataset = async (datasetId: string, request: UpdateDatasetRequest) => {
  const { data } = await api.put(`/datasets/${datasetId}`, request)
  return data
}

export const sealDataset = async (datasetId: string, force: boolean = false, splitRatio?: { train: number, val: number, test: number }) => {
  const { data } = await api.post(`/datasets/${datasetId}/seal`, { force, split_ratio: splitRatio })
  return data
}

// 审核通过：人工确认标注无误后标记，之后才允许自动灌桶/封板
export const approveDatasetReview = async (datasetId: string) => {
  const { data } = await api.post(`/datasets/${datasetId}/approve`)
  return data
}

// 撤回：把已灌入接收桶但未封板的图片退回，恢复可编辑
export const unpourDatasetBucket = async (datasetId: string) => {
  const { data } = await api.post(`/datasets/${datasetId}/unpour`)
  return data
}

// 2.1 入料校验：手动重新执行标注格式校验
export const validateDataset = async (datasetId: string) => {
  const { data } = await api.post(`/datasets/${datasetId}/validate`)
  return data
}

// 挂载前校验：数据集换绑到某算法库后，已标注类别的序号会不会错位（同类别排不同序号 = 会串号）
export interface BindCheckResult {
  dataset_id: string
  target_library_id: string
  target_names: string[]
  annotated_images: number
  verifiable: boolean
  conflict: boolean
  summary: string
  blocking: { class_id: number | string; english_code: string; want_index: number | null; actual_index: number; images: number; boxes: number; reason: string }[]
  warnings: { class_id: number | string; english_code: string; images: number; boxes: number; reason: string }[]
}

export const checkDatasetBind = async (datasetId: string, modelId: string): Promise<BindCheckResult> => {
  const { data } = await api.get(`/datasets/${datasetId}/bind-check`, { params: { model_id: modelId } })
  return data
}

// 1.7 模型仓库：绑定/解绑数据集到模型（model_id 传 null 表示解绑；force 用于确认串号风险后放行）
export const bindDatasetModel = async (datasetId: string, modelId: string | null, force = false) => {
  const { data } = await api.put(`/datasets/${datasetId}/model`, { model_id: modelId, force })
  return data
}

export const deleteDataset = async (datasetId: string) => {
  const { data } = await api.delete(`/datasets/${datasetId}`)
  return data
}

export const exportAnnotatedDataset = async (datasetId: string, version: string = 'v1'): Promise<Blob> => {
  const response = await api.get(`/datasets/${datasetId}/export/annotated`, {
    params: { version },
    responseType: 'blob'
  })
  return response.data
}

export const exportOriginalDataset = async (datasetId: string, version: string = 'v1'): Promise<Blob> => {
  const response = await api.get(`/datasets/${datasetId}/export/original`, {
    params: { version },
    responseType: 'blob'
  })
  return response.data
}

// B 系列：按模型滚动灌桶（A 包标注完成自动灌桶，桶满 SEAL_MIN_IMAGES 自动封板，幂等）
export const advanceBuckets = async () => {
  const { data } = await api.post('/datasets/advance-buckets')
  return data
}

// 封板/桶灌数量门槛：查询与在线修改（持久化到 data/seal_threshold.json）
// modelId 传入 = 单独覆盖该库；不传 = 改全局默认；value 传 -1 = 该库恢复跟随全局
export const getSealThreshold = async (): Promise<{ seal_min_images: number; default: number; seal_max_days: number; per_model: Record<string, number> }> => {
  const { data } = await api.get('/datasets/config/seal-threshold')
  return data
}

export const setSealThreshold = async (sealMinImages: number, modelId?: string | null) => {
  const { data } = await api.put('/datasets/config/seal-threshold', { seal_min_images: sealMinImages, model_id: modelId ?? null })
  return data
}

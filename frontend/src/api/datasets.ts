import api from './axios'

export interface UpdateDatasetRequest {
  description?: string
  tags?: string[]
}

/** 上传数据集压缩包；modelId 为归属的算法库（必填，后端会换算成库根记录）
 *  onProgress 可选：上传字节进度百分比回调，不传则与旧调用方完全一致
 *  options 可选：uploadId（本次上传标识，供取消时精确清理服务端产物）+ signal（AbortSignal）
 *  第 4 个参数缺省时与旧调用方完全兼容（ModelDetail.vue 的两参调用不受影响） */
export const uploadDataset = async (
  file: File, modelId: string,
  onProgress?: (percent: number) => void,
  options?: { uploadId?: string; signal?: AbortSignal; role?: 'train' | 'test' }
) => {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post('/datasets/upload', formData, {
    params: {
      model_id: modelId,
      ...(options?.uploadId ? { upload_id: options.uploadId } : {}),
      ...(options?.role ? { role: options.role } : {})
    },
    signal: options?.signal,
    timeout: 0,
    onUploadProgress: (e) => {
      if (!onProgress) return
      const total = e.total || e.loaded || 1
      onProgress(Math.min(100, Math.round((e.loaded / total) * 100)))
    }
  })
  return data
}

/** 取消上传：通知服务端按 upload_id 清理在途产物（数据集目录/压缩包/PG 影子库）。
 *  幂等，可与上传并发；即使上传请求已断开、拿不到 dataset_id 也能清干净。 */
export const cancelUpload = async (uploadId: string) => {
  const { data } = await api.post('/datasets/upload/cancel', { upload_id: uploadId })
  return data
}

export const prepareDataset = async (datasetId: string, splitRatio?: any, classes?: string[]) => {
  const { data } = await api.post(`/datasets/${datasetId}/prepare`, {
    split_ratio: splitRatio,
    classes
  }, { timeout: 0 })
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

// 方案 B：多数据集合并为持久新数据集（源只读，产出新数据集可直接训练/封板/导出）
export const mergeDatasets = async (
  datasetIds: string[],
  libraryRootId?: string | null,
  displayName?: string | null
) => {
  const { data } = await api.post('/datasets/merge', {
    dataset_ids: datasetIds,
    library_root_id: libraryRootId ?? null,
    display_name: displayName ?? null,
  })
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

// B-47：把一个已标注数据集封板为所属算法库的独立测试集（held-out）。
// force=true 用于泄漏检查命中（测试集与同库训练图 md5 重合）时强制放行。
export const setDatasetAsTestSet = async (datasetId: string, force: boolean = false) => {
  const { data } = await api.post(`/datasets/${datasetId}/set-as-test-set`, { force })
  return data
}

// 测试集动态增删图片后重新封板（版本号 +1，同库旧指标标 stale）
export const resealTestSet = async (datasetId: string) => {
  const { data } = await api.post(`/datasets/${datasetId}/reseal-test-set`)
  return data
}

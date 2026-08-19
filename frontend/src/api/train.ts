import api from './axios'

export interface TrainJobRequest {
  dataset_id: string
  version?: string
  model_name?: string
  epochs: number
  imgsz?: number
  batch?: number
  base_model_id?: string  // 用于微调的已有模型ID
  business?: string       // 业务/算法类型（守门员按此隔离对比），默认 "general"
  // 阶段1.3（训练任务聚合）：多数据集聚合训练
  dataset_ids?: string[]      // 待聚合训练的数据集列表（缺省=仅 dataset_id）
  aggregate_incomplete?: boolean // 阶段1.4（雪球）：自动带同业务未完成训练数据集
  // 高级训练参数（可选，缺省则由系统默认）
  lr0?: number          // 初始学习率
  optimizer?: string    // auto / SGD / Adam / AdamW
  weight_decay?: number // 权重衰减
  patience?: number     // 早停轮数（0 = 关闭）
  gpu_index?: number | null // 训练节点：指定 GPU 索引（null/undefined = 自动）
  // 训练增强（1.5/1.6）：样本池抽样 + 回忆集混训
  hard_sample_ratio?: number      // 困难样本库按比例抽样并入训练（0=关闭，默认 0.1）
  background_sample_ratio?: number // 空白样本库抽样（负样本，默认 0.05）
  recall_enabled?: boolean        // 回忆集混训：增量训练时混入同业务旧数据防遗忘（默认 true）
}

export interface GpuInfo {
  index: number
  name: string
  total_gb: number
  used_gb: number
  free_gb: number
}

export interface GpuListResult {
  cuda_available: boolean
  gpus: GpuInfo[]
}

export interface LabelFilterInfo {
  model_code?: string
  kept: string[]
  dropped: string[]
}

export interface CreateTrainJobResult {
  job_id: string
  status: string
  label_filter?: LabelFilterInfo
  dataset_ids?: string[]
  aggregated?: boolean
  aggregation?: {
    dataset_count: number
    unified_names: string[]
    total_images: number
  }
}

export const createTrainJob = async (params: TrainJobRequest): Promise<CreateTrainJobResult> => {
  const { data } = await api.post('/train/jobs', params)
  return data
}

/**
 * 训练前确保预训练权重已本地化（镜像加速下载缓存），避免启动阶段联网下载卡顿。
 * 已缓存时立即返回（status=ready），自定义/微调模型同样秒返。
 */
export const prepareWeights = async (model_name: string) => {
  const { data } = await api.post('/train/prepare-weights', { model_name })
  return data
}

/**
 * 获取当前服务器的 GPU 列表（训练节点），无 GPU 时 cuda_available=false
 */
export const listGpus = async (): Promise<GpuListResult> => {
  const { data } = await api.get('/train/gpus')
  return data
}

// 根据数据类别名自动推断业务/算法类型（无需手动选择业务场景）
export const inferBusiness = async (dataset_id: string, version: string = 'v1') => {
  const { data } = await api.get('/train/business', {
    params: { dataset_id, version }
  })
  return data
}

/**
 * 根据当前硬件与数据集自动推荐训练参数（基础 + 高级，可手动覆盖）
 */
export const suggestTrainParams = async (
  dataset_id: string,
  version: string = 'v1',
  base_model_id?: string
) => {
  const { data } = await api.get('/train/suggest', {
    params: { dataset_id, version, base_model_id: base_model_id || undefined }
  })
  return data
}

export const listTrainJobs = async () => {
  const { data } = await api.get('/train/jobs')
  return data
}

export const getTrainJob = async (jobId: string) => {
  const { data } = await api.get(`/train/jobs/${jobId}`)
  return data
}

export const getTrainJobTree = async (jobId: string) => {
  const { data } = await api.get(`/train/jobs/${jobId}/tree`)
  return data
}

export const stopTrainJob = async (jobId: string) => {
  const { data } = await api.post(`/train/jobs/${jobId}/stop`)
  return data
}

export const resumeTrainJob = async (jobId: string) => {
  const { data } = await api.post(`/train/jobs/${jobId}/resume`)
  return data
}

export const deleteTrainJob = async (jobId: string) => {
  const { data } = await api.delete(`/train/jobs/${jobId}`)
  return data
}

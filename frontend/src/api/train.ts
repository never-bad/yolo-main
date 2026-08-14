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
  // 高级训练参数（可选，缺省则由系统默认）
  lr0?: number          // 初始学习率
  optimizer?: string    // auto / SGD / Adam / AdamW
  weight_decay?: number // 权重衰减
  patience?: number     // 早停轮数（0 = 关闭）
  gpu_index?: number | null // 训练节点：指定 GPU 索引（null/undefined = 自动）
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

export const createTrainJob = async (params: TrainJobRequest) => {
  const { data } = await api.post('/train/jobs', params)
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

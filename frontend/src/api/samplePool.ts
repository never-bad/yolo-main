import api from './axios'

export interface AddHardRequest {
  dataset_id: string
  model_id: string
  image_names: string[]
  version?: string
}

export interface AddBackgroundRequest {
  dataset_id: string
  image_names: string[]
  version?: string
}

export interface HardPoolInfo {
  model_id: string
  model_code?: string
  names: string[]
  image_count: number
  label_count: number
  updated_at: string
}

export interface SamplePoolResult {
  hard: HardPoolInfo[]
  background: number
}

/** 将某数据集的指定图片（连同标注）加入指定模型的困难样本库 */
export const addHardSamples = async (params: AddHardRequest) => {
  const { data } = await api.post('/sample-pool/hard', params)
  return data
}

/** 将某数据集的指定图片作为无目标背景加入空白样本库 */
export const addBackgroundSamples = async (params: AddBackgroundRequest) => {
  const { data } = await api.post('/sample-pool/background', params)
  return data
}
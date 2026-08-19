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

/** 汇总样本池状态：困难样本（按模型）+ 空白样本（全局） */
export const listSamplePool = async (): Promise<SamplePoolResult> => {
  const { data } = await api.get('/sample-pool')
  return data
}

/** 清空某模型的困难样本库 */
export const clearHardPool = async (modelId: string) => {
  const { data } = await api.delete(`/sample-pool/hard/${modelId}`)
  return data
}

/** 清空空白样本库 */
export const clearBackgroundPool = async () => {
  const { data } = await api.delete('/sample-pool/background')
  return data
}
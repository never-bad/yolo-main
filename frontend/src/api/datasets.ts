import api from './axios'

export interface UpdateDatasetRequest {
  description?: string
  tags?: string[]
}

export const uploadDataset = async (file: File, modelId?: string | null, modelCode?: string | null) => {
  const formData = new FormData()
  formData.append('file', file)
  const params: Record<string, string> = {}
  if (modelCode) params.model_code = modelCode
  else if (modelId) params.model_id = modelId
  const { data } = await api.post('/datasets/upload', formData, {
    params: Object.keys(params).length ? params : undefined
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

// 2.1 入料校验：手动重新执行标注格式校验
export const validateDataset = async (datasetId: string) => {
  const { data } = await api.post(`/datasets/${datasetId}/validate`)
  return data
}

// 1.7 模型仓库：绑定/解绑数据集到模型（model_id 传 null 表示解绑）
export const bindDatasetModel = async (datasetId: string, modelId: string | null) => {
  const { data } = await api.put(`/datasets/${datasetId}/model`, { model_id: modelId })
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

import api from './axios'

export interface UpdateModelRequest {
  name?: string
  description?: string
  tags?: string[]
  model_code?: string
  display_name?: string
  status?: 'active' | 'inactive' | string
}

/** 模型统一标签字典项（四字段：index / english_code / chinese_name / chinese_desc） */
export interface LabelItem {
  index: number
  english_code: string
  chinese_name?: string
  chinese_desc?: string
}

export interface LabelsDict {
  model_code?: string
  labels: LabelItem[]
}

/** 模型守门员评估报告（训练结束后自动对比新旧模型生成） */
export interface GatekeeperReport {
  result?: 'promoted' | 'rejected' | 'first_version' | string
  promoted?: boolean
  eval_split?: string       // 评估集：test（独立测试集）或 val
  new_metrics?: {
    mAP50?: number | null
    mAP50_95?: number | null
    precision?: number | null
    recall?: number | null
    speed_ms?: number | null
  }
  old_metrics?: {
    mAP50?: number | null
    mAP50_95?: number | null
    precision?: number | null
    recall?: number | null
    speed_ms?: number | null
  } | null
  class_ap?: Record<string, {
    old_ap?: number | null
    new_ap?: number | null
    delta_pct?: number | null
    regressed?: boolean
  }>
  regressed_classes?: string[]
  report?: string           // 中文诊断报告
}

export interface ModelDetails {
  model_id: string
  job_id?: string
  base_model?: string
  task?: string
  classes?: string[]
  imgsz?: number
  epochs?: number
  created_at?: string
  weights_path?: string
  name?: string
  description?: string
  tags?: string[]
  file_size?: number
  file_size_mb?: number
  // 模型仓库 / 守门员字段
  business?: string                 // 业务/算法类型
  status?: 'production_ready' | 'rejected' | 'superseded' | 'training' | 'evaluating' | string
  version?: string                  // 版本号（同业务内递增，如 v1.0 → v1.1）
  lineage?: {                       // 血缘关系
    parent_model_id?: string | null
    base_model?: string
    dataset_id?: string
    job_id?: string
  }
  gatekeeper?: GatekeeperReport
  override?: {
    from_status?: string
    to_status?: string
    operated_at?: string
    reason?: string
  }
  training_metrics?: {
    training_history?: {
      epochs?: number[]
      train_box_loss?: number[]
      train_cls_loss?: number[]
      train_dfl_loss?: number[]
      val_box_loss?: number[]
      val_cls_loss?: number[]
      val_dfl_loss?: number[]
      metrics_precision?: number[]
      metrics_recall?: number[]
      metrics_mAP50?: number[]
      metrics_mAP50_95?: number[]
      final_metrics?: {
        mAP50?: number
        mAP50_95?: number
        precision?: number
        recall?: number
      }
    }
    job_config?: {
      dataset_id?: string
      epochs?: number
      imgsz?: number
      batch?: number
      model_name?: string
      status?: string
      created_at?: string
      completed_at?: string
    }
  }
  model_info?: {
    task?: string
    model_type?: string
    total_params?: number
    trainable_params?: number
    total_params_m?: number
  }
}

export const listModels = async () => {
  const { data } = await api.get('/models')
  return data
}

// 1.7 模型仓库：归属于模型的数据集（含状态机字段）
export interface ModelDataset {
  dataset_id: string
  filename?: string
  name?: string
  stage?: string                 // collecting/annotating/sealed/training/completed/failed
  training_status?: string       // incomplete/completed 未完成训练/已完成训练
  status?: string                // uploaded/prepared 等（待封板判定用）
  image_count?: number
  annotated?: boolean
  sealed_at?: string | null
  trained_at?: string | null
  model_id?: string | null
  classes?: string[]
  label_count?: number
  label_validation?: {
    status?: 'ok' | 'failed'
    checked_files?: number
    lines_checked?: number
    invalid_files?: number
    invalid_lines?: number
    errors?: string[]
    checked_at?: string
  } | null
  uploaded_at?: string
}

export const getModelDatasets = async (modelId: string): Promise<{ model_id: string; datasets: ModelDataset[] }> => {
  const { data } = await api.get(`/models/${modelId}/datasets`)
  return data
}

export const getModel = async (modelId: string): Promise<ModelDetails> => {
  const { data } = await api.get(`/models/${modelId}`)
  return data
}

export const updateModel = async (modelId: string, request: UpdateModelRequest) => {
  const { data } = await api.put(`/models/${modelId}`, request)
  return data
}

// 阶段0：统一标签字典（四字段 CRUD）
export const getLabelsDict = async (modelId: string): Promise<LabelsDict> => {
  const { data } = await api.get(`/models/${modelId}/labels`)
  return data
}

export const updateLabelsDict = async (modelId: string, labels: LabelItem[]): Promise<LabelsDict> => {
  const { data } = await api.put(`/models/${modelId}/labels`, { labels })
  return data
}

export interface LabelSuggestion {
  english_code: string
  chinese_name: string
  chinese_desc: string
  images: string[]
}

export const suggestModelLabels = async (modelId: string, datasetId: string, limit = 3) => {
  const { data } = await api.post(`/models/${modelId}/labels/suggest`, { dataset_id: datasetId, limit })
  return data as { ok: boolean; message?: string; candidates?: LabelSuggestion[] }
}

// 采纳 AI 建议的新标签：追加到模型标签字典末尾（跳过已存在项）
export const adoptSuggestedLabels = async (modelId: string, suggestions: { english_code: string; chinese_name: string; chinese_desc: string }[]) => {
  const { data } = await api.post(`/models/${modelId}/labels/suggest/adopt`, { suggestions })
  return data as { added: string[]; skipped: string[] }
}

export const deleteModel = async (modelId: string) => {
  const { data } = await api.delete(`/models/${modelId}`)
  return data
}

// ===== 2.7 相似模型排查与合并（人工工具） =====
export interface SimilarPair {
  a: { model_id: string; name?: string; code?: string; dataset_count: number; empty?: boolean; status?: string }
  b: { model_id: string; name?: string; code?: string; dataset_count: number; empty?: boolean; status?: string }
  similarity: number
  common_classes: string[]
  suggested_main: string
  suggested_main_datasets: number
  sub_datasets: number
}

export const findSimilarModels = async (minSimilarity: number = 0.5) => {
  const { data } = await api.get('/models/similar-scan', { params: { min_similarity: minSimilarity } })
  return data
}

export const mergeModels = async (mainModelId: string, mergedModelIds: string[], reason?: string) => {
  const { data } = await api.post('/models/merge', {
    main_model_id: mainModelId,
    merged_model_ids: mergedModelIds,
    reason: reason || undefined
  })
  return data
}

export const getMergeLogs = async (limit: number = 20) => {
  const { data } = await api.get('/models/merge-log', { params: { limit } })
  return data
}

export const rollbackMerge = async (logIndex: number = -1) => {
  const { data } = await api.post('/models/rollback-merge', { log_index: logIndex })
  return data
}

// 列出当前在役的生产模型（可按业务/算法类型过滤）
export const listProductionModels = async (business?: string) => {
  const { data } = await api.get('/models/production', {
    params: { business: business || undefined }
  })
  return data
}

// 人工强制覆盖（Override）：将守门员拦截的模型强制设为生产版本
export const overrideModel = async (
  modelId: string,
  options?: { business?: string; reason?: string }
) => {
  const { data } = await api.post(`/models/${modelId}/override`, options || {})
  return data
}

// 将训练好的模型升级为 AI 预标注的检测模型（热切换）
export const promoteToDetector = async (modelId: string) => {
  const { data } = await api.post(`/models/${modelId}/promote-to-detector`)
  return data
}

export const uploadModel = async (file: File) => {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post('/models/upload', formData)
  return data
}

// 上传自定义 .pt 预训练模型
export const uploadPretrainedPt = async (file: File) => {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post('/models/upload-pt', formData)
  return data
}

// 列出已上传的自定义预训练模型
export const listCustomModels = async () => {
  const { data } = await api.get('/models/custom')
  return data
}

export const exportModel = async (modelId: string): Promise<Blob> => {
  const response = await api.get(`/models/${modelId}/export`, {
    responseType: 'blob'
  })
  return response.data
}

export const generateTrainingCharts = async (
  modelId: string, 
  chartType: 'loss' | 'metrics' | 'all' = 'all'
): Promise<Blob> => {
  const response = await api.get(`/models/${modelId}/charts`, {
    params: { chart_type: chartType },
    responseType: 'blob'
  })
  return response.data
}

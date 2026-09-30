import api from './axios'

export interface UpdateModelRequest {
  name?: string
  description?: string
  tags?: string[]
  model_code?: string
  display_name?: string
  status?: 'active' | 'inactive' | string
  model_library_id?: string | null  // 归入算法库；传空/null 表示移出库（降级为独立模型）
}

/** 模型统一标签字典项（五字段：index / english_code / chinese_name / chinese_desc / ai_prompt_en） */
export interface LabelItem {
  index: number
  english_code: string
  chinese_name?: string
  chinese_desc?: string
  /** 英文检测提示词：告诉 GroundingDINO / YOLO-World 这个类别在画面里长什么样（只认英文） */
  ai_prompt_en?: string
}

export interface LabelsDict {
  model_code?: string
  labels: LabelItem[]
}

/** 标签引用统计：判断每个类别"有没有被数据用过"，用过的英文名锁死不可改 */
export interface LabelUsageItem {
  index: number
  english_code: string
  images: number
  datasets: number
  in_use: boolean
  /** 该行是否被锁死（英文名不可改、不可删）：在用的那一行及其之前的行 */
  locked: boolean
}

export interface LabelsUsage {
  model_id: string
  library_root_id: string
  dict_locked: boolean
  /** 最后一个"在用"类别所在行号；-1 表示本库还没有任何数据用到标签，字典可自由编辑 */
  frozen_until: number
  labels: LabelUsageItem[]
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
  display_name?: string         // 算法库显示名（新建时人工指定，可改）
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

// ===== 算法库（model_libraries：base_algorithm / 标签字典 / 当前生产模型指针） =====
export interface ModelLibrary {
  model_library_id: string
  display_name?: string
  description?: string
  business_scenario?: string
  base_algorithm?: string
  architecture_family?: string
  label_dict_id?: string | null
  current_production_model_id?: string | null
  production_model_name?: string    // 附加字段：该库当前生产模型名
  model_count?: number              // 附加字段：库内挂接模型数
  created_at?: string
  updated_at?: string
}

export const listLibraries = async (): Promise<{ libraries: ModelLibrary[] }> => {
  const { data } = await api.get('/models/libraries')
  return data
}

export const createLibrary = async (req: Partial<ModelLibrary>): Promise<ModelLibrary> => {
  const { data } = await api.post('/models/libraries', req)
  return data
}

export const deleteLibrary = async (libraryId: string) => {
  const { data } = await api.delete(`/models/libraries/${libraryId}`)
  return data
}

// 首页「+ 新建算法库」：创建空白模型（空标签字典 + 采集库），code 已存在则返回现有模型
export const createModel = async (req: {
  display_name: string
  model_code: string
  business?: string
  description?: string
}) => {
  const { data } = await api.post('/models', req)
  return data
}

// 1.7 模型仓库：归属于模型的数据集（含状态机字段）
export interface ModelDataset {
  dataset_id: string
  filename?: string
  name?: string
  display_name?: string        // 数据集显示名（展示优先于 filename）
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

/** 标签引用统计（弹窗打开时拉一次：决定哪些英文名锁死、显示已用/未用） */
export const getLabelsUsage = async (modelId: string): Promise<LabelsUsage> => {
  const { data } = await api.get(`/models/${modelId}/labels/usage`)
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

/** 标签文案同步结果：把标签字典的中文名/描述刷进本库已标注数据 */
export interface LabelSyncResult {
  model_id: string
  dry_run: boolean
  stats: {
    datasets: number
    files_scanned: number
    files_changed: number
    boxes_changed: number
  }
  samples: {
    dataset_id: string
    image_id: string
    english_code: string
    field: string
    before: string
    after: string
  }[]
  backup_id: string | null
  last_backup: { backup_id?: string; at?: string; files?: number; boxes?: number } | null
}

export const syncLabelsToAnnotations = async (modelId: string, dryRun: boolean): Promise<LabelSyncResult> => {
  const { data } = await api.post(`/models/${modelId}/labels/sync-annotations`, { dry_run: dryRun })
  return data
}

export const rollbackLabelSync = async (modelId: string) => {
  const { data } = await api.post(`/models/${modelId}/labels/sync-annotations/rollback`)
  return data as { model_id: string; restored_files: number; backup_id: string }
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

// ===== P8 半自动发布（守门员通过 → 待确认发布 → 管理员确认后为唯一生产版本） =====
export interface PendingPublishModel extends ModelDetails {
  publish_status?: 'pending' | 'published' | string   // pending=待确认 published=已确认
  ready_at?: string                                   // 守门员通过(待确认)时间，用于冷却期判定
  published_at?: string                               // 确认发布时间
  migration?: {                                       // 2.4 迁移训练产出标记
    enabled?: boolean
    source?: string
    target?: string
  } | null
}

export const listPendingPublish = async (business?: string): Promise<{ models: PendingPublishModel[] }> => {
  const { data } = await api.get('/models/publish/pending', {
    params: { business: business || undefined }
  })
  return data
}

export const confirmPublish = async (
  modelId: string,
  options?: { force?: boolean }
): Promise<{ meta: PendingPublishModel; superseded: string[] }> => {
  const { data } = await api.post(`/models/publish/${modelId}`, options || {})
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

export const uploadModel = async (file: File, name?: string, libraryId?: string) => {
  const formData = new FormData()
  formData.append('file', file)
  if (name && name.trim()) formData.append('name', name.trim())
  if (libraryId && libraryId.trim()) formData.append('library_id', libraryId.trim())
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

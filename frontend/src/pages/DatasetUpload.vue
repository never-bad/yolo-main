<template>
  <div class="dataset-upload">
    <div class="card">
      <h2>上传数据集</h2>
      <div class="form-group">
        <label>选择 ZIP 文件</label>
        <div v-if="selectedFile" class="selected-file">
          <span class="selected-file-icon">📦</span>
          <span class="selected-file-name">{{ selectedFile.name }}</span>
          <span class="selected-file-size">({{ (selectedFile.size / 1024 / 1024).toFixed(2) }} MB)</span>
        </div>
        <div v-else class="selected-file empty">尚未选择文件</div>
        <div class="upload-row">
          <input type="file" accept=".zip" id="datasetZip" @change="handleFileSelect" :disabled="uploading" class="file-input" />
          <label for="datasetZip" class="file-btn" :disabled="uploading">选择文件</label>
          <button class="upload-btn" @click="uploadFile" :disabled="!selectedFile || uploading">
            <span v-if="uploading" class="loading-spinner"></span>
            {{ uploading ? '上传中...' : '上传' }}
          </button>
        </div>
      </div>
      <div v-if="uploadResult" class="result">
        <p>✓ 上传成功: {{ uploadResult.dataset_id }}</p>
        <button @click="prepareDataset" :disabled="preparingNew">
          <span v-if="preparingNew" class="loading-spinner"></span>
          {{ preparingNew ? '准备中...' : '准备数据集' }}
        </button>
      </div>
    </div>

    <div class="card">
      <h2>数据集列表</h2>
      <button @click="loadDatasets" :disabled="loading" class="secondary">
        <span v-if="loading" class="loading-spinner"></span>
        {{ loading ? '加载中...' : '刷新' }}
      </button>
      <div v-if="loading" class="loading-state">
        <span class="loading-spinner large"></span>
        <span>加载数据集列表...</span>
      </div>
      <div v-else class="dataset-list">
        <div v-for="dataset in datasets" :key="dataset.dataset_id" class="dataset-item">
          <div class="dataset-header">
            <h3 class="dataset-name" @click="viewDataset(dataset)" title="点击查看数据集文件夹结构">{{ dataset.dataset_id }}</h3>
            <div class="dataset-actions">
              <button 
                v-if="dataset.status === 'uploaded'" 
                @click="prepareDatasetFromList(dataset.dataset_id)" 
                :disabled="preparingDataset === dataset.dataset_id"
              >
                <span v-if="preparingDataset === dataset.dataset_id" class="loading-spinner"></span>
                {{ preparingDataset === dataset.dataset_id ? '准备中...' : '准备' }}
              </button>
              <button 
                @click="viewDataset(dataset)" 
                class="secondary"
                title="查看数据集图片"
              >查看</button>
              <button 
                @click="editDataset(dataset)" 
                class="secondary"
                :disabled="editingDataset === dataset.dataset_id"
              >
                <span v-if="editingDataset === dataset.dataset_id" class="loading-spinner"></span>
                {{ editingDataset === dataset.dataset_id ? '保存中...' : '编辑' }}
              </button>
              <button 
                v-if="dataset.status === 'prepared'"
                @click="exportAnnotated(dataset.dataset_id)" 
                class="secondary"
                :disabled="exportingDataset === dataset.dataset_id + '_annotated'"
              >
                <span v-if="exportingDataset === dataset.dataset_id + '_annotated'" class="loading-spinner"></span>
                {{ exportingDataset === dataset.dataset_id + '_annotated' ? '导出中...' : '导出（标注后）' }}
              </button>
              <button 
                v-if="dataset.status === 'prepared'"
                @click="exportOriginal(dataset.dataset_id)" 
                class="secondary"
                :disabled="exportingDataset === dataset.dataset_id + '_original'"
              >
                <span v-if="exportingDataset === dataset.dataset_id + '_original'" class="loading-spinner"></span>
                {{ exportingDataset === dataset.dataset_id + '_original' ? '导出中...' : '导出（标注前）' }}
              </button>
              <button 
                @click="deleteDatasetItem(dataset.dataset_id)" 
                class="danger"
                :disabled="deletingDataset === dataset.dataset_id"
              >
                <span v-if="deletingDataset === dataset.dataset_id" class="loading-spinner"></span>
                {{ deletingDataset === dataset.dataset_id ? '删除中...' : '删除' }}
              </button>
            </div>
          </div>
          <p>文件名: {{ dataset.filename }}</p>
          <p>大小: {{ (dataset.size / 1024 / 1024).toFixed(2) }} MB</p>
          <p>状态: <span :class="'status-badge status-' + dataset.status">{{ dataset.status }}</span></p>
          <p v-if="dataset.image_count">图片数: {{ dataset.image_count }}</p>
          <p v-if="dataset.label_count">标签数: {{ dataset.label_count }}</p>
          <p v-if="dataset.classes">类别: {{ dataset.classes.join(', ') }}</p>
          <p v-if="dataset.description">描述: {{ dataset.description }}</p>
          <p v-if="dataset.tags && dataset.tags.length">标签: {{ dataset.tags.join(', ') }}</p>
        </div>
        <div v-if="datasets.length === 0" class="empty-state">
          暂无数据集，请上传数据集
        </div>
      </div>
    </div>

    <!-- 数据集查看弹窗（统计信息 + 图片预览） -->
    <div v-if="viewingDataset" class="viewer-mask" @click.self="closeViewer">
      <div class="viewer-dialog">
        <div class="viewer-header">
          <h3>{{ viewingDataset.dataset_id }}</h3>
          <button class="secondary viewer-close" @click="closeViewer">关闭</button>
        </div>
        <div v-if="viewerLoading" class="loading-state">
          <span class="loading-spinner large"></span>
          <span>加载数据集详情...</span>
        </div>
        <template v-else>
          <!-- 统计信息 -->
          <div class="viewer-stats">
            <div class="stat-item">
              <span class="stat-label">状态</span>
              <span class="stat-value">{{ viewerDetail?.status || viewingDataset.status }}</span>
            </div>
            <div class="stat-item" v-if="viewerDetail?.image_count || viewingDataset.image_count">
              <span class="stat-label">图片数</span>
              <span class="stat-value">{{ viewerDetail?.image_count || viewingDataset.image_count }}</span>
            </div>
            <div class="stat-item" v-if="viewerDetail?.label_count || viewingDataset.label_count">
              <span class="stat-label">标注数</span>
              <span class="stat-value">{{ viewerDetail?.label_count || viewingDataset.label_count }}</span>
            </div>
            <div class="stat-item" v-if="viewerDetail?.images">
              <span class="stat-label">预览图</span>
              <span class="stat-value">{{ viewerDetail.images.length }} 张</span>
            </div>
            <div class="stat-item" v-if="viewerDetail?.prepared_at">
              <span class="stat-label">准备时间</span>
              <span class="stat-value">{{ formatTime(viewerDetail.prepared_at) }}</span>
            </div>
          </div>
          <div class="viewer-classes" v-if="viewerDetail?.classes?.length || viewingDataset.classes?.length">
            <span class="stat-label">类别:</span>
            <span class="viewer-class-tags">
              <span v-for="(c, i) in (viewerDetail?.classes || viewingDataset.classes)" :key="i" class="class-tag">{{ c }}</span>
            </span>
          </div>
          <div class="viewer-description" v-if="viewerDetail?.description || viewingDataset.description">
            {{ viewerDetail?.description || viewingDataset.description }}
          </div>

          <!-- 文件夹结构 -->
          <div class="viewer-section-title">文件夹结构</div>
          <div v-if="treeLoading" class="loading-state">
            <span class="loading-spinner large"></span>
            <span>加载文件夹结构...</span>
          </div>
          <div v-else-if="treeData" class="viewer-tree">
            <FolderTree
              :nodes="treeData.children"
              :depth="0"
              :expanded="treeExpanded"
              @toggle-dir="toggleDir"
              @open-file="openFile"
            />
          </div>
          <div v-else class="empty-state">暂无文件夹信息</div>

          <!-- 图片预览 -->
          <div class="viewer-section-title">图片预览</div>
          <div v-if="viewerImages.length === 0" class="empty-state">
            该数据集尚无图片（或未执行"准备"操作）
          </div>
          <div class="viewer-grid" v-else>
            <div v-for="img in viewerImages" :key="img" class="viewer-thumb">
              <img :src="'/static/' + img" loading="lazy" @error="handleImgError" />
            </div>
          </div>
        </template>
      </div>
    </div>

    <!-- 文件内容预览弹窗 -->
    <div v-if="filePreview" class="file-preview-mask" @click.self="closeFilePreview">
      <div class="file-preview-dialog">
        <div class="viewer-header">
          <h3>{{ filePreview.name }}</h3>
          <button class="secondary viewer-close" @click="closeFilePreview">关闭</button>
        </div>
        <div v-if="filePreviewIsImage" class="file-preview-img-wrap">
          <img :src="'/static/' + filePreview.path" @error="handleImgError" />
        </div>
        <pre v-else class="file-preview-text">{{ filePreviewContent || '加载中...' }}</pre>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { uploadDataset, prepareDataset as prepareDst, listDatasets, getDataset, getDatasetTree, updateDataset, deleteDataset, exportAnnotatedDataset, exportOriginalDataset } from '@/api/datasets'
import { downloadFile } from '@/utils/download'
import { showConfirm, showPrompt } from '@/composables/useDialog'
import FolderTree from '@/components/FolderTree.vue'

const selectedFile = ref<File | null>(null)
const uploading = ref(false)
const uploadResult = ref<any>(null)
const datasets = ref<any[]>([])
const loading = ref(false)
const preparingDataset = ref<string | null>(null)  // 正在准备的数据集ID（列表中）
const preparingNew = ref(false)  // 正在准备新上传的数据集
const editingDataset = ref<string | null>(null)  // 正在编辑的数据集ID
const deletingDataset = ref<string | null>(null)  // 正在删除的数据集ID
const exportingDataset = ref<string | null>(null)  // 正在导出的数据集ID
const viewingDataset = ref<any | null>(null)  // 正在预览的数据集
const viewerImages = ref<string[]>([])  // 预览图片列表
const viewerLoading = ref(false)  // 预览加载状态
const viewerDetail = ref<any | null>(null)  // 数据集详情（统计信息）
const treeData = ref<any | null>(null)  // 文件夹结构树
const treeLoading = ref(false)  // 文件夹结构加载状态
const treeExpanded = reactive(new Set<string>())  // 已展开的目录节点
const filePreview = ref<any | null>(null)  // 正在预览的文件
const filePreviewIsImage = ref(false)  // 预览的是否为图片
const filePreviewContent = ref('')  // 文本文件内容

// 打开数据集查看（统计信息 + 图片预览）
const viewDataset = async (dataset: any) => {
  if (dataset.status !== 'prepared') {
    alert('该数据集尚未"准备"，无法查看。请先执行"准备"操作。')
    return
  }
  viewingDataset.value = dataset
  viewerLoading.value = true
  viewerImages.value = []
  viewerDetail.value = null
  try {
    const result = await getDataset(dataset.dataset_id)
    viewerDetail.value = result
    viewerImages.value = result.images || []
  } catch (e: any) {
    alert('加载数据集详情失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    viewerLoading.value = false
  }
  loadTree(dataset.dataset_id)
}

// 关闭预览
const closeViewer = () => {
  viewingDataset.value = null
  viewerImages.value = []
  viewerDetail.value = null
  treeData.value = null
  treeExpanded.clear()
  closeFilePreview()
}

// 加载数据集文件夹结构
const loadTree = async (datasetId: string) => {
  treeLoading.value = true
  treeData.value = null
  treeExpanded.clear()
  try {
    const result = await getDatasetTree(datasetId)
    treeData.value = result.tree
  } catch (e) {
    treeData.value = null
  } finally {
    treeLoading.value = false
  }
}

// 展开/折叠目录
const toggleDir = (node: any) => {
  if (treeExpanded.has(node.path)) treeExpanded.delete(node.path)
  else treeExpanded.add(node.path)
}

const FILE_IMG_EXTS = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']

// 点击文件：图片直接预览，文本读取内容
const openFile = async (node: any) => {
  filePreview.value = node
  filePreviewContent.value = ''
  filePreviewIsImage.value = FILE_IMG_EXTS.includes((node.ext || '').toLowerCase())
  if (filePreviewIsImage.value) return
  try {
    const res = await fetch('/static/' + node.path)
    if (!res.ok) throw new Error('HTTP ' + res.status)
    filePreviewContent.value = await res.text()
  } catch (e: any) {
    filePreviewContent.value = '无法读取该文件：' + (e.message || e)
  }
}

// 关闭文件预览
const closeFilePreview = () => {
  filePreview.value = null
  filePreviewContent.value = ''
}

// 图片加载失败时的兜底（隐藏占位）
const handleImgError = (e: any) => {
  const el = e.target as HTMLImageElement
  el.style.display = 'none'
}

// 时间格式化
const formatTime = (t: any) => {
  if (!t) return ''
  const d = new Date(t)
  if (isNaN(d.getTime())) return String(t).slice(0, 19).replace('T', ' ')
  return d.toLocaleString('zh-CN')
}

const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files && target.files[0]) {
    selectedFile.value = target.files[0]
    uploadResult.value = null
  }
}

const uploadFile = async () => {
  if (!selectedFile.value) return
  
  uploading.value = true
  try {
    const result = await uploadDataset(selectedFile.value)
    uploadResult.value = result
    alert('上传成功!')
    loadDatasets()
  } catch (error: any) {
    alert('上传失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    uploading.value = false
  }
}

const prepareDataset = async () => {
  if (!uploadResult.value) return
  
  preparingNew.value = true
  try {
    await prepareDst(uploadResult.value.dataset_id)
    alert('准备成功!')
    uploadResult.value = null
    loadDatasets()
  } catch (error: any) {
    alert('准备失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    preparingNew.value = false
  }
}

const prepareDatasetFromList = async (datasetId: string) => {
  preparingDataset.value = datasetId
  try {
    await prepareDst(datasetId)
    alert('准备成功!')
    loadDatasets()
  } catch (error: any) {
    alert('准备失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    preparingDataset.value = null
  }
}

const loadDatasets = async () => {
  loading.value = true
  try {
    const result = await listDatasets()
    datasets.value = result.datasets
  } catch (error: any) {
    alert('加载失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    loading.value = false
  }
}

const editDataset = async (dataset: any) => {
  const description = await showPrompt('请输入数据集描述（可选）:', dataset.description || '')
  if (description === null) return
  
  const tagsInput = await showPrompt('请输入标签（逗号分隔，可选）:', dataset.tags ? dataset.tags.join(',') : '')
  const tags = tagsInput ? tagsInput.split(',').map(t => t.trim()).filter(t => t) : undefined
  
  editingDataset.value = dataset.dataset_id
  try {
    await updateDataset(dataset.dataset_id, {
      description: description || undefined,
      tags: tags
    })
    alert('更新成功!')
    loadDatasets()
  } catch (error: any) {
    alert('更新失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    editingDataset.value = null
  }
}

const deleteDatasetItem = async (datasetId: string) => {
  if (!(await showConfirm(`确定要删除数据集 ${datasetId} 吗？此操作不可恢复！`))) return
  
  deletingDataset.value = datasetId
  try {
    await deleteDataset(datasetId)
    alert('删除成功!')
    loadDatasets()
  } catch (error: any) {
    alert('删除失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    deletingDataset.value = null
  }
}

// 导出标注后的数据集
const exportAnnotated = async (datasetId: string) => {
  exportingDataset.value = datasetId + '_annotated'
  try {
    const blob = await exportAnnotatedDataset(datasetId)
    downloadFile(blob, `${datasetId}_annotated.zip`)
    alert('导出成功!')
  } catch (error: any) {
    alert('导出失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    exportingDataset.value = null
  }
}

// 导出标注前的数据集
const exportOriginal = async (datasetId: string) => {
  exportingDataset.value = datasetId + '_original'
  try {
    const blob = await exportOriginalDataset(datasetId)
    downloadFile(blob, `${datasetId}_original.zip`)
    alert('导出成功!')
  } catch (error: any) {
    alert('导出失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    exportingDataset.value = null
  }
}

onMounted(() => {
  loadDatasets()
})
</script>

<style scoped>
.upload-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-top: 0.25rem;
}

.file-input {
  display: none;
}

.file-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 38px;
  padding: 0 1rem;
  background: #eef0f3;
  border: 1px solid #d0d3d8;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.95rem;
  color: #333;
  white-space: nowrap;
}

.file-btn:hover {
  background: #e2e5ea;
}

.upload-btn {
  background: #8e44ad;
  color: white;
  border: none;
  height: 38px;
  padding: 0 1.2rem;
  border-radius: 4px;
  cursor: pointer;
  font-weight: bold;
  font-size: 0.95rem;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  box-shadow: 0 2px 4px rgba(142, 68, 173, 0.3);
}

.upload-btn:disabled {
  background: #b9a0c9;
  cursor: not-allowed;
}

.selected-file {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
  padding: 0.75rem 1rem;
  background: #eaf4fb;
  border: 1px solid #3498db;
  border-radius: 6px;
  font-size: 1rem;
  color: #1a5276;
}

.selected-file.empty {
  background: #f8f9fa;
  border: 1px dashed #c0c4cc;
  color: #9aa0a6;
}

.selected-file-icon {
  font-size: 1.25rem;
}

.selected-file-name {
  font-weight: bold;
  color: #2c3e50;
  word-break: break-all;
}

.selected-file-size {
  color: #7f8c8d;
  white-space: nowrap;
}

.result {
  margin-top: 1rem;
  padding: 1rem;
  background: #d5f4e6;
  border-radius: 4px;
}

.dataset-list {
  margin-top: 1rem;
  display: grid;
  gap: 1rem;
}

.dataset-item {
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.dataset-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.dataset-header h3 {
  margin: 0;
  color: #2c3e50;
}

.dataset-actions {
  display: flex;
  gap: 0.5rem;
}

.dataset-actions button {
  padding: 0.25rem 0.75rem;
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.dataset-item p {
  margin: 0.25rem 0;
  color: #7f8c8d;
}

.loading-state {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 2rem;
  color: #7f8c8d;
}

.empty-state {
  padding: 2rem;
  text-align: center;
  color: #7f8c8d;
  background: #f8f9fa;
  border-radius: 4px;
}

button {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}

/* 图片预览弹窗 */
.viewer-mask {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 1.5rem;
}

.viewer-dialog {
  background: white;
  border-radius: 8px;
  width: 90%;
  max-width: 1100px;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.viewer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.25rem;
  border-bottom: 1px solid #eee;
}

.viewer-header h3 {
  margin: 0;
  color: #2c3e50;
}

.viewer-close {
  padding: 0.35rem 0.9rem;
  font-size: 0.875rem;
}

.viewer-stats {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 0.75rem 1.25rem;
  padding: 1rem 1.25rem 0.5rem;
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}

.stat-label {
  font-size: 0.75rem;
  color: #95a5a6;
}

.stat-value {
  font-size: 0.95rem;
  color: #2c3e50;
  font-weight: 600;
}

.viewer-classes {
  padding: 0.5rem 1.25rem;
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.viewer-class-tags {
  display: flex;
  gap: 0.35rem;
  flex-wrap: wrap;
}

.class-tag {
  background: #eef4fb;
  color: #2c6fbb;
  border-radius: 12px;
  padding: 0.15rem 0.6rem;
  font-size: 0.8rem;
}

.viewer-description {
  padding: 0.25rem 1.25rem;
  color: #7f8c8d;
  font-size: 0.85rem;
}

.viewer-section-title {
  padding: 0.75rem 1.25rem 0.25rem;
  font-weight: 600;
  color: #2c3e50;
  border-top: 1px solid #f0f0f0;
  margin-top: 0.5rem;
}

.viewer-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 0.75rem;
  padding: 0.75rem 1.25rem 1.25rem;
  overflow-y: auto;
}

.viewer-thumb {
  aspect-ratio: 1;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid #e0e0e0;
  background: #f5f5f5;
}

.viewer-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

/* 数据集名称可点击 */
.dataset-name {
  cursor: pointer;
}
.dataset-name:hover {
  color: #2c6fbb;
  text-decoration: underline;
}

/* 文件夹结构树 */
.viewer-tree {
  max-height: 400px;
  overflow-y: auto;
  padding: 0.5rem 1.25rem 0.75rem;
  border-bottom: 1px solid #f0f0f0;
}

/* 文件内容预览弹窗 */
.file-preview-mask {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1100;
  padding: 1.5rem;
}

.file-preview-dialog {
  background: white;
  border-radius: 8px;
  width: 80%;
  max-width: 900px;
  max-height: 88vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.file-preview-img-wrap {
  flex: 1;
  overflow: auto;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f5f5f5;
  padding: 1rem;
}

.file-preview-img-wrap img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.file-preview-text {
  flex: 1;
  overflow: auto;
  margin: 0;
  padding: 1rem 1.25rem;
  font-size: 0.8rem;
  line-height: 1.5;
  background: #fafbfc;
  white-space: pre-wrap;
  word-break: break-all;
  color: #2c3e50;
}
</style>

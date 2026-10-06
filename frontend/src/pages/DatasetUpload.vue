<template>
  <div class="dataset-upload">
    <div class="card">
      <h2>上传数据集</h2>
      <div class="form-group">
        <label>选择压缩包文件（支持 .zip / .tar / .tar.gz / .tgz）</label>
        <small class="form-hint">
          内置标准化引擎：VOC(XML) / COCO(JSON) 标注自动转换 YOLO 格式；文件夹结构不统一时自动扫描重组（图片与同名标签自动配对）。
          仅原图无标注时直接入池（status=unlabeled），随后到标注页画框。
        </small>
        <div v-if="selectedFile" class="selected-file">
          <span class="selected-file-icon">📦</span>
          <span class="selected-file-name">{{ selectedFile.name }}</span>
          <span class="selected-file-size">({{ (selectedFile.size / 1024 / 1024).toFixed(2) }} MB)</span>
        </div>
        <div v-else class="selected-file empty">尚未选择文件</div>
        <div class="upload-row">
          <input type="file" accept=".zip,.tar,.tgz,.tar.gz" id="datasetZip" @change="handleFileSelect" :disabled="uploading" class="file-input" />
          <label for="datasetZip" class="file-btn" :disabled="uploading">选择文件</label>
          <button class="upload-btn" @click="uploadFile" :disabled="!selectedFile || uploading || !uploadLibraryId">
            <span v-if="uploading" class="loading-spinner"></span>
            {{ uploading ? '上传中...' : '上传' }}
          </button>
        </div>
      </div>
      <div class="form-group">
        <label>归属算法库 <i class="required">*</i></label>
        <div class="library-row">
          <select v-model="uploadLibraryId" class="model-code-input" :class="{ invalid: libraryTouched && !uploadLibraryId }">
            <option value="">— 请选择该数据集归属的算法库 —</option>
            <option v-for="lib in libraryOptions" :key="lib.model_id" :value="lib.model_id">
              {{ lib.display_name || lib.model_code || lib.model_id }}{{ lib.dataset_count ? `（已挂 ${lib.dataset_count} 个数据集）` : '' }}
            </option>
          </select>
          <button class="secondary" @click="createLibraryOpen = true" title="没有合适的算法库？先新建一个再归属">+ 新建算法库</button>
        </div>
        <small v-if="libraryTouched && !uploadLibraryId" class="form-hint error-hint">数据集必须归属一个算法库，请先选择；没有合适的库请点「+ 新建算法库」</small>
        <small v-else class="form-hint">归属决定训练产出的模型进入哪个算法库；导入后可在数据集列表里改归属</small>
      </div>
      <div v-if="uploadResult" class="result">
        <p>✓ 上传成功: {{ uploadResult.dataset_id }}<span class="auto-create-tag">（已归属算法库 {{ libraryName(uploadLibraryId) }}）</span></p>
        <p v-if="preparingNew" class="auto-prepare-tip">
          <span class="loading-spinner"></span>正在自动准备数据集（解压/重组/统计），请稍候...
        </p>
      </div>
    </div>

    <div class="card">
      <h2>数据集列表</h2>
      <button @click="loadDatasets()" :disabled="loading" class="secondary">
        <span v-if="loading" class="loading-spinner"></span>
        {{ loading ? '加载中...' : '刷新' }}
      </button>
      <div v-if="bucketNotice" class="bucket-notice" @click="bucketNotice = ''" title="点击关闭">{{ bucketNotice }}<span class="bucket-notice-close">✕</span></div>
      <div v-if="bucketBars.length" class="bucket-progress">
        <div class="bucket-progress-title">桶进度（每桶满 {{ sealMinImages }} 张自动封板，标完的 A 包自动灌入）</div>
        <div v-for="bar in bucketBars" :key="bar.model_id" class="bucket-model">
          <span class="bucket-model-tag">{{ modelName(bar.model_id) }}</span>
          <span v-for="s in bar.sealed" :key="s.version" class="bucket-sealed" :title="'已封板 ' + s.images + ' 张（' + formatTime(s.sealed_at) + '）'">B{{ String(s.version).replace('v', '') }} 已封板</span>
          <span class="bucket-recv">
            接收 B{{ bar.curIndex }}：
            <span class="bar"><span class="bar-fill" :style="{ width: bar.pct + '%' }"></span></span>
            <span class="bar-num">{{ bar.poured }} / {{ bar.cap || '--' }}</span>
          </span>
        </div>
      </div>
      <div v-if="loading" class="loading-state">
        <span class="loading-spinner large"></span>
        <span>加载数据集列表...</span>
      </div>
      <div v-else class="dataset-list">
        <!-- 数据池 ↔ 训练仓库分离（数据模块规范 §3）：未封板数据在池中可编辑，封板后为只读快照 -->
        <div v-for="section in datasetSections" :key="section.key" class="ds-section">
          <div class="ds-section-head">
            <span class="ds-section-title">{{ section.title }}</span>
            <small class="ds-section-desc">{{ section.desc }}</small>
            <span class="ds-section-count">{{ section.items.length }} 个</span>
          </div>
          <div v-if="!section.items.length" class="empty-state mini">{{ section.emptyText }}</div>
          <div v-for="dataset in section.items" :key="dataset.dataset_id" class="dataset-item" :data-ds="dataset.dataset_id">
          <div class="dataset-header">
            <h3 class="dataset-name" @click="viewDataset(dataset)" title="点击查看数据集文件夹结构">
              {{ dataset.display_name || dataset.dataset_id }}
              <span v-if="bucketBadge(dataset).text" :class="'badge ' + bucketBadge(dataset).cls">{{ bucketBadge(dataset).text }}</span>
            </h3>
            <div class="dataset-actions">
              <button 
                @click="viewDataset(dataset)" 
                class="secondary"
                title="查看数据集图片"
              >查看</button>
              <button 
                v-if="dataset.status === 'prepared'"
                @click="goAnnotate(dataset)"
                class="secondary"
                title="打开该数据集的标注任务（无任务自动创建，直达标注页）"
              >去标注</button>
              <button
                v-if="canApprove(dataset)"
                @click="approveItem(dataset)"
                class="primary"
                :disabled="approvingDataset === dataset.dataset_id"
                title="人工审核确认标注无误后，该包才允许自动灌桶/封板"
              >
                <span v-if="approvingDataset === dataset.dataset_id" class="loading-spinner"></span>
                {{ approvingDataset === dataset.dataset_id ? '确认中...' : '审核通过' }}
              </button>
              <button
                v-if="canUnpour(dataset)"
                @click="unpourItem(dataset)"
                class="secondary"
                :disabled="unpouringDataset === dataset.dataset_id"
                title="撤回该包：把已灌入接收桶但尚未封板的图片退回，恢复可编辑"
              >
                <span v-if="unpouringDataset === dataset.dataset_id" class="loading-spinner"></span>
                {{ unpouringDataset === dataset.dataset_id ? '撤回中...' : '撤回' }}
              </button>
              <button 
                v-if="canSeal(dataset)"
                @click="sealDatasetItem(dataset)" 
                class="primary"
                :disabled="sealingDataset === dataset.dataset_id"
                title="封板后数据集只读，等待进入训练队列"
              >
                <span v-if="sealingDataset === dataset.dataset_id" class="loading-spinner"></span>
                {{ sealingDataset === dataset.dataset_id ? '封板中...' : '封板' }}
              </button>
              <button 
                v-else-if="dataset.stage === 'sealed'"
                class="secondary"
                disabled
                title="已封板（只读，等待训练）"
              >已封板</button>
              <button 
                @click="editDataset(dataset)" 
                class="secondary"
                :disabled="editingDataset === dataset.dataset_id || dataset.stage === 'sealed'"
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
          <p class="lib-row">
            归属算法库:
            <b v-if="datasetLibraryId(dataset)">{{ libraryName(datasetLibraryId(dataset)) }}</b>
            <span v-else class="badge badge-unassigned">未归属</span>
            <button class="link-btn" @click="openBindDialog(dataset)" title="数据集必须归属一个算法库，训练产出才会进入该库">
              {{ datasetLibraryId(dataset) ? '改归属' : '去归属' }}
            </button>
          </p>
          <p>大小: {{ (dataset.size / 1024 / 1024).toFixed(2) }} MB</p>
          <p class="life-row">
            <span class="life-pill" :class="'life-' + lifecycleOf(dataset).key" :title="lifecycleOf(dataset).hint">{{ lifecycleOf(dataset).text }}</span>
            <span v-if="['training', 'completed'].includes(dataset.stage) && dataset.training_status" class="life-sub" :title="'training_status'">{{ trainingLabel(dataset.training_status) }}</span>
            <span v-if="dataset.stage === 'sealed'" class="life-lock" title="封板后为不可变快照（硬链接共享数据，仅供训练调用，不允许修改/删除）">🔒 不可变快照</span>
          </p>
          <!-- 生命周期进度条（数据模块 §3 状态机：入池 → 标注 → 审核 → 封板） -->
          <div class="life-track" :class="{ sk: lifecycleOf(dataset).skip }">
            <div v-for="(seg, i) in lifeSegments" :key="seg" class="life-seg" :class="{ on: i < lifecycleOf(dataset).idx, cur: i === lifecycleOf(dataset).idx }">
              <span class="life-dot"></span>
              <span class="life-seg-label">{{ seg }}</span>
            </div>
          </div>
          <p v-if="dataset.sealed_at">封板时间: {{ formatTime(dataset.sealed_at) }}</p>
          <p v-if="dataset.image_count">图片数: {{ dataset.image_count }}<span v-if="dataset.stage === 'annotating' && sealMinImages > 0" style="color:#b26a00;font-size:0.8rem">（待封板 {{ dataset.image_count }} / {{ sealMinImages }} 张{{ dataset.image_count < sealMinImages ? `，还差 ${sealMinImages - dataset.image_count} 张` : '' }}）</span></p>
          <p v-if="dataset.label_count">标签数: {{ dataset.label_count }}</p>
          <p v-if="dataset.classes">类别: {{ dataset.classes.join(', ') }}</p>
          <p v-if="dataset.annotation_task_id" class="auto-task-hint">
              ✓ 已自动创建标注任务（{{ dataset.annotation_task_id }}）→ 点「去标注」直接标注
            </p>
          <p v-if="dataset.description">描述: {{ dataset.description }}</p>
          <p v-if="dataset.tags && dataset.tags.length">标签: {{ dataset.tags.join(', ') }}</p>
          </div><!-- /dataset-item -->
          </div><!-- /ds-section -->
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

          <!-- 图片预览 + 样本池入库（1.6） -->
          <div class="viewer-section-title">
            图片预览
            <span class="section-hint">勾选图片可加入样本池（困难样本/背景负样本），训练时自动抽样并入</span>
          </div>
          <div v-if="viewerImages.length === 0" class="empty-state">
            该数据集尚无图片（或未执行"准备"操作）
          </div>
          <template v-else>
            <div class="pool-toolbar">
              <label class="pool-model-label">困难样本归属模型:</label>
              <select v-model="poolModelId" class="pool-model-select" title="困难样本必须归属模型，训练该模型时才会抽样并入">
                <option value="">-- 选择模型（困难样本必填）--</option>
                <option v-for="m in poolModels" :key="m.model_id" :value="m.model_id">
                  {{ m.display_name || m.name || m.model_code || m.model_id }}
                </option>
              </select>
              <button class="secondary pool-btn" @click="addSelectionToHardPool" :disabled="poolImagesSelected.length === 0 || !poolModelId || !!poolBusy">
                <span v-if="poolBusy === 'hard'" class="loading-spinner"></span>
                {{ poolBusy === 'hard' ? '入库中...' : `加入困难样本库（${poolImagesSelected.length}张）` }}
              </button>
              <button class="secondary pool-btn" @click="addSelectionToBackground" :disabled="poolImagesSelected.length === 0 || !!poolBusy">
                <span v-if="poolBusy === 'bg'" class="loading-spinner"></span>
                {{ poolBusy === 'bg' ? '入库中...' : `加入空白样本库（${poolImagesSelected.length}张）` }}
              </button>
              <button class="secondary pool-btn" @click="poolSelected = {}" :disabled="poolImagesSelected.length === 0">取消选择</button>
            </div>
            <div class="viewer-grid">
              <div v-for="img in viewerImages" :key="img" class="viewer-thumb" :class="{ 'thumb-selected': !!poolSelected[img] }" @click="togglePoolSelect(img)">
                <img :src="'/static/' + img" loading="lazy" @error="handleImgError" />
                <span class="thumb-check" :class="{ checked: !!poolSelected[img] }">{{ poolSelected[img] ? '✓' : '' }}</span>
              </div>
            </div>
          </template>
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

    <!-- 归属算法库：导入时必选，老数据可在这里补选，已归属的可改挂到其它库 -->
    <LibraryCreateModal v-model:open="createLibraryOpen" confirm-text="创建并选中" @created="onLibraryCreated" />

    <div v-if="bindDialog" class="modal-mask" @click.self="bindDialog = null">
      <div class="modal bind-modal">
        <h3>{{ datasetLibraryId(bindDialog) ? '修改归属算法库' : '选择归属算法库' }}</h3>
        <p class="modal-sub">{{ bindDialog.dataset_id }}<br />数据集必须归属一个算法库，训练产出的模型才会进入该库</p>
        <div class="form-field">
          <label>归属算法库 <i>*</i></label>
          <select v-model="bindLibraryId">
            <option value="">— 请选择算法库 —</option>
            <option v-for="lib in libraryOptions" :key="lib.model_id" :value="lib.model_id">
              {{ lib.display_name || lib.model_code || lib.model_id }}
            </option>
          </select>
        </div>
        <div v-if="bindError" class="form-error">{{ bindError }}</div>
        <div class="modal-actions">
          <button class="secondary" @click="createLibraryOpen = true">+ 新建算法库</button>
          <button class="secondary" @click="bindDialog = null">取消</button>
          <button class="primary" @click="submitBind" :disabled="!bindLibraryId || binding">
            {{ binding ? '保存中...' : '保存' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onActivated, nextTick } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { uploadDataset, prepareDataset as prepareDst, listDatasets, getDataset, getDatasetTree, updateDataset, deleteDataset, sealDataset as sealDst, exportAnnotatedDataset, exportOriginalDataset, advanceBuckets, approveDatasetReview, unpourDatasetBucket, bindDatasetModel } from '@/api/datasets'
import { listModels } from '@/api/models'
import { addHardSamples, addBackgroundSamples } from '@/api/samplePool'
import { downloadFile } from '@/utils/download'
import { showConfirm, showPrompt } from '@/composables/useDialog'
import FolderTree from '@/components/FolderTree.vue'
import LibraryCreateModal from '@/components/LibraryCreateModal.vue'

const selectedFile = ref<File | null>(null)
const router = useRouter()
const route = useRoute()
const uploadLibraryId = ref('')   // 导入时必选的归属算法库（库根记录 id）
const libraryTouched = ref(false) // 点过上传但没选库 → 显示红字提示
const createLibraryOpen = ref(false)
const allModels = ref<any[]>([]) // registry 记录（算法库下拉数据源）
const uploading = ref(false)
const uploadResult = ref<any>(null)
const sealMinImages = ref(0)  // #6 封板数量门槛（来自后端），列表展示待封板进度
const datasets = ref<any[]>([])
const buckets = ref<any[]>([])      // B 系列：按模型桶进度（receiving + sealed）
const bucketNotice = ref('')        // 灌桶/封板结果提示（点击关闭）
const loading = ref(false)
const preparingNew = ref(false)  // 正在准备新上传的数据集
const editingDataset = ref<string | null>(null)  // 正在编辑的数据集ID
const deletingDataset = ref<string | null>(null)  // 正在删除的数据集ID
const exportingDataset = ref<string | null>(null)  // 正在导出的数据集ID
const sealingDataset = ref<string | null>(null)  // 正在封板的数据集ID
const approvingDataset = ref<string | null>(null)  // 正在审核确认的数据集ID
const unpouringDataset = ref<string | null>(null)  // 正在撤回（退桶）的数据集ID
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

// ===== 样本池入库（1.6）=====
const poolModels = ref<any[]>([])      // 模型列表（困难样本归属选择）
const poolModelId = ref('')            // 选中的模型 id
const poolSelected = ref<Record<string, boolean>>({})  // 勾选的图片路径
const poolBusy = ref<'' | 'hard' | 'bg'>('')

/** 勾选图片路径集合（返回相对路径数组，入库时后端按文件名 stem 匹配） */
const poolImagesSelected = computed(() => Object.keys(poolSelected.value).filter(k => poolSelected.value[k]))

const togglePoolSelect = (imgPath: string) => {
  poolSelected.value[imgPath] = !poolSelected.value[imgPath]
}

// 加载模型列表（困难样本入库时选择归属模型）
const loadPoolModels = async () => {
  try {
    const result = await listModels()
    poolModels.value = (result.models || []).filter((m: any) => m.status !== 'deleted')
  } catch (e) {
    poolModels.value = []
  }
}

/** 加载算法库列表，供导入时必选归属（库 = 自身即库根的 registry 记录） */
const loadUploadModels = async () => {
  try {
    const result = await listModels()
    allModels.value = (result.models || []).filter((m: any) => m.status !== 'deleted')
    if (uploadLibraryId.value && !libraryOptions.value.some((l: any) => l.model_id === uploadLibraryId.value)) {
      uploadLibraryId.value = ''
    }
  } catch (e) {
    allModels.value = []
  }
}

// 算法库下拉数据源：library_root_id 指向自身（或老数据未写该字段）的记录的才是库，
// 训练产出的版本记录 library_root_id 指向所属库，不会出现在这里
const libraryOptions = computed(() =>
  allModels.value.filter((m: any) => (m.library_root_id || m.model_id) === m.model_id)
)

const libraryName = (id: string) => {
  const m = allModels.value.find(x => x.model_id === id)
  return m ? (m.display_name || m.model_code || m.model_id) : (id || '未归属')
}

// 数据集归属展示：新数据写 library_root_id，老数据只有 model_id（可能就是库根本身）；
// 指向已删除模型记录的悬空 id 不算归属，按「未归属」处理并允许补选
const datasetLibraryId = (d: any) => {
  const id = d?.library_root_id || d?.model_id || ''
  if (!id) return ''
  return libraryOptions.value.some((l: any) => l.model_id === id) ? id : ''
}

// 导入页新建算法库：只选中新库，不做页面跳转
function onLibraryCreated(meta: any) {
  loadUploadModels()
  if (meta?.model_id) {
    uploadLibraryId.value = meta.model_id
    libraryTouched.value = false
    if (bindDialog.value) bindLibraryId.value = meta.model_id
  }
}

// ===== 补选 / 改挂归属算法库（历史未归属的老数据集用这个）=====
const bindDialog = ref<any | null>(null)
const bindLibraryId = ref('')
const bindError = ref('')
const binding = ref(false)

function openBindDialog(dataset: any) {
  bindDialog.value = dataset
  bindLibraryId.value = datasetLibraryId(dataset)
  bindError.value = ''
}

async function submitBind() {
  if (!bindDialog.value || !bindLibraryId.value) return
  binding.value = true
  bindError.value = ''
  try {
    await bindDatasetModel(bindDialog.value.dataset_id, bindLibraryId.value)
    bindDialog.value = null
    loadDatasets()
    loadUploadModels()
  } catch (e: any) {
    bindError.value = '保存失败: ' + (e.response?.data?.detail || e.message)
  } finally {
    binding.value = false
  }
}

// 从图片相对路径提取 stem（文件名去扩展名），供后端匹配图片与标注
const imageStems = (paths: string[]) =>
  paths.map(p => {
    const name = p.split('/').pop() || p
    const extIdx = name.lastIndexOf('.')
    return extIdx > 0 ? name.slice(0, extIdx) : name
  })

const addSelectionToHardPool = async () => {
  if (!viewingDataset.value || poolImagesSelected.value.length === 0) return
  if (!poolModelId.value) {
    alert('请先选择困难样本归属模型（困难样本库按模型 1:1 隔离，训练该模型时才会抽样并入）')
    return
  }
  const ok = await showConfirm(`将 ${poolImagesSelected.value.length} 张图片（连同标注）加入模型 ${poolModelId.value} 的困难样本库？\n标注缺失的图片会跳过。`)
  if (!ok) return
  poolBusy.value = 'hard'
  try {
    const res = await addHardSamples({
      dataset_id: viewingDataset.value.dataset_id,
      model_id: poolModelId.value,
      image_names: imageStems(poolImagesSelected.value),
      version: 'v1'
    })
    alert(`入库完成：新增 ${res.added} 张，跳过 ${res.skipped} 张（未找到/已存在）。\n该模型困难样本库现有 ${res.pool_image_count} 张。`)
  } catch (e: any) {
    alert('加入困难样本库失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    poolBusy.value = ''
  }
}

const addSelectionToBackground = async () => {
  if (!viewingDataset.value || poolImagesSelected.value.length === 0) return
  const ok = await showConfirm(`将 ${poolImagesSelected.value.length} 张图片作为无目标背景（负样本，配空标注）加入全局空白样本库？\n训练任意模型时可按比例抽样并入。`)
  if (!ok) return
  poolBusy.value = 'bg'
  try {
    const res = await addBackgroundSamples({
      dataset_id: viewingDataset.value.dataset_id,
      image_names: imageStems(poolImagesSelected.value),
      version: 'v1'
    })
    alert(`入库完成：新增 ${res.added} 张，跳过 ${res.skipped} 张。`)
  } catch (e: any) {
    alert('加入空白样本库失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    poolBusy.value = ''
  }
}

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
  poolSelected.value = {}  // 打开新数据集时清空勾选
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

// 雪球闭环（1.8）：直达标注页 —— ?dataset_id=xxx，标注页优先复用已有任务/自动创建
// P-01：改用 router.push 走 SPA 跳转，避免整页刷新丢失前端状态
const goAnnotate = (dataset: any) => {
  router.push('/annotate?dataset_id=' + dataset.dataset_id)
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

// ===== 数据集生命周期状态（MLOps 1.1）：文字由 lifecycleOf 胶囊提供；训练子状态单独展示 =====
const TRAIN_STATUS_LABELS: Record<string, string> = {
  incomplete: '未完成训练',
  completed: '已完成训练',
}
const trainingLabel = (s?: string) => TRAIN_STATUS_LABELS[s || ''] || s || ''

// ===== B 系列：A包/归档/成品 标识 + 桶进度 =====
const modelName = (mid: string) => {
  const m = allModels.value.find(x => x.model_id === mid)
  return m ? (m.display_name || m.name || m.model_code || mid) : mid
}

// 卡片徽标：素材袋=A包，已归档=已归档→B{N}，桶成品=成品 B{N}（恒返回对象，便于模板直接用）
const bucketBadge = (d: any) => {
  if (!d) return { text: '', cls: '' }
  if (d.is_bucket_snapshot) return { text: `成品 B${d.bucket_index ?? ''}`, cls: 'badge-b' }
  if (d.bucket_archived) return { text: `已归档 → B${d.bucket_index ?? ''}`, cls: 'badge-archived' }
  if (d.model_id && d.stage && ['annotating', 'collecting'].includes(d.stage)) {
    const total = Number(d.image_count || 0)
    const done = Number(d.label_count || 0)
    if (d.review_approved) return { text: 'A包·已审核', cls: 'badge-a' }
    if (total > 0 && done >= total) return { text: 'A包·待审核', cls: 'badge-pending-review' }
    return { text: 'A包', cls: 'badge-a' }
  }
  return { text: '', cls: '' }
}

// 桶进度条（每个模型一行：已封板版本 + 接收桶已灌/容量）
const bucketBars = computed(() => buckets.value.map(b => {
  const recv = b.receiving || { index: 1, poured: 0 }
  const cap = sealMinImages.value || 0
  const poured = Number(recv.poured) || 0
  return {
    model_id: b.model_id,
    sealed: b.sealed || [],
    curIndex: recv.index,
    poured,
    cap,
    pct: cap > 0 ? Math.min(100, Math.round((poured / cap) * 100)) : 0,
  }
}))

// 是否可封板：已准备 且 未进入 sealed/training/completed/failed 阶段 且 未归档（已归档的包已灌入桶）
const canSeal = (d: any) => {
  if (!d) return false
  if (d.bucket_archived) return false
  const s = d.stage
  if (['sealed', 'training', 'completed', 'failed'].includes(s)) return false
  return d.status === 'prepared'
}

// ===== 数据池 ↔ 训练仓库双分区（数据模块 §3/§4：未封板=池，封板=不可变快照） =====
const poolDatasets = computed(() =>
  datasets.value.filter((d: any) => !['sealed', 'training', 'completed'].includes(d.stage))
)
const frozenDatasets = computed(() =>
  datasets.value.filter((d: any) => ['sealed', 'training', 'completed'].includes(d.stage))
)
const datasetSections = computed(() => [
  {
    key: 'pool',
    title: '🌊 数据池（未封板 · 可编辑）',
    desc: '上传即入池：标注完成后达到封板阈值 → 点击「封板」生成不可变快照',
    emptyText: '数据池为空，请先在上方上传数据集',
    items: poolDatasets.value
  },
  {
    key: 'frozen',
    title: '🧊 训练数据集（已封板快照 · 只读）',
    desc: '封板后为不可变数据集版本，可被多个训练任务同时复用（硬链接零额外存储）',
    emptyText: '暂无已封板数据集：完成标注后在数据池中点击「封板」',
    items: frozenDatasets.value
  }
])

// 生命周期状态机（数据模块 §4.3）：Raw 入池 → 标注 → 审核待封板 → Frozen 封板
const lifeSegments = ['入池', '标注', '审核', '封板']
const lifecycleOf = (d: any) => {
  const s = d?.stage
  if (['sealed', 'training', 'completed'].includes(s)) {
    const text = s === 'sealed' ? '已封板' : s === 'training' ? '训练中' : '已完成训练'
    return { key: 'sealed', text, idx: 3, hint: '数据已封板为不可变快照，仅供训练任务调用', skip: false }
  }
  if (s === 'failed') {
    return { key: 'failed', text: '训练失败', idx: 2, hint: '该数据集训练未通过，保持未完成状态，可重新标注/封板参与下一轮训练', skip: true }
  }
  if (!d?.model_id && !d?.bucket_archived) {
    return { key: 'unassigned', text: '未归属', idx: 0, hint: '尚未归属任何算法库，点数据集上的「去归属」补选后才能用于训练', skip: false }
  }
  if (s === 'collecting') {
    return { key: 'todo', text: '待标注', idx: 0, hint: '数据已导入数据池，尚未开始标注', skip: false }
  }
  if (canSeal(d)) {
    return { key: 'pending', text: '待封板', idx: 2, hint: '标注完成且达到封板门槛（或准备就绪），等待封板确认', skip: false }
  }
  return { key: 'annotating', text: '标注中', idx: 1, hint: '标注工作正在进行', skip: false }
}

// 封板结果提示：把「这次到底导出了多少分割轮廓」说清楚，避免误以为整包都是分割标注
const sealMsg = (res: any, head: string) => {
  const s = res?.sealed_split
  if (!s || typeof s.segment_boxes !== 'number') return head
  if (!s.has_segment) return `${head}\n本批标注均为普通框（无轮廓），data.yaml 标记 task: detect`
  const extra = s.rect_boxes ? `\n另有 ${s.rect_boxes} 个普通框（无轮廓），训练分割时不会生成掩码` : ''
  return `${head}\n导出分割轮廓 ${s.segment_boxes} 个，data.yaml 已标记 task: segment${extra}`
}

// 封板（标注不完整时后端拒绝，可走强制封板 = 时间窗口兜底）
const sealDatasetItem = async (d: any) => {
  // 数量门槛提示（#6）：标注中且数量未达门槛 → 明确提示差额，避免封板被后端拒绝
  const imgN = Number(d.image_count || 0)
  const minN = sealMinImages.value
  const sealNote = minN > 0 && imgN > 0 && imgN < minN
    ? `\n⚠ 当前 ${imgN} 张 < 封板门槛 ${minN} 张：直接封板会被拒绝，需走「强制封板」兜底`
    : ''
  const ratio = d.split_ratio || { train: 0.7, val: 0.15, test: 0.15 }
  const input = window.prompt(
    '请输入数据集划分比例（训练,验证,测试，逗号分隔，如 0.7,0.15,0.15）：' + sealNote,
    `${ratio.train},${ratio.val},${ratio.test ?? 0}`
  )
  if (input === null) return
  const parts = input.split(/[,，、\s]+/).map(Number)
  if (parts.length < 2 || parts.some(n => Number.isNaN(n) || n < 0) || parts.reduce((a, b) => a + b, 0) <= 0) {
    alert('输入无效，请用逗号分隔三个非负数字，如：0.7,0.15,0.15')
    return
  }
  const t = parts[0], v = parts[1], te = parts[2] ?? 0
  const ok = await showConfirm(
    `确定封板数据集 ${d.dataset_id} 吗？\n` +
    `划分比例：训练:验证:测试 = ${t} : ${v} : ${te}\n` +
    `封板后数据集变为只读，等待进入训练队列。`
  )
  if (!ok) return
  sealingDataset.value = d.dataset_id
  try {
    const res = await sealDst(d.dataset_id, false, { train: t, val: v, test: te })
    alert(sealMsg(res, '封板成功'))
  } catch (e: any) {
    const msg = e.response?.data?.detail || e.message
    const force = await showConfirm(`${msg}\n\n是否需要强制封板（时间窗口兜底，跳过标注完成条件）？`)
    if (force) {
      try {
        const res2 = await sealDst(d.dataset_id, true, { train: t, val: v, test: te })
        alert(sealMsg(res2, '强制封板成功'))
      } catch (e2: any) {
        alert('封板失败: ' + (e2.response?.data?.detail || e2.message))
      }
    }
  } finally {
    sealingDataset.value = null
    loadDatasets()
  }
}

// ===== 审核门槛 / 撤回（灌桶前人工确认；未封板的已灌桶包可撤回恢复编辑） =====
// 可审核：已准备、未封板、未归档、尚未审核通过
const canApprove = (d: any) => {
  if (!d) return false
  if (d.review_approved || d.bucket_archived) return false
  if (['sealed', 'training', 'completed', 'failed'].includes(d.stage)) return false
  return d.status === 'prepared'
}

// 可撤回：已灌入接收桶（archived/pending>0）但桶尚未封板（bucket_sealed==0）
const canUnpour = (d: any) => {
  if (!d) return false
  if (['sealed', 'training', 'completed'].includes(d.stage)) return false
  if ((Number(d.bucket_sealed) || 0) > 0) return false
  return !!d.bucket_archived || (Number(d.bucket_pending) || 0) > 0
}

const approveItem = async (d: any) => {
  const ok = await showConfirm(
    `确认数据集 ${d.dataset_id} 的标注已人工审核无误？\n` +
    `审核通过后该包才允许自动灌桶/封板（灌桶后仍可撤回，封板后不可逆）。`
  )
  if (!ok) return
  approvingDataset.value = d.dataset_id
  try {
    await approveDatasetReview(d.dataset_id)
  } catch (e: any) {
    alert('审核确认失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    approvingDataset.value = null
    loadDatasets()
  }
}

const unpourItem = async (d: any) => {
  const ok = await showConfirm(
    `撤回数据集 ${d.dataset_id}？\n` +
    `将把该包已灌入接收桶但尚未封板的图片退回，恢复为可编辑状态（需重新审核通过后才会再次灌桶）。`
  )
  if (!ok) return
  unpouringDataset.value = d.dataset_id
  try {
    await unpourDatasetBucket(d.dataset_id)
  } catch (e: any) {
    alert('撤回失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    unpouringDataset.value = null
    loadDatasets()
  }
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
  libraryTouched.value = true
  if (!uploadLibraryId.value) {
    alert('请先选择该数据集归属的算法库（没有合适的库请点「+ 新建算法库」）')
    return
  }

  uploading.value = true
  try {
    const result = await uploadDataset(selectedFile.value, uploadLibraryId.value)
    uploadResult.value = result
    loadDatasets()
    // 上传成功后自动准备（无需用户额外操作）
    preparingNew.value = true
    try {
      await prepareDst(result.dataset_id)
      alert(`上传并准备成功！数据集已归属算法库「${libraryName(uploadLibraryId.value)}」`)
      uploadResult.value = null
      loadDatasets()
      loadUploadModels()
    } catch (error: any) {
      alert('准备失败: ' + (error.response?.data?.detail || error.message))
    } finally {
      preparingNew.value = false
    }
  } catch (error: any) {
    alert('上传失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    uploading.value = false
  }
}

const loadDatasets = async (silent = false) => {
  if (!silent) loading.value = true
  try {
    // B 系列：先触发按模型滚动灌桶（幂等，标注完成返回列表时自动执行），再拉取最新列表
    try {
      const adv = await advanceBuckets()
      const sealedN = (adv.sealed || []).length
      const pouredN = (adv.poured || []).length
      if (sealedN > 0) {
        bucketNotice.value = `自动灌桶完成：${pouredN} 个 A 包已入池，${sealedN} 个桶已封板`
      } else if (pouredN > 0) {
        bucketNotice.value = `已自动灌桶 ${pouredN} 个 A 包`
      }
      const reviewN = (adv.pending_review || []).length
      if (reviewN > 0) {
        bucketNotice.value = (bucketNotice.value ? bucketNotice.value + '；' : '') + `${reviewN} 个包标注完成，待人工审核通过后才会灌桶`
      }
    } catch (e) {
      // 灌桶失败不阻塞列表展示
    }
    const result = await listDatasets()
    datasets.value = result.datasets
    sealMinImages.value = result.seal_min_images || 0
    buckets.value = result.buckets || []
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

// D-02：滚动 + 高亮指定数据集卡片（列表渲染完成后再定位）
// 用内联 style 而非 classList：Vue patch 元素时会把 class 重置回模板值，外部加的 class 会被抹掉，
// 但未绑定的内联 style 不受 class patch 影响，故高亮更稳。
const applyFocus = (id: string) => {
  const tryLocate = (attempt: number) => {
    const el = document.querySelector(`[data-ds="${CSS.escape(id)}"]`) as HTMLElement | null
    if (!el) {
      if (attempt < 10) setTimeout(() => tryLocate(attempt + 1), 100)  // 列表尚未渲染，重试
      return
    }
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    el.style.setProperty('outline', '3px solid #f59e0b')
    el.style.setProperty('outline-offset', '1px')
    el.style.setProperty('box-shadow', '0 0 0 6px rgba(245, 158, 11, 0.35)')
    setTimeout(() => {
      el.style.removeProperty('outline')
      el.style.removeProperty('outline-offset')
      el.style.removeProperty('box-shadow')
    }, 4000)  // 高亮 4s 后淡出
  }
  nextTick(() => tryLocate(0))
}

const consumeFocus = () => {
  const f = (route.query.focus as string) || ''
  if (f) applyFocus(f)
}

onMounted(async () => {
  // D-02：消费 ?focus= —— 列表加载后滚动到目标数据集并高亮
  await loadDatasets()
  loadPoolModels()
  loadUploadModels()
  consumeFocus()
})

// B-10：keep-alive 缓存后，切回本页静默刷新（不闪骨架屏）；
// 且从模型详情页带 ?focus= 跳转进来时组件不会重新 onMounted，
// 需在 activated 里消费 focus，否则高亮定位失效。首次挂载跳过避免重复请求。
// 注：不做滚动位置恢复——导航栏非 sticky，点导航前必然已滚回顶部，恢复无意义。
let _dsFirstActivate = true
onActivated(() => {
  const skipRefresh = _dsFirstActivate
  _dsFirstActivate = false
  if (!skipRefresh) {
    loadDatasets(true)
    consumeFocus()
  }
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

.auto-prepare-tip {
  margin-top: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #1f6f4a;
}

.model-code-input {
  width: 100%;
  padding: 8px 10px;
  border: 1px solid #d5dee8;
  border-radius: 8px;
  font-size: 0.9rem;
}

.model-code-input:focus {
  border-color: #2c6ec5;
  outline: none;
}

.form-hint {
  color: #9aa7b6;
  font-size: 0.78rem;
}

.auto-create-tag {
  color: #2c6ec5;
  font-weight: 600;
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

/* D-02：从模型详情跳转聚焦时的临时高亮 */
.dataset-item.ds-focus {
  border-color: #f59e0b;
  box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.35);
  transition: box-shadow 0.6s ease, border-color 0.6s ease;
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

.auto-task-hint {
  color: #2e7d32 !important;
  background: #e8f5e9;
  border: 1px solid #c8e6c9;
  border-radius: 4px;
  padding: 0.15rem 0.5rem;
  display: inline-block;
  font-size: 0.8rem;
}

/* ===== B 系列：A包/归档/成品徽标 + 桶进度 ===== */
.badge {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 600;
  border-radius: 10px;
  padding: 0.1rem 0.55rem;
  margin-left: 0.5rem;
  vertical-align: middle;
}
.badge-a {
  background: #e8d9f5;
  color: #6c3483;
  border: 1px solid #c9a8e0;
}
.badge-pending-review {
  background: #fdecea;
  color: #b23b2e;
  border: 1px solid #f0b8b1;
}
.badge-archived {
  background: #eee5d5;
  color: #8a6d1b;
  border: 1px solid #d8c490;
}
.badge-b {
  background: #d4efdf;
  color: #1e7e4a;
  border: 1px solid #9ed4b4;
}

.bucket-notice {
  margin-top: 0.75rem;
  padding: 0.5rem 0.9rem;
  background: #fdf3d8;
  border: 1px solid #ead9a8;
  border-radius: 6px;
  color: #8a6d1b;
  font-size: 0.85rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.bucket-notice-close {
  margin-left: 0.75rem;
  font-weight: bold;
  color: #b39d5e;
}

.bucket-progress {
  margin-top: 0.9rem;
  padding: 0.75rem 1rem;
  background: #f7f9fb;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
}
.bucket-progress-title {
  font-size: 0.8rem;
  color: #5a6a7a;
  margin-bottom: 0.5rem;
  font-weight: 600;
}
.bucket-model {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
  padding: 0.3rem 0;
}
.bucket-model-tag {
  font-size: 0.8rem;
  font-weight: 700;
  color: #2c3e50;
  background: #eaf0f6;
  border-radius: 10px;
  padding: 0.1rem 0.6rem;
  white-space: nowrap;
}
.bucket-sealed {
  font-size: 0.75rem;
  color: #1e7e4a;
  background: #d4efdf;
  border: 1px solid #9ed4b4;
  border-radius: 10px;
  padding: 0.08rem 0.5rem;
  white-space: nowrap;
}
.bucket-recv {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  font-size: 0.8rem;
  color: #5a6a7a;
  white-space: nowrap;
}
.bar {
  width: 120px;
  height: 8px;
  background: #e3e8ee;
  border-radius: 4px;
  overflow: hidden;
  display: inline-block;
}
.bar-fill {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, #8e44ad, #a569bd);
  border-radius: 4px;
  transition: width 0.3s;
}
.bar-num {
  font-size: 0.75rem;
  color: #7f8c8d;
  font-variant-numeric: tabular-nums;
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

/* 样本池入库（1.6）*/
.section-hint {
  font-size: 0.75rem;
  color: #95a5a6;
  font-weight: normal;
  margin-left: 0.5rem;
}

.pool-toolbar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  padding: 0.5rem 1.25rem;
  border-bottom: 1px solid #f0f0f0;
}

.pool-model-label {
  font-size: 0.8rem;
  color: #555;
}

.pool-model-select {
  max-width: 240px;
  padding: 0.35rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
  font-size: 0.85rem;
}

.pool-btn {
  padding: 0.3rem 0.7rem;
  font-size: 0.8rem;
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
}

.viewer-thumb {
  position: relative;
  cursor: pointer;
  transition: border-color 0.15s, box-shadow 0.15s;
}

.viewer-thumb:hover {
  border-color: #8e44ad;
}

.viewer-thumb.thumb-selected {
  border: 2px solid #8e44ad;
  box-shadow: 0 0 0 2px rgba(142, 68, 173, 0.25);
}

.thumb-check {
  position: absolute;
  top: 6px;
  right: 6px;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.45);
  color: white;
  font-size: 14px;
  font-weight: bold;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid rgba(255, 255, 255, 0.7);
}

.thumb-check.checked {
  background: #8e44ad;
  border-color: #8e44ad;
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

/* ===== 数据池 ↔ 训练仓库双分区（数据模块 §3） ===== */
.ds-section {
  margin-top: 0.6rem;
  padding: 0.75rem 0.9rem;
  background: #fbfcfe;
  border: 1px solid #e7ecf2;
  border-radius: 8px;
}
.ds-section-head {
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  flex-wrap: wrap;
  margin-bottom: 0.55rem;
}
.ds-section-title {
  font-weight: 700;
  font-size: 0.95rem;
  color: #2c3e50;
}
.ds-section-desc {
  font-size: 0.75rem;
  color: #95a5a6;
  flex: 1 1 auto;
  min-width: 200px;
}
.ds-section-count {
  font-size: 0.75rem;
  color: #7f8c8d;
  background: #eef0f3;
  border-radius: 10px;
  padding: 0.1rem 0.55rem;
  white-space: nowrap;
}
.empty-state.mini {
  padding: 0.8rem;
  font-size: 0.82rem;
}

/* ===== 生命周期状态胶囊（3.3 六色：灰/蓝/橙/绿/白/红，与算法库详情页一致） ===== */
.life-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.life-pill {
  font-size: 0.75rem;
  padding: 3px 10px;
  border-radius: 10px;
  border: 1px solid transparent;
  white-space: nowrap;
  display: inline-block;
  font-weight: 600;
}
.life-todo { background: #eef0f3; color: #6b7c93; }
.life-annotating { background: #e3f2fd; color: #1565c0; }
.life-pending { background: #fff3e0; color: #b26a00; border-color: #ffd9a0; font-weight: 700; }
.life-sealed { background: #e8f5e9; color: #2e7d32; border-color: #c8e6c9; }
.life-unassigned { background: #fff; color: #9aa7b6; border-color: #eaedf1; }
.life-failed { background: #ffebee; color: #c62828; border-color: #ffcdd2; }
.life-sub {
  font-size: 0.72rem;
  color: #7f8c8d;
  background: #f4f6f8;
  border-radius: 8px;
  padding: 2px 8px;
}
.life-lock {
  font-size: 0.72rem;
  color: #2e7d32;
  background: #e8f5e9;
  border: 1px solid #c8e6c9;
  border-radius: 8px;
  padding: 2px 8px;
  font-weight: 600;
}

/* ===== 生命周期进度条（Raw → 标注 → 审核 → 封板） ===== */
.life-track {
  display: flex;
  margin: 0.4rem 0 0.2rem;
  max-width: 420px;
}
.life-seg {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  position: relative;
}
.life-seg:not(:last-child)::after {
  content: '';
  position: absolute;
  top: 4px;
  left: 50%;
  width: 100%;
  height: 2px;
  background: #e3e8ee;
}
.life-dot {
  position: relative;
  z-index: 1;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #e3e8ee;
  border: 2px solid #fff;
  box-shadow: 0 0 0 1px #e3e8ee;
}
.life-seg-label {
  font-size: 0.7rem;
  color: #aab5c0;
}
.life-seg.on .life-dot {
  background: #8e44ad;
  box-shadow: 0 0 0 1px #8e44ad;
}
.life-seg.on .life-seg-label {
  color: #6c2d82;
  font-weight: 600;
}
.life-seg.on:not(:last-child)::after {
  background: #c9a8e0;
}
.life-seg.cur .life-dot {
  background: #6c2d82;
  box-shadow: 0 0 0 3px rgba(108, 45, 130, 0.25);
}
.life-track.sk .life-seg.cur .life-dot {
  background: #e74c3c;
  box-shadow: 0 0 0 3px rgba(231, 76, 60, 0.25);
}
.life-track.sk .life-seg.cur .life-seg-label {
  color: #c0392b;
}

/* ===== 归属算法库：导入必选 + 列表展示 + 补选弹窗 ===== */
.library-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.library-row .model-code-input {
  flex: 1;
}

.model-code-input.invalid {
  border-color: #c62828;
}

.required {
  color: #c62828;
  font-style: normal;
}

.error-hint {
  color: #c62828;
}

.lib-row {
  display: flex;
  align-items: center;
  gap: 0.35rem;
}

.link-btn {
  background: none;
  border: none;
  color: #2c6ec5;
  font-size: 0.78rem;
  cursor: pointer;
  padding: 0 0.2rem;
  text-decoration: underline;
}

.badge-unassigned {
  background: #fdecea;
  color: #b23b2e;
  border: 1px solid #f0b8b1;
}

.modal-mask {
  position: fixed;
  inset: 0;
  background: rgba(20, 30, 45, 0.35);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.modal {
  background: #fff;
  border-radius: 14px;
  padding: 22px;
  width: 480px;
  max-width: 90vw;
  box-shadow: 0 18px 48px rgba(20, 40, 70, 0.28);
}

.modal h3 {
  color: #1f2d3d;
  margin: 0 0 4px;
}

.modal-sub {
  color: #7a8aa0;
  font-size: 0.82rem;
  margin: 0 0 14px;
  word-break: break-all;
}

.form-field {
  margin-bottom: 12px;
  text-align: left;
}

.form-field label {
  display: block;
  font-size: 0.85rem;
  color: #445;
  margin-bottom: 5px;
}

.form-field label i {
  color: #c62828;
  font-style: normal;
}

.form-field select {
  width: 100%;
  padding: 8px 10px;
  border: 1px solid #d5dee8;
  border-radius: 8px;
  font-size: 0.88rem;
  box-sizing: border-box;
  outline: none;
  font-family: inherit;
}

.form-field select:focus {
  border-color: #2c6ec5;
}

.form-error {
  color: #c62828;
  font-size: 0.82rem;
  margin-bottom: 10px;
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 8px;
}

.primary {
  background: #2c6ec5;
  color: #fff;
  border: 1px solid #2c6ec5;
  border-radius: 8px;
  padding: 8px 16px;
  cursor: pointer;
  font-size: 0.9rem;
}

.primary:hover {
  background: #2459a5;
}

.primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>

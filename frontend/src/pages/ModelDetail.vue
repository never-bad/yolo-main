<template>
  <div class="model-detail">
    <button class="back-btn" @click="$router.push('/')">← 返回仓库</button>

    <div v-if="loading" class="loading-state"><span class="loading-spinner"></span>加载算法库详情...</div>

    <div v-else-if="!model" class="empty-state">算法库不存在或已被删除</div>

    <template v-else>
      <!-- ===== A. 顶部 Header 区域（算法库工作台） ===== -->
      <div class="card wk-header">
        <div class="wk-header-top">
          <div class="wk-title">
            <h1>{{ libraryRoot?.display_name || libraryRoot?.name || libraryRoot?.model_code || '未命名算法库' }}</h1>
            <code class="model-code">{{ model.model_code || model.model_id }}</code>
            <span class="status-tag" :class="model.status">{{ statusLabel(model.status) }}</span>
          </div>
          <button class="ghost small" @click="goSettings" title="进入算法库设置">
            ⚙ 设置
          </button>
        </div>
        <div class="wk-desc">
          {{ libraryRoot?.description || '暂无业务场景描述，点击「编辑简介」补充该算法库的用途说明。' }}
        </div>
        <div class="wk-meta">
          <span>算法架构：{{ archLabel }}</span>
          <span class="meta-sep">|</span>
          <span>标签版本：{{ labelVersionLabel }}</span>
        </div>
        <div class="wk-actions">
          <button class="ghost small" @click="editDescOpen = !editDescOpen">
            {{ editDescOpen ? '收起' : '编辑简介' }}
          </button>
          <button class="ghost small" @click="openLabelModal">管理标签</button>
        </div>
        <div v-if="editDescOpen" class="wk-edit-desc">
          <textarea v-model="descDraft" rows="3" placeholder="描述该算法库的业务场景、识别要求等"></textarea>
          <div class="wk-edit-desc-actions">
            <button class="primary small" :disabled="savingDesc" @click="saveDescription">
              {{ savingDesc ? '保存中...' : '保存' }}
            </button>
            <button class="secondary small" @click="editDescOpen = false">取消</button>
          </div>
          <p v-if="descSaveMsg" class="wk-save-msg" :class="{ err: descSaveErr }">{{ descSaveMsg }}</p>
        </div>
      </div>

      <!-- ===== B. 主操作栏 ===== -->
      <div class="wk-ops">
        <button class="primary" @click="goTrain">
          ▶ 发起训练
        </button>
        <button class="secondary" @click="importFileInput?.click()">
          ⬆ 导入外部模型
        </button>
        <input ref="importFileInput" type="file" accept=".zip,.pt" class="pt-file-input" @change="handleImportModel" />
      </div>

      <!-- ===== C. Tab 切换区域 ===== -->
      <div class="wk-tabs">
        <button
          v-for="t in wkTabs"
          :key="t.key"
          class="wk-tab"
          :class="{ on: activeTab === t.key }"
          @click="activeTab = t.key"
        >
          {{ t.label }}
        </button>
      </div>

      <!-- ---------- C1. 概览 ---------- -->
      <div v-if="activeTab === 'overview'" class="card wk-pane">
        <div class="wk-pane-head"><h2>算法库概览</h2></div>
        <div class="ov-grid">
          <div class="ov-card">
            <div class="ov-label">业务场景</div>
            <div class="ov-value">{{ model.description || '—' }}</div>
          </div>
          <div class="ov-card">
            <div class="ov-label">算法架构</div>
            <div class="ov-value">{{ archLabel || '—' }}</div>
          </div>
          <div class="ov-card">
            <div class="ov-label">标签体系（{{ labelsDict.labels?.length || 0 }} 类）</div>
            <div class="ov-tags" v-if="labelsDict.labels?.length">
              <span v-for="l in labelsDict.labels" :key="l.index" class="ov-tag">
                {{ l.chinese_name || l.english_code }} <small class="ov-tag-sub">({{ l.index }}·{{ l.english_code }})</small>
              </span>
            </div>
            <div class="ov-value" v-else>尚未定义标签体系，可在「设置」中创建</div>
          </div>
          <div class="ov-card">
            <div class="ov-label">最新模型权重</div>
            <div v-if="newestVersion">
              <div class="ov-value">
                版本 {{ normVer(newestVersion.version) }}<span v-if="latestMAP">（mAP {{ latestMAP }}）</span>
              </div>
              <button
                class="secondary small mt4"
                :disabled="downloadingVer === newestVersion.model_id || !newestVersion.weights_path"
                @click="downloadVersionWeights(newestVersion)"
              >{{ downloadingVer === newestVersion.model_id ? '下载中...' : newestVersion.weights_path ? '⬇ 下载 .pt' : '暂无权重' }}</button>
            </div>
            <div class="ov-value" v-else>该算法库尚无训练产出，发起训练后即可下载</div>
          </div>
        </div>
      </div>

      <!-- ---------- C2. 数据集管理（核心改造区） ---------- -->
      <div v-else-if="activeTab === 'datasets'" class="card wk-pane">
        <div class="wk-pane-head">
          <h2>数据集管理（{{ datasetsView.length }}）</h2>
        </div>

        <!-- 搜索 + 筛选 + 导入新数据 -->
        <div class="ds-toolbar">
          <input v-model="dsKeyword" class="ds-search" placeholder="搜索数据集..." />
          <select v-model="dsFilter" class="ds-filter">
            <option value="">全部状态</option>
            <option value="todo">待标注</option>
            <option value="annotating">标注中</option>
            <option value="pending">待封板</option>
            <option value="sealed">已封板</option>
            <option value="failed">训练失败</option>
          </select>
          <button class="secondary small" @click="openMountModal">
            ⤵ 挂载已有数据集{{ mountableDatasets.length ? ` (${mountableDatasets.length})` : '' }}
          </button>
          <button class="primary small" @click="openDsImportModal">+ 导入新数据</button>
        </div>

        <!-- 空状态：新手引导 -->
        <div v-if="filteredDatasets.length === 0" class="wk-empty">
          <div v-if="datasetsView.length === 0" class="empty-state">
            <h3>该算法库还没有数据集</h3>
            <p>完成以下三步即可训练出第一个模型：</p>
            <ol class="wk-steps">
              <li>① 上传数据：点上方「+ 导入新数据」上传压缩包（留在本页即可完成）</li>
              <li>② 标注：在标注工具中完成类别标注并导出</li>
              <li>③ 训练：点击「发起训练」开始训练，产出模型版本</li>
            </ol>
            <p class="form-hint">如果数据已经导入过平台，也可以直接点上方「⤵ 挂载已有数据集」把它归入本算法库。</p>
          </div>
          <div v-else class="empty-state">没有符合当前筛选条件的数据集</div>
        </div>

        <table class="ds-table" v-else>
          <thead>
            <tr>
              <th>数据集名称</th>
              <th>阶段</th>
              <th>图片数</th>
              <th>标注框</th>
              <th>状态</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in filteredDatasets" :key="d.dataset_id">
              <td>
                <div class="ds-name-main">{{ d.display_name || d.filename || d.dataset_id }}</div>
              </td>
              <td><span class="life-pill" :class="`life-${lifecycleOf(d).key}`">{{ lifecycleOf(d).text }}</span></td>
              <td>{{ d.image_count ?? '—' }}</td>
              <td>{{ d.label_count ?? '—' }}</td>
              <td>
                <span class="train-mark" :class="d.training_status === 'completed' ? 'ok' : 'todo'">
                  {{ d.training_status === 'completed' ? '已完成训练' : (d.training_status && d.training_status !== 'incomplete' ? d.training_status : (d.status === 'prepared' ? '已就绪' : '待处理')) }}
                </span>
              </td>
              <td>
                <div class="row-actions">
                  <!-- 行主按钮（实心，仅一个） -->
                  <router-link v-if="mainActionOf(d).to" :to="mainActionOf(d).to">
                    <button class="primary small">{{ mainActionOf(d).label }}</button>
                  </router-link>
                  <button v-else class="primary small" @click="onMainAction(d)">{{ mainActionOf(d).label }}</button>
                  <!-- 次要操作折叠 -->
                  <div class="more-menu" v-if="d.model_id">
                    <button class="ghost small more-btn" @click="toggleMore(d)">⋯</button>
                    <div v-if="moreOpen === d.dataset_id" class="more-dropdown" @click="moreOpen = ''">
                      <button class="dropdown-item" @click="revalidate(d)">校验</button>
                      <button class="dropdown-item" @click="unbind(d)">解绑</button>
                      <router-link :to="`/datasets?focus=${d.dataset_id}`" class="dropdown-item link">查看详情</router-link>
                      <button class="dropdown-item danger" @click="removeDs(d)">删除</button>
                    </div>
                  </div>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
        <p class="form-hint">说明：本列表只显示归属本算法库的数据集；要把别处或尚未归属的数据集放进来，点上方「挂载已有数据集」。待封板数据集数量达到门槛后可直接封板。</p>
      </div>

      <!-- ---------- C3. 模型版本 ---------- -->
      <div v-else-if="activeTab === 'versions'" class="card wk-pane">
        <div class="wk-pane-head">
          <h2>模型版本（{{ versions.length }}）</h2>
          <button class="secondary small" :disabled="charting" @click="downloadCharts">
            {{ charting ? '生成中...' : '📈 生成训练图表' }}
          </button>
        </div>

        <div v-if="versions.length === 0" class="empty-state">该算法库还没有训练产出的模型版本</div>

        <!-- 主视图：使用中 / 待发布 / 训练中 / 已入库；退役版本折叠到底部「历史版本」 -->
        <div v-for="v in mainVersions" :key="v.model_id || ''" class="ver-card">
          <div class="ver-head">
            <span class="ver-name">{{ verTitle(v) }}<span v-if="v.model_id === latestOutputId && v.version" class="ver-latest">最新</span><span v-if="v.__lineage" class="ver-lineage">血缘上一代</span><span v-if="!v.version" class="ver-latest">上传导入</span></span>
            <span class="status-tag" :class="verBadge(v).cls">{{ verBadge(v).text }}</span>
            <button
              v-if="v.status === 'production_ready' && v.publish_status !== 'published'"
              class="primary small"
              :disabled="publishingId === v.model_id"
              title="发布后本版本成为唯一使用中版本，旧使用中版本自动变为已换代"
              @click="doPublish(v)"
            >{{ publishingId === v.model_id ? '发布中...' : '✓ 确认发布' }}</button>
            <button
              class="secondary small ver-dl"
              :disabled="downloadingVer === v.model_id"
              @click="downloadVersionWeights(v)"
            >{{ downloadingVer === v.model_id ? '下载中...' : '⬇ 下载 .pt' }}</button>
          </div>
          <div v-if="hasVerMetrics(v)" class="ver-metrics">
            <div class="ver-metric">
              <span class="vm-label">mAP50-95</span>
              <b>{{ fmtPct(metricOf(v, 'mAP50_95')) }}</b>
            </div>
            <div class="ver-metric">
              <span class="vm-label">Precision</span>
              <b>{{ fmtPct(metricOf(v, 'precision')) }}</b>
            </div>
            <div class="ver-metric">
              <span class="vm-label">Recall</span>
              <b>{{ fmtPct(metricOf(v, 'recall')) }}</b>
            </div>
            <div class="ver-metric">
              <span class="vm-label">参数量</span>
              <b>{{ v.model_info?.total_params_m ? v.model_info.total_params_m + 'M' : '—' }}</b>
            </div>
          </div>
          <div v-else class="ver-meta">
            <span>手工上传的权重，无训练指标记录（可用于预标注与推理）</span>
          </div>
          <div class="ver-meta">
            <span>产出时间：{{ fmtTime(v.created_at) }}</span>
            <span v-if="v.model_id" title="模型页 / 训练页「查看模型」打开的就是这条记录">模型记录：{{ v.model_id }}</span>
            <span v-if="v.lineage?.job_id" title="训练页任务列表里对应的任务号">训练任务：{{ v.lineage.job_id }}</span>
            <span v-if="parentLabel(v)">基于 {{ parentLabel(v) }} 迭代</span>
            <span v-if="baseLabel(v)">基底：{{ baseLabel(v) }}</span>
          </div>
          <!-- 守门员对比（当前 vs 上一代） -->
          <div v-if="v.gatekeeper" class="gk-compare">
            <div class="gk-box">
              <div class="gk-title">守门员评估</div>
              <div class="gk-report-text">{{ v.gatekeeper.report }}</div>
              <div v-if="v.gatekeeper.regressed_classes?.length" class="gk-regressed">
                退化类别：
                <span v-for="c in v.gatekeeper.regressed_classes" :key="c" class="gk-regressed-item">{{ c }}</span>
              </div>
              <div v-if="deltaOf(v)" class="gk-delta">
                对比上一代：mAP50-95 {{ deltaOf(v)?.deltaText }}
              </div>
            </div>
          </div>
        </div>

        <!-- 退役版本（已换代 / 守门员拦截）默认折叠 -->
        <div v-if="histVersions.length" class="ver-hist-wrap">
          <button class="link-btn ver-hist-toggle" @click="showHistVersions = !showHistVersions">
            {{ showHistVersions ? '▾' : '▸' }} 历史版本（{{ histVersions.length }}）· 已换代 / 守门员拦截
          </button>
          <div v-if="showHistVersions" class="ver-hist-list">
            <div v-for="v in histVersions" :key="v.model_id" class="ver-card ver-card-hist">
              <div class="ver-head">
                <span class="ver-name">{{ verTitle(v) }}</span>
                <span class="status-tag" :class="verBadge(v).cls">{{ verBadge(v).text }}</span>
                <button
                  class="secondary small ver-dl"
                  :disabled="downloadingVer === v.model_id"
                  @click="downloadVersionWeights(v)"
                >{{ downloadingVer === v.model_id ? '下载中...' : '⬇ 下载 .pt' }}</button>
              </div>
              <div class="ver-meta">
                <span>产出时间：{{ fmtTime(v.created_at) }}</span>
                <span v-if="v.model_id">模型记录：{{ v.model_id }}</span>
                <span v-if="v.lineage?.job_id">训练任务：{{ v.lineage.job_id }}</span>
                <span v-if="baseLabel(v)">基底：{{ baseLabel(v) }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ---------- C4. 训练任务历史 ---------- -->
      <div v-else-if="activeTab === 'history'" class="card wk-pane">
        <div class="wk-pane-head">
          <h2>训练任务历史（{{ historyJobs.length }}）</h2>
          <button class="secondary small" :disabled="loadingJobs" @click="loadJobs">刷新</button>
        </div>
        <div v-if="historyJobs.length === 0" class="empty-state">该算法库还没有训练任务</div>
        <table class="ds-table" v-else>
          <thead>
            <tr>
              <th>任务</th>
              <th>状态</th>
              <th>数据集</th>
              <th>模型 / 轮数</th>
              <th>启动时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="j in historyJobs" :key="j.job_id">
              <td><div class="ds-name-main">{{ j.job_id }}</div></td>
              <td>
                <span class="life-pill" :class="`job-${j.status}`">{{ jobStatusText(j.status) }}</span>
              </td>
              <td>{{ j.dataset_id }}</td>
              <td>{{ j.model_name }} · {{ j.epochs }} 轮</td>
              <td>{{ fmtTime(j.created_at) }}</td>
              <td>
                <div class="row-actions">
                  <router-link :to="`/train`"><button class="ghost small">查看</button></router-link>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- ---------- C5. 设置（简介 / 标签字典管理 / 静态信息） ---------- -->
      <div v-else-if="activeTab === 'settings'" ref="settingsPane" class="card wk-pane">
        <div class="wk-pane-head"><h2>算法库设置</h2></div>

        <div class="set-group">
          <div class="set-label">显示名称</div>
          <div class="set-row">
            <input v-model="nameDraft" class="ds-search" style="max-width: 340px;" />
            <button class="secondary small" :disabled="savingDesc" @click="saveName">保存</button>
          </div>
        </div>
        <div class="set-group">
          <div class="set-label">静态信息</div>
          <div class="static-info">
            <div><span>算法库 ID</span><code>{{ libraryRoot?.model_id || model.model_id }}</code></div>
            <div v-if="model.library_root_id && model.library_root_id !== model.model_id"><span>当前查看版本</span><code>{{ model.model_id }}</code></div>
            <div v-if="model.model_code"><span>唯一标识</span><code>{{ model.model_code }}</code></div>
            <div><span>业务标签</span><span>{{ bizLabel }}</span></div>
            <div v-if="model.weights_path"><span>权重路径</span><code class="path">{{ model.weights_path }}</code></div>
            <div><span>创建时间</span><span>{{ fmtTime(model.created_at) }}</span></div>
          </div>
        </div>

        <!-- 危险区 -->
        <div class="set-group danger-zone">
          <div class="set-label">危险操作</div>
          <button class="danger" :disabled="deleting" @click="removeModel">
            {{ deleting ? '删除中...' : '删除算法库' }}
          </button>
          <p class="form-hint">删除算法库将同时清除其模型版本与权重文件，且不可恢复！</p>
        </div>
      </div>

      <!-- ---------- D. 管理标签弹窗（标签字典编辑） ---------- -->
      <div v-if="labelModalOpen" class="modal-mask" @click.self="closeLabelModal">
        <div class="modal-panel">
          <div class="modal-head">
            <h3>管理标签字典（{{ labelsDict.labels?.length || 0 }} 类）</h3>
            <button class="ghost small" @click="closeLabelModal">✕</button>
          </div>
          <p class="form-hint">
            哪些能改、哪些不能改：<b>Class ID</b> 是训练类别号，不能重排；<b>英文名</b> 是标签身份
            （标注、过滤、导出都靠它），<u>只要本库有数据用过它，名字和位置就锁死</u>（格子会变灰、无法输入），
            新类别只能加在最后一行之后；<b>中文名</b> 可以改，但早年只写了中文名的老标注可能认不出类别；
            <b>描述</b> 与 <b>AI 提示词</b> 都是给 AI 预标注用的，<u>随时可改，下一次 AI 标注立即生效</u>：
            描述写给千问 VL（它能读中文），AI 提示词写给 GroundingDINO / YOLO-World（这两个只认英文，
            填中文或拼音等于没填，例如 jishui 要写成 <code>standing water or puddle on the ground</code>）。
          </p>
          <p class="usage-tip">
            <span v-if="usageLoading">正在核对每个标签被多少张图用过...</span>
            <span v-else-if="labelUsage && !labelUsage.dict_locked">
              本库还没有任何数据用上这些标签，英文名可以随便改。
            </span>
            <span v-else-if="labelUsage">
              已有数据在用：第 1 ~ {{ labelUsage.frozen_until + 1 }} 行的英文名与顺序已锁死。
            </span>
          </p>
          <table class="ds-table label-table">
            <thead>
              <tr>
                <th style="width: 80px;">Class ID<div class="th-tag">不可重排</div></th>
                <th>英文名<div class="th-tag">身份标识·在用不可改</div></th>
                <th>中文名<div class="th-tag">可改·老数据有风险</div></th>
                <th>描述<div class="th-tag">千问用·随时可改</div></th>
                <th>AI 提示词（英文）<div class="th-tag">Grounding / YOLO-World 用·随时可改</div></th>
                <th style="width: 60px;"></th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(l, idx) in labelRows" :key="l._tmp || l.index">
                <td>{{ l.index }}</td>
                <td>
                  <input v-model="l.english_code" class="cell-input" :class="{ locked: rowLocked(l) }"
                         :readonly="rowLocked(l)" :title="rowLocked(l) ? '已有数据在用，改名会让已标注整批丢框' : ''"
                         placeholder="如 person" />
                  <span v-if="rowUsage(l)" class="usage-tag" :class="{ used: rowUsage(l).in_use }">{{ usageText(l) }}</span>
                </td>
                <td><input v-model="l.chinese_name" class="cell-input" placeholder="如 行人" /></td>
                <td><input v-model="l.chinese_desc" class="cell-input" placeholder="可选" /></td>
                <td>
                  <input v-model="l.ai_prompt_en" class="cell-input"
                         placeholder="英文，如 standing water or puddle on the ground" />
                </td>
                <td>
                  <button class="ghost small danger" :disabled="rowLocked(l)"
                          :title="rowLocked(l) ? '已被数据使用，不能删除' : ''"
                          @click="labelRows.splice(idx, 1)">删</button>
                </td>
              </tr>
            </tbody>
          </table>
          <div class="set-row modal-actions">
            <button class="ghost small" @click="addLabelRow">+ 添加类别</button>
            <button class="primary small" :disabled="savingLabels" @click="saveLabels">
              {{ savingLabels ? '保存中...' : '保存标签字典' }}
            </button>
            <button class="secondary small" @click="closeLabelModal">关闭</button>
            <span v-if="labelsMsg" class="labels-msg">{{ labelsMsg }}</span>
          </div>

          <!-- 已标注数据的文案同步（改描述只影响之后，老标注里存的是当时那份） -->
          <div class="label-sync">
            <div class="ls-title">已标注数据的文案同步</div>
            <p class="form-hint">
              每张图片的标注里额外存了一份当时的中文名/描述（便于追溯）。改字典不会自动改它们，
              需要统一时手动同步；同步以<b>已保存</b>的标签字典为准，只改文案、不动类别号，训练结果不受影响。
            </p>
            <div class="set-row">
              <button class="secondary small" :disabled="labelSyncing" @click="previewLabelSync">
                {{ labelSyncing ? '统计中...' : '① 预览差异' }}
              </button>
              <button class="secondary small" :disabled="labelSyncing || !hasLabelDiff" @click="runLabelSync">
                ② 同步到已标注数据
              </button>
              <button v-if="labelSyncInfo?.last_backup" class="ghost small" :disabled="labelSyncing" @click="rollbackLabelSyncAction">
                ↩ 回退上次同步
              </button>
            </div>
            <div v-if="labelSyncInfo" class="ls-detail">
              <span v-if="!labelSyncInfo.stats.boxes_changed">
                已扫描 {{ labelSyncInfo.stats.files_scanned }} 张标注图，文案与标签字典一致，无需同步。
              </span>
              <template v-else>
                <div>
                  共 {{ labelSyncInfo.stats.boxes_changed }} 处文案不同，涉及
                  {{ labelSyncInfo.stats.files_changed }} 张图 / {{ labelSyncInfo.stats.datasets }} 个数据集。前 {{ labelSyncInfo.samples.length }} 条示例：
                </div>
                <table class="ds-table ls-table">
                  <thead>
                    <tr>
                      <th>标签</th><th>字段</th><th>标注里的旧文案</th><th>同步后</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(s, i) in labelSyncInfo.samples" :key="i">
                      <td>{{ s.english_code }}</td>
                      <td>{{ s.field }}</td>
                      <td class="ls-before">{{ s.before || '（空）' }}</td>
                      <td class="ls-after">{{ s.after }}</td>
                    </tr>
                  </tbody>
                </table>
              </template>
              <div v-if="labelSyncInfo.last_backup" class="ls-backup">
                上次同步备份：{{ labelSyncInfo.last_backup.backup_id }}
                （{{ labelSyncInfo.last_backup.files }} 个文件 / {{ labelSyncInfo.last_backup.boxes }} 处文案），可点「回退上次同步」还原。
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ---------- E. 挂载已有数据集弹窗 ---------- -->
      <div v-if="mountModalOpen" class="modal-mask" @click.self="mountModalOpen = false">
        <div class="modal-panel">
          <div class="modal-head">
            <h3>挂载已有数据集到本算法库（{{ mountableDatasets.length }} 个可选）</h3>
            <button class="ghost small" @click="mountModalOpen = false">✕</button>
          </div>
          <p class="form-hint">这里列出尚未归属本算法库的数据集。挂载后它会出现在本库的「数据集管理」中；若原本属于其它算法库，则从那个库移出。</p>
          <input v-model="mountKeyword" class="ds-search" placeholder="按名称 / 编号搜索..." />
          <div v-if="mountableDatasets.length === 0" class="empty-state">没有可挂载的数据集，可点「+ 导入新数据」上传</div>
          <table class="ds-table" v-else>
            <thead>
              <tr>
                <th>数据集名称</th>
                <th>当前归属</th>
                <th>阶段</th>
                <th>图片数</th>
                <th style="width: 80px;"></th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="d in mountableDatasets" :key="d.dataset_id">
                <td>
                  <div class="ds-name-main">{{ d.display_name || d.filename || d.dataset_id }}</div>
                  <div class="ds-sub">{{ d.dataset_id }}</div>
                </td>
                <td>{{ ownedLibraryNameOf(d) }}</td>
                <td><span class="life-pill" :class="`life-${lifecycleOf(d).key}`">{{ lifecycleOf(d).text }}</span></td>
                <td>{{ d.image_count ?? '—' }}</td>
                <td><button class="primary small" @click="mountDataset(d)">挂载</button></td>
              </tr>
            </tbody>
          </table>
          <div class="set-row modal-actions">
            <button class="secondary small" @click="mountModalOpen = false">关闭</button>
          </div>
        </div>
      </div>

      <!-- ---------- F. 导入新数据弹窗（留在本页，归属自动是本库） ---------- -->
      <div v-if="dsImportOpen" class="modal-mask" @click.self="dsImportOpen = false">
        <div class="modal-panel">
          <div class="modal-head">
            <h3>导入新数据到本算法库</h3>
            <button class="ghost small" @click="dsImportOpen = false">✕</button>
          </div>
          <p class="form-hint">
            归属算法库：<b>{{ model?.display_name || model?.name || modelId }}</b>
            —— 导入的数据集会自动出现在下方列表中，无需跳转到「数据集」页。
          </p>
          <div class="ds-file-box">
            <div v-if="dsImportFile" class="ds-selected">
              📦 {{ dsImportFile.name }}
              <span class="ds-selected-size">({{ (dsImportFile.size / 1024 / 1024).toFixed(2) }} MB)</span>
            </div>
            <div v-else class="ds-selected empty">尚未选择文件（支持 .zip / .tar / .tar.gz / .tgz）</div>
            <input id="dsImportZip" type="file" accept=".zip,.tar,.tgz,.tar.gz" class="ds-file-input"
                   :disabled="dsImporting" @change="onDsFilePick" />
            <label for="dsImportZip" class="ds-file-btn">选择压缩包</label>
          </div>
          <p class="form-hint">
            导入后自动完成准备（解压 / 标注转换 / 图片标签配对 / 统计）：VOC(XML)、COCO(JSON) 标注会自动转成
            YOLO 格式，文件夹结构不统一时自动扫描重组；只有原图没有标注时也会入池，之后到标注页画框即可。
          </p>
          <div class="set-row modal-actions">
            <button class="primary small" :disabled="!dsImportFile || dsImporting" @click="doImportDataset">
              <span v-if="dsImporting" class="loading-spinner"></span>
              {{ dsPreparing ? '正在准备数据...' : (dsImporting ? '上传中...' : '开始导入') }}
            </button>
            <button class="secondary small" :disabled="dsImporting" @click="dsImportOpen = false">取消</button>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import {
  getModel, getModelDatasets, updateModel, exportModel,
  getLabelsDict, updateLabelsDict, generateTrainingCharts, listModels,
  uploadModel,
  syncLabelsToAnnotations, rollbackLabelSync, getLabelsUsage, confirmPublish,
  type ModelDataset, type LabelsDict, type LabelSyncResult, type LabelsUsage
} from '@/api/models'
import {
  listDatasets, bindDatasetModel, checkDatasetBind, validateDataset, sealDataset, deleteDataset,
  uploadDataset, prepareDataset
} from '@/api/datasets'
import { listTrainJobs } from '@/api/train'
import { deleteModelWithGuard, showConfirm } from '@/composables/useDialog'
import { showToast } from '@/utils/toast'

const route = useRoute()
const modelId = String(route.params.modelId)

const loading = ref(true)
const model = ref<any>(null)
const allModels = ref<any[]>([])
const mountedDatasets = ref<ModelDataset[]>([])
const allDatasets = ref<ModelDataset[]>([])
const labelsDict = ref<LabelsDict>({ labels: [] })
const historyJobs = ref<any[]>([])
const loadingJobs = ref(false)
const sealMinImages = ref(0)

// ===== Header / 概览 =====
const editDescOpen = ref(false)
const descDraft = ref('')
const savingDesc = ref(false)
const descSaveMsg = ref('')
const descSaveErr = ref(false)
const nameDraft = ref('')
const charting = ref(false)
const importFileInput = ref<HTMLInputElement | null>(null)

// ===== Tab =====
const activeTab = ref('overview')

// 设置定位与标签弹窗（二取一：⚙ 设置 → 定位设置区；管理标签 → 弹窗就地编辑）
const settingsPane = ref<HTMLElement | null>(null)
const labelModalOpen = ref(false)

function goSettings() {
  activeTab.value = 'settings'
  nextTick(() => settingsPane.value?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
}

function openLabelModal() {
  syncLabelRows()
  labelSyncInfo.value = null
  labelUsage.value = null
  labelModalOpen.value = true
  loadLabelUsage() // 不阻塞弹窗：统计回来后再把在用的英文名格子锁住
}

function closeLabelModal() {
  labelModalOpen.value = false
  labelSyncInfo.value = null
  syncLabelRows() // 关闭即回滚未保存的编辑
}

const normVer = (v?: string) => (v && !String(v).toLowerCase().startsWith('v') ? 'v' + v : v || '—')

const archOf = (m: any) => {
  return m?.architecture_name || (/(yolov?\d+[a-z]*?)([nslmxbce])?/i.exec(m?.model_name || '') || []).join('')
}
const archLabel = computed(() => {
  const a = archOf(model.value)
  const s = model.value?.model_scale
  return a ? `${a}${s ? '-' + s : ''}` : '未指定'
})
const labelVersionLabel = computed(() => {
  const n = labelsDict.value.labels?.length ?? 0
  const ver = model.value?.version
  return `${ver ? 'v' + String(ver).replace(/^v/i, '') : '—'} (${n}类)`
})
const latestMAP = computed(() => {
  // 以库内最新版本为准（库详情页打开的是"库"，其自身记录可能没有训练指标）
  const cur = newestVersion.value || model.value
  const g = cur?.gatekeeper
  const m = g?.new_metrics?.mAP50_95 ?? cur?.training_metrics?.training_history?.final_metrics?.mAP50_95
  return m != null ? fmtPct(m) : ''
})
const bizLabel = computed(() => {
  const v = model.value?.business
  const map: Record<string, string> = {
    general: '通用检测', pedestrian: '行人检测', vehicle: '车辆识别', defect: '高精度质检', package: '包裹分拣'
  }
  return (v && (map[v as string] || v)) || '—'
})

function statusLabel(status?: string): string {
  if (!status) return '未知'
  const map: Record<string, string> = {
    production_ready: '生产就绪', rejected: '守门员拦截', superseded: '已换代',
    training: '训练中', evaluating: '评估中', active: '已入库', inactive: '停用'
  }
  return map[status] || status
}

function fmtTime(t?: string | null): string {
  if (!t) return '—'
  const d = new Date(t)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

function fmtPct(v: number | null | undefined): string {
  if (v == null) return '—'
  return (v >= 0 && v <= 1.5 ? (v * 100).toFixed(1) + '%' : Number(v).toFixed(2))
}

function metricOf(v: any, key: 'mAP50_95' | 'precision' | 'recall') {
  return v?.gatekeeper?.new_metrics?.[key] ??
    v?.training_metrics?.training_history?.final_metrics?.[key] ?? null
}

// 手工上传的记录没有 version / 训练指标，全走 normVer() 会渲染成一排破折号，
// 标题回退到显示名，指标全空时整行不渲染（改由文案说明）。
const verTitle = (v: any) =>
  v?.version ? normVer(v.version) : (v?.display_name || v?.model_id || '未命名模型')

// ===== 血缘/溯源可读化：让 v1.1/v1.2 能对上「训练页的任务」和「基底模型」 =====
const modelById = (id?: string | null): any =>
  id ? allModels.value.find((m: any) => m.model_id === id) || null : null
// 上一代（微调父）→ 「名字（model_id）」；找不到记录时退回原始 id
const parentLabel = (v: any) => {
  const pid = v?.lineage?.parent_model_id
  if (!pid) return ''
  const p = modelById(pid)
  return p ? `${verTitle(p)}（${pid}）` : String(pid)
}
// 基底权重路径 → 人话：库内某模型的权重 / 某次训练任务的产物 / 预训练权重
const baseLabel = (v: any) => {
  const bm = String(v?.lineage?.base_model || '')
  if (!bm) return ''
  const reg = bm.match(/models\/registry\/([^/]+)\//)
  if (reg) {
    const m = modelById(reg[1])
    return m ? `${verTitle(m)}（${m.model_id}）的权重` : `${reg[1]} 的权重`
  }
  const job = bm.match(/jobs\/([^/]+)\/train\//)
  if (job) {
    const m = allModels.value.find((x: any) => x.lineage?.job_id === job[1])
    return m ? `训练任务 ${job[1]} 产出的 ${verTitle(m)}（${m.model_id}）` : `训练任务 ${job[1]} 的产物`
  }
  return `预训练权重 ${bm.split('/').pop() || bm}`
}
const hasVerMetrics = (v: any) =>
  ['mAP50_95', 'precision', 'recall'].some((k) => metricOf(v, k as any) != null) ||
  !!v?.model_info?.total_params_m

const deltaOf = (v: any) => {
  const g = v?.gatekeeper
  const old = g?.old_metrics?.mAP50_95
  const neu = g?.new_metrics?.mAP50_95
  if (old == null || neu == null) return null
  const d = (neu - old) * 100
  return { deltaText: `${d >= 0 ? '+' : ''}${d.toFixed(1)}%` }
}

// ===== 数据集管理 =====
const dsKeyword = ref('')
const dsFilter = ref('')

type Lifecycle = { key: string; text: string }

function lifecycleOf(d: any): Lifecycle {
  if (!d.model_id) return { key: 'unassigned', text: '未分配' }
  if (d.stage === 'failed') return { key: 'failed', text: '训练失败' }
  if (['sealed', 'training', 'completed'].includes(d.stage)) return { key: 'sealed', text: '已封板' }
  if (d.stage === 'annotating') {
    const hasData = d.annotated || Number(d.label_count || 0) > 0 || Number(d.image_count || 0) > 0
    return hasData && d.status === 'prepared'
      ? { key: 'pending', text: '待封板' }
      : { key: 'annotating', text: '标注中' }
  }
  if (d.annotated) return { key: 'pending', text: '待封板' }
  return { key: 'todo', text: '待标注' }
}

function mainActionOf(d: any): { label: string; to: string } {
  const lc = lifecycleOf(d)
  switch (lc.key) {
    case 'todo': case 'annotating': case 'failed':
      return { label: '去标注', to: `/annotate?dataset_id=${d.dataset_id}` }
    case 'pending': return { label: '封板', to: '' }
    case 'sealed': return { label: '训练', to: `/train?dataset_ids=${d.dataset_id}` }
    default: return { label: '查看', to: '' }
  }
}

// 库详情的数据集列表只放「归属本库」的数据集（后端按库根聚合）；
// 未归属的数据集不再混进每个库，改由「挂载已有数据集」弹窗按需拉入本库
const datasetsView = computed<ModelDataset[]>(() => mountedDatasets.value)

// 数据集当前归属的算法库名（用于挂载弹窗提示，避免误把他库数据拉走）
function ownedLibraryNameOf(d: any): string {
  const id = d?.library_root_id || d?.model_id || ''
  if (!id) return '未归属'
  const m = allModels.value.find((x: any) => (x.library_root_id || x.model_id) === x.model_id && x.model_id === id)
  return m ? (m.display_name || m.model_code || id) : '未归属'
}

// ===== 挂载已有数据集（把未归属/其它库的数据集改挂到本库） =====
const mountModalOpen = ref(false)
const mountKeyword = ref('')

const mountableDatasets = computed(() => {
  const inLibrary = new Set(mountedDatasets.value.map((d) => d.dataset_id))
  const kw = mountKeyword.value.trim().toLowerCase()
  return allDatasets.value
    .filter((d) => !inLibrary.has(d.dataset_id))
    .filter((d) => {
      if (!kw) return true
      return `${d.dataset_id} ${d.display_name || ''} ${d.filename || ''}`.toLowerCase().includes(kw)
    })
})

function openMountModal() {
  mountKeyword.value = ''
  mountModalOpen.value = true
}

async function mountDataset(d: any) {
  await bindCurrent(d)
  mountModalOpen.value = false
}

// ===== 页内导入新数据：归属自动是本库，不用跳到「数据集」页 =====
const dsImportOpen = ref(false)
const dsImportFile = ref<File | null>(null)
const dsImporting = ref(false)
const dsPreparing = ref(false)

function openDsImportModal() {
  dsImportFile.value = null
  dsImporting.value = false
  dsPreparing.value = false
  dsImportOpen.value = true
}

function onDsFilePick(e: Event) {
  const target = e.target as HTMLInputElement
  if (target.files && target.files[0]) dsImportFile.value = target.files[0]
}

async function doImportDataset() {
  const file = dsImportFile.value
  if (!file || dsImporting.value) return
  const libName = model.value?.display_name || model.value?.name || modelId
  dsImporting.value = true
  try {
    const res = await uploadDataset(file, modelId)
    dsPreparing.value = true
    try {
      await prepareDataset(res.dataset_id)
      dsImportOpen.value = false
      dsImportFile.value = null
      await refresh()
      showToast(`导入成功：数据集已归属算法库「${libName}」并完成准备`, 'info', 4000)
    } catch (e: any) {
      // 上传成功但准备失败：数据集已入库，刷新后可在列表里重试准备
      await refresh()
      if (!e?.__toastShown) showToast('上传成功，但自动准备失败：' + (e?.response?.data?.detail || e?.message) + '\n数据集已保留在列表中，可稍后重新准备。', 'error', 4000)
    } finally {
      dsPreparing.value = false
    }
  } catch (e: any) {
    if (!e?.__toastShown) showToast('导入失败：' + (e?.response?.data?.detail || e?.message), 'error', 4000)
  } finally {
    dsImporting.value = false
  }
}

const filteredDatasets = computed(() => {
  const kw = dsKeyword.value.trim().toLowerCase()
  return datasetsView.value.filter((d) => {
    if (dsFilter.value && lifecycleOf(d).key !== dsFilter.value) return false
    if (!kw) return true
    const hay = `${d.dataset_id} ${d.display_name || ''} ${d.filename || ''}`.toLowerCase()
    return hay.includes(kw)
  })
})

const moreOpen = ref('')
function toggleMore(d: any) {
  moreOpen.value = moreOpen.value === d.dataset_id ? '' : d.dataset_id
}

async function onMainAction(d: any) {
  const lc = lifecycleOf(d)
  if (lc.key === 'pending') await sealAction(d)
}

async function sealAction(d: any) {
  const cnt = Number(d.label_count || 0)
  const within = sealMinImages.value > 0 && cnt >= sealMinImages.value
  const msg = `数据集 ${d.dataset_id} 共 ${cnt} 张标注，${within ? '达到封板门槛，确认封板？' : `未达到封板门槛（${sealMinImages.value} 张），确认强制封板？`}`
  if (!(await showConfirm(msg))) return
  try {
    await sealDataset(d.dataset_id, !within)
    showToast('已封板：数据集将作为该模型的训练数据（只读）。', 'info', 4000)
    await refresh()
  } catch (e: any) {
    if (!e?.__toastShown) showToast('封板失败：' + (e?.response?.data?.detail || e?.message), 'error', 4000)
  }
}

async function bindCurrent(d: any) {
  const libName = model.value?.display_name || model.value?.name || modelId
  const owner = ownedLibraryNameOf(d)
  const base = owner === '未归属'
    ? `将数据集 ${d.display_name || d.dataset_id} 挂载到算法库「${libName}」？`
    : `数据集 ${d.display_name || d.dataset_id} 当前归属算法库「${owner}」，确认改挂到「${libName}」？（原算法库的数据集列表将不再显示它）`
  // 先问后端：这个数据集已经画好的框，换到本库后类别序号会不会错位
  // （两个库有同名英文标签但排列顺序不同 → 封板时框会被认成别的类）
  let check: any = null
  try {
    check = await checkDatasetBind(d.dataset_id, modelId)
  } catch {
    check = null // 预览失败不挡路：真正挂载时后端还会再校验一次
  }
  let force = false
  let msg = base
  if (check?.conflict) {
    msg = `${base}\n\n⚠ 有串号风险，请先看清再决定：\n${check.summary}\n\n仍要挂载吗？（拿不准就点「取消」，先到本算法库把标签顺序调成一致）`
    force = true
  } else if (check && (check.warnings || []).length) {
    msg = `${base}\n\n${check.summary}`
  }
  if (!(await showConfirm(msg))) return
  try {
    await bindDatasetModel(d.dataset_id, modelId, force)
    await refresh()
  } catch (e: any) {
    if (!e?.__toastShown) showToast('挂载失败：' + (e?.response?.data?.detail || e?.message), 'error', 4000)
  }
}

async function revalidate(d: ModelDataset) {
  try {
    const res = await validateDataset(d.dataset_id)
    const ok = res?.status === 'ok'
    if (ok) {
      showToast(`校验通过：${res.checked_files} 个标注文件 / ${res.lines_checked} 行标注均符合 YOLO 格式`, 'info', 4000)
    } else {
      const errs = (res?.errors || []).slice(0, 5).join('\n')
      showToast(`校验未通过：${res.invalid_lines} 行异常（${res.invalid_files} 个文件）\n示例：\n${errs || '(无详情)'}`, 'info', 4000)
    }
    await refresh()
  } catch (e: any) {
    if (!e?.__toastShown) showToast('校验失败：' + (e?.response?.data?.detail || e?.message || '未知错误'), 'error', 4000)
  }
}

async function unbind(d: ModelDataset) {
  if (!(await showConfirm(`解绑数据集 ${d.dataset_id}？它将不再归属当前算法库。`))) return
  try {
    await bindDatasetModel(d.dataset_id, null)
    await refresh()
  } catch (e: any) {
    if (!e?.__toastShown) showToast('解绑失败：' + (e?.response?.data?.detail || e?.message), 'error', 4000)
  }
}

async function removeDs(d: ModelDataset) {
  if (!(await showConfirm(`删除数据集 ${d.dataset_id}？此操作不可恢复！`))) return
  try {
    await deleteDataset(d.dataset_id)
    await refresh()
  } catch (e: any) {
    if (!e?.__toastShown) showToast('删除失败：' + (e?.response?.data?.detail || e?.message), 'error', 4000)
  }
}

// ===== 模型版本 =====
// 库根记录：训练页「查看模型」等入口可能直接打开库下某个版本记录（/models/{versionId}），
// 此时页面标题/简介必须展示「归属算法库」的信息，而不是兜底成版本记录自己的 model_id
const libraryRoot = computed<any>(() => {
  const cur: any = model.value
  if (!cur) return null
  const rootId = cur.library_root_id || cur.model_id
  return allModels.value.find((x: any) => x.model_id === rootId) || cur
})
// 版本号 → 可比较数字（v1.10 → 1.1）；空库占位记录没有版本号，视为最旧排在最后
const verNum = (v?: string) => {
  const n = parseFloat(String(v || '').replace(/^v/i, ''))
  return Number.isFinite(n) ? n : -1
}

const versions = computed(() => {
  const cur: any = model.value
  if (!cur) return []
  // 本库的全部训练产出：带同一「库根」标识的记录归属同一个算法库。
  // 首页「+ 新建算法库」创建的空白占位记录（empty=true）不是训练产出，不能当版本展示。
  const root = cur.library_root_id || cur.model_id
  const list: any[] = allModels.value.filter(
    (m: any) => !m.empty && (m.library_root_id || m.model_id) === root
  )
  // 兼容改动前的历史数据（产出没有库标识）：按血缘把上一代补进来。
  // 补入的记录打上 __lineage 标记：它们不属于本库产出，不得挂「最新」徽标
  const parentId = cur.lineage?.parent_model_id
  if (parentId && !list.some((m) => m.model_id === parentId)) {
    const p = allModels.value.find((m: any) => m.model_id === parentId)
    if (p && !p.empty) list.push({ ...p, __lineage: true })
  }
  // 倒序（最新在前）：先比版本号，版本号相同/缺失时比产出时间
  return list.sort((a: any, b: any) => {
    const d = verNum(b.version) - verNum(a.version)
    if (d !== 0) return d
    return String(b.created_at || '').localeCompare(String(a.created_at || ''))
  })
})
const newestVersion = computed(() => versions.value[0] || null)
// 「最新」徽标只挂在本库最新的训练产出上（按产出时间比，不比版本号）：
// 血缘补入的历史模型（__lineage）版本号可能更大（如老链路的 v1.3），按版本号排会抢走徽标
const latestOutputId = computed(() => {
  const outs = versions.value.filter((v: any) => !v.__lineage && v.version)
  if (!outs.length) return null
  return [...outs]
    .sort((a: any, b: any) => String(b.created_at || '').localeCompare(String(a.created_at || '')))[0]?.model_id || null
})

// ===== 版本状态简化展示：每卡只挂 1 个徽标，退役版本折叠 =====
// 状态机不变（active/production_ready/superseded/rejected/training/evaluating + publish_status），
// 只把「生产中」改叫「使用中」（用户只关心哪个正在用），待确认发布叫「待发布」并给一键发布。
function verBadge(v: any): { text: string; cls: string } {
  if (v.status === 'production_ready') {
    return v.publish_status === 'published'
      ? { text: '✓ 使用中', cls: 'using' }
      : { text: '待发布', cls: 'pending' }
  }
  if (v.status === 'superseded') return { text: '已换代', cls: 'superseded' }
  if (v.status === 'rejected') return { text: '守门员拦截', cls: 'rejected' }
  if (v.status === 'training') return { text: '训练中', cls: 'training' }
  if (v.status === 'evaluating') return { text: '评估中', cls: 'evaluating' }
  return { text: '已入库', cls: 'active' }  // active：手工上传权重，可下载/预标注，未进生产流程
}
function isHistVer(v: any): boolean {
  return v.status === 'superseded' || v.status === 'rejected'
}
const showHistVersions = ref(false)
const histVersions = computed(() => versions.value.filter(isHistVer))
// 主视图排序：使用中置顶 → 其余按版本号新到旧
const mainVersions = computed(() => {
  const main = versions.value.filter((v: any) => !isHistVer(v))
  return main.sort((a: any, b: any) => {
    const au = a.status === 'production_ready' && a.publish_status === 'published' ? 0 : 1
    const bu = b.status === 'production_ready' && b.publish_status === 'published' ? 0 : 1
    if (au !== bu) return au - bu
    return 0
  })
})
const publishingId = ref<string | null>(null)
async function doPublish(v: any) {
  const cur = versions.value.find((x: any) => x.status === 'production_ready' && x.publish_status === 'published')
  const extra = cur ? `当前使用中版本为 ${verTitle(cur)}，发布后它将变为「已换代」。` : ''
  if (!(await showConfirm(`确认将 ${verTitle(v)} 发布为使用中版本？${extra}`))) return
  publishingId.value = v.model_id
  try {
    await confirmPublish(v.model_id)
    await refresh()
  } catch (e: any) {
    if (!e?.__toastShown) showToast('发布失败：' + (e?.response?.data?.detail || e?.message), 'error', 4000)
  } finally {
    publishingId.value = null
  }
}

// ===== Tab（依赖引用于此处定义之后） =====
const wkTabs = computed(() => [
  { key: 'overview', label: '概览' },
  { key: 'datasets', label: `数据集管理 (${datasetsView.value.length})` },
  { key: 'versions', label: `模型版本 (${versions.value.length})` },
  { key: 'history', label: `训练任务历史 (${historyJobs.value.length})` },
  { key: 'settings', label: '设置' }
])

// ===== 训练任务历史 =====
async function loadJobs() {
  loadingJobs.value = true
  try {
    const res = await listTrainJobs()
    const jobs = res.jobs || []
    const rela = model.value?.lineage?.job_id
    historyJobs.value = jobs.filter((j: any) =>
      j.model_id === modelId || j.base_model_id === modelId || j.job_id === rela
    ).sort((a: any, b: any) => String(b.created_at).localeCompare(String(a.created_at)))
  } catch (e) {
    console.error('加载训练任务失败:', e)
  } finally {
    loadingJobs.value = false
  }
}

function jobStatusText(s: string): string {
  const map: Record<string, string> = {
    running: '训练中', queued: '排队中', completed: '已完成',
    failed: '失败', crashed: '崩溃', stopped: '已停止',
    ready: '就绪', evaluating: '评估中'
  }
  return map[s] || s
}

// ===== 设置：标签字典编辑 =====
const labelRows = ref<any[]>([])
let labelTmpSeq = 0
function addLabelRow() {
  labelRows.value.push({ _tmp: `new_${labelTmpSeq++}`, index: labelRows.value.length, english_code: '', chinese_name: '', chinese_desc: '', ai_prompt_en: '' })
}
function syncLabelRows() {
  labelRows.value = (labelsDict.value.labels || []).map((l) => ({ ...l }))
}

// 标签使用情况（打开弹窗时向后端要一次）：决定哪些英文名锁死、显示 已用/未用
const labelUsage = ref<LabelsUsage | null>(null)
const usageLoading = ref(false)
const usageByCode = computed<Record<string, LabelsUsage['labels'][number]>>(() => {
  const map: Record<string, LabelsUsage['labels'][number]> = {}
  for (const row of labelUsage.value?.labels || []) {
    map[String(row.english_code || '').trim().toLowerCase()] = row
  }
  return map
})
function rowUsage(l: any) {
  return usageByCode.value[String(l?.english_code || '').trim().toLowerCase()]
}
// 只锁"字典里原有、且不晚于最后一个在用类别"的行（新加的行不在统计里，永远可编辑）
function rowLocked(l: any): boolean {
  if (l?._tmp) return false
  return !!rowUsage(l)?.locked
}
function usageText(l: any): string {
  const u = rowUsage(l)
  if (!u) return ''
  if (u.images) return `已用 · ${u.images} 张图画了框`
  if (u.datasets) return `已用 · ${u.datasets} 个数据集导入过（还没画框）`
  return u.locked ? '未用 · 排在已用类别前面，位置不能动' : '未用 · 可改可删'
}
async function loadLabelUsage() {
  usageLoading.value = true
  try {
    labelUsage.value = await getLabelsUsage(modelId)
  } catch {
    labelUsage.value = null // 统计失败不影响查看/改文案，保存时后端仍会拦
  } finally {
    usageLoading.value = false
  }
}

async function saveLabels() {
  savingLabels.value = true
  labelsMsg.value = ''
  try {
    const rows = labelRows.value.map((r, i) => ({
      index: i,
      english_code: (r.english_code || '').trim(),
      chinese_name: (r.chinese_name || '').trim() || undefined,
      chinese_desc: (r.chinese_desc || '').trim() || undefined,
      // 英文检测提示词：给 GroundingDINO / YOLO-World 用，留空则后端回退到英文名
      ai_prompt_en: (r.ai_prompt_en || '').trim() || undefined
    })).filter((r) => r.english_code)
    console.log('🔍 [saveLabels] 发送给后端的完整数据:', JSON.stringify(rows, null, 2))
    const result = await updateLabelsDict(modelId, rows)
    console.log('✅ [saveLabels] 后端返回:', JSON.stringify(result, null, 2))
    labelsMsg.value = '✓ 标签字典已保存'
    labelModalOpen.value = false
    await refresh()
  } catch (e: any) {
    console.error('❌ [saveLabels] 错误:', e?.response?.data || e.message)
    labelsMsg.value = '保存失败：' + (e?.response?.data?.detail || e?.message)
  } finally {
    savingLabels.value = false
  }
}

const savingLabels = ref(false)
const labelsMsg = ref('')

// ===== 已标注数据的文案同步：把字典里的中文名/描述刷进本库各图的标注 =====
const labelSyncing = ref(false)
const labelSyncInfo = ref<LabelSyncResult | null>(null)
const hasLabelDiff = computed(() => (labelSyncInfo.value?.stats?.boxes_changed || 0) > 0)

function labelSyncErr(e: any): string {
  return e?.response?.data?.detail || e?.message || String(e)
}

async function previewLabelSync() {
  if (labelSyncing.value) return
  labelSyncing.value = true
  labelsMsg.value = ''
  try {
    const res = await syncLabelsToAnnotations(modelId, true)
    labelSyncInfo.value = res
    labelsMsg.value = res.stats.boxes_changed
      ? `发现 ${res.stats.boxes_changed} 处文案与标签字典不一致，可执行同步`
      : `已扫描 ${res.stats.files_scanned} 张标注图，文案一致，无需同步`
  } catch (e: any) {
    labelSyncInfo.value = null
    labelsMsg.value = '统计失败：' + labelSyncErr(e)
  } finally {
    labelSyncing.value = false
  }
}

async function runLabelSync() {
  if (labelSyncing.value || !hasLabelDiff.value) return
  const n = labelSyncInfo.value?.stats?.boxes_changed || 0
  if (!(await showConfirm(`将改写本库已标注数据中 ${n} 处文案（中文名/描述），类别号不动。\n原文件会自动备份，可一键回退。确认同步？`))) return
  labelSyncing.value = true
  labelsMsg.value = ''
  try {
    const res = await syncLabelsToAnnotations(modelId, false)
    labelSyncInfo.value = res
    labelsMsg.value = `✓ 已同步 ${res.stats.boxes_changed} 处文案（${res.stats.files_changed} 个文件），备份号 ${res.backup_id || '—'}`
  } catch (e: any) {
    labelsMsg.value = '同步失败：' + labelSyncErr(e)
  } finally {
    labelSyncing.value = false
  }
}

async function rollbackLabelSyncAction() {
  if (labelSyncing.value) return
  const id = labelSyncInfo.value?.last_backup?.backup_id || ''
  if (!(await showConfirm(`回退到上次同步之前？将用备份覆盖 ${labelSyncInfo.value?.last_backup?.files || 0} 个标注文件（备份号 ${id}）。`))) return
  labelSyncing.value = true
  labelsMsg.value = ''
  try {
    const res = await rollbackLabelSync(modelId)
    labelsMsg.value = `✓ 已还原 ${res.restored_files} 个标注文件`
    labelSyncInfo.value = null
    labelSyncing.value = false // 释放状态后再重新统计，确认已回到同步前
    await previewLabelSync()
  } catch (e: any) {
    labelsMsg.value = '回退失败：' + labelSyncErr(e)
  } finally {
    labelSyncing.value = false
  }
}

// ===== 主操作 =====
function goTrain() {
  const ids = mountedDatasets.value.map((d) => d.dataset_id)
  const qs = new URLSearchParams()
  if (ids.length) qs.set('dataset_ids', ids.join(','))
  // 带上本库标识：训练产出直接归入这个算法库，成为库下的新版本
  qs.set('library_root_id', String(model.value?.library_root_id || modelId))
  window.location.href = `/train?${qs.toString()}`
}

async function handleImportModel(e: any) {
  const file = e.target.files?.[0]
  if (!file) return
  // 详情页可能打开的是库下某个版本记录，必须归到「库根」，否则会在首页多挂出一张库卡
  const rootId = String(model.value?.library_root_id || modelId)
  try {
    // 首选上传即归库（一次写盘，不存在"传完了再补挂"的中间态）
    const res = await uploadModel(file, '', rootId)
    const name = res?.display_name || res?.name || res?.file_name || file.name
    let bound = !!res && (res.library_root_id || res.model_library_id) === rootId
    if (!bound && res?.model_id) {
      // 兜底：旧后端会静默忽略 library_id 字段，这里补一次归库
      try {
        await updateModel(res.model_id, { model_library_id: rootId })
        bound = true
      } catch {
        bound = false
      }
    }
    // 归库没成就是要如实说，否则用户会在库里找不到模型、以为导入丢了
    showToast(
      bound
        ? '导入成功：' + name + '（已归入本算法库）'
        : '导入成功：' + name + '，但归入本算法库未生效，请到模型管理页手动归库（后端需重启以启用上传即归库）',
      'info',
      4000
    )
    await refresh()
  } catch (err: any) {
    if (!err?.__toastShown) showToast('导入失败：' + (err?.response?.data?.detail || err?.message), 'error', 4000)
  } finally {
    e.target.value = ''
  }
}

// 外层 zip 名可读化：算法库名_版本_产出日期（如 安全帽检测_v1.2_20261028.zip）；
// display_name/model_code 本身可读（非 model_日期_时间 的 id 形态）时优先用它；
// 兜底 model_id。磁盘存储名保持 best.pt 不变（预标注引擎按约定路径找它）
const downloadingVer = ref<string | null>(null)
const ID_LIKE_RE = /^model[_-]?\d{8}[_-]?\d{6}/i
function sanitizeFilePart(s: string): string {
  return String(s || '').trim().replace(/[\\/:*?"<>|\s]+/g, '_')
}
function datePartOf(t?: string | null): string {
  if (!t) return ''
  const d = new Date(t)
  if (isNaN(d.getTime())) return ''
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}`
}
function exportZipName(v: any): string {
  const lib = sanitizeFilePart(
    libraryRoot.value?.display_name || libraryRoot.value?.name || libraryRoot.value?.model_code || ''
  )
  const readable = (s: string) => !!s && s !== v.model_id && !ID_LIKE_RE.test(s)
  const dn = String(v.display_name || '').trim()
  const code = String(v.model_code || '').trim()
  const base = readable(dn) ? dn : readable(code) ? code : (lib || v.model_id)
  const parts = [base]
  const ver = String(v.version || '').trim()
  if (ver) {
    const vp = ver.toLowerCase().startsWith('v') ? ver : 'v' + ver
    if (!base.toLowerCase().includes(vp.toLowerCase()) && !base.toLowerCase().includes(vp.slice(1).toLowerCase())) {
      parts.push(vp)
    }
  }
  const dp = datePartOf(v.created_at)
  if (dp && !base.includes(dp)) parts.push(dp)
  return parts.join('_') + '.zip'
}
async function downloadVersionWeights(v: any) {
  if (!v?.model_id) return
  downloadingVer.value = v.model_id
  try {
    const blob = await exportModel(v.model_id)
    saveBlob(blob, exportZipName(v))
  } catch (e: any) {
    if (!e?.__toastShown) showToast('下载失败：' + (e?.response?.data?.detail || e?.message), 'error', 4000)
  } finally {
    downloadingVer.value = null
  }
}

async function downloadCharts() {
  charting.value = true
  try {
    const blob = await generateTrainingCharts(modelId, 'all')
    window.open(URL.createObjectURL(blob), '_blank')
  } catch (e: any) {
    if (!e?.__toastShown) showToast('图表生成失败（可能该模型尚无训练结果）：' + (e?.response?.data?.detail || e?.message), 'error', 4000)
  } finally {
    charting.value = false
  }
}

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

// ===== 简介 / 名称保存 =====
// 详情页可能打开的是库下某个版本记录：显示的是库根的名称/简介，保存也必须写回库根
const rootEditId = computed(() => model.value?.library_root_id || modelId)

function saveDescription() {
  savingDesc.value = true
  descSaveMsg.value = ''
  descSaveErr.value = false
  updateModel(rootEditId.value, { description: descDraft.value })
    .then(async () => {
      descSaveMsg.value = '✓ 简介已保存'
      await refresh()
      editDescOpen.value = false
    })
    .catch((e: any) => {
      descSaveErr.value = true
      descSaveMsg.value = '保存失败：' + (e?.response?.data?.detail || e?.message)
    })
    .finally(() => { savingDesc.value = false })
}

function saveName() {
  savingDesc.value = true
  descSaveMsg.value = ''
  descSaveErr.value = false
  updateModel(rootEditId.value, { display_name: nameDraft.value, name: nameDraft.value })
    .then(async () => {
      descSaveMsg.value = '✓ 名称已保存'
      await refresh()
    })
    .catch((e: any) => {
      descSaveErr.value = true
      descSaveMsg.value = '保存失败：' + (e?.response?.data?.detail || e?.message)
    })
    .finally(() => { savingDesc.value = false })
}

const deleting = ref(false)
async function removeModel() {
  if (!(await showConfirm('确定删除该算法库及其全部模型版本与权重文件？此操作不可恢复！'))) return
  deleting.value = true
  try {
    // P2-5：挂接非空时后端不删，守卫流程弹清单让用户选「转移归属 / 保留未归属」
    const name = model.value?.display_name || model.value?.name || modelId
    if (await deleteModelWithGuard(modelId, name) !== 'deleted') return
    showToast('算法库已删除', 'success')
    window.location.href = '/'
  } catch (e: any) {
    if (!e?.__toastShown) showToast('删除失败：' + (e?.response?.data?.detail || e?.message), 'error', 4000)
  } finally {
    deleting.value = false
  }
}

// ===== 刷新 =====
async function refresh() {
  loading.value = true
  try {
    // 6 个接口各自独立，串行 await 会把 65~170ms 逐个累加成 1~2s；改并行后总耗时≈最慢的一个
    const [m, dsRes, allRes, ld, lm] = await Promise.all([
      getModel(modelId),
      getModelDatasets(modelId),
      listDatasets(),
      getLabelsDict(modelId).catch(() => null),      // 标签字典可能尚未初始化
      listModels().catch(() => null),                 // 忽略
    ])
    model.value = m?.model_id ? m : null
    if (!model.value) return
    mountedDatasets.value = dsRes.datasets || []
    const all: any[] = allRes.datasets || []
    allDatasets.value = all
    sealMinImages.value = allRes.seal_min_images || 0
    if (ld) {
      labelsDict.value = { labels: ld?.labels || [], model_code: ld?.model_code }
      syncLabelRows()
    }
    if (lm) allModels.value = lm.models || []
    // 简介/名称草稿取库根记录（与标题展示、保存目标保持一致）
    const rid = m?.library_root_id || m?.model_id
    const rootRec = allModels.value.find((x: any) => x.model_id === rid) || m
    descDraft.value = rootRec?.description || ''
    nameDraft.value = rootRec?.name || rootRec?.display_name || ''
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  await refresh()
  await loadJobs()
})
</script>

<style scoped>
.model-detail {
  max-width: 1100px;
}

.back-btn {
  background: none;
  border: none;
  color: #2c6ec5;
  cursor: pointer;
  padding: 0;
  font-size: 0.9rem;
  margin-bottom: 12px;
}
.back-btn:hover { text-decoration: underline; }

/* ===== Header（工作台） ===== */
.wk-header { margin-bottom: 12px; }
.wk-header-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }
.wk-title { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.wk-title h1 { margin: 0; color: #1f2d3d; font-size: 1.5rem; }
.model-code { font-size: 0.75rem; color: #7a8aa0; background: #f3f6fa; padding: 2px 8px; border-radius: 6px; }
.status-tag { font-size: 0.75rem; padding: 2px 10px; border-radius: 10px; background: #f1f3f5; color: #6b7c93; }
.status-tag.using { background: #e8f5e9; color: #1e8449; font-weight: 600; }
.status-tag.pending { background: #e3f2fd; color: #1565c0; }
.status-tag.production_ready, .status-tag.active { background: #e8f5e9; color: #2e7d32; }
.status-tag.rejected { background: #ffebee; color: #c62828; }
.status-tag.evaluating { background: #e3f2fd; color: #1565c0; }
.status-tag.superseded { background: #fff8e1; color: #b26a00; }
.status-tag.training { background: #e8eaf6; color: #3949ab; }
.ver-hist-wrap { margin-top: 10px; }
.ver-hist-toggle { font-size: 0.85rem; color: #6b7c93; }
.ver-hist-list { margin-top: 8px; display: flex; flex-direction: column; gap: 8px; }
.ver-card-hist { opacity: 0.75; }

.wk-desc { margin-top: 10px; color: #7f8c8d; font-size: 0.9rem; line-height: 1.6; }
.wk-meta { margin-top: 8px; font-size: 0.85rem; color: #34495e; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.meta-sep { color: #cdd6e0; }
.wk-actions { margin-top: 10px; display: flex; gap: 6px; }
.wk-edit-desc { margin-top: 10px; padding-top: 10px; border-top: 1px dashed #e0e6ee; }
.wk-edit-desc textarea { width: 100%; border: 1px solid #d5dee8; border-radius: 8px; padding: 8px 10px; font-family: inherit; }
.wk-edit-desc-actions { margin-top: 8px; display: flex; gap: 8px; }
.wk-save-msg { margin: 8px 0 0; font-size: 0.8rem; color: #27ae60; }
.wk-save-msg.err { color: #e74c3c; }

/* ===== 主操作栏 ===== */
.wk-ops { display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
.pt-file-input { display: none; }

/* ===== Tab 栏 ===== */
.wk-tabs { display: flex; gap: 4px; border-bottom: 2px solid #eef2f6; margin-bottom: 14px; flex-wrap: wrap; }
.wk-tab {
  padding: 9px 16px;
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
  color: #6b7c93;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 600;
}
.wk-tab:hover { color: #2c6ec5; }
.wk-tab.on { color: #2c6ec5; border-bottom-color: #2c6ec5; }

.wk-pane { padding-bottom: 20px; }
.wk-pane-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
.wk-pane-head h2 { margin: 0; font-size: 1.1rem; color: #2c3e50; }

/* ===== 概览 ===== */
.ov-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
.ov-card { border: 1px solid #e3e9f1; border-radius: 8px; padding: 12px 14px; background: #fbfcfe; }
.ov-label { font-size: 0.75rem; color: #94a3b3; margin-bottom: 6px; font-weight: 600; }
.ov-value { font-size: 0.9rem; color: #2c3e50; line-height: 1.5; }
.ov-tags { display: flex; flex-wrap: wrap; gap: 5px; }
.ov-tag { font-size: 0.72rem; background: #eef4ff; color: #2c6ec5; padding: 2px 8px; border-radius: 10px; }
.ov-tag-sub { color: #7f8c8d; font-size: 11px; margin-left: 4px; }
.mt4 { margin-top: 8px; }

/* ===== 数据集管理 ===== */
.ds-toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }
.ds-search { flex: 1; min-width: 200px; max-width: 360px; padding: 8px 12px; border: 1px solid #d5dee8; border-radius: 8px; font-size: 0.85rem; }
.ds-filter { padding: 8px 10px; border: 1px solid #d5dee8; border-radius: 8px; font-size: 0.85rem; background: #fff; }

.ds-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.ds-table th { text-align: left; padding: 8px 10px; border-bottom: 2px solid #eef2f6; color: #7f8c8d; font-weight: 600; }
.ds-table td { padding: 10px; border-bottom: 1px solid #f0f3f7; vertical-align: middle; }
.ds-name-main { font-weight: 600; color: #1f2d3d; max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ds-sub { font-size: 0.72rem; color: #9aa7b6; font-family: Consolas, monospace; }
.modal-panel .ds-search { display: block; width: 100%; max-width: none; margin-bottom: 10px; box-sizing: border-box; }

/* 生命周期状态胶囊（3.3：灰/蓝/橙/绿/白） */
.life-pill { font-size: 0.75rem; padding: 3px 10px; border-radius: 10px; border: 1px solid transparent; white-space: nowrap; display: inline-block; }
.life-todo { background: #eef0f3; color: #6b7c93; }
.life-annotating { background: #e3f2fd; color: #1565c0; }
.life-pending { background: #fff3e0; color: #b26a00; border-color: #ffd9a0; font-weight: 700; }
.life-sealed { background: #e8f5e9; color: #2e7d32; border-color: #c8e6c9; }
.life-unassigned { background: #fff; color: #9aa7b6; border-color: #eaedf1; }
.life-failed { background: #ffebee; color: #c62828; border-color: #ffcdd2; }

.job-running, .job-queued { background: #e3f2fd; color: #1565c0; }
.job-completed, .job-ready { background: #e8f5e9; color: #2e7d32; }
.job-failed, .job-crashed { background: #ffebee; color: #c62828; }
.job-stopped { background: #fff3e0; color: #b26a00; }

.train-mark { font-size: 0.72rem; padding: 2px 8px; border-radius: 10px; white-space: nowrap; }
.train-mark.ok { background: #e0f7fa; color: #00695c; }
.train-mark.todo { background: #f3f6fa; color: #8a94a6; }

.row-actions { display: flex; gap: 6px; align-items: center; position: relative; }
.more-menu { position: relative; }
.more-btn { min-width: 30px; padding: 0 8px; }
.more-dropdown {
  position: absolute; right: 0; top: 30px; z-index: 20;
  background: #fff; border: 1px solid #e3e8f0; border-radius: 8px;
  box-shadow: 0 4px 14px rgba(31, 45, 61, 0.14); min-width: 120px; padding: 4px;
  display: flex; flex-direction: column;
}
.dropdown-item {
  padding: 7px 12px; text-align: left; background: none; border: none;
  font-size: 0.82rem; color: #34495e; cursor: pointer; border-radius: 6px;
}
.dropdown-item:hover { background: #f0f5fb; }
.dropdown-item.danger { color: #e74c3c; }
.dropdown-item.link { text-decoration: none; }

/* 空状态引导 */
.wk-empty .empty-state { text-align: left; }
.wk-steps { margin: 8px 0 0; padding-left: 4px; list-style: none; line-height: 2; color: #5a6b7d; font-size: 0.9rem; }

/* ===== 模型版本 ===== */
.ver-card { border: 1px solid #e3e9f1; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; background: #fff; }
.ver-head { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }
.ver-name { font-size: 1.05rem; font-weight: 700; color: #1f2d3d; }
.ver-latest { font-size: 0.7rem; background: #27ae60; color: #fff; padding: 1px 8px; border-radius: 10px; margin-left: 6px; }
.ver-latest.prod { margin-left: 0; background: #1e8449; }
.ver-lineage { font-size: 0.7rem; background: #8e9aaf; color: #fff; padding: 1px 8px; border-radius: 10px; margin-left: 6px; }
.ver-dl { margin-left: auto; }
.ver-metrics { display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 8px; }
.ver-metric { display: flex; flex-direction: column; gap: 2px; }
.ver-metric b { font-size: 1.2rem; color: #8e44ad; }
.vm-label { font-size: 0.72rem; color: #94a3b3; }
.ver-meta { display: flex; gap: 14px; flex-wrap: wrap; font-size: 0.78rem; color: #7f8c8d; }

.gk-compare { margin-top: 10px; padding: 10px 12px; background: #fafbfc; border: 1px solid #dde3ea; border-radius: 8px; }
.gk-title { font-size: 0.78rem; font-weight: 700; color: #34495e; margin-bottom: 6px; }
.gk-report-text { font-size: 0.8rem; line-height: 1.6; color: #4a5568; word-break: break-word; }
.gk-regressed { margin-top: 6px; font-size: 0.76rem; color: #c0392b; }
.gk-regressed-item { display: inline-block; margin: 2px 4px 0 0; padding: 1px 8px; background: #fdecea; border: 1px solid #f5b7b1; border-radius: 8px; font-weight: bold; }
.gk-delta { margin-top: 6px; font-size: 0.8rem; color: #1e8449; font-weight: 600; }

/* ===== 设置 ===== */
.set-group { padding: 14px 0; border-bottom: 1px dashed #e3e9f1; }
.set-group:last-child { border-bottom: none; }
.set-label { font-size: 0.88rem; font-weight: 700; color: #2c3e50; margin-bottom: 8px; }
.set-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.cell-input { border: 1px solid #d5dee8; border-radius: 6px; padding: 6px 8px; font-size: 0.82rem; width: 100%; min-width: 90px; }
.label-table input { width: 100%; }
.labels-msg { font-size: 0.8rem; color: #27ae60; }
.static-info { display: flex; flex-direction: column; gap: 6px; font-size: 0.83rem; color: #4a5568; }
.static-info div { display: flex; gap: 12px; align-items: baseline; }
.static-info span:first-child { color: #94a3b3; min-width: 110px; }
.static-info code { background: #f3f6fa; padding: 2px 6px; border-radius: 4px; font-size: 0.78rem; }
.static-info .path { word-break: break-all; }

.form-hint { color: #9aa7b6; font-size: 0.8rem; }
.danger-zone .danger { background: #fff; border: 1px solid #e57373; color: #e53935; border-radius: 8px; padding: 8px 16px; font-size: 0.85rem; cursor: pointer; font-weight: 600; }
.danger-zone .danger:hover { background: #e53935; color: #fff; }
.danger-zone .danger:disabled { opacity: 0.6; cursor: not-allowed; }
.empty-state { text-align: center; color: #7f8c8d; padding: 40px 24px; background: #fff; border: 1px dashed #cfd8e3; border-radius: 12px; }

/* ===== 管理标签弹窗 ===== */
.modal-mask {
  position: fixed; inset: 0; z-index: 100;
  background: rgba(31, 45, 61, 0.45);
  display: flex; align-items: flex-start; justify-content: center;
  padding: 48px 16px; overflow-y: auto;
}
.modal-panel {
  background: #fff; border-radius: 12px; box-shadow: 0 8px 30px rgba(31, 45, 61, 0.2);
  padding: 18px 20px; width: 100%; max-width: 760px;
}
.modal-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.modal-head h3 { margin: 0; font-size: 1.05rem; color: #1f2d3d; }
.modal-actions { margin-top: 12px; }
.modal-panel .label-table input { width: 100%; }
.modal-panel .cell-input { border: 1px solid #d5dee8; border-radius: 6px; padding: 6px 8px; font-size: 0.82rem; min-width: 90px; box-sizing: border-box; }

/* ===== 页内导入新数据弹窗 ===== */
.ds-file-box { margin: 12px 0 10px; }
.ds-file-input { display: none; }
.ds-file-btn {
  display: inline-flex; align-items: center; justify-content: center; height: 34px; padding: 0 14px;
  background: #eef0f3; border: 1px solid #d0d3d8; border-radius: 6px; cursor: pointer;
  font-size: 0.85rem; color: #333; white-space: nowrap;
}
.ds-file-btn:hover { background: #e3e7ec; }
.ds-selected {
  display: flex; align-items: center; gap: 6px; margin-bottom: 8px; padding: 10px 12px;
  background: #eaf4fb; border: 1px solid #3498db; border-radius: 6px; font-size: 0.86rem; color: #1a5276;
  word-break: break-all;
}
.ds-selected.empty { background: #f8f9fa; border: 1px dashed #c8d2dc; color: #9aa7b6; }
.ds-selected-size { color: #5d6d7e; font-size: 0.78rem; }

/* ===== 标签表头说明小标签 ===== */
.th-tag { margin-top: 2px; font-size: 0.68rem; font-weight: 400; color: #9aa7b6; }

/* ===== 标签在用/未用标识 + 锁死的英文名格子 ===== */
.usage-tip { margin: -4px 0 8px; font-size: 0.78rem; color: #e08a3c; }
.usage-tag { display: block; margin-top: 3px; font-size: 0.7rem; color: #9aa7b6; }
.usage-tag.used { color: #2f8f5b; }
.cell-input.locked { background: #f3f6fa; color: #8b98a8; cursor: not-allowed; }
.label-table button.danger:disabled { opacity: 0.45; cursor: not-allowed; }

/* ===== 已标注数据的文案同步 ===== */
.label-sync {
  margin-top: 14px; padding: 12px 14px;
  background: #fbfcfe; border: 1px solid #e3e9f1; border-radius: 8px;
}
.ls-title { font-size: 0.86rem; font-weight: 700; color: #2c3e50; margin-bottom: 6px; }
.label-sync .form-hint { margin: 0 0 10px; line-height: 1.6; }
.ls-detail { margin-top: 10px; font-size: 0.8rem; color: #4a5568; line-height: 1.6; }
.ls-table { font-size: 0.78rem; margin-top: 6px; }
.ls-table th, .ls-table td { padding: 5px 8px; }
.ls-before { color: #c0392b; text-decoration: line-through; word-break: break-all; }
.ls-after { color: #1e8449; word-break: break-all; }
.ls-backup { margin-top: 8px; font-size: 0.75rem; color: #9aa7b6; }
</style>
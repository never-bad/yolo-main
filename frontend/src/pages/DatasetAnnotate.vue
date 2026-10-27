<template>
  <div class="annotate">
    <div class="card">
      <h2>创建标注任务</h2>
      <div class="info-box">
        ⚠ 请确保数据集已完成准备（prepare）操作，否则无法创建标注任务
      </div>
      <div class="form-group">
        <label>选择数据集</label>
        <div class="select-with-refresh">
          <select v-model="newTask.datasetId" :disabled="loadingDatasets || creatingTask" @change="onDatasetSelectChange">
            <option value="">-- 请选择数据集 --</option>
            <option 
              v-for="ds in preparedDatasets" 
              :key="ds.dataset_id" 
              :value="ds.dataset_id"
            >
              {{ ds.display_name || ds.dataset_id }} ({{ getDatasetStatus(ds) }}) 
              {{ ds.image_count ? `- ${ds.image_count}张图片` : '' }}
            </option>
          </select>
          <button type="button" @click="loadDatasets" class="refresh-btn" :disabled="loadingDatasets">
            <span v-if="loadingDatasets" class="loading-spinner"></span>
            {{ loadingDatasets ? '加载中...' : '刷新' }}
          </button>
        </div>
        <div v-if="preparedDatasets.length > 0" class="ds-pager">
          <span class="ds-pager-info">共 {{ preparedDatasets.length }} 个 · 第 {{ dsPage }}/{{ dsPageCount }} 页</span>
          <span class="ds-pager-nav">
            <button type="button" class="ds-pager-btn" :disabled="dsPage <= 1" @click="dsPage = dsPage - 1">上一页</button>
            <button type="button" class="ds-pager-btn" :disabled="dsPage >= dsPageCount" @click="dsPage = dsPage + 1">下一页</button>
          </span>
          <select v-model.number="dsPageSize" class="ds-pager-size" :disabled="loadingDatasets">
            <option v-for="n in DS_PAGE_SIZES" :key="n" :value="n">每页 {{ n }} 个</option>
          </select>
        </div>
        <div class="ds-grid">
          <div
            v-for="ds in pagedDatasets"
            :key="ds.dataset_id"
            :class="['ds-card', { selected: newTask.datasetId === ds.dataset_id }]"
            @click="selectDataset(ds)"
          >
            <div class="ds-card-header">
              <span class="ds-name">{{ dsTitle(ds) }}</span>
              <span :class="'status-badge status-' + ds.status">{{ getDatasetStatus(ds) }}</span>
            </div>
            <div class="ds-card-info">
              <span>数据集id：{{ ds.dataset_id }}</span>
              <span v-if="ds.image_count">{{ ds.image_count }} 张图片</span>
              <span v-if="ds.classes && ds.classes.length" class="ds-classes" :title="ds.classes.join(', ')">
                {{ ds.classes.length }} 类: {{ ds.classes.slice(0, 3).join(', ') }}<template v-if="ds.classes.length > 3"> 等</template>
              </span>
            </div>
          </div>
          <div v-if="!loadingDatasets && preparedDatasets.length === 0" class="empty-state">暂无已准备的数据集，请先在数据集页执行"准备"</div>
        </div>
        <small>提示：仅展示已准备（prepared）的数据集</small>
      </div>
      <div class="form-group">
        <label>类别（逗号分隔，可选）</label>
        <input v-model="classesInput" type="text" placeholder="留空则自动读取类别文件；多个类别用逗号分隔（中英文逗号均可）" :disabled="creatingTask" />
        <small>提示：切换数据集会自动填充其已有类别，可在此基础上追加/删改；留空则创建时自动从类别文件读取</small>
      </div>
      <button @click="createTask" :disabled="creatingTask || !newTask.datasetId">
        <span v-if="creatingTask" class="loading-spinner"></span>
        {{ creatingTask ? '创建中...' : '创建任务' }}
      </button>
    </div>

    <div v-if="currentTask" class="annotate-workspace">
      <div class="sidebar">
        <!-- 计数跟着真正显示的图片走：筛/搜之后括号里是结果数，后面灰字是全量 -->
        <h3 class="list-title" @dblclick="jumpToIndex" title="双击可跳转到指定张数">
          图片列表 ({{ visibleItems.length }}<span v-if="visibleItems.length !== items.length" class="list-count-total"> / {{ items.length }}</span>)
        </h3>
        <div v-if="importedCount > 0" class="import-info">
          已导入 {{ importedCount }} 个标注
        </div>
        <div class="image-filter">
          <div class="image-search">
            <input
              v-model="imageQuery"
              type="text"
              class="image-search-input"
              placeholder="搜索图片名，如 hsz / 0012"
              title="按图片文件名子串匹配，不区分大小写，与下方的类别筛选叠加生效"
            />
            <button v-if="imageQuery" class="image-search-clear" title="清空搜索" @click="imageQuery = ''">✕</button>
          </div>
          <div class="filter-dropdown" @click.stop>
            <button class="filter-trigger" @click="filterOpen = !filterOpen">
              <span>{{ currentFilterLabel }}</span>
              <span class="filter-caret">{{ filterOpen ? '▲' : '▼' }}</span>
            </button>
            <div v-if="filterOpen" class="filter-menu">
              <div
                class="filter-menu-item"
                :class="{ active: imageFilter === 'all' }"
                @click="setFilter('all')"
              >全部（{{ counts.total }}）</div>
              <div class="filter-menu-group">
                <div
                  class="filter-menu-item group-header"
                  :class="{ active: imageFilter === 'annotated' }"
                  @click="setFilter('annotated')"
                >已标注（{{ counts.annotated }}）</div>
                <div
                  class="filter-menu-item sub"
                  :class="{ active: imageFilter === 'manual' }"
                  @click="setFilter('manual')"
                >↳ 人工标注（{{ counts.manual }}）</div>
                <div
                  class="filter-menu-item sub"
                  :class="{ active: imageFilter === 'ai' }"
                  @click="setFilter('ai')"
                >↳ AI标注（{{ counts.ai }}）</div>
              </div>
              <!-- 轮廓质量：按"图上有没有轮廓 + 人工有没有修过"分三档，挑图精修不用自己翻 -->
              <div class="filter-menu-group">
                <div class="filter-menu-item group-header label-only">轮廓质量</div>
                <div
                  class="filter-menu-item sub"
                  :class="{ active: imageFilter === 'poly_draft' }"
                  @click="setFilter('poly_draft')"
                  title="SAM/AI 批量描的轮廓还没人工复核过，需要挑着修"
                >↳ 待修正·AI稿（{{ counts.poly_draft }}）</div>
                <div
                  class="filter-menu-item sub"
                  :class="{ active: imageFilter === 'poly_refined' }"
                  @click="setFilter('poly_refined')"
                  title="在画布上编辑过并保存过的轮廓，算人工把关过"
                >↳ 已修过（{{ counts.poly_refined }}）</div>
                <div
                  class="filter-menu-item sub"
                  :class="{ active: imageFilter === 'poly_none' }"
                  @click="setFilter('poly_none')"
                  title="图上还没有任何分割轮廓（可能没标、也可能只画了方框），可拿去描新轮廓"
                >↳ 还没有轮廓（{{ counts.poly_none }}）</div>
              </div>
              <div
                class="filter-menu-item"
                :class="{ active: imageFilter === 'unannotated' }"
                @click="setFilter('unannotated')"
              >未标注（{{ counts.unannotated }}）</div>
            </div>
          </div>
        </div>
        <div v-if="loadingItems" class="loading-state">
          <span class="loading-spinner"></span>
          <span>加载图片列表...</span>
        </div>
        <div v-else ref="imageListRef" class="image-list" @scroll.passive="onListScroll">
          <!-- 列表只渲染当前图附近的一段；滚到边缘时自动平移窗口，体感是连续滚动 -->
          <div
            v-for="item in windowedItems"
            :key="item.image_id"
            :data-img="item.image_id"
            :class="['image-item', { active: currentImage?.image_id === item.image_id, annotated: item.annotated }]"
            @click="selectImageByItem(item)"
          >
            <span class="image-item-name">{{ item.image_id }}</span>
            <span class="image-item-badges">
              <span v-if="item.sample_type === 'hard'" class="img-badge hard" title="困难样本：AI 低置信度/检空，需重点审核">难</span>
              <span v-else-if="item.sample_type === 'background'" class="img-badge bg" title="空白样本（负样本）：无目标框">空</span>
              <span v-if="item.annotated" class="img-badge ok" title="已有标注">✓</span>
              <span v-if="item.ai_annotated" class="img-badge ai" title="AI 预标注，建议人工审核">AI</span>
            </span>
          </div>
          <div v-if="visibleItems.length === 0" class="empty-filter">
            <template v-if="imageQuery.trim() && imageFilter !== 'all'">
              「{{ filterLabels[imageFilter] }}」里没有文件名包含「{{ imageQuery.trim() }}」的图片
            </template>
            <template v-else-if="imageQuery.trim()">没有文件名包含「{{ imageQuery.trim() }}」的图片</template>
            <template v-else>该筛选条件下没有图片</template>
          </div>
        </div>
        <!-- 翻页用画布两侧的 ‹ › 或键盘 ←/→，这里只留一个游标，表示当前图在筛选/搜索结果里的位置 -->
        <div class="nav-pos">{{ visibleIdx >= 0 ? `第 ${visibleIdx + 1} / ${visibleItems.length} 张` : '当前无匹配图片' }}</div>
      </div>

      <div class="canvas-area">
        <h3>
          {{ currentImage?.image_id }}
          <span v-if="loadingAnnotation" class="loading-indicator">
            <span class="loading-spinner"></span> 加载标注中...
          </span>
        </h3>
        <div class="ai-toolbar">
          <button class="ai-btn" @click="runAutoLabel" :disabled="!currentImage">
            <span v-if="aiLabeling" class="loading-spinner"></span>
            {{ aiLabeling ? 'AI标注中(点击取消)...' : 'AI 预标注' }}
          </button>
          <button class="ai-btn batch-btn" @click="runBatchLabel" :disabled="batchRunning || !items.length" title="对任务内全部未标注图片执行 AI 预标注">
            <span v-if="batchRunning" class="loading-spinner"></span>
            {{ batchRunning ? '批量预标注中...' : '批量预标注' }}
          </button>
          <!-- 一级入口：选哪个 AI 模型（原来藏在 ⋯更多 → 标注模型设置，不好找） -->
          <button class="ai-btn model-btn" @click="openModelCfg"
            :title="'当前 AI 识别模型：' + detectorLabel + '\n点击可选择/切换模型（支持多个引擎同时检测）'">
            模型: {{ detectorLabel }}
          </button>
          <!-- 一级开关：AI 出的是「轮廓（多边形/分割）」还是「方框（四边形/检测）」 -->
          <span class="shape-seg" :title="samOn
            ? '当前 AI 沿目标边缘描轮廓 → 多边形分割标注（可训练分割模型，较慢）。\n图中特征：轮廓内有半透明填充、拐角是实心小方块，右侧列表带「分割」徽标'
            : '当前 AI 只画外接方框 → 普通检测标注（速度快）。\n图中特征：只有一圈边框线，选中后八向手柄可拉伸'">
            <button class="seg-btn" :class="{ active: samOn }" :disabled="samSaving" @click="setSamEnabled(true)">轮廓·分割</button>
            <button class="seg-btn" :class="{ active: !samOn }" :disabled="samSaving" @click="setSamEnabled(false)">方框·检测</button>
          </span>
          <!-- 描轮廓绘制开关：选「轮廓·分割」时会自动进入，这里给一个显性的手动入口/退出 -->
          <button class="ai-btn poly-toggle" :class="{ active: polyDrawMode }" @click="togglePolyDrawMode"
            :title="polyDrawMode
              ? '正在逐点描轮廓：左键逐个点顶点 · 回车或点回起点闭合 · Esc 取消本条 · 右键顶点删点。点此退出，恢复拖框绘制'
              : '手动描分割轮廓（多边形）：逐点落顶点，用于修正 AI 轮廓或精雕边界'">
            {{ polyDrawMode ? '描轮廓中·退出' : '手动描轮廓' }}
          </button>
          <!-- 批量预标注是另一套开关（默认只出方框），这里明示避免分不清 -->
          <button class="shape-note" @click="openModelCfg"
            title="批量预标注默认只出方框（比出轮廓快一倍以上）；点这里在「AI 标注模型设置」里勾选「批量也描轮廓」">
            批量出: {{ batchSamOn ? '轮廓' : '方框' }}
          </button>
          <span class="ai-classes-label" v-if="classes.length" :title="'检测类别（由当前算法库自动同步）：\n' + classes.join('、')">
            检测类别: {{ classes.length }}个
          </span>
          <details class="ai-params">
            <summary title="置信度 / IoU 阈值参数（AI 预标注即用）">参数</summary>
            <div class="ai-params-body">
              <label class="ai-conf-label">置信度:</label>
              <input type="number" v-model.number="aiConf" min="0" max="1" step="0.05" class="conf-input" />
              <label class="ai-conf-label">IoU:</label>
              <input type="number" v-model.number="aiIou" min="0" max="1" step="0.05" class="conf-input" />
              <button class="ai-btn small" @click="onThreshChange" title="保存置信度/IoU 阈值，AI 标注即用">保存阈值</button>
            </div>
          </details>
          <div v-if="batchId" class="batch-progress">
            <span>批量预标注: {{ batchProgress?.done || 0 }}/{{ batchProgress?.total || 0 }}</span>
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: batchPercent + '%' }"></div>
            </div>
            <div v-if="batchProgress?.audit" class="audit-batch-line">
              L2 审核: {{ batchProgress.audit.checked }} 图/{{ batchProgress.audit.boxes }} 框
              · <span class="a-keep">✓{{ batchProgress.audit.stats.keep }}</span>
              <span class="a-modify">~{{ batchProgress.audit.stats.modify }}</span>
              <span class="a-reject">✗{{ batchProgress.audit.stats.reject }}</span>
              <span class="a-split">⧉{{ batchProgress.audit.stats.split }}</span>
              <span class="a-missed">漏{{ batchProgress.audit.stats.missed }}</span>
              <span v-if="batchProgress.audit.errors" class="a-reject">({{ batchProgress.audit.errors }} 次失败)</span>
            </div>
            <button class="danger small" @click="stopBatch" :disabled="!batchRunning">停止</button>
          </div>
        </div>
        <div class="canvas-container" @mousedown="startDrawing" @mousemove="drawing" @mouseup="endDrawing" @wheel="onWheel" @mouseleave="onCanvasLeave" @contextmenu="onCanvasContextMenu">
          <button
            class="img-nav-btn prev"
            @click.stop="prevImage"
            @mousedown.stop
            :disabled="!canPrev || loadingAnnotation"
            title="上一张（← / A）"
          >‹</button>
          <canvas ref="canvasRef" @click="onCanvasClick" @dblclick="onCanvasDblClick"></canvas>
          <button
            class="img-nav-btn next"
            @click.stop="nextImage"
            @mousedown.stop
            :disabled="!canNext || loadingAnnotation"
            title="下一张（→ / D）"
          >›</button>
        </div>
        <!-- 缩放/绘制工具条：放在画面下方独占一行，不再浮在画布上挡住要标的区域 -->
        <div class="view-toolbar">
          <!-- 提示语保持精简：详细操作写在 title 悬浮提示里 -->
          <span
            v-if="cutoutMode"
            class="view-hint poly-hint"
            title="左键点目标内部 → 后台排队抠轮廓，点完立刻可以点下一个（也可以直接切到别的图接着点，结果各回各图）· 同一个目标上继续左键=补点微调、右键=点抠多了的地方 · 回车或双击换下一个目标 · 再按一次 G 或 Esc 退出（退出会撤掉还没算的任务，已落地的框用 Ctrl+Z 撤销）"
          >抠图中：点哪儿抠哪儿 · 可切图连着点
            <em v-if="cutoutQueue.length">· 排队中 {{ cutoutQueue.length }} 个</em>
            <em v-if="cutoutDone">· 已抠好 {{ cutoutDone }} 个</em>
            · G/Esc 退出</span>
          <span
            v-else-if="polyDrawMode"
            class="view-hint poly-hint"
            title="空白处左键逐个点顶点 · 点回起点或按回车闭合成一个分割标注 · 绘制中右键=删上一个点 · Esc 取消本条 · 按 F 复位视图 · 已有目标：点住白色方块拖动改拐点、点轮廓内部拖动整体平移、边上空心圆加点、右键点删点"
          >描轮廓中：空白处逐点画新轮廓 · 点已有目标先选中，拖方块改拐点、拖内部平移</span>
          <span v-else class="view-hint" title="滚轮缩放 · 空格+拖拽平移 · 按 F 或双击复位 · 拖框移动/角点缩放 · 分割目标：点住拐点直接拖、拖轮廓内部整体平移 · 按 G 进入智能抠图（点哪儿抠哪儿）· 按 C 清空本图所有标注框（Ctrl+Z 撤销）">滚轮缩放 · 拖框绘制 · 按 F 复位 · 按 G 抠图 · 按 C 清空</span>
          <span>缩放 {{ Math.round(viewScale * 100) }}%</span>
          <button class="small" @click.stop="zoomBy(1.2)" title="放大">+</button>
          <button class="small" @click.stop="zoomBy(1 / 1.2)" title="缩小">−</button>
          <button class="small" @click.stop="fitView" title="适应窗口">适应窗口</button>
          <button
            class="small"
            :class="{ active: polyDrawMode }"
            @click.stop="togglePolyDrawMode"
            title="手动描分割轮廓（多边形）：用于修正 AI 轮廓或给目标精雕边界"
          >{{ polyDrawMode ? '退出轮廓绘制' : '轮廓绘制' }}</button>
          <button
            class="small"
            :class="{ active: edgeOverlayOn }"
            @click.stop="edgeOverlayOn = !edgeOverlayOn; onEdgeOverlayToggle()"
            title="边缘增强叠加层（快捷键 E）：青色细线显形 Sobel 强梯度（水陆过渡带、纹理恢复线），用于规范 v1.3 §1.6 的 B/C 类模糊边界判边"
          >边缘</button>
          <button
            class="small"
            :class="{ active: snapToEdgeOn }"
            @click.stop="snapToEdgeOn = !snapToEdgeOn; onSnapToggle()"
            title="顶点吸附（快捷键 S）：落点/拖点自动贴到半径内最强边缘（A/B 类边界可压到 ±2px）。C 类不可见边界按物理线索画时建议关闭"
          >吸附</button>
          <label class="alpha-ctl" title="分割标注内部填充的遮盖度：调低看清水面原貌，调高看清标注范围；0=只显示描边">
            填充
            <input type="range" min="0" max="0.4" step="0.02" v-model.number="maskAlpha" @input="onMaskAlphaChange" />
          </label>
        </div>
        <div class="controls">
          <label>当前类别:</label>
          <select v-model="currentClass" title="仅控制画布显示哪些类别的框，不影响新画框的类别">
            <option :value="-1">全部标签</option>
            <option v-for="(cls, idx) in classes" :key="idx" :value="idx">{{ cls }}</option>
          </select>
          <label>绘制类别:</label>
          <select v-model="drawClass" title="新画框落盘使用的类别，与显示过滤无关">
            <option v-for="(cls, idx) in classes" :key="idx" :value="idx">{{ cls }}</option>
          </select>
        </div>
      </div>

      <div class="annotations-panel">
        <div class="ann-panel-header">
          <h3>当前标注</h3>
          <div class="more-wrap">
            <button class="more-btn" :class="{ active: moreOpen }" @click="moreOpen = !moreOpen" title="全局 / 批量操作">⋯ 更多</button>
            <div v-if="moreOpen" class="menu-backdrop" @click="moreOpen = false"></div>
            <div v-if="moreOpen" class="more-menu" @click.stop>
              <button class="cfg-entry" @click="openModelCfg(); moreOpen = false" title="选择检测引擎（可多选同时检测）、调阈值、配置千问 VL">
                标注模型设置
              </button>
              <button
                @click="auditCurrentImage(); moreOpen = false"
                :disabled="auditing || !currentBoxes.length"
                title="对当前图片标注框调用千问 L2 审核：漏标/粘连/标签错误/框质量/合规。绿=通过、橙=待修正、红=建议删除、蓝虚线=漏标建议，可一键采纳"
              >
                <span v-if="auditing" class="loading-spinner"></span>
                {{ auditing ? 'AI 审核中...' : 'AI 审核当前图（千问 L2）' }}
              </button>
              <button
                @click="openAuditIssues(); moreOpen = false"
                :disabled="loadingIssueList"
                title="列出任务中带审核异议的图片（建议删除/待修正/粘连），点击直达，不用逐张翻图"
              >
                <span v-if="loadingIssueList" class="loading-spinner"></span>
                审核问题图片
              </button>
              <button
                @click="toggleInteractiveMode(); moreOpen = false"
                :disabled="interactiveBusy || !items.length"
                :class="{ active: interactiveMode }"
                title="框选局部区域 + 文本提示（当前类别）检测，避免全图盲扫误标"
              >
                {{ interactiveMode ? '退出交互式标注' : '交互式标注' }}
              </button>
              <button class="danger" @click="clearAllAiLabels(); moreOpen = false" :disabled="counts.ai === 0" title="误标过多时一键清除，重新标注">
                清除全部AI标注（{{ counts.ai }}）
              </button>
              <button
                class="danger"
                @click="clearAllLabels(); moreOpen = false"
                :disabled="counts.annotated === 0 || clearingAll"
                title="清空本数据集所有标注框（AI 标注 + 人工标注一视同仁）；清空前后端会自动打包备份，可解压回退"
              >
                {{ clearingAll ? '清除中...' : `清除全部标注（${counts.annotated}）` }}
              </button>
              <button class="danger" @click="cleanOverlaps(); moreOpen = false" :disabled="counts.annotated === 0" title="清理历史遗留的高度重合标注框（跨类别/同类重复）">
                清理重叠框
              </button>
              <button @click="exportTask(); moreOpen = false" :disabled="exporting">
                <span v-if="exporting" class="loading-spinner"></span>
                {{ exporting ? '导出中...' : '导出' }}
              </button>
              <button
                v-if="approveEligible"
                class="primary"
                @click="approveCurrentDataset(); moreOpen = false"
                :disabled="approvingReview"
                title="本批标注已人工复核完毕，确认审核通过：数据集即可自动灌入训练桶（也可到数据集页点同一按钮，效果一样）"
              >
                <span v-if="approvingReview" class="loading-spinner"></span>
                {{ approvingReview ? '确认中...' : `✓ 审核通过本数据集（${counts.annotated}/${counts.total}）` }}
              </button>
              <div v-else-if="taskDatasetId && counts.total > 0" class="menu-hint" :title="approveHintTitle">
                {{ approveHintText }}
              </div>
              <button @click="exportReviewedPackage(); moreOpen = false" :disabled="exportingReviewed || counts.reviewed === 0" title="只把人工修过（筛子里“轮廓·已修过”）的图片+YOLO标签打包成 zip 下载，labels 为分割格式，解压即可用">
                <span v-if="exportingReviewed" class="loading-spinner"></span>
                {{ exportingReviewed ? '打包中...' : `导出已修过 zip（${counts.reviewed}）` }}
              </button>
              <label class="import-label" title="导入 X-AnyLabeling 等外部工具导出的 YOLO 标注（zip 内含 labels/*.txt）">
                <input type="file" accept=".zip" class="import-input" @change="onImportYolo" />
                <span v-if="importing" class="loading-spinner"></span>
                {{ importing ? '导入中...' : '导入YOLO标注' }}
              </label>
            </div>
          </div>
        </div>
        <!-- 审核问题图片弹窗：任务级检索有异议的图片 -->
        <div v-if="auditIssuesOpen" class="issue-mask" @click.self="auditIssuesOpen = false">
          <div class="issue-dialog">
            <div class="issue-dialog-head">
              <strong>审核问题图片（{{ auditIssues.length }}）</strong>
              <span class="issue-hint">建议删除 ✗ · 待修正 ~ · 粘连 ⧉</span>
              <button class="small" @click="refreshAuditIssues" :disabled="loadingIssueList">刷新</button>
              <button class="small" @click="auditIssuesOpen = false">关闭</button>
            </div>
            <div v-if="loadingIssueList && !auditIssues.length" class="issue-loading">加载中...</div>
            <div v-else-if="!auditIssues.length" class="issue-empty">没有发现审核异议的图片</div>
            <ul v-else class="issue-list">
              <li v-for="it in auditIssues" :key="it.image_id" @click="gotoIssueImage(it.image_id)" title="点击跳转到该图">
                <span class="issue-img">{{ it.image_id }}</span>
                <span class="issue-badges">
                  <em v-if="it.actions.reject" class="a-reject">✗{{ it.actions.reject }}</em>
                  <em v-if="it.actions.modify" class="a-modify">~{{ it.actions.modify }}</em>
                  <em v-if="it.actions.split" class="a-split">⧉{{ it.actions.split }}</em>
                </span>
              </li>
            </ul>
          </div>
        </div>
        <div class="box-list">
          <div
            v-for="(box, idx) in currentBoxes"
            :key="idx"
            class="box-item"
            :class="{ 'box-item-selected': idx === selectedBoxIndex }"
            @click="selectBoxFromPanel(idx)"
          >
            <span class="box-color" :style="{ background: classColors[box.class_id] || '#999' }"></span>
            <span class="box-class-name" :title="classes[box.class_id] || ('类别' + box.class_id)">{{ classes[box.class_id] || ('类别' + box.class_id) }}</span>
            <span v-if="box.polygon && box.polygon.length >= 3" class="seg-chip" :title="'分割标注：' + box.polygon.length + ' 个轮廓顶点（选中后可在图上拖点微调）'">分割</span>
            <span
              v-if="box.audit"
              class="audit-chip"
              :class="'a-' + (box.audit.action || 'keep')"
              :title="'AI 审核: ' + (box.audit.reason || box.audit.action) + (box.audit.kind === 'label_error' ? '（标签错误）' : '') + (box.audit.kind === 'merged' ? '（粘连）' : '') + (box.audit.tags?.length ? ' ' + box.audit.tags.join(' ') : '')"
            >
              {{ box.audit.action === 'keep' ? '✓' : box.audit.action === 'reject' ? '✗' : box.audit.action === 'modify' ? '~' : '⧉' }}
            </span>
            <button v-if="box.audit?.suggested_class_id != null" class="small fix-btn" @click.stop="fixBoxClass(idx)" :title="'采纳 AI 建议类别：' + (classes[box.audit.suggested_class_id] ?? '')">
              用{{ classes[box.audit.suggested_class_id] ?? '' }}
            </button>
            <select :value="box.class_id" class="box-class-select" :title="classes[box.class_id] || ''" @click.stop @change="onBoxClassChange($event, idx)">
              <option v-for="(cls, cIdx) in classes" :key="cIdx" :value="cIdx">{{ cls }}</option>
            </select>
            <button class="danger small" @click.stop="removeBox(idx)">删除</button>
          </div>
          <div v-if="!currentBoxes.length" class="box-empty">暂无标注框，可手动绘制或使用 AI 预标注</div>
        </div>
        <!-- 千问 L2 审核汇总（设计文档 §6.3）：统计 + 漏标补框 + 一键操作 -->
        <div v-if="imageAudit?.ok" class="audit-summary">
          <div class="audit-head">
            <span class="audit-title">AI 审核 · 千问 L2</span>
            <button class="small" @click="clearAudit" title="去掉审核颜色/标签/漏标虚线，保留标注框">清除审核标记</button>
          </div>
          <div class="audit-stats">
            <span class="a-keep">✓ 通过 {{ imageAudit.stats?.keep || 0 }}</span>
            <span class="a-modify">~ 待修正 {{ imageAudit.stats?.modify || 0 }}</span>
            <span class="a-reject">✗ 建议删除 {{ imageAudit.stats?.reject || 0 }}</span>
            <span class="a-split">⧉ 粘连 {{ imageAudit.stats?.split || 0 }}</span>
            <span class="a-miss">漏标 {{ imageAudit.missed_targets?.length || 0 }}</span>
          </div>
          <div v-if="auditRejectCount" class="audit-actions">
            <button class="danger small" @click="removeRejectedBoxes">删除被拒框（{{ auditRejectCount }}）</button>
          </div>
          <div v-if="imageAudit.missed_targets?.length" class="audit-missed">
            <div class="audit-missed-title">漏标建议（画布蓝虚线位置，点击补标即落框）</div>
            <div v-for="(mt, i) in imageAudit.missed_targets" :key="'mt' + i" class="audit-mt-row">
              <span class="mt-info">{{ classes[mt.class_id] || mt.class_id }}<em v-if="mt.reason">（{{ mt.reason }}）</em></span>
              <button class="small primary" @click="addMissedTarget(mt)">补标</button>
            </div>
          </div>
        </div>
        <div v-if="cutoutMode && !cutoutBusy" class="interactive-hint">
          抠图模式：左键连续点水洼（相连的点自动并成一个闭合区域，不相连的自动拆成多个框）· 已抠出来的目标上继续点＝微调 · 右键拖框＝挖掉多余那块（只点不拖＝抠掉那个点）· 等的时候可以直接切图接着点，抠好的会各自回到原图 · 回车/双击换下一个目标 · Esc 退出
        </div>
        <div v-if="cutoutMode && cutoutBusy" class="interactive-hint">
          <span class="loading-spinner"></span> 正在判断点位连通性并抠图：排队 {{ cutoutQueue.length }} 个 · 已完成 {{ cutoutDone }} 个，可以先切图继续点
        </div>
        <div v-if="interactiveMode && !interactiveBusy" class="interactive-hint">
          在图上拖拽框选要标注的区域（建议先选好"当前类别"），Esc 退出
        </div>
        <div v-if="interactiveBusy" class="interactive-hint">
          <span class="loading-spinner"></span> 正在解析所选区域...
        </div>
        <div class="ann-fixed-actions" title="">
          <div class="sample-type-group" title="标注完成后给该图片分级：普通/困难（AI 低置信度或漏检，重点审核）/空白（无目标负样本）；不选则自动判定">
            <span class="sample-type-label">样本</span>
            <template v-for="st in SAMPLE_TYPES" :key="st.value">
              <button
                v-if="st.value !== ''"
                class="sample-type-btn"
                :class="{ active: currentSampleType === st.value }"
                @click="setSampleType(st.value)"
              >{{ st.label }}</button>
            </template>
          </div>
          <div class="ann-action-row">
            <button class="primary" @click="saveAnnotation" :disabled="savingAnnotation" title="快捷键 Ctrl+S / Cmd+S 保存当前标注">
              <span v-if="savingAnnotation" class="loading-spinner"></span>
              {{ savingAnnotation ? '保存中...' : '保存' }}
            </button>
            <button class="danger" @click="clearCurrentBoxes" :disabled="!currentBoxes.length" title="清空本图所有标注框">
              清空本图
            </button>
            <button class="danger" @click="deleteCurrentImage" :disabled="deletingImage || !currentImage" title="快捷键 X：删除当前图片（同时删除其标注与同名标签文件），不可恢复，批量标注中也可删除">
              {{ deletingImage ? '删除中...' : '删除图片' }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
  <!-- AI 标注模型设置弹窗（多引擎可同时检测） -->
  <div v-if="showModelCfg" class="modal-overlay" @click.self="closeModelCfg">
    <div class="modal-box model-cfg-modal">
      <h3>AI 标注模型设置</h3>
      <p class="hint">支持同时启用多个检测引擎，多引擎检测结果自动合并去重。保存后立即生效，无需重启后端。<br />启用千问后，AI 预标注会自动附加 <b>L2 审核</b>（漏标/粘连/标签错误/框质量/合规），也可通过"⋯更多 → AI 审核当前图"手动审核。</p>
      <div class="cfg-block">
        <span class="cfg-title">检测引擎（可多选，同时使用）</span>
        <label class="row">
          <input type="checkbox" value="grounding_dino" v-model="modelCfg.detectors" />
          <span>GroundingDINO <em class="dim">（零样本文本检测，无需权重）</em></span>
        </label>
        <label class="row">
          <input type="checkbox" value="yolo_world" v-model="modelCfg.detectors" />
          <span>YOLO-World（.pt 权重）</span>
        </label>
        <div class="row indent" v-if="modelCfg.detectors.includes('yolo_world')">
          <span>权重文件：</span>
          <select v-model="modelCfg.detector_weights">
            <option v-for="m in availableModels" :key="m.name" :value="m.name">{{ m.name }}（{{ m.size_mb }}MB）</option>
            <option v-if="!availableModels.length" value="yolov8s-world.pt">yolov8s-world.pt（默认）</option>
          </select>
        </div>
        <label class="row">
          <input type="checkbox" value="qwen_vl" v-model="modelCfg.detectors" />
          <span>千问 VL（云端 / 本地 VLM 大模型）</span>
        </label>
        <label class="row">
          <input type="checkbox" value="trained_model" v-model="modelCfg.detectors" />
          <span>本库已训模型 <em class="dim">（用这个算法库自己训练出来的模型预标注，比只靠文字提示的引擎准得多）</em></span>
        </label>
        <template v-if="modelCfg.detectors.includes('trained_model')">
          <div class="row indent">
            <span>模型版本：</span>
            <select v-model="modelCfg.trained_model_id">
              <option value="">（请选择）</option>
              <option v-for="v in trainedVersions" :key="v.model_id" :value="v.model_id">{{ trainedVersionLabel(v) }}</option>
            </select>
          </div>
          <p class="hint" v-if="!trainedVersions.length">
            目前还没有能用于预标注的模型：<b>需要先在这个算法库里训练完成一次</b>（训练产出权重后才会出现在这里）。
          </p>
          <p class="hint" v-else>
            用法：人工先标一小批 → 训练出第一版 → 回来勾上它批量预标剩下的图 → 人只删错的、补漏的。<br />
            模型学到的类别名要和标签的<b>英文名</b>一致（例如 jishui），对不上的类别会自动忽略，不会乱标。<br />
            标了「分割」的版本会直接沿边缘描出轮廓；标了「检测」的版本只出方框。
          </p>
        </template>
      </div>
      <div class="cfg-block">
        <span class="cfg-title">标注形态：轮廓（多边形）还是方框（四边形）</span>
        <label class="row">
          <input type="checkbox" v-model="samEnabledModel" />
          <span>单图「AI 预标注」与「交互式标注」沿边缘描轮廓 <em class="dim">（勾=分割标注，不勾=只出方框）</em></span>
        </label>
        <label class="row">
          <input type="checkbox" v-model="batchSamModel" />
          <span>「批量预标注」也描轮廓 <em class="dim">（默认不勾：批量只出方框，速度快一倍以上）</em></span>
        </label>
        <p class="hint">怎么在图上分辨两种标注：<b>轮廓</b>=形状内有半透明填充、拐角是实心小方块（可拖动微调）、右侧列表带「分割」徽标；<b>方框</b>=只有一圈边框线、选中后八个手柄拉伸。</p>
      </div>
      <div class="cfg-block">
        <span class="cfg-title">AI 阈值</span>
        <label class="row">
          <span>置信度</span>
          <input type="number" v-model.number="modelCfg.conf" min="0" max="1" step="0.05" class="w-input short" />
          <span>IoU</span>
          <input type="number" v-model.number="modelCfg.iou" min="0" max="1" step="0.05" class="w-input short" />
        </label>
      </div>
      <div class="cfg-block" v-if="modelCfg.detectors.includes('qwen_vl')">
        <span class="cfg-title">千问 VL 配置</span>
        <!-- 原来这里还有一个「启用千问 VL」复选框，勾了引擎还要再勾一次它才生效，
             容易漏（表现为保存后千问其实没跑）。现在以引擎勾选框为唯一开关。 -->
        <div class="row">
          <span>✓ 已随上方引擎勾选自动启用</span>
        </div>
        <label class="row">
          <span>接入方式：</span>
          <select v-model="modelCfg.qwen_backend">
            <option value="ollama">Ollama（本地大模型，免费）</option>
            <option value="dashscope">阿里云百炼 DashScope（云端 API）</option>
          </select>
        </label>
        <label class="row">
          <span>服务地址：</span>
          <input type="text" v-model="modelCfg.qwen_endpoint" class="w-input" />
        </label>
        <label class="row">
          <span>模型名：</span>
          <input type="text" v-model="modelCfg.qwen_model" class="w-input" />
        </label>
        <label class="row">
          <span>API Key：</span>
          <input type="password" v-model="modelCfg.qwen_api_key" class="w-input" />
        </label>
        <label class="row">
          <span>单图超时(秒)：</span>
          <input type="number" v-model.number="modelCfg.qwen_timeout" min="10" max="600" class="w-input short" />
        </label>
        <label class="row" v-if="modelCfg.qwen_backend === 'ollama'">
          <span>keep_alive(驻留)：</span>
          <input type="text" v-model="modelCfg.qwen_keep_alive" placeholder="5m / 30m / -1=常驻" class="w-input short" />
          <span>并行数：</span>
          <input type="number" v-model.number="modelCfg.qwen_num_parallel" min="1" max="16" class="w-input short" />
        </label>
        <label class="row">
          <input type="checkbox" v-model="modelCfg.qwen_mock" />
          <span>模拟框模式（本地联调，不调用真实模型）</span>
        </label>
        <!-- 速度开关：实测 qwen3-vl:4b 上"检测"约 0.45s、"L2 审核"约 1.35s，
             审核比检测还贵，所以单独给开关；批量里审核只进统计，默认不做。 -->
        <span class="cfg-title" style="display:block;margin-top:10px">速度</span>
        <label class="row">
          <input type="checkbox" v-model="modelCfg.qwen_audit_enabled" />
          <span>单图预标注后自动做 L2 审核
            <em class="dim">（关掉单图约 0.5s，开着约 2.2s；需要审核时点「⋯更多 → AI 审核当前图」）</em></span>
        </label>
        <label class="row">
          <input type="checkbox" v-model="modelCfg.qwen_audit_batch" />
          <span>批量预标注也做 L2 审核
            <em class="dim">（默认关：批量里审核只进统计不改框，却让每图从 ~0.5s 涨到 ~2.2s）</em></span>
        </label>
        <label class="row">
          <span>送模型图片最长边：</span>
          <input type="number" v-model.number="modelCfg.qwen_image_max_side" min="320" max="2048" step="64" class="w-input short" />
          <span class="dim">px（1024 够用；实测压到 768 并不会更快）</span>
        </label>
      </div>
      <div class="modal-footer">
        <button @click="closeModelCfg">取消</button>
        <button class="primary" @click="saveModelCfg">保存并应用</button>
      </div>
    </div>
  </div>

  <div v-if="suggestDialogOpen" class="modal-overlay" @click.self="suggestDialogOpen = false">
      <div class="modal suggest-modal">
        <h3>🤖 AI 识别新标签</h3>
        <p class="suggest-hint">千问 VL 正在查看本任务中困难/空白样本图，找出已知标签之外的新目标</p>
        <div v-if="suggestLoading" class="suggest-msg">✨ 正在分析候选图片，请稍候…</div>
        <template v-else>
          <div v-if="suggestMsg" class="suggest-msg">{{ suggestMsg }}</div>
          <label v-for="c in suggestCandidates" :key="c.english_code" class="suggest-item">
            <input type="checkbox" :value="c" v-model="suggestSelected" />
            <div class="suggest-item-body">
              <b>{{ c.chinese_name }}</b>
              <code>{{ c.english_code }}</code>
              <span class="suggest-desc">{{ c.chinese_desc }}</span>
              <span class="suggest-imgs">命中 {{ (c.images || []).length }} 张图</span>
            </div>
          </label>
          <div v-if="suggestCandidates.length === 0" class="suggest-msg">未发现已知类别之外的新目标。</div>
          <div v-if="suggestCandidates.length" class="suggest-actions">
            <button class="primary" @click="adoptSuggest" :disabled="suggestAdopting || suggestSelected.length === 0">
              {{ suggestAdopting ? '采纳中…' : `采纳选中（${suggestSelected.length}）` }}
            </button>
            <button @click="suggestDialogOpen = false">关闭</button>
          </div>
        </template>
      </div>
    </div>

    <!-- 困难/空白样本提醒弹窗：进入任务且存在这类样本时弹一次，不再常驻横幅 -->
    <div v-if="suggestPromptOpen" class="modal-overlay" @click.self="suggestPromptOpen = false">
      <div class="modal suggest-modal">
        <h3>⚠ 困难/空白样本提醒</h3>
        <p class="suggest-hint">
          本任务有 <b>{{ suggestBannerCount }}</b> 张困难/空白样本，AI 检空或低置信度，可能漏检了某些目标
        </p>
        <div class="suggest-actions">
          <button v-if="taskModelId" class="btn-soft" @click="suggestPromptOpen = false; openSuggestDialog()">🤖 分析新类别</button>
          <button @click="suggestPromptOpen = false">知道了</button>
        </div>
      </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import type { BBox } from '@/api/annotations'
import { createAnnotationTask, findAnnotationTask, getTaskItems, getImageAnnotation, saveAnnotation as saveAnn, exportAnnotations, exportReviewedZip, clearAiAnnotations, clearAllAnnotations, cleanTaskOverlaps, importYoloLabels, deleteTaskImage } from '@/api/annotations'
import { autoLabelImage, interactiveLabelImage, pointLabelImage, erasePolygonRegion, startBatchLabel, getBatchProgress, stopBatchLabel, getSamConfig, updateSamConfig, getSamModels, getSamTrainedVersions, auditImage, listAuditIssues } from '@/api/sam'
import { listDatasets, approveDatasetReview } from '@/api/datasets'
import { showConfirm, showPrompt, showAlert } from '@/composables/useDialog'
import { downloadFile } from '@/utils/download'
import { showToast } from '@/utils/toast'
import { dsTitle } from '@/utils/dsDisplay'
import { suggestModelLabels, adoptSuggestedLabels } from '@/api/models'

const newTask = ref({ datasetId: '', version: 'v1' })
const datasets = ref<any[]>([])
const loadingDatasets = ref(false)
const creatingTask = ref(false)
const loadingItems = ref(false)
const savingAnnotation = ref(false)
const deletingImage = ref(false)
// 右侧面板"⋯ 更多"（全局/批量操作）菜单开关
const moreOpen = ref(false)
// AI 标注模型设置弹窗（多引擎可同时检测）
const showModelCfg = ref(false)
const availableModels = ref<any[]>([])
// 「本库已训模型」引擎的可选版本（后端只返回权重已落盘的模型版本）
const trainedVersions = ref<any[]>([])
const trainedVersionLabel = (v: any) => {
  // 主标识用模型管理页的重命名：version（如 v1）和 model_id 对人毫无辨识度，
  // 多个库都叫 v1 时根本挑不出该用哪个权重。没改过名才退回 version / model_id。
  const name = v.display_name || v.version || v.model_id
  const scope = v.same_library ? '本库' : '其他库·类别可能对不上'
  const task = v.task === 'segment' ? '分割' : v.task === 'detect' ? '检测' : (v.task || '未知')
  const cls = (v.classes && v.classes.length) ? `${v.classes.length}类` : '未知类别'
  // 跨库时补上所属算法库；未归库的模型库名等于自身名字，不重复显示
  const lib = v.library_name && v.library_name !== name ? `｜${v.library_name}` : ''
  return `${scope}｜${name}${lib}｜${task}｜${cls}｜${v.size_mb}MB`
}
const modelCfg = ref<any>({
  detectors: [] as string[],
  detector_weights: 'yolov8s-world.pt',
  trained_model_id: '',
  conf: 0.25,
  iou: 0.4,
  qwen_enabled: false,
  qwen_backend: 'ollama',
  qwen_endpoint: 'http://localhost:11434',
  qwen_model: 'qwen3-vl:4b-instruct',
  qwen_api_key: '',
  qwen_timeout: 90,
  qwen_mock: false,
  qwen_keep_alive: '5m',
  qwen_num_parallel: 1,
  qwen_image_max_side: 1024,
  qwen_audit_enabled: true,
  qwen_audit_batch: false,
})
// 自动保存（防抖）
const autoSaveTimer = ref<any>(null)
const dirty = ref(false)
// 当前图是否被"人工编辑"过：画布上真正改动标注（pushUndo）才置真，AI 结果自动合入不算；
// 切图时初始值取后端已存的 reviewed，保存时随请求落盘（粘性），供"轮廓质量"筛选挑图
const manualTouched = ref(false)
// 导入 YOLO 标注
const importing = ref(false)
const exporting = ref(false)
const exportingReviewed = ref(false)
// 清除全部标注（AI + 人工）进行中，防止重复触发
const clearingAll = ref(false)
// 审核通过（标注页顺手点）：与数据集页同一接口，成功后该包进入灌桶资格
const approvingReview = ref(false)
const taskDatasetMeta = computed(() =>
  (datasets.value || []).find((d: any) => d.dataset_id === taskDatasetId.value) || null)
const approveEligible = computed(() => {
  const d = taskDatasetMeta.value
  if (!d || !taskDatasetId.value) return false
  if (d.review_approved || d.bucket_archived) return false
  if (['sealed', 'training', 'completed', 'failed'].includes(d.stage)) return false
  // 标注页现场判定：每张图都有标注 JSON 才允许审核（与后端校验同口径）
  return counts.value.total > 0 && counts.value.annotated === counts.value.total
})
const approveHintText = computed(() => {
  const d = taskDatasetMeta.value
  if (d?.review_approved) return '✓ 本数据集已审核通过'
  if (counts.value.total > 0 && counts.value.annotated < counts.value.total) {
    return `还有 ${counts.value.total - counts.value.annotated} 张未标注，全部标完才能审核通过`
  }
  return ''
})
const approveHintTitle = computed(() => '审核通过条件：数据集已归属算法库、全部图片完成标注、尚未审核')
const approveCurrentDataset = async () => {
  const dsId = taskDatasetId.value
  if (!dsId || approvingReview.value) return
  const ok = await showConfirm(
    `确认审核通过本数据集（${counts.value.annotated} 张已全部标注）？\n\n` +
    `通过后该包将自动灌入所属算法库的训练桶，桶满即封板（封板后只读）。\n` +
    `封板前如需继续修改，可到数据集页点「撤回」。`)
  if (!ok) return
  approvingReview.value = true
  try {
    await approveDatasetReview(dsId)
    await loadDatasets()  // 刷新徽标状态（按钮变为"已审核通过"）
    showToast('已审核通过 ✓ 该数据集进入灌桶队列，下次刷新数据集页自动灌入训练桶', 'success', 4000)
  } catch (e: any) {
    if (!e?.__toastShown) showToast('审核确认失败: ' + (e.response?.data?.detail || e.message), 'error', 4000)
  } finally {
    approvingReview.value = false
  }
}
// 标注任务持久化 key，用于中途离开页面后返回时恢复
const ANNOTATE_STORE_KEY = 'annotate_task_resume'

const loadDatasets = async () => {
  loadingDatasets.value = true
  try {
    const result = await listDatasets()
    datasets.value = result.datasets || []
  } catch (error: any) {
    console.error('加载数据集失败:', error)
    if (!error?.__toastShown) showToast('加载数据集失败: ' + (error.response?.data?.detail || error.message), 'error', 4000)
  } finally {
    loadingDatasets.value = false
  }
}

// 获取数据集准备状态
const getDatasetStatus = (dataset: any) => {
  if (dataset.status === 'prepared') return '已准备'
  if (dataset.status === 'uploaded') return '已上传'
  return dataset.status || '未知'
}

// 标注页只展示"已准备"的数据集（未准备无法创建标注任务）
const preparedDatasets = computed(() => {
  return (datasets.value || []).filter((d: any) => d.status === 'prepared')
})

// 卡片网格分页：已准备的数据集有几十个，一次性铺满会把「新建任务」卡片拉得很长，
// 这里只切分渲染范围，下拉框仍然是全量（下拉自身可滚动，不受分页影响）。
const dsPageSize = ref(10)          // 每页条数，默认 10
const dsPage = ref(1)               // 当前页（1-based）
const DS_PAGE_SIZES = [10, 20, 30, 50]
const dsPageCount = computed(() => Math.max(1, Math.ceil(preparedDatasets.value.length / dsPageSize.value)))
const pagedDatasets = computed(() => {
  const start = (dsPage.value - 1) * dsPageSize.value
  return preparedDatasets.value.slice(start, start + dsPageSize.value)
})
// 某个数据集落在第几页；不在已准备列表里返回 0
const dsPageOf = (id: string) => {
  const i = preparedDatasets.value.findIndex((d: any) => d.dataset_id === id)
  return i < 0 ? 0 : Math.floor(i / dsPageSize.value) + 1
}
// 换页大小、数据集增删都回到第 1 页，避免停在已不存在的页码上
watch(dsPageSize, () => { dsPage.value = 1 })
watch(preparedDatasets, () => { dsPage.value = 1 })

// 切换数据集时自动刷新类别输入框：跟随所选数据集的类别；
// 数据集无类别则清空，避免把上一个数据集的类别误带到新数据集
const syncClassesFromDataset = (dsId: string) => {
  const ds = datasets.value.find(d => d.dataset_id === dsId)
  if (!ds) return
  const dsClasses = Array.isArray(ds.classes) ? ds.classes : []
  classesInput.value = dsClasses.join(', ')
}

// 卡片式选择数据集
const selectDataset = (ds: any) => {
  newTask.value.datasetId = ds.dataset_id
  syncClassesFromDataset(ds.dataset_id)
  const p = dsPageOf(ds.dataset_id)
  if (p > 0) dsPage.value = p
}
// 下拉框选中：下拉是全量的，选中的数据集卡片可能不在当前页，顺带跳到它所在的页
const onDatasetSelectChange = () => {
  const id = newTask.value.datasetId
  syncClassesFromDataset(id)
  const p = dsPageOf(id)
  if (p > 0) dsPage.value = p
}
const classesInput = ref('')  // 默认为空，从数据集自动读取
const classes = ref<string[]>([])
const currentTask = ref<string | null>(null)
const items = ref<any[]>([])
const currentIndex = ref(0)
const currentImage = ref<any>(null)
const currentClass = ref(-1)  // -1 表示"全部标签"，仅控制画布显示哪些类别的框
const drawClass = ref(0)      // 新画框落盘使用的类别索引，与显示过滤无关
const currentBoxes = ref<BBox[]>([])
const selectedBoxIndex = ref(-1)  // 当前选中的标注框索引（点击画布或左侧列表时更新）
const importedCount = ref(0)  // 导入的标注数量
const loadingAnnotation = ref(false)  // 加载标注状态
// 当前图片样本类型（人工可改）：'' 未定 | normal 普通 | hard 困难 | background 空白负样本
const currentSampleType = ref('')
const SAMPLE_TYPES = [
  { value: '', label: '自动' },
  { value: 'normal', label: '普通' },
  { value: 'hard', label: '困难' },
  { value: 'background', label: '空白' },
]
const setSampleType = (t: string) => {
  if (currentSampleType.value === t) return
  currentSampleType.value = t
  if (currentImage.value) currentImage.value.sample_type = t
  if (currentIndex.value >= 0 && items.value[currentIndex.value]) {
    items.value[currentIndex.value].sample_type = t
  }
  markDirty()  // 触发防抖自动保存，无需手动点保存
}

// ===== 新类别主动提醒：任务中有困难/空白样本 → 横幅提示 → 点击才跑 AI 识别 =====
const taskDatasetId = ref('')
const taskModelId = ref('')
const suggestPromptOpen = ref(false)
const suggestPrompted = ref(false)
const suggestDialogOpen = ref(false)
const suggestLoading = ref(false)
const suggestAdopting = ref(false)
const suggestCandidates = ref<any[]>([])
const suggestSelected = ref<any[]>([])
const suggestMsg = ref('')

// 横幅计数：困难/空白样本数（不调用千问，仅统计，提醒用户可能漏检新类别）
const suggestBannerCount = computed(() => {
  return items.value.filter((it: any) => it.sample_type === 'hard' || it.sample_type === 'background').length
})

const openSuggestDialog = async () => {
  suggestDialogOpen.value = true
  suggestCandidates.value = []
  suggestSelected.value = []
  suggestMsg.value = ''
  await runSuggest()
}

const runSuggest = async () => {
  if (!taskModelId.value || !taskDatasetId.value) {
    suggestMsg.value = '该任务未绑定模型，无法识别新类别'
    return
  }
  suggestLoading.value = true
  suggestCandidates.value = []
  suggestSelected.value = []
  suggestMsg.value = ''
  try {
    const res = await suggestModelLabels(taskModelId.value, taskDatasetId.value, 3)
    if (res.ok) {
      suggestCandidates.value = res.candidates || []
      if (suggestCandidates.value.length === 0 && res.message) suggestMsg.value = res.message
    } else {
      suggestMsg.value = res.message || '识别失败'
    }
  } catch (error: any) {
    suggestMsg.value = '识别失败: ' + (error.response?.data?.detail || error.message)
  } finally {
    suggestLoading.value = false
  }
}

const adoptSuggest = async () => {
  if (!taskModelId.value || suggestSelected.value.length === 0) return
  suggestAdopting.value = true
  try {
    const res = await adoptSuggestedLabels(taskModelId.value, suggestSelected.value.map(c => ({
      english_code: c.english_code,
      chinese_name: c.chinese_name,
      chinese_desc: c.chinese_desc
    })))
    const added = res.added || []
    showToast(`已采纳 ${added.length} 个新标签，追加到模型标签字典末尾：${added.join('、') || '（无新增）'}`, 'info', 4000)
    suggestDialogOpen.value = false
  } catch (error: any) {
    if (!error?.__toastShown) showToast('采纳失败: ' + (error.response?.data?.detail || error.message), 'error', 4000)
  } finally {
    suggestAdopting.value = false
  }
}

const canvasRef = ref<HTMLCanvasElement | null>(null)
const ctx = ref<CanvasRenderingContext2D | null>(null)
const img = ref<HTMLImageElement | null>(null)
const isDrawing = ref(false)
const startPos = ref({ x: 0, y: 0 })
const currentPos = ref({ x: 0, y: 0 })

// 画布视图变换（缩放/平移）
const viewScale = ref(1)
const viewOffset = ref({ x: 0, y: 0 })
const isPanning = ref(false)
// 分割标注内部填充的遮盖度：0.18 在暗光图上会把积水本身盖住、看不清轮廓贴不贴合，
// 做成可调（0=只显示描边），选择记在 localStorage 里下次自动恢复
const maskAlpha = ref((() => {
  const v = Number(localStorage.getItem("annotate_mask_alpha"))
  return Number.isFinite(v) && v >= 0 && v <= 0.5 ? v : 0.08
})())
const onMaskAlphaChange = () => {
  localStorage.setItem("annotate_mask_alpha", String(maskAlpha.value))
  redrawCanvas()
}
// ----------------------------------------------------------------------
// 边缘辅助（规范 v1.3 §1.6）：Sobel 梯度叠加层 + 顶点吸附
// 图片加载后离线算一次梯度；叠加层把"水陆过渡带/纹理恢复线"显形，
// 吸附让落点/拖点自动贴到半径内最强边缘（A/B 类边界误差压到 ±2px；C 类可关）
// ----------------------------------------------------------------------
const edgeOverlayOn = ref(localStorage.getItem("annotate_edge_overlay") === "1")
const snapToEdgeOn = ref(localStorage.getItem("annotate_edge_snap") !== "0")  // 默认开
let edgeData: { w: number; h: number; grad: Float32Array } | null = null
let edgeOverlayCanvas: HTMLCanvasElement | null = null
const onEdgeOverlayToggle = () => {
  localStorage.setItem("annotate_edge_overlay", edgeOverlayOn.value ? "1" : "0")
  redrawCanvas()
}
const onSnapToggle = () => {
  localStorage.setItem("annotate_edge_snap", snapToEdgeOn.value ? "1" : "0")
}
// 构建当前图的 Sobel 梯度场与青色边缘叠加层（1080p 约 100-200ms，随图片加载做一次）
const buildEdgeMap = () => {
  edgeData = null
  edgeOverlayCanvas = null
  const im = img.value
  if (!im || !im.width) return
  const w = im.width, h = im.height
  const cv = document.createElement("canvas")
  cv.width = w; cv.height = h
  const c2 = cv.getContext("2d", { willReadFrequently: true })
  if (!c2) return
  c2.drawImage(im, 0, 0)
  let data: ImageData
  try { data = c2.getImageData(0, 0, w, h) } catch { return }  // 跨域污染时静默降级
  const px = data.data
  const gray = new Float32Array(w * h)
  for (let i = 0, j = 0; i < px.length; i += 4, j++) gray[j] = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]
  const grad = new Float32Array(w * h)
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i0 = y * w + x
      const gx = -gray[i0 - w - 1] - 2 * gray[i0 - 1] - gray[i0 + w - 1] + gray[i0 - w + 1] + 2 * gray[i0 + 1] + gray[i0 + w + 1]
      const gy = -gray[i0 - w - 1] - 2 * gray[i0 - w] - gray[i0 - w + 1] + gray[i0 + w - 1] + 2 * gray[i0 + w] + gray[i0 + w + 1]
      grad[i0] = Math.hypot(gx, gy)
    }
  }
  edgeData = { w, h, grad }
  const oc = document.createElement("canvas")
  oc.width = w; oc.height = h
  const octx = oc.getContext("2d")
  if (!octx) return
  const od = octx.createImageData(w, h)
  const thr = 90  // Sobel 幅值阈值：只显形真实边缘，噪声不干扰
  for (let j = 0; j < grad.length; j++) {
    const g = grad[j]
    if (g > thr) {
      od.data[j * 4] = 0
      od.data[j * 4 + 1] = 220
      od.data[j * 4 + 2] = 255
      od.data[j * 4 + 3] = Math.min(255, (g - thr) * 3)
    }
  }
  octx.putImageData(od, 0, 0)
  edgeOverlayCanvas = oc
}
// 吸附：搜索半径内（屏幕 ~8px）最强梯度点；弱于阈值 80 不吸附，避免贴到噪声上
const snapToEdge = (p: { x: number; y: number }): { x: number; y: number } => {
  if (!snapToEdgeOn.value || !edgeData) return p
  const { w, h, grad } = edgeData
  const r = Math.max(2, Math.min(16, Math.round(8 / viewScale.value)))
  const cx = Math.round(p.x), cy = Math.round(p.y)
  let bx = -1, by = -1, best = 80
  const y1 = Math.max(1, cy - r), y2 = Math.min(h - 2, cy + r)
  const x1 = Math.max(1, cx - r), x2 = Math.min(w - 2, cx + r)
  for (let y = y1; y <= y2; y++) {
    for (let x = x1; x <= x2; x++) {
      const g = grad[y * w + x]
      if (g > best) { best = g; bx = x; by = y }
    }
  }
  return bx >= 0 ? { x: bx, y: by } : p
}
const panStart = ref({ x: 0, y: 0 })
const spaceDown = ref(false)

// 交互/编辑状态
const editMode = ref<'idle' | 'draw' | 'move' | 'resize' | 'vertex'>('idle')
const resizeHandle = ref(-1)  // 控制点索引 0-3 角 / 4-7 边
const moveStartPt = ref<{ x: number; y: number } | null>(null)
const origBox = ref<BBox | null>(null)

// ----------------------------------------------------------------------
// 分割轮廓（多边形）交互状态
// ----------------------------------------------------------------------
const polyDrawMode = ref(false)        // 多边形绘制模式（左键逐点，回车/点起点闭合，Esc 取消）
const polyDraft = ref<number[][]>([])  // 正在绘制中的顶点（图像坐标）
const dragVertexIdx = ref(-1)          // 正在拖动的顶点下标；插入模式为 -(边下标+1)
const moveUndoPushed = ref(false)      // 本次平移是否已压撤销快照（真拖动才算改）
const polyMoveHintShown = ref('')      // 平移提示每条图只弹一次，避免刷屏
const suppressCanvasClick = ref(false) // 描轮廓模式下"按下选中"后，跟随的 click 不再改选中态
// 描轮廓模式下的"按下待判定"：按住已有目标时先挂起，移动了 = 改拐点 / 整体平移，没移动 = 选中该目标
const polyPending = ref<{ kind: 'vertex' | 'move'; idx: number; v: number; pt: { x: number; y: number }; sx: number; sy: number } | null>(null)
const hitRadius = () => 9 / viewScale.value

// 取框的轮廓（不足 3 点视为无轮廓 = 普通检测框）
const polyOf = (b: BBox | null | undefined): number[][] | null =>
  (Array.isArray(b?.polygon) && (b!.polygon as number[][]).length >= 3 ? (b!.polygon as number[][]) : null)

// 有轮廓时矩形必须跟着轮廓走，否则导出会出现"框与分割掩码不符"
const syncBBoxFromPolygon = (b: BBox) => {
  const p = polyOf(b)
  if (!p) return
  let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity
  p.forEach(pt => {
    x1 = Math.min(x1, pt[0]); y1 = Math.min(y1, pt[1])
    x2 = Math.max(x2, pt[0]); y2 = Math.max(y2, pt[1])
  })
  b.x1 = x1; b.y1 = y1; b.x2 = x2; b.y2 = y2
}

// 整体平移（框 + 轮廓一起动）：每次从快照重放，避免 mousemove 累加漂移
const applyMove = (box: BBox, orig: BBox, dx: number, dy: number) => {
  box.x1 = orig.x1 + dx; box.y1 = orig.y1 + dy
  box.x2 = orig.x2 + dx; box.y2 = orig.y2 + dy
  const op = polyOf(orig), p = polyOf(box)
  if (op && p) {
    for (let i = 0; i < p.length && i < op.length; i++) {
      p[i][0] = op[i][0] + dx
      p[i][1] = op[i][1] + dy
    }
  }
}

// 射线法：点是否落在多边形轮廓内
const pointInPolygon = (p: { x: number; y: number }, poly: number[][]) => {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1]
    const xj = poly[j][0], yj = poly[j][1]
    if ((yi > p.y) !== (yj > p.y) && p.x < ((xj - xi) * (p.y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

// 复制一份框（轮廓必须深拷贝，否则拖动时 origBox 与当前框共享同一数组）
const cloneBox = (b: BBox): BBox => ({ ...b, polygon: polyOf(b)?.map(pt => [pt[0], pt[1]]) })

// 命中选中框的轮廓顶点：返回 ≥0 为顶点下标，-(边下标+2) 为该边中点（点击即插入新顶点），-1 未命中
const hitVertex = (idx: number, p: { x: number; y: number }) => {
  const poly = polyOf(currentBoxes.value[idx])
  if (!poly) return -1
  const r = hitRadius()
  for (let i = 0; i < poly.length; i++) {
    if (Math.abs(p.x - poly[i][0]) <= r && Math.abs(p.y - poly[i][1]) <= r) return i
  }
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length]
    const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2
    if (Math.abs(p.x - mx) <= r && Math.abs(p.y - my) <= r) return -(i + 2)
  }
  return -1
}

// 描轮廓模式下判断"点是否落在某个分割目标轮廓内部"。
// 只认轮廓内、不认外接矩形，否则两个相邻目标之间的空白会被误判成"要平移"。
const hitPolygonInterior = (p: { x: number; y: number }) => {
  const cur = currentClass.value
  const visible = (b: BBox) => String(cur) === '-1' || b.class_id === Number(cur)
  for (let i = currentBoxes.value.length - 1; i >= 0; i--) {
    const b = currentBoxes.value[i]
    if (!visible(b)) continue
    const poly = polyOf(b)
    if (poly && pointInPolygon(p, poly)) return i
  }
  return -1
}

// 抠图模式：判断"点是否落在某个分割目标上"。
// 有轮廓就只认轮廓内（和外接矩形不同，轮廓外的空白不该算命中），没轮廓才退回外接矩形。
const pointInsideBox = (b: BBox, p: { x: number; y: number }) => {
  const poly = polyOf(b)
  if (poly) return pointInPolygon(p, poly)
  return p.x >= b.x1 && p.x <= b.x2 && p.y >= b.y1 && p.y <= b.y2
}

// 同一个位置连点（习惯性双击、手抖）不能算成两个目标，否则图上会叠出两个几乎一样的框。
// 两道判断：① 离本目标上一个点太近；② 离最近一个"还在排队/在算的新目标"的点太近。
let cutoutLastNew: { imageId: string; x: number; y: number } | null = null
const cutoutTooClose = (p: { x: number; y: number }) => {
  const r = 12 / viewScale.value
  const last = cutoutPts.value[cutoutPts.value.length - 1]
  if (last && Math.hypot(p.x - last.x, p.y - last.y) <= r) return true
  const curId = currentImage.value ? currentImage.value.image_id : ''
  if (cutoutLastNew && cutoutLastNew.imageId === curId &&
      Math.hypot(p.x - cutoutLastNew.x, p.y - cutoutLastNew.y) <= r) return true
  return cutoutQueue.value.some(j => !j.box && j.imageId === curId &&
    j.pts.some(pt => pt.label === 1 && Math.hypot(p.x - pt.x, p.y - pt.y) <= r))
}

// 抓轮廓顶点：优先选中框，其次图上任意可见分割框（不用先选中就能点住拐点改位置）
// 返回 [框下标, 命中码]，命中码含义同 hitVertex；未命中返回 [-1, -1]
const hitAnyVertex = (p: { x: number; y: number }): [number, number] => {
  const sel = selectedBoxIndex.value
  if (sel >= 0) {
    const v = hitVertex(sel, p)
    if (v !== -1) return [sel, v]
  }
  const cur = currentClass.value
  const visible = (b: BBox) => String(cur) === '-1' || b.class_id === Number(cur)
  for (let i = currentBoxes.value.length - 1; i >= 0; i--) {
    if (i === sel || !visible(currentBoxes.value[i])) continue
    const v = hitVertex(i, p)
    if (v !== -1) return [i, v]
  }
  return [-1, -1]
}

// 进入"顶点微调"：命中边中点时先插一个点再拖，返回是否成功进入
const grabVertex = (idx: number, v: number): boolean => {
  if (idx < 0 || v === -1) return false
  const box = currentBoxes.value[idx]
  const poly = polyOf(box)
  if (!poly) return false
  let vi = v
  pushUndo()
  if (v <= -2) {
    const edge = -v - 2
    const a = poly[edge], b = poly[(edge + 1) % poly.length]
    poly.splice(edge + 1, 0, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2])
    vi = edge + 1
  }
  selectedBoxIndex.value = idx
  editMode.value = 'vertex'
  dragVertexIdx.value = vi
  redrawCanvas()
  return true
}

// 点是否贴住某个分割框的轮廓边（屏幕上 ~14px 内，且落在外接矩形附近）：
// 返回 [框下标, 边下标, 投影点x, 投影点y]。放大后想微调拐点却没点准时走这条，
// 在投影点插顶点直接拖，而不是落进轮廓内部误触发整体平移。
const hitPolyEdgeNear = (p: { x: number; y: number }): [number, number, number, number] | null => {
  const r = 14 / viewScale.value
  const cur = currentClass.value
  const visible = (b: BBox) => String(cur) === '-1' || b.class_id === Number(cur)
  for (let i = currentBoxes.value.length - 1; i >= 0; i--) {
    const b = currentBoxes.value[i]
    if (!visible(b)) continue
    const poly = polyOf(b)
    if (!poly) continue
    if (p.x < b.x1 - r || p.x > b.x2 + r || p.y < b.y1 - r || p.y > b.y2 + r) continue
    let best = -1, bd = r, bpx = 0, bpy = 0
    for (let e = 0; e < poly.length; e++) {
      const a = poly[e], q = poly[(e + 1) % poly.length]
      const dx = q[0] - a[0], dy = q[1] - a[1]
      const len2 = dx * dx + dy * dy
      let t = len2 ? ((p.x - a[0]) * dx + (p.y - a[1]) * dy) / len2 : 0
      t = Math.max(0, Math.min(1, t))
      const px = a[0] + t * dx, py = a[1] + t * dy
      const d = Math.hypot(p.x - px, p.y - py)
      if (d <= bd) { bd = d; best = e; bpx = px; bpy = py }
    }
    if (best >= 0) return [i, best, bpx, bpy]
  }
  return null
}

// 在轮廓边投影点插入新顶点并立即进入拖动；投影点已贴近端点则直接抓端点（避免叠点）
const grabEdgeVertex = (idx: number, edge: number, px: number, py: number): boolean => {
  const box = currentBoxes.value[idx]
  const poly = polyOf(box)
  if (!poly) return false
  const a = poly[edge], q = poly[(edge + 1) % poly.length]
  const near = 3 / viewScale.value
  if (Math.hypot(px - a[0], py - a[1]) <= near) return grabVertex(idx, edge)
  if (Math.hypot(px - q[0], py - q[1]) <= near) return grabVertex(idx, (edge + 1) % poly.length)
  pushUndo()
  const sp = snapToEdge({ x: px, y: py })   // 插点也吸附到最强边缘
  poly.splice(edge + 1, 0, [sp.x, sp.y])
  selectedBoxIndex.value = idx
  editMode.value = 'vertex'
  dragVertexIdx.value = edge + 1
  redrawCanvas()
  return true
}

// 进入"整体平移"：框和轮廓一起动（撤销快照等真正拖动起来再压，避免单击就变脏）
const grabMove = (idx: number, p: { x: number; y: number }) => {
  if (idx < 0) return
  selectedBoxIndex.value = idx
  editMode.value = 'move'
  moveStartPt.value = p
  origBox.value = cloneBox(currentBoxes.value[idx])
  moveUndoPushed.value = false
  redrawCanvas()
}

// 结束绘制：顶点数合法则收为一个分割框，否则丢弃
const finishPolygonDraft = () => {
  if (polyDraft.value.length >= 3) {
    let cls = Number(drawClass.value)
    if (Number.isNaN(cls) || cls < 0) cls = 0
    if (classes.value.length && cls >= classes.value.length) cls = classes.value.length - 1
    const poly = polyDraft.value.map(pt => [pt[0], pt[1]])
    const box: BBox = { class_id: cls, x1: 0, y1: 0, x2: 0, y2: 0, polygon: poly, source: 'manual' }
    syncBBoxFromPolygon(box)
    pushUndo()
    currentBoxes.value.push(box)
    selectedBoxIndex.value = currentBoxes.value.length - 1
  }
  polyDraft.value = []
  persistPointDraft()
  redrawCanvas()
}

const cancelPolygonDraft = () => {
  polyDraft.value = []
  polyPending.value = null
  persistPointDraft()
  redrawCanvas()
}

// 描轮廓：落一个顶点。点回起点（≥3 点）用全部点闭合；点更早的顶点 = 把最后一条边
// 收拢到该点，用 [该点..最新点] 这段闭合（沿边界描回来的自然语义），区域点数须 >5。
const addDraftPoint = (p: { x: number; y: number }) => {
  selectedBoxIndex.value = -1
  const d = polyDraft.value
  let hitIdx = -1
  // 排除最后一个点：快速双击时第二下会命中刚落的点，那不是闭合意图（闭合请点起点/更早的顶点）
  for (let i = 0; i < d.length - 1; i++) {
    if (Math.abs(p.x - d[i][0]) <= hitRadius() && Math.abs(p.y - d[i][1]) <= hitRadius()) { hitIdx = i; break }
  }
  if (hitIdx === 0 && d.length >= 3) {
    finishPolygonDraft()
    return
  }
  if (hitIdx > 0) {
    const seg = d.slice(hitIdx)   // 该点之后的所有点（含最新点）
    if (seg.length > 5) {
      polyDraft.value = seg
      finishPolygonDraft()
    } else {
      showToast(`从该点到最新点只有 ${seg.length} 个点（需大于 5）；请点更靠前的顶点或点回起点闭合`, 'info')
    }
    return
  }
  const sp = snapToEdge(p)   // 顶点吸附：新落点贴到半径内最强边缘
  polyDraft.value = [...d, [sp.x, sp.y]]
  persistPointDraft()
  redrawCanvas()
}

// 启用/退出多边形绘制模式（与交互式标注互斥，两者都要用鼠标框选）
const togglePolyDrawMode = () => {
  polyDrawMode.value = !polyDrawMode.value
  polyPending.value = null
  if (!polyDrawMode.value) cancelPolygonDraft()
  if (polyDrawMode.value) {
    interactiveMode.value = false
    interactiveRegion.value = null
    selectedBoxIndex.value = -1
    exitCutoutMode()
  }
  redrawCanvas()
}

// 删除选中框的某个顶点（少于 3 点则该框退回普通检测框，由调用方判断）
const removeVertex = (idx: number, vIdx: number) => {
  const box = currentBoxes.value[idx]
  const poly = polyOf(box)
  if (!box || !poly || vIdx < 0 || vIdx >= poly.length) return
  pushUndo()
  poly.splice(vIdx, 1)
  if (poly.length >= 3) {
    syncBBoxFromPolygon(box)
  } else {
    delete box.polygon   // 点不够了 → 退回矩形框
  }
  redrawCanvas()
}

// 撤销/重做
const undoStack = ref<BBox[][]>([])
const redoStack = ref<BBox[][]>([])

// 类别颜色（渲染时自动分配）
const classColors = ref<string[]>([])

// SAM 大模型预标注状态
const aiLabeling = ref(false)
const aiAbortCtrl = ref<AbortController | null>(null)
const aiConf = ref(0.25)
const aiIou = ref(0.4)
// 交互式标注状态（框选局部区域 + 文本提示，避免全图盲扫误标）
const interactiveMode = ref(false)
const interactiveBusy = ref(false)
const interactiveRegion = ref<{ x1: number; y1: number; x2: number; y2: number } | null>(null)
// 点选抠图（智能抠图）：快捷键 G 进入，点哪儿抠哪儿，一次点击直接落一个分割标注
const cutoutMode = ref(false)
// 当前目标累积的提示点（同一目标可连点微调）+ 这些点已经落成了哪个框
const cutoutPts = ref<{ x: number; y: number; label: number }[]>([])
const cutoutBox = ref<BBox | null>(null)
// 后端回传的"这些点是否连成一片"结论：1=全部相连（一个闭合区域），<1=有断开（自动拆框）
const cutoutCoverage = ref(1)
// 目标的会话编号：每换一次目标 / 切一次图 +1。
// 结果回来时只有编号还对得上的才允许认领"当前目标"，否则会把用户后来的点击带偏。
const cutoutSid = ref(0)
// 抠图队列：点一下就把任务丢进队列，后台按顺序一个一个算（CPU 一次只能算一张，
// 串行才不会互相拖慢）。每个任务记着自己属于哪张图，结果只往那张图上落 ——
// 这样"这张点完切下一张接着点"才不会把 A 图的轮廓掉进 B 图（脏标签）。
type CutoutJob = {
  taskId: string
  imageId: string
  classId: number
  pts: { x: number; y: number; label: number }[]
  box: BBox | null      // 非空 = 给这个已落地的目标补点微调；空 = 新目标
  sid: number
  exitSeq: number       // 提交时的"退出抠图模式"计数，回来时对不上就作废
}
const cutoutQueue = ref<CutoutJob[]>([])
const cutoutRunning = ref(false)
// 退出抠图模式的次数：任务提交时记下的值和当前值不一致 = 中途用户已退出 → 结果作废
let cutoutExitSeq = 0
// 离页标记：队列清空 + 在途结果不再落地，免得往已卸载的画布上写
let cutoutGone = false
// 本次会话累计抠好了几个（提示条上给用户个准数，方便"一起检验"时核对有没有漏）
const cutoutDone = ref(0)
// 有任务在算或在排 = 忙（提示条转圈用）
const cutoutBusy = computed(() => cutoutRunning.value || cutoutQueue.value.length > 0)
// 右键拖拽=区域擦除：按下时命中的分割目标 + 拖出的矩形（松开即把那块从轮廓里挖掉）
const cutoutErase = ref<{ box: BBox | null; x1: number; y1: number; x2: number; y2: number } | null>(null)
const currentImageHasAi = ref(false)  // 当前图片是否由 AI 生成了标注框（用于保存时标记 AI 来源）
const imageAudit = ref<any>(null)     // 当前图千问 L2 审核结果 {ok,box_reviews,missed_targets,stats}
const auditing = ref(false)           // 审核请求中
// 问题图片检索（任务级）：列出带审核异议的图片，点击直达
const auditIssues = ref<any[]>([])
const auditIssuesOpen = ref(false)
const loadingIssueList = ref(false)
const samConfig = ref<any>({ detector_weights: 'yolov8s-world.pt', conf: 0.15 })
// ---- 工具条上的 AI 形态开关：轮廓（多边形/分割）还是方框（四边形/检测）----
const samSaving = ref(false)
const DETECTOR_LABELS: Record<string, string> = {
  grounding_dino: 'GroundingDINO',
  yolo_world: 'YOLO-World',
  qwen_vl: '千问VL',
}
// 后端未显式关闭时按默认（开启轮廓）处理，避免配置里没有这个键时显示反了
const samOn = computed(() => samConfig.value?.sam_enabled !== false)
const batchSamOn = computed(() => samConfig.value?.batch_sam_enabled === true)
const detectorLabel = computed(() => {
  const cfg = samConfig.value || {}
  const list: string[] = (cfg.detectors && cfg.detectors.length)
    ? cfg.detectors
    : ((cfg.detector && cfg.detector !== 'none') ? [cfg.detector] : [])
  if (!list.length) return '未选择'
  return list.map((d) => DETECTOR_LABELS[d] || d).join(' + ')
})
// 只提交单个字段，保存后立即回读，无需重启后端
const setSamEnabled = async (v: boolean) => {
  if (samSaving.value || samOn.value === v) return
  samSaving.value = true
  try {
    await updateSamConfig({ sam_enabled: v })
    samConfig.value = await getSamConfig()
    showToast(v ? '已切换为「轮廓·分割」：AI 预标注会沿边缘描轮廓' : '已切换为「方框·检测」：AI 预标注只出方框', 'success')
  } catch (e: any) {
    showToast('切换失败: ' + (e.response?.data?.detail || e.message), 'error')
  } finally {
    samSaving.value = false
  }
}
// 弹窗里的两个勾与工具条开关共用同一份数据（samConfig），杜绝"弹窗保存把工具条开关打回去"
const samEnabledModel = computed({
  get: () => samOn.value,
  set: (v: boolean) => { samConfig.value.sam_enabled = !!v },
})
const batchSamModel = computed({
  get: () => batchSamOn.value,
  set: (v: boolean) => { samConfig.value.batch_sam_enabled = !!v },
})
// 形态开关联动画布绘制方式：切到「轮廓·分割」= 自动进入逐点描轮廓（和 CVAT 切换工具一致）
watch(samOn, (on) => {
  if (!img.value) return // 还没打开图片时不预设，避免一进页面就被锁成描轮廓
  polyDrawMode.value = on
  if (on) {
    interactiveMode.value = false
    interactiveRegion.value = null
    selectedBoxIndex.value = -1
    exitCutoutMode()
    showToast('已进入「描轮廓」绘制：图上左键逐个点顶点，回车或点回起点闭合成一个分割标注', 'info')
  } else {
    cancelPolygonDraft()
    showToast('已退出描轮廓：可以直接在图上拖框绘制', 'info')
  }
  redrawCanvas()
})
const batchId = ref<string | null>(null)
const batchProgress = ref<any>(null)
const batchTimer = ref<any>(null)

// 常见中文类别 → 英文提示词映射，用于提升 YOLO-World 对中文类别的识别率
const CLASS_EN_MAP: Record<string, string> = {
  '人': 'person', '行人': 'person', '人体': 'person',
  '车': 'car', '汽车': 'car', '轿车': 'car', '小汽车': 'car',
  '卡车': 'truck', '货车': 'truck', '公交车': 'bus', '大巴': 'bus',
  '自行车': 'bicycle', '摩托车': 'motorcycle', '电动车': 'motorcycle',
  '狗': 'dog', '猫': 'cat', '马': 'horse', '牛': 'cow', '羊': 'sheep',
  '猪': 'pig', '鸟': 'bird', '飞机': 'airplane', '轮船': 'boat', '船': 'boat',
  '火车': 'train', '瓶子': 'bottle', '杯子': 'cup', '碗': 'bowl',
  '椅子': 'chair', '桌子': 'table', '餐桌': 'dining table', '沙发': 'sofa',
  '床': 'bed', '电视': 'tv', '显示器': 'tv monitor', '电脑': 'laptop',
  '键盘': 'keyboard', '鼠标': 'mouse', '手机': 'cell phone', '手机架': 'cell phone',
  '遥控器': 'remote', '书': 'book', '钟': 'clock', '花瓶': 'vase',
  '剪刀': 'scissors', '香蕉': 'banana', '苹果': 'apple', '橙子': 'orange',
  '胡萝卜': 'carrot', '披萨': 'pizza', '蛋糕': 'cake', '面包': 'bread',
  '安全帽': 'helmet', '头盔': 'helmet', '帽子': 'hat',
  '背心': 'vest', '手套': 'glove', '口罩': 'mask',
  '消防栓': 'fire hydrant', '斑马线': 'zebra crossing', '交通灯': 'traffic light',
  '红绿灯': 'traffic light', '停靠牌': 'stop sign', '停车标志': 'stop sign',
  '限速牌': 'speed limit sign', '路标': 'traffic sign', '指示牌': 'signboard',
  '树木': 'tree', '树': 'tree', '花': 'flower', '草': 'grass',
  '建筑': 'building', '楼房': 'building', '房子': 'house', '烟囱': 'chimney',
  '路灯': 'street light', '电线杆': 'pole', '电线': 'wire',
  '摄像头': 'camera', '监控': 'cctv camera',
  '箱子': 'box', '纸箱': 'cardboard box', '包裹': 'package', '包裹箱': 'package',
  '垃圾桶': 'trash can', '垃圾箱': 'trash can', '饮料瓶': 'bottle',
  '雨伞': 'umbrella', '球': 'sports ball', '棒球棒': 'baseball bat',
  '网球拍': 'tennis racket', '滑雪板': 'skis', '滑板': 'skateboard',
  '冲浪板': 'surfboard', '风筝': 'kite', '棒球手套': 'baseball glove',
  '方向盘': 'steering wheel', '轮胎': 'tire', '车轮': 'wheel',
  '停车场': 'parking meter', '水杯': 'cup', '水壶': 'kettle',
  '书包': 'backpack', '行李箱': 'suitcase', '手提包': 'handbag',
  '领带': 'tie', '围巾': 'scarf', '外套': 'coat', '裤子': 'pants',
  '裙子': 'skirt', '鞋': 'shoe', '衬衫': 'shirt', '短裤': 'shorts',
  '袜子': 'socks', '蝴蝶结': 'tie', '领结': 'tie',
  '火车头': 'train', '地铁': 'train',
  '车牌': 'license plate', '车门': 'car door', '车窗': 'car window',
  '运动员': 'person', '工人': 'person',
  '油桶': 'bucket', '桶': 'bucket', '灭火器': 'fire extinguisher',
}

// 将类别列表转换为与 classes 一一对应的英文提示词（用于 YOLO-World 文本检测）
const buildPrompts = (classesList: string[]): string[] => {
  return classesList.map(c => CLASS_EN_MAP[c] || c)
}

const batchRunning = computed(() =>
  !!batchProgress.value && ['running', 'pending'].includes(batchProgress.value.status)
)
const batchPercent = computed(() => {
  if (!batchProgress.value || !batchProgress.value.total) return 0
  return Math.round((batchProgress.value.done / batchProgress.value.total) * 100)
})

// ===== 图片列表分类过滤（级联下拉菜单） =====
// poly_* 三档按"轮廓质量"分：修过（人工复核）/ 待修（AI 歪稿）/ 还没轮廓，方便挑图精修
type FilterKey = 'all' | 'annotated' | 'ai' | 'manual' | 'unannotated' | 'poly_refined' | 'poly_draft' | 'poly_none'
const imageFilter = ref<FilterKey>('all')
const filterOpen = ref(false)
const counts = computed(() => {
  const total = items.value.length
  const annotated = items.value.filter(i => i.annotated).length
  const ai = items.value.filter(i => i.ai_annotated).length
  const manual = items.value.filter(i => i.annotated && !i.ai_annotated).length
  const unannotated = items.value.filter(i => !i.annotated).length
  const poly_refined = items.value.filter(i => i.has_polygon && i.reviewed).length
  const poly_draft = items.value.filter(i => i.has_polygon && !i.reviewed).length
  const poly_none = items.value.filter(i => !i.has_polygon).length
  const reviewed = items.value.filter(i => i.reviewed).length
  return { total, annotated, ai, manual, unannotated, poly_refined, poly_draft, poly_none, reviewed }
})
const filterLabels: Record<string, string> = {
  all: '全部',
  annotated: '已标注',
  ai: 'AI标注',
  manual: '人工标注',
  unannotated: '未标注',
  poly_refined: '轮廓·已修过',
  poly_draft: '轮廓·待修正',
  poly_none: '轮廓·还没有',
}
const currentFilterLabel = computed(() => {
  const label = filterLabels[imageFilter.value] || '全部'
  const count = counts.value[imageFilter.value === 'all' ? 'total' : imageFilter.value]
  return `${label}（${count}）`
})
const setFilter = (val: FilterKey) => {
  imageFilter.value = val
  filterOpen.value = false
}
const filteredItems = computed(() => {
  switch (imageFilter.value) {
    case 'annotated': return items.value.filter(i => i.annotated)
    case 'ai': return items.value.filter(i => i.ai_annotated)
    case 'manual': return items.value.filter(i => i.annotated && !i.ai_annotated)
    case 'unannotated': return items.value.filter(i => !i.annotated)
    case 'poly_refined': return items.value.filter(i => i.has_polygon && i.reviewed)
    case 'poly_draft': return items.value.filter(i => i.has_polygon && !i.reviewed)
    case 'poly_none': return items.value.filter(i => !i.has_polygon)
    default: return items.value
  }
})
// 点击过滤后的图片，定位到原始 items 索引
const selectImageByItem = (item: any) => {
  const idx = items.value.findIndex(i => i.image_id === item.image_id)
  if (idx >= 0) selectImage(idx)
}

// ===== 列表真正呈现的集合：类别过滤 ∩ 文件名搜索 =====
// 之前列表按 filteredItems 渲染，而标题计数、上一张/下一张都走全量 items，
// 三者口径不一致：筛完只剩 120 张，标题仍写 787，翻图还会翻到列表里找不到的图上。
// 现在统一以 visibleItems 为唯一口径（列表、计数、翻页、窗口都用它）。
const imageQuery = ref('')
const visibleItems = computed(() => {
  const base = filteredItems.value
  const q = imageQuery.value.trim().toLowerCase()
  if (!q) return base
  return base.filter(i => String(i.image_id || '').toLowerCase().includes(q))
})
// 当前图在可见集合里的位置；-1 表示这张被筛选/搜索条件挡掉了
const visibleIdx = computed(() => {
  const id = currentImage.value?.image_id
  if (!id) return -1
  return visibleItems.value.findIndex(i => i.image_id === id)
})
const canPrev = computed(() => visibleIdx.value > 0)
const canNext = computed(() => visibleIdx.value >= 0 && visibleIdx.value < visibleItems.value.length - 1)

// 窗口化渲染：只把当前图附近的一段条目放进 DOM（千张级的任务里
// 全量渲染会让滚动和切换都变钝），窗口跟着当前图滑动，
// 列表更长时由滚动到边缘触发平移。
const LIST_WINDOW = 80
const clampIdx = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(v, hi))
// 手动浏览产生的平移偏移：选图或改筛选/搜索条件后归零，重新回到跟随当前图
const browseOffset = ref(0)
const windowStart = computed(() => {
  const n = visibleItems.value.length
  if (n <= LIST_WINDOW) return 0
  const anchor = visibleIdx.value >= 0 ? visibleIdx.value - Math.floor(LIST_WINDOW / 2) : 0
  // H-2r：先把 anchor 收敛成「以当前图为中心」的合法基础位，再叠加浏览偏移整体收敛。
  // 原实现 anchor 未先收敛，当前图在列表头部（anchor=-40）时首次触底的 +40 偏移
  // 被下限 0 吞掉，表现为「滚到底没反应，须第二次滚动才平移」（批次 H 验收实测发现）。
  const base = clampIdx(anchor, 0, n - LIST_WINDOW)
  return clampIdx(base + browseOffset.value, 0, n - LIST_WINDOW)
})
const windowEnd = computed(() => Math.min(windowStart.value + LIST_WINDOW, visibleItems.value.length))
const windowedItems = computed(() => visibleItems.value.slice(windowStart.value, windowEnd.value))
const hasEarlier = computed(() => windowStart.value > 0)
const hasLater = computed(() => windowEnd.value < visibleItems.value.length)
// 滚到列表边缘时把窗口平移一段，取代原来的「更早/更晚」按钮
const LIST_SHIFT = 40          // 每次平移条数
const isShifting = ref(false)
const shiftWindow = async (delta: number) => {
  if (isShifting.value) return
  const el = imageListRef.value
  if (!el) return
  isShifting.value = true
  // 锚定：取当前视口顶部第一个可见项
  const listRect = el.getBoundingClientRect()
  let anchor: HTMLElement | null = null
  const kids = el.querySelectorAll<HTMLElement>('.image-item[data-img]')
  for (const k of Array.from(kids)) {
    if (k.getBoundingClientRect().bottom > listRect.top + 1) { anchor = k; break }
  }
  const anchorId = anchor?.dataset.img || ''
  const beforeTop = anchor ? anchor.getBoundingClientRect().top : 0
  browseOffset.value += delta
  await nextTick()
  if (anchorId) {
    const again = el.querySelector<HTMLElement>(`.image-item[data-img="${anchorId}"]`)
    if (again) {
      const afterTop = again.getBoundingClientRect().top
      el.scrollTop += (afterTop - beforeTop)   // 视觉位置保持不跳
    }
  }
  isShifting.value = false
}
const onListScroll = () => {
  const el = imageListRef.value
  if (!el || isShifting.value) return
  const NEAR = 8
  if (el.scrollTop <= NEAR && hasEarlier.value) shiftWindow(-LIST_SHIFT)
  else if (el.scrollHeight - el.scrollTop - el.clientHeight <= NEAR && hasLater.value) shiftWindow(LIST_SHIFT)
}

// 双击「图片列表 (N)」计数：输入序号直接跳到那一张
const jumpToIndex = async () => {
  const n = visibleItems.value.length
  if (!n) return
  const input = await showPrompt(`跳转到第几张？（1 ~ ${n}）`, String(visibleIdx.value + 1), '跳转图片')
  if (input == null) return
  const p = parseInt(String(input).trim(), 10)
  if (!Number.isFinite(p) || p < 1 || p > n) { showAlert(`请输入 1 ~ ${n} 之间的整数`); return }
  selectImageByItem(visibleItems.value[p - 1])   // 复用：会触发 selectImage → watch(currentImage) 重置 browseOffset 并把窗口对准
}

const imageListRef = ref<HTMLElement | null>(null)
// block:'nearest' 只在当前项真看不见时才动，避免每次切图列表都跳到顶上
const scrollActiveIntoView = async () => {
  await nextTick()
  imageListRef.value?.querySelector('.image-item.active')?.scrollIntoView({ block: 'nearest' })
}
// 换筛选/搜索条件后当前图可能被挡掉：自动落到结果里的第一张，
// 否则画布显示的图在列表里找不到，翻页也会因为没有游标而彻底卡住。
watch([imageFilter, imageQuery], () => {
  browseOffset.value = 0
  const list = visibleItems.value
  if (list.length && visibleIdx.value < 0) selectImageByItem(list[0])
  else scrollActiveIntoView() // 当前图仍在结果里时，窗口重锚了也要把它滚回视野
})
// 切图后把当前项滚进视野
watch(currentImage, () => {
  browseOffset.value = 0
  scrollActiveIntoView()
})

const createTask = async () => {
  if (!newTask.value.datasetId) {
    showToast('请选择数据集', 'info', 4000)
    return
  }
  
  // 检查数据集是否已准备
  const selectedDataset = datasets.value.find(ds => ds.dataset_id === newTask.value.datasetId)
  if (selectedDataset && selectedDataset.status === 'uploaded') {
    showToast('数据集尚未准备（prepare）。请先在数据集上传页面执行"准备"操作后再创建标注任务。', 'info', 4000)
    return
  }
  
  // 类别优先级：数据集类别文件能读到则以后端读取为准（真实类别名）；
  // 读不到（如尚未标注过的新数据集）时用输入框填写的类别兜底，避免标注任务无类别导致"请先设置类别"
  const rawClasses = (classesInput.value || '').trim()
  // 兼容中英文逗号分隔（", " / "， "混用均可）
  const inputClasses = rawClasses
    ? rawClasses.split(/[,，]/).map((s: string) => s.trim()).filter(Boolean)
    : undefined
  
  creatingTask.value = true
  try {
    const result = await createAnnotationTask(newTask.value.datasetId, newTask.value.version, inputClasses)
    currentTask.value = result.task_id
    
    // 使用返回的类别（从数据集类别文件读取，覆盖输入框内容）
    if (result.classes && result.classes.length > 0) {
      classes.value = result.classes
      classesInput.value = result.classes.join(', ')
    }
    
    // 显示导入信息
    if (result.imported_annotations > 0) {
      importedCount.value = result.imported_annotations
      showToast(`任务创建成功！\n共 ${result.total_images} 张图片\n已导入 ${result.imported_annotations} 个已有标注`, 'info', 4000)
    }
    
    await loadTaskItems()
    saveAnnotateState()
  } catch (error: any) {
    if (!error?.__toastShown) showToast('创建任务失败: ' + (error.response?.data?.detail || error.message), 'error', 4000)
  } finally {
    creatingTask.value = false
  }
}

const loadTaskItems = async () => {
  if (!currentTask.value) return
  
  loadingItems.value = true
  try {
    const result = await getTaskItems(currentTask.value)
    items.value = result.items
    taskDatasetId.value = result.dataset_id || ''
    taskModelId.value = result.model_id || ''
    // 每次进入任务重置：存在困难/空白样本时弹一次提醒（「知道了」仅本次会话生效）
    suggestPrompted.value = false
    nextTick(() => {
      if (suggestBannerCount.value > 0 && !suggestPrompted.value) {
        suggestPromptOpen.value = true
        suggestPrompted.value = true
      }
    })
    
    // 更新类别（如果后端返回了）
    if (result.classes && result.classes.length > 0) {
      classes.value = result.classes
      if (!classesInput.value) {
        classesInput.value = result.classes.join(', ')
      }
    }
    
    // 记录导入的标注数量
    if (result.imported_annotations) {
      importedCount.value = result.imported_annotations
    }
    
    if (items.value.length > 0) {
      selectImage(0)
    }
  } catch (error: any) {
    if (!error?.__toastShown) showToast('加载失败: ' + (error.response?.data?.detail || error.message), 'error', 4000)
  } finally {
    loadingItems.value = false
  }
}

// 持久化当前标注任务，便于中途离开页面后返回继续标注
const saveAnnotateState = () => {
  try {
    localStorage.setItem(ANNOTATE_STORE_KEY, JSON.stringify({
      taskId: currentTask.value,
      imageId: currentImage.value?.image_id || null,
      index: currentIndex.value,
      batchId: batchId.value,  // 恢复批量预标注轮询
    }))
  } catch (e) {
    console.error('保存标注状态失败:', e)
  }
}

// 清空持久化的标注任务（任务删除或完成后）
const clearAnnotateState = () => {
  try {
    localStorage.removeItem(ANNOTATE_STORE_KEY)
  } catch (e) {
    console.error('清空标注状态失败:', e)
  }
}

// ------------------------------------------------------------------
// 未闭合的点位草稿（描轮廓顶点 / 抠图会话点）单独持久化：
// 页面刷新、浏览器崩溃、误关标签页后回来，点了一半的区域还能接着连。
// 每落一个点就同步写一次 localStorage（不依赖 watch/卸载事件，刷新瞬间也不丢）；
// 闭合、取消、切图时清掉 —— 草稿是"这一张图、这一次未完成的描点"，不该跨图复活。
// ------------------------------------------------------------------
const POINT_DRAFT_KEY = 'annotate_point_draft'
const POINT_DRAFT_TTL_MS = 7 * 24 * 3600 * 1000   // 超过 7 天的残留草稿直接作废
// 刷新恢复后、用户还没按 G 进抠图模式前，抠图点位暂存这里（toggleCutoutMode 时接上）
let pendingCutoutPts: { x: number; y: number; label: number }[] | null = null

const persistPointDraft = () => {
  try {
    const poly = polyDraft.value
    const cutout = cutoutPts.value.length ? cutoutPts.value : (pendingCutoutPts || [])
    if (!poly.length && !cutout.length) { localStorage.removeItem(POINT_DRAFT_KEY); return }
    localStorage.setItem(POINT_DRAFT_KEY, JSON.stringify({
      taskId: currentTask.value,
      imageId: currentImage.value?.image_id || null,
      poly: poly.map(pt => [pt[0], pt[1]]),
      cutout: cutout.map(p => ({ x: p.x, y: p.y, label: p.label })),
      ts: Date.now(),
    }))
  } catch (e) {
    console.error('保存点位草稿失败:', e)
  }
}

// 读取指定任务+图片的未完成草稿；对不上号或过期一律视为没有
const peekPointDraft = (taskId: string, imageId: string) => {
  try {
    const raw = localStorage.getItem(POINT_DRAFT_KEY)
    if (!raw) return null
    const d = JSON.parse(raw)
    if (!d || d.taskId !== taskId || d.imageId !== imageId) return null
    if (!d.ts || Date.now() - d.ts > POINT_DRAFT_TTL_MS) { localStorage.removeItem(POINT_DRAFT_KEY); return null }
    return d
  } catch {
    return null
  }
}

// 把恢复出来的草稿接回会话：轮廓草稿直接进描轮廓模式；抠图点位等按 G 时再接上
const applyPointDraft = (draft: any) => {
  const poly: number[][] = Array.isArray(draft.poly) ? draft.poly : []
  const cutout: { x: number; y: number; label: number }[] = Array.isArray(draft.cutout) ? draft.cutout : []
  if (poly.length >= 2) {
    polyDrawMode.value = true
    polyDraft.value = poly
    showToast(`已恢复上次未闭合的轮廓草稿（${poly.length} 个顶点）：继续描点，点回起点或回车闭合`, 'info', 8000)
  } else if (cutout.length) {
    pendingCutoutPts = cutout
    showToast(`已恢复上次未完成的抠图点位（${cutout.length} 个点）：按 G 进入抠图模式即可接着点`, 'info', 8000)
  }
  redrawCanvas()
  persistPointDraft()
}

// 恢复上次未完成的标注任务
const restoreAnnotateState = async () => {
  // URL 直达（?dataset_id=xxx）优先，跳过本地恢复，避免旧任务覆盖直达任务
  if (new URLSearchParams(window.location.search).has('dataset_id')) return
  let saved: any = null
  try {
    const raw = localStorage.getItem(ANNOTATE_STORE_KEY)
    if (raw) saved = JSON.parse(raw)
  } catch (e) {
    return
  }
  if (!saved?.taskId) return
  // 优先恢复批量预标注轮询：不因下面的任务/图片加载失败而中断（批量任务在后端后台仍在运行）
  if (saved.batchId) {
    batchId.value = saved.batchId
    batchProgress.value = { status: 'running', total: 1, done: 0, boxes_written: 0, current_image: '' }
    pollBatch()
  }
  currentTask.value = saved.taskId
  if (!currentTask.value) return
  try {
    const result = await getTaskItems(currentTask.value)
    items.value = result.items
    if (result.classes && result.classes.length > 0) {
      classes.value = result.classes
      classesInput.value = result.classes.join(', ')
    }
    if (result.imported_annotations) {
      importedCount.value = result.imported_annotations
    }
    if (items.value.length > 0) {
      const targetIdx = (saved.index && saved.index < items.value.length) ? saved.index : 0
      // 必须在 selectImage 之前读草稿：selectImage 会清会话状态并同步删除本地草稿记录
      const draft = peekPointDraft(saved.taskId, items.value[targetIdx]?.image_id || '')
      await selectImage(targetIdx)
      // 刷新/崩溃恢复：上一张图点了一半还没闭合的轮廓草稿 / 抠图点位接回来
      if (draft) applyPointDraft(draft)
    }
  } catch (e: any) {
    // 任务已不存在或加载失败：清空本地恢复记录，避免每次刷新都重复失败而清空图片列表
    currentTask.value = null
    items.value = []
    clearAnnotateState()
    if (!e?.__toastShown) showToast('恢复标注任务失败，已清空记录，请重新创建任务: ' + (e.response?.data?.detail || e.message), 'error', 4000)
  }
}

const selectImage = async (index: number) => {
  // 若有未保存改动，先立即保存再切换（避免切图丢弃 1.5s 防抖窗口内的修改）
  if (dirty.value) {
    clearTimeout(autoSaveTimer.value)
    await persistAnnotation(true)
  }
  currentIndex.value = index
  currentImage.value = items.value[index]
  currentBoxes.value = []
  selectedBoxIndex.value = -1
  currentImageHasAi.value = false
  imageAudit.value = null  // 切图清空上一张的审核结果
  currentSampleType.value = ''  // 切换图片重置样本类型（已有标注时由 loadExistingAnnotation 回填）
  // 切换图片：取消上一张的待保存状态（未保存的改动在离页时统一处理）
  dirty.value = false
  clearTimeout(autoSaveTimer.value)
  // 切换图片时重置编辑/撤销状态与视图
  editMode.value = 'idle'
  isDrawing.value = false
  isPanning.value = false
  moveStartPt.value = null
  origBox.value = null
  resizeHandle.value = -1
  polyDraft.value = []      // 未闭合的轮廓草稿不能跨图残留
  polyPending.value = null
  clearCutoutSession()      // 抠图的提示点同样不能跨图残留（模式本身保留，方便连抠）
  dragVertexIdx.value = -1
  undoStack.value = []
  redoStack.value = []
  // 人工编辑标记先清零，已有标注的图由 loadExistingAnnotation 回读后端 reviewed
  manualTouched.value = false

  await nextTick()
  
  // 如果图片已有标注，先加载标注
  if (currentImage.value.annotated && currentTask.value) {
    await loadExistingAnnotation()
  }
  
  loadImageToCanvas()
  saveAnnotateState()
}

// 加载已有标注
const loadExistingAnnotation = async () => {
  if (!currentTask.value || !currentImage.value) return
  
  loadingAnnotation.value = true
  try {
    const result = await getImageAnnotation(currentTask.value, currentImage.value.image_id)
    if (result) {
      currentSampleType.value = result.sample_type || ''
      // 回读"人工复核过"标记：之前修过的图再打开仍算已修，不会被筛回"待修正"
      manualTouched.value = !!result.reviewed
      if (result.boxes && result.boxes.length > 0) {
        currentBoxes.value = result.boxes.map((box: any) => ({
          class_id: box.class_id,
          x1: box.x1,
          y1: box.y1,
          x2: box.x2,
          y2: box.y2,
          // 分割轮廓必须回填，否则重新打开图片时分割标注只剩矩形（顶点也就无从微调）
          ...(box.polygon && box.polygon.length >= 3 ? { polygon: box.polygon as number[][] } : {})
        }))
      }
    }
  } catch (error: any) {
    console.error('加载标注失败:', error)
  } finally {
    loadingAnnotation.value = false
  }
}

const loadImageToCanvas = () => {
  if (!currentImage.value || !canvasRef.value) return
  // 切图先清掉上一张的边缘数据：onload 前若有重绘（fitView/标注刷新），旧图叠加层会"残留"到新图上
  edgeData = null
  edgeOverlayCanvas = null

  const canvas = canvasRef.value
  ctx.value = canvas.getContext('2d')
  
  // 处理图片路径
  // 后端返回的路径应该是相对于 DATA_DIR 的（例如：datasets/ds_xxx/images/xxx.jpg）
  // 静态文件服务挂载在 /static，目录是 backend/data
  // 所以需要：/static/datasets/...
  let imagePath = currentImage.value.image_path
  // 统一转换为正斜杠
  imagePath = imagePath.replace(/\\/g, '/')
  // 去掉可能的 backend/data/ 或 data/ 前缀（兼容旧格式）
  imagePath = imagePath.replace(/^(backend\/)?data\//, '')
  // 确保路径不以 / 开头（因为会加上 /static/ 前缀）
  imagePath = imagePath.replace(/^\//, '')
  // 静态文件路径不使用 API 基础路径，直接使用 /static/
  const imageUrl = '/static/' + imagePath
  
  img.value = new Image()
  img.value.crossOrigin = 'anonymous'
  
  img.value.onload = () => {
    if (!img.value || !ctx.value) return
    canvas.width = img.value.width
    canvas.height = img.value.height

    // 边缘辅助：离线构建 Sobel 梯度场与叠加层（异步不阻塞首帧）
    setTimeout(buildEdgeMap, 0)

    // 新图片加载后重置视图并居中
    fitView()

    // 绘制已有标注
    if (currentBoxes.value.length > 0) {
      redrawCanvas()
    }
  }
  
  img.value.onerror = (error) => {
    console.error('Image load error:', {
      url: imageUrl,
      originalPath: currentImage.value.image_path,
      error
    })
    // 显示更友好的错误提示
    const errorMsg = `图片加载失败\n路径: ${imageUrl}\n\n请检查：\n1. 后端服务是否运行\n2. 静态文件服务是否正常\n3. 图片文件是否存在`
    showToast(errorMsg, 'error', 5000)
  }
  
  img.value.src = imageUrl
}

const screenToImg = (e: MouseEvent) => {
  const rect = canvasRef.value!.getBoundingClientRect()
  return {
    x: (e.clientX - rect.left - viewOffset.value.x) / viewScale.value,
    y: (e.clientY - rect.top - viewOffset.value.y) / viewScale.value
  }
}

const getContainerSize = () => {
  const el = canvasRef.value?.parentElement
  return el ? { w: el.clientWidth, h: el.clientHeight } : { w: 800, h: 600 }
}

// 视图适应窗口（图片居中显示）
const fitView = () => {
  if (!img.value || !canvasRef.value) return
  const { w, h } = getContainerSize()
  canvasRef.value.width = Math.max(w, 1)
  canvasRef.value.height = Math.max(h, 1)
  if (!img.value.width) return
  const s = Math.min(w / img.value.width, h / img.value.height)
  viewScale.value = s
  viewOffset.value = {
    x: Math.max((w - img.value.width * s) / 2, 0),
    y: Math.max((h - img.value.height * s) / 2, 0)
  }
  redrawCanvas()
}

const zoomBy = (factor: number) => {
  if (!img.value || !canvasRef.value) return
  const { w, h } = getContainerSize()
  const cx = w / 2
  const cy = h / 2
  const ns = Math.min(Math.max(viewScale.value * factor, 0.05), 20)
  viewOffset.value = {
    x: cx - (cx - viewOffset.value.x) * (ns / viewScale.value),
    y: cy - (cy - viewOffset.value.y) * (ns / viewScale.value)
  }
  viewScale.value = ns
  redrawCanvas()
}

// 滚轮缩放（以鼠标为中心）
const onWheel = (e: WheelEvent) => {
  e.preventDefault()
  if (!img.value || !canvasRef.value) return
  const rect = canvasRef.value.getBoundingClientRect()
  const mx = e.clientX - rect.left
  const my = e.clientY - rect.top
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
  const ns = Math.min(Math.max(viewScale.value * factor, 0.05), 20)
  viewOffset.value = {
    x: mx - (mx - viewOffset.value.x) * (ns / viewScale.value),
    y: my - (my - viewOffset.value.y) * (ns / viewScale.value)
  }
  viewScale.value = ns
  redrawCanvas()
}

const onCanvasLeave = () => {
  // 挂起的"按下待判定"移出画布后可能再也等不到 mouseup，直接取消，避免残留
  polyPending.value = null
  cutoutErase.value = null   // 右键擦除拖到画布外收不到 mouseup，移出即取消
  if (isDrawing.value) {
    isDrawing.value = false
    editMode.value = 'idle'
  }
  // 顶点拖动中途移出画布：不收尾会卡在 vertex 态，鼠标一动就乱改轮廓
  if (editMode.value === 'vertex') {
    const box = currentBoxes.value[selectedBoxIndex.value]
    if (box) syncBBoxFromPolygon(box)
    editMode.value = 'idle'
    dragVertexIdx.value = -1
  }
  isPanning.value = false
}

// 命中点 p 是否在某个可见框内（从后往前，后画的在上层）
// 分割框按轮廓判定：外接矩形会盖住紧挨着的相邻目标，只看矩形极易误选错框
const hitBox = (p: { x: number; y: number }) => {
  const cur = currentClass.value
  const visible = (b: BBox) => String(cur) === '-1' || b.class_id === Number(cur)
  // 第一轮：精确命中（分割框=点在轮廓内，检测框=点在矩形内）
  for (let i = currentBoxes.value.length - 1; i >= 0; i--) {
    const b = currentBoxes.value[i]
    if (!visible(b)) continue
    const poly = polyOf(b)
    if (poly) {
      if (pointInPolygon(p, poly)) return i
    } else if (p.x >= b.x1 && p.x <= b.x2 && p.y >= b.y1 && p.y <= b.y2) {
      return i
    }
  }
  // 第二轮：兜底按外接矩形命中，保证分割框在小图上依然可点选/拖动
  for (let i = currentBoxes.value.length - 1; i >= 0; i--) {
    const b = currentBoxes.value[i]
    if (!visible(b)) continue
    if (p.x >= b.x1 && p.x <= b.x2 && p.y >= b.y1 && p.y <= b.y2) return i
  }
  return -1
}

// 命中选中框的控制点（0-3 角，4-7 边中点），未命中返回 -1
const hitHandle = (idx: number, p: { x: number; y: number }) => {
  const b = currentBoxes.value[idx]
  // 分割框不允许拖角缩放：矩形是由轮廓反推出来的，单独改矩形会让框与轮廓对不上
  if (!b || polyOf(b)) return -1
  const hw = 10 / viewScale.value
  const pts = [
    { x: b.x1, y: b.y1 }, { x: (b.x1 + b.x2) / 2, y: b.y1 }, { x: b.x2, y: b.y1 },
    { x: b.x2, y: (b.y1 + b.y2) / 2 }, { x: b.x2, y: b.y2 }, { x: (b.x1 + b.x2) / 2, y: b.y2 },
    { x: b.x1, y: b.y2 }, { x: b.x1, y: (b.y1 + b.y2) / 2 }
  ]
  for (let i = 0; i < 8; i++) {
    if (Math.abs(p.x - pts[i].x) <= hw && Math.abs(p.y - pts[i].y) <= hw) return i
  }
  return -1
}

// 根据控制点缩放选中框
const applyResize = (p: { x: number; y: number }) => {
  const b = origBox.value
  const idx = selectedBoxIndex.value
  if (!b || idx < 0) return
  const h = resizeHandle.value
  let x1 = b.x1, y1 = b.y1, x2 = b.x2, y2 = b.y2
  if (h === 0 || h === 6 || h === 7) x1 = p.x
  if (h === 2 || h === 4 || h === 3) x2 = p.x
  if (h === 0 || h === 1 || h === 2) y1 = p.y
  if (h === 4 || h === 5 || h === 6) y2 = p.y
  if (x1 > x2) [x1, x2] = [x2, x1]
  if (y1 > y2) [y1, y2] = [y2, y1]
  const box = currentBoxes.value[idx]
  if (box) { box.x1 = x1; box.y1 = y1; box.x2 = x2; box.y2 = y2 }
}

// 画框/移动/缩放交互入口
const startDrawing = (e: MouseEvent) => {
  if (!canvasRef.value || !img.value) return
  suppressCanvasClick.value = false
  // 按住空格 + 左键 = 平移
  if (spaceDown.value) {
    isPanning.value = true
    panStart.value = { x: e.clientX, y: e.clientY }
    return
  }
  const p = screenToImg(e)
  // 抠图模式：左键点目标 → 丢进队列后台算，点完立刻可以点下一个，不用干等
  if (cutoutMode.value) {
    // 右键按下：先记下"擦除候选"（命中带轮廓的目标才有效），拖出距离够就在松手时挖掉那块；
    // 只点不拖 = 老行为（补一个"抠多了"的负点），判定都放到 endDrawing
    if (e.button === 2) {
      const hit = hitBox(p)
      const box = hit >= 0 ? currentBoxes.value[hit] : null
      cutoutErase.value = {
        box: box && polyOf(box) ? box : null,
        x1: p.x, y1: p.y, x2: p.x, y2: p.y
      }
      redrawCanvas()
      return
    }
    if (e.button !== 0) return
    if (cutoutTooClose(p)) return
    // 离散点击点的"同一片区域"判定交给后端（逐点掩码互含分组）：
    // 点在当前目标上 = 补点微调；点在另一个已落地目标上 = 换目标继续累积；
    // 点在空白处 = 先当作"可能相连"累积进当前会话，等后端回 coverage 再决定
    // 是并入当前目标还是自动拆成新目标（见 landCutout）。
    const hitExisting = currentBoxes.value.find(b => pointInsideBox(b, p))
    if (hitExisting && hitExisting !== cutoutBox.value) {
      // 命中另一个已有目标：并入它。若当前会话还攥着上一个目标，先给它收尾，
      // 免得会话点位被两个目标混在一起。
      if (cutoutBox.value) clearCutoutSession()
      cutoutSid.value++
      cutoutBox.value = hitExisting
      cutoutPts.value = []
    } else if (!cutoutBox.value && !hitExisting && !cutoutPts.value.length) {
      clearCutoutSession()   // 全新目标：换会话编号，在途结果不再认领这个位置
      // 注：cutoutPts 非空 = 刷新恢复出来的未处理点位（还没落成框），不清会话、继续累积，
      // 由后端的连通判断决定并入哪片区域
    }
    cutoutPts.value = [...cutoutPts.value, { x: p.x, y: p.y, label: 1 }]
    persistPointDraft()
    redrawCanvas()
    queueCutout()
    return
  }
  // 多边形绘制模式：左键逐点落轮廓顶点，点回起点附近（≥3 点）即闭合成一个分割框
  if (polyDrawMode.value) {
    if (e.button !== 0 || interactiveBusy.value) return
    // 草稿还没起头时，若按在已有分割目标的拐点或轮廓内，先挂起不决定：
    // 真的拖了 → 改拐点 / 整体平移；只是点了一下 → 仍按"落一个轮廓顶点"处理。
    if (polyDraft.value.length === 0) {
      const [vi, vh] = hitAnyVertex(p)
      const en = vh !== -1 ? null : hitPolyEdgeNear(p)
      const inside = (vh === -1 && !en) ? hitPolygonInterior(p) : -1
      if (vh !== -1 || en || inside >= 0) {
        polyPending.value = {
          kind: (vh !== -1 || en) ? 'vertex' : 'move',
          idx: vh !== -1 ? vi : (en ? en[0] : inside),
          v: vh,
          pt: p,
          sx: e.clientX,
          sy: e.clientY
        }
        return
      }
    }
    addDraftPoint(p)
    return
  }
  // 交互式标注模式：拖拽框选局部提示区域（不生成类别框）
  if (interactiveMode.value) {
    if (interactiveBusy.value) return
    selectedBoxIndex.value = -1
    editMode.value = 'draw'
    startPos.value = p
    currentPos.value = p
    isDrawing.value = true
    redrawCanvas()
    return
  }
  // 1) 命中分割轮廓顶点 → 进入顶点微调（不用先选中，点住拐点即可拖；点边中点先插点再拖）
  const [vi, vh] = hitAnyVertex(p)
  if (vh !== -1 && grabVertex(vi, vh)) return
  // 1.5) 贴住轮廓边（放大后没点准拐点）→ 在投影点插顶点并拖动，避免误触发整体平移
  const en = hitPolyEdgeNear(p)
  if (en && grabEdgeVertex(en[0], en[1], en[2], en[3])) return
  // 2) 命中矩形控制点 → 缩放
  if (selectedBoxIndex.value >= 0) {
    const h = hitHandle(selectedBoxIndex.value, p)
    if (h >= 0) {
      editMode.value = 'resize'
      resizeHandle.value = h
      origBox.value = cloneBox(currentBoxes.value[selectedBoxIndex.value])
      pushUndo()
      return
    }
  }
  // 3) 命中某个框内部 → 选中并移动
  const hit = hitBox(p)
  if (hit >= 0) {
    grabMove(hit, p)
    return
  }
  // 3) 空白处 → 画新框
  selectedBoxIndex.value = -1
  editMode.value = 'draw'
  startPos.value = p
  currentPos.value = p
  isDrawing.value = true
  redrawCanvas()
}

const drawing = (e: MouseEvent) => {
  if (!canvasRef.value) return
  // 抠图模式右键拖拽：更新擦除矩形（拖到哪挖到哪的预览）
  if (cutoutErase.value) {
    const p = screenToImg(e)
    cutoutErase.value.x2 = p.x
    cutoutErase.value.y2 = p.y
    redrawCanvas()
    return
  }
  if (isPanning.value) {
    viewOffset.value = {
      x: viewOffset.value.x + (e.clientX - panStart.value.x),
      y: viewOffset.value.y + (e.clientY - panStart.value.y)
    }
    panStart.value = { x: e.clientX, y: e.clientY }
    redrawCanvas()
    return
  }
  if (editMode.value === 'draw' && isDrawing.value) {
    currentPos.value = screenToImg(e)
    redrawCanvas()
    return
  }
  // 描轮廓模式的"按下待判定"：移动超过几个像素才算编辑（改拐点 / 整体平移），否则算落点
  const pend = polyPending.value
  if (pend) {
    if (e.buttons !== 1) { polyPending.value = null; return }
    if (Math.abs(e.clientX - pend.sx) + Math.abs(e.clientY - pend.sy) < 4) return
    polyPending.value = null
    if (pend.kind === 'vertex') {
      if (pend.v !== -1) { grabVertex(pend.idx, pend.v) }
      else {
        const e2 = hitPolyEdgeNear(pend.pt)
        if (e2) grabEdgeVertex(e2[0], e2[1], e2[2], e2[3])
      }
    } else grabMove(pend.idx, pend.pt)
    // 不 return：本次移动直接交给下面的顶点/平移分支处理
  }
  // 鼠标在画布外松开时收不到 mouseup：按键已抬起就立即收尾，否则框/拐点会跟着光标跑
  if (e.buttons !== 1 && editMode.value !== 'idle' && editMode.value !== 'draw') {
    endDrawing(e)
    return
  }
  if (editMode.value === 'move' && selectedBoxIndex.value >= 0 && moveStartPt.value && origBox.value) {
    const p = screenToImg(e)
    const dx = p.x - moveStartPt.value.x
    const dy = p.y - moveStartPt.value.y
    if (!moveUndoPushed.value && (Math.abs(dx) > 1 || Math.abs(dy) > 1)) { pushUndo(); moveUndoPushed.value = true }
    const box = currentBoxes.value[selectedBoxIndex.value]
    applyMove(box, origBox.value, dx, dy)
    redrawCanvas()
    return
  }
  // 顶点微调：只动被抓住的那个点（矩形在最后由轮廓反推）
  if (editMode.value === 'vertex' && selectedBoxIndex.value >= 0 && dragVertexIdx.value >= 0) {
    const poly = polyOf(currentBoxes.value[selectedBoxIndex.value])
    if (poly && dragVertexIdx.value < poly.length) {
      const p = snapToEdge(screenToImg(e))   // 拖点吸附：松手前顶点持续贴边
      poly[dragVertexIdx.value] = [p.x, p.y]
      redrawCanvas()
    }
    return
  }
  if (editMode.value === 'resize' && selectedBoxIndex.value >= 0) {
    applyResize(screenToImg(e))
    redrawCanvas()
  }
}

const endDrawing = async (e: MouseEvent) => {
  // 抠图模式右键收尾：拖出距离够 = 区域擦除（把矩形那块从轮廓挖掉）；只点不拖 = 补一个负点
  if (cutoutErase.value) {
    const er = cutoutErase.value
    cutoutErase.value = null
    const p = screenToImg(e)
    const w = Math.abs(p.x - er.x1)
    const h = Math.abs(p.y - er.y1)
    if (er.box && (w > 8 || h > 8)) {
      await applyErase(er.box, { x1: Math.min(er.x1, p.x), y1: Math.min(er.y1, p.y),
                                 x2: Math.max(er.x1, p.x), y2: Math.max(er.y1, p.y) })
      redrawCanvas()
      return
    }
    // 没拖出矩形 = 老行为：右键点一下抠多了的地方
    if (!er.box && !cutoutBox.value) {
      showToast(cutoutBusy.value ? '这个目标还在算，等它出轮廓后再用右键点多余处'
                                 : '先左键点一下目标内部，抠出来后再用右键点抠多了的地方', 'info')
      redrawCanvas()
      return
    }
    if (cutoutBox.value) {
      cutoutPts.value = [...cutoutPts.value, { x: p.x, y: p.y, label: 0 }]
      persistPointDraft()
      redrawCanvas()
      queueCutout()
    } else {
      showToast('右键点一下=抠掉这个点；拖一个框=整块挖掉', 'info')
      redrawCanvas()
    }
    return
  }
  if (isPanning.value) {
    isPanning.value = false
    return
  }
  // 描轮廓模式下按下却没拖动 = 想选中这个已有目标（拖起来才是改拐点/平移），
  // 不能把它当成新轮廓的第一个点，否则点一下就冒出个草稿尖角。
  if (polyPending.value) {
    const pd = polyPending.value
    polyPending.value = null
    suppressCanvasClick.value = true
    editMode.value = 'idle'
    dragVertexIdx.value = -1
    selectedBoxIndex.value = pd.idx
    if (polyMoveHintShown.value !== (currentImage.value?.image_id || '')) {
      polyMoveHintShown.value = currentImage.value?.image_id || ''
      showToast('已选中分割目标：按住白色方块拖动改拐点 · 按住轮廓内部拖动整体平移 · 空心圆加点 · 右键点删点', 'info')
    }
    redrawCanvas()
    return
  }
  if (editMode.value === 'draw' && isDrawing.value) {
    isDrawing.value = false
    currentPos.value = screenToImg(e)
    const x1 = Math.min(startPos.value.x, currentPos.value.x)
    const y1 = Math.min(startPos.value.y, currentPos.value.y)
    const x2 = Math.max(startPos.value.x, currentPos.value.x)
    const y2 = Math.max(startPos.value.y, currentPos.value.y)
    editMode.value = 'idle'
    // 交互式标注：框选完成 → 调接口在该区域内检测
    if (interactiveMode.value) {
      if (x2 - x1 > 5 && y2 - y1 > 5) {
        interactiveRegion.value = { x1, y1, x2, y2 }
        runInteractiveLabel()
      }
      redrawCanvas()
      return
    }
    if (x2 - x1 > 5 && y2 - y1 > 5) {
      // 新框类别只取"绘制类别"下拉框，与显示过滤器（currentClass）无关
      let cls = Number(drawClass.value)
      if (Number.isNaN(cls) || cls < 0) cls = 0
      if (classes.value.length && cls >= classes.value.length) cls = classes.value.length - 1
      pushUndo()
      currentBoxes.value.push({ class_id: cls, x1, y1, x2, y2 })
      selectedBoxIndex.value = currentBoxes.value.length - 1
    }
    redrawCanvas()
    return
  }
  if (editMode.value === 'vertex') {
    const box = currentBoxes.value[selectedBoxIndex.value]
    if (box) syncBBoxFromPolygon(box)
    editMode.value = 'idle'
    dragVertexIdx.value = -1
    redrawCanvas()
    return
  }
  if (editMode.value === 'move' || editMode.value === 'resize') {
    editMode.value = 'idle'
    moveStartPt.value = null
    origBox.value = null
    resizeHandle.value = -1
    moveUndoPushed.value = false
  }
}

const onCanvasClick = (e: MouseEvent) => {
  if (!canvasRef.value || !img.value) return
  if (suppressCanvasClick.value) { suppressCanvasClick.value = false; return }
  if (editMode.value !== 'idle') return
  if (interactiveMode.value) return
  if (cutoutMode.value) return
  const p = screenToImg(e)
  // 描轮廓模式：正在落点时交给 mousedown 逐点处理；空闲时点击仍可选中已有分割目标
  if (polyDrawMode.value) {
    if (polyDraft.value.length) return
    const hit = hitBox(p)
    if (hit >= 0) { selectedBoxIndex.value = hit; redrawCanvas() }
    return
  }
  selectedBoxIndex.value = hitBox(p)
  redrawCanvas()
}

// 右键：抠图模式下 = 拖框挖掉多余那块（点不拖=负点微调，见 endDrawing）；描轮廓模式下 = 删掉某个轮廓顶点
const onCanvasContextMenu = (e: MouseEvent) => {
  if (!canvasRef.value || !img.value) return
  e.preventDefault()
  // 抠图模式的右键语义已由 mousedown/endDrawing 接管（拖=擦除，点=负点），这里只拦默认菜单
  if (cutoutMode.value) return
  const p = screenToImg(e)
  if (polyDrawMode.value && polyDraft.value.length) {
    // 绘制中右键 = 单点回退：命中哪个顶点删哪个，没命中则删最后一个点（不再清空整条草稿）
    const d = polyDraft.value
    let vi = -1
    for (let i = d.length - 1; i >= 0; i--) {
      if (Math.abs(p.x - d[i][0]) <= hitRadius() && Math.abs(p.y - d[i][1]) <= hitRadius()) { vi = i; break }
    }
    const del = vi >= 0 ? vi : d.length - 1
    polyDraft.value.splice(del, 1)
    if (!polyDraft.value.length) cancelPolygonDraft()
    else redrawCanvas()
    return
  }
  let idx = selectedBoxIndex.value
  let v = idx >= 0 ? hitVertex(idx, p) : -1
  if (v === -1) {
    const [i, vv] = hitAnyVertex(p)
    idx = i
    v = vv
  }
  if (v >= 0) removeVertex(idx, v)
}

// 双击：抠图模式下表示"这个目标抠好了，换下一个"；描轮廓落点中双击=误触，吞掉
// （既不闭合也不复位视图，防止打断标注）；其余情况仍是复位视图
const onCanvasDblClick = () => {
  if (cutoutMode.value) { finishCutoutTarget(); return }
  if (polyDrawMode.value && polyDraft.value.length) return
  fitView()
}

// 交互式标注：在用户框选的局部区域内按文本提示检测目标，追加到当前标注
const runInteractiveLabel = async () => {
  if (!interactiveRegion.value || !currentTask.value || !currentImage.value) return
  if (!classes.value.length) {
    showToast('当前任务没有类别，无法进行交互式标注', 'info', 4000)
    return
  }
  interactiveBusy.value = true
  try {
    // 交互式标注只按"绘制类别"检测（文本提示），返回的 class_id 是子集索引需映射回全局
    const single = drawClass.value >= 0 && drawClass.value < classes.value.length
    const cls = single ? [classes.value[drawClass.value]] : classes.value
    const res = await interactiveLabelImage(
      currentTask.value,
      currentImage.value.image_id,
      cls,
      aiConf.value,
      buildPrompts(cls),
      interactiveRegion.value
    )
    const base = single ? drawClass.value : 0
    // SAM 精修返回的分割轮廓必须透传，否则前端只显示矩形，分割标注就白做了
    const newBoxes: BBox[] = (res.boxes || []).map((b: any) => ({
      class_id: base + (b.class_id ?? 0),
      x1: b.x1,
      y1: b.y1,
      x2: b.x2,
      y2: b.y2,
      ...(b.polygon && b.polygon.length >= 3 ? { polygon: b.polygon as number[][] } : {})
    }))
    if (newBoxes.length) {
      pushUndo()
      currentBoxes.value.push(...newBoxes)
      showToast(`已在所选区域标注 ${newBoxes.length} 个目标`, 'info', 4000)
    } else {
      showToast('该区域未检测到目标，可调整区域范围或类别后重试', 'info', 4000)
    }
    redrawCanvas()
  } catch (e: any) {
    if (!e?.__toastShown) showToast('交互式标注失败: ' + (e.response?.data?.detail || e.message), 'error', 4000)
  } finally {
    interactiveBusy.value = false
    interactiveRegion.value = null
  }
}

// 启用/退出交互式标注模式
const toggleInteractiveMode = () => {
  interactiveMode.value = !interactiveMode.value
  if (!interactiveMode.value) {
    interactiveRegion.value = null
  }
  // 两种模式都要占鼠标，互斥切换
  if (interactiveMode.value) {
    if (polyDrawMode.value) {
      polyDrawMode.value = false
      cancelPolygonDraft()
    }
    exitCutoutMode()
  }
  redrawCanvas()
}

// ------------------------------------------------------------------
// 点选抠图：把修图软件里"点一下抠一个"的交互搬到分割标注上。
// 和「轮廓·分割」的区别：那条要检测器先猜出方框，碎水洼、连片浅水面猜不出来；
// 这条完全由用户点的位置决定，点哪儿抠哪儿，拐角不用人一个一个描。
// ------------------------------------------------------------------
// 放弃"当前正在抠的这个目标"（已落到标注列表里的框保留，Ctrl+Z 可撤销）
const clearCutoutSession = () => {
  cutoutSid.value++      // 换目标/切图：在途任务回来后不再认领"当前目标"这个位置
  cutoutPts.value = []
  cutoutBox.value = null
  cutoutCoverage.value = 1
  pendingCutoutPts = null   // 会话作废：刷新恢复的暂存点同样不再接上（调用方若要先恢复，须先取走）
  persistPointDraft()
}

const exitCutoutMode = () => {
  if (!cutoutMode.value) return
  cutoutMode.value = false
  // 退出即"别再抠了"：排队中的直接撤掉，在途那份回来后也只字不落地。
  // 否则会出现"人都退出了，图框还在往外冒"，切了图甚至会落到别的图上。
  cutoutExitSeq++
  cutoutQueue.value = []
  clearCutoutSession()
}

// 区域擦除：把矩形那块从目标的轮廓里挖掉（后端纯几何、毫秒级）。
// 剩多块 → 第一块改回原框，其余作为新框插入；全挖空 → 删掉该框。Ctrl+Z 可撤销。
const applyErase = async (box: BBox, rect: { x1: number; y1: number; x2: number; y2: number }) => {
  const poly = polyOf(box)
  if (!poly || !currentImage.value) return
  try {
    const res = await erasePolygonRegion(poly, [rect],
      currentImage.value.width, currentImage.value.height, Number(box.class_id))
    const boxes: any[] = Array.isArray(res?.boxes) ? res.boxes : []
    pushUndo()
    const i = currentBoxes.value.indexOf(box)
    if (i < 0) return
    if (!boxes.length) {
      currentBoxes.value.splice(i, 1)
      if (cutoutBox.value === box) clearCutoutSession()
      if (selectedBoxIndex.value === i) selectedBoxIndex.value = -1
      showToast('这块挖没了，已删除该标注（Ctrl+Z 可撤销）', 'success')
      return
    }
    const b0 = boxes[0]
    Object.assign(currentBoxes.value[i], {
      x1: b0.x1, y1: b0.y1, x2: b0.x2, y2: b0.y2,
      polygon: b0.polygon && b0.polygon.length >= 3 ? b0.polygon : undefined
    })
    if (!b0.polygon) delete (currentBoxes.value[i] as any).polygon
    for (let k = 1; k < boxes.length; k++) {
      const nb = boxes[k]
      const box2: any = { class_id: nb.class_id, x1: nb.x1, y1: nb.y1, x2: nb.x2, y2: nb.y2 }
      if (nb.polygon && nb.polygon.length >= 3) box2.polygon = nb.polygon
      currentBoxes.value.push(box2)
    }
    showToast(boxes.length > 1
      ? `已挖掉那块，轮廓断成 ${boxes.length} 块（其余为新标注）`
      : '已挖掉那块（Ctrl+Z 可撤销）', 'success')
  } catch (e: any) {
    showToast('擦除失败: ' + (e.response?.data?.detail || e.message), 'error')
  }
}

// 把当前累积的点位打成一个任务丢进队列：点完就走，不等结果（这才敢连着点）
const queueCutout = () => {
  if (!currentTask.value || !currentImage.value || !cutoutPts.value.length) return
  if (drawClass.value < 0 || drawClass.value >= classes.value.length) {
    showToast('请先在下方「绘制类别」里选好类别，再点目标', 'error')
    return
  }
  cutoutQueue.value.push({
    taskId: currentTask.value,
    imageId: currentImage.value.image_id,
    // 微调已落地的目标时沿用那个目标原本的类别，别被「绘制类别」悄悄改掉
    classId: cutoutBox.value ? Number(cutoutBox.value.class_id) : Number(drawClass.value),
    pts: cutoutPts.value.map(p => ({ ...p })),
    box: cutoutBox.value,
    sid: cutoutSid.value,
    exitSeq: cutoutExitSeq
  })
  if (!cutoutBox.value) {
    const fg = cutoutPts.value.filter(pt => pt.label === 1).pop()
    cutoutLastNew = fg ? { imageId: currentImage.value.image_id, x: fg.x, y: fg.y } : null
  }
  pumpCutout()
}

// 结果落地：先分"还是眼前这张图"和"已经切走的图"两种走法。
// 后端可能回多个框（用户连点的点不连通 → 自动拆成多个闭合区域）：
// 第 1 框并入当前目标（或成为新目标），其余框作为新标注插入。
// keepIdx = 第 1 框实际吸收的会话点位下标（后端 point_indices），用于把已拆走的点移出会话。
const landCutout = async (job: CutoutJob, geoms: Partial<BBox>[], split: boolean, keepIdx?: number[]) => {
  const curId = currentImage.value ? currentImage.value.image_id : ''
  if (job.imageId === curId) {
    // 当前图：改内存里的标注列表，自动保存（1.5s 防抖）会带上去
    pushUndo()   // 人工点过目标 = 动过标注，pushUndo 内部会置「已修过」并起自动保存
    const i = job.box ? currentBoxes.value.indexOf(job.box) : -1
    if (i >= 0) {
      Object.assign(currentBoxes.value[i], geoms[0])
      showToast(split
        ? `这些点不在同一片区域：已按最近的点重抠该目标，断开处另起 ${geoms.length - 1} 个新目标`
        : '已按新的点位重抠该目标（不准可继续点，或 Ctrl+Z 回退）', 'success')
    } else {
      const box = { ...geoms[0] } as BBox
      currentBoxes.value.push(box)
      // 只有"这个目标还是当前会话的目标"时才把它认作微调对象，
      // 否则用户已经点向下一个目标了，抢这个位置会把后续点击带偏
      if (!job.box && job.sid === cutoutSid.value) {
        cutoutBox.value = box
        // 拆出去的点已另起新框，从会话中剔除（保留本次任务之后新补的点）
        if (split && keepIdx) {
          const keep = new Set(keepIdx)
          cutoutPts.value = cutoutPts.value.filter((_, k) => k >= job.pts.length || keep.has(k))
        }
      }
      showToast(split
        ? `检测到 ${geoms.length} 片不相连的区域，已各自成框（第 1 片是当前目标）`
        : '已抠出一个目标：继续点它是微调，点别处开新目标', 'success')
    }
    for (let k = 1; k < geoms.length; k++) {
      currentBoxes.value.push({ ...geoms[k] } as BBox)
    }
    persistPointDraft()   // 拆出去的点已从会话里剔除，同步刷新草稿
    redrawCanvas()
    return
  }
  // 已经切走的图：直接写那张图的标注文件，切回去就能看到，不用等用户回来补存
  const ann = await getImageAnnotation(job.taskId, job.imageId)
  const boxes: any[] = Array.isArray(ann && ann.boxes) ? ann.boxes : []
  const sameGeom = (x: any) =>
    Math.abs(x.x1 - job.box!.x1) < 1 && Math.abs(x.y1 - job.box!.y1) < 1 &&
    Math.abs(x.x2 - job.box!.x2) < 1 && Math.abs(x.y2 - job.box!.y2) < 1
  const i = job.box ? boxes.findIndex(sameGeom) : -1
  if (i >= 0) boxes[i] = { ...boxes[i], ...geoms[0] }
  else boxes.push({ ...geoms[0] })
  for (let k = 1; k < geoms.length; k++) boxes.push({ ...geoms[k] })
  await saveAnn(job.taskId, job.imageId, boxes as BBox[], !!ann && !!ann.ai_annotated,
                ann && ann.sample_type ? ann.sample_type : undefined, undefined, true)
  const ii = items.value.findIndex(it => it.image_id === job.imageId)
  if (ii >= 0) {
    items.value[ii].annotated = true
    items.value[ii].reviewed = true    // 后端 reviewed 是粘性标记，这里同步本地筛选计数
    items.value[ii].has_polygon = true
  }
  showToast('有张图抠好了（你已切走），切回那张图核对轮廓', 'success')
}

// 队列工人：一次只算一个。SAM 推理在后端是加锁串行的，CPU 上一次也只吃得下一张，
// 并发发请求只会让每一个都变慢，所以排队发、算完一个再发下一个。
const pumpCutout = async () => {
  if (cutoutRunning.value) return
  const job = cutoutQueue.value.shift()
  if (!job) return
  cutoutRunning.value = true
  const done = cutoutDone.value
  try {
    const res = await pointLabelImage(job.taskId, job.imageId, job.classId, job.pts)
    // 算完这几秒里，用户可能已经退出抠图模式，甚至已经离开了这个页面。
    // 这两种情况下结果一律作废 —— 修的就是"人退出了，框还在往外冒"。
    if (cutoutGone || job.exitSeq !== cutoutExitSeq) return
    const raw: any[] = res.boxes || []
    if (!raw.length) {
      cutoutCoverage.value = 1
      showToast('这个点没抠出目标，往水洼内部再点一下', 'error')
    } else if (job.box && cutoutQueue.value.some(j => j.box === job.box)) {
      // 同一个目标后面还排着更新的点位 → 这份中间结果不落地，免得轮廓来回跳、白费算力
      cutoutCoverage.value = Number(raw[0].coverage ?? 1)
    } else {
      // coverage / point_indices 是"点是否连成一片"的判定元数据，不能混进标注体系落库
      const geoms = raw.map(b => ({
        class_id: job.classId,
        x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2,
        ...(b.polygon && b.polygon.length >= 3 ? { polygon: b.polygon as number[][] } : {})
      }))
      const split = raw.length > 1 || Number(raw[0].coverage ?? 1) < 1
      cutoutCoverage.value = Number(raw[0].coverage ?? 1)
      await landCutout(job, geoms, split, Array.isArray(raw[0].point_indices) ? raw[0].point_indices : undefined)
      cutoutDone.value = done + 1
    }
  } catch (e: any) {
    if (!cutoutGone && job.exitSeq === cutoutExitSeq)
      showToast('抠图失败: ' + (e.response?.data?.detail || e.message), 'error')
  } finally {
    cutoutRunning.value = false
    if (cutoutQueue.value.length) pumpCutout()
  }
}

// 抠图模式的开/关（只做快捷键入口，工具条上不摆按钮）
const toggleCutoutMode = () => {
  if (cutoutMode.value) {
    exitCutoutMode()
    showToast('已退出抠图模式', 'info')
    redrawCanvas()
    return
  }
  if (!img.value) {
    showToast('先打开一张图再抠图', 'info')
    return
  }
  cutoutMode.value = true
  // 三种鼠标交互互斥：抠图占了左右键，描轮廓和框选区域都得让开
  polyDrawMode.value = false
  polyPending.value = null
  cancelPolygonDraft()
  interactiveMode.value = false
  interactiveRegion.value = null
  selectedBoxIndex.value = -1
  // 刷新恢复出来的未处理点位：进模式即接上，用户接着点就继续累积（连通判断在后端）
  const restored = pendingCutoutPts
  pendingCutoutPts = null
  clearCutoutSession()
  persistPointDraft()
  redrawCanvas()
  if (restored?.length) {
    cutoutPts.value = restored
    cutoutSid.value++
    persistPointDraft()
    showToast(`接上之前点的 ${restored.length} 个点：继续点水洼即可，系统会判断它们连不连成一片`, 'info', 8000)
    return
  }
  showToast('抠图模式：左键连续点水洼，相连的点自动连成一个闭合区域、不相连的自动拆成多个框；右键点抠多了的地方；回车/双击换下一个目标；Esc 退出', 'info')
}

// 当前目标抠完了：放掉点位，下一次左键开一个新目标
const finishCutoutTarget = () => {
  if (!cutoutMode.value) return
  if (!cutoutPts.value.length) {
    exitCutoutMode()
    showToast('已退出抠图模式', 'info')
    redrawCanvas()
    return
  }
  clearCutoutSession()
  showToast('这个目标已落定，直接点下一个', 'info')
  redrawCanvas()
}

// 点击左侧列表：选中该标注框并在画布高亮
const selectBoxFromPanel = (index: number) => {
  selectedBoxIndex.value = index
  redrawCanvas()
}

const drawBox = (box: BBox, index = -1) => {
  if (!ctx.value) return
  const selected = index === selectedBoxIndex.value
  // 框本体颜色 = 类别色（保持原有语义，不与审核结论混淆）
  const color = classColors.value[box.class_id] || 'lime'
  const a = box.audit
  const poly = polyOf(box)
  ctx.value.strokeStyle = selected ? 'orange' : color
  // 线宽按屏幕像素恒定：画布坐标系已乘 viewScale，不除的话放大后线条成倍变粗糊住边界
  ctx.value.lineWidth = (selected ? 3 : 2) / viewScale.value
  if (poly) {
    // 分割标注：沿轮廓描边 + 半透明填充，一眼能和普通检测框区分开
    ctx.value.beginPath()
    poly.forEach((pt, i) => (i === 0 ? ctx.value!.moveTo(pt[0], pt[1]) : ctx.value!.lineTo(pt[0], pt[1])))
    ctx.value.closePath()
    ctx.value.save()
    ctx.value.globalAlpha = maskAlpha.value
    ctx.value.fillStyle = selected ? 'orange' : color
    ctx.value.fill()
    ctx.value.restore()
    ctx.value.stroke()
  } else {
    ctx.value.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1)
  }
  ctx.value.fillStyle = selected ? 'orange' : color
  ctx.value.font = '14px Arial'
  let label = classes.value[box.class_id] || ''
  if (a?.tags?.length) label += ' ' + a.tags.join(' ')
  ctx.value.fillText(label, box.x1, Math.max(box.y1 - 5, 12))
  // 审核结论走独立通道：框右上角徽标（reject ✗红 / modify ~橙 / split⧉黄），keep 无标记；
  // 类别色与审核色不再混叠，只突出有异议的框
  if (a && a.action && a.action !== 'keep') {
    const sym = a.action === 'reject' ? '✗' : a.action === 'modify' ? '~' : '⧉'
    const bg = a.action === 'reject' ? '#e74c3c' : a.action === 'modify' ? '#f39c12' : '#f1c40f'
    const bs = 16
    ctx.value.fillStyle = bg
    ctx.value.fillRect(box.x2 - bs, box.y1, bs, bs)
    ctx.value.fillStyle = '#fff'
    ctx.value.font = 'bold 13px Arial'
    ctx.value.textAlign = 'center'
    ctx.value.fillText(sym, box.x2 - bs / 2, box.y1 + 12)
    ctx.value.textAlign = 'left'
  }
  if (selected) {
    const s = 8 / viewScale.value
    if (poly) {
      // 顶点=实心方块（可拖动），边中点=空心圆（点上去即插入新顶点）
      ctx.value.fillStyle = '#fff'
      ctx.value.strokeStyle = 'orange'
      ctx.value.lineWidth = 1.5 / viewScale.value
      poly.forEach(pt => ctx.value!.fillRect(pt[0] - s / 2, pt[1] - s / 2, s, s))
      const r = s * 0.45
      ctx.value.beginPath()
      poly.forEach((pt, i) => {
        const q = poly[(i + 1) % poly.length]
        ctx.value!.moveTo((pt[0] + q[0]) / 2 + r, (pt[1] + q[1]) / 2)
        ctx.value!.arc((pt[0] + q[0]) / 2, (pt[1] + q[1]) / 2, r, 0, Math.PI * 2)
      })
      ctx.value.stroke()
      // 轮廓中心的"米"字标记 = 提示这块可以整体拖动（放在质心，避免与顶点混淆）
      const cx = poly.reduce((sum, pt) => sum + pt[0], 0) / poly.length
      const cy = poly.reduce((sum, pt) => sum + pt[1], 0) / poly.length
      const k = s * 0.9
      ctx.value.beginPath()
      ctx.value.moveTo(cx - k, cy); ctx.value.lineTo(cx + k, cy)
      ctx.value.moveTo(cx, cy - k); ctx.value.lineTo(cx, cy + k)
      ctx.value.stroke()
    } else {
      const pts = [
        { x: box.x1, y: box.y1 }, { x: (box.x1 + box.x2) / 2, y: box.y1 }, { x: box.x2, y: box.y1 },
        { x: box.x2, y: (box.y1 + box.y2) / 2 }, { x: box.x2, y: box.y2 }, { x: (box.x1 + box.x2) / 2, y: box.y2 },
        { x: box.x1, y: box.y2 }, { x: box.x1, y: (box.y1 + box.y2) / 2 }
      ]
      ctx.value.fillStyle = '#fff'
      ctx.value.strokeStyle = 'orange'
      ctx.value.lineWidth = 1.5 / viewScale.value
      pts.forEach(pt => {
        ctx.value!.fillRect(pt.x - s / 2, pt.y - s / 2, s, s)
        ctx.value!.strokeRect(pt.x - s / 2, pt.y - s / 2, s, s)
      })
    }
  }
}

const redrawCanvas = () => {
  if (!ctx.value || !canvasRef.value || !img.value) return
  const c = canvasRef.value
  ctx.value.clearRect(0, 0, c.width, c.height)
  ctx.value.save()
  ctx.value.translate(viewOffset.value.x, viewOffset.value.y)
  ctx.value.scale(viewScale.value, viewScale.value)
  ctx.value.drawImage(img.value, 0, 0)
  // 边缘增强叠加层：青色 Sobel 边缘显形"水陆过渡带/纹理恢复线"（规范 v1.3 §1.6）
  if (edgeOverlayOn.value && edgeOverlayCanvas) {
    ctx.value.globalAlpha = 0.5
    ctx.value.drawImage(edgeOverlayCanvas, 0, 0)
    ctx.value.globalAlpha = 1
  }
  const cur = currentClass.value
  currentBoxes.value.forEach((box, idx) => {
    if (String(cur) === '-1' || box.class_id === Number(cur)) {
      drawBox(box, idx)
    }
  })
  // 千问审核漏标补框：蓝色虚线框（不参与框选中/移动，是"建议"，点面板补标才落框）
  ;(imageAudit.value?.missed_targets || []).forEach((mt: any) => {
    if (String(cur) !== '-1' && mt.class_id !== Number(cur)) return
    ctx.value!.strokeStyle = '#00bcd4'
    ctx.value!.lineWidth = 2 / viewScale.value
    ctx.value!.setLineDash([8, 6])
    ctx.value!.strokeRect(mt.x1, mt.y1, mt.x2 - mt.x1, mt.y2 - mt.y1)
    ctx.value!.setLineDash([])
    ctx.value!.fillStyle = '#00bcd4'
    ctx.value!.font = '14px Arial'
    ctx.value!.fillText('漏:' + (classes.value[mt.class_id] || mt.class_id), mt.x1, Math.max(mt.y1 - 5, 12))
  })
  // 正在画的新框
  if (isDrawing.value && editMode.value === 'draw') {
    ctx.value.strokeStyle = 'red'
    ctx.value.lineWidth = 2 / viewScale.value
    ctx.value.strokeRect(
      startPos.value.x, startPos.value.y,
      currentPos.value.x - startPos.value.x, currentPos.value.y - startPos.value.y
    )
  }
  // 正在绘制中的分割轮廓草稿：青色虚线 + 已落顶点，起点套圈提示"可闭合"
  if (polyDraft.value.length) {
    const pts = polyDraft.value
    ctx.value.strokeStyle = '#00e5ff'
    ctx.value.lineWidth = 2 / viewScale.value
    ctx.value.setLineDash([6, 4])
    ctx.value.beginPath()
    pts.forEach((pt, i) => (i === 0 ? ctx.value!.moveTo(pt[0], pt[1]) : ctx.value!.lineTo(pt[0], pt[1])))
    ctx.value.stroke()
    ctx.value.setLineDash([])
    const s = 8 / viewScale.value
    // 顶点用白底黑边：青色叠加层开着时纯青方块会淹在边缘线里，白底在任何背景上都醒目
    pts.forEach(pt => {
      ctx.value!.fillStyle = '#ffffff'
      ctx.value!.fillRect(pt[0] - s / 2, pt[1] - s / 2, s, s)
      ctx.value!.strokeStyle = '#003a44'
      ctx.value!.lineWidth = 1.5 / viewScale.value
      ctx.value!.strokeRect(pt[0] - s / 2, pt[1] - s / 2, s, s)
    })
    ctx.value.beginPath()
    ctx.value.arc(pts[0][0], pts[0][1], s, 0, Math.PI * 2)
    ctx.value.strokeStyle = pts.length >= 3 ? '#ff4d4f' : '#00e5ff'
    ctx.value.lineWidth = 2 / viewScale.value
    ctx.value.stroke()
  }
  // 抠图模式的提示点：绿点=要这块（前景），红点=抠掉这块（背景），让用户看清自己点了哪儿
  if (cutoutMode.value && cutoutPts.value.length) {
    const r = 6 / viewScale.value
    // 实时预览：前景点按点击顺序虚线串联，直观展示"这些点被当成同一片区域在判断"
    const fgs = cutoutPts.value.filter(pt => pt.label === 1)
    if (fgs.length >= 2) {
      ctx.value!.save()
      ctx.value!.strokeStyle = 'rgba(0, 230, 118, 0.85)'
      ctx.value!.lineWidth = 1.5 / viewScale.value
      ctx.value!.setLineDash([8 / viewScale.value, 6 / viewScale.value])
      ctx.value!.beginPath()
      fgs.forEach((pt, k) => (k === 0 ? ctx.value!.moveTo(pt.x, pt.y) : ctx.value!.lineTo(pt.x, pt.y)))
      ctx.value!.stroke()
      ctx.value!.restore()
    }
    cutoutPts.value.forEach(pt => {
      ctx.value!.beginPath()
      ctx.value!.arc(pt.x, pt.y, r, 0, Math.PI * 2)
      ctx.value!.fillStyle = pt.label === 1 ? '#00e676' : '#ff1744'
      ctx.value!.fill()
      ctx.value!.strokeStyle = '#ffffff'
      ctx.value!.lineWidth = 1.5 / viewScale.value
      ctx.value!.stroke()
    })
    // 最后一个前景点上标一圈脉冲环，提示"系统正在判断它与前面的点是否连成一片"
    const last = fgs[fgs.length - 1]
    if (last && cutoutBusy.value) {
      ctx.value!.beginPath()
      ctx.value!.arc(last.x, last.y, r * 2.2, 0, Math.PI * 2)
      ctx.value!.strokeStyle = '#00e5ff'
      ctx.value!.lineWidth = 2 / viewScale.value
      ctx.value!.stroke()
    }
  }
  // 右键拖拽中的擦除矩形：红色半透明，预览"这块要被挖掉"
  if (cutoutErase.value) {
    const er = cutoutErase.value
    ctx.value!.save()
    ctx.value!.fillStyle = 'rgba(255, 23, 68, 0.18)'
    ctx.value!.strokeStyle = '#ff1744'
    ctx.value!.lineWidth = 1.5 / viewScale.value
    ctx.value!.setLineDash([6 / viewScale.value, 4 / viewScale.value])
    ctx.value!.fillRect(Math.min(er.x1, er.x2), Math.min(er.y1, er.y2),
                        Math.abs(er.x2 - er.x1), Math.abs(er.y2 - er.y1))
    ctx.value!.strokeRect(Math.min(er.x1, er.x2), Math.min(er.y1, er.y2),
                          Math.abs(er.x2 - er.x1), Math.abs(er.y2 - er.y1))
    ctx.value!.restore()
  }
  ctx.value.restore()
}

const removeBox = (index: number) => {
  pushUndo()
  currentBoxes.value.splice(index, 1)
  if (selectedBoxIndex.value === index) selectedBoxIndex.value = -1
  redrawCanvas()
}

// 快捷键 C：一键清空当前图片上的所有标注框（走 pushUndo，可用 Ctrl+Z 整体撤销）
const clearAllBoxes = () => {
  if (!currentImage.value) return
  const n = currentBoxes.value.length
  if (!n) {
    showToast('这张图上没有标注框', 'info')
    return
  }
  pushUndo()
  currentBoxes.value = []
  selectedBoxIndex.value = -1
  redrawCanvas()
  showToast(`已清空 ${n} 个标注框，Ctrl+Z 可撤销`, 'info')
}

// 撤销/重做
const pushUndo = () => {
  // 走到这里的一定是人工编辑动作（画框/描轮廓/拖拐点/删框/改类等），标记本图已被人工动过
  manualTouched.value = true
  undoStack.value.push(JSON.parse(JSON.stringify(currentBoxes.value)))
  if (undoStack.value.length > 50) undoStack.value.shift()
  redoStack.value = []
  markDirty()
}

const undo = () => {
  if (!undoStack.value.length) return
  redoStack.value.push(JSON.parse(JSON.stringify(currentBoxes.value)))
  currentBoxes.value = undoStack.value.pop()!
  selectedBoxIndex.value = -1
  redrawCanvas()
  markDirty()
}

const redo = () => {
  if (!redoStack.value.length) return
  undoStack.value.push(JSON.parse(JSON.stringify(currentBoxes.value)))
  currentBoxes.value = redoStack.value.pop()!
  selectedBoxIndex.value = -1
  redrawCanvas()
  markDirty()
}

// 修改标注框类别（支持撤销）
const onBoxClassChange = (e: Event, idx: number) => {
  const old = currentBoxes.value[idx].class_id
  const val = Number((e.target as HTMLSelectElement).value)
  if (val === old) return
  pushUndo()
  currentBoxes.value[idx].class_id = val
  redrawCanvas()
}

// 类别颜色：按黄金角在 HSL 色环上均匀分配，肉眼可区分
const assignClassColors = (count: number) => {
  const colors: string[] = []
  for (let i = 0; i < count; i++) {
    colors.push(`hsl(${(i * 137.508) % 360}, 70%, 50%)`)
  }
  return colors
}

// 切换"当前类别"时重绘画布，实现按类别过滤显示框
watch(currentClass, () => {
  redrawCanvas()
})

// 类别列表变化时重新生成颜色
watch(classes, (val) => {
  classColors.value = assignClassColors(val.length)
}, { immediate: true })

// 一键清空当前图片的所有标注（针对误把置信度调成0导致标注过多的情况）
const clearCurrentBoxes = async () => {
  if (!currentBoxes.value.length) return
  if (!(await showConfirm(`确定清空当前图片的 ${currentBoxes.value.length} 个标注框吗？\n\n清空后需点击"保存"才会写入文件。`))) return
  pushUndo()
  currentBoxes.value = []
  redrawCanvas()
}

// 删除当前图片（连同标注 JSON 与同名标签文件，从数据集移除，不可恢复）
const deleteCurrentImage = async () => {
  if (!currentTask.value || !currentImage.value) return
  deletingImage.value = true
  try {
    const res = await deleteTaskImage(currentTask.value, currentImage.value.image_id)
    const removedId = res.image_id
    const idx = items.value.findIndex(it => it.image_id === removedId)
    items.value = items.value.filter(it => it.image_id !== removedId)
    if (items.value.length === 0) {
      currentImage.value = null
      currentBoxes.value = []
      currentSampleType.value = ''
      redrawCanvas()
    } else {
      selectImage(Math.min(Math.max(idx, 0), items.value.length - 1))
    }
    saveAnnotateState()
    showToast(`已删除图片 ${removedId}，当前图片总数：${res.image_count}`, 'success')
  } catch (error: any) {
    showToast('删除图片失败: ' + (error.response?.data?.detail || error.message), 'error')
  } finally {
    deletingImage.value = false
  }
}

// ===== AI 预标注配置 =====
const loadSamConfigAndModels = async () => {
  try {
    const cfg = await getSamConfig()
    samConfig.value = cfg
    // 页面打开时按已保存的形态对齐绘制工具：上次选的是「轮廓·分割」就直接进入描轮廓
    polyDrawMode.value = samOn.value
    if (typeof cfg.conf === 'number') aiConf.value = cfg.conf
    if (typeof cfg.iou === 'number') aiIou.value = cfg.iou
  } catch (e: any) {
    console.error('加载 SAM 配置失败:', e)
  }
}

const openModelCfg = async () => {
  try {
    const [cfg, models, trained] = await Promise.all([
      getSamConfig(), getSamModels(), getSamTrainedVersions(currentTask.value || '').catch(() => null),
    ])
    availableModels.value = models
    trainedVersions.value = (trained && trained.versions) || []
    const dets = (cfg.detectors && cfg.detectors.length)
      ? cfg.detectors
      : ((cfg.detector && cfg.detector !== 'none') ? [cfg.detector] : [])
    modelCfg.value.detectors = dets
    modelCfg.value.detector_weights = cfg.detector_weights || 'yolov8s-world.pt'
    // 本库已训模型：没选过或选的那个版本已不存在时，默认帮用户挑第一个（本库优先）
    const savedTid = cfg.trained_model_id || ''
    modelCfg.value.trained_model_id = trainedVersions.value.some((v: any) => v.model_id === savedTid)
      ? savedTid
      : (trainedVersions.value[0]?.model_id || '')
    modelCfg.value.conf = typeof cfg.conf === 'number' ? cfg.conf : 0.25
    modelCfg.value.iou = typeof cfg.iou === 'number' ? cfg.iou : 0.4
    // 分割形态两个开关直接读自 samConfig（与工具条同源），弹窗里改勾选即时反映到工具条
    samConfig.value = cfg
    // 千问是否启用 = 上方「检测引擎」里有没有勾千问 VL（原本还要再勾一次开关，容易漏）
    modelCfg.value.qwen_enabled = dets.includes('qwen_vl')
    modelCfg.value.qwen_backend = cfg.qwen_backend || 'ollama'
    modelCfg.value.qwen_endpoint = cfg.qwen_endpoint || 'http://localhost:11434'
    modelCfg.value.qwen_model = cfg.qwen_model || 'qwen3-vl:4b-instruct'
    modelCfg.value.qwen_api_key = cfg.qwen_api_key || ''
    modelCfg.value.qwen_timeout = cfg.qwen_timeout ?? 90
    modelCfg.value.qwen_mock = !!cfg.qwen_mock
    modelCfg.value.qwen_keep_alive = cfg.qwen_keep_alive || '5m'
    modelCfg.value.qwen_num_parallel = typeof cfg.qwen_num_parallel === 'number' ? cfg.qwen_num_parallel : 1
    modelCfg.value.qwen_image_max_side = typeof cfg.qwen_image_max_side === 'number' ? cfg.qwen_image_max_side : 1024
    modelCfg.value.qwen_audit_enabled = cfg.qwen_audit_enabled !== false
    modelCfg.value.qwen_audit_batch = cfg.qwen_audit_batch === true
    showModelCfg.value = true
  } catch (e: any) {
    if (!e?.__toastShown) showToast('读取 AI 标注配置失败: ' + (e.response?.data?.detail || e.message), 'error', 4000)
  }
}

// 取消/关闭弹窗：形态两个开关是即时生效的本地状态，这里回读后端，未保存的勾选不落到工具条
const closeModelCfg = async () => {
  showModelCfg.value = false
  try { samConfig.value = await getSamConfig() } catch { /* 回读失败不影响关闭 */ }
}

const saveModelCfg = async () => {
  try {
    if (!modelCfg.value.detectors.length) {
      showToast('请至少勾选一个检测引擎', 'info', 4000)
      return
    }
    // 勾了「本库已训模型」却没选版本 = 这个引擎永远出不了框，保存前先拦住
    if (modelCfg.value.detectors.includes('trained_model') && !modelCfg.value.trained_model_id) {
      showToast('请为「本库已训模型」选择一个模型版本（列表为空说明这个库还没训练过）', 'info', 4000)
      return
    }
    await updateSamConfig({
      detectors: [...modelCfg.value.detectors],
      detector: modelCfg.value.detectors[0],   // 兼容旧字段（批量兜底重试等仍引用）
      detector_weights: modelCfg.value.detector_weights,
      trained_model_id: modelCfg.value.trained_model_id || '',
      conf: modelCfg.value.conf,
      iou: modelCfg.value.iou,
      // 分割轮廓开关：单图 AI 与批量各一个（关掉即只出方框，不跑 SAM 所以更快）
      // 取自 samConfig（与工具条同一份数据），不会把工具条刚切好的形态覆盖回去
      sam_enabled: samEnabledModel.value,
      batch_sam_enabled: batchSamModel.value,
      // 勾了引擎即启用千问，未勾引擎自然关闭（与弹窗里显示的勾选框一致）
      qwen_enabled: modelCfg.value.detectors.includes('qwen_vl'),
      qwen_backend: modelCfg.value.qwen_backend,
      qwen_endpoint: modelCfg.value.qwen_endpoint,
      qwen_model: modelCfg.value.qwen_model,
      qwen_api_key: modelCfg.value.qwen_api_key,
      qwen_timeout: modelCfg.value.qwen_timeout,
      qwen_mock: !!modelCfg.value.qwen_mock,
      qwen_keep_alive: modelCfg.value.qwen_keep_alive,
      qwen_num_parallel: modelCfg.value.qwen_num_parallel,
      qwen_image_max_side: modelCfg.value.qwen_image_max_side,
      qwen_audit_enabled: !!modelCfg.value.qwen_audit_enabled,
      qwen_audit_batch: !!modelCfg.value.qwen_audit_batch,
    })
    // 同步页面阈值并刷新本地配置
    aiConf.value = modelCfg.value.conf
    aiIou.value = modelCfg.value.iou
    samConfig.value = await getSamConfig()
    showModelCfg.value = false
    showToast('AI 标注模型配置已保存并生效', 'success')
  } catch (e: any) {
    if (!e?.__toastShown) showToast('保存 AI 标注配置失败: ' + (e.response?.data?.detail || e.message), 'error', 4000)
  }
}

// 计算两个框的 IoU（交并比），用于去重
const iou = (a: any, b: any) => {
  const x1 = Math.max(a.x1, b.x1)
  const y1 = Math.max(a.y1, b.y1)
  const x2 = Math.min(a.x2, b.x2)
  const y2 = Math.min(a.y2, b.y2)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
  const union = areaA + areaB - inter
  return union > 0 ? inter / union : 0
}

const runAutoLabel = async () => {
  if (!currentTask.value || !currentImage.value) return
  // 若正在标注中，再次点击则询问是否取消
  if (aiLabeling.value) {
    if (!(await showConfirm('是否取消当前 AI 标注？'))) return
    aiAbortCtrl.value?.abort()
    aiLabeling.value = false
    aiAbortCtrl.value = null
    return
  }
  if (!classes.value.length) {
    showToast('请先设置类别', 'info', 4000)
    return
  }
  aiLabeling.value = true
  aiAbortCtrl.value = new AbortController()
  try {
    const result = await autoLabelImage(
      currentTask.value,
      currentImage.value.image_id,
      classes.value,
      aiConf.value,
      buildPrompts(classes.value.map((c: any) => (typeof c === 'string' ? c : c.name))),
      aiAbortCtrl.value.signal
    )
    if (result.error) {
      showToast('AI 预标注失败: ' + result.error, 'error', 4000)
      return
    }
    const auditMap = new Map<number, any>()
    ;(result.audit?.box_reviews || []).forEach((r: any) => auditMap.set(r.box_id, r))
    const aiBoxes = (result.boxes || []).map((b: any) => ({
      class_id: b.class_id,
      x1: b.x1,
      y1: b.y1,
      x2: b.x2,
      y2: b.y2,
      source: b.source || 'ai',
      // SAM 分割轮廓（有则透传，无则仍是检测框）
      ...(b.polygon && b.polygon.length >= 3 ? { polygon: b.polygon as number[][] } : {}),
      box_id: b.box_id != null ? b.box_id : undefined
    }))
    // 按 IoU 去重：与已有任意类别框重叠 >=0.5 的视为同一目标，避免重复标注（含跨类别）
    const newAiBoxes = aiBoxes.filter(
      (ab: any) => !currentBoxes.value.some((cb: any) => iou(cb, ab) >= 0.5)
    )
    // 把对应审核结论（box_id 映射回预标注下标）粘到新增框上，供画布/列表渲染；box_id 不落盘
    newAiBoxes.forEach((ab: any) => {
      if (ab.box_id != null) {
        const r = auditMap.get(ab.box_id)
        if (r) ab.audit = { ...r }
      }
      delete ab.box_id
    })
    currentBoxes.value.push(...newAiBoxes)
    if (newAiBoxes.length > 0) {
      currentImageHasAi.value = true
      markDirty()
    }
    // 持有审核汇总（统计 + 漏标补框），面板与画布虚线框据此渲染
    imageAudit.value = result.audit && result.audit.ok ? result.audit : null
    redrawCanvas()
    const st = imageAudit.value?.stats
    if (st) {
      const j = (x: any) => x || 0
      const parts: string[] = []
      if (j(st.keep)) parts.push(`通过 ${j(st.keep)}`)
      if (j(st.modify)) parts.push(`待修正 ${j(st.modify)}`)
      if (j(st.reject)) parts.push(`建议删除 ${j(st.reject)}`)
      if (j(st.split)) parts.push(`粘连 ${j(st.split)}`)
      if (j(st.missed)) parts.push(`漏标 ${j(st.missed)}`)
      showToast(`L2 审核：${parts.join(' · ') || '全部通过'}`, parts.length ? 'info' : 'success')
    }
    if (!aiBoxes.length) {
      showToast('未检测到目标，可调低置信度重试', 'info', 4000)
    } else if (!newAiBoxes.length) {
      showToast(`检测到 ${aiBoxes.length} 个目标，但均与已有标注重复，未新增`, 'info', 4000)
    }
  } catch (e: any) {
    if (e?.code === 'ERR_CANCELED' || e?.message === 'canceled') {
      showToast('已取消 AI 标注', 'info', 4000)
    } else {
      if (!e?.__toastShown) showToast('AI 预标注失败: ' + (e.response?.data?.detail || e.message), 'error', 4000)
    }
  } finally {
    aiLabeling.value = false
    aiAbortCtrl.value = null
  }
}

// ===== 千问 L2 审核（设计文档 §6.2.2/§6.3）=====
// 审核结论按 box_id（= 提交审核时的数组下标）回写；reject 红框 / modify 橙框 / keep 绿框 / split 黄框，
// missed_targets 蓝虚线框可一键补标；label_errors 一键修正类别，tags（#occluded/#blur/#compliance_issue）随框显示。
const applyAudit = (audit: any) => {
  imageAudit.value = audit && audit.ok ? audit : null
  if (audit && !audit.ok) {
    showToast('AI 审核失败: ' + (audit.error || '未知异常'), 'error', 4000)
    return
  }
  const map = new Map<number, any>()
  ;(audit?.box_reviews || []).forEach((r: any) => map.set(r.box_id, r))
  currentBoxes.value.forEach((b: any, idx: number) => {
    const r = map.get(idx)
    if (r) b.audit = { ...r }
    else delete b.audit
  })
  redrawCanvas()
  const st = imageAudit.value?.stats
  if (st) {
    const j = (x: any) => x || 0
    const parts: string[] = []
    if (j(st.keep)) parts.push(`通过 ${j(st.keep)}`)
    if (j(st.modify)) parts.push(`待修正 ${j(st.modify)}`)
    if (j(st.reject)) parts.push(`建议删除 ${j(st.reject)}`)
    if (j(st.split)) parts.push(`粘连 ${j(st.split)}`)
    if (j(st.missed)) parts.push(`漏标 ${j(st.missed)}`)
    showToast(`L2 审核：${parts.join(' · ') || '全部通过'}`, parts.length ? 'info' : 'success')
  }
}

// 手动审核当前图（⋯更多 → AI 审核当前图）：把画布上当前框发给千问审核，未保存也可审
const auditCurrentImage = async () => {
  if (!currentTask.value || !currentImage.value) return
  if (!currentBoxes.value.length) {
    showToast('当前图片没有标注框，无法审核', 'info', 4000)
    return
  }
  auditing.value = true
  try {
    const res = await auditImage(
      currentTask.value,
      currentImage.value.image_id,
      currentBoxes.value.map((b: any) => ({
        class_id: b.class_id, x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2
      }))
    )
    if (res.message && !res.audit) {
      showToast(res.message, 'info', 4000)
      return
    }
    applyAudit(res.audit)
  } catch (e: any) {
    if (!e?.__toastShown) showToast('AI 审核失败: ' + (e.response?.data?.detail || e.message), 'error', 4000)
  } finally {
    auditing.value = false
  }
}

// 问题图片检索：列出任务中带审核异议（reject/modify/split）的图片，点击直达
const openAuditIssues = async () => {
  if (!currentTask.value) return
  auditIssuesOpen.value = true
  await refreshAuditIssues()
}

const refreshAuditIssues = async () => {
  if (!currentTask.value) return
  loadingIssueList.value = true
  try {
    const res = await listAuditIssues(currentTask.value)
    auditIssues.value = res?.issues || []
  } catch (e: any) {
    if (!e?.__toastShown) showToast('加载审核问题列表失败: ' + (e.response?.data?.detail || e.message), 'error', 4000)
  } finally {
    loadingIssueList.value = false
  }
}

const gotoIssueImage = (imageId: string) => {
  const idx = items.value.findIndex((it: any) => it.image_id === imageId)
  if (idx < 0) return
  auditIssuesOpen.value = false
  selectImage(idx)
}

// 审核建议的漏标补框：一键添加为新标注框
const addMissedTarget = (mt: any) => {
  pushUndo()
  currentBoxes.value.push({
    class_id: mt.class_id, x1: mt.x1, y1: mt.y1, x2: mt.x2, y2: mt.y2,
    source: 'ai_audit'
  })
  currentImageHasAi.value = true
  markDirty()
  redrawCanvas()
  if (imageAudit.value) {
    imageAudit.value.missed_targets = (imageAudit.value.missed_targets || []).filter(
      (m: any) => m !== mt
    )
    if (imageAudit.value.stats) imageAudit.value.stats.missed = imageAudit.value.missed_targets.length
  }
  showToast(`已补标：${classes.value[mt.class_id] || mt.class_id}`, 'success')
}

// 一键采纳"建议类别"（label_errors 修正）
const fixBoxClass = (idx: number) => {
  const b = currentBoxes.value[idx]
  if (!b?.audit || b.audit.suggested_class_id == null) return
  pushUndo()
  b.class_id = b.audit.suggested_class_id
  delete b.audit
  redrawCanvas()
  showToast(`已修正类别为：${classes.value[b.class_id] || b.class_id}`, 'success')
}

// 一键删除被审核拒绝的框（reject / label_error / merged-reject）
const auditRejectCount = computed(() =>
  currentBoxes.value.filter((b) => b.audit && b.audit.action === 'reject').length
)

const removeRejectedBoxes = async () => {
  const n = auditRejectCount.value
  if (!n) return
  if (!(await showConfirm(`${n} 个审核拒绝的框将被删除，是否继续？`))) return
  pushUndo()
  currentBoxes.value = currentBoxes.value.filter((b: any) => !(b.audit && b.audit.action === 'reject'))
  redrawCanvas()
  markDirty()
  showToast(`已删除 ${n} 个审核拒绝的框`, 'success')
}

// 清除审核标记：保留框，去掉颜色/标签/漏标虚线
const clearAudit = () => {
  currentBoxes.value.forEach((b: any) => delete b.audit)
  imageAudit.value = null
  redrawCanvas()
  showToast('已清除 AI 审核标记（标注框保留）', 'info')
}

const runBatchLabel = async () => {
  if (!currentTask.value) return
  if (!classes.value.length) {
    showToast('请先设置类别', 'info', 4000)
    return
  }
  if (!(await showConfirm(`确认对 ${items.value.length} 张图片执行批量 AI 预标注？已标注的类别不会被覆盖，仅会为尚未标注的新类别追加检测框。`))) return
  try {
    const result = await startBatchLabel(
      currentTask.value,
      classes.value,
      aiConf.value,
      buildPrompts(classes.value.map((c: any) => (typeof c === 'string' ? c : c.name)))
    )
    batchId.value = result.batch_id
    batchProgress.value = { status: 'running', total: result.total, done: 0, boxes_written: 0, current_image: '' }
    pollBatch()
  } catch (e: any) {
    if (!e?.__toastShown) showToast('启动批量预标注失败: ' + (e.response?.data?.detail || e.message), 'error', 4000)
  }
}

const pollBatch = async () => {
  if (!batchId.value) return
  let res: any
  try {
    res = await getBatchProgress(batchId.value)
  } catch {
    batchTimer.value = setTimeout(pollBatch, 2000)
    return
  }
  if (!res) {
    // 批量任务不存在（例如后端重启导致内存任务丢失），已标注的图片已保存，不丢数据
    batchId.value = null
    batchProgress.value = null
    saveAnnotateState()
    showToast('批量预标注任务已中断（可能因后端重启），之前已完成图片的标注已保存。', 'info', 4000)
    await loadTaskItems()
    return
  }
  batchProgress.value = res
  // 批量增量刷新：只更新左侧列表的标注状态徽标/计数，**绝不重载当前画布**
  // （避免打断正在进行的标注）。当前图被批量写入后，切走再切回时
  // selectImage 会自然加载最新标注，无需主动刷新。
  if (Array.isArray(res.annotated_images) && res.annotated_images.length) {
    const written = new Set<string>(res.annotated_images)
    for (const it of items.value) {
      if (written.has(it.image_id) && !it.annotated) {
        it.annotated = true
        it.ai_annotated = true
      }
    }
  }
  if (res.status === 'done') {
    batchId.value = null
    saveAnnotateState()
    const annotated = res.summary?.annotated ?? res.boxes_written ?? 0
    const retried = res.retried || 0
    const skipped = (res.skipped_candidates || []).length
    let msg = `批量预标注完成！共标注 ${annotated} 张图片`
    if (retried > 0) msg += `\n其中 ${retried} 张通过降阈值/双引擎兜底补标成功`
    if (skipped > 0) msg += `\n仍有 ${skipped} 张两次兜底都未检出目标（可能需要人工复核或调整类别）`
    const au = res.audit
    if (au) {
      msg += `\nL2 审核: ${au.checked} 图 / ${au.boxes} 框`
      msg += ` · 通过 ${au.stats.keep} · 待修正 ${au.stats.modify} · 建议删除 ${au.stats.reject} · 粘连 ${au.stats.split} · 漏标 ${au.stats.missed}`
      if (au.errors > 0) msg += `（${au.errors} 张审核失败）`
    }
    showToast(msg, 'info', 5000)
    await loadTaskItems()
  } else if (res.status === 'cancelled') {
    batchId.value = null
    saveAnnotateState()
    const done = res.done || 0
    const annotated = res.annotated_images ?? done
    showToast(`批量预标注已取消，已标注 ${annotated} 张图片`, 'info', 4000)
    await loadTaskItems()
  } else if (res.status === 'error') {
    batchId.value = null
    saveAnnotateState()
    showToast('批量预标注出错: ' + (res.error || '未知错误'), 'error', 4000)
  } else {
    saveAnnotateState()
    batchTimer.value = setTimeout(pollBatch, 1500)
  }
}

// 一键清除本任务所有 AI 预标注（误标过多时清理重标）
const clearAllAiLabels = async () => {
  if (!currentTask.value) return
  if (!(await showConfirm(`确定清除本任务全部 ${counts.value.ai} 张 AI 预标注图片吗？\n将删除这些图片的标注（含之后人工修改），不可恢复。`))) return
  try {
    const res = await clearAiAnnotations(currentTask.value)
    showToast(`已清除 ${res.removed ?? 0} 张图片的 AI 标注，可重新标注`, 'info', 4000)
    await loadTaskItems()
    if (currentImage.value?.annotated) {
      await loadExistingAnnotation()
    } else {
      currentBoxes.value = []
      currentImageHasAi.value = false
      loadImageToCanvas()
    }
  } catch (e: any) {
    if (!e?.__toastShown) showToast('清除失败: ' + (e.response?.data?.detail || e.message), 'error', 4000)
  }
}

// 一键清除本数据集全部标注（AI 标注 + 人工标注一视同仁）
// 后端在清空前会把 annotations/*.json、labels/*.txt、meta.json 打包备份，误清可解压回退；
// 已封板 / 已入桶 / 已归档的数据集会被后端拒绝（400）。
const clearAllLabels = async () => {
  if (!currentTask.value || clearingAll.value) return
  const ok = await showConfirm(
    `将清空本数据集全部 ${counts.value.annotated} 张已标注图片的标注框：\n` +
      'AI 预标注与人工标注一视同仁，全部删除。\n\n' +
      '· 清空前后端会自动打包备份，误清可解压回退\n' +
      '· 样本类型标记（正常/困难/背景）保留，只清框\n' +
      '· 派生的 YOLO txt 标签同步删除，避免两份真相\n' +
      '· 数据集的审核通过状态会回退，需重新复核\n' +
      '· 已封板 / 已入桶 / 已归档的数据集不允许执行\n\n' +
      '是否继续？',
    '清除全部标注'
  )
  if (!ok) return
  const word = await showPrompt('此操作影响整个数据集，请输入「删除」以最终确认：', '', '二次确认')
  if (word === null) return
  if (word.trim() !== '删除') {
    await showAlert('输入不匹配，已取消清除全部标注', '已取消')
    return
  }
  clearingAll.value = true
  try {
    const res = await clearAllAnnotations(currentTask.value)
    await showAlert(
      `已清除 ${res.cleared_images ?? 0} 张图片、${res.cleared_boxes ?? 0} 个标注框` +
        (res.removed_label_files ? `，并删除 ${res.removed_label_files} 个派生 YOLO 标签文件` : '') +
        `\n\n备份文件：${res.backup || '（无可备份内容，未生成）'}`,
      '清除完成'
    )
    await loadTaskItems()
    currentBoxes.value = []
    currentImageHasAi.value = false
    dirty.value = false
    manualTouched.value = false
    loadImageToCanvas()
  } catch (e: any) {
    await showAlert('清除失败: ' + (e.response?.data?.detail || e.message), '错误')
  } finally {
    clearingAll.value = false
  }
}

// 一键清理本任务全部图片的高度重合标注框（历史遗留重叠框统一根治，对所有检测模型生效）
const cleanOverlaps = async () => {
  if (!currentTask.value) return
  if (!(await showConfirm(
    '将对本任务全部图片执行重叠框清理：\n' +
    '1. 跨类别高度重合（IoU≥0.5）只保留一个\n' +
    '2. 同类高度重叠/紧邻的框合并\n\n' +
    '此操作会覆盖图片标注，不可恢复，是否继续？'
  ))) return
  try {
    const res = await cleanTaskOverlaps(currentTask.value)
    showToast(`清理完成：${res.images_cleaned ?? 0} 张图片、${res.boxes_removed ?? 0} 个重叠框被清理`, 'info', 4000)
    await loadTaskItems()
    if (currentImage.value?.annotated) {
      await loadExistingAnnotation()
    } else {
      currentBoxes.value = []
      currentImageHasAi.value = false
      loadImageToCanvas()
    }
  } catch (e: any) {
    if (!e?.__toastShown) showToast('清理失败: ' + (e.response?.data?.detail || e.message), 'error', 4000)
  }
}

const stopBatch = async () => {
  if (!batchId.value) return
  try {
    await stopBatchLabel(batchId.value)
    // 提示已标注的图片数量后清理本地状态
    const done = batchProgress.value?.done || 0
    const annotated = batchProgress.value?.annotated_images ?? done
    batchId.value = null
    batchProgress.value = null
    saveAnnotateState()
    showToast(`批量预标注已取消，已标注 ${annotated} 张图片`, 'info', 4000)
    await loadTaskItems()
  } catch (e: any) {
    // 任务不存在（例如后端重启后内存任务清空），清理本地状态并友好提示
    if (e.response?.status === 404) {
      batchId.value = null
      batchProgress.value = null
      saveAnnotateState()
      showToast('预标注任务已不存在（可能已重启），已停止', 'info', 4000)
      return
    }
    if (!e?.__toastShown) showToast('停止失败: ' + (e.response?.data?.detail || e.message), 'error', 4000)
  }
}

onUnmounted(() => {
  if (batchTimer.value) {
    clearTimeout(batchTimer.value)
  }
})

// 保存标注（silent=true 用于自动保存，不弹提示打扰）
const persistAnnotation = async (silent = false): Promise<boolean> => {
  if (!currentTask.value || !currentImage.value) return false
  if (savingAnnotation.value) return false
  savingAnnotation.value = true
  try {
    const aiFlag = currentImageHasAi.value || !!currentImage.value.ai_annotated
    // 人工保存：显式样本类型时附带 manual 判据（人工确认），未选择时留空由后端自动判定
    const manualType = currentSampleType.value || undefined
    // 人工框标 source=manual（AI 框保存 source=ai_x），后端据此识别"人工补框→困难样本"
    // audit 落盘（供"问题图片检索"），box_id 是预标注临时索引不落盘
    const payloadBoxes = currentBoxes.value.map((b: any) => {
      const { box_id, ...rest } = b
      return rest.source ? rest : { ...rest, source: 'manual' }
    })
    await saveAnn(currentTask.value, currentImage.value.image_id, payloadBoxes, aiFlag, manualType, manualType ? 'manual' : undefined, manualTouched.value)
    items.value[currentIndex.value].annotated = true
    if (aiFlag) {
      items.value[currentIndex.value].ai_annotated = true
    }
    if (manualType) {
      items.value[currentIndex.value].sample_type = manualType
    }
    // 即时刷新"轮廓质量"筛选所需的本地标记（免得重进页面才更新列表）
    items.value[currentIndex.value].reviewed = items.value[currentIndex.value].reviewed || manualTouched.value
    items.value[currentIndex.value].has_polygon = currentBoxes.value.some((b: any) => (b.polygon?.length || 0) >= 3)
    dirty.value = false
    if (!silent) showToast('保存成功!', 'success')
    return true
  } catch (error: any) {
    if (!silent && !error?.__toastShown) showToast('保存失败: ' + (error.response?.data?.detail || error.message), 'error', 4000)
    return false
  } finally {
    savingAnnotation.value = false
  }
}

const saveAnnotation = () => persistAnnotation(false)

// 标记脏状态并防抖自动保存（所有编辑操作经 pushUndo 触发）
const markDirty = () => {
  dirty.value = true
  clearTimeout(autoSaveTimer.value)
  autoSaveTimer.value = setTimeout(() => persistAnnotation(true), 1500)
}

// 在线调置信度/IoU 阈值：保存到后端配置，AI 标注与交互式标注立即生效
const onThreshChange = async () => {
  try {
    await updateSamConfig({ conf: aiConf.value, iou: aiIou.value })
    showToast(`已保存阈值：置信度 ${aiConf.value}，IoU ${aiIou.value}`, 'info', 4000)
  } catch (e: any) {
    if (!e?.__toastShown) showToast('保存阈值失败: ' + (e.response?.data?.detail || e.message), 'error', 4000)
  }
}

// 导入 YOLO 标注（zip 内含 labels/*.txt）
const onImportYolo = async (e: any) => {
  const file = e.target.files?.[0]
  e.target.value = ''
  if (!file || !currentTask.value) return
  if (!(await showConfirm(
    `导入 ${file.name}？\n\n` +
    `将按文件名匹配任务图片，写入 YOLO 标注。\n` +
    `同名图片的现有标注会被导入内容覆盖，是否继续？`
  ))) return
  importing.value = true
  try {
    const res = await importYoloLabels(currentTask.value, file)
    const msg = `导入完成：${res.imported ?? 0} 张图片`
    const skipped = (res.skipped || []).slice(0, 5).join('、')
    showToast(skipped ? `${msg}\n跳过未匹配: ${skipped}${res.skipped.length > 5 ? ' ...' : ''}` : msg, 'info', 4000)
    await loadTaskItems()
    if (currentImage.value?.annotated) {
      await loadExistingAnnotation()
    }
  } catch (err: any) {
    if (!err?.__toastShown) showToast('导入失败: ' + (err.response?.data?.detail || err.message), 'error', 4000)
  } finally {
    importing.value = false
  }
}

const exportTask = async () => {
  if (!currentTask.value) return
  if (!(await showConfirm(
    `确认导出已标注数据集？\n\n` +
    `将把标注结果导出为 YOLO 格式（labels/*.txt）写入数据集目录。\n` +
    `数据集划分将在「封板」时进行。`
  ))) return

  exporting.value = true
  try {
    const result = await exportAnnotations(currentTask.value)
    // 后端返回字段是 exported_count（原来读的 total 并不存在，弹框里一直显示 undefined）
    const cnt = result.exported_count ?? result.total ?? 0
    // segment_boxes = 本次导出带分割轮廓的框数；为 0 说明标注里根本没有轮廓，
    // 导出的标签行只有 5 列，只能训检测、不能训分割，这里直接告诉用户
    const segNote = result.segment_boxes
      ? `\n其中 ${result.segment_boxes} 条含分割轮廓（标签行大于 5 列，可训练分割模型）`
      : '\n注意：本次导出的标签全部只有 5 列（纯方框检测标注），无法用于训练分割模型'
    showToast(
      `导出成功！共 ${cnt} 张已标注图片${segNote}\n` +
      `YOLO 标注已写入数据集目录，划分随封板进行。`,
      'info',
      5000
    )
  } catch (error: any) {
    if (!error?.__toastShown) showToast('导出失败: ' + (error.response?.data?.detail || error.message), 'error', 4000)
  } finally {
    exporting.value = false
  }
}

// 单独提取"已修过"的图：后端只打包人工修过且有标注框的图片 + YOLO 分割标签，zip 直接下载
const exportReviewedPackage = async () => {
  if (!currentTask.value) return
  exportingReviewed.value = true
  try {
    const blob = await exportReviewedZip(currentTask.value)
    // 文件名前置数据集业务名，方便下载后一眼认出是哪个库；查不到元数据时退化为原名
    const meta = taskDatasetMeta.value
    const title = meta ? String(dsTitle(meta) || '').trim() : ''
    const prefix = title ? `${title}_` : ''
    downloadFile(blob, `${prefix}${currentTask.value}_reviewed.zip`)
    showToast(
      `已开始下载 zip：包含您修过的 ${counts.value.reviewed} 张图片。\n` +
      `解压后：images/ 是图片，labels/ 是 YOLO 标签（含分割轮廓），还有 classes.txt 和 data.yaml，可直接用于训练。`,
      'info',
      5000
    )
  } catch (error: any) {
    // 出错时后端返回的也是 blob（JSON 内容），需要读出来才能看到具体原因
    let msg = ''
    const data = error?.response?.data
    if (data instanceof Blob) {
      try {
        msg = JSON.parse(await data.text())?.detail || ''
      } catch {
        msg = ''
      }
    } else {
      msg = data?.detail || error?.message || ''
    }
    if (!error?.__toastShown) showToast('导出失败: ' + (msg || '未知错误'), 'error', 4000)
  } finally {
    exportingReviewed.value = false
  }
}

// 翻页走可见集合（类别过滤 ∩ 搜索），不再跑到列表里根本看不到的图上；
// 索引换算交给 selectImageByItem，它负责映射回全量 items 的下标。
const prevImage = () => {
  const p = visibleIdx.value
  if (p > 0) selectImageByItem(visibleItems.value[p - 1])
}

const nextImage = () => {
  const p = visibleIdx.value
  if (p >= 0 && p < visibleItems.value.length - 1) selectImageByItem(visibleItems.value[p + 1])
}

const handleKeydown = (e: KeyboardEvent) => {
  const tag = (e.target as HTMLElement)?.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
  if (!currentTask.value || loadingAnnotation.value) return

  // Ctrl/Cmd 组合键
  if (e.ctrlKey || e.metaKey) {
    const k = e.key.toLowerCase()
    if (k === 's') {
      e.preventDefault()
      saveAnnotation()
    } else if (k === 'z') {
      e.preventDefault()
      if (e.shiftKey) redo()
      else undo()
    } else if (k === 'y') {
      e.preventDefault()
      redo()
    }
    return
  }

  // 空格：按住平移（只在画布未被聚焦交互时启用）
  if (e.code === 'Space') {
    e.preventDefault()
    spaceDown.value = true
    return
  }

  switch (e.key) {
    case 'Escape':
      // 分层退出：先丢弃绘制中的轮廓 → 再退抠图模式 → 退多边形模式 → 最后退交互式标注
      if (cutoutMode.value) {
        exitCutoutMode()
        redrawCanvas()
      } else if (polyPending.value) {
        polyPending.value = null
      } else if (polyDraft.value.length) {
        cancelPolygonDraft()
      } else if (polyDrawMode.value) {
        polyDrawMode.value = false
        redrawCanvas()
      } else if (interactiveMode.value) {
        interactiveMode.value = false
        interactiveRegion.value = null
        redrawCanvas()
      }
      break
    case 'Enter':
      // 回车：抠图模式下落定当前目标，描轮廓模式下闭合当前轮廓
      if (cutoutMode.value) {
        e.preventDefault()
        finishCutoutTarget()
      } else if (polyDrawMode.value && polyDraft.value.length) {
        e.preventDefault()
        finishPolygonDraft()
      }
      break
    case 'f':
    case 'F':
      // 快捷键 F：恢复画面大小（适配窗口）。描轮廓/抠图时左键被占用，复位视图走键盘
      e.preventDefault()
      fitView()
      break
    case 'e':
    case 'E':
      // 快捷键 E：边缘增强叠加层开关（规范 v1.3 §1.6 判边辅助）
      edgeOverlayOn.value = !edgeOverlayOn.value
      onEdgeOverlayToggle()
      break
    case 's':
    case 'S':
      // 快捷键 S：顶点吸附开关（C 类不可见边界按物理线索画时关掉）
      snapToEdgeOn.value = !snapToEdgeOn.value
      onSnapToggle()
      break
    case 'g':
    case 'G':
      // 快捷键 G：进出智能抠图（点哪儿抠哪儿，一键出分割标注）
      e.preventDefault()
      toggleCutoutMode()
      break
    case 'ArrowRight':
    case 'd':
    case 'D':
      e.preventDefault()
      nextImage()
      break
    case 'ArrowLeft':
    case 'a':
    case 'A':
      e.preventDefault()
      prevImage()
      break
    case 'c':
    case 'C':
      // 快捷键 C：一键清空当前图片的所有标注框（不删图片本身，Ctrl+Z 可撤销）
      e.preventDefault()
      clearAllBoxes()
      break
    case 'x':
    case 'X':
      // 快捷键 X：直接删除当前图片（不弹确认；批量标注中也可删）
      e.preventDefault()
      deleteCurrentImage()
      break
    case 'Delete':
    case 'Backspace':
      if (selectedBoxIndex.value >= 0) {
        e.preventDefault()
        removeBox(selectedBoxIndex.value)
      }
      break
  }
}

// 松开空格等按键时复位状态
const handleKeyup = (e: KeyboardEvent) => {
  if (e.code === 'Space') {
    spaceDown.value = false
    if (isPanning.value) {
      isPanning.value = false
      redrawCanvas()
    }
  }
}

// 点击页面其他区域时关闭筛选下拉菜单
const handleDocClick = () => {
  filterOpen.value = false
}

onMounted(() => {
  loadDatasets()
  loadSamConfigAndModels()
  window.addEventListener('keydown', handleKeydown)
  window.addEventListener('keyup', handleKeyup)
  window.addEventListener('resize', fitView)
  document.addEventListener('click', handleDocClick)
  handleUrlEntry()
  restoreAnnotateState()
})

// 雪球闭环（1.8）：?dataset_id=xxx 直达 —— 优先复用该数据集已有标注任务，
// 无任务则自动创建并进入（对应数据集页"准备"后自动建任务 + "去标注"按钮跳转）
const handleUrlEntry = async () => {
  const params = new URLSearchParams(window.location.search)
  const dsId = params.get('dataset_id')
  if (!dsId) return
  await loadDatasets()
  const ds = datasets.value.find((d: any) => d.dataset_id === dsId)
  if (!ds) {
    showToast('未找到数据集 ' + dsId + '，请先上传并准备', 'info', 4000)
    return
  }
  newTask.value.datasetId = dsId
  syncClassesFromDataset(dsId)
  try {
    const found = await findAnnotationTask(dsId, 'v1')
    if (found?.task) {
      currentTask.value = found.task.task_id
      if ((found.task.classes || []).length) {
        classes.value = found.task.classes
        classesInput.value = found.task.classes.join(', ')
      }
      if (found.task.imported_annotations) importedCount.value = found.task.imported_annotations
      await loadTaskItems()
      // 直达成功（自动任务已存在），清掉 URL 参数避免刷新时重复触发
      window.history.replaceState({}, '', '/annotate')
    } else {
      await createTask()  // 无任务 → 自动创建（后端 prepare 已建则由 find 命中，此处兜底）
    }
  } catch (e: any) {
    if (!e?.__toastShown) showToast('进入标注任务失败: ' + (e.response?.data?.detail || e.message), 'error', 4000)
  }
}

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
  window.removeEventListener('keyup', handleKeyup)
  window.removeEventListener('resize', fitView)
  document.removeEventListener('click', handleDocClick)
  // 离开页面：抠图队列清空，在途结果也不再落地，免得往已经卸载的画布上写
  cutoutGone = true
  cutoutQueue.value = []
  // 离页时若有未保存改动，尝试立即保存
  if (dirty.value) persistAnnotation(true)
  saveAnnotateState()
})
</script>

<style scoped>
/* 千问 VL 配置弹窗 */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.modal-box {
  background: #fff;
  border-radius: 10px;
  padding: 20px 24px;
  width: 460px;
  max-width: 92vw;
  max-height: 86vh;
  overflow: auto;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}
.modal-box h3 {
  margin: 0 0 8px;
  font-size: 16px;
}
.modal-box .row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 10px 0;
  font-size: 13px;
  color: #333;
}
/* 批次E：仅收紧"勾选框行"。不能改 .modal-box .row 本体 —— 同一弹窗里
   「接入方式/服务地址/模型名/API Key/超时」等行内是 .w-input{flex:1}，
   若整条 .row 收成 fit-content，这些输入框会塌成默认窄宽（回归）。 */
.modal-box .row:has(> input[type="checkbox"]) {
  width: fit-content;
}
.modal-box .hint {
  font-size: 12px;
  color: #7f8c8d;
  line-height: 1.5;
  margin: 4px 0 12px;
  background: #f8f9fa;
  border-radius: 6px;
  padding: 8px 10px;
}
.modal-box .w-input {
  flex: 1;
  min-width: 0;
  padding: 6px 8px;
  border: 1px solid #d0d7de;
  border-radius: 6px;
  font-size: 13px;
}
.modal-box .w-input.short {
  flex: 0 0 80px;
}
.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 16px;
}
.modal-footer button {
  padding: 7px 18px;
  border-radius: 6px;
  border: 1px solid #d0d7de;
  background: #fff;
  cursor: pointer;
}
.modal-footer button.primary {
  background: #3d7eff;
  color: #fff;
  border-color: #3d7eff;
}
.info-box {
  background: #fff3cd;
  border: 1px solid #ffc107;
  padding: 0.75rem;
  border-radius: 4px;
  margin-bottom: 1rem;
  color: #856404;
}

.form-group small {
  display: block;
  margin-top: 0.25rem;
  color: #7f8c8d;
  font-size: 0.875rem;
}

.annotate-workspace {
  display: grid;
  grid-template-columns: 200px 1fr 250px;
  gap: 1rem;
  margin-top: 1rem;
}

.sidebar, .canvas-area, .annotations-panel {
  background: white;
  border-radius: 8px;
  padding: 1rem;
}

/* grid 列允许收缩：防止大尺寸画布把中间列撑到原图宽度，顶开右侧面板 */
.canvas-area, .annotations-panel {
  min-width: 0;
}

/* 右侧标注面板：贴住视口顶部，长列表滚动时操作区始终可见 */
.annotations-panel {
  position: sticky;
  top: 1rem;
  align-self: start;
  display: flex;
  flex-direction: column;
}

.ann-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.ann-panel-header h3 {
  margin: 0;
}

.more-wrap {
  position: relative;
}

.more-btn {
  width: auto !important;
  margin-bottom: 0 !important;
  padding: 0.3rem 0.6rem;
  font-size: 0.8rem;
  background: #eef0f3;
  border: 1px solid #d0d3d8;
  border-radius: 4px;
  color: #333;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  white-space: nowrap;
}

.more-btn:hover,
.more-btn.active {
  background: #e2e5ea;
}

.menu-backdrop {
  position: fixed;
  inset: 0;
  z-index: 24;
}

.more-menu {
  position: absolute;
  right: 0;
  top: calc(100% + 4px);
  width: 210px;
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
  z-index: 25;
  padding: 0.4rem;
}

.more-menu button,
.more-menu .import-label {
  width: 100%;
  margin-bottom: 0.25rem;
  font-size: 0.85rem;
}

.more-menu button.primary {
  background: #1c64f2;
  color: #fff;
  border: none;
}
.more-menu button.primary:hover { background: #1a56d8; }
.more-menu .menu-hint {
  padding: 0.35rem 0.5rem;
  margin-bottom: 0.25rem;
  font-size: 0.78rem;
  color: #92400e;
  background: #fffbeb;
  border: 1px dashed #f5c26b;
  border-radius: 4px;
  text-align: center;
}

.more-menu .import-label {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.25rem;
  padding: 0.5rem;
  border: 1px solid #d0d3d8;
  border-radius: 4px;
  background: #eef0f3;
  color: #333;
  cursor: pointer;
  text-align: center;
  box-sizing: border-box;
}

.more-menu .import-label:hover {
  background: #e2e5ea;
}

/* 单图操作固定栏：标签列表滚动时保持可见 */
.ann-fixed-actions {
  margin-top: 0.5rem;
  border-top: 1px solid #eee;
  padding-top: 0.5rem;
}

.ann-action-row {
  display: flex;
  gap: 0.4rem;
  margin-top: 0.5rem;
}

.ann-action-row button {
  flex: 1;
  margin-bottom: 0 !important;
  min-width: 0;
}

.ann-action-row .primary {
  background: #3498db;
  border-color: #3498db;
  color: #fff;
}

.ann-action-row .primary:hover:not(:disabled) {
  background: #2980b9;
}

.image-list {
  max-height: 400px;
  overflow-y: auto;
  margin: 1rem 0;
}

/* 「图片列表 (N)」可双击跳转到指定张数 */
.list-title {
  cursor: pointer;
  user-select: none;
}

/* 图片名搜索框 */
.image-search {
  position: relative;
  display: flex;
  align-items: center;
  margin-bottom: 0.4rem;
}

.image-search-input {
  width: 100%;
  padding: 0.4rem 1.7rem 0.4rem 0.6rem;
  font-size: 0.85rem;
  border-radius: 6px;
  border: 1px solid #ddd;
  background: #fff;
  color: #333;
  box-sizing: border-box;
}

.image-search-input:focus {
  outline: none;
  border-color: #3498db;
}

.image-search-clear {
  position: absolute;
  right: 0.35rem;
  padding: 0.1rem 0.3rem;
  font-size: 0.8rem;
  line-height: 1;
  color: #95a5a6;
  background: none;
  border: none;
  cursor: pointer;
}

.image-search-clear:hover {
  color: #2c3e50;
}

/* 筛选/搜索生效时，标题里的全量数用灰字弱化 */
.list-count-total {
  color: #95a5a6;
  font-weight: normal;
  font-size: 0.85em;
}

.nav-pos {
  padding: 0.4rem 0 0.1rem;
  text-align: center;
  font-size: 0.78rem;
  color: #7f8c8d;
  font-variant-numeric: tabular-nums;
}

.image-filter {
  position: relative;
  margin: 0.75rem 0 0.25rem;
}

.filter-dropdown {
  position: relative;
}

.filter-trigger {
  width: 100%;
  padding: 0.4rem 0.6rem;
  font-size: 0.85rem;
  border-radius: 6px;
  border: 1px solid #ddd;
  background: #fff;
  color: #333;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.filter-trigger:hover {
  border-color: #3498db;
}

.filter-caret {
  font-size: 0.7rem;
  color: #95a5a6;
}

.filter-menu {
  position: absolute;
  top: calc(100% + 4px);
  left: 0;
  right: 0;
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
  z-index: 20;
  padding: 0.25rem 0;
}

.filter-menu-item {
  padding: 0.4rem 0.75rem;
  font-size: 0.85rem;
  color: #333;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.filter-menu-item:hover {
  background: #f0f6ff;
}

.filter-menu-item.active {
  background: #3498db;
  color: #fff;
}

.filter-menu-item.sub {
  padding-left: 1.5rem;
  color: #555;
}

.filter-menu-group {
  border-top: 1px solid #f0f0f0;
  border-bottom: 1px solid #f0f0f0;
  margin: 0.15rem 0;
}

.filter-menu-item.group-header {
  font-weight: 600;
}

/* "轮廓质量"这类纯分组标题：不可点、无悬停态 */
.filter-menu-item.group-header.label-only {
  cursor: default;
  pointer-events: none;
  color: #888;
}

.filter-count {
  font-size: 0.7rem;
  background: rgba(0, 0, 0, 0.08);
  border-radius: 8px;
  padding: 0 0.35rem;
  line-height: 1.3;
  flex-shrink: 0;
}

.filter-menu-item.active .filter-count {
  background: rgba(255, 255, 255, 0.25);
}

.empty-filter {
  padding: 1rem;
  color: #95a5a6;
  text-align: center;
  font-size: 0.85rem;
}

.image-item {
  padding: 0.5rem;
  cursor: pointer;
  border-radius: 4px;
  margin-bottom: 0.25rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.4rem;
}

.image-item-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.image-item-badges {
  display: flex;
  gap: 0.25rem;
  flex-shrink: 0;
}

.img-badge {
  font-size: 0.7rem;
  line-height: 1;
  padding: 0.2rem 0.35rem;
  border-radius: 4px;
  font-weight: 600;
}

.img-badge.ok {
  color: #2ecc71;
  background: rgba(46, 204, 113, 0.12);
}

.img-badge.ai {
  color: #fff;
  background: #e67e22;
}

.img-badge.hard {
  color: #e74c3c;
  background: rgba(231, 76, 60, 0.15);
}

.img-badge.bg {
  color: #7f8c8d;
  background: rgba(127, 140, 141, 0.15);
}

.sample-type-group {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  margin-right: 0.5rem;
}

.sample-type-label {
  font-size: 0.78rem;
  color: #7f8c8d;
}

.sample-type-btn {
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
  border: 1px solid #bdc3c7;
  border-radius: 4px;
  background: #fff;
  color: #555;
  cursor: pointer;
}

.sample-type-btn.active {
  border-color: #3498db;
  background: #3498db;
  color: #fff;
}

.image-item:hover {
  background: #ecf0f1;
}

.image-item.active {
  background: #3498db;
  color: white;
}

.image-item.active .img-badge.ok {
  color: #fff;
  background: transparent;
}

.image-item.active .img-badge.ai {
  color: #fff;
  background: rgba(255, 255, 255, 0.35);
}

.image-item.annotated {
  border-left: 3px solid #2ecc71;
}

.canvas-container {
  position: relative;
  border: 2px solid #ddd;
  overflow: hidden;
  /* 确定高度（而非依赖画布内容自撑）：大图不再把布局顶开，小图也有稳定的适配区域
     底部让出工具条那一行的高度，整块区域总高保持不变 */
  height: min(600px, calc(100vh - 376px));
  min-height: 240px;
  cursor: crosshair;
  /* 图片小于显示框时居中；图片过大溢出时 margin 自动归零，从左上滚动不受影响 */
  display: flex;
}

/* 缩放/绘制工具条：画布下方独占一行，不再浮在画面上挡住要标注的区域 */
.view-toolbar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: #33363d;
  color: #fff;
  border: 2px solid #ddd;
  border-top: none;
  border-radius: 0 0 6px 6px;
  padding: 5px 8px;
  font-size: 0.8rem;
  /* 窗口太窄时折行，而不是把右边的按钮裁掉（这一行在画面外，变高也不挡图） */
  flex-wrap: wrap;
}

/* 分割填充遮盖度滑块 */
.alpha-ctl {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: #ccc;
  font-size: 0.75rem;
  white-space: nowrap;
}
.alpha-ctl input[type="range"] {
  width: 72px;
  accent-color: #4a9eff;
  cursor: pointer;
}

.view-hint {
  color: #ccc;
  font-size: 0.75rem;
  margin-right: 0.5rem;
  /* 独占剩余宽度并把按钮顶到右侧；窗口太窄时自己省略号，不撑破这一行 */
  flex: 1;
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.view-toolbar .small {
  min-width: 26px;
  height: 26px;
  padding: 0;
  line-height: 1;
}

/* 工具条开关按钮（边缘/吸附/轮廓绘制）激活态：红色 = 功能已开启，一眼可辨 */
.view-toolbar .small.active {
  background: #dc3545;
  border-color: #dc3545;
  color: #fff;
  width: auto;
  padding: 0 0.5rem;
}

.view-hint.poly-hint {
  color: #17a2b8;
}

/* 左侧标注框条目标签前的颜色块 */
.box-color {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 3px;
  margin-right: 0.5rem;
  flex: none;
}

/* 交互式标注提示条与激活按钮 */
.interactive-hint {
  font-size: 0.8rem;
  color: #e67e22;
  background: #fdf3e7;
  border: 1px solid #f5c6a0;
  border-radius: 4px;
  padding: 0.4rem 0.6rem;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.annotations-panel button.active {
  background: #e67e22;
  color: #fff;
  border-color: #e67e22;
}

/* 导入 YOLO 标注按钮（label 包裹 file input） */
.import-label {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.45rem 0.8rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: #fff;
  cursor: pointer;
  font-size: 0.9rem;
}

.import-label:hover {
  border-color: #42b983;
  color: #42b983;
}

.import-input {
  display: none;
}

.ai-btn.small {
  padding: 0.3rem 0.6rem;
  font-size: 0.8rem;
}

.img-nav-btn {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 44px;
  height: 44px;
  padding: 0;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.45);
  color: #fff;
  font-size: 28px;
  font-weight: bold;
  line-height: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  z-index: 5;
  border: none;
  transition: background 0.2s;
}

.img-nav-btn.prev {
  left: 8px;
}

.img-nav-btn.next {
  right: 8px;
}

.img-nav-btn:hover:not(:disabled) {
  background: rgba(52, 152, 219, 0.9);
}

.img-nav-btn:disabled {
  background: rgba(0, 0, 0, 0.25);
  opacity: 0.4;
  cursor: not-allowed;
}

canvas {
  display: block;
  margin: auto;
  /* 显示尺寸上限=容器，避免任何时刻画布缓冲过大把布局撑开 */
  max-width: 100%;
  max-height: 100%;
  min-width: 0;
  min-height: 0;
}

.controls {
  margin-top: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.controls select {
  flex: 1;
}

.box-list {
  margin: 1rem 0;
  max-height: 400px;
  overflow-y: auto;
}

.box-item {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-bottom: 0.5rem;
  cursor: pointer;
  transition: border-color 0.15s, background-color 0.15s;
}

.box-item:hover {
  border-color: #aaa;
}

.box-item-selected {
  border-color: orange;
  background: rgba(255, 165, 0, 0.12);
}

.box-class-select {
  flex: 1;
  margin-right: 0.5rem;
  padding: 0.25rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  min-width: 0;
}

.box-class-name {
  font-size: 0.85rem;
  color: #333;
  margin-right: 0.5rem;
  white-space: normal;
  word-break: break-all;
  flex: 1 1 100%;
  min-width: 0;
}

.box-empty {
  color: #999;
  font-size: 0.85rem;
  padding: 0.5rem 0;
  text-align: center;
}

/* ===== 千问 L2 审核（设计文档 §6.3：绿/橙/红/黄框 + 蓝色虚线漏标 + 一键操作） ===== */
.audit-chip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-right: 0.35rem;
  padding: 0.05rem 0.4rem;
  border-radius: 10px;
  font-size: 0.7rem;
  font-weight: 700;
  min-width: 1.1rem;
  color: #fff;
}
.audit-chip.a-keep { background: #2ecc71; }
.audit-chip.a-modify { background: #f39c12; }
.audit-chip.a-reject { background: #e74c3c; }
.audit-chip.a-split, .audit-chip.a-merged { background: #f1c40f; color: #333; }
/* 分割标注标记：列表里一眼看出这个框带轮廓（可微调顶点） */
.seg-chip {
  display: inline-flex;
  align-items: center;
  margin-right: 0.35rem;
  padding: 0.05rem 0.35rem;
  border-radius: 3px;
  font-size: 0.68rem;
  color: #fff;
  background: #17a2b8;
  flex: none;
}
.fix-btn {
  flex: initial !important;
  width: auto !important;
  margin-right: 0.4rem !important;
  margin-bottom: 0 !important;
  padding: 0.15rem 0.4rem !important;
  font-size: 0.72rem !important;
  background: #fff3cd;
  color: #856404;
  border: 1px solid #ffc107;
  border-radius: 4px;
  cursor: pointer;
}
.fix-btn:hover { background: #ffe69c; }

.audit-summary {
  margin-top: 0.75rem;
  padding: 0.6rem;
  border: 1px solid #d1e7ff;
  border-radius: 6px;
  background: #f6faff;
  font-size: 0.8rem;
}
.audit-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  margin-bottom: 0.4rem;
}
.audit-title { font-weight: 600; color: #1c64f2; }
.audit-summary button { width: auto !important; margin-bottom: 0 !important; }
.audit-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 0.6rem;
  margin-bottom: 0.4rem;
}
.audit-stats span { font-size: 0.75rem; padding: 0.1rem 0.35rem; border-radius: 4px; }
.a-keep { background: #e6f9ef; color: #0e9f6e; }
.a-modify { background: #fef3e2; color: #b45309; }
.a-reject { background: #fde8e8; color: #e02424; }
.a-split { background: #fef9e7; color: #a16207; }
.a-missed { background: #e0f7fa; color: #0097a7; }
.audit-batch-line { margin-top: 6px; font-size: 12px; color: #666; line-height: 1.5; word-break: break-all; }
.audit-batch-line .a-keep, .audit-batch-line .a-modify, .audit-batch-line .a-reject,
.audit-batch-line .a-split, .audit-batch-line .a-missed { padding: 0 4px; border-radius: 3px; margin: 0 2px; }
.issue-mask { position: fixed; inset: 0; background: rgba(0, 0, 0, .45); z-index: 1000; display: flex; align-items: center; justify-content: center; }
.issue-dialog { background: #fff; border-radius: 10px; width: 580px; max-width: 92vw; max-height: 76vh; display: flex; flex-direction: column; box-shadow: 0 8px 30px rgba(0, 0, 0, .25); padding: 14px 16px; }
.issue-dialog-head { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 14px; }
.issue-hint { color: #999; font-size: 12px; flex: 1; }
.issue-loading, .issue-empty { padding: 24px; text-align: center; color: #999; font-size: 13px; }
.issue-list { list-style: none; margin: 0; padding: 0; overflow-y: auto; }
.issue-list li { display: flex; align-items: center; gap: 10px; padding: 8px 10px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.issue-list li:hover { background: #f0f7ff; }
.issue-img { flex: 1; font-family: monospace; word-break: break-all; }
.issue-badges em { font-style: normal; margin-left: 4px; padding: 1px 6px; border-radius: 4px; font-size: 12px; }
.a-miss { background: #e0f7fa; color: #0097a7; }
.audit-actions { margin-bottom: 0.4rem; }
.audit-missed-title { font-size: 0.75rem; font-weight: 600; color: #0097a7; margin-bottom: 0.3rem; }
.audit-mt-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.25rem 0;
  border-top: 1px dashed #c7e9ef;
}
.mt-info { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mt-info em { color: #999; font-style: normal; font-size: 0.72rem; }
.audit-summary .primary { background: #1c64f2; color: #fff; border: none; }
.audit-summary .danger { background: #e02424; color: #fff; border: none; }

.annotations-panel button {
  width: 100%;
  margin-bottom: 0.5rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.25rem;
}

.select-with-refresh {
  display: flex;
  gap: 0.5rem;
}

.select-with-refresh select {
  flex: 1;
}

.ds-selectbar {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 0.5rem;
}

.ds-pager {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
  font-size: 0.78rem;
  color: #5b6770;
}

.ds-pager-info {
  white-space: nowrap;
}

.ds-pager-nav {
  display: flex;
  gap: 0.4rem;
}

.ds-pager-btn {
  padding: 0.2rem 0.6rem;
  font-size: 0.78rem;
  color: #2c3e50;
  background: #fff;
  border: 1px solid #d0d3d8;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap;
}

.ds-pager-btn:hover:not(:disabled) {
  border-color: #8e44ad;
  color: #8e44ad;
}

.ds-pager-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 全局 select 是 width:100%，分页条里要收成内容宽度 */
.ds-pager-size {
  width: auto;
  padding: 0.2rem 0.4rem;
  font-size: 0.78rem;
}

.ds-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 0.6rem;
  margin-bottom: 0.5rem;
}

.ds-card {
  border: 1px solid #d0d3d8;
  border-radius: 6px;
  padding: 0.6rem 0.75rem;
  cursor: pointer;
  background: #fff;
  transition: all 0.15s ease;
}

.ds-card:hover {
  border-color: #8e44ad;
  box-shadow: 0 2px 6px rgba(142, 68, 173, 0.15);
}

.ds-card.selected {
  border-color: #8e44ad;
  background: #f7f0fb;
  box-shadow: 0 0 0 2px rgba(142, 68, 173, 0.25);
}

.ds-card.disabled {
  opacity: 0.55;
  cursor: not-allowed;
  background: #f8f9fa;
}

.ds-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
}

.ds-name {
  font-weight: bold;
  font-size: 0.9rem;
  word-break: break-all;
}

.status-badge {
  font-size: 0.7rem;
  padding: 0.1rem 0.4rem;
  border-radius: 8px;
  white-space: nowrap;
  color: #fff;
}

.status-prepared {
  background: #2ecc71;
}

.status-uploaded {
  background: #f39c12;
}

.status-unknown {
  background: #95a5a6;
}

.ds-card-info {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  font-size: 0.8rem;
  color: #7f8c8d;
}

.ds-card-warn {
  margin-top: 0.35rem;
  font-size: 0.75rem;
  color: #b9770e;
}

.empty-state {
  grid-column: 1 / -1;
  text-align: center;
  color: #95a5a6;
  padding: 1rem;
  border: 1px dashed #d0d3d8;
  border-radius: 6px;
}

.refresh-btn {
  padding: 0.5rem 1rem;
  white-space: nowrap;
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}

.import-info {
  background: #d4edda;
  border: 1px solid #c3e6cb;
  color: #155724;
  padding: 0.5rem;
  border-radius: 4px;
  font-size: 0.875rem;
  margin-bottom: 0.5rem;
}

.loading-state {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 2rem;
  color: #7f8c8d;
}

.loading-indicator {
  font-size: 0.875rem;
  color: #7f8c8d;
  font-weight: normal;
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}

button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.25rem;
}

button.small {
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
}

.ai-btn {
  background: #8e44ad;
  color: white;
  border: none;
  padding: 0.4rem 0.75rem;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap;
}
.ai-btn:hover:not(:disabled) {
  background: #732d91;
}

/* 批量预标注：与单图 AI 预标注区分的高亮操作，显眼醒目 */
.ai-btn.batch-btn {
  background: #0e9f6e;
  box-shadow: 0 2px 4px rgba(14, 159, 110, 0.35);
}

.ai-btn.batch-btn:hover:not(:disabled) {
  background: #0b7a54;
}

.ai-toolbar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.6rem 0.75rem;
  margin-bottom: 0.75rem;
  background: linear-gradient(90deg, #f3e8f9, #eef3fb);
  border: 1px solid #d9c4e8;
  border-radius: 6px;
  flex-wrap: wrap;
}

.ai-btn {
  background: #8e44ad;
  color: white;
  border: none;
  padding: 0.5rem 1.1rem;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap;
  font-weight: bold;
  font-size: 0.95rem;
  box-shadow: 0 2px 4px rgba(142, 68, 173, 0.3);
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}

.ai-conf-label {
  margin-left: 0.25rem;
  color: #555;
}

.conf-input {
  width: 70px;
  padding: 0.35rem;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.ai-classes-label {
  display: inline-block;
  background: #eef0f3;
  border: 1px dashed #b8bcc4;
  border-radius: 999px;
  padding: 0.25rem 0.7rem;
  color: #666;
  font-size: 0.85rem;
  cursor: help;
  white-space: nowrap;
}

/* 一级入口：当前 AI 识别模型（点开=标注模型设置弹窗），与紫色 AI 按钮区分 */
.ai-btn.model-btn {
  background: #2d6cdf;
  box-shadow: 0 2px 4px rgba(45, 108, 223, 0.3);
  font-size: 0.85rem;
  font-weight: normal;
  padding: 0.4rem 0.8rem;
}

.ai-btn.model-btn:hover:not(:disabled) {
  background: #1f52ad;
}

/* 手动描轮廓：默认低调描边，进入绘制后与「轮廓·分割」同色，一眼看出当前画法 */
.ai-btn.poly-toggle {
  background: #fff;
  color: #8e44ad;
  border: 1px solid #c9a8dd;
  box-shadow: none;
  font-size: 0.85rem;
  padding: 0.4rem 0.8rem;
}

.ai-btn.poly-toggle.active {
  background: #8e44ad;
  color: #fff;
  border-color: #8e44ad;
  font-weight: bold;
}

/* 结果形态分段开关：轮廓（多边形/分割） / 方框（四边形/检测） */
.shape-seg {
  display: inline-flex;
  align-items: center;
  border: 1px solid #b9a3c9;
  border-radius: 999px;
  overflow: hidden;
  background: #fff;
}

.shape-seg .seg-btn {
  border: none;
  background: transparent;
  color: #6b5b78;
  padding: 0.32rem 0.7rem;
  font-size: 0.85rem;
  cursor: pointer;
  white-space: nowrap;
}

.shape-seg .seg-btn.active {
  background: #8e44ad;
  color: #fff;
  font-weight: bold;
}

.shape-seg .seg-btn:disabled {
  opacity: 0.6;
  cursor: default;
}

/* 批量预标注走的是独立开关，明示当前出轮廓还是出方框，点击进设置 */
button.shape-note {
  border: 1px dashed #b8bcc4;
  background: #eef0f3;
  color: #666;
  border-radius: 999px;
  padding: 0.25rem 0.65rem;
  font-size: 0.82rem;
  cursor: pointer;
  white-space: nowrap;
}

button.shape-note:hover {
  border-color: #8e44ad;
  color: #8e44ad;
}

.ai-params {
  position: relative;
}

.ai-params summary {
  cursor: pointer;
  color: #8e44ad;
  font-size: 0.9rem;
  user-select: none;
  list-style: none;
}

.ai-params summary::-webkit-details-marker {
  display: none;
}

.ai-params-body {
  position: absolute;
  top: calc(100% + 4px);
  left: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  background: white;
  border: 1px solid #ddd;
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
  white-space: nowrap;
}

.batch-progress {
  margin: 0.75rem 0;
  padding: 0.5rem;
  background: #eef3fb;
  border-radius: 4px;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  font-size: 0.875rem;
}

.progress-bar {
  height: 8px;
  background: #dfe6ed;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #3498db;
  transition: width 0.3s ease;
}

/* ===== 新类别主动提醒（困难/空白样本弹窗 + AI 识别弹窗） ===== */
.btn-soft {
  background: #8e44ad;
  color: #fff;
  border: none;
  padding: 0.4rem 0.9rem;
  border-radius: 4px;
  cursor: pointer;
  font-weight: bold;
}

.btn-ghost {
  background: transparent;
  border: 1px solid #d0d3d8;
  color: #666;
  padding: 0.4rem 0.9rem;
  border-radius: 4px;
  cursor: pointer;
}

.model-cfg-modal .cfg-block {
  margin-bottom: 0.6rem;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid #f0f0f0;
}

.model-cfg-modal .cfg-title {
  display: block;
  font-weight: bold;
  color: #444;
  margin-bottom: 0.35rem;
}

.model-cfg-modal .indent {
  margin-left: 1.3rem;
}

.model-cfg-modal .dim {
  font-style: normal;
  color: #999;
}

.model-cfg-modal select {
  padding: 0.35rem;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.suggest-modal .suggest-hint {
  color: #777;
  font-size: 0.88rem;
  margin-bottom: 0.8rem;
}

.suggest-modal .suggest-msg {
  color: #8e44ad;
  padding: 0.5rem 0;
  font-size: 0.92rem;
}

.suggest-modal .suggest-item {
  display: flex;
  align-items: flex-start;
  gap: 0.6rem;
  padding: 0.55rem 0.6rem;
  border: 1px solid #eee;
  border-radius: 6px;
  margin-bottom: 0.5rem;
  cursor: pointer;
}

.suggest-modal .suggest-item:hover {
  background: #f7f4fb;
}

.suggest-item-body {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  flex-wrap: wrap;
  flex: 1;
}

.suggest-item-body code {
  background: #f0ecf5;
  color: #7d3c98;
  border-radius: 3px;
  padding: 0.05rem 0.35rem;
  font-size: 0.82rem;
}

.suggest-desc {
  color: #777;
  font-size: 0.86rem;
}

.suggest-imgs {
  color: #e67e22;
  font-size: 0.82rem;
  margin-left: auto;
}

.suggest-actions {
  display: flex;
  gap: 0.6rem;
  margin-top: 0.8rem;
}
</style>

<template>
  <div class="folder-tree">
    <div v-for="node in nodes" :key="node.path" class="tree-node">
      <div
        class="tree-row"
        :class="{ 'is-file': node.type === 'file' }"
        :style="{ paddingLeft: (depth ?? 0) * 18 + 'px' }"
        :title="node.type === 'file' ? node.path : (['train','val','test','images','labels'].includes(node.name) ? '' : node.path)"
        @click="node.type === 'dir' ? toggleDir(node) : openFile(node)"
      >
        <span class="tree-caret">{{ caretText(node) }}</span>
        <span class="tree-icon">{{ iconText(node) }}</span>
        <span class="tree-name">{{ node.name }}</span>
        <span v-if="node.type === 'dir' && (node.file_count ?? node.children?.length ?? 0)" class="tree-count">{{ node.file_count ?? (node.children ?? []).length }}</span>
        <span v-if="node.type === 'file'" class="tree-size">{{ formatSize(node.size) }}</span>
        <span v-if="node.truncated" class="tree-truncated" title="条目过多已省略">…</span>
      </div>
      <template v-if="node.type === 'dir'">
        <FolderTree
          v-if="expanded.has(node.path)"
          :nodes="node.children ?? []"
          :depth="(depth ?? 0) + 1"
          :expanded="expanded"
          @toggle-dir="toggleDir"
          @open-file="openFile"
        />
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
interface TreeNode {
  name: string
  type: 'dir' | 'file'
  path: string
  ext?: string
  size?: number
  children?: TreeNode[]
  truncated?: boolean
  file_count?: number
}

const props = defineProps<{
  nodes: TreeNode[]
  depth?: number
  expanded: Set<string>
}>()

const emit = defineEmits<{
  (e: 'toggle-dir', node: TreeNode): void
  (e: 'open-file', node: TreeNode): void
}>()

const toggleDir = (node: TreeNode) => emit('toggle-dir', node)
const openFile = (node: TreeNode) => emit('open-file', node)

const isImage = (node: TreeNode) =>
  ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'].includes(node.ext || '')

const caretText = (node: TreeNode) => {
  if (node.type !== 'dir') return ''
  return props.expanded.has(node.path) ? '▾' : '▸'
}

const iconText = (node: TreeNode) => {
  if (node.type !== 'dir') return isImage(node) ? '🖼️' : '📄'
  return props.expanded.has(node.path) ? '📂' : '📁'
}

const formatSize = (size?: number) => {
  if (size == null) return ''
  if (size < 1024) return size + ' B'
  if (size < 1024 * 1024) return (size / 1024).toFixed(1) + ' KB'
  return (size / 1024 / 1024).toFixed(2) + ' MB'
}
</script>

<style scoped>
.folder-tree {
  font-size: 0.95rem;
}
.tree-row {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.4rem 0.5rem;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap;
  user-select: none;
}
.tree-row:hover {
  background: #f2f6fc;
}
.tree-row.is-file:hover {
  color: #2c6fbb;
}
.tree-caret {
  width: 1rem;
  color: #9aa0a6;
  flex-shrink: 0;
}
.tree-icon {
  flex-shrink: 0;
  font-size: 1.05rem;
}
.tree-name {
  color: #2c3e50;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tree-row.is-file .tree-name {
  color: #34495e;
}
.tree-count {
  color: #95a5a6;
  font-size: 0.8rem;
  background: #eef0f3;
  border-radius: 8px;
  padding: 0 0.45rem;
  flex-shrink: 0;
}
.tree-size {
  color: #b0b6bd;
  font-size: 0.8rem;
  margin-left: auto;
  flex-shrink: 0;
}
.tree-truncated {
  color: #e67e22;
  margin-left: auto;
}
</style>
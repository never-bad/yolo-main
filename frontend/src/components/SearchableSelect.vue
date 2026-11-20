<template>
  <div ref="root" class="searchable-select">
    <input
      ref="inputEl"
      v-model="keyword"
      class="ss-input"
      :class="{ invalid: invalid }"
      type="text"
      autocomplete="off"
      :placeholder="placeholder || '输入关键字搜索...'"
      :disabled="disabled"
      :title="selectedLabel || placeholder"
      @focus="onFocus"
      @input="onInput"
      @keydown.down.prevent="moveActive(1)"
      @keydown.up.prevent="moveActive(-1)"
      @keydown.enter.prevent="pickActive"
      @keydown.esc.prevent="close"
    />
    <button v-if="!disabled && modelValue" class="ss-clear" type="button" title="清空选择" @click="clear">×</button>
    <button
      v-if="!disabled"
      class="ss-arrow"
      :class="{ open }"
      type="button"
      tabindex="-1"
      title="展开 / 收起"
      @click="toggleArrow"
    >▾</button>
    <div v-if="open" class="ss-menu">
      <div v-if="!filtered.length" class="ss-empty">没有匹配的算法库</div>
      <div
        v-for="(opt, i) in filtered"
        :key="opt.value"
        class="ss-option"
        :class="{ active: i === activeIndex, chosen: opt.value === modelValue }"
        @mousedown.prevent="pick(opt)"
        @mouseenter="activeIndex = i"
      >{{ opt.label }}</div>
      <div v-if="filtered.length" class="ss-hint">输入即过滤 · ↑↓ 选 · Enter 确认 · Esc 关闭</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

type Option = { value: string; label: string }

const props = defineProps<{
  modelValue: string
  options: Option[]
  placeholder?: string
  disabled?: boolean
  invalid?: boolean
}>()
const emit = defineEmits<{ (e: 'update:modelValue', value: string): void }>()

const root = ref<HTMLElement | null>(null)
const inputEl = ref<HTMLInputElement | null>(null)
const open = ref(false)
const keyword = ref('')
const activeIndex = ref(-1)

const selectedLabel = computed(() => {
  const hit = props.options.find(o => o.value === props.modelValue)
  return hit ? hit.label : ''
})

// 未聚焦时输入框展示当前选中的完整名称；聚焦后变成可编辑的搜索词。
watch(() => props.modelValue, () => {
  if (!open.value) keyword.value = selectedLabel.value
})
watch(selectedLabel, (label) => {
  if (!open.value) keyword.value = label
}, { immediate: true })

const filtered = computed(() => {
  const kw = keyword.value.trim().toLowerCase()
  const list = kw
    ? props.options.filter(o => o.label.toLowerCase().includes(kw) || o.value.toLowerCase().includes(kw))
    : props.options
  return list
})

const toggleArrow = () => {
  if (open.value) {
    close()
  } else {
    inputEl.value?.focus()
  }
}

const onFocus = async () => {
  if (props.disabled) return
  open.value = true
  keyword.value = ''
  activeIndex.value = filtered.value.findIndex(o => o.value === props.modelValue)
  await nextTick()
  inputEl.value?.select()
}

const onInput = () => {
  open.value = true
  activeIndex.value = filtered.value.length ? 0 : -1
}

const moveActive = (delta: number) => {
  if (!open.value || !filtered.value.length) return
  const n = filtered.value.length
  activeIndex.value = ((activeIndex.value + delta) % n + n) % n
}

const pick = (opt: Option) => {
  emit('update:modelValue', opt.value)
  keyword.value = opt.label
  open.value = false
  activeIndex.value = -1
  inputEl.value?.blur()
}

const pickActive = () => {
  const opt = filtered.value[activeIndex.value]
  if (opt) pick(opt)
}

const clear = () => {
  emit('update:modelValue', '')
  keyword.value = ''
  open.value = false
  inputEl.value?.focus()
}

const close = () => {
  open.value = false
  keyword.value = selectedLabel.value
  inputEl.value?.blur()
}

// 点击组件外部关闭下拉，避免两个下拉同时展开。
const onDocClick = (e: MouseEvent) => {
  if (open.value && root.value && !root.value.contains(e.target as Node)) {
    open.value = false
    keyword.value = selectedLabel.value
  }
}
onMounted(() => document.addEventListener('mousedown', onDocClick))
onBeforeUnmount(() => document.removeEventListener('mousedown', onDocClick))
</script>

<style scoped>
.searchable-select { position: relative; width: 100%; }
.ss-input {
  width: 100%;
  padding: 8px 50px 8px 10px;
  border: 1px solid #d5dee8;
  border-radius: 8px;
  font-size: 0.9rem;
  background: #fff;
  color: #223;
  box-sizing: border-box;
}
.ss-input:focus { border-color: #2c6ec5; outline: none; }
.ss-input.invalid { border-color: #c62828; }
.ss-input:disabled { background: #f4f6f9; color: #99a; cursor: not-allowed; }
.ss-clear {
  position: absolute;
  right: 27px;
  top: 50%;
  transform: translateY(-50%);
  border: none;
  background: transparent;
  color: #8a95a5;
  font-size: 1.1rem;
  line-height: 1;
  cursor: pointer;
  padding: 0 3px;
}
.ss-clear:hover { color: #c62828; }
.ss-arrow {
  position: absolute;
  right: 7px;
  top: 50%;
  transform: translateY(-50%);
  border: none;
  background: transparent;
  color: #8a95a5;
  font-size: 0.85rem;
  line-height: 1;
  cursor: pointer;
  padding: 3px 4px;
  transition: transform 0.15s;
}
.ss-arrow.open { transform: translateY(-50%) rotate(180deg); }
.ss-menu {
  position: absolute;
  z-index: 60;
  top: calc(100% + 3px);
  left: 0;
  right: 0;
  max-height: 320px;
  overflow-y: auto;
  background: #fff;
  border: 1px solid #d5dee8;
  border-radius: 8px;
  box-shadow: 0 8px 24px rgba(20, 40, 70, 0.16);
  padding: 4px 0;
}
.ss-option {
  padding: 7px 10px;
  font-size: 0.88rem;
  color: #223;
  cursor: pointer;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ss-option:hover, .ss-option.active { background: #eaf2ff; }
.ss-option.chosen { color: #2c6ec5; font-weight: 600; }
.ss-empty { padding: 10px; font-size: 0.88rem; color: #99a; text-align: center; }
.ss-hint {
  padding: 7px 10px 4px;
  margin-top: 4px;
  border-top: 1px solid #edf1f6;
  font-size: 0.74rem;
  color: #8a95a5;
  white-space: nowrap;
}
</style>

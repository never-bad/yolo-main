<template>
  <teleport to="body">
    <transition name="dialog-fade">
      <div v-if="state.visible" class="dialog-overlay" @mousedown.self="handleOverlay">
        <div class="dialog-box">
          <div class="dialog-title">{{ state.title }}</div>
          <div class="dialog-message">{{ state.message }}</div>
          <div v-if="state.type === 'prompt'" class="dialog-input">
            <input
              ref="inputRef"
              v-model="state.value"
              type="text"
              @keydown.enter="confirm"
              @keydown.esc="cancel"
            />
          </div>
          <div class="dialog-actions">
            <template v-if="state.type === 'confirm'">
              <button class="secondary" @click="cancel">取消</button>
              <button class="danger" @click="confirm">确定</button>
            </template>
            <template v-else-if="state.type === 'prompt'">
              <button class="secondary" @click="cancel">取消</button>
              <button @click="confirm">确定</button>
            </template>
            <template v-else>
              <button @click="confirm">确定</button>
            </template>
          </div>
        </div>
      </div>
    </transition>
  </teleport>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { useDialog } from '@/composables/useDialog'

const { state, resolveDialog } = useDialog()
const inputRef = ref<HTMLInputElement | null>(null)

function confirm() {
  const value = state.type === 'confirm' ? true : state.type === 'prompt' ? state.value : null
  resolveDialog(value)
}

function cancel() {
  resolveDialog(state.type === 'confirm' ? false : null)
}

function handleOverlay() {
  cancel()
}

watch(
  () => state.visible,
  (visible) => {
    if (visible && state.type === 'prompt') {
      nextTick(() => inputRef.value?.focus())
    }
  }
)
</script>

<style scoped>
.dialog-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}

.dialog-box {
  background: #fff;
  border-radius: 10px;
  padding: 1.5rem 1.5rem 1.25rem;
  width: 380px;
  max-width: 90vw;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
}

.dialog-title {
  font-size: 1.15rem;
  font-weight: bold;
  color: #2c3e50;
  margin-bottom: 0.75rem;
}

.dialog-message {
  font-size: 1rem;
  color: #555;
  line-height: 1.6;
  margin-bottom: 0.5rem;
  white-space: pre-line;
  word-break: break-word;
}

.dialog-input {
  margin: 1rem 0 0.5rem;
}

.dialog-input input {
  width: 100%;
}

.dialog-actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
  margin-top: 1.25rem;
}

.dialog-fade-enter-active,
.dialog-fade-leave-active {
  transition: opacity 0.2s ease;
}

.dialog-fade-enter-from,
.dialog-fade-leave-to {
  opacity: 0;
}
</style>
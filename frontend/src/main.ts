import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'
import App from './App.vue'
import './style.css'
import { showAlert } from './composables/useDialog'

// 将原生 alert 替换为居中弹框，使整个平台的提示都显示在屏幕正中间
window.alert = (message?: string) => {
  showAlert(message ?? '')
}

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.mount('#app')

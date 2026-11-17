<template>
  <div id="app">
    <nav class="navbar">
      <div class="nav-brand">YOLO训练平台</div>
      <div class="nav-links">
        <router-link to="/">首页</router-link>
        <router-link to="/datasets">数据集</router-link>
        <router-link to="/annotate">标注</router-link>
        <router-link to="/train">训练</router-link>
        <router-link to="/models">模型管理</router-link>
      </div>
    </nav>
    <main class="main-content">
      <!-- 列表类页面 + 标注页 + 训练页全部缓存实例，切回不再整页重建：
           - 列表类：切回直接画旧数据
           - 标注页：保住当前任务/图片/缩放/筛选（原来切走就销毁，回来从第1张重来）
           - 训练页：保住填了一半的训练参数（原来只有点"保存配置"才持久化）
           标注页/训练页自行在 onDeactivated 停掉全局监听与轮询、onActivated 恢复。 -->
      <router-view v-slot="{ Component }">
        <keep-alive :include="['Dashboard', 'DatasetUpload', 'Models', 'DatasetAnnotate', 'Train']">
          <component :is="Component" />
        </keep-alive>
      </router-view>
    </main>
    <GlobalDialog />
  </div>
</template>

<script setup lang="ts">
import GlobalDialog from '@/components/GlobalDialog.vue'
</script>

<style scoped>
.navbar {
  background: #2c3e50;
  color: white;
  padding: 1rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.nav-brand {
  font-size: 1.5rem;
  font-weight: bold;
}

.nav-links {
  display: flex;
  gap: 1.5rem;
}

.nav-links a {
  color: white;
  text-decoration: none;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  transition: background 0.3s;
}

.nav-links a:hover,
.nav-links a.router-link-active {
  background: #34495e;
}

.main-content {
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
}
</style>

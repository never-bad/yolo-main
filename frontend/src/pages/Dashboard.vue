<template>
  <div class="dashboard">
    <div class="card">
      <h1>欢迎使用 YOLO 训练平台</h1>
      <p>这是一个基于 YOLOv8 的目标检测模型训练平台</p>
    </div>

    <div class="grid grid-2">
      <div class="card">
        <h2>📦 数据集管理</h2>
        <p>上传、准备和管理训练数据集</p>
        <router-link to="/datasets">
          <button>进入数据集</button>
        </router-link>
      </div>

      <div class="card">
        <h2>✏️ 数据标注</h2>
        <p>使用 Canvas 进行 BBox 标注</p>
        <router-link to="/annotate">
          <button>开始标注</button>
        </router-link>
      </div>

      <div class="card">
        <h2>🚀 模型训练</h2>
        <p>配置训练参数并启动训练任务</p>
        <router-link to="/train">
          <button>开始训练</button>
        </router-link>
      </div>

      <div class="card">
        <h2>📦 模型管理</h2>
        <p>上传、查看和管理训练好的模型</p>
        <router-link to="/models">
          <button>进入模型管理</button>
        </router-link>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
</script>

<style scoped>
.dashboard h1 {
  color: #2c3e50;
  margin-bottom: 0.5rem;
}

.dashboard p {
  color: #7f8c8d;
  margin-bottom: 1rem;
}

.card h2 {
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
}

.card button {
  margin-top: 1rem;
}
</style>

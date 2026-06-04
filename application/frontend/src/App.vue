<template>
  <div class="app-container">
    <!-- 顶部标题栏 -->
    <header class="app-header">
      <div class="header-content">
        <h1>
          <span class="header-icon">🚗</span>
          二手车智能估价系统
        </h1>
        <p class="header-subtitle">
          LightGBM 引擎 · 18.8 万条真实交易训练 · R²=0.66 · MAE≈$17,200
        </p>
      </div>
    </header>

    <!-- 主体：左右两栏 -->
    <main class="app-main">
      <!-- 左栏：输入表单 -->
      <section class="panel panel-input">
        <InputForm
          :brands="brands"
          :models="filteredModels"
          :loading="loading"
          @predict="handlePredict"
          @brand-change="handleBrandChange"
        />
      </section>

      <!-- 右栏：结果展示 -->
      <section class="panel panel-result">
        <template v-if="!result && !loading">
          <div class="empty-state">
            <el-icon :size="64" color="#c0c4cc"><svg viewBox="0 0 1024 1024" width="1em" height="1em"><path d="M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z" fill="currentColor"/><path d="M464 688a48 48 0 1 0 96 0 48 48 0 1 0-96 0zm24-112h48c4.4 0 8-3.6 8-8V296c0-4.4-3.6-8-8-8h-48c-4.4 0-8 3.6-8 8v272c0 4.4 3.6 8 8 8z" fill="currentColor"/></svg></el-icon>
            <p>输入车辆信息，点击"开始估价"查看预测结果</p>
          </div>
        </template>

        <template v-else>
          <div v-if="loading" class="loading-state">
            <el-icon class="is-loading" :size="48" color="#409eff"><svg viewBox="0 0 1024 1024" width="1em" height="1em"><path d="M512 64c247.4 0 448 200.6 448 448h-64c0-212.1-171.9-384-384-384V64zm0 832c-247.4 0-448-200.6-448-448H0c0 247.4 200.6 448 448 448v-64z" fill="currentColor"/></svg></el-icon>
            <p>正在分析车辆信息...</p>
          </div>

          <template v-if="result && !loading">
            <ResultCard :result="result" :stats="stats" />
            <ShapChart :items="result.shap_explanation" />
            <div class="nl-section">
              <h3>📝 估价解读</h3>
              <div
                v-for="(line, i) in result.natural_language"
                :key="i"
                class="nl-item"
              >
                {{ line }}
              </div>
              <div v-if="result.validation_warnings?.length" class="warnings-section">
                <h4>⚠ 注意事项</h4>
                <el-tag
                  v-for="(w, i) in result.validation_warnings"
                  :key="i"
                  type="warning"
                  style="margin: 4px 4px 4px 0"
                >
                  {{ w }}
                </el-tag>
              </div>
            </div>
          </template>
        </template>
      </section>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import InputForm from './components/InputForm.vue'
import ResultCard from './components/ResultCard.vue'
import ShapChart from './components/ShapChart.vue'
import { getBrands, getModels, getStats, predictPrice } from './api/index.js'

const brands = ref([])
const filteredModels = ref([])
const stats = ref({})
const result = ref(null)
const loading = ref(false)

onMounted(async () => {
  try {
    const [b, s] = await Promise.all([getBrands(), getStats()])
    brands.value = b
    stats.value = s
  } catch (e) {
    console.error('Failed to load initial data:', e)
  }
})

async function handleBrandChange(brand) {
  if (!brand) {
    filteredModels.value = []
    return
  }
  try {
    const models = await getModels(brand)
    filteredModels.value = models
  } catch (e) {
    filteredModels.value = []
  }
}

async function handlePredict(formData) {
  loading.value = true
  result.value = null
  try {
    const data = await predictPrice(formData)
    result.value = data
  } catch (e) {
    ElMessage.error('预测请求失败，请检查后端服务是否启动')
    console.error(e)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="result-card">
    <h3>📊 估价结果</h3>

    <!-- 价格卡片 -->
    <div class="price-cards">
      <div class="price-card price-card-main">
        <div class="price-label">预测价格</div>
        <div class="price-value">${{ result.predicted_price?.toLocaleString(undefined, { maximumFractionDigits: 0 }) }}</div>
        <el-tag
          :type="tierTagType"
          size="large"
          effect="dark"
          round
        >
          {{ result.price_tier }}
        </el-tag>
      </div>

      <div class="price-card price-card-sub">
        <div class="price-label">市场均价</div>
        <div class="price-value sub">${{ result.market_avg_price?.toLocaleString(undefined, { maximumFractionDigits: 0 }) }}</div>
        <div class="price-compare" :class="{ above: result.predicted_price > result.market_avg_price }">
          {{ result.predicted_price > result.market_avg_price ? '高于' : '低于' }}均价
          {{ Math.abs(((result.predicted_price - result.market_avg_price) / result.market_avg_price * 100)).toFixed(1) }}%
        </div>
      </div>
    </div>

    <!-- 市场位置条 -->
    <div class="market-bar">
      <div class="bar-label">
        <span>市场位置</span>
        <span>超过 {{ result.percentile }}% 的车辆</span>
      </div>
      <el-progress
        :percentage="result.percentile"
        :color="progressColor"
        :stroke-width="12"
        :show-text="false"
      />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  result: { type: Object, required: true },
  stats: { type: Object, default: () => ({}) },
})

const tierTagType = computed(() => {
  const tier = props.result.price_tier
  if (tier?.includes('低端')) return 'info'
  if (tier?.includes('中端')) return 'success'
  if (tier?.includes('高档')) return 'warning'
  return 'danger'
})

const progressColor = computed(() => {
  const p = props.result.percentile
  if (p < 30) return '#67c23a'
  if (p < 60) return '#409eff'
  if (p < 85) return '#e6a23c'
  return '#f56c6c'
})
</script>

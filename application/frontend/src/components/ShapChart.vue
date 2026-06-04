<template>
  <div class="shap-chart" v-if="items?.length">
    <h3>📈 关键影响因素 (SHAP)</h3>
    <p class="chart-hint">各特征对预测价格的正负贡献（美元）</p>
    <div class="shap-bars">
      <div
        v-for="item in sortedItems"
        :key="item.feature"
        class="shap-row"
      >
        <span class="shap-name">{{ item.display_name }}</span>
        <div class="shap-bar-track">
          <div
            class="shap-bar-fill"
            :class="{ positive: item.direction === 'positive', negative: item.direction === 'negative' }"
            :style="{ width: barWidth(item) + '%' }"
          />
        </div>
        <span
          class="shap-value"
          :class="{ positive: item.direction === 'positive', negative: item.direction === 'negative' }"
        >
          {{ item.direction === 'positive' ? '+' : '-' }}${{ Math.abs(item.contribution).toLocaleString(undefined, { maximumFractionDigits: 0 }) }}
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  items: { type: Array, default: () => [] },
})

const sortedItems = computed(() => {
  return [...props.items].sort((a, b) => {
    if (a.direction === 'positive' && b.direction === 'negative') return -1
    if (a.direction === 'negative' && b.direction === 'positive') return 1
    return Math.abs(b.contribution) - Math.abs(a.contribution)
  })
})

const maxContribution = computed(() => {
  if (!props.items.length) return 1
  return Math.max(...props.items.map(i => Math.abs(i.contribution)))
})

function barWidth(item) {
  return (Math.abs(item.contribution) / maxContribution.value) * 100
}
</script>

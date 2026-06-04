<template>
  <div class="input-form">
    <h3>📋 车辆参数</h3>

    <el-form
      ref="formRef"
      :model="form"
      :rules="rules"
      label-width="100px"
      label-position="top"
      size="large"
      @submit.prevent="handleSubmit"
    >
      <!-- 品牌 -->
      <el-form-item label="品牌" prop="brand">
        <el-select
          v-model="form.brand"
          filterable
          placeholder="选择品牌"
          style="width: 100%"
          @change="onBrandChange"
        >
          <el-option
            v-for="b in brands"
            :key="b"
            :label="b"
            :value="b"
          />
        </el-select>
      </el-form-item>

      <!-- 车型 -->
      <el-form-item label="车型" prop="model">
        <el-select
          v-model="form.model"
          filterable
          placeholder="先选品牌，再选车型"
          style="width: 100%"
          :disabled="!form.brand"
        >
          <el-option
            v-for="m in models"
            :key="m"
            :label="m"
            :value="m"
          />
        </el-select>
      </el-form-item>

      <!-- 年份 + 里程 -->
      <el-row :gutter="16">
        <el-col :span="12">
          <el-form-item label="出厂年份" prop="model_year">
            <el-input-number
              v-model="form.model_year"
              :min="1990"
              :max="2026"
              :step="1"
              controls-position="right"
              style="width: 100%"
              @change="onBoundary('model_year', 1990, 2026, '年')"
            />
            <span v-if="hints.model_year" class="field-warn">{{ hints.model_year }}</span>
            <span v-else class="field-hint">范围 1990 - 2026 年</span>
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="里程(英里)" prop="milage">
            <el-input-number
              v-model="form.milage"
              :min="100"
              :max="400000"
              :step="1000"
              controls-position="right"
              style="width: 100%"
              @change="onBoundary('milage', 100, 400000, '英里')"
            />
            <span v-if="hints.milage" class="field-warn">{{ hints.milage }}</span>
            <span v-else class="field-hint">范围 100 - 400,000 英里</span>
          </el-form-item>
        </el-col>
      </el-row>

      <!-- 马力 + 排量 -->
      <el-row :gutter="16">
        <el-col :span="12">
          <el-form-item label="马力(HP)" prop="engine_hp">
            <el-input-number
              v-model="form.engine_hp"
              :min="50"
              :max="1500"
              :step="10"
              controls-position="right"
              style="width: 100%"
              @change="onBoundary('engine_hp', 50, 1500, 'HP')"
            />
            <span v-if="hints.engine_hp" class="field-warn">{{ hints.engine_hp }}</span>
            <span v-else class="field-hint">范围 50 - 1,500 HP</span>
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="排量(L)" prop="engine_liter">
            <el-input-number
              v-model="form.engine_liter"
              :min="0.5"
              :max="8.0"
              :step="0.1"
              :precision="1"
              controls-position="right"
              style="width: 100%"
              @change="onBoundary('engine_liter', 0.5, 8.0, 'L')"
            />
            <span v-if="hints.engine_liter" class="field-warn">{{ hints.engine_liter }}</span>
            <span v-else class="field-hint">范围 0.5 - 8.0 L</span>
          </el-form-item>
        </el-col>
      </el-row>

      <!-- 燃油类型 -->
      <el-form-item label="燃油类型" prop="fuel_type">
        <el-select v-model="form.fuel_type" style="width: 100%">
          <el-option v-for="f in fuelTypes" :key="f" :label="f" :value="f" />
        </el-select>
      </el-form-item>

      <!-- 事故记录 + 产权 -->
      <el-row :gutter="16">
        <el-col :span="12">
          <el-form-item label="事故记录" prop="accident">
            <el-select v-model="form.accident" style="width: 100%">
              <el-option
                v-for="a in accidentOptions"
                :key="a.value"
                :label="a.label"
                :value="a.value"
              />
            </el-select>
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="产权状态" prop="clean_title">
            <el-select v-model="form.clean_title" style="width: 100%">
              <el-option label="清晰产权" value="Yes" />
              <el-option label="非清晰产权" value="No" />
            </el-select>
          </el-form-item>
        </el-col>
      </el-row>

      <!-- 提交按钮 -->
      <el-form-item>
        <el-button
          type="primary"
          size="large"
          style="width: 100%"
          :loading="loading"
          @click="handleSubmit"
        >
          {{ loading ? '分析中...' : '💰 开始估价' }}
        </el-button>
      </el-form-item>
    </el-form>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'

const props = defineProps({
  brands: { type: Array, default: () => [] },
  models: { type: Array, default: () => [] },
  loading: { type: Boolean, default: false },
})

const emit = defineEmits(['predict', 'brand-change'])

const formRef = ref(null)

const fuelTypes = [
  'Gasoline', 'Diesel', 'Electric', 'Hybrid',
  'E85 Flex Fuel', 'Plug-In Hybrid', 'Unknown',
]

const accidentOptions = [
  { label: '无事故记录', value: 'None reported' },
  { label: '有事故记录', value: 'At least 1 accident or damage reported' },
  { label: '未知', value: '' },
]

const form = reactive({
  brand: '',
  model: '',
  model_year: 2018,
  milage: 50000,
  engine_hp: 200,
  engine_liter: 2.0,
  fuel_type: 'Gasoline',
  accident: 'None reported',
  clean_title: 'Yes',
})

const rules = {
  brand: [{ required: true, message: '请选择品牌', trigger: 'change' }],
  model: [{ required: true, message: '请选择车型', trigger: 'change' }],
  model_year: [
    { required: true, message: '请输入出厂年份', trigger: 'blur' },
    { type: 'number', min: 1990, max: 2026, message: '年份需在1990-2026之间', trigger: 'blur' },
  ],
  milage: [
    { required: true, message: '请输入里程', trigger: 'blur' },
    { type: 'number', min: 100, max: 400000, message: '里程需在合理范围', trigger: 'blur' },
  ],
  engine_hp: [
    { required: true, message: '请输入马力', trigger: 'blur' },
    { type: 'number', min: 50, max: 1500, message: '马力需在50-1500之间', trigger: 'blur' },
  ],
  engine_liter: [
    { required: true, message: '请输入排量', trigger: 'blur' },
    { type: 'number', min: 0.5, max: 8.0, message: '排量需在0.5-8.0L之间', trigger: 'blur' },
  ],
}

// 边界警告提示
const hints = reactive({
  model_year: '',
  milage: '',
  engine_hp: '',
  engine_liter: '',
})

function onBoundary(field, min, max, unit) {
  const v = form[field]
  if (v === min) {
    hints[field] = `已自动调整为下限 ${min.toLocaleString()} ${unit}`
    setTimeout(() => { hints[field] = '' }, 3000)
  } else if (v === max) {
    hints[field] = `已自动调整为上限 ${max.toLocaleString()} ${unit}`
    setTimeout(() => { hints[field] = '' }, 3000)
  } else {
    hints[field] = ''
  }
}

function onBrandChange(brand) {
  form.model = ''
  emit('brand-change', brand)
}

async function handleSubmit() {
  if (!formRef.value) return
  try {
    await formRef.value.validate()
    emit('predict', { ...form })
  } catch {
    ElMessage.warning('请检查输入参数')
  }
}
</script>

<style scoped>
.field-hint {
  display: block;
  font-size: 11px;
  color: #a8abb2;
  margin-top: 2px;
  line-height: 1.4;
}
.field-warn {
  display: block;
  font-size: 11px;
  color: #e6a23c;
  margin-top: 2px;
  line-height: 1.4;
}
</style>

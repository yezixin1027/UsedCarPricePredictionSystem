import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

export function getBrands() {
  return api.get('/brands').then(r => r.data.brands)
}

export function getModels(brand) {
  return api.get('/models', { params: { brand } }).then(r => r.data.models)
}

export function getStats() {
  return api.get('/stats').then(r => r.data)
}

export function predictPrice(data) {
  return api.post('/predict', data).then(r => r.data)
}

export function validateFields(data) {
  return api.post('/validate', data).then(r => r.data)
}

import axios from 'axios'

const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export const apiClient = axios.create({
  baseURL,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Interceptor for basic error normalization
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const normalized = new Error(
      error?.response?.data?.detail || error?.message || 'request_failed'
    )
    normalized.status = error?.response?.status
    normalized.data = error?.response?.data
    return Promise.reject(normalized)
  }
)

// Core endpoints
export const getHealth = () => apiClient.get('/health').then(r => r.data)
export const getModelInfo = () => apiClient.get('/model/info').then(r => r.data)
export const getAnalyticsSummary = () => apiClient.get('/analytics/summary').then(r => r.data)

// Prediction
export const predictText = (text) => apiClient.post('/predict', { text }).then(r => r.data)
export const predictUrl = (url) => apiClient.post('/predict/url', { url }).then(r => r.data)
export const predictBatch = (items) => apiClient.post('/predict/batch', { items }).then(r => r.data)
export const predictExplain = (payload) => apiClient.post('/predict/explain', payload).then(r => r.data)
export const predictCounterfactual = (payload) => apiClient.post('/predict/counterfactual', payload).then(r => r.data)

// Datasets
export const listDatasets = () => apiClient.get('/datasets').then(r => r.data)
export const getDatasetInfo = (name) => apiClient.get(`/datasets/${name}/info`).then(r => r.data)
export const getDatasetSample = (name, size = 5) => apiClient.get(`/datasets/${name}/sample`, { params: { size } }).then(r => r.data)
export const runFullPipeline = (name) => apiClient.post(`/datasets/full-pipeline/${name}`).then(r => r.data)
export const runCrossDomainEvaluation = (body = {}) => apiClient.post('/datasets/evaluate/cross-domain', body, { timeout: 180000 }).then(r => r.data)
export const runDatasetEvaluation = (name, body = {}) => apiClient.post(`/datasets/evaluate/${name}`, body, { timeout: 180000 }).then(r => r.data)

// Reports
export const listReports = () => apiClient.get('/reports').then(r => r.data)
export const getReportMarkdown = (filename) => apiClient.get(`/reports/${encodeURIComponent(filename)}`, { responseType: 'text' }).then(r => r.data)

// Feedback
export const listFeedback = () => apiClient.get('/feedback/').then(r => r.data)
export const submitFeedback = (entry) => apiClient.post('/feedback/', entry).then(r => r.data)
export const deleteFeedback = (id) => apiClient.delete(`/feedback/${id}`).then(r => r.data)
export const exportFeedback = (params) => apiClient.post('/feedback/export', params).then(r => r.data)



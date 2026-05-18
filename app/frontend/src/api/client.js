import axios from 'axios';

const apiBaseURL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

const client = axios.create({
  baseURL: apiBaseURL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

client.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.error ||
      error.response?.data?.detail?.error ||
      error.response?.data?.detail ||
      error.message ||
      'Unknown API error';

    return Promise.reject({
      message,
      status: error.response?.status,
      details: error.response?.data?.details || error.response?.data?.detail || null,
      raw: error,
    });
  },
);

export const getGenes = async () => {
  const response = await client.get('/genes');
  return response.data;
};

export const getMetadata = async () => {
  const response = await client.get('/metadata');
  return response.data;
};

export const getHealth = async () => {
  const response = await client.get('/health');
  return response.data;
};

export const getMetrics = async () => {
  const response = await client.get('/metrics');
  return response.data;
};

export const predict = async (geneValues) => {
  const response = await client.post('/predict', { gene_values: geneValues });
  return response.data;
};

export default client;

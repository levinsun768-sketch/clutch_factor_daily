import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    allowedHosts: [
      'u567908-88a3-a93f4741.bjb1.seetacloud.com',
      'uu567908-88a3-a93f4741.bjb1.seetacloud.com',
      'localhost',
      '127.0.0.1'
    ],
    proxy: {
      '/api': {
        target: process.env.VITE_PROXY_TARGET || process.env.VITE_API_BASE_URL || 'http://127.0.0.1:6008',
        changeOrigin: true
      }
    }
  }
});

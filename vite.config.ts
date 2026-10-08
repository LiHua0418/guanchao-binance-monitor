import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: { '/api/b4': 'http://127.0.0.1:8765' },
  },
  preview: {
    host: '0.0.0.0',
    port: 5173,
    proxy: { '/api/b4': 'http://127.0.0.1:8765' },
  },
})

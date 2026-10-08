import { spawn, spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'

const root = fileURLToPath(new URL('..', import.meta.url))
const mode = process.argv[2] ?? 'dev'
const venvPython = resolve(root, '.venv-training', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')
const python = process.env.GUANCHAO_PYTHON || (existsSync(venvPython) ? venvPython : 'python')
const children = new Set()
let stopping = false
function stop(code = 0) {
  if (stopping) return
  stopping = true
  for (const child of children) {
    // Windows venv launchers have a Python descendant; stop only the process
    // trees this launcher owns. A pre-existing inference service is not owned.
    if (process.platform === 'win32' && child.pid) spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' })
    else child.kill()
  }
  process.exitCode = code
}
function launch(command, args) {
  const child = spawn(command, args, { cwd: root, stdio: 'inherit', windowsHide: true })
  children.add(child)
  child.on('error', (error) => { console.error(error.message); stop(1) })
  child.on('exit', (code) => { children.delete(child); stop(code ?? 0) })
  return child
}
async function healthy() {
  try {
    const response = await fetch('http://127.0.0.1:8765/api/b4/health', { signal: AbortSignal.timeout(1000) })
    if (!response.ok) return false
    const body = await response.json()
    return body.ok === true && body.modelId === 'guanchao-b4' && body.service === 'local-frozen-inference'
  } catch { return false }
}
process.on('SIGINT', () => stop())
process.on('SIGTERM', () => stop())
if (await healthy()) {
  console.log('B4 本地推理服务已在 127.0.0.1:8765 运行。')
} else {
  launch(python, ['scripts/serve-b4.py'])
  for (let attempt = 0; attempt < 30 && !stopping; attempt++) {
    if (await healthy()) break
    await new Promise((done) => setTimeout(done, 500))
  }
  if (!stopping && !(await healthy())) {
    console.error('B4 服务启动失败，请检查 Python 环境和训练权重。')
    stop(1)
  }
}
if (!stopping && mode !== 'model') {
  const args = [resolve(root, 'node_modules/vite/bin/vite.js')]
  if (mode === 'preview') args.push('preview')
  args.push(...process.argv.slice(3))
  launch(process.execPath, args)
}

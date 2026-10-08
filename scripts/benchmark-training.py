"""Measure B3 CPU training throughput without changing any deployed artifact.

Fixed synthetic batch, same model/loss/AdamW optimizer as train-b3.py. This is a
compute benchmark, not a predictive-performance experiment or a GPU benchmark.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import torch
from torch.nn import functional as F

MODULE_PATH = Path(__file__).with_name("train-b3.py")
spec = importlib.util.spec_from_file_location("b3_training", MODULE_PATH)
b3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b3)


def hardware():
    info = {"cpu": platform.processor(), "logicalProcessors": os.cpu_count(), "physicalCores": None,
            "gpu": None, "pytorch": str(torch.__version__), "cudaAvailable": torch.cuda.is_available(),
            "pytorchCudaRuntime": torch.version.cuda, "platform": platform.platform()}
    if os.name == "nt":
        try:
            result = subprocess.run(["powershell", "-NoProfile", "-Command",
                                     "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"],
                                    capture_output=True, text=True, timeout=15, check=True)
            cpu = json.loads(result.stdout.strip())
            if isinstance(cpu, list):
                cpu = cpu[0]
            info.update(cpu=cpu["Name"], physicalCores=cpu["NumberOfCores"])
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=15, check=True)
        info["gpu"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def measure(threads, seconds, batch_size):
    torch.set_num_threads(threads)
    torch.manual_seed(b3.SEED)
    model = b3.B3().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.015)
    generator = torch.Generator().manual_seed(b3.SEED + 1)
    x = torch.randn(batch_size, b3.SEQUENCE, 8, generator=generator)
    target = torch.randn(batch_size, b3.HORIZON, generator=generator)
    weights = torch.tensor([.6] + [.4 / 11] * 11)

    def step():
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = (F.smooth_l1_loss(output, target, beta=.5, reduction="none") * weights).sum(dim=1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

    for _ in range(12):
        step()
    started = time.perf_counter()
    steps = 0
    while time.perf_counter() - started < seconds:
        step()
        steps += 1
    elapsed = time.perf_counter() - started
    return {"threads": threads, "steps": steps, "seconds": elapsed,
            "stepsPerSecond": steps / elapsed, "samplesPerSecond": steps * batch_size / elapsed,
            "millisecondsPerStep": elapsed / steps * 1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4, 6, 8, 12, 16])
    parser.add_argument("--seconds", type=float, default=.9, help="Measured duration for each repetition; minimum .5 seconds.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.seconds < .5 or args.repeats < 1 or any(n < 1 for n in args.threads):
        parser.error("Use positive threads/repeats and at least .5 seconds per measurement.")
    torch.set_num_interop_threads(2)
    info = hardware()
    print(json.dumps(info), flush=True)
    rows = {n: [] for n in args.threads}
    order_rng = random.Random(b3.SEED)
    for repeat in range(args.repeats):
        order = args.threads.copy()
        order_rng.shuffle(order)
        for threads in order:
            result = measure(threads, args.seconds, args.batch_size)
            result["repeat"] = repeat + 1
            rows[threads].append(result)
            print(f"CPU threads={threads:2d} repeat={repeat+1} samples/s={result['samplesPerSecond']:.1f} step_ms={result['millisecondsPerStep']:.3f}", flush=True)
    summary = [{"threads": n, "medianSamplesPerSecond": statistics.median(r["samplesPerSecond"] for r in runs),
                "medianStepMilliseconds": statistics.median(r["millisecondsPerStep"] for r in runs),
                "runs": runs} for n, runs in sorted(rows.items())]
    best = max(summary, key=lambda r: r["medianSamplesPerSecond"])
    base = next((r["medianSamplesPerSecond"] for r in summary if r["threads"] == 1), None)
    for row in summary:
        row["speedupVsOneThread"] = row["medianSamplesPerSecond"] / base if base else None
    training_report = json.loads((b3.REPORTS / "b3-training-report.json").read_text(encoding="utf-8"))
    durations = {r["interval"]: r["durationSeconds"] for r in training_report["intervals"]}
    report = {"benchmark": "B3 CPU forward/backward/AdamW step", "hardware": info,
              "batchSize": args.batch_size, "sequence": b3.SEQUENCE, "features": 8,
              "parameters": sum(p.numel() for p in b3.B3().parameters()), "seed": b3.SEED,
              "warmupStepsPerRun": 12, "minimumMeasuredSecondsPerRun": args.seconds,
              "repeats": args.repeats, "interOpThreads": 2, "syntheticInputs": True,
              "recommendedCpuThreads": best["threads"], "results": summary,
              "previousTrainingThreads": 6, "previousIntervalTrainingSeconds": durations,
              "limitations": ["Synthetic steady-state batch excludes download/feature construction/validation/export.",
                              "Measured recommendation is for this small B3 architecture and batch size, not every model.",
                              "GPU throughput was not tested; this benchmark measures CPU only.",
                              "Compute throughput does not measure forecasting accuracy."]}
    b3.write_json(b3.REPORTS / "b3-compute-benchmark.json", report)
    lines = ["# B3 训练计算基准", "", f"CPU：{info['cpu']}；物理核：{info['physicalCores']}；逻辑线程：{info['logicalProcessors']}。",
             f"GPU：{info['gpu']}。PyTorch：{info['pytorch']}；CUDA 可用：{info['cudaAvailable']}；PyTorch CUDA runtime：{info['pytorchCudaRuntime']}。", "",
             "现有训练已经使用 6 个 intra-op 线程和 2 个 inter-op 线程，不是单核。" +
             ("当前安装的是 CPU 版 PyTorch，因此尚未使用显卡。" if torch.version.cuda is None else "当前 PyTorch 包含 CUDA runtime，但本基准仅测 CPU。"), "",
             f"本基准使用相同 B3 网络（{report['parameters']} 参数）、batch {args.batch_size}、48×8 输入、下一根优先 Huber loss、反向传播、梯度clip和AdamW。输入为固定合成数据；每次先预热12步，每档线程测{args.repeats}次、每次至少{args.seconds:.1f}秒；顺序固定seed随机打乱，表格取中位数。", "",
             "| CPU线程 | 样本/秒（中位数） | 每训练步ms | 相对单线程吞吐 |", "|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['threads']} | {row['medianSamplesPerSecond']:.0f} | {row['medianStepMilliseconds']:.3f} | {row['speedupVsOneThread']:.2f}× |")
    lines += ["", f"本机实测推荐：**{best['threads']} 个 CPU 线程**。小网络可能因同步和线程调度开销无法随核数线性提速，应以实测为准。",
              "", "最近一次已完成训练使用6线程，各周期从特征准备至导出耗时如下（不含公共行情下载）：", "",
              *[f"- {interval}：{seconds:.2f} 秒" for interval, seconds in durations.items()],
              "", "这些耗时与吞吐只说明计算效率，不说明预测精度。加CPU线程、启用GPU或增加训练轮数都不会自动增加行情中的可预测信号。",
              "本次没有安装 CUDA 版 PyTorch，也没有执行 GPU 训练测试。新增 CUDA 训练路径仅在环境支持时使用；显式指定不可用 CUDA 会清楚报错。",
              "合成batch吞吐未计下载、特征构造、验证或导出，不能直接当作完整训练耗时提速比例。基准文件没有修改任何已冻结模型权重或预测质量指标。", "",
              "```powershell", ".\\.venv-training\\Scripts\\python.exe scripts/benchmark-training.py", "```"]
    (b3.REPORTS / "b3-compute-benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Recommended CPU threads: {best['threads']}", flush=True)


if __name__ == "__main__":
    main()

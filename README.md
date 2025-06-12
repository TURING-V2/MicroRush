# MicroRush HFT

**MicroRush** is a high-performance, real-time high-frequency trading (HFT) engine written in [Zig](https://ziglang.org/). It is designed for ultra-low-latency signal processing and execution using cutting-edge system-level features.

## Features

- **Ultra-Fast Signal Engine**  
  SIMD-accelerated signal generation using AVX2 across multiple CPU cores and thread pools.

- **GPU-Accelerated Analytics**  
  Real-time RSI, StochRSI, and order book metrics computed using CUDA kernels.

- **Lock-Free Queues**  
  Inter-thread communication built with lock-free data structures to avoid contention.

- **Multithreaded Execution Engine**  
  Uses Zig’s thread pools and affinity pinning for high throughput and core-level balancing.

- **Atomic Portfolio Management**  
  All trade operations are lock-protected and atomic for safe concurrent access.

- **Live Exchange Integration**  
  Fetches real-time market data from **Binance** using WebSocket streams and REST APIs.

- **Balanced Load Metrics**  
  `metrics.zig` implements core-optimized, lock-free load reporting for adaptive tuning and performance analysis.

- **Built-in Benchmarking Tools**  
  Track system performance using built-in metrics collectors and CLI flags.

---

## Key Components

- **`core/`** – Core HFT runtime loop, lock-free logic, signal dispatch.
- **`signal_engine.zig`** – SIMD-powered, batched signal calculations.
- **`statcalc/`** – GPU-based technical indicator computation.
- **`trade_handler.zig`** – Portfolio and execution manager, thread-safe.
- **`metrics.zig`** – Real-time metrics collection, lock-free load tracker.

---

## Requirements

- Zig (latest [master build](https://ziglang.org/download/))
- CUDA Toolkit nvcc (>=release 12.8, V12.8.93)
- AVX2-capable CPU (modern Intel or AMD)
- Linux (tested on Gentoo Amd64)

---

## Building

Use the provided Makefile:

```sh
make build         # Standard build
make build-fast    # Optimized with release-fast


make run           # Run the engine
make run-metrics   # Run with metrics collection

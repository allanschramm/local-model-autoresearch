# CPU Inference Optimization Guide for llama.cpp

*See also: [`docs/llamacpp-toolset.md`](../llamacpp-toolset.md) for llama.cpp build targets and CLI flags, and [`advanced-inference-optimizations.md`](./advanced-inference-optimizations.md) for general runtime & memory optimizations.*

## 1. Build Configuration

### Native vs Universal Builds

| Option | Command | When to Use |
|--------|---------|-------------|
| **Native** | `-DGGML_NATIVE=ON` | Single rig, maximum speed |
| **Universal** | `-DGGML_NATIVE=OFF` | Cross-machine binary |

**Native builds** optimize for your CPU at compile time. **Universal builds** include multiple instruction set variants but are larger and compile slower.

### x86 CPU Instruction Sets

| Flag | Intel | AMD | Purpose |
|------|-------|-----|---------|
| `GGML_AVX` | All x86-64 | Zen1+ | Baseline vectorized ops |
| `GGML_AVX2` | All x86-64 | Zen1+ | Enhanced vector ops |
| `GGML_AVX512` | 6th-gen+ | Genoa+ | 512-bit vectorization |
| `GGML_AVX512_VNNI` | Ice Lake+ | - | INT8 matrix ops |
| `GGML_AVX512_BF16` | Cooper Lake+ | - | BF16 operations |
| `GGML_AMX_TILE` | Sapphire Rapids+ | - | AMX tile ops |
| `GGML_AMX_INT8` | Sapphire Rapids+ | - | AMX INT8 |
| `GGML_AMX_BF16` | Sapphire Rapids+ | - | AMX BF16 |

**AMD note:** Zen2/Zen3 support AVX2 only. Zen4+ and EPYC (Genoa+) support AVX-512. No AMX on AMD.

### Build Commands

**Linux / macOS (Native build with OpenMP & AVX-512 / AMX):**

```bash
# Intel Sapphire Rapids+ (AVX-512 + AMX)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
  -DGGML_AVX512=ON -DGGML_AMX_TILE=ON -DGGML_AMX_INT8=ON -DGGML_AMX_BF16=ON

# Standard x86-64 Native Build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON
cmake --build build --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

**Windows (MSVC + Ninja with Native AVX2):**

```powershell
cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DGGML_AVX2=ON
cmake --build build --config Release
```

### OpenMP Parallelization

`-DGGML_OPENMP=ON` enables OpenMP for parallel layer execution. Essential for multi-core CPU inference. Confirm at runtime via the `system_info` line printed on startup (`OPENMP = 1`).

---

## 2. NUMA Configuration

Non-Uniform Memory Access (NUMA) matters on multi-socket systems.

> **Platform:** NUMA, CPU affinity, and allocator preload below are Linux-focused. Windows/macOS use the native build in §1; allocator preload is Linux-only.

### NUMA Strategies (`llama-server --numa`)

| Mode | Description | Best For |
|------|-------------|----------|
| `distribute` | Spread threads across nodes | Multi-socket servers |
| `isolate` | Bind threads to the start node only | Single-node locality |
| `numactl` | Honor the CPU map from the `numactl` wrapper | Custom core/memory binding |

### Enable NUMA

```bash
llama-server --numa distribute

# Custom map: wrap with numactl, then tell llama.cpp to honor that map
numactl --cpunodebind=0 --membind=0 llama-server --numa numactl -m model.gguf
```

### Linux NUMA Inspection

```bash
lscpu | grep -i numa
numactl --hardware
```

---

## 3. Thread Affinity & Scheduling

### Thread Configuration

```bash
llama-server -t 8          # 8 threads
llama-server -t 12 -tb 8   # 12 main, 8 batch threads
```

### CPU Affinity Options

| Option | Example | Description |
|--------|---------|-------------|
| `--cpu-mask` | `-C 0xFF` | Hex mask (cores 0-7) |
| `--cpu-range` | `-Cr 0-7` | Core range |
| `--cpu-strict` | `--cpu-strict 1` | Strict placement |
| `--cpu-mask-batch` | `-Cb 0xFF` | Batch thread mask |
| `--cpu-range-batch` | `-Crb 0-7` | Batch thread range |

### Physical Core Affinity

On systems with SMT/hyperthreading, bind to physical cores to avoid cache thrashing and ensure stable latency:

```bash
# Discover physical core mapping (Linux)
lscpu -e=CPU,CORE,SOCKET

# Option A: Direct llama-server CPU core binding (set range to match physical core IDs from `lscpu -e`)
llama-server -m model.gguf -t 8 --cpu-range 0-7 --cpu-strict 1

# Option B: OS-level thread binding via taskset (8 physical cores; match -t)
# Inspect topology first with `lscpu -e` to pick physical core IDs on your machine.
sudo taskset -cp 0,2,4,6,8,10,12,14 $(pidof llama-server)
```

---

## 4. Memory Allocators

Resolve the library path on your distro first, then preload:

```bash
# Find the .so on this machine (path differs by distro)
ldconfig -p | grep -E 'tcmalloc|jemalloc'

# Example once you have the path:
LD_PRELOAD="$(ldconfig -p | awk '/libtcmalloc\.so/{print $NF; exit}')" \
  llama-server -m model.gguf -t 8

LD_PRELOAD="$(ldconfig -p | awk '/libjemalloc\.so/{print $NF; exit}')" \
  llama-server -m model.gguf -t 8
```

Also covered in [`advanced-inference-optimizations.md`](./advanced-inference-optimizations.md) §2.

---

## 5. GGUF Quantization for CPU

### Model Quants

| Type | BPW | CPU Recommendation |
|------|-----|-------------------|
| Q2_K | 2.56 | Lowest, fast |
| Q3_K_M | 3.35 | Low, compact |
| Q4_K_M | 4.35 | **Best balance** |
| Q5_K_M | 5.34 | Quality focus |
| Q6_K | 6.56 | Near-fp16 quality |
| Q8_0 | 8.50 | Highest, slowest |

**CPU Memory Bandwidth & Cache Strategy:**
CPU inference speed is strictly bound by RAM bandwidth and L3 cache footprint.
- **Cache Fit:** Smaller quants (Q4_K_M, IQ4_XS) fit a larger portion of layer weights into L3 cache during repeat tokens/attention calculations.
- **Recommendation:** Start with `Q4_K_M`. For memory-bandwidth constrained CPUs (dual-channel DDR4/DDR5), prefer `Q4_K_M` or `IQ4_XS` over heavy `Q8_0` to maintain token generation speed. Use `Q5_K_M` when high accuracy is required and RAM bandwidth allows.

### KV Cache Compression

```bash
llama-server -m model.gguf -ctk q4_0 -ctv q4_0
```

Saves 50-75% KV cache memory. Trade-off: minor quality loss.

### CPU Cache & Context Overhead

KV cache size directly impacts L2/L3 cache hits during generation:

| Cache Tier | Typical Size | Considerations |
|------------|--------------|----------------|
| L2 | 256 KB–1 MB per core | Holds immediate attention vector computations |
| L3 | 8 MB–128 MB shared | Holds active KV cache layers & small quant blocks |

> **Formula**: KV cache size ≈ `2 × n_kv_heads × head_dim × bytes_per_elem × CTX_SIZE × n_layers`. Check L3 cache size via `lscpu` or `cat /sys/devices/system/cpu/cpu0/cache/index*/size`. When KV cache exceeds L3 cache capacity, attention layers fall back to main system RAM bandwidth. Use KV cache quantizing (`-ctk q4_0 -ctv q4_0`) to fit longer contexts within L3 cache bounds.

---

## 6. Practical Tuning Workflow

### 1. Quick Validation

```bash
# Smoke test
llama-cli -m model.gguf -p "Hello" -n 128 -ngl 0
```

### 2. Thread Benchmarking

```bash
for t in 4 6 8 10 12; do
  llama-bench -m model.gguf -t $t -p 512 -n 128 -ngl 0 -o json
done
```

### 3. NUMA Comparison

```bash
llama-bench -m model.gguf -t 8 --numa distribute -p 512 -n 128
llama-bench -m model.gguf -t 8 --numa isolate -p 512 -n 128
```

### 4. Allocator Comparison

Run the same `llama-bench` command under the default allocator, then with `tcmalloc` / `jemalloc` via the `LD_PRELOAD` patterns in §4. Compare wall time / tokens/s.

---

## 7. Hardware-Specific Notes

### Intel CPUs

| Generation | AVX-512 | AMX | Notes |
|------------|---------|-----|-------|
| 6th-gen HEDT/Xeon (Skylake-X/SP) | Yes | - | Early AVX-512 (consumer Skylake lacks it) |
| Ice Lake | Yes | - | VNNI (INT8 matrix extension; no BF16) |
| Cooper Lake | Yes | - | AVX-512 BF16 extension |
| Sapphire Rapids | Yes | Yes | AMX tile/int8/bf16 |

Build: `-DGGML_NATIVE=ON -DGGML_AVX512=ON`

### AMD CPUs

| Family | AVX-512 | Notes |
|--------|---------|-------|
| Zen1-Zen3 | No | AVX2 only |
| Zen4+ (Genoa/EPYC 9004+) | Yes | AVX-512 support |
| EPYC Milan (Zen3) | No | AVX2 only; Genoa replaced it |

Build: `-DGGML_NATIVE=ON -DGGML_AVX2=ON` (or AVX512 on Zen4+)

---

## 8. Common Pitfalls

1. **Over-threading:** More threads ≠ faster. Match physical cores, not logical cores.
2. **NUMA on single-socket:** Can hurt performance. Test before enabling.
3. **Allocator preload:** Missing library crashes the process. Verify path exists.

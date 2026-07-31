# CPU Inference Optimization Guide for llama.cpp

*See also: [`docs/llamacpp-toolset.md`](../llamacpp-toolset.md) for llama.cpp build targets and CLI flags, and [`advanced-inference-optimizations.md`](./advanced-inference-optimizations.md) for general runtime & memory optimizations (including why `tcmalloc` / `jemalloc` help).*

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
| `GGML_AVX` | Sandy Bridge+ | Bulldozer+ / Zen1+ | Baseline vectorized ops |
| `GGML_AVX2` | Haswell+ | Zen1+ | Enhanced vector ops |
| `GGML_AVX512` | Skylake-X / Xeon SP+ (not mainstream 6th–10th Core) | Zen4+ / Genoa+ | 512-bit vectorization |
| `GGML_AVX512_VNNI` | Ice Lake+ | - | INT8 matrix ops |
| `GGML_AVX512_BF16` | Cooper Lake+ | - | BF16 operations |
| `GGML_AMX_TILE` | Sapphire Rapids+ | - | AMX tile ops |
| `GGML_AMX_INT8` | Sapphire Rapids+ | - | AMX INT8 |
| `GGML_AMX_BF16` | Sapphire Rapids+ | - | AMX BF16 |

**Intel note:** Consumer Skylake–Comet Lake desktop Core lacks AVX-512; only Skylake-X / Xeon SP (and later server/HEDT lines) have it. Ice Lake adds VNNI; Cooper Lake adds BF16; Sapphire Rapids adds AMX.

**AMD note:** Zen1–Zen3 are AVX2-only. Zen4+ and EPYC Genoa+ add AVX-512. No AMX on AMD.

### Build Commands

```bash
# Intel Sapphire Rapids+ (AVX-512 + AMX)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
  -DGGML_AVX512=ON -DGGML_AMX_TILE=ON -DGGML_AMX_INT8=ON -DGGML_AMX_BF16=ON

# Standard native build (AVX2 on typical desktops)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON
cmake --build build --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

### OpenMP Parallelization

`-DGGML_OPENMP=ON` enables OpenMP for parallel layer execution. Essential for multi-core CPU inference. Confirm at runtime via the `system_info` line printed on startup (`OPENMP = 1`).

---

## 2. NUMA Configuration

Non-Uniform Memory Access (NUMA) matters on multi-socket systems.

> **Platform:** §§2–4 are Linux-only (`lscpu`, `numactl`, `taskset`, `ldconfig` / `LD_PRELOAD`).

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
| `--cpu-range` | `-Cr 0-7` | Contiguous CPU id range only |
| `--cpu-strict` | `--cpu-strict 1` | Strict placement |
| `--cpu-mask-batch` | `-Cb 0xFF` | Batch thread mask |
| `--cpu-range-batch` | `-Crb 0-7` | Batch thread range |

`--cpu-range` accepts a single `lo-hi` span. On SMT/HT, physical cores are often non-contiguous CPU ids — prefer discovering one logical CPU per physical `CORE` and binding with `taskset` (below).

### Physical Core Affinity

```bash
# Inspect topology
lscpu -e=CPU,CORE,SOCKET

# First N logical CPU ids with distinct physical CORE (skips SMT siblings)
N=8
CORES=$(lscpu -p=CPU,CORE | awk -F, -v n="$N" '
  !/^#/ && !seen[$2]++ { ids[++c]=$1 }
  END {
    if (c < n) { printf "need %d physical cores, found %d\n", n, c > "/dev/stderr"; exit 1 }
    for (i = 1; i <= n; i++) printf "%s%s", ids[i], (i < n ? "," : "")
    print ""
  }')

# Option A (preferred): bind at process start
taskset -c "$CORES" llama-server -m model.gguf -t "$N" --cpu-strict 1

# Option B: OS-level bind on one running PID (fails closed if 0 or >1 matches)
pid=$(pgrep -x llama-server)
case "$pid" in
  '') echo "llama-server not running" >&2; exit 1 ;;
  *[!0-9]*) echo "multiple llama-server PIDs: $pid" >&2; exit 1 ;;
esac
sudo taskset -cp "$CORES" "$pid"
```

---

## 4. Memory Allocators

Why and when: see [`advanced-inference-optimizations.md`](./advanced-inference-optimizations.md) §2. **Linux + glibc `ldconfig` only** — abort outside that environment (macOS / musl lack this lookup path):

```bash
command -v ldconfig >/dev/null || { echo "ldconfig required (Linux glibc)" >&2; exit 1; }
name=tcmalloc   # or: jemalloc
lib="$(ldconfig -p | awk -v n="$name" '$0 ~ "lib"n"\\.so"{print $NF; exit}')"
[ -n "$lib" ] || { echo "$name not found — install the matching package" >&2; exit 1; }
LD_PRELOAD="$lib" llama-server -m model.gguf -t 8
```

---

## 5. GGUF Quantization for CPU

| Type | BPW | CPU Recommendation |
|------|-----|-------------------|
| Q2_K | 2.56 | Lowest, fast |
| Q3_K_M | 3.35 | Low, compact |
| Q4_K_M | 4.35 | **Best balance** |
| Q5_K_M | 5.34 | Quality focus |
| Q6_K | 6.56 | Near-fp16 quality |
| Q8_0 | 8.50 | Highest, slowest |

**CPU memory bandwidth & L3 cache:** smaller quants (Q4_K_M, IQ4_XS) keep more layer weights in L3 during decode. Start with `Q4_K_M`. On dual-channel DDR4/DDR5, prefer `Q4_K_M` or `IQ4_XS` over `Q8_0` for TPS; use `Q5_K_M` when accuracy matters and bandwidth allows. Check L3 size with `lscpu` (or `/sys/devices/system/cpu/cpu0/cache/index*/size` on Linux).

# 2026-08-22 — Ornith NVFP4 vs GGUF Compare Setup + TPS Result (8 GB-class)

## Goal
Document that `Ornith-1.5-35B-A3B` `NVFP4` HF (`MIXED W4A16_NVFP4`) is `rejected` on the 8 GB-class SGLang/vLLM path while `GGUF Q4_K_M` remains `on_front` via `llama.cpp` `n-cpu-moe`; capture staged packs, measured TPS, and the VRAM guard.

## Hardware
- discrete_gpu, 8 GB VRAM class, Windows host, WSL2 Ubuntu 24.04
- Existing GGUF store already had `Ornith-1.5-35B-Q4_K_M.gguf` (21.7G, `block_count 41`, `n-cpu-moe 41`)
- Disk free before: ~48G on D:; after NVFP4: ~24G free (NVFP4 23.5G)

## Downloads (network only, no GPU)
- `hf download ornith-ai/Ornith-1.5-35B-A3B-NVFP4 --local-dir models/ornith-ai/Ornith-1.5-35B-A3B-NVFP4` — verified complete via `dry-run 0 files` 2026-08-22. Layout:
  - `model-00001-of-00003.safetensors` 9.4G
  - `model-00002-of-00003.safetensors` 9.4G
  - `model-00003-of-00003.safetensors` 3.2G
  - `hf_quant_config.json` `quant_algo: MIXED_PRECISION`, `kv_cache_quant_algo: FP8`, layers: `mlp.experts / shared_expert.*` = `W4A16_NVFP4 group 16`, `linear_attn.*` = `FP8`
- Existing GGUF untouched: `models/ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf` (21G)
- No model execution; no `llama-server` / `vllm` spawn; `nvidia-smi` untouched.

## Config prepared (not applied)
### A — GGUF path (llama.cpp, existing harness)
No Baseline change applied now. Staged Baseline for later Trial (when GPU free):

```python
ENGINE_DEFAULTS = {
 'MODEL': 'Ornith-1.5-35B-Q4_K_M.gguf',
 'CTX_SIZE': 65536,  # Night floor; Day also test 32768
 'N_GPU_LAYERS': -1,
 'KV_CACHE_K': 'q4_0',
 'KV_CACHE_V': 'q4_0',
 'BATCH_SIZE': 512,
 'UBATCH_SIZE': 128,
 'FLASH_ATTN': 'on',
 'N_CPU_MOE': None,  # auto block_count (40 layers)
 'VRAM_LIMIT_MB': 7900,
 'TPS_FLOOR': 20.0,
}
SAMPLER_DEFAULTS = {
 # from Ornith-1.5 card + thinking: temp 0.6 / top_p 0.95 / top_k 20 for agentic; temp 1.0 for bench repro
 'TEMP': 0.6, 'TOP_P': 0.95, 'TOP_K': 20, 'MIN_P': 0.05,
 'REPEAT_PENALTY': 1.0,
}
# + REASONING_PRESERVE=True via /props supports_preserve_reasoning (see 2026-08-19 log)
```
Run when free: `.\venv\Scripts\python.exe benchmark_search.py --agentic-full --desc "trial Ornith-1.5-35B-Q4_K_M"` (or `--validation` for smoke).

### B — NVFP4 path (vLLM, WSL2, Marlin W4A16 fallback on SM89)
Not Blackwell-native W4A4; on RTX 4060 it will use `MarlinNvFp4LinearKernel` W4A16 (memory win, no FP4 speed). Requires `float16` activations (`BF16+Marlin` garbled per vllm#34694).

**Env separation (do not reuse `venv-sglang`):** `venv` = Windows llama.cpp harness. `venv-sglang` = WSL2 SGLang (0.5.18, sgl-kernel/flashinfer/DeepGEMM) — do not pollute with vLLM. Create `venv-vllm` for NVFP4.

WSL2 install (once, no GPU):
```bash
# WSL2 Ubuntu 24.04, repo root
python3 -m venv venv-vllm
source venv-vllm/bin/activate
pip install -U pip
pip install "vllm>=0.10" "transformers>=5.8.1" huggingface_hub
# optional: pip install flashinfer  # vLLM pulls its own kernels
```

Launch (when GPU free, no --quantization flag needed — auto from hf_quant_config.json):
```bash
# WSL2, from repo root
source venv-vllm/bin/activate
VLLM_NVFP4_GEMM_BACKEND=marlin \
python -m vllm.entrypoints.openai.api_server \
  --model models/ornith-ai/Ornith-1.5-35B-A3B-NVFP4 \
  --served-model-name Ornith-1.5-35B-A3B-NVFP4 \
  --dtype float16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --trust-remote-code
```

Alternative SGLang (separate env, untouched):
```bash
source venv-sglang/bin/activate
python -m sglang.launch_server --model-path models/ornith-ai/Ornith-1.5-35B-A3B-NVFP4 --quantization modelopt_fp4 --context-length 65536 --mem-fraction-static 0.90 --trust-remote-code
## Errors / notes
- First `hf download` timed out at 300s (23.5G); resumed with 1800s timeout, completed. Dry-run now 0 files.
- `venv` (Windows) has no `vllm`/`sglang` — expected; WSL `venv-sglang` is SGLang-only per docs/sessions/2026-08-18-issue-59-sglang.md; `venv-vllm` created empty, **not installed yet** (no pip run per your GPU-in-use request — scaffold only).
- No Baseline edit, no server spawn per user request.

## Decisions
- Keep both packs on disk; do not delete GGUF root.
- Stage comparison via this session log; actual Trials on operator go-ahead when GPU idle.

## References
- HF repos: `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` (safetensors, 23.5G) and `ornith-ai/Ornith-1.5-35B-A3B-GGUF` (`Ornith-1.5-35B-Q4_K_M.gguf` 21.7G)
- Docs patched 2026-08-22: `docs/discovery/nvfp4-quantization.md`, `sglang-inference-engine.md`, `inference-engines-landscape.md`

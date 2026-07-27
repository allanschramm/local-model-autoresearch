# AutoResearch

Autonomous multi-objective Search that benchmarks local LLM runtime configurations and maintains a Pareto frontier of non-dominated Trials (context × throughput × agentic × coding) for a given hardware budget.

## Language

### Search Process

**Search**:
The overall optimization process. An indefinite sequence of Rounds that continues until manually stopped.
_Avoid_: loop, sweep, experiment

**Round**:
One iteration of the Search: evaluate Neighbors of the active Baseline until the per-model frontier stops improving or Neighbors are exhausted.
_Avoid_: step, iteration, cycle

**Trial**:
One execution of chosen benchmarks against a single Fingerprint. The atomic unit of work. May be partial (subset of axes) or complete (all four Objective Vector axes measured).
_Avoid_: run, evaluation, pass, execution

**Objective Vector**:
The four maximize axes of a Trial: configured context (`CTX_SIZE`), TPS, agentic (Claw-Eval full), coding (coding-10). Domination compares Objective Vectors.
_Avoid_: Val Score, blended intelligence, single score

**Pareto Set**:
The set of Trials whose Objective Vectors are not dominated. Global ranking is the union of per-model fronts for a hardware+budget identity; Neighbor generation stays inside one model.
_Avoid_: leaderboard winner, single champion, keep list

**Domination**:
Trial A dominates B when A is ≥ on every Objective Vector axis and > on at least one. Dominated Trials are not failures — a better tradeoff exists on the front.
_Avoid_: discard, reject, worse score

**Fingerprint**:
Identity of a configuration for merge and frontier membership: the full `ENGINE_DEFAULTS` + `SAMPLER_DEFAULTS` used for the Trial (model, ctx, KV, batch/threads, MTP/spec, offload, sampler, …).
_Avoid_: model-only key, engine-only key

**Usage Profile**:
A selection lens over the Pareto Set, not a separate frontier. **Day** (supervised): among points with `min(agentic, coding) ≥ DAY_IQ_RATIO × max(min on the set)`, maximize TPS; ties → higher `min`, then ctx (default `DAY_IQ_RATIO=0.75`). Empty band → max `min(agentic, coding)` then TPS. **Night** requires `CTX_SIZE ≥ NIGHT_CTX_FLOOR` then max `min(agentic, coding)`, with fallback to max ctx if none qualify (unsupervised long loops).
_Avoid_: separate day/night frontiers, Day = pure max TPS, Day = speed-band-first, TPS Floor as keep rule

**DAY_IQ_RATIO**:
Fraction of the Pareto Set’s best `min(agentic, coding)` required to enter the Day IQ band (default 0.75). Day then maximizes TPS inside that band. Raise toward 0.8 when Day work is as quality-sensitive as Night.
_Avoid_: absolute IQ floor, Day = max TPS, Day = speed band first

**DAY_TPS_RATIO**:
Deprecated Day gate from ADR 0007 (speed band first). Superseded by `DAY_IQ_RATIO` ([ADR 0008](docs/adr/0008-day-iq-epsilon-then-tps.md)).
_Avoid_: using as current Day rule

**NIGHT_CTX_FLOOR**:
Minimum configured `CTX_SIZE` for Night profile selection (default 65536). Revisitable when project architecture / ticket size / compaction change how much context night loops need.
_Avoid_: ctx axis, hard reject below floor

**Local Maxima**:
A state where all valid Neighbors from the active Baseline have been evaluated and none join or improve the per-model Pareto Set.
_Avoid_: stuck state, convergence

**Random Restart**:
The mechanism used to escape a Local Maxima. Generates a random configuration far from the active Baseline that isn't in visited memory, sets it as the new Baseline, and resumes the Search.
_Avoid_: random jump, memory wipe

**SearchStrategy**:
A deep module encapsulating Neighbor generation, Pareto Set updates, Usage Profile Baseline pick, and Random Restarts across Search Spaces.
_Avoid_: heuristic loop, search script, Pareto Tie-Breaker

### Configuration

**Baseline**:
The active Neighbor origin for Search — a Fingerprint persisted in `autoresearch/core/config.py` (`ENGINE_DEFAULTS` / `SAMPLER_DEFAULTS`). Chosen by Usage Profile (or manual override) from the per-model front; not “the single best config”. `.autoresearch_state.json` holds visited memory only.
_Avoid_: sole champion, default, current config, state baseline

**Neighbor**:
A configuration derived from the Baseline by changing exactly one parameter. The Search evaluates Neighbors to grow or improve the per-model Pareto Set.
_Avoid_: candidate, variant, mutation

**Search Space**:
The set of parameters and their candidate values that the Search explores. Defines which Neighbors are reachable from any Baseline. Neighbors do not jump across models.
_Avoid_: grid, parameter space

### Evaluation

**Validation**:
The pre-check before an expensive Trial: (1) local backend throughput validation, then (2) Claw-Eval quick. Optional direct-coding preflight always uses exactly 10 tasks per dataset. The `--validation` flag runs throughput plus Claw-Eval quick and exits.

**To validate a single model**: (1) set `MODEL` (and other flags) in `config.py` Baseline, (2) run `python3 benchmark_search.py --validation --desc "validate <model>"` with no CLI flag soup. One model at a time — never parallel. See GOLDEN-RULES.md §5 for the full step-by-step.
_Avoid_: bench-only, speed check, smoke test

**Trial Status**:
Canonical outcome labels: `on_front` (complete vector, non-dominated), `dominated` (complete vector, dominated), `incomplete` (missing axes; may merge into a Fingerprint), `rejected` (invalid config, infra/VRAM kill, crash). 
_Avoid_: keep, discard (legacy; `keep` may alias `on_front` only during migration)

**Val Score**:
Legacy scalar (historically Claw-Eval full) retained for display/compat only. Not the Search keep rule. Prefer the Objective Vector.
_Avoid_: score, result, metric (when meaning frontier truth)

**TPS Floor**:
Legacy minimum throughput knob in Baseline `ENGINE_DEFAULTS['TPS_FLOOR']`. Does **not** gate Pareto Set membership. Day selection uses relative `DAY_IQ_RATIO` ([ADR 0008](docs/adr/0008-day-iq-epsilon-then-tps.md)), not an absolute tok/s floor. Removable once callers stop depending on it.
_Avoid_: threshold as frontier rule, minimum TPS for on_front, Day = TPS Floor

### Benchmarks

**Nexus**:
Retrieval benchmark. Tests context-stress with synthetic history — the model must find a needle in a haystack of padding.
_Avoid_: retrieval, context stress

**Claw**:
Agency benchmark. Tests tool-use (JSON browser calls) and instruction-following. Claw-Eval full supplies the **agentic** Objective Vector axis; quick remains observational smoke.
_Avoid_: agency, ClawBench, tool-use benchmark

**Coding**:
Benchmark using LiveCodeBench v6, HumanEval+, MBPP+, and BigCodeBench Hard (exactly 10 tasks per dataset when enabled). Supplies the **coding** Objective Vector axis.
_Avoid_: EvalPlus, HumanEval

### Runtime

**ServerIntent**:
A pure data object describing the full configuration for a Trial — model path, context size, KV cache types, threads, speculative draft tokens, etc.
_Avoid_: config object, server config

**SGLang Backend**:
Directory model paths under `models/` are served through SGLang instead of `llama-server`. SGLang Trials still flow through the harness, run the same Coding benchmark, and use the configured CTX_SIZE.
_Avoid_: raw SGLang run, direct server launch

**TurboQuant**:
Hardware-accelerated KV cache compression formats (`turbo2`, `turbo3`, `turbo4`) that fit large contexts within tight VRAM budgets.
_Avoid_: quantized cache, compressed KV

**Multi-Token Prediction (MTP)**:
Speculative decoding using specialized draft heads (built into the model) to predict multiple tokens ahead, improving throughput. Distinct from "speculative decoding with separate draft model", which fails on MoE+SSM models. MTP is a Search dial (buy ctx or TPS), not a required default.
_Avoid_: speculative decoding (when referring specifically to MTP)

### Generic Configuration Skeleton

The example below uses portable placeholders. Replace with your actual paths/values.

**Model file**: place in `models/` (relative to repo root), e.g. `models/<model-filename>.gguf`. Symlinks to absolute paths under your home directory are also OK.

**Working llama-server command template**:
```
llama-server \
  -m models/<model-filename>.gguf \
  --host 0.0.0.0 --port 8083 \
  -ngl 999 \
  --n-cpu-moe 32 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -c 8192 \
  --override-tensor "v\\..*=CPU" \
  --flash-attn on --no-warmup
```

**MTP flags (only when the GGUF was downloaded from the `-MTP-GGUF` repo variant)**:
```
--spec-type mtp --spec-draft-n-max 2
```
Notes:
- Turboquant and similar forks accept `--spec-type mtp` (NOT `draft-mtp`).
- Upstream `ggml-org/llama.cpp` accepts `--spec-type draft-mtp`.
- `scripts/setup-check.sh` probes your build's `--help` and validates compatibility.
- MTP adds ~1 GB VRAM headroom. Speedup is **1.15–1.25× for MoE**, **1.4–2.0× for dense**.

**Key flags explained**:
- `-ngl 999`: lets auto-fit adjust GPU layers. Avoid combining with explicit `--n-cpu-moe` smaller than auto-fit targets.
- `--n-cpu-moe 32`: first 32 layers' MoE experts on CPU, remaining layers' experts on GPU. Adjust based on your MoE layer count.
- `--override-tensor "v\\..*=CPU"`: value projection weights forced to CPU (saves ~500MB VRAM on multimodal GGUFs).
- `--no-warmup`: required to avoid OOM during empty-run warmup on tight VRAM (e.g., 8GB).
- `--parallel 1`: reduces RS buffer (recurrent state for delta net) significantly on hybrid architectures.

**Performance expectations** (illustrative — depends on hardware + flags):
- q4_0 KV at 8k ctx on RTX 4060 8GB with MoE offload: ~11 tok/s (no MTP), ~13 tok/s (with MTP).
- TurboQuant does not always help — on GQA 8:1 architectures, `turbo4` K cache auto-upgrades to `q8_0` with no speed/VRAM gain over plain `q4_0`. Run `whichllm plan` to inspect your specific model.

**Filesystem caveats**:
- Models on 9p bridges (e.g. `/mnt/c/...`, `/mnt/d/...`) load very slowly via `mmap` (10–50× slower than native ext4). Copy or symlink model files into `models/` (native ext4) for normal speed.
- For WSL2: ensure `vm.overcommit_memory=1` and ample WSL `.wslconfig` memory (≥24 GB) when serving 20+ GB models.

## Discovery Workflow (cross-reference)

For users selecting which model to autotune, see [`docs/discovery/discover-models.md`](docs/discovery/discover-models.md). It documents the **whichllm → Pareto Set → Baseline handoff** flow. Canonical frontier rules: [ADR 0006](docs/adr/0006-pareto-frontier-search.md); Day/Night pick: [ADR 0008](docs/adr/0008-day-iq-epsilon-then-tps.md) + [pareto-selection.md](docs/discovery/pareto-selection.md).

## Cached lessons (general, not user-specific)

- **MoE offload is mandatory on 8GB VRAM**: without explicit `--n-cpu-moe`, auto-fit can put 36/42 layers with GATE overflow, dropping throughput to ~0.7 tok/s. Always pair `--n-cpu-moe` with explicit `--override-tensor`.
- **Speculative decoding with separate draft models fails on MoE+SSM**: verification becomes PCIe-bound (MoE expert fetch per token) and SSM layers can't parallelize across a draft window. MTP is a different mechanism and works.
- **`whichllm` score ≠ coding benchmark**: whichllm blends AA Intelligence Index, Aider, LiveBench (intelligence weighted). For Claude Code / Pi Agent loops, cross-reference SWE-bench Verified — Gemma-4-26B-A4B ranks top in whichllm but scores only ~17% on SWE-bench Verified (bad coding agent despite high general intelligence).
- **Pareto Set beats "highest score"**: pick a point on the frontier with a Usage Profile (Day/Night), not the highest single-axis leader. Day = IQ ε-band then max TPS ([ADR 0008](docs/adr/0008-day-iq-epsilon-then-tps.md)); Night = ctx floor then maximin IQ.
- **Configured ctx is the ctx axis**: `llama-server` reserves KV for full `CTX_SIZE` even when a short coding prompt uses few tokens. Night loops that fill 65k+ need that reservation; coding-10 alone does not prove lung capacity.

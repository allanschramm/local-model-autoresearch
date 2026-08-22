# ADR 0016 — Measurement hygiene and Morris screen

- **Status:** Accepted
- **Date:** 2026-08-21
- **Context:** llama-optimize-style ideas (thermal settle, median TPS, crash skip, Morris elementary effects) can cut noisy benches and reboot loops without replacing Pareto Neighbor Search (ADR 0006). Taguchi / orthogonal-array Search and a `robust` submodule are out of scope.

## Decision

1. **Pareto hill-climb is unchanged.** `SearchStrategy.get_neighbors` still emits one-parameter Neighbors. Morris is a **pre-Round Search Space pin** on engine knobs (`ENGINE_DEFAULTS` keys only), using llama-cli TPS as y. Sampler keys are never screened. `--no-screen` / `--mode quality` skip it. Pins persist in `.autoresearch_state.json` schema 3 (`morris`). `--reset-visited` clears pins so a fresh Search re-screens.
2. **Thermal settle + TPS median.** Capture idle GPU temp once on `ExperimentRunner`. Before each llama-cli / SGLang bench rep and before server spawn, wait until temp is near idle (`idle + 8°C`, 90s timeout). Bench TPS is `statistics.median` of `TPS_REPS` (default 3). Coding-generation TPS is unchanged. New TSV columns: `gpu_temp_c` (not sampler `temp`), `tps_reps`, `tps_spread`. `schema_version` stays `"2"` (additive columns).
3. **Crash journal.** Gitignored `.autoresearch_crash.journal` next to visited state. Autoloop writes it immediately before a Neighbor/Baseline `run_trial` and clears it in `finally`. On next process start, consume as `rejected` / `outcome=CRASH` unless `--retry-crashed`. `--validation` single shots do not journal.
4. **No Taguchi, no `robust` submodule, no Neighbor replacement.**

## Consequences

- Engine Search Space can shrink per model after a cheap Morris screen (`reps=1`); full Trials still use median-of-`TPS_REPS`.
- A config that reboots the host is skipped on resume (visited + rejected row).
- v1/v2 state files still load; `morris` defaults to `{}`.
- Operators who find Morris too expensive use `--no-screen`.
- Screen rows (`evaluation_profile=morris-screen`) are excluded at every
  competition seam — `rank_results.build_vectors` TPS/ctx collection,
  `recompute_rows` group formation, `classify._known_vectors`, and autoloop's
  `_seed_known_vectors` — so a reps=1 probe never sets the basename TPS axis
  or seeds/demotes any front.

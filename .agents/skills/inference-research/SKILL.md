---
name: inference-research
description: >
  Research inference-performance levers — llama.cpp flags and features,
  inference engines and forks, quantization and KV-cache techniques,
  speculative decoding, context optimization — on the operator rig, and turn
  findings into falsifiable Search neighbors. Scope is engine-side only: this
  skill does NOT search for or shortlist new models (that is discover-models /
  the Trial skill). Use whenever the user asks to research inference, find
  flags/engines/techniques to raise TPS/ctx or cut VRAM, check whether "X is
  worth trying" on this rig, or verify a vendor claim against local
  measurement — including questions about dflash/dspark, MTP, speculative
  decoding, KV cache, quantization, ngram, or a specific engine. Workflow:
  read repo docs → research primary sources → land discoveries in docs →
  commit.
---

# inference-research

Investigate an inference-performance question against primary sources, then
land the finding in one of three buckets: a candidate **Neighbor** (one
Fingerprint, engine-side), a **TBD:** probe, or a **dead end** with measured
evidence. IQ is guarded by measurement, never assumed. The IQ lever — model
choice — lives in `docs/discovery/discover-models.md` and the `trial` skill;
this skill feeds only the engine side of a chosen model.

## Scope

**In:** engine flags/features on the pinned build; inference engines and
forks; quantization formats and KV-cache techniques; speculative decoding
(MTP, ngram, DFlash/DSpark, Eagle); context optimization (ctx axis, KV
compression, compaction); hardware-specific tuning for the 8 GB-class rig
(Baseline `VRAM_LIMIT_MB`); vendor-claim verification.

**Out (do not drift):** new model search / candidate shortlists (`whichllm`,
HF browsing for models → `discover-models.md`); harness-side IQ extraction
(parked until a harness fork); training or fine-tuning; running Trials
(`trial` skill); validation (`validation` skill); operator-only autoloop.

## Method

Workflow: **read repo docs → research → land discoveries in docs →
commit.** Each phase below.

1. **Read repo docs first.** Walk the repo's own knowledge before touching
   external sources: root `AGENTS.md` + `CONTEXT.md` (domain language,
   Fingerprint/Trial contracts), `results.tsv` (measured history),
   `docs/sessions/` and `docs/discovery/` (prior findings and dead ends),
   `models/TASK.md` (open queue), and the pinned-build notes
   (`docs/llamacpp-toolset.md`, `docs/models/` cards). A measured dead end
   stays dead unless the Fingerprint or engine changed. Do not re-research
   what the repo already knows.
2. **Research primary sources.** Official docs, llama.cpp source and
   `--help`, HF model cards + API (`/api/models/<org>/<repo>/tree/main` for
   exact file sizes, `gguf` metadata fields), vendor papers and blogs.
   Follow each claim to the source that owns it. Reference external sources
   by URL, never copy-paste.
3. **Ground on this rig.** A vendor claim is marketing-grade until
   measured. Probe flag support against the pinned engine
   (`scripts/setup-check.sh` reads `--help`). Use venv Python + harness
   modules for file facts (`gguf_has_mtp`, `gguf_kv_f16_mb`, `is_moe_model`,
   `resolve_model_file`) — never guess arch or size from filenames.
4. **Land every finding in one bucket:**
   - **Neighbor** — an engine-side change, falsifiable as one Fingerprint
     (same model, same tasks, switch on/off; e.g. a flag, KV format, spec
     type, ctx size).
   - **TBD probe** — flag support or claim not yet verifiable on the pinned
     build; mark **TBD:** and keep it listed in the doc's Open questions.
   - **Dead end** — with measured evidence; do not re-run without new
     evidence or a changed Fingerprint.
5. **Land discoveries in docs.** Durable technique facts →
   `docs/discovery/` (stable contracts only). Single-day capture →
   `docs/sessions/YYYY-MM-DD-<topic>.md` following the sessions schema
   (Goal/Hardware/Findings/Decisions/Open questions; hardware-class
   language; no SKU, no absolute paths, no PII; cross-link related logs).
   Actionable queue → `models/TASK.md` checklist items with evidence links.
   Session logs are not edited after completion; add a follow-up file for
   new work.
6. **Commit the discoveries.** Commit the docs and queue changes from this
   pass only: `docs/discovery/`, `docs/sessions/`, `models/TASK.md`. Never
   commit `results.tsv`, `models/aliases/`, machine Baseline, private
   paths, or harness code — a research pass changes docs and the queue
   only.
7. **Constraints.** Investigation is read-only: no Trials, downloads, or
   `config.py` edits without operator go-ahead. One command at a time, no
   execution timeouts, venv Python only, never autoloop. Creating a new
   file (doc or skill) requires operator confirmation.

## Frame the answer

Verdict first, then evidence, then the falsifiable experiment, then the
rank. State uncertainty at the claim and name the tradeoff; pick the boring
option when evidence is thin. Link measured numbers (TSV rows, session
logs) instead of repeating them. Mark anything unverified as
**unverified** — an edit or assumption is not proof.

## Anti-patterns (learned in this repo)

- Re-running a measured dead end on the same Fingerprint (e.g. DFlash/DSpark
  on MoE + `n-cpu-moe`; separate draft models on hybrid SSM).
- Trusting vendor TPS/IQ numbers over the repo's own measurements.
- Answering with a model pick from this skill — model selection is out of
  scope.
- Editing code while investigating; a research pass changes docs and the
  queue only.
- Researching without the repo-docs read first — re-discovering ground the
  repo already measured.
- Committing more than the discoveries with a research pass (`results.tsv`,
  `models/aliases/`, machine Baseline, harness code) — the commit covers
  docs and the queue only.

## Verification

- Every claim cites its source: a URL or a measured Trial/session log.
- Vendor claims are labeled unverified until a local measurement exists.
- TBD items appear as **TBD:** and live in the doc's Open questions until
  resolved.
- A research pass ends committed: the discoveries (`docs/discovery/`,
  `docs/sessions/`, `models/TASK.md`) land in git, and nothing else
  (`results.tsv`, aliases, Baseline, code stay out).

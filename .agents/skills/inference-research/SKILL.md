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
  worth trying" on this rig, or verify a vendor claim against primary sources
  and documented measurements — including questions about dflash/dspark, MTP,
  speculative decoding, KV cache, quantization, ngram, or a specific engine.
  Pure desk research: read repo + web/source research only — never probe, run,
  or download. Workflow: read repo docs → walk its open questions → research
  what the repo cannot answer → land discoveries in docs → commit on operator
  go-ahead.
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

Workflow: **read repo docs → walk open questions → research what the repo
cannot answer → land discoveries in docs → commit on operator go-ahead.**
Desk research only: no commands executed, no engine probes, no downloads, no
Trials. Each phase below.

1. **Read repo docs first.** Walk the repo's own knowledge before touching
   external sources: root `AGENTS.md` + `CONTEXT.md` (domain language,
   Fingerprint/Trial contracts), `results.tsv` (measured history),
   `docs/sessions/` and `docs/discovery/` (prior findings, TBDs, dead ends),
   and the pinned-build notes (`docs/llamacpp-toolset.md`, `docs/models/`
   cards). A measured dead end stays dead unless the Fingerprint or engine
   changed. Do not re-research what the repo already knows.
2. **Walk the open questions.** Collect every `**TBD:**` / open-question item
   in the repo docs. For each, apply the clarity gate: *"Is it clear how X
   works and how it affects inference?"* Answer **yes** from repo knowledge →
   state it and move to the next item. Answer **no** → that item is a
   research target.
3. **Research primary sources.** Official docs, llama.cpp source (vendored
   `llama.cpp/` tree and upstream master), GitHub PR/issue API
   (`api.github.com`), HF model cards + API (tree, metadata, **discussions** —
   community measurements), vendor papers and blogs. Follow each claim to the
   source that owns it; date it; reference by URL, never copy-paste.
   External measurements are evidence with an explicit label, not proof for
   this rig.
4. **Land every finding in one bucket:**
   - **Neighbor** — an engine-side change, falsifiable as one Fingerprint
     (same model, same tasks, switch on/off; e.g. a flag, KV format, spec
     type, ctx size). A research pass states the prior from primary sources
     and repo measurements; only a later Trial closes it.
   - **TBD probe** — flag support or claim not verifiable without running the
     pinned engine; mark **TBD:** and keep it listed in the doc's Open
     questions. Resolving it is a separate, later pass (probe/Trial) — a
     research pass only sharpens the prior.
   - **Dead end** — with measured evidence (repo rows/session logs, or an
     external measurement cited with URL + date and labeled **external /
     unverified on this rig**); do not re-run without new evidence or a
     changed Fingerprint.
5. **Land discoveries in docs.** Prefer **patching existing** docs. Durable
   technique facts → `docs/discovery/` (stable contracts only). Resolve TBDs
   with evidence or sharpen their prior; never silently drop an open
   question. Single-day capture → `docs/sessions/YYYY-MM-DD-<topic>.md`
   following the sessions schema (Goal/Hardware/Findings/Decisions/Open
   questions; hardware-class language; no SKU, no absolute paths, no PII;
   cross-link related logs). Session logs are not edited after completion;
   add a follow-up file for new work.
6. **Commit the discoveries — on operator go-ahead.** Prepare a docs-only
   commit (`docs/discovery/`, `docs/sessions/`, skill edits) and ask for the
   explicit command before committing. Never commit `results.tsv`,
   `models/aliases/`, machine Baseline, private paths, or harness code —
   a research pass changes docs only.
7. **Constraints.** Pure desk research: no shell commands, no engine probes
   (`--help`, `setup-check.sh`), no venv/harness module runs, no downloads,
   no Trials, no `config.py` edits. Investigation = `read` / `grep` / `glob`
   / `web_search` on repo files and URLs only. Creating a new file (doc or
   skill) requires operator confirmation. Never autoloop.

## Frame the answer

Answer per item: verdict first (clear from repo knowledge, or researched),
then evidence (URLs with dates, repo rows/session logs), then the
falsifiable experiment, then the rank. If the item was already clear in repo
docs, say so and move on — research only what the repo cannot answer. State
uncertainty at the claim and name the tradeoff; pick the boring option when
evidence is thin. Link measured numbers (TSV rows, session logs) instead of
repeating them. Mark anything unverified as **unverified** — an edit or an
external measurement is not local proof.

## Anti-patterns (learned in this repo)

- Executing anything during a research pass — engine probes (`--help`,
  `setup-check.sh`), venv/harness module runs, downloads, Trials. This skill
  is desk research; probing and measuring are separate passes.
- Answering a repo-doc question the docs already answer — run the clarity
  gate first; research only what is unclear.
- Treating external measurements as local proof — label them **external /
  unverified on this rig** even when N-reproducible on other hardware.
- Re-running a measured dead end on the same Fingerprint (e.g. DFlash/DSpark
  on MoE + `n-cpu-moe`; separate draft models on hybrid SSM).
- Trusting vendor TPS/IQ numbers over the repo's own measurements.
- Answering with a model pick from this skill — model selection is out of
  scope.
- Editing code while investigating; a research pass changes docs only.
- Researching without the repo-docs read first — re-discovering ground the
  repo already measured.
- Committing more than the discoveries with a research pass (`results.tsv`,
  `models/aliases/`, machine Baseline, harness code) — the commit covers
  docs only, and only on operator go-ahead.

## Verification

- Every claim cites its source: a URL with access/extraction date, or a repo
  session log / `results.tsv` row.
- External measurements are labeled **external / unverified on this rig**
  until a local measurement exists; vendor claims are labeled unverified.
- TBD items appear as **TBD:** and live in the doc's Open questions until
  resolved; evidence-backed resolution removes the marker, and
  measurement-only gaps stay open with the falsifiable experiment named.
- A research pass ends with docs updated and a docs-only commit prepared —
  the commit lands only on explicit operator command, and contains nothing
  else (`results.tsv`, aliases, Baseline, code stay out).

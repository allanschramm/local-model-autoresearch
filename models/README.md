# `models/` — Shared GGUF store (LM Studio + harness)

## Layout (canonical)

LM Studio needs `publisher/model/file.gguf`. This repo resolves by **basename**, so both work:

```
models/
  lmstudio-community/gemma-4-e4b-it-gguf/<file>.gguf   # LM Studio My Models
  local/<name>/<file>.gguf                             # other GGUFs
  draft/<file>.gguf                                    # speculative drafts (keep flat here)
  vision/mmproj-*.gguf                                 # multimodal projectors
  aliases/<name>/config.yaml                           # model-up launcher configs
```

Config / CLI still use short names: `MODEL=gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` — harness finds nested path via `resolve_model_path`.

## Download rules

**LM Studio:** downloads already land nested under this folder (set as downloads folder). Repo finds them by filename.

**Repo (`hf`):** put files under publisher/model, not the root:

```bash
hf download <org>/<repo> <file>.gguf --local-dir models/<publisher>/<model-name>
```

Example:

```bash
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf --local-dir models/lmstudio-community/qwen3.5-9b-gguf
```

Special dirs stay as-is: `--local-dir models/draft` for MTP/DFlash drafts; `--local-dir models/vision` for mmproj.

## Aliases (`model-up`)

**Local only — never commit.** Per-user `aliases/<name>/config.yaml`, optional `aliases/INDEX.md` / `REMOVED.md` for launcher notes, plus root `models/REMOVED.md` for SSD keep/delete inventory. Tracked docs carry **model** scores and Pareto picks (GGUF basename), not which alias name you use or what is on disk.

## Do not

- Drop main GGUFs in `models/` root (LM Studio will not list them).
- Put drafts under `publisher/model` unless you also update `SPEC_DRAFT_MODEL` paths.
- Commit alias names, ports, or personal `config.yaml` into the repo.

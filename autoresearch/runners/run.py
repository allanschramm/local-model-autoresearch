import argparse
import csv
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from autoresearch.benchmarks import bench_config, format_agentic_benchmarks, format_claw_tiers
from autoresearch.core import classify, config, engine_version_tag, recompute, resolve_llama_server
from autoresearch.runners.evaluation import ExperimentRunner, resolve_tps_floor

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR.parent.parent / "results.tsv"
MODELS_DIR = BASE_DIR.parent.parent / "models"

# ── Run categories for TSV ────────────────────────────────────────────
CATEGORY_VALIDATION = "validation"
CATEGORY_10_TASK = "10-task"
CATEGORY_FULL_SUITE = "full-suite"

# Trial Status (ADR 0006): on_front | dominated | incomplete | rejected.
# Legacy keep/discard are not accepted on write; migrate keep→on_front offline.
TRIAL_STATUSES = {"on_front", "dominated", "incomplete", "rejected"}

BASELINE_CLI_FLAGS = {
    "--model",
    "--kv",
    "--kv-k",
    "--cache-type-k",
    "-ctk",
    "--kv-v",
    "--cache-type-v",
    "-ctv",
    "--max-tokens",
    "--ctx-size",
    "-c",
    "--threads",
    "-t",
    "--threads-batch",
    "--n-cpu-moe",
    "-ncmoe",
    "--ngl",
    "--n-gpu-layers",
    "-ngl",
    "--parallel",
    "--context-tokens",
    "--batch-size",
    "-b",
    "--ubatch-size",
    "-ub",
    "--flash-attn",
    "-fa",
    "--spec-type",
    "--spec-draft-n-max",
    "--spec-draft-model",
    "--no-mmap",
    "--jinja",
    "--reasoning-budget",
    "--reasoning-budget-message",
    "--reasoning",
    "--cont-batching",
    "--temp",
    "--top-p",
    "--min-p",
    "--top-k",
    "--repeat-penalty",
    "--presence-penalty",
    "--frequency-penalty",
    "--coding-task-limit",
    "--lcb-task-limit",
    "--bigcode-task-limit",
    "--bench-tts-threshold",
}


def determine_category(args) -> str:
    """Infer run category from CLI args."""
    if getattr(args, "validation", False):
        return CATEGORY_VALIDATION
    if getattr(args, "agentic_full", False):
        return "agentic-full"
    if getattr(args, "agentic_quick", False):
        return "agentic-quick"
    coding = getattr(args, "coding_task_limit", 10)
    lcb = getattr(args, "lcb_task_limit", 10)
    bigcode = getattr(args, "bigcode_task_limit", 10)
    if coding <= 10 and lcb <= 10 and bigcode <= 10:
        return CATEGORY_10_TASK
    return CATEGORY_FULL_SUITE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified AutoResearch Benchmark Runner", allow_abbrev=False
    )
    parser.add_argument(
        "--desc", type=str, help="Description of the experiment (required for logging single runs)"
    )
    parser.add_argument(
        "--model", type=str, default=config.MODEL, help="Model filename in models/ directory"
    )
    parser.add_argument(
        "--kv", type=str, default=config.KV_CACHE, help="KV cache type (e.g. q4_0, q4_1, f16)"
    )
    parser.add_argument(
        "--kv-k",
        "--cache-type-k",
        "-ctk",
        dest="kv_k",
        type=str,
        default=config.KV_CACHE_K,
        help="Key cache type (overrides --kv if set)",
    )
    parser.add_argument(
        "--kv-v",
        "--cache-type-v",
        "-ctv",
        dest="kv_v",
        type=str,
        default=config.KV_CACHE_V,
        help="Value cache type (overrides --kv if set)",
    )
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generation tokens")
    parser.add_argument("--ctx-size", "-c", type=int, default=config.CTX_SIZE, help="Context size")
    parser.add_argument("--port", type=int, default=18080, help="Port for llama-server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for llama-server")
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=config.THREADS,
        help="Number of threads for llama-server",
    )
    parser.add_argument(
        "--threads-batch",
        type=int,
        default=config.THREADS_BATCH,
        help="Number of batch/prefill threads for llama-server",
    )
    parser.add_argument(
        "--ngl",
        "--n-gpu-layers",
        "-ngl",
        type=int,
        default=99,
        help="Number of GPU layers to offload",
    )
    parser.add_argument(
        "--n-cpu-moe",
        "-ncmoe",
        type=int,
        default=getattr(config, "N_CPU_MOE", None),
        help="Keep MoE expert weights of first N layers on CPU (VITRIOL)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size for llama-server",
    )
    parser.add_argument(
        "--ubatch-size",
        "-ub",
        type=int,
        default=config.UBATCH_SIZE,
        help="Micro-batch size for llama-server",
    )
    parser.add_argument("--parallel", type=int, default=1, help="Parallel slots count")
    parser.add_argument(
        "--flash-attn",
        "-fa",
        nargs="?",
        const="on",
        default=config.FLASH_ATTN,
        choices=["on", "off", "auto"],
        help="Enable/disable/auto Flash Attention",
    )
    parser.add_argument(
        "--spec-type",
        type=str,
        default=config.SPEC_TYPE,
        help="Speculative decoding type (e.g. draft-mtp, mtp)",
    )
    parser.add_argument(
        "--spec-draft-n-max",
        type=int,
        default=config.SPEC_DRAFT_N_MAX,
        help="Speculative draft max tokens count for MTP",
    )
    parser.add_argument(
        "--spec-draft-model",
        type=str,
        default=config.SPEC_DRAFT_MODEL,
        help="Speculative draft model filename/path",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=config.CTX_SIZE,
        help="Context tokens padding length (100k minimum)",
    )
    parser.add_argument(
        "--include-coding",
        action="store_true",
        default=getattr(bench_config, "INCLUDE_CODING", False),
        help="Run the optional 10-task direct-coding preflight",
    )
    parser.add_argument(
        "--no-coding", dest="include_coding", action="store_false", help="Disable Coding benchmark"
    )
    parser.add_argument(
        "--agentic-quick",
        action=argparse.BooleanOptionalAction,
        default=getattr(bench_config, "INCLUDE_AGENTIC_QUICK", False),
        help="Run Claw-Eval quick tier smoke (5 tasks; use --no-agentic-quick to disable)",
    )
    parser.add_argument(
        "--agentic-full",
        action=argparse.BooleanOptionalAction,
        default=getattr(bench_config, "INCLUDE_AGENTIC_FULL", False),
        help="Run Claw-Eval full tier quality gate (15 tasks; use --no-agentic-full to disable)",
    )
    parser.add_argument(
        "--list-agentic-benchmarks",
        action="store_true",
        help="List long-horizon agentic benchmark targets and exit",
    )
    parser.add_argument(
        "--list-claw-tiers",
        action="store_true",
        help="List Claw-Eval quick/full task tiers and exit",
    )
    parser.add_argument(
        "--coding-task-limit",
        type=int,
        default=getattr(bench_config, "CODING_TASK_LIMIT", 30),
        help="Tasks per dataset (0=full dataset)",
    )
    parser.add_argument(
        "--lcb-task-limit",
        type=int,
        default=getattr(bench_config, "LCB_TASK_LIMIT", 10),
        help="LiveCodeBench task limit",
    )
    parser.add_argument(
        "--bigcode-task-limit",
        type=int,
        default=getattr(bench_config, "BIGCODE_TASK_LIMIT", 10),
        help="BigCodeBench task limit",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Validation mode: run llama-bench + Claw quick smoke evaluation. "
        "Validates model load, throughput, and basic agentic behavior. "
        "No extended eval, no keep/discard. Useful for quick config sanity checks.",
    )
    parser.add_argument(
        "--bench-tts-threshold",
        type=float,
        default=resolve_tps_floor(),
        help="Minimum text generation t/s (default: Baseline TPS_FLOOR; config.py-only)",
    )
    parser.add_argument(
        "--no-mmap", action="store_true", default=config.NO_MMAP, help="Disable mmap"
    )
    parser.add_argument(
        "--jinja",
        action="store_true",
        default=config.JINJA,
        help="Enable Jinja chat template engine",
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=config.REASONING_BUDGET,
        help="Thinking budget tokens limit",
    )
    parser.add_argument(
        "--reasoning-budget-message",
        type=str,
        default=config.REASONING_BUDGET_MESSAGE,
        help="Message on thinking budget exhaust",
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        choices=["on", "off", "auto"],
        default=config.REASONING,
        help="Reasoning mode (on/off/auto)",
    )
    parser.add_argument(
        "--cont-batching",
        action="store_true",
        default=config.CONT_BATCHING,
        help="Enable continuous batching",
    )
    parser.add_argument("--temp", type=float, default=config.TEMP, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=config.TOP_P, help="Top-p sampling")
    parser.add_argument("--min-p", type=float, default=config.MIN_P, help="Min-p sampling")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help="Top-k sampling")
    parser.add_argument(
        "--repeat-penalty", type=float, default=config.REPEAT_PENALTY, help="Repeat penalty"
    )
    parser.add_argument(
        "--presence-penalty", type=float, default=config.PRESENCE_PENALTY, help="Presence penalty"
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=config.FREQUENCY_PENALTY,
        help="Frequency penalty",
    )

    for action in parser._actions:
        if BASELINE_CLI_FLAGS.intersection(action.option_strings):
            action.help = argparse.SUPPRESS

    forbidden = [
        arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.split("=", 1)[0] in BASELINE_CLI_FLAGS
    ]
    if forbidden:
        parser.error(f"Baseline flags are config.py-only: {', '.join(forbidden)}")
    return parser.parse_args()


def tsv_fields_from_cfg(baseline: dict[str, Any]) -> dict[str, Any]:
    """Map a Baseline-style config dict to write_row flat-column kwargs."""
    recorded = {key.lower(): value for key, value in baseline.items()}
    return {
        "model": baseline.get("MODEL", ""),
        "kv": baseline.get("KV_CACHE", ""),
        "ctx": baseline.get("CTX_SIZE"),
        "threads": baseline.get("THREADS"),
        "threads_batch": baseline.get("THREADS_BATCH"),
        "batch_size": baseline.get("BATCH_SIZE"),
        "ubatch_size": baseline.get("UBATCH_SIZE"),
        "n_cpu_moe": baseline.get("N_CPU_MOE"),
        "temp": baseline.get("TEMP"),
        "top_p": baseline.get("TOP_P"),
        "top_k": baseline.get("TOP_K"),
        "min_p": baseline.get("MIN_P"),
        "repeat_penalty": baseline.get("REPEAT_PENALTY"),
        "presence_penalty": baseline.get("PRESENCE_PENALTY"),
        "cont_batching": baseline.get("CONT_BATCHING"),
        "flash_attn": baseline.get("FLASH_ATTN", ""),
        "no_mmap": baseline.get("NO_MMAP"),
        "spec_draft_n_max": baseline.get("SPEC_DRAFT_N_MAX"),
        "config_json": json.dumps(recorded, separators=(",", ":"), sort_keys=True),
    }


def _result_config() -> dict[str, Any]:
    return tsv_fields_from_cfg(config.load_config())


def get_git_commit() -> str:
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        if status:
            commit += "-dirty"
        return commit
    except Exception:
        return "unknown"


def read_rows(results_file: Path) -> list[dict[str, str]]:
    """All results.tsv rows as dicts ([] when missing or empty)."""
    if not results_file.exists() or results_file.stat().st_size == 0:
        return []
    with open(results_file, encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter="\t")]


def recompute_statuses(results_file: Path) -> None:
    """Store-wide status refresh after a Trial write (issue #5).

    A new on_front point demotes rows it dominates to dominated; incomplete
    and rejected rows are left out; fingerprint-less legacy rows without a
    config_json fingerprint are untouched. Idempotent: rerunning changes
    nothing, so a no-change store is not rewritten.
    """
    rows = read_rows(results_file)
    updated = recompute.recompute_rows(rows)
    if updated == rows:
        return
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=CATEGORY_FIELDNAMES, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(updated)


def get_previous_best(results_file: Path, model_name: str | None = None) -> float:
    if not results_file.exists():
        return 0.0
    best_score = 0.0
    try:
        with open(results_file, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if classify.is_on_front(row.get("status")):
                    if model_name:
                        row_model = row.get("model", "")
                        if row_model and row_model != model_name:
                            continue
                        if not row_model and model_name not in row.get("description", ""):
                            continue
                    try:
                        score = float(row.get("val_score", 0.0))
                        if score > best_score:
                            best_score = score
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Error reading results.tsv: {e}")
    return best_score


CATEGORY_FIELDNAMES = [
    "schema_version",
    "trial_id",
    "commit",
    "model",
    "model_id",
    "backend",
    "category",
    "evaluation_profile",
    "scoring_benchmark",
    "outcome",
    "diagnostic",
    "status",
    "val_score",
    "swe_score",
    "lcb_score",
    "he_score",
    "mbpp_score",
    "bigcode_score",
    "agentic",
    "coding",
    "memory_gb",
    "elapsed_sec",
    "tps",
    "bench_tg",
    "kv",
    "ctx",
    "threads",
    "threads_batch",
    "batch_size",
    "ubatch_size",
    "n_cpu_moe",
    "temp",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "presence_penalty",
    "cont_batching",
    "flash_attn",
    "no_mmap",
    "spec_draft_n_max",
    "task_ids",
    "random_seed",
    "config_json",
    "binary_version",
    "tps_source",
    "description",
]


def _ensure_category_column(results_file: Path) -> None:
    """One-time migration: add missing columns."""
    if not results_file.exists() or results_file.stat().st_size == 0:
        return
    with open(results_file, encoding="utf-8") as f:
        header = f.readline().strip()
    cols = header.split("\t")
    if cols == CATEGORY_FIELDNAMES:
        return  # already migrated
    backup = results_file.with_suffix(results_file.suffix + ".bak")
    if results_file.is_file() and not backup.exists():
        shutil.copy2(results_file, backup)
    rows = []
    with open(results_file, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=CATEGORY_FIELDNAMES, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _tsv_cell(value: Any, fmt: str | None = None) -> str:
    """Format a TSV cell; None stays blank, zeros stay zeros."""
    if value is None:
        return ""
    return format(value, fmt) if fmt else str(value)


def _engine_version() -> str:
    """Engine identity for the Trial evidence (binary_version cell).

    Resolves the engine actually used; falls back to empty when no
    llama-server is resolvable (e.g. pure row-serialization contexts).
    """
    try:
        return engine_version_tag(resolve_llama_server())
    except FileNotFoundError:
        return ""


def write_row(
    results_file: Path,
    commit: str,
    val_score: float,
    swe_score: float,
    he_score: float,
    mbpp_score: float,
    memory_gb: float,
    status: str,
    description: str,
    lcb_score: float = 0.0,
    bigcode_score: float = 0.0,
    agentic: float | None = None,
    coding: float | None = None,
    category: str = "",
    elapsed_sec: float = 0.0,
    model: str = "",
    tps: float | None = None,
    bench_tg: float | None = None,
    kv: str = "",
    ctx: int | None = None,
    threads: int | None = None,
    threads_batch: int | None = None,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    n_cpu_moe: int | None = None,
    temp: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    repeat_penalty: float | None = None,
    presence_penalty: float | None = None,
    cont_batching: Any = None,
    flash_attn: str = "",
    no_mmap: Any = None,
    spec_draft_n_max: int | None = None,
    outcome: str = "",
    diagnostic: str = "",
    evaluation_profile: str = "",
    scoring_benchmark: str = "",
    task_ids: str = "",
    config_json: str = "",
    tps_source: str = "",
):
    _ensure_category_column(results_file)
    if status not in TRIAL_STATUSES:
        raise ValueError(f"invalid trial status: {status!r}; allowed: {sorted(TRIAL_STATUSES)}")
    new_file = not results_file.exists() or results_file.stat().st_size == 0
    with open(results_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CATEGORY_FIELDNAMES, delimiter="\t")
        if new_file:
            writer.writeheader()
        row = {
            "schema_version": "2",
            "trial_id": str(uuid.uuid4()),
            "commit": commit,
            "model": model,
            "model_id": model,
            "backend": "sglang" if model and not model.lower().endswith(".gguf") else "llama.cpp",
            "category": category,
            "evaluation_profile": evaluation_profile or category,
            "scoring_benchmark": scoring_benchmark
            or ("claw-eval" if category.startswith("agentic") else "coding"),
            "outcome": outcome
            or ("OK" if not status.lower().startswith("fail") else "MODEL_REJECTED"),
            "diagnostic": diagnostic,
            "status": status,
            "val_score": f"{val_score:.6f}",
            "swe_score": f"{swe_score:.6f}",
            "lcb_score": f"{lcb_score:.6f}",
            "he_score": f"{he_score:.6f}",
            "mbpp_score": f"{mbpp_score:.6f}",
            "bigcode_score": f"{bigcode_score:.6f}",
            "agentic": _tsv_cell(agentic, ".4f"),
            "coding": _tsv_cell(coding, ".6f"),
            "memory_gb": f"{memory_gb:.1f}",
            "elapsed_sec": f"{elapsed_sec:.0f}",
            "tps": _tsv_cell(tps, ".1f"),
            "bench_tg": _tsv_cell(bench_tg, ".1f"),
            "kv": kv,
            "ctx": _tsv_cell(ctx),
            "threads": _tsv_cell(threads),
            "threads_batch": _tsv_cell(threads_batch),
            "batch_size": _tsv_cell(batch_size),
            "ubatch_size": _tsv_cell(ubatch_size),
            "n_cpu_moe": _tsv_cell(n_cpu_moe),
            "temp": _tsv_cell(temp),
            "top_p": _tsv_cell(top_p),
            "top_k": _tsv_cell(top_k),
            "min_p": _tsv_cell(min_p),
            "repeat_penalty": _tsv_cell(repeat_penalty),
            "presence_penalty": _tsv_cell(presence_penalty),
            "cont_batching": _tsv_cell(cont_batching),
            "flash_attn": flash_attn,
            "no_mmap": _tsv_cell(no_mmap),
            "spec_draft_n_max": _tsv_cell(spec_draft_n_max),
            "task_ids": task_ids,
            "random_seed": "",
            "config_json": config_json
            or json.dumps(
                {
                    "kv": kv,
                    "ctx": ctx,
                    "threads": threads,
                    "threads_batch": threads_batch,
                    "batch_size": batch_size,
                    "ubatch_size": ubatch_size,
                    "flash_attn": flash_attn,
                    "spec_draft_n_max": spec_draft_n_max,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            "binary_version": _engine_version(),
            "tps_source": tps_source,
            "description": description,
        }
        writer.writerow(row)


def run_evaluation(cfg: dict | Any, skip_bench: bool = False, **overrides) -> dict[str, Any]:
    """Run one trial and return results as a dict (backward-compat wrapper).

    New code should use ExperimentRunner.run_trial() directly for a typed TrialResult.
    """
    runner = ExperimentRunner(MODELS_DIR)
    tr = runner.run_trial(cfg, skip_bench=skip_bench, **overrides)
    return {
        "status": tr.status,
        "val_score": tr.val_score,
        "coding_val": tr.coding_val,
        "coding_tps": tr.coding_tps,
        "lcb_val": tr.lcb_val,
        "he_val": tr.he_val,
        "mbpp_val": tr.mbpp_val,
        "bigcode_val": tr.bigcode_val,
        "swe_val": tr.swe_val,
        "agentic_val": tr.agentic_val,
        "agentic_tier": tr.agentic_tier,
        "agentic_task_count": tr.agentic_task_count,
        "avg_tps": tr.avg_tps,
        "peak_vram_gb": tr.peak_vram_gb,
        "bench_tg_tps": tr.bench_tg_tps,
        "bench_pp_tps": tr.bench_pp_tps,
        "elapsed_sec": tr.elapsed_sec,
        "outcome": tr.outcome.value,
        "diagnostic": tr.diagnostic,
        "task_ids": list(tr.task_ids),
        "tps_source": tr.tps_source,
    }


def handle_single_run(args):
    if not args.desc:
        print(
            "Error: --desc is required for logging single runs. Example: --desc 'Tweak system prompt'"
        )
        sys.exit(1)

    print(f"Starting single run for model: {args.model}")
    commit = get_git_commit()

    agentic_quick = getattr(args, "agentic_quick", False) is True
    agentic_full = getattr(args, "agentic_full", False) is True

    # Run evaluation
    res = run_evaluation(
        args,
        agentic_quick=agentic_quick,
        agentic_full=agentic_full,
    )

    failed = res["status"] != "OK" or res.get("outcome", "OK") != "OK"
    if failed:
        print(f"Evaluation failed: {res['status'] or res.get('outcome', 'OK')}")
        write_row(
            RESULTS_FILE,
            commit,
            0.0,
            0.0,
            0.0,
            0.0,
            res["peak_vram_gb"],
            "rejected",  # classifier: hard failure (VRAM policy / crash / invalid)
            f"FAIL: {res['status'] or res.get('outcome', 'OK')} | {args.desc}",
            category=determine_category(args),
            elapsed_sec=res.get("elapsed_sec", 0.0),
            tps=res.get("avg_tps"),
            bench_tg=res.get("bench_tg_tps"),
            outcome=res.get("outcome", ""),
            diagnostic=res.get("diagnostic", ""),
            task_ids=",".join(res.get("task_ids", [])),
            tps_source=res.get("tps_source", ""),
            **_result_config(),
        )
        recompute_statuses(RESULTS_FILE)  # literal "after every write", issue #5
        sys.exit(1)

    val_score = res["val_score"]

    # Classify via the Pareto nucleus (issue #4): rejected on hard failure,
    # incomplete while any axis is missing, else on_front/dominated vs the
    # known Set for this hardware+budget bucket.
    # coding-10 is canonical-Trial work; validation = smoke gates only (run_trial
    # forces include_coding off). Mirror that so the vector never claims a
    # 0.0 coding axis on a validation row.
    coding_measured = getattr(args, "include_coding", False) is True and not getattr(
        args, "validation", False
    )
    # Claw quick is smoke, not the agentic axis (ADR 0006: agentic = Claw full).
    agentic_full = res.get("agentic_tier") == "full"
    vector = classify.ObjectiveVector(
        ctx=args.ctx_size,
        tps=res["avg_tps"] or None,
        agentic=res["agentic_val"] if agentic_full else None,
        coding=res["coding_val"] if coding_measured else None,
    )
    _ensure_category_column(RESULTS_FILE)
    rows = read_rows(RESULTS_FILE)
    fp = classify.fp_from_baseline(config.load_config())
    status, _ = classify.plan_write(
        rows,
        fp=fp,
        vector=vector,
        bucket_gb=classify.bucket(res["peak_vram_gb"]),
    )

    details = f"{args.model} kv={args.kv} ctx={args.ctx_size} TPS={res['avg_tps']:.1f} VRAM={res['peak_vram_gb']:.1f}GB coding={res['coding_val']:.4f}"
    details += f" lcb={res.get('lcb_val', 0.0):.4f} he={res.get('he_val', 0.0):.4f} mbpp={res.get('mbpp_val', 0.0):.4f} bigcode={res.get('bigcode_val', 0.0):.4f}"
    if res.get("agentic_tier"):
        details += f" agentic_{res['agentic_tier']}={res.get('agentic_val', 0.0):.4f} (n={res.get('agentic_task_count', 0)})"
    details += f" bench_tg={res.get('bench_tg_tps', 0.0):.1f}"
    details += f" | {args.desc}"

    # Log to results.tsv
    write_row(
        RESULTS_FILE,
        commit,
        val_score,
        res.get("swe_val", 0.0),
        res.get("he_val", 0.0),
        res.get("mbpp_val", 0.0),
        res["peak_vram_gb"],
        status,
        details,
        lcb_score=res.get("lcb_val", 0.0),
        bigcode_score=res.get("bigcode_val", 0.0),
        agentic=vector.agentic,
        coding=vector.coding,
        category=determine_category(args),
        elapsed_sec=res.get("elapsed_sec", 0.0),
        tps=res.get("avg_tps"),
        bench_tg=res.get("bench_tg_tps"),
        outcome=res.get("outcome", ""),
        diagnostic=res.get("diagnostic", ""),
        task_ids=",".join(res.get("task_ids", [])),
        tps_source=res.get("tps_source", ""),
        **_result_config(),
    )
    recompute_statuses(RESULTS_FILE)

    print("\n" + "=" * 40)
    print("EVALUATION COMPLETE")
    print("=" * 40)
    print(f"Model:          {args.model}")
    print(f"KV Cache:       {args.kv}")
    print(f"Context Size:   {args.ctx_size}")
    print("-" * 40)
    print(f"Coding Score:     {res['coding_val']:.4f}")
    print(f"  LCB:            {res.get('lcb_val', 0.0):.4f}")
    print(f"  HumanEval+:     {res.get('he_val', 0.0):.4f}")
    print(f"  MBPP+:          {res.get('mbpp_val', 0.0):.4f}")
    print(f"  BigCode Hard:   {res.get('bigcode_val', 0.0):.4f}")
    print(f"Combined TPS:     {res['avg_tps']:.1f} (Threshold: >= {resolve_tps_floor():.1f})")
    print(f"Bench tg:         {res.get('bench_tg_tps', 0.0):.1f} t/s")
    print(f"Peak VRAM:        {res['peak_vram_gb']:.1f} GB")
    print(f"Current Score:    {val_score:.6f}")
    print("-" * 40)
    print(f"\n>>> TRIAL STATUS: {status}")


def main():
    args = parse_args()
    if args.list_agentic_benchmarks:
        print(format_agentic_benchmarks())
        return
    if args.list_claw_tiers:
        print(format_claw_tiers())
        return
    handle_single_run(args)


if __name__ == "__main__":
    main()

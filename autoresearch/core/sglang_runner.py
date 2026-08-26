import os
import selectors
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

from autoresearch.core.llama_runner import (
    ROOT_DIR,
    ServerIntent,
    candidate_ports,
    sweep_leftover_processes,
)
from autoresearch.core.process_guard import TEARDOWN_GRACE_SECONDS, ProcessGuard
from autoresearch.core.single_load import enforce_single_load, resolve_allow_multi

IS_WINDOWS = os.name == "nt"
REPO_ROOT = ROOT_DIR.parent.parent
SGLANG_BIN = REPO_ROOT / "venv-sglang" / ("Scripts" if IS_WINDOWS else "bin")
SGLANG_PYTHON = SGLANG_BIN / ("python.exe" if IS_WINDOWS else "python3")


def _sglang_nvcc_dir() -> Path | None:
    """nvcc bundled by the pip ``cuda-toolkit`` inside the SGLang venv.

    SGLang JIT-compiles Triton/FlashInfer kernels on first load; the compiler
    (``nvcc``) and CUDA runtime must be discoverable. The venv ships them under
    ``site-packages/nvidia/cu*/`` — not on PATH, and no system CUDA install is
    assumed on the operator host. POSIX venv layout only: SGLang has no
    Windows build (WSL2 is the only supported host here).
    """
    try:
        candidates = sorted(SGLANG_BIN.parent.glob("lib/python*/site-packages/nvidia/cu*/bin/nvcc"))
    except OSError:
        return None
    for nvcc in candidates:
        if nvcc.is_file():
            return nvcc.parent
    return None


# Backend flags shared by bench and server (issue #59): SGLang 0.5.17 on an
# 8 GB-class host needs an explicit static-pool fraction (auto-computed rejects
# the hybrid linear-attention arch), the Triton attention path (FlashInfer JIT
# fails with the pip cu13 toolchain), and the model's native bf16 dtype (fp16
# trips a state-cache dtype bug). Keep the two command builders in sync.
SGLANG_BACKEND_FLAGS = [
    "--dtype",
    "bfloat16",
    "--mem-fraction-static",
    "0.9",
    "--attention-backend",
    "triton",
    "--sampling-backend",
    "pytorch",
]


def _sglang_env() -> dict[str, str]:
    """Process env for SGLang subprocesses: venv bin + JIT toolchain on PATH."""
    env = os.environ.copy()
    sglang_bin = str(SGLANG_BIN)
    path_dirs = [sglang_bin]
    nvcc_dir = _sglang_nvcc_dir()
    if nvcc_dir is not None:
        path_dirs.append(str(nvcc_dir))
        env["CUDA_HOME"] = str(nvcc_dir.parent)
    env["PATH"] = f"{os.pathsep.join(path_dirs)}{os.pathsep}{env.get('PATH', '')}"
    env["SGLANG_MAMBA_CONV_DTYPE"] = "bfloat16"
    env["SGLANG_MAMBA_SSM_DTYPE"] = "bfloat16"
    return env


def _popen_group_kwargs() -> dict[str, Any]:
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"preexec_fn": os.setsid}


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, text=True
        )
        return
    import signal

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    # Grace window, then force-kill the whole group so SIGTERM-ignoring
    # workers (torch multiprocessing) cannot survive a failed startup.
    deadline = time.monotonic() + TEARDOWN_GRACE_SECONDS
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


def run_sglang_bench_validation(
    model_path: Path,
    batch_size: int,
    n_prompt: int,
    n_gen: int,
) -> float:
    # --- Guard against 8GB VRAM OOM on large models ---
    model_name = model_path.name.upper()
    is_large_model = "35B" in model_name or "32B" in model_name
    if is_large_model:
        vram_gb: float | None = None
        try:
            import torch

            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            pass

        if vram_gb is None:
            try:
                res = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
                )
                vram_gb = float(res.stdout.splitlines()[0].strip()) / 1024
            except Exception:
                # This benchmark path has already crashed WSL on 8GB GPUs. If
                # VRAM cannot be queried, fail closed for 32B/35B SGLang models.
                vram_gb = 0.0

        if vram_gb < 10.0:
            raise RuntimeError(
                f"SGLang disabled for {model_path.name}: {vram_gb:.1f}GB VRAM "
                "detected for a 32B/35B model. Refusing bench/server validation "
                "to prevent WSL crash."
            )

    print(f"  [bench] Running sglang.bench_one_batch for {model_path.name}")
    cmd = [
        str(SGLANG_PYTHON),
        "-m",
        "sglang.bench_one_batch",
        "--model-path",
        str(model_path),
        "--batch-size",
        str(batch_size),
        "--input-len",
        str(n_prompt),
        "--output-len",
        str(n_gen),
        *SGLANG_BACKEND_FLAGS,
    ]
    if "GPTQ" in model_path.name.upper():
        cmd += ["--quantization", "gptq_marlin"]
    if "AWQ" in model_path.name.upper():
        cmd += ["--quantization", "awq"]

    server_env = _sglang_env()

    try:
        res = subprocess.run(cmd, env=server_env, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"sglang.bench_one_batch failed:\n{e.stderr}")
        raise

    # Parse output, looking for "Decode token/s:" (legacy) or the 0.5.x
    # "Decode.  median ... median throughput: N token/s" format. We want the
    # decode (generation) tokens per second (tg).
    tg_tps = 0.0
    for line in res.stdout.splitlines():
        if "Decode token/s:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                try:
                    # Strip spaces and "tokens/s" if present, though split by ':' might just leave " 45.2"
                    tg_tps = float(parts[1].split()[0].replace("tokens/s", "").strip())
                except ValueError:
                    pass
        elif "median throughput:" in line and "Decode" in line:
            try:
                # e.g. "Decode.  median latency: 0.01738 s, median throughput: 57.55 token/s"
                tg_tps = float(line.split("median throughput:")[1].split()[0])
            except (ValueError, IndexError):
                pass
    return tg_tps


class SGLangServerRunner:
    def __init__(self, intent: ServerIntent, log_path: Path | None = None):
        self.intent = intent
        self.log_path = log_path

        self.port: int | None = None
        self.peak_vram_mb: float = 0.0

        self._server_proc: subprocess.Popen[str] | None = None
        self._server_log: Any = None
        self._stop_event = threading.Event()
        self._guard: ProcessGuard | None = None

    def _build_cmd(self, target_port: int) -> list[str]:
        print(
            f"  [SGLang] Directory detected. Using SGLang backend for {self.intent.model_path.name}"
        )
        cmd = [
            str(SGLANG_PYTHON),
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(self.intent.model_path),
            "--served-model-name",
            self.intent.model_path.name,
            "--host",
            str(self.intent.host),
            "--port",
            str(target_port),
            "--context-length",
            str(self.intent.ctx_size),
            *SGLANG_BACKEND_FLAGS,
            "--disable-cuda-graph",
            "--mm-feature-transport",
            "cpu",
        ]
        if os.environ.get("SGLANG_TRUST_REMOTE_CODE", "1") != "0":
            cmd.append("--trust-remote-code")
        if "GPTQ" in self.intent.model_path.name.upper():
            cmd += ["--quantization", "gptq_marlin"]
        if "AWQ" in self.intent.model_path.name.upper():
            cmd += ["--quantization", "awq"]
        return cmd

    def is_port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.intent.host, port)) == 0

    def is_server_ready(self, port: int) -> bool:
        try:
            req = urllib.request.Request(f"http://{self.intent.host}:{port}/v1/models")
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def start(self) -> int:
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._server_log = open(self.log_path, "w", encoding="utf-8")
        else:
            self._server_log = subprocess.DEVNULL

        server_env = _sglang_env()

        # Single-load gate (#41): refuse a second full server while one is
        # live. The pre-flight orphan sweep would kill a live sibling on a
        # harness port, so it only runs when the gate passes and allow-multi
        # is off.
        allow_multi = resolve_allow_multi()
        enforce_single_load(allow_multi=allow_multi)
        if not allow_multi:
            sweep_leftover_processes()
        self._guard = ProcessGuard()

        startup_tail: deque[str] = deque(maxlen=20)
        for port in candidate_ports(self.intent.port):
            cmd = self._build_cmd(port)
            if self.is_port_in_use(port):
                continue

            print(f"Starting server: {' '.join(cmd)}")
            self._server_proc = self._guard.spawn(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=server_env,
                **_popen_group_kwargs(),
            )

            ready = False
            output_selector = None
            if not IS_WINDOWS and self._server_proc.stdout is not None:
                try:
                    output_selector = selectors.DefaultSelector()
                    output_selector.register(self._server_proc.stdout, selectors.EVENT_READ)
                except (OSError, ValueError):
                    output_selector = None

            while True:
                if self._server_proc.poll() is not None:
                    break

                try:
                    line = ""
                    if output_selector is None or output_selector.select(timeout=0.1):
                        line = self._server_proc.stdout.readline()  # type: ignore
                    if line:
                        if self.log_path:
                            self._server_log.write(line)
                            self._server_log.flush()
                        startup_tail.append(line.rstrip())

                        if (
                            "Uvicorn running on" in line
                            or "The server is fired up and ready to roll!" in line
                        ):
                            ready = True
                            break
                except Exception:
                    pass

                if self.is_server_ready(port):
                    ready = True
                    break

                # POSIX readiness is polled frequently; avoid adding 0.5 s
                # latency after selector-based health checks.
                time.sleep(0.1 if output_selector is not None else 0.5)

            if output_selector is not None:
                output_selector.close()
            if ready:
                self.port = port

                def consume_output():
                    try:
                        for log_line in self._server_proc.stdout:  # type: ignore
                            if self.log_path:
                                self._server_log.write(log_line)
                                self._server_log.flush()
                    except Exception:
                        pass

                threading.Thread(target=consume_output, daemon=True).start()
                return port

            else:
                if self._server_proc.poll() is None:
                    print(f"Failed to start on port {port}, trying next...")
                    _terminate_process_tree(self._server_proc)
                    self._server_proc.wait()
                else:
                    print("FAIL: Server crashed during startup.")
                    print("Tail of startup log:")
                    print("\n".join(startup_tail))
                    break

        raise RuntimeError("Failed to start SGLang server on any candidate port.")

    def stop(self) -> None:
        self._stop_event.set()

        if self._server_proc and self._server_proc.poll() is None and self._guard is None:
            _terminate_process_tree(self._server_proc)
            self._server_proc.wait()

        if self._guard:
            self._guard.teardown()
            self._guard = None

        if self._server_log and self._server_log != subprocess.DEVNULL:
            try:
                self._server_log.close()
            except Exception:
                pass

    def __enter__(self) -> "SGLangServerRunner":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        self.stop()

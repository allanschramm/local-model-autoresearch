"""
check_hardware.py — Diagnóstico de hardware para IA local (Windows / macOS / Linux).

Detecta RAM, GPU NVIDIA (VRAM dedicada) ou memória unificada (Apple Silicon / sem
NVIDIA discreta) e recomenda GGUF/contexto conservadores. Dense deve caber no pool
detectado; nunca sugerir partial dense offload.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any


MEMORY_CLASS_UNIFIED = "unified_memory"
MEMORY_CLASS_DISCRETE = "discrete_gpu"


def classify_memory_class(*, has_cuda: bool, has_metal: bool = False) -> str:
    """Discrete only when NVIDIA CUDA VRAM is present; else one shared host pool.

    ``has_metal`` is accepted for call-site clarity (Darwin) but does not change
    the class: without CUDA, the host is always ``unified_memory``.
    """
    del has_metal  # API clarity only
    if has_cuda:
        return MEMORY_CLASS_DISCRETE
    return MEMORY_CLASS_UNIFIED


def model_pool_gb(info: dict[str, Any]) -> float:
    """Reported capacity GB: dedicated VRAM, or total unified RAM (not a safe fill target)."""
    if info.get("memory_class") == MEMORY_CLASS_DISCRETE:
        return float(info.get("vram_gb") or 0.0)
    return float(info.get("ram_gb") or 0.0)


def _detect_ram_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass

    system = sys.platform
    if system == "win32":
        try:
            res = subprocess.run(
                ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                capture_output=True,
                text=True,
                check=False,
            )
            lines = [line.strip() for line in res.stdout.splitlines() if line.strip().isdigit()]
            if lines:
                return round(int(lines[0]) / (1024**3), 1)
        except Exception:
            pass
        return 0.0

    if system == "darwin":
        try:
            res = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=False,
            )
            raw = (res.stdout or "").strip()
            if raw.isdigit():
                return round(int(raw) / (1024**3), 1)
        except Exception:
            pass
        return 0.0

    # Linux / other POSIX
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    # MemTotal is in kB
                    return round(int(parts[1]) / (1024**2), 1)
    except Exception:
        pass
    return 0.0


def _detect_nvidia() -> tuple[str | None, float, bool]:
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            line = res.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                return parts[0], round(float(parts[1]) / 1024.0, 1), True
    except Exception:
        pass
    return None, 0.0, False


def _detect_apple_metal() -> tuple[bool, str | None]:
    """On macOS, Metal is the GPU backend (Intel + Apple Silicon)."""
    if sys.platform != "darwin":
        return False, None
    chip = None
    try:
        res = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        brand = (res.stdout or "").strip()
        if brand:
            chip = brand
    except Exception:
        pass
    machine = platform.machine().lower()
    if not chip:
        chip = f"macOS ({machine})"
    return True, chip


def get_system_info() -> dict[str, Any]:
    ram_gb = _detect_ram_gb()
    gpu_name, vram_gb, has_cuda = _detect_nvidia()
    has_metal, chip = _detect_apple_metal()

    if has_cuda and gpu_name:
        display_gpu = gpu_name
    elif has_metal:
        display_gpu = "Apple / macOS (Metal)"
    else:
        display_gpu = "Não detectada (CPU)"

    memory_class = classify_memory_class(has_cuda=has_cuda, has_metal=has_metal)

    return {
        "ram_gb": ram_gb,
        "vram_gb": vram_gb if has_cuda else 0.0,
        "gpu_name": display_gpu,
        "has_cuda": has_cuda,
        "has_metal": has_metal,
        "memory_class": memory_class,
        "chip": chip,
        "platform": sys.platform,
        "detection_complete": ram_gb > 0.0 and (has_cuda or has_metal or ram_gb > 0.0),
    }


def generate_recommendations(info: dict[str, Any]) -> None:
    ram = float(info.get("ram_gb") or 0.0)
    vram = float(info.get("vram_gb") or 0.0)
    has_cuda = bool(info.get("has_cuda"))
    has_metal = bool(info.get("has_metal"))
    memory_class = info.get("memory_class") or classify_memory_class(
        has_cuda=has_cuda, has_metal=has_metal
    )
    pool = model_pool_gb({**info, "memory_class": memory_class})
    chip = info.get("chip")
    detection_ok = ram > 0.0

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print(" [PC] RECOMENDADOR DE IA LOCAL (Diagnóstico de Hardware)")
    print("=" * 60)
    print(f" * Sistema: {info.get('platform') or sys.platform}" + (f" / {chip}" if chip else ""))
    print(f" * Processador / RAM: {ram} GB RAM")
    print(f" * Placa de Vídeo (GPU): {info.get('gpu_name')}")
    if not detection_ok:
        print(" * DETECÇÃO INCOMPLETA: RAM = 0 — confirme manualmente antes de baixar modelos")
        print("   macOS: Sobre este Mac / sysctl -n hw.memsize")
        print("   Windows: Configurações → Sistema → Sobre, ou Task Manager")
        print("   Linux: free -h / /proc/meminfo")
        print("   NVIDIA: nvidia-smi")
        print("=" * 60)
        return

    if memory_class == MEMORY_CLASS_DISCRETE:
        print(f" * Classe de memória: discrete_gpu (VRAM dedicada)")
        print(f" * VRAM Dedicada: {vram} GB VRAM")
        print(f" * Capacidade reportada: {pool} GB (VRAM física — dense deve caber aqui)")
    else:
        print(f" * Classe de memória: unified_memory (um pool = RAM)")
        if has_metal:
            print(f" * Memória unificada (Metal): {ram} GB total (mesmo pool do sistema)")
        else:
            print(f" * Sem VRAM NVIDIA dedicada — pool = RAM do sistema ({ram} GB)")
        print(
            f" * Capacidade reportada: {pool} GB RAM total — NÃO encha: "
            "GGUF + contexto devem deixar folga para SO/IDE/browser"
        )
    print("-" * 60)

    if memory_class == MEMORY_CLASS_DISCRETE:
        if pool >= 15.0:
            tier = "Alto Desempenho (16GB+ VRAM)"
            model = "GGUF denso que caiba integralmente na VRAM física"
            ngl = "99 (100% GPU; modelo denso deve caber na VRAM física)"
            ctx = "32768"
        elif pool >= 7.5:
            tier = "Intermediário (8GB-12GB VRAM)"
            model = "GGUF denso compacto que caiba integralmente na VRAM física"
            ngl = "99 (100% GPU; modelo denso deve caber na VRAM física)"
            ctx = "16384"
        elif pool >= 3.5:
            tier = "Básico GPU (4GB-6GB VRAM)"
            model = "GGUF denso pequeno que caiba integralmente na VRAM física"
            ngl = "99 (100% GPU; modelo denso deve caber na VRAM física)"
            ctx = "8192"
        else:
            tier = "VRAM muito baixa"
            model = "GGUF mínimo que caiba na VRAM física (ou rejeitar o modelo)"
            ngl = "99 (100% GPU; modelo denso deve caber na VRAM física)"
            ctx = "4096"
    elif has_metal:
        if pool >= 24.0:
            tier = "macOS Metal unificado (24GB+ RAM)"
            model = (
                "GGUF que caiba na RAM unificada com folga clara para o sistema "
                "(OS + IDE + modelo dividem o mesmo pool — não use quase toda a RAM)"
            )
            ngl = "99 (Metal; modelo deve caber na memória unificada)"
            ctx = "16384"
        elif pool >= 12.0:
            tier = "macOS Metal unificado (12–24GB RAM)"
            model = (
                "GGUF compacto (tipicamente poucos GB / ~3–9B Q4) com folga para o SO — "
                "não use GGUF ~12GB+ em máquinas de 16GB unificado"
            )
            ngl = "99 (Metal; modelo deve caber na memória unificada)"
            ctx = "8192"
        else:
            tier = "macOS Metal unificado (RAM limitada)"
            model = "GGUF bem pequeno adequado à RAM unificada com folga para o SO"
            ngl = "99 (Metal; modelo deve caber na memória unificada)"
            ctx = "4096"
    else:
        tier = "Modo CPU / Sem GPU Dedicada (memória unificada = RAM)"
        model = (
            "GGUF pequeno adequado à RAM disponível, com folga para o sistema"
        )
        ngl = "0 (Somente CPU)"
        ctx = "4096"

    print(f"\nCLASSIFICAÇÃO: {tier}")
    print(f" Modelo recomendado: {model}")
    print(f" Flag recomendada -ngl: {ngl}")
    print(f" Tamanho de contexto -c: {ctx}")
    print(" Velocidade: meça no seu hardware; não use estimativas como resultado")
    if memory_class == MEMORY_CLASS_UNIFIED:
        print(
            "\nATENÇÃO (memória unificada): whichllm/llmfit podem tratar RAM como "
            "VRAM cheia e recomendar modelos grandes demais. Use este diagnóstico "
            "como autoridade de fit; deixe folga para o sistema antes de baixar."
        )
    print("\nAplique estes valores em autoresearch/core/config.py e inicie com:")
    if sys.platform == "win32":
        print("  .\\venv\\Scripts\\python.exe scripts\\serve-config.py serve")
    else:
        print("  ./venv/bin/python scripts/serve-config.py serve")
    print("=" * 60)


if __name__ == "__main__":
    generate_recommendations(get_system_info())

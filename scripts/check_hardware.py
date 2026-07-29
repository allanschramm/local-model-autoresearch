"""
check_hardware.py — Diagnóstico de hardware para IA local (Windows / macOS / Linux).

Detecta RAM, GPU NVIDIA (VRAM dedicada) ou memória unificada (Apple Silicon / sem
NVIDIA discreta) e recomenda GGUF/contexto conservadores. Dense deve caber no pool
detectado; nunca sugerir partial dense offload.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Repo root on path when run as scripts/check_hardware.py
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from autoresearch.core.hardware import (
    MEMORY_CLASS_DISCRETE,
    MEMORY_CLASS_UNIFIED,
    classify_memory_class,
    get_system_info,
    model_pool_gb,
)

# Re-export for tests that import from scripts.check_hardware
__all__ = [
    "MEMORY_CLASS_DISCRETE",
    "MEMORY_CLASS_UNIFIED",
    "classify_memory_class",
    "generate_recommendations",
    "get_system_info",
    "model_pool_gb",
]


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
        print(" * Classe de memória: discrete_gpu (VRAM dedicada)")
        print(f" * VRAM Dedicada: {vram} GB VRAM")
        print(f" * Capacidade reportada: {pool} GB (VRAM física — dense deve caber aqui)")
    else:
        print(" * Classe de memória: unified_memory (um pool = RAM)")
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
        model = "GGUF pequeno adequado à RAM disponível, com folga para o sistema"
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

from scripts.check_hardware import (
    classify_memory_class,
    generate_recommendations,
    model_pool_gb,
)
from scripts.verify_setup import performance_advice


def test_dense_gpu_recommendation_never_uses_partial_offload(capsys):
    generate_recommendations(
        {
            "ram_gb": 16.0,
            "vram_gb": 4.0,
            "gpu_name": "Test GPU",
            "has_cuda": True,
            "has_metal": False,
            "memory_class": "discrete_gpu",
        }
    )

    output = capsys.readouterr().out
    assert "-ngl: 99" in output
    assert "Parcial na GPU" not in output
    assert "deve caber" in output
    assert "discrete" in output.lower() or "VRAM Dedicada" in output


def test_cpu_recommendation_remains_cpu_only(capsys):
    generate_recommendations(
        {
            "ram_gb": 16.0,
            "vram_gb": 0.0,
            "gpu_name": "Não detectada (CPU)",
            "has_cuda": False,
            "has_metal": False,
            "memory_class": "unified_memory",
            "physical_cores": 8,
            "logical_cores": 16,
        }
    )

    output = capsys.readouterr().out
    assert "-ngl: 0" in output
    assert "Somente CPU" in output
    assert "unificada" in output.lower() or "unificado" in output.lower()
    assert "8 físicos / 16 lógicos" in output
    assert "-t 8" in output
    assert "NUMA" in output


def test_cores_and_simd_lines(capsys):
    generate_recommendations(
        {
            "ram_gb": 16.0,
            "vram_gb": 0.0,
            "gpu_name": "Não detectada (CPU)",
            "has_cuda": False,
            "has_metal": False,
            "memory_class": "unified_memory",
            "physical_cores": 8,
            "logical_cores": 16,
            "simd_hints": ["avx512f", "avx2", "sse4_2"],
        }
    )

    output = capsys.readouterr().out
    assert "SIMD (CPU): avx512f, avx2, sse4_2" in output


def test_simd_line_omitted_when_absent(capsys):
    generate_recommendations(
        {
            "ram_gb": 16.0,
            "vram_gb": 0.0,
            "gpu_name": "Não detectada (CPU)",
            "has_cuda": False,
            "has_metal": False,
            "memory_class": "unified_memory",
        }
    )

    output = capsys.readouterr().out
    assert "SIMD" not in output
    assert "-ngl: 0" in output


def test_mac_unified_metal_uses_gpu_not_cpu_only(capsys):
    generate_recommendations(
        {
            "ram_gb": 16.0,
            "vram_gb": 0.0,
            "gpu_name": "Apple / macOS (Metal)",
            "has_cuda": False,
            "has_metal": True,
            "memory_class": "unified_memory",
            "chip": "Apple M4",
        }
    )

    output = capsys.readouterr().out
    assert "unificada" in output.lower() or "unificado" in output.lower()
    assert "Metal" in output or "metal" in output.lower()
    assert "-ngl: 99" in output
    assert "Somente CPU" not in output
    assert "16" in output
    assert "ATENÇÃO (memória unificada)" in output
    assert "whichllm" in output.lower()


def test_incomplete_ram_detection_stops_before_tiers(capsys):
    generate_recommendations(
        {
            "ram_gb": 0.0,
            "vram_gb": 0.0,
            "gpu_name": "Não detectada (CPU)",
            "has_cuda": False,
            "has_metal": False,
            "memory_class": "unified_memory",
        }
    )
    output = capsys.readouterr().out
    assert "DETECÇÃO INCOMPLETA" in output
    assert "CLASSIFICAÇÃO:" not in output


def test_classify_nvidia_is_discrete():
    assert classify_memory_class(has_cuda=True, has_metal=False) == "discrete_gpu"


def test_classify_metal_or_no_cuda_is_unified():
    assert classify_memory_class(has_cuda=False, has_metal=True) == "unified_memory"
    assert classify_memory_class(has_cuda=False, has_metal=False) == "unified_memory"


def test_model_pool_discrete_uses_vram():
    assert (
        model_pool_gb(
            {
                "ram_gb": 32.0,
                "vram_gb": 8.0,
                "memory_class": "discrete_gpu",
            }
        )
        == 8.0
    )


def test_model_pool_unified_uses_ram():
    assert (
        model_pool_gb(
            {
                "ram_gb": 16.0,
                "vram_gb": 0.0,
                "memory_class": "unified_memory",
                "has_metal": True,
            }
        )
        == 16.0
    )


def test_unified_16gb_does_not_claim_high_vram_tier(capsys):
    generate_recommendations(
        {
            "ram_gb": 16.0,
            "vram_gb": 0.0,
            "gpu_name": "Apple / macOS (Metal)",
            "has_cuda": False,
            "has_metal": True,
            "memory_class": "unified_memory",
            "chip": "Apple M4",
        }
    )
    output = capsys.readouterr().out
    assert "16GB+ VRAM" not in output
    assert "Alto Desempenho (16GB+ VRAM)" not in output
    assert "12GB+" in output or "16GB" in output


def test_low_tps_advice_does_not_suggest_dense_layer_spill():
    advice = performance_advice(10)

    assert "aumente a flag -ngl" not in advice
    assert "contexto" in advice.lower()
    assert "modelo menor" in advice.lower()

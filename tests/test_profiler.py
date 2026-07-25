"""Tests for the profiler's math (CPU-only, no model needed)."""

import math
import torch
import pytest
from src.profiler import (
    hadamard_matrix,
    load_centroids,
    simulate_turbo_quant_keys,
    simulate_turbo_quant_values,
    compute_features,
    LayerFeatures,
)


def test_hadamard_orthogonality():
    """H @ H.T should be identity (normalized Hadamard is orthogonal)."""
    for d in [4, 8, 16, 64, 128]:
        H = hadamard_matrix(d)
        eye = H @ H.T
        assert torch.allclose(eye, torch.eye(d), atol=1e-5), f"d={d}"


def test_hadamard_bad_dim():
    with pytest.raises(AssertionError):
        hadamard_matrix(6)


def test_centroids_sorted_and_symmetric():
    for bits in [2, 3, 4]:
        c = load_centroids(bits)
        assert len(c) == 2**bits, f"expected {2**bits} centroids for {bits}-bit, got {len(c)}"
        assert (c == c.sort().values).all(), "centroids must be sorted"
        assert torch.allclose(c, -c.flip(0), atol=1e-4), "centroids must be symmetric"


def test_key_quant_roundtrip_low_error():
    """Quantizing near-Gaussian data should have bounded error."""
    torch.manual_seed(42)
    keys = torch.randn(100, 128)  # 100 tokens, head_dim=128
    restored = simulate_turbo_quant_keys(keys, n_bits=4)
    mse = (restored - keys).pow(2).mean().item()
    assert mse < 0.5, f"4-bit key quantization MSE too high: {mse}"


def test_key_quant_3bit_worse_than_4bit():
    torch.manual_seed(42)
    keys = torch.randn(100, 128)
    mse_3 = (simulate_turbo_quant_keys(keys, 3) - keys).pow(2).mean().item()
    mse_4 = (simulate_turbo_quant_keys(keys, 4) - keys).pow(2).mean().item()
    assert mse_3 > mse_4, "3-bit should have higher error than 4-bit"


def test_value_quant_preserves_range():
    torch.manual_seed(42)
    values = torch.randn(100, 128) * 3 + 1
    restored = simulate_turbo_quant_values(values, n_bits=4)
    assert restored.min() >= values.min() - 0.5
    assert restored.max() <= values.max() + 0.5


def test_value_quant_4bit_vs_2bit():
    torch.manual_seed(42)
    values = torch.randn(100, 128)
    mse_2 = (simulate_turbo_quant_values(values, 2) - values).pow(2).mean().item()
    mse_4 = (simulate_turbo_quant_values(values, 4) - values).pow(2).mean().item()
    assert mse_2 > mse_4


def test_compute_features_shapes():
    torch.manual_seed(42)
    keys = torch.randn(50, 8, 128)    # 50 tokens, 8 heads, head_dim=128
    values = torch.randn(50, 8, 128)
    f = compute_features(keys, values, layer_idx=5)
    assert f.layer_idx == 5
    assert 0.0 <= f.key_outlier_frac <= 1.0
    assert not math.isnan(f.post_wht_excess_kurtosis)
    assert f.key_norm_cv > 0
    assert f.value_dynamic_range > 0
    assert f.simulated_quant_error > 0


def test_features_outlier_detection():
    """Layer with extreme outliers should have higher outlier fraction."""
    torch.manual_seed(42)
    normal_keys = torch.randn(100, 4, 64)
    outlier_keys = normal_keys.clone()
    # Inject a spike in a few tokens only (not all), so channel std stays finite
    outlier_keys[:5, :, 0] = 50.0

    f_normal = compute_features(normal_keys, torch.randn(100, 4, 64), 0)
    f_outlier = compute_features(outlier_keys, torch.randn(100, 4, 64), 1)
    assert f_outlier.key_outlier_frac > f_normal.key_outlier_frac


def test_features_quant_error_sensitive_to_outliers():
    """Outlier keys should have higher quantization error."""
    torch.manual_seed(42)
    normal_keys = torch.randn(100, 4, 64)
    outlier_keys = normal_keys.clone()
    outlier_keys[:, :, 0] *= 10.0

    f_normal = compute_features(normal_keys, torch.randn(100, 4, 64), 0)
    f_outlier = compute_features(outlier_keys, torch.randn(100, 4, 64), 1)
    assert f_outlier.simulated_quant_error > f_normal.simulated_quant_error

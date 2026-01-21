"""
Tests for AttentivePooler probe integration with EEGLeJEPA models.

Run as standalone script:
    python eeg_bench/models/clinical/test_attentive_probe.py
"""
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

from eeg_bench.models.clinical.attentive_probe import AttentivePooler


def test_attentive_pooler_initialization():
    """Test that AttentivePooler initializes correctly."""
    pooler = AttentivePooler(
        num_queries=1,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4.0,
        depth=1,
    )
    assert pooler is not None
    assert isinstance(pooler, nn.Module)
    print("  PASSED: AttentivePooler initializes correctly")


def test_attentive_pooler_forward_shape():
    """Test AttentivePooler forward pass produces correct output shape."""
    batch_size = 4
    seq_len = 10
    embed_dim = 384
    num_queries = 1

    pooler = AttentivePooler(
        num_queries=num_queries,
        embed_dim=embed_dim,
        num_heads=6,
        mlp_ratio=4.0,
        depth=1,
    )

    x = torch.randn(batch_size, seq_len, embed_dim)
    output = pooler(x)

    assert output.shape == (batch_size, num_queries, embed_dim)
    print(f"  PASSED: Output shape {output.shape} is correct")


def test_attentive_pooler_multi_query():
    """Test AttentivePooler with multiple queries."""
    batch_size = 4
    seq_len = 10
    embed_dim = 384
    num_queries = 4

    pooler = AttentivePooler(
        num_queries=num_queries,
        embed_dim=embed_dim,
        num_heads=6,
        mlp_ratio=4.0,
        depth=1,
    )

    x = torch.randn(batch_size, seq_len, embed_dim)
    output = pooler(x)

    assert output.shape == (batch_size, num_queries, embed_dim)
    print(f"  PASSED: Multi-query output shape {output.shape} is correct")


def test_attentive_pooler_with_depth():
    """Test AttentivePooler with transformer blocks after pooling."""
    batch_size = 4
    seq_len = 10
    embed_dim = 384
    num_queries = 1

    pooler = AttentivePooler(
        num_queries=num_queries,
        embed_dim=embed_dim,
        num_heads=6,
        mlp_ratio=4.0,
        depth=3,  # Add transformer blocks
    )

    x = torch.randn(batch_size, seq_len, embed_dim)
    output = pooler(x)

    assert output.shape == (batch_size, num_queries, embed_dim)
    assert pooler.blocks is not None
    print("  PASSED: Deep pooler (depth=3) works correctly")


def test_attentive_pooler_gradient_flow():
    """Test that gradients flow through AttentivePooler."""
    batch_size = 2
    seq_len = 5
    embed_dim = 384

    pooler = AttentivePooler(
        num_queries=1,
        embed_dim=embed_dim,
        num_heads=6,
        mlp_ratio=4.0,
        depth=1,
    )

    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    output = pooler(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    print("  PASSED: Gradients flow correctly")


def test_probe_types_defined():
    """Test that PROBE_TYPES is correctly defined in model modules."""
    from eeg_bench.models.clinical.EEGLejepa_model import PROBE_TYPES as CLINICAL_PROBE_TYPES
    from eeg_bench.models.bci.EEGLeJEPA_model import PROBE_TYPES as BCI_PROBE_TYPES

    assert "linear" in CLINICAL_PROBE_TYPES
    assert "attentive" in CLINICAL_PROBE_TYPES
    assert len(CLINICAL_PROBE_TYPES) == 2

    assert "linear" in BCI_PROBE_TYPES
    assert "attentive" in BCI_PROBE_TYPES
    assert len(BCI_PROBE_TYPES) == 2
    print("  PASSED: PROBE_TYPES correctly defined in both models")


def test_lejepa_config_probe_type():
    """Test LeJEPAConfig has probe_type field."""
    from eeg_bench.config import LeJEPAConfig

    config = LeJEPAConfig(probe_type="attentive")
    assert config.probe_type == "attentive"

    config_linear = LeJEPAConfig()
    assert config_linear.probe_type == "linear"
    print("  PASSED: LeJEPAConfig has probe_type field with correct default")


def test_clinical_forward_with_attentive_pooler():
    """Test clinical model forward logic with attentive pooler."""
    batch_size = 2
    n_chunks = 5
    embed_dim = 384
    num_classes = 4

    # Mock the aggregation logic used in ConcreteLeJEPAClinical
    attentive_pooler = AttentivePooler(
        num_queries=1,
        embed_dim=embed_dim,
        num_heads=6,
        mlp_ratio=4.0,
        depth=1,
    )
    head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    # Simulate cls tokens from chunks: (B, n_chunks, embed_dim)
    cls = torch.randn(batch_size, n_chunks, embed_dim)

    # Attentive pooling
    pooled = attentive_pooler(cls)  # (B, 1, embed_dim)
    pooled = pooled.squeeze(1)  # (B, embed_dim)

    # Classification
    logits = head(pooled)

    assert logits.shape == (batch_size, num_classes)
    print(f"  PASSED: Clinical forward mock produces correct output shape {logits.shape}")


def test_bci_forward_with_attentive_pooler():
    """Test BCI model forward logic with attentive pooler."""
    batch_size = 2
    seq_len = 60  # Typical sequence length for BCI
    embed_dim = 384
    num_classes = 4

    # Mock the aggregation logic used in ConcreteLeJEPABCI
    attentive_pooler = AttentivePooler(
        num_queries=1,
        embed_dim=embed_dim,
        num_heads=6,
        mlp_ratio=4.0,
        depth=1,
    )
    head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    # Simulate sequence embeddings: (B, seq_len, embed_dim)
    seq_emb = torch.randn(batch_size, seq_len, embed_dim)

    # Attentive pooling
    pooled = attentive_pooler(seq_emb)  # (B, 1, embed_dim)
    pooled = pooled.squeeze(1)  # (B, embed_dim)

    # Classification
    logits = head(pooled)

    assert logits.shape == (batch_size, num_classes)
    print(f"  PASSED: BCI forward mock produces correct output shape {logits.shape}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running AttentivePooler Tests for EEGLeJEPA")
    print("=" * 60)
    print()

    tests = [
        ("1. AttentivePooler initialization", test_attentive_pooler_initialization),
        ("2. AttentivePooler forward shape", test_attentive_pooler_forward_shape),
        ("3. AttentivePooler multi-query", test_attentive_pooler_multi_query),
        ("4. AttentivePooler with depth", test_attentive_pooler_with_depth),
        ("5. AttentivePooler gradient flow", test_attentive_pooler_gradient_flow),
        ("6. PROBE_TYPES constants", test_probe_types_defined),
        ("7. LeJEPAConfig probe_type", test_lejepa_config_probe_type),
        ("8. Clinical forward mock", test_clinical_forward_with_attentive_pooler),
        ("9. BCI forward mock", test_bci_forward_with_attentive_pooler),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"Test {name}...")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

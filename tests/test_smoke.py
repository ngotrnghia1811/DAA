"""
Basic smoke tests for the DAA package.

Run with:  python -m pytest tests/ -v
"""
import pytest
import torch


# ---------------------------------------------------------------------------
# Adapter module
# ---------------------------------------------------------------------------

def test_bottleneck_adapter_forward():
    from daa.modules.adapter import BottleneckAdapter
    adapter = BottleneckAdapter(hidden_size=32, bottleneck_dim=8)
    x = torch.randn(2, 5, 32)
    out = adapter(x)
    assert out.shape == x.shape


def test_domain_adapter_stack_layer():
    from daa.modules.adapter import DomainAdapterStack
    stack = DomainAdapterStack(num_layers=4, hidden_size=32, bottleneck_dim=8)
    x = torch.randn(2, 5, 32)
    out = stack.forward_layer(2, x)
    assert out.shape == x.shape


def test_domain_adapter_stack_bottleneck():
    from daa.modules.adapter import DomainAdapterStack
    stack = DomainAdapterStack(num_layers=4, hidden_size=32, bottleneck_dim=8)
    x = torch.randn(2, 5, 32)
    out, bn = stack.forward_layer_with_bottleneck(2, x)
    assert out.shape == x.shape
    assert bn.shape == (2, 5, 8)


# ---------------------------------------------------------------------------
# Wasserstein estimator
# ---------------------------------------------------------------------------

def test_wasserstein_select_samples():
    from daa.modules.wasserstein import WassersteinEstimator, select_source_samples
    est = WassersteinEstimator(input_dim=16, hidden_dim=32)
    src = torch.randn(10, 16)
    selected = select_source_samples(est, src, selection_ratio=0.5)
    assert len(selected) == 5


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------

def test_vocab_roundtrip():
    from daa.vocabs import Vocab
    v = Vocab(["None", "Life.Die", "Attack"])
    assert v.token2id("Life.Die") == 1
    assert v.id2token(2) == "Attack"
    assert v.token2id("UNKNOWN", default=-1) == -1


def test_build_event_vocab_ace05():
    from daa.vocabs import build_event_vocab
    v = build_event_vocab("event_detection")
    assert len(v) == 34
    assert v.token2id("None") == 0


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def test_scorer_perfect():
    from daa.scorer import compute_prf
    gold = [0, 1, 2, 1, 0]
    pred = [0, 1, 2, 1, 0]
    m = compute_prf(gold, pred, none_label_id=0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_scorer_all_none_pred():
    from daa.scorer import compute_prf
    gold = [0, 1, 2]
    pred = [0, 0, 0]
    m = compute_prf(gold, pred, none_label_id=0)
    assert m["precision"] == pytest.approx(0.0)
    assert m["recall"] == pytest.approx(0.0)
    assert m["f1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MLM masking utility
# ---------------------------------------------------------------------------

def test_mlm_mask_shape():
    from daa.util import apply_mlm_mask
    input_ids = torch.randint(100, 1000, (3, 20))
    masked, labels = apply_mlm_mask(
        input_ids, vocab_size=1000, mask_token_id=103,
        pad_token_id=0, mask_prob=0.15
    )
    assert masked.shape == input_ids.shape
    assert labels.shape == input_ids.shape

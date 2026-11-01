"""Unit tests for Stage 2. Run with: python -m pytest -q tests"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tta.methods import BNAdapt, NoAdapt, Tent, collect_bn_params  # noqa: E402
from tta.metrics import collapse_score, ece, error, paired_bootstrap  # noqa: E402
from tta.models import ResNet18, load_resnet18  # noqa: E402
from tta.streams import build_conditions, run_stream  # noqa: E402


def tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 10))


def fake_loader(n=64):
    def load(c, s):
        rng = np.random.default_rng(hash((c, s)) % 2**32)
        return rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8), rng.integers(0, 10, n)
    return load


# ---- methods ---------------------------------------------------------------

def test_noadapt_uses_stored_stats_and_changes_nothing():
    model = tiny_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    m = NoAdapt(model)
    m(torch.rand(16, 3, 32, 32))
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_bnadapt_uses_batch_stats_and_no_weight_changes():
    model = tiny_model()
    w_before = [p.clone() for p in model.parameters()]
    m = BNAdapt(model)
    bn = model[1]
    assert bn.training and bn.running_mean is None
    x = torch.rand(32, 3, 32, 32)
    out1, out2 = m(x), m(x)
    assert torch.allclose(out1, out2)
    for p, q in zip(model.parameters(), w_before):
        assert torch.equal(p, q)


def test_tent_updates_only_bn_affine():
    model = tiny_model()
    before = {n: p.clone() for n, p in model.named_parameters()}
    m = Tent(model, lr=1e-2)
    _, bn_names = collect_bn_params(model)
    assert bn_names == ["1.weight", "1.bias"]
    m(torch.rand(32, 3, 32, 32))
    for n, p in model.named_parameters():
        changed = not torch.equal(p, before[n])
        assert changed == (n in bn_names), n


def test_tent_predicts_before_update():
    x = torch.rand(32, 3, 32, 32)
    ref = BNAdapt(tiny_model())(x)
    out = Tent(tiny_model(), lr=1.0)(x)
    assert torch.allclose(out, ref, atol=1e-5)


def test_tent_entropy_goes_down():
    m = Tent(tiny_model(), lr=1e-2)
    x = torch.rand(64, 3, 32, 32)
    from tta.methods import softmax_entropy
    ents = [softmax_entropy(m(x)).mean().item() for _ in range(20)]
    assert ents[-1] < ents[0]


def test_tent_reset_restores_model_and_optimizer():
    model = tiny_model()
    m = Tent(model, lr=1e-2)
    x = torch.rand(32, 3, 32, 32)
    first = m(x)
    for _ in range(5):
        m(x)
    m.reset()
    assert len(m.optimizer.state) == 0
    assert torch.allclose(m(x), first, atol=1e-6)


# ---- streams ---------------------------------------------------------------

def test_conditions():
    assert len(build_conditions("episodic")) == 75
    cont = build_conditions("continual")
    assert len(cont) == 15 and all(s == 5 for _, s in cont)
    assert cont[0][0] == "gaussian_noise" and cont[-1][0] == "jpeg_compression"


def test_episodic_reset_makes_conditions_independent():
    load = fake_loader()
    conds = [("a", 5), ("b", 5)]
    both = run_stream(Tent(tiny_model(), lr=1e-2), conds, load, 16, True, "cpu", log=lambda *_: None)
    only_b = run_stream(Tent(tiny_model(), lr=1e-2), conds[1:], load, 16, True, "cpu", log=lambda *_: None)
    assert np.allclose(both[1]["logits"], only_b[0]["logits"], atol=1e-5)


def test_continual_carries_state():
    load = fake_loader()
    conds = [("a", 5), ("b", 5)]
    both = run_stream(Tent(tiny_model(), lr=1e-1), conds, load, 16, False, "cpu", log=lambda *_: None)
    only_b = run_stream(Tent(tiny_model(), lr=1e-1), conds[1:], load, 16, False, "cpu", log=lambda *_: None)
    assert not np.allclose(both[1]["logits"], only_b[0]["logits"], atol=1e-5)


def test_shuffle_keeps_index_and_labels_aligned():
    load = fake_loader()
    res = run_stream(NoAdapt(tiny_model()), [("a", 1)], load, 16, True, "cpu",
                     shuffle_seed=3, log=lambda *_: None)[0]
    _, y = load("a", 1)
    assert not np.array_equal(res["index"], np.arange(len(y)))
    assert np.array_equal(res["labels"], y[res["index"]])


# ---- metrics ---------------------------------------------------------------

def test_error_and_ece():
    logits = np.array([[10.0, 0], [0, 10.0], [10.0, 0], [0, 10.0]])
    labels = np.array([0, 1, 1, 0])
    assert error(logits, labels) == 0.5
    # Confidence ~1.0 with 50% accuracy -> ECE ~0.5.
    assert abs(ece(logits, labels) - 0.5) < 1e-3
    assert ece(np.array([[10.0, 0]] * 4), np.zeros(4, int)) < 1e-3


def test_paired_bootstrap_detects_harm():
    rng = np.random.default_rng(0)
    ref = rng.random(5000) < 0.8
    worse = ref & (rng.random(5000) < 0.9)  # loses about 8 points
    diff, low, high = paired_bootstrap(ref, worse, n_boot=200)
    assert diff > 0 and low > 0
    _, low, high = paired_bootstrap(ref, ref, n_boot=200)
    assert low <= 0 <= high


def test_collapse_score():
    balanced = np.tile(np.arange(10), 300)
    assert np.allclose(collapse_score(balanced), 0.1)
    collapsed = np.zeros(3000, int)
    assert np.allclose(collapse_score(collapsed), 1.0)
    half = np.concatenate([balanced[:2000], np.zeros(2000, int)])
    s = collapse_score(half)
    assert s[0] == pytest.approx(0.1) and s[-1] == 1.0


# ---- model -----------------------------------------------------------------

def test_resnet18_shape_params_and_checkpoint_roundtrip(tmp_path):
    model = ResNet18()
    n_params = sum(p.numel() for p in model.parameters())
    assert 11.0e6 < n_params < 11.3e6
    assert model(torch.rand(2, 3, 32, 32)).shape == (2, 10)
    path = tmp_path / "ckpt.pt"
    state = {"module." + k: v for k, v in model.state_dict().items() if not k.startswith("normalize.")}
    torch.save({"model": state, "epoch": 5}, path)
    loaded = load_resnet18(str(path)).eval()
    x = torch.rand(4, 3, 32, 32)
    assert torch.allclose(loaded(x), model.eval()(x), atol=1e-5)

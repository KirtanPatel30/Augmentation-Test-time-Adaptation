"""Evaluation metrics (Section 7 of the proposal). NumPy only, so they run without a GPU."""
import numpy as np


def softmax(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def error(logits, labels):
    """Top-1 error in [0, 1]."""
    return float((logits.argmax(1) != labels).mean())


def ece(logits, labels, n_bins=15):
    """Expected calibration error with equal-width confidence bins (Guo et al., 2017)."""
    probs = softmax(logits.astype(np.float64))
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    # Bin edges (0, 1/15], (1/15, 2/15], ... like the usual implementation.
    bins = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def paired_bootstrap(correct_ref, correct_new, n_boot=1000, seed=0, alpha=0.05):
    """Bootstrap CI of error(new) - error(ref) over the same test images.

    ``correct_ref`` and ``correct_new`` are boolean arrays aligned by image.
    Returns (diff, low, high), each in [−1, 1].
    """
    correct_ref = np.asarray(correct_ref, dtype=np.float32)
    correct_new = np.asarray(correct_new, dtype=np.float32)
    if correct_ref.shape != correct_new.shape:
        raise ValueError("arrays must be aligned by image")
    per_image = correct_ref - correct_new  # +1 where only ref is right -> new is worse
    n = len(per_image)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(0, n_boot, 100):
        idx = rng.integers(0, n, size=(min(100, n_boot - i), n))
        boots[i:i + len(idx)] = per_image[idx].mean(1)
    low, high = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(per_image.mean()), float(low), float(high)


def condition_effect(correct_ref, correct_new, **kwargs):
    """+1 if adaptation harms (CI above 0), -1 if it helps (CI below 0), else 0."""
    _, low, high = paired_bootstrap(correct_ref, correct_new, **kwargs)
    if low > 0:
        return 1
    if high < 0:
        return -1
    return 0


def collapse_score(preds, window=1000, n_classes=10):
    """Share of predictions that go to the most predicted class, per sliding window.

    ``preds`` must be in stream order. Returns one value per window position
    (stride 1). About 0.1 is normal for 10 balanced classes; 1.0 is full collapse.
    """
    preds = np.asarray(preds)
    if len(preds) < window:
        window = len(preds)
    onehot = np.zeros((len(preds) + 1, n_classes), dtype=np.int32)
    onehot[np.arange(1, len(preds) + 1), preds] = 1
    counts = np.cumsum(onehot, axis=0)
    in_window = counts[window:] - counts[:-window]
    return in_window.max(1) / window

"""Test stream protocols (Section 4.4 of the proposal).

* episodic:    all 75 conditions, batch 200, reset before each condition (official Tent protocol).
* continual:   15 corruptions at severity 5, one after another, no reset (CoTTA protocol).
* small_batch: 15 corruptions at severity 5, batch 16, reset before each condition.

The streams only change the order and batch size of the same images.
"""
import time

import numpy as np
import torch

from .data import CORRUPTIONS, SEVERITIES, to_tensor

PROTOCOLS = {
    "episodic": dict(severities=SEVERITIES, batch_size=200, reset_each=True),
    "continual": dict(severities=[5], batch_size=200, reset_each=False),
    "small_batch": dict(severities=[5], batch_size=16, reset_each=True),
}

# Tent's official learning rate is set for batch 200; smaller batches scale it linearly.
REFERENCE_BATCH_SIZE = 200


def build_conditions(protocol, corruptions=CORRUPTIONS, severities=None):
    """List of (corruption, severity) in stream order.

    Episodic follows the Tent code: severity in the outer loop. Continual and
    small-batch use one severity, so the order is just the corruption order.
    """
    severities = severities or PROTOCOLS[protocol]["severities"]
    return [(c, s) for s in severities for c in corruptions]


def run_stream(method, conditions, load_fn, batch_size, reset_each, device,
               shuffle_seed=None, reset_every=None, log=print):
    """Run ``method`` over the conditions in order and collect its predictions.

    Args:
        load_fn: (corruption, severity) -> (x uint8 (n,32,32,3), y int64 (n,)).
        reset_each: reset the method before every condition.
        shuffle_seed: if set, shuffle the image order inside each condition.
        reset_every: optional reset every N batches over the whole stream (RDumb, E6).

    Returns a list with one dict per condition: corruption, severity, logits
    (float32, stream order), labels, index (position in the original test set),
    and seconds (wall time spent in the method).
    """
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    method.reset()
    results = []
    n_batches = 0
    for corruption, severity in conditions:
        if reset_each:
            method.reset()
        x_all, y_all = load_fn(corruption, severity)
        order = np.arange(len(y_all))
        if rng is not None:
            order = rng.permutation(order)
        logits_out = []
        seconds = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            x = to_tensor(x_all[idx]).to(device, non_blocking=True)
            t0 = time.perf_counter()
            logits = method(x)
            logits_out.append(logits.float().cpu())
            seconds += time.perf_counter() - t0
            n_batches += 1
            if reset_every and n_batches % reset_every == 0:
                method.reset()
        logits = torch.cat(logits_out).numpy()
        labels = y_all[order]
        err = float((logits.argmax(1) != labels).mean())
        log(f"  {corruption:>18s} s{severity}  error {100 * err:5.2f}%  ({seconds:.1f}s)")
        results.append(dict(corruption=corruption, severity=severity, logits=logits,
                            labels=labels, index=order, seconds=seconds))
    return results

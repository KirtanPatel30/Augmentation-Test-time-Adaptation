"""Test-time methods: no adaptation, BN-adapt, and Tent.

Tent is adapted from the official code (https://github.com/DequanWang/tent,
tent.py, MIT License, Copyright (c) 2021 Dequan Wang). The repository is
archived, so the core logic is copied here with small changes:
``BNAdapt`` reuses the same model setup with no gradient step.

Every method returns the logits for a batch *before* it updates on that batch,
so each test image is predicted once and never with weights that saw it.
"""
from copy import deepcopy

import torch
import torch.nn as nn


def softmax_entropy(logits):
    """Entropy of softmax(logits), one value per row."""
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def configure_bn_for_adaptation(model):
    """Use test-batch BN statistics and freeze every parameter.

    Same as tent.configure_model, except that the BN affine parameters are
    frozen here; Tent unfreezes them afterwards.
    """
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def collect_bn_params(model):
    """BN scale and shift (gamma, beta), the only parameters Tent updates."""
    params, names = [], []
    for mod_name, m in model.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            for p_name, p in m.named_parameters():
                if p_name in ("weight", "bias"):
                    params.append(p)
                    names.append(f"{mod_name}.{p_name}")
    return params, names


class Method(nn.Module):
    """Base class. ``forward(x)`` returns logits and may adapt the model."""

    name = "base"

    def __init__(self, model):
        super().__init__()
        self.model = model

    def reset(self):
        """Go back to the source model. No-op for methods without state."""


class NoAdapt(Method):
    """Source model with stored BN statistics, no update."""

    name = "none"

    def __init__(self, model):
        super().__init__(model)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)


class BNAdapt(Method):
    """Normalize with the statistics of the current test batch (Schneider et al., 2020)."""

    name = "bn"

    def __init__(self, model):
        super().__init__(configure_bn_for_adaptation(model))

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)


class Tent(Method):
    """Tent (Wang et al., 2021): test-batch BN stats + entropy minimization on BN affine params."""

    name = "tent"

    def __init__(self, model, lr=1e-3, steps=1, betas=(0.9, 0.999), weight_decay=0.0):
        super().__init__(configure_bn_for_adaptation(model))
        if steps < 1:
            raise ValueError("Tent needs at least one step; use BNAdapt for zero steps")
        self.steps = steps
        params, _ = collect_bn_params(self.model)
        for p in params:
            p.requires_grad_(True)
        self.optimizer = torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
        self._initial_state = (deepcopy(self.model.state_dict()),
                               deepcopy(self.optimizer.state_dict()))

    def reset(self):
        model_state, opt_state = self._initial_state
        self.model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(opt_state)

    @torch.enable_grad()
    def forward(self, x):
        logits = None
        for _ in range(self.steps):
            out = self.model(x)
            if logits is None:
                # Prediction made before the first update on this batch.
                logits = out.detach()
            loss = softmax_entropy(out).mean(0)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return logits


def build_method(name, model, lr=1e-3, steps=1):
    if name == "none":
        return NoAdapt(model)
    if name == "bn":
        return BNAdapt(model)
    if name == "tent":
        return Tent(model, lr=lr, steps=steps)
    raise ValueError(f"unknown method {name!r}; choose none, bn, or tent")

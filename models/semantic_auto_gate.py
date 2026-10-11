from __future__ import annotations

import torch
import torch.nn as nn

GATE_FEATURE_NAMES = (
    "hist_scale",
    "hist_diff",
    "level_shift",
    "scale_shift",
    "lag1_autocorr",
    "branch_disagreement",
    "uncertainty_scale",
    "semantic_magnitude",
    "semantic_consistency",
)

NUM_GATE_FEATURES = len(GATE_FEATURE_NAMES)


DEFAULT_GATE_INIT_BIAS = -2.1972246


def _log1p_abs(t):
    return torch.log1p(t.abs())


def build_semantic_gate_features(
    x_norm,
    branch_preds,
    log_beta,
    semantic_residual,
    numeric_anomaly,
    signed_consistency=False,
    eps=1e-6,
):
    if x_norm.dim() != 3:
        raise ValueError(f"x_norm must be (B, S, N), got {tuple(x_norm.shape)}")
    B, S, N = x_norm.shape

    half = max(S // 2, 1)
    first = x_norm[:, :half]
    second = x_norm[:, half:] if S > half else x_norm[:, :half]

    hist_scale = _log1p_abs(x_norm.std(dim=1, unbiased=False))
    if S > 1:
        diff = x_norm[:, 1:] - x_norm[:, :-1]
        hist_diff = _log1p_abs(diff.abs().mean(dim=1))
        a = x_norm[:, :-1] - x_norm[:, :-1].mean(dim=1, keepdim=True)
        b = x_norm[:, 1:] - x_norm[:, 1:].mean(dim=1, keepdim=True)
        lag1 = (a * b).sum(dim=1) / (
            a.pow(2).sum(dim=1).sqrt() * b.pow(2).sum(dim=1).sqrt() + eps
        )
    else:
        hist_diff = torch.zeros_like(hist_scale)
        lag1 = torch.zeros_like(hist_scale)

    level_shift = _log1p_abs(second.mean(dim=1) - first.mean(dim=1))
    scale_shift = torch.log(
        (second.std(dim=1, unbiased=False) + eps)
        / (first.std(dim=1, unbiased=False) + eps)
    )

    if branch_preds is not None and len(branch_preds) > 1:
        stacked = torch.stack(list(branch_preds), dim=0)
        branch_disagreement = _log1p_abs(
            stacked.std(dim=0, unbiased=False).mean(dim=1)
        )
    else:
        branch_disagreement = torch.zeros_like(hist_scale)

    uncertainty_scale = log_beta.mean(dim=1)
    semantic_magnitude = _log1p_abs(semantic_residual.abs().mean(dim=1))

    sem_c = semantic_residual - semantic_residual.mean(dim=1, keepdim=True)
    num_c = numeric_anomaly - numeric_anomaly.mean(dim=1, keepdim=True)


    semantic_consistency = ((sem_c * num_c).sum(dim=1) / (
        sem_c.pow(2).sum(dim=1).sqrt() * num_c.pow(2).sum(dim=1).sqrt() + eps
    ))
    if not signed_consistency:
        semantic_consistency = semantic_consistency.abs()

    features = torch.stack(
        [
            hist_scale,
            hist_diff,
            level_shift,
            scale_shift,
            lag1,
            branch_disagreement,
            uncertainty_scale,
            semantic_magnitude,
            semantic_consistency,
        ],
        dim=-1,
    )
    if tuple(features.shape) != (B, N, NUM_GATE_FEATURES):
        raise RuntimeError(
            f"gate features have shape {tuple(features.shape)}, "
            f"expected {(B, N, NUM_GATE_FEATURES)}"
        )
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class SemanticAutoGate(nn.Module):

    def __init__(
        self,
        n_features=NUM_GATE_FEATURES,
        init_bias=DEFAULT_GATE_INIT_BIAS,
        zero_init=True,
        signed_output=False,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.init_bias = float(init_bias)
        self.signed_output = bool(signed_output)
        self.norm = nn.LayerNorm(self.n_features)
        self.proj = nn.Linear(self.n_features, 1)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
        else:
            nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.proj.bias, self.init_bias)

    def extra_repr(self):
        return (
            f"n_features={self.n_features}, init_bias={self.init_bias:.6f}, "
            f"signed_output={self.signed_output}"
        )

    def forward(self, features):
        if features.shape[-1] != self.n_features:
            raise ValueError(
                f"gate expects {self.n_features} features, got {features.shape[-1]}"
            )
        logits = self.proj(self.norm(features)).squeeze(-1)
        if self.signed_output:


            return torch.tanh(0.5 * logits)
        return torch.sigmoid(logits)


class SemanticAutoDecayGate(nn.Module):

    def __init__(
        self,
        n_features=NUM_GATE_FEATURES,
        init_bias=DEFAULT_GATE_INIT_BIAS,
        zero_init=True,
        log_tau_range=3.0,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.init_bias = float(init_bias)
        self.log_tau_range = float(log_tau_range)
        self.norm = nn.LayerNorm(self.n_features)
        self.proj = nn.Linear(self.n_features, 2)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
        else:
            nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        with torch.no_grad():
            self.proj.bias[0].fill_(self.init_bias)
            self.proj.bias[1].zero_()

    def extra_repr(self):
        return (
            f"n_features={self.n_features}, init_bias={self.init_bias:.6f}, "
            f"log_tau_range={self.log_tau_range:.3f}"
        )

    def forward(self, features, pred_len):
        if features.shape[-1] != self.n_features:
            raise ValueError(
                f"gate expects {self.n_features} features, got {features.shape[-1]}"
            )
        logits = self.proj(self.norm(features))
        amplitude = torch.sigmoid(logits[..., 0])
        tau = float(pred_len) * torch.exp(
            self.log_tau_range * torch.tanh(logits[..., 1])
        )
        lead = torch.arange(
            int(pred_len), device=features.device, dtype=features.dtype
        ) + 0.5
        temporal_gate = amplitude.unsqueeze(1) * torch.exp(
            -lead.view(1, -1, 1) / tau.unsqueeze(1).clamp_min(1e-6)
        )
        return temporal_gate, amplitude, tau

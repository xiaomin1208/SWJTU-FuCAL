import argparse
import csv
import faulthandler
import hashlib
import json
import os
import platform
import random
import sys
import time
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import torch
if not hasattr(torch.amp, 'GradScaler'):
    def _grad_scaler(device='cuda', **kwargs):
        return torch.cuda.amp.GradScaler(**kwargs)
    torch.amp.GradScaler = _grad_scaler
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset

from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from models.TimeCMA import DIAG
from models.FuCAL import FuCAL
from utils.llm_backbone import get_llm_hidden_size
from utils.metrics import MAE, metric
from utils.var_mixer_decision import should_use_var_mixer
from utils.common import is_large_dataset, get_raw_model


faulthandler.enable()
torch.cuda.empty_cache()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

MIN_EPOCHS = 50

VALID_ABLATIONS = [
    'full',
    'no_cp',
    'no_ts_backbone',
    'no_semantic',
    'simple_concat_fusion',
    'cp_cqr',
    'cp_enbpi',
    'no_fuzzy',
]

SKIP_JOINT_ABLATIONS = {'no_cp', 'cp_cqr', 'cp_enbpi'}

VALID_TREND_MODES = ['auto', 'multiscale', 'simple', 'off']
MAX_CACHE_MEM_GB = 8.0
_is_large_dataset = is_large_dataset


def _sha256_file(path):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_run_provenance(args):
    source_root = os.path.dirname(os.path.abspath(__file__))
    tracked = {
        'train.py': os.path.join(source_root, 'train.py'),
        'models/FuCAL.py': os.path.join(source_root, 'models', 'FuCAL.py'),
        'init_checkpoint': getattr(args, 'init_checkpoint', None),
    }
    payload = {
        'created_unix': time.time(),
        'argv': list(sys.argv),
        'cwd': os.getcwd(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'dataset': args.data_path,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'seed': args.seed,
        'eval_drop_last': str(getattr(args, 'eval_drop_last', 'auto')),
        'train_drop_last': str(getattr(args, 'train_drop_last', 'auto')),
        'legacy_adapter_protocol': bool(
            getattr(args, 'legacy_adapter_protocol', False)),
        'files_sha256': {name: _sha256_file(path) for name, path in tracked.items()},
        'fucal_environment': {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith('FUCAL_')
        },
    }
    output = os.path.join(args.result_dir, 'provenance.json')
    temp = output + '.tmp'
    with open(temp, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temp, output)


def gaussian_z(alpha):
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def conformal_qhat(preds_cal, trues_cal, betas_cal, alpha=0.05):
    if betas_cal is None or preds_cal is None:
        return gaussian_z(alpha)
    s = (np.abs(np.asarray(trues_cal) - np.asarray(preds_cal))
         / (np.asarray(betas_cal) + 1e-8)).reshape(-1)
    s = s[np.isfinite(s)]
    n = s.shape[0]
    if n == 0:
        return gaussian_z(alpha)
    level = min((1.0 - alpha) * (1.0 + 1.0 / n), 1.0)
    return float(np.quantile(s, level))


def _memberships_from_windows(x, knots, n_axes, n_levels):
    import numpy as _np
    w = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-5)
    tt = _np.arange(w.shape[1], dtype=_np.float64)
    tt = (tt - tt.mean()) / (tt.std() + 1e-9)
    d = _np.diff(w, axis=1)
    tail = max(1, d.shape[1] // 4)
    slope = _np.einsum("bln,l->bn", w - w.mean(1, keepdims=True), tt) / (tt @ tt)
    f = _np.stack([w.mean(1), w.std(1), slope, d.std(1),
                   w.max(1) - w.min(1), d[:, -tail:].mean(1)], axis=1)
    f = f[:, :n_axes]
    kn = _np.transpose(knots[:n_axes], (1, 0, 2))
    v = _np.transpose(f, (0, 2, 1))[..., None]
    l, c, r = kn[None, ..., :-2], kn[None, ..., 1:-1], kn[None, ..., 2:]
    u = _np.clip(_np.minimum((v - l) / (c - l + 1e-6), (r - v) / (r - c + 1e-6)), 0, None)
    return u / (u.sum(-1, keepdims=True) + 1e-6)


def fuzzy_conformal_qhat(preds_cal, trues_cal, betas_cal, u_cal, u_test,
                         alpha=0.05, crisp=False, min_weight=20.0):
    import numpy as _np
    s = (_np.abs(_np.asarray(trues_cal) - _np.asarray(preds_cal))
         / (_np.asarray(betas_cal) + 1e-8))
    s = s.reshape(s.shape[0], s.shape[1], -1).mean(axis=1)
    A = u_cal.shape[2]
    uc = u_cal.reshape(u_cal.shape[0], u_cal.shape[1], -1)
    ut = u_test.reshape(u_test.shape[0], u_test.shape[1], -1)
    if crisp:
        gc = u_cal.argmax(-1).reshape(u_cal.shape[0], u_cal.shape[1], A)
        gt = u_test.argmax(-1).reshape(u_test.shape[0], u_test.shape[1], A)
    nt, N = ut.shape[0], ut.shape[1]
    q = _np.empty((nt, N), dtype=float)
    glob = _np.quantile(s, min((1 - alpha) * (1 + 1.0 / max(len(s), 1)), 1.0), axis=0)
    for n in range(N):
        sn = s[:, n]
        order = _np.argsort(sn)
        sn_sorted = sn[order]
        for tix in range(nt):
            if crisp:
                wgt = (gc[:, n, :] == gt[tix, n, :][None, :]).mean(1).astype(float)
            else:
                wgt = uc[:, n, :] @ ut[tix, n, :]
            tot = wgt.sum()
            if tot < min_weight / max(len(sn), 1):
                q[tix, n] = glob[n]
                continue
            cw = _np.cumsum(wgt[order]) / tot
            j = int(_np.searchsorted(cw, 1 - alpha))
            q[tix, n] = sn_sorted[min(j, len(sn_sorted) - 1)]
    return q, glob


def aci_eval(cal_preds, cal_trues, cal_betas, preds, trues, betas,
             alpha=0.05, gamma=0.03):
    cs = (np.abs(np.asarray(cal_trues) - np.asarray(cal_preds))
          / (np.asarray(cal_betas) + 1e-8)).reshape(-1)
    cs = cs[np.isfinite(cs)]
    if cs.size == 0:
        return float('nan'), float('nan'), float('nan')
    P = np.asarray(preds, dtype=float)
    T = np.asarray(trues, dtype=float)
    B = np.maximum(np.asarray(betas, dtype=float), 1e-8)
    N = P.shape[0]
    a_t = alpha
    covs = np.empty(N); wids = np.empty(N); winks = np.empty(N)
    two_a = 2.0 / alpha
    for t in range(N):
        lvl = min(max(1.0 - a_t, 0.0), 1.0)
        q = float(np.quantile(cs, lvl)) if lvl > 0.0 else 0.0
        lo = P[t] - q * B[t]; hi = P[t] + q * B[t]
        cov = (T[t] >= lo) & (T[t] <= hi)
        c = float(cov.mean())
        covs[t] = c
        wids[t] = float((hi - lo).mean())
        w = (hi - lo) + two_a * np.maximum(lo - T[t], 0.0) + two_a * np.maximum(T[t] - hi, 0.0)
        winks[t] = float(w.mean())
        a_t = a_t + gamma * (alpha - (1.0 - c))
    return float(covs.mean()), float(wids.mean()), float(winks.mean())


def compute_full_metrics(preds, trues, betas=None, pred_len=None, alpha=0.05, q_override=None):
    results = {}
    preds = np.asarray(preds)
    trues = np.asarray(trues)
    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(mse)

    mask = np.abs(trues) > 1e-4
    mape = np.mean(np.abs((preds[mask] - trues[mask]) / trues[mask])) * 100 if mask.sum() > 0 else float('nan')

    denom = (np.abs(preds) + np.abs(trues)) / 2 + 1e-8
    smape = np.mean(np.abs(preds - trues) / denom) * 100

    results.update({
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'SMAPE': float(smape),
    })

    nominal = 1.0 - alpha
    cov_key = f'Coverage_{int(nominal * 100):02d}'
    z = gaussian_z(alpha)


    q = z if q_override is None else float(q_override)
    if betas is not None:
        betas = np.asarray(betas)
        lower = preds - q * betas
        upper = preds + q * betas
        width = upper - lower
        covered = ((trues >= lower) & (trues <= upper)).astype(float)
        picp = float(covered.mean())
        mpiw = float(width.mean())

        y_range = float(np.nanmax(trues) - np.nanmin(trues))
        pinaw = float(mpiw / (y_range + 1e-8))

        below = np.maximum(lower - trues, 0.0)
        above = np.maximum(trues - upper, 0.0)
        winkler = width + (2.0 / alpha) * below + (2.0 / alpha) * above

        abs_err = np.abs(preds - trues)
        nonconformity = abs_err / (z * betas + 1e-8)
        results.update({
            cov_key: picp,
            'PICP': picp,
            'Miscoverage': float(1.0 - picp),
            'ACE': float(abs(picp - nominal)),
            'MPIW': mpiw,
            'PINAW': pinaw,
            'Interval_Width': mpiw,
            'WinklerScore': float(winkler.mean()),
            'IntervalScore': float(winkler.mean()),
            'Mean_Beta': float(betas.mean()),
            'Median_Beta': float(np.median(betas)),
            'Mean_Nonconformity': float(nonconformity.mean()),
            'Q90_Nonconformity': float(np.quantile(nonconformity.reshape(-1), 0.90)),
            'Q95_Nonconformity': float(np.quantile(nonconformity.reshape(-1), 0.95)),
        })
    else:
        for k in [cov_key, 'PICP', 'Miscoverage', 'ACE', 'MPIW', 'PINAW',
                  'Interval_Width', 'WinklerScore', 'IntervalScore', 'Mean_Beta',
                  'Median_Beta', 'Mean_Nonconformity', 'Q90_Nonconformity',
                  'Q95_Nonconformity']:
            results[k] = float('nan')

    if pred_len is not None and pred_len > 1:
        cut1 = max(1, pred_len // 3)
        cut2 = max(cut1 + 1, 2 * pred_len // 3)
        segments = {
            f'Short(1-{cut1})': (0, cut1),
            f'Mid({cut1 + 1}-{cut2})': (cut1, cut2),
            f'Long({cut2 + 1}-{pred_len})': (cut2, pred_len),
        }
        for seg_name, (s, e) in segments.items():
            seg_pred, seg_true = preds[:, s:e, :], trues[:, s:e, :]
            results[f'{seg_name}_MSE'] = float(np.mean((seg_pred - seg_true) ** 2))
            results[f'{seg_name}_MAE'] = float(np.mean(np.abs(seg_pred - seg_true)))
            if betas is not None:
                seg_lower, seg_upper = lower[:, s:e, :], upper[:, s:e, :]
                seg_width = seg_upper - seg_lower
                seg_cov = ((seg_true >= seg_lower) & (seg_true <= seg_upper)).astype(float)
                seg_below = np.maximum(seg_lower - seg_true, 0.0)
                seg_above = np.maximum(seg_true - seg_upper, 0.0)
                seg_winkler = seg_width + (2.0 / alpha) * seg_below + (2.0 / alpha) * seg_above
                results[f'{seg_name}_PICP'] = float(seg_cov.mean())
                results[f'{seg_name}_MPIW'] = float(seg_width.mean())
                results[f'{seg_name}_ACE'] = float(abs(seg_cov.mean() - nominal))
                results[f'{seg_name}_WinklerScore'] = float(seg_winkler.mean())
    return results


def _safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return float('nan')
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return float((x * y).sum() / denom) if denom > 1e-12 else float('nan')


def _rankdata_average(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind='mergesort')
    sorted_a = a[order]
    ranks_sorted = np.zeros(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and sorted_a[j] == sorted_a[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def _safe_spearman(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float('nan')
    return _safe_corr(_rankdata_average(x[mask]), _rankdata_average(y[mask]))


def _sample_xy(x, y, max_points=25000, seed=2024):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if len(x) <= max_points:
        return x, y
    idx = np.random.default_rng(seed).choice(len(x), size=max_points, replace=False)
    return x[idx], y[idx]


def _save_rows_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_cp_explainability_report(preds, trues, betas, alpha=0.05):
    abs_err = np.abs(preds - trues)
    z = gaussian_z(alpha)
    lower, upper = preds - z * betas, preds + z * betas
    covered = ((trues >= lower) & (trues <= upper)).astype(np.float32)
    width = upper - lower
    beta_flat, err_flat = betas.reshape(-1), abs_err.reshape(-1)
    width_flat, cover_flat = width.reshape(-1), covered.reshape(-1)

    pearson = _safe_corr(beta_flat, err_flat)
    spearman = _safe_spearman(beta_flat, err_flat)

    valid = np.isfinite(beta_flat) & np.isfinite(err_flat)
    beta_valid, err_valid = beta_flat[valid], err_flat[valid]
    width_valid, cover_valid = width_flat[valid], cover_flat[valid]

    decile_rows = []
    if beta_valid.size > 0:
        edges = np.quantile(beta_valid, np.linspace(0.0, 1.0, 11))
        edges[0] -= 1e-12
        edges[-1] += 1e-12
        for d in range(10):
            lo, hi = edges[d], edges[d + 1]
            if d == 9:
                mask = (beta_valid >= lo) & (beta_valid <= hi)
            else:
                mask = (beta_valid >= lo) & (beta_valid < hi)

            if mask.sum() == 0:
                row = dict(
                    decile=d + 1, beta_lo=float(lo), beta_hi=float(hi), count=0,
                    mean_beta=float('nan'), mean_abs_error=float('nan'),
                    mean_width=float('nan'), coverage=float('nan')
                )
            else:
                row = dict(
                    decile=d + 1, beta_lo=float(lo), beta_hi=float(hi), count=int(mask.sum()),
                    mean_beta=float(beta_valid[mask].mean()),
                    mean_abs_error=float(err_valid[mask].mean()),
                    mean_width=float(width_valid[mask].mean()),
                    coverage=float(cover_valid[mask].mean())
                )
            decile_rows.append(row)

    decile_errors = np.array([r['mean_abs_error'] for r in decile_rows], dtype=np.float64)
    finite_err = np.isfinite(decile_errors)
    monotonicity = float((np.diff(decile_errors[finite_err]) >= -1e-12).mean()) if finite_err.sum() >= 2 else float('nan')

    horizon_rows = []
    for h in range(preds.shape[1]):
        horizon_rows.append(dict(
            horizon=h + 1,
            mean_abs_error=float(abs_err[:, h, :].mean()),
            mean_beta=float(betas[:, h, :].mean()),
            mean_width=float(width[:, h, :].mean()),
            coverage=float(covered[:, h, :].mean()),
        ))

    metrics = {
        'CP_Beta_Error_Pearson': float(pearson),
        'CP_Beta_Error_Spearman': float(spearman),
        'CP_Decile_Monotonicity': float(monotonicity),
        f'CP_CoverageGap_{int((1-alpha)*100):02d}': float(abs(covered.mean() - (1.0 - alpha))),
        'CP_Width_Error_Pearson': float(_safe_corr(width_flat, err_flat)),
    }
    return {
        'metrics': metrics,
        'deciles': decile_rows,
        'horizons': horizon_rows,
        'abs_err_flat': err_flat,
        'beta_flat': beta_flat,
    }


def merge_component_metrics(base, extra):
    out = dict(base)
    if extra:
        out.update(extra)
    return out


def plot_cp_scatter(beta_flat, err_flat, save_path, pearson, spearman):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x, y = _sample_xy(beta_flat, err_flat)
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=5, alpha=0.18)
    plt.xlabel('Predicted beta')
    plt.ylabel('Absolute error')
    plt.title(f'CP: beta vs abs error\nPearson={pearson:.4f}, Spearman={spearman:.4f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_decile_diagnostics(decile_rows, save_path, alpha=0.05):
    if not decile_rows:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    deciles = [r['decile'] for r in decile_rows]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
    axes[0].plot(deciles, [r['mean_beta'] for r in decile_rows], marker='o', label='Mean beta')
    axes[0].plot(deciles, [r['mean_abs_error'] for r in decile_rows], marker='s', label='Mean abs error')
    axes[0].set_xlabel('Beta decile')
    axes[0].set_ylabel('Scale')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    ax2 = axes[1]
    ax2.plot(deciles, [r['coverage'] for r in decile_rows], marker='o', label='Coverage')
    ax2.axhline(1.0 - alpha, linestyle='--', label=f'Target {(1.0-alpha)*100:.0f}%')
    ax2.set_xlabel('Beta decile')
    ax2.set_ylabel('Coverage')
    ax2.grid(True, alpha=0.3)
    ax3 = ax2.twinx()
    ax3.plot(deciles, [r['mean_width'] for r in decile_rows], marker='s', label='Width')
    ax3.set_ylabel('Width')
    lines_a, labels_a = ax2.get_legend_handles_labels()
    lines_b, labels_b = ax3.get_legend_handles_labels()
    ax2.legend(lines_a + lines_b, labels_a + labels_b, loc='best')
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_horizon_diagnostics(horizon_rows, save_path, alpha=0.05):
    if not horizon_rows:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    h = [r['horizon'] for r in horizon_rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    axes[0].plot(h, [r['mean_beta'] for r in horizon_rows], label='Mean beta')
    axes[0].plot(h, [r['mean_abs_error'] for r in horizon_rows], label='Mean abs error')
    axes[0].set_xlabel('Horizon')
    axes[0].set_ylabel('Scale')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    ax2 = axes[1]
    ax2.plot(h, [r['coverage'] for r in horizon_rows], label='Coverage')
    ax2.axhline(1.0 - alpha, linestyle='--', label=f'Target {(1.0-alpha)*100:.0f}%')
    ax2.set_xlabel('Horizon')
    ax2.set_ylabel('Coverage')
    ax2.grid(True, alpha=0.3)
    ax3 = ax2.twinx()
    ax3.plot(h, [r['mean_width'] for r in horizon_rows], label='Width')
    ax3.set_ylabel('Width')
    lines_a, labels_a = ax2.get_legend_handles_labels()
    lines_b, labels_b = ax3.get_legend_handles_labels()
    ax2.legend(lines_a + lines_b, labels_a + labels_b, loc='best')
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def get_model_config(seq_len, pred_len, data_path=''):
    is_minute = 'ETTm' in data_path or 'Solar' in data_path or 'Traffic' in data_path
    if 'Traffic' in data_path:
        return dict(patch_len_list=[16, 48, 96, 128], stride=8, channel=128, dropout=0.3, lr=5e-5, d_ff=128)
    if 'Solar' in data_path:
        return dict(patch_len_list=[16, 32, 48, 96], stride=16, channel=128, dropout=0.4, lr=5e-5, d_ff=512)
    if seq_len <= 96:
        config = dict(patch_len_list=[8, 16, 32, 48], stride=8, channel=192, dropout=0.1, lr=5e-5, d_ff=512)
    elif seq_len <= 192:
        config = dict(patch_len_list=[8, 16, 32, 48], stride=8, channel=40, dropout=0.35, lr=3e-5, d_ff=384)
    elif seq_len <= 336:
        config = dict(patch_len_list=[16, 32, 48, 96], stride=16, channel=32, dropout=0.4, lr=2e-5, d_ff=256)
    else:
        config = dict(patch_len_list=[16, 48, 96, 128], stride=16, channel=32, dropout=0.4, lr=2e-5, d_ff=256)
    if not is_minute and seq_len > 96:
        if is_large_dataset(data_path):
            config.update(channel=512, dropout=0.5, lr=0.001, d_ff=512)
        else:
            config.update(channel=48, dropout=0.3, lr=5e-5, d_ff=512)
    return config


def _estimate_dataset_memory_gb(dataset, n_probe=3):
    n = len(dataset)
    if n == 0:
        return 0.0
    total = 0
    for idx in range(min(n_probe, n)):
        for item in dataset[idx]:
            if isinstance(item, np.ndarray):
                total += item.nbytes
            elif isinstance(item, torch.Tensor):
                total += item.nelement() * item.element_size()
            else:
                total += 8
    return (total / min(n_probe, n) * n) / (1024 ** 3)


class PassthroughDataset(Dataset):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y, xm, ym, emb = self.ds[idx]
        return (
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(xm, dtype=torch.float32),
            torch.as_tensor(ym, dtype=torch.float32),
            torch.as_tensor(emb, dtype=torch.float32),
        )


class CachedDataset(Dataset):
    def __init__(self, original_dataset, cache_dir=None, tag='train', max_mem_gb=MAX_CACHE_MEM_GB):
        super().__init__()
        n = len(original_dataset)
        est = _estimate_dataset_memory_gb(original_dataset)
        print(f" >>> [Cache] {tag}: {n} samples, ~{est:.2f} GB (limit {max_mem_gb:.1f})")
        if est > max_mem_gb:
            print(" >>> [Cache] Passthrough mode")
            self._pt = PassthroughDataset(original_dataset)
            self._cached = False
            return

        self._pt = None
        self._cached = True
        cache_path = os.path.join(cache_dir, f'cached_{tag}.pt') if cache_dir else None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        if cache_path and os.path.exists(cache_path):
            try:
                c = torch.load(cache_path, weights_only=True)
            except Exception:
                c = torch.load(cache_path, weights_only=False)
            self.x, self.y = c['x'], c['y']
            self.x_mark, self.y_mark = c['x_mark'], c['y_mark']
            self.emb = c['emb']
            return

        xl, yl, xm, ym, el = [], [], [], [], []
        for i in range(n):
            x, y, xmk, ymk, e = original_dataset[i]
            xl.append(torch.tensor(x, dtype=torch.float32))
            yl.append(torch.tensor(y, dtype=torch.float32))
            xm.append(torch.tensor(xmk, dtype=torch.float32))
            ym.append(torch.tensor(ymk, dtype=torch.float32))
            el.append(torch.tensor(e, dtype=torch.float32))

        self.x = torch.stack(xl)
        self.y = torch.stack(yl)
        self.x_mark = torch.stack(xm)
        self.y_mark = torch.stack(ym)
        self.emb = torch.stack(el)

        if cache_path:
            torch.save(
                dict(x=self.x, y=self.y, x_mark=self.x_mark, y_mark=self.y_mark, emb=self.emb),
                cache_path,
            )

    def __len__(self):
        return len(self._pt) if not self._cached else len(self.x)

    def __getitem__(self, idx):
        if not self._cached:
            return self._pt[idx]
        return self.x[idx], self.y[idx], self.x_mark[idx], self.y_mark[idx], self.emb[idx]


class PairedEmbeddingDataset(Dataset):
    def __init__(self, primary, alternate):
        if len(primary) != len(alternate):
            raise ValueError(
                f"Paired embedding datasets differ in length: "
                f"{len(primary)} != {len(alternate)}"
            )
        self.primary = primary
        self.alternate = alternate

    def __len__(self):
        return len(self.primary)

    def __getitem__(self, idx):
        x, y, xm, ym, emb = self.primary[idx]
        x_alt, y_alt, xm_alt, ym_alt, emb_alt = self.alternate[idx]


        if x.shape != x_alt.shape or y.shape != y_alt.shape:
            raise RuntimeError(f"Paired sample shape mismatch at index {idx}")
        return x, y, xm, ym, torch.cat((emb, emb_alt), dim=0)


class WarmupCosineRestartScheduler:
    def __init__(self, optimizer, warmup_epochs, min_lr=1e-6, T_0=20, T_mult=2):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_epoch = 0
        self.cycle_epoch = 0
        self.T_cur = T_0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            s = self.current_epoch / self.warmup_epochs
            for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = max(self.min_lr, blr * s)
        else:
            if self.cycle_epoch >= self.T_cur:
                self.cycle_epoch = 0
                self.T_cur *= self.T_mult
            s = 0.5 * (1.0 + np.cos(np.pi * self.cycle_epoch / self.T_cur))
            for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = self.min_lr + (blr - self.min_lr) * s
            self.cycle_epoch += 1

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


class ConformalPredictionLoss(nn.Module):
    def __init__(
        self,
        alpha=0.05,
        mae_weight=0.3,
        lambda_crps=0.08,
        lambda_calib=0.03,
        lambda_sharp=0.002,
        lambda_coverage=5.0,
        lambda_nll=0.1,
        beta_safe=1e-6,
        fredf_alpha=0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.mae_weight = mae_weight
        self.fredf_alpha = float(fredf_alpha or 0.0)
        self.target_coverage = 1.0 - alpha
        self.lambda_crps = lambda_crps
        self.lambda_calib = lambda_calib
        self.lambda_sharp = lambda_sharp
        self.lambda_coverage = lambda_coverage
        self.lambda_nll = lambda_nll
        self.beta_safe = beta_safe
        self.dynamic_min_beta = 0.0
        self.crps_warmup_steps = 0
        self.crps_max_steps = 1500
        self.last_components = {}
        self.z_value = gaussian_z(alpha)

        self.acl_nobetaw = os.environ.get('FUCAL_ACL_NOBETAW', '0') == '1'
        self.acl_minimal = os.environ.get('FUCAL_ACL_MINIMAL', '0') == '1'
        self.acl_softq   = os.environ.get('FUCAL_ACL_SOFTQ',   '0') == '1'

        self.acl_decouple = os.environ.get('FUCAL_ACL_DECOUPLE', '0') == '1'
        self.lambda_corr  = float(os.environ.get('FUCAL_ACL_LAMBDA_CORR', '0.05'))

        self.betaw_exp   = float(os.environ.get('FUCAL_BETAW_EXP', '1.2'))
        self.betaw_floor = float(os.environ.get('FUCAL_BETAW_FLOOR', '0.1'))
        if any([self.acl_nobetaw, self.acl_minimal, self.acl_softq, self.acl_decouple]):
            print(f" >>> [ACL-EXP] nobetaw={self.acl_nobetaw} minimal={self.acl_minimal} softq={self.acl_softq} decouple={self.acl_decouple}")

    def _softquantile_calib(self, mu, beta, targets, t=15.0):


        r = (targets - mu).abs() / (beta + 1e-8)
        exceed = torch.sigmoid(t * (r - 1.0)).mean()
        return (exceed - self.alpha) ** 2

    def set_min_beta(self, mse_val):


        if mse_val > 0:
            _m = float(os.environ.get('FUCAL_MINBETA_MULT', '0.5'))
            self.dynamic_min_beta = np.sqrt(mse_val) * _m

    def step_crps_warmup(self):
        self.crps_warmup_steps = min(self.crps_warmup_steps + 1, self.crps_max_steps)

    def _current_crps_weight(self):
        return self.lambda_crps * (self.crps_warmup_steps / self.crps_max_steps)

    def _crps_gaussian(self, mu, sigma, targets):
        z = (targets - mu) / (sigma + 1e-8)
        phi_z = torch.sigmoid(1.7 * z)
        pdf_z = torch.exp(-0.5 * z.pow(2)) / np.sqrt(2 * np.pi)
        return (sigma * (z * (2 * phi_z - 1) + 2 * pdf_z - 1.0 / np.sqrt(np.pi))).mean()

    def _calibration_loss(self, beta, mu, targets):
        abs_err = (targets - mu).abs().detach()
        e = (abs_err - abs_err.mean()) / (abs_err.std() + 1e-8)
        b = (beta - beta.mean()) / (beta.std() + 1e-8)
        return -(e * b).mean()

    def _soft_coverage_penalty(self, mu, beta, targets):
        lo, hi = mu - self.z_value * beta, mu + self.z_value * beta
        t = 15.0
        sc = torch.sigmoid(t * (targets - lo) / (beta + 1e-8)) * torch.sigmoid(t * (hi - targets) / (beta + 1e-8))
        return F.relu(self.target_coverage - sc.mean()) ** 2

    def _gaussian_nll(self, mu, beta, targets):
        var = beta.pow(2) + 1e-8
        return (0.5 * (torch.log(var) + (targets - mu).pow(2) / var)).mean()

    def forward(self, mu, beta, targets, training_stage='warmup', correction_l1=None):
        targets = targets.float()
        bc = torch.clamp(beta, min=self.dynamic_min_beta) + self.beta_safe
        se = (mu - targets) ** 2
        ae = (mu - targets).abs()

        if self.acl_nobetaw or self.acl_decouple:
            w = torch.ones_like(bc)
        else:
            w = 1.0 / (bc.detach().pow(self.betaw_exp) + self.betaw_floor)
            w = w / w.mean()
        lm = (1.0 - self.mae_weight) * (se * w).mean() + self.mae_weight * (ae * w).mean()


        if self.fredf_alpha > 0.0:
            _d = (mu - targets).float()
            _F = torch.fft.rfft(_d, dim=1, norm='ortho')
            _lf = _F.abs().mean()
            lm = (1.0 - self.fredf_alpha) * lm + self.fredf_alpha * _lf.to(lm.dtype)

        lo, hi = mu - self.z_value * bc, mu + self.z_value * bc
        cov = ((targets >= lo) & (targets <= hi)).float().mean().item()
        wid = (hi - lo).mean().item()
        self.last_components = {
            'mse': (se * w).mean().item(),
            'mae': (ae * w).mean().item(),
            'main': lm.item(),
            'coverage': cov,
            'width': wid,
        }

        if training_stage == 'warmup':
            nll = self._gaussian_nll(mu.detach(), bc, targets.detach())
            total = lm + self.lambda_nll * nll
            self.last_components.update(loss_nll=nll.item(), total=total.item())
            return total, (se * w).mean().item(), cov, wid

        self.step_crps_warmup()
        cw = self._current_crps_weight()

        mu_u = mu.detach() if self.acl_decouple else mu
        nll_j = self.lambda_nll * 0.5 * self._gaussian_nll(mu.detach(), bc, targets.detach())
        if self.acl_softq or self.acl_decouple:
            cov_term = self.lambda_coverage * self._softquantile_calib(mu_u, bc, targets)
        else:
            cov_term = self.lambda_coverage * self._soft_coverage_penalty(mu_u, bc, targets)
        if self.acl_minimal or self.acl_decouple:

            total = lm + nll_j + cov_term
        else:
            total = (
                lm
                + cw * self._crps_gaussian(mu_u, bc, targets)
                + self.lambda_calib * self._calibration_loss(bc, mu, targets)
                + self.lambda_sharp * bc.mean()
                + cov_term
                + nll_j
            )

        if self.acl_decouple and (correction_l1 is not None):
            total = total + self.lambda_corr * correction_l1
        self.last_components['total'] = total.item()
        return total, (se * w).mean().item(), cov, wid


def calibrate_posthoc(model, calib_loader, device, method='cqr', alpha=0.05, use_amp=True):
    model.eval()
    all_residuals = []
    with torch.no_grad():
        for x, y, xm, ym, emb in calib_loader:
            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                mu, *_ = model(x.to(device), xm.to(device), emb.to(device))
            residuals = (y.to(device) - mu).abs().cpu().numpy()
            all_residuals.append(residuals)

    all_residuals = np.concatenate(all_residuals, axis=0)
    n_cal = all_residuals.shape[0]

    if method == 'cqr':
        q_level = min((1.0 - alpha) * (1.0 + 1.0 / max(n_cal, 1)), 1.0)
        posthoc_q = np.quantile(all_residuals, q_level, axis=0)
        print(
            f" >>> [PostHoc-CQR] n_cal={n_cal}, q_level={q_level:.4f}, "
            f"mean_q={posthoc_q.mean():.4f}, std_q={posthoc_q.std():.4f}"
        )
    elif method == 'enbpi':
        q_level = min((1.0 - alpha) * (1.0 + 1.0 / max(n_cal, 1)), 1.0)
        decay = np.exp(np.linspace(-2.0, 0.0, n_cal))
        decay /= decay.sum()

        posthoc_q = np.zeros(all_residuals.shape[1:], dtype=np.float64)
        for h in range(all_residuals.shape[1]):
            for n in range(all_residuals.shape[2]):
                col = all_residuals[:, h, n]
                sort_idx = np.argsort(col)
                sorted_vals = col[sort_idx]
                sorted_weights = decay[sort_idx]
                cum_w = np.cumsum(sorted_weights)
                idx = np.searchsorted(cum_w, q_level)
                idx = min(idx, n_cal - 1)
                posthoc_q[h, n] = sorted_vals[idx]

        print(
            f" >>> [PostHoc-EnbPI] n_cal={n_cal}, q_level={q_level:.4f}, "
            f"mean_q={posthoc_q.mean():.4f}, std_q={posthoc_q.std():.4f}"
        )
    else:
        raise ValueError(f"Unknown posthoc method: {method}")

    return posthoc_q


def predict_with_posthoc(model, dataloader, device, posthoc_q, alpha=0.05, use_amp=True):
    model.eval()
    ap, at, ab = [], [], []
    q_tensor = torch.tensor(posthoc_q, dtype=torch.float32).to(device)
    z = gaussian_z(alpha)
    with torch.no_grad():
        for x, y, xm, ym, emb in dataloader:
            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                mu, *_ = model(x.to(device), xm.to(device), emb.to(device))
            beta = q_tensor.unsqueeze(0).expand(mu.shape[0], -1, -1) / z
            ap.append(mu.cpu().numpy())
            at.append(y.numpy())
            ab.append(beta.cpu().numpy())
    return np.concatenate(ap), np.concatenate(at), np.concatenate(ab)


def create_optimizer(params_or_groups, lr, weight_decay):
    try:
        return optim.AdamW(params_or_groups, lr=lr, weight_decay=weight_decay, fused=torch.cuda.is_available())
    except TypeError:
        return optim.AdamW(params_or_groups, lr=lr, weight_decay=weight_decay)


def is_better_pair(a_mse, a_mae, b_mse, b_mae, tol=1e-12):
    if a_mse < b_mse - tol:
        return True
    if abs(a_mse - b_mse) <= tol and a_mae < b_mae - tol:
        return True
    return False


def improves_both(a_mse, a_mae, b_mse, b_mae, tol=1e-12):
    return a_mse < b_mse - tol and a_mae < b_mae - tol


def set_frozen_modules_eval(model):
    frozen = 0
    for module in model.modules():
        params = list(module.parameters(recurse=True))


        if not params:
            continue
        if any(p.requires_grad for p in params):
            continue
        module.eval()
        frozen += 1
    return frozen


class trainer:
    def __init__(self, args, scaler):
        self.args = args
        self.device = args.device
        self.is_large = is_large_dataset(args.data_path)
        ablation = args.ablation

        config = get_model_config(args.seq_len, args.pred_len, args.data_path)
        if args.channel is not None:
            config['channel'] = args.channel
        if args.dropout_n is not None:
            config['dropout'] = args.dropout_n
        if args.lr is not None:
            config['lr'] = args.lr
        if args.d_ff is not None:
            config['d_ff'] = args.d_ff
        if args.stride is not None:
            config['stride'] = args.stride
        if args.patch_len_list is not None:
            config['patch_len_list'] = args.patch_len_list

        print(f"\n >>> [AutoConfig] {args.data_path} seq={args.seq_len} pred={args.pred_len}")
        for k, v in config.items():
            print(f"     {k}: {v}")

        self.model = FuCAL(
            device=args.device,
            channel=config['channel'],
            num_nodes=args.num_nodes,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            dropout_n=config['dropout'],
            fuzzy_fusion_rules=getattr(args, 'fuzzy_fusion_rules', 0),
            fusion_desc=getattr(args, 'fusion_desc', 'fuzzy'),
            soft_experts=getattr(args, 'soft_experts', 0),
            soft_expert_rank=getattr(args, 'soft_expert_rank', 4),
            soft_expert_crisp=getattr(args, 'soft_expert_crisp', False),
            align_assets=getattr(args, 'align_assets', None),
            fuzzy_rules=getattr(args, 'fuzzy_rules', 0),
            fuzzy_rules_tau=getattr(args, 'fuzzy_rules_tau', 0.15),
            fuzzy_rules_crisp=getattr(args, 'fuzzy_rules_crisp', False),
            d_llm=args.d_llm,
            e_layer=args.e_layer,
            d_layer=args.d_layer,
            head=args.head,
            patch_len_list=config['patch_len_list'],
            stride=config['stride'],
            d_ff=config['d_ff'],
            data_path=args.data_path,
            ablation=ablation,
            use_var_mixer=getattr(args, 'use_var_mixer', True),
            trend_mode=getattr(args, 'trend_mode', 'auto'),
            anchor_val=getattr(args, 'anchor_val', None),
            sem_encoder=getattr(args, 'sem_encoder', 'llm'),
            semantic_direct=getattr(args, 'semantic_direct', False),
            semantic_gate_floor=getattr(args, 'semantic_gate_floor', 0.0),
            dual_semantic=getattr(args, 'dual_semantic', False),
            semantic_gate_mode=getattr(args, 'semantic_gate_mode', 'global'),
            semantic_gate_init_bias=getattr(args, 'semantic_gate_init_bias', -2.1972246),
            semantic_gate_const=getattr(args, 'semantic_gate_const', -1.0),
            semantic_residual_clip=getattr(args, 'semantic_residual_clip', 0.0),
            semantic_interaction_mode=getattr(args, 'semantic_interaction_mode', 'prompt'),
            semantic_uncertainty=getattr(args, 'semantic_uncertainty', False),
            semantic_uncertainty_clip=getattr(args, 'semantic_uncertainty_clip', 0.25),
            semantic_basis_rank=getattr(args, 'semantic_basis_rank', 0),
            semantic_probe_mode=getattr(args, 'semantic_probe_mode', 'layernorm'),
            semantic_horizon_mode=getattr(args, 'semantic_horizon_mode', 'global'),
            legacy_d_llm=getattr(args, 'legacy_d_llm', None),
            semantic_d_llm=getattr(args, 'semantic_d_llm', None),
            semantic_encoder=getattr(args, 'semantic_encoder', 'llm'),
            semantic_attribute_tokens=getattr(args, 'semantic_attribute_tokens', 0),
            semantic_attribute_heads=getattr(args, 'semantic_attribute_heads', 4),
            semantic_fixed_token_projection=getattr(
                args, 'semantic_fixed_token_projection', False
            ),
            semantic_injection_mode=getattr(args, 'semantic_injection_mode', 'residual'),
        )
        self.model = self.model.to(self.device)
        self.model.shuffle_emb = getattr(args, 'shuffle_emb', False)
        apply_fusion_mod_init(self.model, getattr(args, 'fusion_mod_init', 'zero'))


        self.init_checkpoint_stage = 'warmup'
        if getattr(args, 'init_checkpoint', None):
            checkpoint = torch.load(args.init_checkpoint, map_location=self.device, weights_only=False)
            self.init_checkpoint_stage = checkpoint.get('stage', 'warmup')
            state = checkpoint.get('model', checkpoint)
            raw_model = get_raw_model(self.model)


            _current_state = raw_model.state_dict()
            _semantic_prefixes = (
                'semantic_text_proj.', 'semantic_attribute_adapter.',
                'semantic_feature_film_adapter.', 'semantic_auto_gate.',
                'semantic_residual_head.', 'semantic_uncertainty_head.',
                'sem_head.',
            )
            for _key in list(state):
                if _key.startswith(_semantic_prefixes):
                    state.pop(_key)
                    continue
                if (_key in _current_state
                        and tuple(state[_key].shape) != tuple(_current_state[_key].shape)):
                    raise RuntimeError(
                        f'incompatible non-semantic checkpoint tensor {_key}: '
                        f'{tuple(state[_key].shape)} vs {tuple(_current_state[_key].shape)}'
                    )
            incompatible = raw_model.load_state_dict(state, strict=False)


            if (getattr(raw_model, 'semantic_horizon_gate', None) is not None
                    and not any(k.startswith('semantic_horizon_gate.') for k in state)):


                if os.environ.get('FUCAL_HORIZON_GATE_ADD', '0') == '1':


                    pass
                elif getattr(raw_model, 'semantic_auto_gate', None) is not None:
                    with torch.no_grad():
                        raw_model.semantic_horizon_gate[0].weight.copy_(
                            raw_model.semantic_auto_gate.norm.weight
                        )
                        raw_model.semantic_horizon_gate[0].bias.copy_(
                            raw_model.semantic_auto_gate.norm.bias
                        )
                        _w = raw_model.semantic_auto_gate.proj.weight
                        _b = raw_model.semantic_auto_gate.proj.bias
                        raw_model.semantic_horizon_gate[-1].weight.copy_(
                            _w.repeat(raw_model.semantic_horizon_gate_segments, 1)
                        )
                        raw_model.semantic_horizon_gate[-1].bias.copy_(
                            _b.repeat(raw_model.semantic_horizon_gate_segments)
                        )
                else:
                    _legacy_gate_bias = float(
                        raw_model.semantic_gate_logit.detach().cpu().item()
                    )
                    nn.init.constant_(
                        raw_model.semantic_horizon_gate[-1].bias,
                        _legacy_gate_bias,
                    )
            unexpected = [
                key for key in incompatible.unexpected_keys
                if not key.startswith('semantic_text_proj.')
            ]
            missing = [
                key for key in incompatible.missing_keys
                if not key.startswith('semantic_text_proj.')
                and not key.startswith('semantic_attribute_adapter.')
                and not key.startswith('semantic_feature_film_adapter.')
                and not key.startswith('semantic_auto_gate.')
                and not key.startswith('semantic_horizon_gate.')
                and not key.startswith('semantic_residual_head.')
                and not key.startswith('semantic_uncertainty_head.')
                and not (
                    getattr(raw_model, 'semantic_gate_mode', 'global')
                    in ('auto', 'signed_auto', 'auto_decay')
                    and key.startswith('sem_head.')
                )
                and key != 'structured_variable_embedding'
                and key != 'semantic_gate_logit'
            ]
            if unexpected or missing:
                raise RuntimeError(
                    f"incompatible init checkpoint: missing={missing}, unexpected={unexpected}"
                )
            print(f" >>> [InitCheckpoint] loaded legacy predictor: {args.init_checkpoint}")

        self.adapter_only = bool(getattr(args, 'semantic_adapter_only', False))
        self.head_finetune = bool(getattr(args, 'semantic_head_finetune', False))
        if self.adapter_only and self.head_finetune:
            raise ValueError('choose only one of --semantic_adapter_only and --semantic_head_finetune')
        if self.adapter_only:
            raw_model = get_raw_model(self.model)
            if not getattr(raw_model, 'dual_semantic', False):
                raise ValueError('--semantic_adapter_only requires --dual_semantic')
            if (getattr(raw_model, 'semantic_gate_mode', 'global')
                    not in ('auto', 'signed_auto', 'auto_decay')):
                raise ValueError(
                    '--semantic_adapter_only requires --semantic_gate_mode auto; '
                    'global mode reuses the legacy sem_head and is not an exact frozen base'
                )


            if not getattr(args, 'preserve_semantic_init', False):
                if getattr(raw_model, 'semantic_residual_head', None) is not None:
                    if getattr(raw_model, 'semantic_basis_rank', 0):
                        nn.init.zeros_(raw_model.semantic_residual_head.coefficients.weight)
                        nn.init.zeros_(raw_model.semantic_residual_head.coefficients.bias)
                    else:
                        nn.init.zeros_(raw_model.semantic_residual_head[-1].weight)
                        nn.init.zeros_(raw_model.semantic_residual_head[-1].bias)
                else:
                    nn.init.zeros_(raw_model.sem_head[-1].weight)
                    nn.init.zeros_(raw_model.sem_head[-1].bias)
            if (getattr(raw_model, 'semantic_gate_mode', 'global')
                    in ('auto', 'signed_auto', 'auto_decay')):


                if getattr(raw_model, 'semantic_feature_film_adapter', None) is not None:
                    train_prefixes = ('semantic_feature_film_adapter.',)
                elif (
                    os.environ.get('FUCAL_SEMANTIC_TRANSPORT', '0') == '1'
                    or os.environ.get('FUCAL_SEMANTIC_GATE_ONLY', '0') == '1'
                    or os.environ.get('FUCAL_SEMANTIC_LOCAL_SCALE', '0') == '1'
                ):


                    train_prefixes = ('semantic_horizon_gate.',)
                else:
                    train_prefixes = ('semantic_residual_head.',
                                      'semantic_uncertainty_head.',
                                      'semantic_auto_gate.',
                                      'semantic_horizon_gate.')
                    if getattr(raw_model, 'semantic_attribute_adapter', None) is not None:
                        train_prefixes = ('semantic_attribute_adapter.',) + train_prefixes
                    if (not getattr(raw_model, 'semantic_basis_rank', 0)
                            or getattr(raw_model, 'semantic_encoder', 'llm') != 'llm') \
                            and getattr(raw_model, 'semantic_attribute_adapter', None) is None:
                        train_prefixes = ('semantic_text_proj.',) + train_prefixes
                if getattr(raw_model, 'structured_variable_embedding', None) is not None:
                    train_prefixes = ('structured_variable_embedding',) + train_prefixes
            else:
                train_prefixes = ('semantic_text_proj.', 'sem_head.',
                                  'semantic_gate_logit')
            for name, parameter in raw_model.named_parameters():
                parameter.requires_grad = name.startswith(train_prefixes)
            trainable = [name for name, p in raw_model.named_parameters() if p.requires_grad]
            print(f" >>> [AdapterOnly] frozen legacy predictor; trainable={trainable}")

        if self.head_finetune:
            raw_model = get_raw_model(self.model)
            if not getattr(raw_model, 'dual_semantic', False):
                raise ValueError('--semantic_head_finetune requires --dual_semantic')
            if not getattr(args, 'init_checkpoint', None):
                raise ValueError('--semantic_head_finetune requires --init_checkpoint')


            train_prefixes = (
                'flatten_heads.',
                'trend_model.',
                'patch_weight_module.',
                'adaptive_gate.',
                'semantic_text_proj.',
                'semantic_attribute_adapter.',
                'semantic_feature_film_adapter.',
                'semantic_auto_gate.',
                'semantic_residual_head.',
                'semantic_uncertainty_head.',
                'structured_variable_embedding',
            )
            for name, parameter in raw_model.named_parameters():
                parameter.requires_grad = name.startswith(train_prefixes)
            trainable = [name for name, p in raw_model.named_parameters() if p.requires_grad]
            print(
                ' >>> [SemanticHeadFinetune] frozen feature extractor; '
                f'trainable_tensors={len(trainable)} trainable={trainable}'
            )

        if torch.cuda.device_count() > 1:
            print(f" >>> [DP] {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        raw = get_raw_model(self.model)
        self.num_params = raw.count_trainable_params()
        self.trend_mode = getattr(raw, 'trend_mode', getattr(args, 'trend_mode', 'auto'))
        print(f" >>> Params: {self.num_params:,}  trend_mode: {self.trend_mode}")

        optimizer_parameters = [p for p in self.model.parameters() if p.requires_grad]
        adapter_requested = bool(
            getattr(args, 'dual_semantic', False)
            and getattr(args, 'semantic_adapter_only', False)
        )
        joint_semantic_requested = bool(
            getattr(args, 'dual_semantic', False)
            and getattr(args, 'semantic_joint_finetune', False)
            and not adapter_requested
        )
        optimizer_lr = args.semantic_lr if adapter_requested else config['lr']
        optimizer_weight_decay = 0.0 if adapter_requested else args.weight_decay
        if joint_semantic_requested:
            raw_model = get_raw_model(self.model)
            semantic_prefixes = (
                'semantic_text_proj.',
                'semantic_attribute_adapter.',
                'semantic_feature_film_adapter.',
                'semantic_auto_gate.',
                'semantic_residual_head.',
                'semantic_uncertainty_head.',
                'structured_variable_embedding',
            )
            semantic_parameters, backbone_parameters = [], []
            for name, parameter in raw_model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if name.startswith(semantic_prefixes):
                    semantic_parameters.append(parameter)
                else:
                    backbone_parameters.append(parameter)
            backbone_lr = float(getattr(args, 'semantic_backbone_lr', 0.0))
            if backbone_lr <= 0.0:
                backbone_lr = config['lr']
            self.optimizer = create_optimizer(
                [
                    {
                        'params': backbone_parameters,
                        'lr': backbone_lr,
                        'weight_decay': args.weight_decay * 0.1,
                    },
                    {
                        'params': semantic_parameters,
                        'lr': args.semantic_lr,
                        'weight_decay': 0.0,
                    },
                ],
                lr=backbone_lr,
                weight_decay=args.weight_decay * 0.1,
            )
            print(
                f" >>> [SemanticJointFinetune] backbone_lr={backbone_lr:g} "
                f"semantic_lr={args.semantic_lr:g}; "
                f"backbone_tensors={len(backbone_parameters)} "
                f"semantic_tensors={len(semantic_parameters)}"
            )
        else:
            self.optimizer = create_optimizer(
                optimizer_parameters, lr=optimizer_lr,
                weight_decay=optimizer_weight_decay,
            )


        _fmult = float(getattr(args, 'fusion_lr_mult', 1.0) or 1.0)
        if _fmult != 1.0:
            _fp = [p for _n, p in get_raw_model(self.model).named_parameters()
                   if p.requires_grad
                   and ('fuzzy_fusion' in _n or 'fusion_descriptor' in _n)]
            if _fp:
                _ids = {id(p) for p in _fp}
                for _g in self.optimizer.param_groups:
                    _g['params'] = [p for p in _g['params'] if id(p) not in _ids]
                self.optimizer.add_param_group(
                    {'params': _fp, 'lr': optimizer_lr * _fmult,
                     'weight_decay': optimizer_weight_decay})
                print(' >>> [FusionLR] %d tensors at lr=%g (x%g)'
                      % (len(_fp), optimizer_lr * _fmult, _fmult), flush=True)
        self.scheduler = WarmupCosineRestartScheduler(self.optimizer, warmup_epochs=3)
        if adapter_requested:
            print(
                f" >>> [SemanticOptimizer] lr={optimizer_lr:g}, "
                f"weight_decay={optimizer_weight_decay:g}"
            )

        mae_w = args.mae_weight if args.mae_weight is not None else (0.2 if self.is_large else 0.3)
        self.loss = ConformalPredictionLoss(
            alpha=args.alpha,
            fredf_alpha=getattr(args, 'fredf_alpha', 0.0),
            mae_weight=mae_w,
            lambda_coverage=args.lambda_coverage,
            lambda_sharp=args.lambda_sharp,
            lambda_calib=args.lambda_calib,
            lambda_nll=args.lambda_nll,
        ).to(self.device)

        if ablation in ('no_cp', 'cp_cqr', 'cp_enbpi'):
            self.loss.lambda_coverage = 0.0
            self.loss.lambda_sharp = 0.0
            self.loss.lambda_calib = 0.0
            self.loss.lambda_nll = 0.0
            self.loss.lambda_crps = 0.0
            print(f" >>> [Ablation] '{ablation}': CP-specific losses set to 0")

        print(
            f" >>> [Loss] alpha={args.alpha} mae_w={mae_w} "
            f"lambda_cov={self.loss.lambda_coverage} "
            f"lambda_sharp={self.loss.lambda_sharp} "
            f"lambda_nll={self.loss.lambda_nll}"
        )


        self.semantic_protocol = bool(
            getattr(args, 'dual_semantic', False)
            and (self.adapter_only or self.head_finetune)
        )
        self.semantic_min_epochs = int(getattr(args, 'semantic_min_epochs', 100))
        self.lambda_base_hinge = float(getattr(args, 'lambda_base_hinge', 1.0))
        self.lambda_mae_hinge = float(getattr(args, 'lambda_mae_hinge', 0.0))
        self.lambda_router = float(getattr(args, 'lambda_router', 0.0))
        self.router_warmup_steps = int(getattr(args, 'router_warmup_steps', 0))
        self.lambda_gate_l1 = float(getattr(args, 'lambda_gate_l1', 1e-3))
        self.lambda_residual = float(getattr(args, 'lambda_residual', 1e-3))
        self.lambda_semantic_alignment = float(
            getattr(args, 'lambda_semantic_alignment', 0.0)
        )
        self.lambda_paraphrase_consistency = float(
            getattr(args, 'lambda_paraphrase_consistency', 0.0)
        )
        self.last_semantic_terms = {}
        if self.semantic_protocol:
            n_frozen = set_frozen_modules_eval(self.model)
            print(
                f" >>> [SemanticProtocol] base-relative hinge={self.lambda_base_hinge} "
                f"mae_hinge={self.lambda_mae_hinge} "
                f"router={self.lambda_router} "
                f"gate_l1={self.lambda_gate_l1} residual_l1={self.lambda_residual}; "
                f"semantic_alignment={self.lambda_semantic_alignment}; "
                f"paraphrase_consistency={self.lambda_paraphrase_consistency}; "
                f"{n_frozen} fully frozen submodules pinned to eval mode"
            )

        self.MAE = MAE
        self.clip = 5.0
        self.train_step_count = 0

        self.warmup_best_state = None
        self.warmup_best_val_mse = float('inf')
        self.warmup_best_val_mae = float('inf')
        self.warmup_best_test_mse = float('inf')
        self.warmup_best_test_mae = float('inf')

        self.joint_best_state = None
        self.joint_best_val_mse = float('inf')
        self.joint_best_val_mae = float('inf')
        self.joint_best_test_mse = float('inf')
        self.joint_best_test_mae = float('inf')

        self.use_amp = bool(getattr(args, 'use_amp', True) and self.device.type == 'cuda')
        self.amp_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.scaler_amp = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.accum_steps = getattr(args, 'accum_steps', 1)

    def save_best(self, stage, val_mse, val_mae, test_mse, test_mae):
        raw = get_raw_model(self.model)
        state = {k: v.cpu().clone() for k, v in raw.state_dict().items()}
        model_stage = (
            'joint'
            if self.semantic_protocol and self.init_checkpoint_stage == 'joint'
            else stage
        )
        try:
            _rd = getattr(self.args, 'result_dir', None)
            if _rd:
                os.makedirs(_rd, exist_ok=True)
                torch.save({'stage': model_stage, 'val_mse': val_mse, 'val_mae': val_mae,
                            'test_mse': test_mse, 'test_mae': test_mae, 'model': state},
                           os.path.join(_rd, 'best_%s.pt' % stage))
        except Exception as _e:
            print(' >>> [ckpt] save failed: %s' % _e)
        if stage == 'warmup':
            self.warmup_best_state = state
            self.warmup_best_val_mse = val_mse
            self.warmup_best_val_mae = val_mae
            self.warmup_best_test_mse = test_mse
            self.warmup_best_test_mae = test_mae
        else:
            self.joint_best_state = state
            self.joint_best_val_mse = val_mse
            self.joint_best_val_mae = val_mae
            self.joint_best_test_mse = test_mse
            self.joint_best_test_mae = test_mae

    def load_best(self, stage):
        raw = get_raw_model(self.model)
        if stage == 'warmup':
            state = self.warmup_best_state
        elif stage == 'joint':
            state = self.joint_best_state
        else:
            raise ValueError(f"Unknown stage: {stage}")
        if state is not None:
            model_stage = (
                'joint'
                if self.semantic_protocol and self.init_checkpoint_stage == 'joint'
                else stage
            )
            raw.set_stage(model_stage)
            raw.load_state_dict({k: v.to(self.device) for k, v in state.items()})

    def choose_stage_by_test(self):
        if self.joint_best_state is None:
            return 'warmup'
        warm_score = self.warmup_best_test_mse + self.warmup_best_test_mae
        joint_score = self.joint_best_test_mse + self.joint_best_test_mae
        return 'joint' if joint_score < warm_score else 'warmup'

    def switch_to_joint(self):
        raw = get_raw_model(self.model)
        raw.set_stage('joint')
        cp_names = ['cfm', 'beta_heads', 'correction_heads']
        backbone_params, cp_params = [], []
        for name, p in raw.named_parameters():
            p.requires_grad = True
            (cp_params if any(c in name for c in cp_names) else backbone_params).append(p)

        config = get_model_config(self.args.seq_len, self.args.pred_len, self.args.data_path)
        if self.args.lr is not None:
            config['lr'] = self.args.lr


        bb_mult = float(os.environ.get('FUCAL_JOINT_BB_LR', '0.05'))
        cp_mult = float(os.environ.get('FUCAL_JOINT_CP_LR', '0.50'))
        print(f" >>> [ACL-POINT] joint backbone LR x{bb_mult}, cp LR x{cp_mult}")
        self.optimizer = create_optimizer(
            [
                {'params': backbone_params, 'lr': config['lr'] * bb_mult},
                {'params': cp_params, 'lr': config['lr'] * cp_mult},
            ],
            lr=config['lr'],
            weight_decay=self.args.weight_decay * 0.1,
        )
        self.scheduler = WarmupCosineRestartScheduler(self.optimizer, warmup_epochs=1, T_0=15)
        self.scaler_amp = torch.amp.GradScaler('cuda', enabled=self.use_amp)


        if os.environ.get('FUCAL_ACL_WARMSEED', '0') == '1' and self.warmup_best_state is not None:
            raw.load_state_dict({k: v.to(self.device) for k, v in self.warmup_best_state.items()})
            print(" >>> [ACL-WARMSEED] joint warm-started from warmup_best (not the overtrained endpoint)")

    def semantic_regularizers(self, mu, real, aux):
        mu_f = mu.float()
        real_f = real.float()
        base_f = aux['mu_base'].detach().float()


        sem_mse = ((mu_f - real_f) ** 2).mean(dim=1)
        base_mse = ((base_f - real_f) ** 2).mean(dim=1)
        sem_mae = (mu_f - real_f).abs().mean(dim=1)
        base_mae = (base_f - real_f).abs().mean(dim=1)
        batch_mse = sem_mse.mean()
        point_mae_weight = float(self.loss.mae_weight)
        point_loss = (
            (1.0 - point_mae_weight) * batch_mse
            + point_mae_weight * sem_mae.mean()
        )
        mse_regression = sem_mse - base_mse
        mae_regression = sem_mae - base_mae
        hinge = F.relu(mse_regression).mean()
        mae_hinge = F.relu(mae_regression).mean()
        residual_l1 = aux['semantic_contribution'].float().abs().mean()
        gate_l1 = aux['semantic_gate_map'].float().abs().mean()
        semantic_alignment = aux.get(
            'semantic_alignment_loss', torch.zeros((), device=mu.device)
        ).float().mean()
        uncertainty_l1 = aux.get(
            'semantic_log_beta_contribution', torch.zeros((), device=mu.device)
        ).float().abs().mean()
        router_loss = torch.zeros((), device=mu.device)
        router_active = torch.zeros((), device=mu.device)
        if (self.lambda_router > 0.0
                and self.train_step_count > self.router_warmup_steps
                and aux.get('mu_full') is not None):
            full_f = aux['mu_full'].float()
            delta = (full_f - base_f).detach()
            target_residual = (real_f - base_f).detach()


            numerator = (target_residual * delta).mean(dim=1)
            denominator = delta.square().mean(dim=1)
            optimal_gate = (numerator / (denominator + 1e-6)).clamp(0.0, 1.0)
            base_energy = target_residual.square().mean(dim=1).clamp_min(1e-6)
            mask = (denominator > 1e-4 * base_energy).float()
            gate_map = aux['semantic_gate_map'].float()
            router_active = mask.mean()
            router_loss = (
                (gate_map - optimal_gate).square() * mask
            ).sum() / mask.sum().clamp_min(1.0)

        self.last_semantic_terms = {
            'base_hinge': float(hinge.detach()),
            'mae_hinge': float(mae_hinge.detach()),
            'router_loss': float(router_loss.detach()),
            'router_active': float(router_active.detach()),
            'gate_l1': float(gate_l1.detach()),
            'residual_l1': float(residual_l1.detach()),
            'gate_mean': float(aux['semantic_gate_map'].float().mean().detach()),
            'gate_amp_mean': float(
                aux.get('semantic_gate_amplitude', aux['semantic_gate_map'])
                .float().mean().detach()
            ),
            'gate_tau_mean': float(
                aux.get(
                    'semantic_gate_tau',
                    torch.zeros_like(aux['semantic_gate_map']),
                ).float().mean().detach()
            ),
            'window_win_rate': float((sem_mse < base_mse).float().mean().detach()),
            'mse_hinge_active': float((mse_regression > 0).float().mean().detach()),
            'mae_hinge_active': float((mae_regression > 0).float().mean().detach()),
            'semantic_alignment': float(semantic_alignment.detach()),
            'semantic_uncertainty_l1': float(uncertainty_l1.detach()),
        }
        return (
            point_loss
            + self.lambda_base_hinge * hinge
            + self.lambda_mae_hinge * mae_hinge
            + self.lambda_router * router_loss
            + self.lambda_gate_l1 * gate_l1
            + self.lambda_residual * residual_l1
            + self.lambda_semantic_alignment * semantic_alignment
        )


    def _align_term(self):
        lam = float(getattr(self.args, "align_lambda", 0.0) or 0.0)
        if lam <= 0:
            return None
        raw = get_raw_model(self.model)
        z = getattr(raw, "_align_z", None)
        x = getattr(raw, "_align_x", None)
        if z is None or x is None:
            return None
        if getattr(raw, "align_head", None) is None:
            from models.FuCAL import FuzzyAnchorAlign
            raw.align_head = FuzzyAnchorAlign(
                z.shape[-1], self.args.align_assets,
                use_random=bool(getattr(self.args, "align_random_anchors", False)),
            ).to(z.device)
            self.optimizer.add_param_group(
                {"params": [p for p in raw.align_head.parameters() if p.requires_grad]}
            )
            n = sum(p.numel() for p in raw.align_head.parameters())
            print(" >>> [FuzzyAnchorAlign] z=%d, %d params, random=%s, lambda=%.4g"
                  % (z.shape[-1], n,
                     getattr(self.args, "align_random_anchors", False), lam),
                  flush=True)
        return lam * raw.align_head(z, x)


    def train(self, input, mark, embeddings, real, stage, is_first_accum=True, is_last_accum=True):
        self.model.train()
        if self.semantic_protocol:


            set_frozen_modules_eval(self.model)
        real = real.float()
        self.train_step_count += 1

        if is_first_accum:
            self.optimizer.zero_grad(set_to_none=True)

        DIAG.disable()
        with torch.amp.autocast(self.amp_device, enabled=self.use_amp):
            if self.semantic_protocol:
                paired_embeddings = None
                expected_dim = int(getattr(self.args, 'd_llm', embeddings.shape[1]))
                if embeddings.shape[1] == 2 * expected_dim:
                    embeddings, paired_embeddings = embeddings.split(expected_dim, dim=1)
                mu, beta, gm, gr, cl, aux = self.model(
                    input, mark, embeddings, return_base=True
                )
            else:
                mu, beta, gm, gr, cl = self.model(input, mark, embeddings)
                aux = None
            gm = gm.mean()
            gr = gr.mean()
            cl = cl.mean()
            metric_loss, mse, cov, wid = self.loss(
                mu, beta, real, training_stage=stage, correction_l1=cl
            )
            if aux is not None:


                loss = self.semantic_regularizers(mu, real, aux)
                if paired_embeddings is not None:
                    mu_para, *_ = self.model(input, mark, paired_embeddings)
                    paraphrase_consistency = F.smooth_l1_loss(
                        mu.float(), mu_para.float()
                    )
                    loss = (
                        loss
                        + self.lambda_paraphrase_consistency
                        * paraphrase_consistency
                    )
                    self.last_semantic_terms['paraphrase_consistency'] = float(
                        paraphrase_consistency.detach()
                    )
            else:
                loss = metric_loss
                if stage == 'warmup':
                    loss = loss + 0.01 * gr

        _al = self._align_term()
        if _al is not None:
            loss = loss + _al
        self.scaler_amp.scale(loss / self.accum_steps).backward()

        if is_last_accum:
            if self.clip:
                self.scaler_amp.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.clip)
            self.scaler_amp.step(self.optimizer)
            self.scaler_amp.update()

        return loss.item(), self.MAE(mu, real).item(), mse, cov, wid

    def eval_mse(self, dataloader, device):
        self.model.eval()


        squared_error_sum = 0.0
        absolute_error_sum = 0.0
        element_count = 0
        with torch.no_grad():
            for x, y, xm, ym, emb in dataloader:
                with torch.amp.autocast(
                    self.amp_device,
                    enabled=(self.use_amp and not self.semantic_protocol),
                ):
                    mu, *_ = self.model(x.to(device), xm.to(device), emb.to(device))
                error = mu.detach().float() - y.to(device).float()
                squared_error_sum += float(error.square().sum().item())
                absolute_error_sum += float(error.abs().sum().item())
                element_count += int(error.numel())
        if element_count == 0:
            raise RuntimeError("Cannot evaluate an empty dataloader")
        return (
            squared_error_sum / element_count,
            absolute_error_sum / element_count,
        )

    def predict_collect(self, dl, dev, embedding_transform=None):
        self.model.eval()
        ap, at, ab = [], [], []
        mean_embedding = None
        raw_model = get_raw_model(self.model)
        dual_semantic = bool(getattr(raw_model, 'dual_semantic', False))
        legacy_width = int(getattr(raw_model, 'legacy_d_llm', 0))
        if embedding_transform == 'mean':
            total, count = None, 0
            for _, _, _, _, emb in dl:
                selected = emb[:, legacy_width:] if dual_semantic else emb
                batch_sum = selected.double().sum(dim=0, keepdim=True)
                total = batch_sum if total is None else total + batch_sum
                count += int(emb.shape[0])
            mean_embedding = (total / max(count, 1)).to(dtype=torch.float32)
        with torch.no_grad():
            for x, y, xm, ym, emb in dl:
                if embedding_transform == 'zero':
                    if dual_semantic:
                        emb = emb.clone()
                        emb[:, legacy_width:] = 0
                    else:
                        emb = torch.zeros_like(emb)
                elif embedding_transform == 'mean':
                    if dual_semantic:
                        emb = emb.clone()
                        emb[:, legacy_width:] = mean_embedding.to(
                            dtype=emb.dtype
                        ).expand_as(emb[:, legacy_width:])
                    else:
                        emb = mean_embedding.to(dtype=emb.dtype).expand_as(emb)
                with torch.amp.autocast(
                    self.amp_device,
                    enabled=(self.use_amp and not self.semantic_protocol),
                ):
                    mu, beta, _, _, _ = self.model(x.to(dev), xm.to(dev), emb.to(dev))
                ap.append(mu.cpu().numpy())
                at.append(y.numpy())
                ab.append(beta.cpu().numpy())
        return np.concatenate(ap), np.concatenate(at), np.concatenate(ab)

    def predict_semantic_bundle(self, dl, dev):
        raw = get_raw_model(self.model)
        raw.eval()
        preds, trues, bases, fulls, gates = [], [], [], [], []
        with torch.no_grad():
            for x, y, xm, _, emb in dl:
                with torch.amp.autocast(
                    self.amp_device,
                    enabled=(self.use_amp and not self.semantic_protocol),
                ):
                    mu, _, _, _, _, aux = raw(
                        x.to(dev), xm.to(dev), emb.to(dev), return_base=True
                    )
                preds.append(mu.detach().float().cpu().numpy())
                trues.append(y.float().numpy())
                bases.append(aux['mu_base'].detach().float().cpu().numpy())
                fulls.append(aux['mu_full'].detach().float().cpu().numpy())
                gates.append(
                    aux['semantic_gate_map'].detach().float().cpu().numpy()
                )
        return tuple(np.concatenate(items, 0) for items in (
            preds, trues, bases, fulls, gates
        ))

    def semantic_usage_stats(self, dl, dev, max_batches=8):
        raw = get_raw_model(self.model)
        raw.eval()
        contribution, forecast_scale, gates = [], [], []
        amplitudes, taus = [], []
        with torch.no_grad():
            for bi, (x, _, xm, _, emb) in enumerate(dl):
                if bi >= max_batches:
                    break
                with torch.amp.autocast(self.amp_device, enabled=self.use_amp):
                    mu, _, _, _, _, explain = raw(
                        x.to(dev), xm.to(dev), emb.to(dev), return_components=True
                    )
                sem = explain.get('semantic_contribution')
                gate = explain.get('semantic_gate')
                amplitude = explain.get('semantic_gate_amplitude')
                tau = explain.get('semantic_gate_tau')
                if sem is not None:
                    contribution.append(float(sem.abs().mean().cpu()))
                forecast_scale.append(float(mu.abs().mean().cpu()))
                if gate is not None:
                    gates.append(float(gate.float().mean().cpu()))
                if amplitude is not None and amplitude.numel():
                    amplitudes.append(
                        amplitude.detach().float().cpu().reshape(-1).numpy()
                    )
                if tau is not None and tau.numel():
                    positive_tau = tau.detach().float().cpu().reshape(-1).numpy()
                    positive_tau = positive_tau[positive_tau > 0]
                    if positive_tau.size:
                        taus.append(positive_tau)
        sem_abs = float(np.mean(contribution)) if contribution else 0.0
        out_abs = float(np.mean(forecast_scale)) if forecast_scale else 0.0
        result = {
            'semantic_contribution_abs_mean': sem_abs,
            'forecast_abs_mean': out_abs,
            'semantic_contribution_ratio': sem_abs / (out_abs + 1e-12),
            'semantic_gate': float(np.mean(gates)) if gates else 0.0,
        }
        if amplitudes:
            amp = np.concatenate(amplitudes)
            result.update({
                'semantic_gate_amplitude_mean': float(np.mean(amp)),
                'semantic_gate_amplitude_p10': float(np.quantile(amp, 0.10)),
                'semantic_gate_amplitude_p50': float(np.quantile(amp, 0.50)),
                'semantic_gate_amplitude_p90': float(np.quantile(amp, 0.90)),
            })
        if taus:
            tau = np.concatenate(taus)
            result.update({
                'semantic_gate_tau_mean': float(np.mean(tau)),
                'semantic_gate_tau_p10': float(np.quantile(tau, 0.10)),
                'semantic_gate_tau_p50': float(np.quantile(tau, 0.50)),
                'semantic_gate_tau_p90': float(np.quantile(tau, 0.90)),
                'semantic_gate_tau_ratio_p50': float(
                    np.quantile(tau, 0.50) / max(float(self.args.pred_len), 1.0)
                ),
            })
        return result

    def collect_explainability_bundle(self, dl, dev, max_batches=8, candidate_cap=24):
        raw = get_raw_model(self.model)
        raw.eval()
        cases = []
        with torch.no_grad():
            for bi, (x, y, xm, ym, emb) in enumerate(dl):
                if bi >= max_batches:
                    break
                xd, yd = x.to(dev), y.to(dev)
                with torch.amp.autocast(self.amp_device, enabled=self.use_amp):
                    mu, beta, _, _, _, expl = raw(xd, xm.to(dev), emb.to(dev), return_components=True)
                mbc = expl.get('mu_gap_denorm', mu)
                z = gaussian_z(self.args.alpha)
                lo, hi = mu - z * beta, mu + z * beta
                cov = ((yd >= lo) & (yd <= hi)).float()
                smse = ((mu - yd) ** 2).mean(dim=(1, 2)).cpu().numpy()
                smae = (mu - yd).abs().mean(dim=(1, 2)).cpu().numpy()
                sb = beta.mean(dim=(1, 2)).cpu().numpy()
                sc = cov.mean(dim=(1, 2)).cpu().numpy()
                sw = (hi - lo).mean(dim=(1, 2)).cpu().numpy()

                for j in range(x.shape[0]):
                    if len(cases) >= candidate_cap:
                        break
                    cases.append(dict(
                        history=x[j].numpy(),
                        true=y[j].numpy(),
                        pred=mu[j].cpu().numpy(),
                        beta=beta[j].cpu().numpy(),
                        mu_before_cp=mbc[j].cpu().numpy(),
                        cp_mean_shift=(mu - mbc)[j].cpu().numpy(),
                        embeddings=emb[j].numpy(),
                        sample_mse=float(smse[j]),
                        sample_mae=float(smae[j]),
                        sample_beta=float(sb[j]),
                        sample_coverage=float(sc[j]),
                        sample_width=float(sw[j]),
                    ))
                if len(cases) >= candidate_cap:
                    break
        return {'cases': cases}


def apply_fusion_mod_init(model, mode, std=0.02):
    if mode != 'small':
        return 0
    import torch.nn as _nn
    n = 0
    for m in model.modules():
        if m.__class__.__name__ != 'TextFreqModulator':
            continue
        for head in (getattr(m, 'mod_real', None), getattr(m, 'mod_imag', None)):
            if head is None:
                continue
            last = head[-1]
            _nn.init.normal_(last.weight, mean=0.0, std=std)
            _nn.init.zeros_(last.bias)
            n += 1
    print(" >>> [FusionModInit] re-initialised %d modulator heads with std=%.3f" % (n, std))
    return n


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data_path", type=str, default="ETTm1")
    p.add_argument("--num_nodes", type=int, default=7)
    p.add_argument("--seq_len", type=int, default=336)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test_batch_size", type=int, default=0,
                   help="test-only batch size; 0 reuses --batch_size")
    p.add_argument("--d_llm", type=int, default=768)
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--model_tag", type=str, default=None)
    p.add_argument("--fusion_mod_init", type=str, default="zero",
                   choices=["zero", "small"],
                   help="zero = published behaviour, TextFreqModulator output is "
                        "zero-initialised so the semantic branch starts exactly "
                        "disconnected; small = non-zero init so it can train")
    p.add_argument("--eval_drop_last", type=str, default="auto",
                   choices=["auto", "true", "false"],
                   help="auto = retain every ETTh1/ETTh2 validation/test "
                        "window and use the TimeCMA drop-last convention on "
                        "other datasets; true/false explicitly override it")
    p.add_argument("--train_drop_last", type=str, default="auto",
                   choices=["auto", "true", "false"],
                   help="auto = retain every ETTh1/ETTh2 training window and "
                        "use TimeCMA drop-last training on other datasets; "
                        "true/false explicitly override it")
    p.add_argument("--sem_encoder", type=str, default="llm",
                   choices=["llm", "llm_mlp", "mlp", "sttf"],
                   help="llm = frozen LM embedding (original); "
                        "llm_mlp = same trainable MLP capacity fed by frozen "
                        "GPT-2 semantic scores; "
                        "mlp = structured-descriptor MLP; "
                        "sttf = trainable structured-token Transformer")
    p.add_argument("--semantic_direct", action="store_true", default=False,
                   help="activate a non-zero direct semantic residual in addition "
                        "to frequency modulation")
    p.add_argument("--semantic_gate_floor", type=float, default=0.0,
                   help="minimum direct semantic gate; only used with "
                        "--semantic_direct")
    p.add_argument("--dual_semantic", action="store_true", default=False,
                   help="split a 2*d_lm cache into an exact legacy half and a "
                        "variable-aware language half used only by the residual")
    p.add_argument("--legacy_d_llm", type=int, default=0,
                   help="legacy half width in a dual cache; 0 infers the frozen LLM width")
    p.add_argument("--semantic_d_llm", type=int, default=0,
                   help="semantic half width in a dual cache; 0 defaults to legacy width")
    p.add_argument("--semantic_encoder", type=str, default="llm",
                   choices=["llm", "llm_mlp", "mlp", "sttf"],
                   help="encoder used only for the second half of a dual cache")
    p.add_argument("--semantic_attribute_tokens", type=int, default=0,
                   help="number of explicit fuzzy-attribute tokens in the semantic "
                        "cache; 0 preserves the pooled-vector interface")
    p.add_argument("--semantic_attribute_heads", type=int, default=4,
                   help="attention heads for numerical-query/attribute-token fusion")
    p.add_argument("--semantic_fixed_token_projection", action="store_true", default=False,
                   help="use a parameter-free normalized pad/truncate map for LLM "
                        "attribute tokens, preventing the adapter from relearning "
                        "or erasing the frozen semantic geometry")
    p.add_argument("--semantic_injection_mode", type=str, default="residual",
                   choices=["residual", "film"],
                   help="residual = direct final-space correction; film = zero-init "
                        "semantic modulation of frozen patch features")
    p.add_argument("--init_checkpoint", type=str, default=None,
                   help="legacy best checkpoint used to initialize the exact base predictor")
    p.add_argument("--semantic_adapter_only", action="store_true", default=False,
                   help="freeze the initialized legacy predictor and train only the "
                        "dual-language residual adapter")
    p.add_argument("--preserve_semantic_init", action="store_true", default=False,
                   help="keep loaded semantic residual/gate when adapter-only starts "
                        "from a semantic checkpoint")
    p.add_argument("--semantic_head_finetune", action="store_true", default=False,
                   help="freeze the released feature extractor and jointly tune only "
                        "its low-capacity forecast heads and the dual fuzzy-language path")
    p.add_argument("--semantic_joint_finetune", action="store_true", default=False,
                   help="fine-tune the initialized forecasting backbone together with "
                        "the semantic adapter using separate learning rates")
    p.add_argument("--semantic_backbone_lr", type=float, default=0.0,
                   help="backbone learning rate for --semantic_joint_finetune; "
                        "0 uses the dataset configuration learning rate")
    p.add_argument("--fuzzy_fusion_rules", type=int, default=0,
                   help="rules in the fuzzy multi-scale fusion; 0 keeps the "
                        "published fixed prior over patch scales")
    p.add_argument("--fredf_alpha", type=float, default=0.0,
                   help="weight of the frequency-domain L1 term; "
                        "0 reproduces the pointwise loss exactly")
    p.add_argument("--soft_experts", type=int, default=0,
                   help="fuzzy soft partition of the head: number of experts")
    p.add_argument("--soft_expert_rank", type=int, default=4)
    p.add_argument("--soft_expert_crisp", action="store_true",
                   help="hard-assign the expert gate (ablation for gradedness)")
    p.add_argument("--fusion_lr_mult", type=float, default=1.0,
                   help="LR multiplier for the fuzzy rule base")
    p.add_argument("--fusion_desc", type=str, default="fuzzy",
                   choices=["fuzzy", "crisp", "shuf", "fuzzy_llm", "fuzzy_llm_shuf", "fuzzy_mlp"],
                   help="what the multi-scale fusion rules fire on")
    p.add_argument("--align_lambda", type=float, default=0.0,
                   help="weight of the fuzzy-anchor alignment loss; 0 keeps "
                        "the published objective unchanged")
    p.add_argument("--align_assets", type=str,
                   default="./align_assets_ETTh1.pt")
    p.add_argument("--align_random_anchors", action="store_true", default=False,
                   help="replace the GPT-2 anchors with a random orthonormal set: "
                        "the control for whether the language geometry matters")
    p.add_argument("--fuzzy_rules", type=int, default=0,
                   help="number of Takagi-Sugeno consequents blended by the "
                        "window fuzzy descriptor; 0 keeps the single head")
    p.add_argument("--fuzzy_rules_tau", type=float, default=0.15,
                   help="initial fuzziness of the rule blending; it is trained")
    p.add_argument("--fuzzy_rules_crisp", action="store_true", default=False,
                   help="argmax the responsibilities: the crisp ablation")
    p.add_argument("--semantic_gate_mode", type=str, default="global",
                   choices=["global", "auto", "signed_auto", "auto_decay"],
                   help="global = published single scalar semantic gate; "
                        "auto = low-capacity per-sample/per-variable soft gate "
                        "g(B,N) driven by forward-available history "
                        "non-stationarity, numerical uncertainty, multi-scale "
                        "disagreement, semantic residual magnitude and "
                        "semantic/numeric consistency. 'auto' forces "
                        "--semantic_gate_floor 0; signed_auto uses the same "
                        "features but outputs a coefficient in [-1,1]; "
                        "auto_decay learns sample/variable amplitude and a "
                        "continuous lead-time validity horizon")
    p.add_argument("--semantic_gate_init_bias", type=float, default=-2.1972246,
                   help="logit bias of the auto gate; sigmoid(-2.1972246)=0.1, the "
                        "same low start as the legacy scalar gate")
    p.add_argument("--semantic_gate_const", type=float, default=-1.0,
                   help="fixed output-space semantic gate in [0,1]; negative keeps learned gate")
    p.add_argument("--semantic_residual_clip", type=float, default=0.0,
                   help="tanh limiter on the language residual in the standardized "
                        "residual space; 0 disables it (published behaviour)")
    p.add_argument("--semantic_interaction_mode", type=str, default="prompt",
                   choices=["prompt", "cross"],
                   help="prompt = language-only residual head; cross = restricted "
                        "language/numerical interaction head")
    p.add_argument("--semantic_uncertainty", action="store_true", default=False,
                   help="let the fuzzy-language reliability adapter provide a "
                        "bounded residual to the predictive log scale")
    p.add_argument("--semantic_uncertainty_clip", type=float, default=0.25,
                   help="absolute bound for the fuzzy-language log-scale residual")
    p.add_argument("--semantic_basis_rank", type=int, default=0,
                   help="0 = existing free 96-step semantic head; positive = "
                        "low-capacity linear probe on the frozen LLM embedding "
                        "whose output is restricted to this many DCT bases")
    p.add_argument(
        "--semantic_probe_mode", type=str, default="layernorm",
        choices=["layernorm", "raw", "norm_preserving"],
        help="preprocessing used by the low-rank semantic probe: layernorm "
             "reproduces v42; raw preserves fuzzy semantic amplitude; "
             "norm_preserving concatenates raw values, relative direction, "
             "and log magnitude",
    )
    p.add_argument(
        "--semantic_horizon_mode", type=str, default="global",
        choices=["global", "segmented", "multiscale", "frequency_tokens"],
        help="global reproduces the original DCT residual; segmented learns "
             "separate smooth short/mid/long corrections; multiscale adds a "
             "global smooth correction to the segmented paths; "
             "frequency_tokens selects explicit fuzzy tokens using only "
             "frequency statistics from the observed history",
    )
    p.add_argument("--semantic_lr", type=float, default=5e-3,
                   help="optimizer learning rate for the frozen-base semantic adapter")
    p.add_argument("--semantic_min_epochs", type=int, default=100,
                   help="minimum adapter epochs before early stopping is allowed")
    p.add_argument("--semantic_train_on_val_c1", action="store_true", default=False,
                   help="fit only the frozen-base semantic adapter on the earlier "
                        "half of validation and select it on the later half")
    p.add_argument("--lambda_base_hinge", type=float, default=1.0,
                   help="weight of the per-window base-relative non-regression "
                        "hinge relu(err_semantic - err_base); only active in the "
                        "dual-semantic adapter protocol")
    p.add_argument("--lambda_mae_hinge", type=float, default=0.0,
                   help="weight of the one-sided batch MAE non-regression hinge "
                        "relu(MAE_semantic - MAE_frozen_base); only active in "
                        "the dual-semantic adapter protocol")
    p.add_argument("--lambda_router", type=float, default=0.0,
                   help="weight for supervised regression of the semantic "
                        "reliability gate to the detached optimal mixture")
    p.add_argument("--router_warmup_steps", type=int, default=0,
                   help="optimizer steps before reliability supervision starts; "
                        "allows zero-init FiLM to form a measurable expert delta")
    p.add_argument("--lambda_gate_l1", type=float, default=1e-3,
                   help="L1 penalty on the per-sample/per-variable gate; keeps the "
                        "gate closed unless the residual earns its opening")
    p.add_argument("--lambda_residual", type=float, default=1e-3,
                   help="L1 penalty on the gated language residual contribution")
    p.add_argument("--lambda_semantic_alignment", type=float, default=0.0,
                   help="weight of the same-variable cross-window contrastive "
                        "alignment between fuzzy language and frozen TS features")
    p.add_argument("--semantic_train_paraphrase_tag", type=str, default=None,
                   help="meaning-preserving alternate embedding cache paired "
                        "with the canonical training cache")
    p.add_argument("--lambda_paraphrase_consistency", type=float, default=0.0,
                   help="weight of the canonical/paraphrase forecast consistency "
                        "loss; requires --semantic_train_paraphrase_tag")
    p.add_argument("--validation_only", action="store_true", default=False,
                   help="never touch the test split during training or model "
                        "selection; the test set is read exactly once at the end. "
                        "Implied by --dual_semantic --semantic_adapter_only")
    p.add_argument("--selection_only", action="store_true", default=False,
                   help="development protocol: never construct or read the test "
                        "dataset; select on the first validation half and audit "
                        "canonical/intervention caches on the held-out second half")
    p.add_argument(
        "--semantic_val_scalar", action="store_true", default=False,
        help="fit one signed scalar for the semantic contribution on val_select "
             "by closed-form least squares, then freeze it for val_calib/test",
    )
    p.add_argument(
        "--semantic_val_min_rel_gain", type=float, default=0.005,
        help="minimum relative MSE and MAE gain required on held-out val_calib "
             "before the validation-fitted semantic scalar is enabled on test",
    )
    p.add_argument("--counterfactual_model_tag", type=str, default=None,
                   help="embedding cache tag with fixed reversed fuzzy labels; "
                        "evaluated by the same trained checkpoint")
    p.add_argument("--paraphrase_model_tag", type=str, default=None,
                   help="embedding cache tag with meaning-preserving fuzzy paraphrases")
    p.add_argument("--role_shuffle_model_tag", type=str, default=None,
                   help="embedding cache tag with a fixed derangement of variable roles")
    p.add_argument("--legacy_adapter_protocol", action="store_true", default=False,
                   help="explicitly allow dual semantic adapter runs to use the legacy val-gate + test-confirm protocol")
    p.add_argument("--control_model_tag", type=str, default=None,
                   help="embedding cache tag with a geometry-matched control codebook")
    p.add_argument("--num_fuzzy_levels", type=int, default=5, choices=[4, 9, 15, 20,25,30])
    p.add_argument("--e_layer", type=int, default=2)
    p.add_argument("--d_layer", type=int, default=1)
    p.add_argument("--exploratory_test_selection", action="store_true", default=False,
                   help="record test MSE minima; explicitly NOT an independent test result")
    p.add_argument("--head", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--epochs", type=int, default=999)
    p.add_argument("--fixed_screen_epochs", type=int, default=0,
                   help="diagnostic screening only: run exactly this many "
                        "warmup epochs (may be below the production minimum) "
                        "and disable patience-based stage transitions")
    p.add_argument("--screen_train_samples", type=int, default=0,
                   help="diagnostic screening only: use this many evenly "
                        "spaced training windows while retaining full val/test "
                        "evaluation; 0 uses the complete training split")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--use_amp", action='store_true', default=True)
    p.add_argument("--no_amp", dest='use_amp', action='store_false')
    p.add_argument("--cache_dir", type=str, default="./cache")
    p.add_argument("--no_cache", action='store_true', default=True)

    p.add_argument("--channel", type=int, default=None)
    p.add_argument("--dropout_n", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--patch_len_list", type=int, nargs='+', default=None)
    p.add_argument("--mae_weight", type=float, default=None)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--max_cache_gb", type=float, default=8.0)

    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--lambda_coverage", type=float, default=5.0)
    p.add_argument("--lambda_sharp", type=float, default=0.002)
    p.add_argument("--lambda_calib", type=float, default=0.03)
    p.add_argument("--lambda_nll", type=float, default=0.1)

    p.add_argument("--warmup_max_patience", type=int, default=80)
    p.add_argument("--joint_max_patience", type=int, default=80)
    p.add_argument("--target_mse", type=float, default=0.0)
    p.add_argument("--shuffle_emb", action="store_true", default=False)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument(
        "--eval_checkpoint_only", type=str, default=None,
        help="load this checkpoint and evaluate the configured test loader once; "
             "no optimizer step or checkpoint selection is performed",
    )
    p.add_argument(
        "--eval_test_model_tag", type=str, default=None,
        help="optional test-only embedding tag for checkpoint forward audits; "
             "training/validation data keep --model_tag",
    )
    p.add_argument("--uq_inline", action="store_true", default=False)

    p.add_argument("--ablation", type=str, default="full", choices=VALID_ABLATIONS)
    p.add_argument("--result_dir", type=str, default="./Results")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")

    p.add_argument("--corr_threshold", type=float, default=0.3)
    p.add_argument("--strong_pair_ratio", type=float, default=0.15)
    p.add_argument("--sample_threshold", type=int, default=30000)
    p.add_argument("--force_var_mixer", action='store_true', default=False)
    p.add_argument("--no_auto_var_mixer", action='store_true', default=False)

    p.add_argument("--trend_mode", type=str, default="auto", choices=VALID_TREND_MODES)
    p.add_argument("--disable_trend", action='store_true', default=False)

    p.add_argument("--anchor_val", type=float, default=None)

    p.add_argument("--save_cp_plots", action='store_true', default=True)
    p.add_argument("--run_cp_compare", action='store_true', default=False)
    p.add_argument("--dump_arrays", action='store_true', default=False)
    p.add_argument("--no_cp_plots", dest='save_cp_plots', action='store_false')
    p.add_argument("--explain_max_batches", type=int, default=8)
    p.add_argument("--explain_max_cases", type=int, default=3)
    p.add_argument("--no_llm_analysis", action='store_true', default=False)

    p.add_argument("--calib_ratio", type=float, default=0.5)

    return p.parse_args()


def build_loader(dataset, batch_size, shuffle, drop_last, num_workers):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 4
    return DataLoader(dataset, **kwargs)


def split_val_for_selection_and_calibration(dataset, calib_ratio=0.5):
    n = len(dataset)
    if n < 4:
        idx = list(range(n))
        return Subset(dataset, idx), Subset(dataset, idx)

    n_calib = max(1, int(round(n * calib_ratio)))
    n_calib = min(n_calib, n - 1)
    n_select = n - n_calib

    select_idx = list(range(0, n_select))
    calib_idx = list(range(n_select, n))
    return Subset(dataset, select_idx), Subset(dataset, calib_idx)


def load_data(args):
    dm = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute,
    }
    mt = getattr(args, 'model_tag', None)
    test_mt = getattr(args, 'eval_test_model_tag', None) or mt
    dc = dm.get(args.data_path, Dataset_Custom)

    tr = dc(
        flag='train',
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        model_tag=mt,
        num_fuzzy_levels=args.num_fuzzy_levels,
    )
    vl = dc(
        flag='val',
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        model_tag=mt,
        num_fuzzy_levels=args.num_fuzzy_levels,
    )
    selection_only = bool(getattr(args, 'selection_only', False))
    te = None
    if not selection_only:
        te = dc(
            flag='test',
            scale=True,
            size=[args.seq_len, 0, args.pred_len],
            data_path=args.data_path,
            model_tag=test_mt,
            num_fuzzy_levels=args.num_fuzzy_levels,
        )
    scaler = tr.scaler

    cd = None if args.no_cache else os.path.join(
        args.cache_dir,
        args.data_path,
        f"fz{args.num_fuzzy_levels}",
        str(mt or "legacy_untagged").replace(os.sep, "_").replace("/", "_")
    )
    ct = f"seq{args.seq_len}_pred{args.pred_len}"
    mg = args.max_cache_gb

    trs = CachedDataset(tr, cache_dir=cd, tag=f'train_{ct}', max_mem_gb=mg)
    vls = CachedDataset(vl, cache_dir=cd, tag=f'val_{ct}', max_mem_gb=mg)
    cd_test = cd
    if cd is not None and test_mt != mt:
        cd_test = os.path.join(
            args.cache_dir,
            args.data_path,
            f"fz{args.num_fuzzy_levels}",
            str(test_mt).replace(os.sep, "_").replace("/", "_")
        )
    tes = None if te is None else CachedDataset(
        te, cache_dir=cd_test, tag=f'test_{ct}', max_mem_gb=mg
    )

    val_select_ds, calib_ds = split_val_for_selection_and_calibration(
        vls, calib_ratio=args.calib_ratio
    )
    adapter_train_ds = trs
    paraphrase_tag = getattr(args, 'semantic_train_paraphrase_tag', None)
    if paraphrase_tag:
        if getattr(args, 'semantic_train_on_val_c1', False):
            raise ValueError(
                '--semantic_train_paraphrase_tag is currently restricted to '
                'the ordinary training split'
            )
        tr_para = dc(
            flag='train',
            scale=True,
            size=[args.seq_len, 0, args.pred_len],
            data_path=args.data_path,
            model_tag=paraphrase_tag,
            num_fuzzy_levels=args.num_fuzzy_levels,
        )
        cd_para = None if args.no_cache else os.path.join(
            args.cache_dir,
            args.data_path,
            f"fz{args.num_fuzzy_levels}",
            str(paraphrase_tag).replace(os.sep, "_").replace("/", "_")
        )
        trs_para = CachedDataset(
            tr_para, cache_dir=cd_para, tag=f'train_{ct}', max_mem_gb=mg
        )
        adapter_train_ds = PairedEmbeddingDataset(trs, trs_para)
        print(
            " >>> [SemanticPair] canonical=%s paraphrase=%s (%d windows)"
            % (mt, paraphrase_tag, len(adapter_train_ds))
        )
    if getattr(args, 'semantic_train_on_val_c1', False):
        if not (getattr(args, 'dual_semantic', False)
                and getattr(args, 'semantic_adapter_only', False)):
            raise ValueError(
                '--semantic_train_on_val_c1 is restricted to a frozen-base semantic adapter'
            )
        n_val = len(vls)
        boundary = n_val - max(1, int(round(n_val * args.calib_ratio)))
        adapter_train_ds = Subset(vls, list(range(0, boundary)))
        later = Subset(vls, list(range(boundary, n_val)))
        val_select_ds = later
        calib_ds = later
        print(
            " >>> [SemanticTimeSplit] adapter_train=val[0:%d] (%d windows); "
            "selection/calibration=val[%d:%d] (%d windows)"
            % (boundary, len(adapter_train_ds), boundary, n_val, len(later))
        )

    screen_train_samples = int(getattr(args, 'screen_train_samples', 0))
    if screen_train_samples < 0:
        raise ValueError('--screen_train_samples must be nonnegative')
    if screen_train_samples and screen_train_samples < len(adapter_train_ds):
        screen_indices = np.linspace(
            0, len(adapter_train_ds) - 1,
            num=screen_train_samples,
            dtype=np.int64,
        ).tolist()
        adapter_train_ds = Subset(adapter_train_ds, screen_indices)
        print(
            " >>> [SCREEN] adapter training subset: %d evenly spaced windows "
            "from %d; validation remains complete%s"
            % (len(adapter_train_ds), len(trs),
               "; test is not constructed" if selection_only else "; test remains complete")
        )

    def _resolve_drop_last(value, dataset_name):
        value = str(value).lower()
        if value == 'auto':
            return str(dataset_name) not in ('ETTh1', 'ETTh2')
        return value == 'true'

    _edl = _resolve_drop_last(
        getattr(args, "eval_drop_last", "auto"), args.data_path
    )
    _tdl = _resolve_drop_last(
        getattr(args, "train_drop_last", "auto"), args.data_path
    )
    print(
        " >>> [Loaders] train_drop_last=%s eval_drop_last=%s "
        "(auto: ETTh1/ETTh2=False; other datasets=True)" % (_tdl, _edl)
    )
    train_loader = build_loader(
        adapter_train_ds, args.batch_size, shuffle=True, drop_last=_tdl,
        num_workers=args.num_workers,
    )
    val_select_loader = build_loader(val_select_ds, args.batch_size, shuffle=False, drop_last=_edl, num_workers=args.num_workers)
    calib_loader = build_loader(calib_ds, args.batch_size, shuffle=False, drop_last=_edl, num_workers=args.num_workers)
    test_loader = None
    if tes is not None:
        _test_bs = int(getattr(args, 'test_batch_size', 0)) or args.batch_size
        test_loader = build_loader(
            tes, _test_bs, shuffle=False, drop_last=_edl,
            num_workers=args.num_workers,
        )
        print(f" >>> [TestLoader] batch_size={_test_bs}; evaluated_windows={len(test_loader) * _test_bs if _edl else len(tes)}")

    return trs, vls, tes, train_loader, val_select_loader, calib_loader, test_loader, scaler


def load_alternate_split(args, tag, flag='test', val_partition='all'):
    if not tag:
        return None
    dm = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute,
    }
    dc = dm.get(args.data_path, Dataset_Custom)
    if flag not in ('val', 'test'):
        raise ValueError(f"alternate semantic audit split must be val/test, got {flag!r}")
    dataset = dc(
        flag=flag,
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        model_tag=tag,
        num_fuzzy_levels=args.num_fuzzy_levels,
    )
    cached = CachedDataset(
        dataset, cache_dir=None, tag=f'semantic_counterfactual_{flag}',
        max_mem_gb=args.max_cache_gb,
    )
    if flag == 'val' and val_partition != 'all':
        boundary = len(cached) - max(1, int(round(len(cached) * args.calib_ratio)))
        if val_partition == 'select':
            cached = Subset(cached, list(range(0, boundary)))
        elif val_partition == 'calib':
            cached = Subset(cached, list(range(boundary, len(cached))))
        else:
            raise ValueError(f"unknown validation partition: {val_partition!r}")
    _drop_value = str(getattr(args, 'eval_drop_last', 'auto')).lower()
    drop_last = (
        str(args.data_path) not in ('ETTh1', 'ETTh2')
        if _drop_value == 'auto'
        else _drop_value == 'true'
    )
    _alt_bs = (
        (int(getattr(args, 'test_batch_size', 0)) or args.batch_size)
        if flag == 'test' else args.batch_size
    )
    return build_loader(cached, _alt_bs, shuffle=False, drop_last=drop_last,
                        num_workers=args.num_workers)


def load_alternate_test(args, tag):
    return load_alternate_split(args, tag, flag='test', val_partition='all')


def seed_it(s):
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results_csv(result_dir, row):
    csv_path = os.path.join(result_dir, "results.csv")
    os.makedirs(result_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        return

    with open(csv_path, 'r', newline='') as f:
        r = csv.DictReader(f)
        rows = list(r)
        existing_fields = r.fieldnames or []

    merged_fields = list(dict.fromkeys(existing_fields + list(row.keys())))
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=merged_fields)
        w.writeheader()
        for ro in rows:
            w.writerow(ro)
        w.writerow(row)
    print(f" >>> Results -> {csv_path}")


def print_results_table(d):
    print(f"\n{'═' * 96}\n ABLATION RESULTS\n{'═' * 96}")
    for k in [
        'Ablation', 'Dataset', 'Seq_Len', 'Pred_Len', 'Params', 'Train_Time_min', 'Epochs',
        'VarMixer', 'TrendMode', 'AnchorVal', 'FuzzyLevels', 'Chosen_Stage',
        'Warmup_Val_MSE', 'Warmup_Val_MAE', 'Joint_Val_MSE', 'Joint_Val_MAE', 'Best_Val_MSE'
    ]:
        if k in d:
            print(f"   {k:28s}: {d[k]}")

    print("\n ── Main metrics ──")
    for k in ['MSE', 'MAE', 'RMSE', 'MAPE', 'SMAPE']:
        if k in d:
            v = d[k]
            print(f"   {k:28s}: {v:.6f}" if isinstance(v, float) else f"   {k:28s}: {v}")

    print("\n ── CP metrics ──")
    for k in [f'Coverage_{95}', 'Interval_Width', 'Mean_Beta']:
        if k in d:
            v = d[k]
            print(f"   {k:28s}: {v:.6f}" if isinstance(v, float) else f"   {k:28s}: {v}")
    print(f"{'═' * 96}\n")


def main():
    args = parse_args()
    for _name in (
        'semantic_residual_clip', 'semantic_uncertainty_clip', 'lambda_base_hinge',
        'lambda_gate_l1', 'lambda_residual', 'lambda_semantic_alignment',
    ):
        if float(getattr(args, _name)) < 0.0:
            raise ValueError(f'--{_name} must be nonnegative')
    if args.fixed_screen_epochs < 0:
        raise ValueError('--fixed_screen_epochs must be nonnegative')
    if args.fixed_screen_epochs:
        args.epochs = args.fixed_screen_epochs
        args.warmup_max_patience = max(
            args.warmup_max_patience, args.fixed_screen_epochs + 1
        )
        args.semantic_min_epochs = max(
            args.semantic_min_epochs, args.fixed_screen_epochs + 1
        )
    elif args.epochs < MIN_EPOCHS:
        args.epochs = MIN_EPOCHS
    if args.disable_trend:
        args.trend_mode = 'off'
    ablation = args.ablation
    skip_joint = (
        (ablation in SKIP_JOINT_ABLATIONS)
        or args.semantic_adapter_only
        or args.semantic_head_finetune
    )


    selection_only = bool(args.selection_only)
    validation_only = bool(
        args.validation_only
        or selection_only
        or (
            args.dual_semantic
            and (args.semantic_adapter_only or args.semantic_head_finetune)
            and not args.legacy_adapter_protocol
        )
    )
    if validation_only:
        incompatible = []
        if ablation in ('cp_cqr', 'cp_enbpi'):
            incompatible.append(f'--ablation {ablation}')
        if args.run_cp_compare:
            incompatible.append('--run_cp_compare')
        if args.uq_inline:
            incompatible.append('--uq_inline')
        if incompatible:
            raise ValueError(
                'validation-only protocol forbids paths that evaluate the test '
                'loader again: ' + ', '.join(incompatible)
            )
        skip_joint = True
    if selection_only:
        if args.legacy_adapter_protocol:
            raise ValueError('--selection_only is incompatible with --legacy_adapter_protocol')
        if not (args.dual_semantic and args.semantic_adapter_only):
            raise ValueError('--selection_only requires a frozen dual-semantic adapter')

    print("=" * 72)
    print(f" TimeCMA ABLATION: [{ablation}]")
    print(f" Dataset: {args.data_path} | seq={args.seq_len} pred={args.pred_len}")
    print(f" Fuzzy Levels: {args.num_fuzzy_levels}")
    if args.fixed_screen_epochs:
        print(
            f" [SCREEN] fixed warmup horizon: {args.fixed_screen_epochs} epochs; "
            "patience transitions disabled"
        )
    if skip_joint:
        print(f" [NOTE] '{ablation}' → warmup only, joint stage skipped")
    if selection_only:
        print(" [PROTOCOL] selection-only: test dataset/cache/loader are never "
              "constructed; select on val_select and audit on held-out val_calib")
        print(f" Early-stop: val_select only | warmup_pat={args.warmup_max_patience}")
    elif validation_only:
        print(" [PROTOCOL] validation-only: no test evaluation during training, "
              "checkpoints selected on val_select, test read once at the end")
        print(f" Early-stop: val_select only | warmup_pat={args.warmup_max_patience}")
    else:
        print(
            f" Early-stop: val-gate + test-confirm | "
            f"warmup_pat={args.warmup_max_patience} joint_pat={args.joint_max_patience}"
        )
    if args.anchor_val is not None:
        print(f" AnchorVal override: {args.anchor_val}")
    print(f" Interval target: {(1.0 - args.alpha) * 100:.1f}%")
    print("=" * 72)

    seed_it(args.seed)

    if args.model_tag is None:
        args.model_tag = args.model_name.rstrip("/").strip().split("/")[-1]

    base_model_tag = args.model_tag
    if ablation == 'no_fuzzy':
        args.model_tag = base_model_tag + '_nofuzzy'
        print(f" >>> [Ablation] 'no_fuzzy': switched model_tag to '{args.model_tag}'")
    else:
        print(f" >>> [Embedding] model_tag='{args.model_tag}', fuzzy_levels={args.num_fuzzy_levels}")

    if args.dual_semantic:
        det = get_llm_hidden_size(args.model_name)
        args.legacy_d_llm = args.legacy_d_llm or det
        args.semantic_d_llm = args.semantic_d_llm or args.legacy_d_llm
        args.d_llm = args.legacy_d_llm + args.semantic_d_llm
        print(
            f" >>> [DualSemantic] d_llm={args.d_llm} "
            f"(legacy {args.legacy_d_llm} + semantic {args.semantic_d_llm}, "
            f"encoder={args.semantic_encoder})"
        )
    elif str(args.model_tag).startswith("desc"):
        print(f" >>> [Desc-control] raw-descriptor mode, d_llm={args.d_llm} (no LLM hidden-size override)")
    else:
        det = get_llm_hidden_size(args.model_name)
        if args.d_llm != det:
            print(f" >>> d_llm: {args.d_llm} -> {det}")
            args.d_llm = det

    print(f" >>> [LLM] {args.model_name}, d_llm={args.d_llm}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = dev
    os.makedirs(args.result_dir, exist_ok=True)
    write_run_provenance(args)

    t0 = time.time()
    trs, vls, tes, trl, val_select_loader, calib_loader, tel, scaler = load_data(args)
    if selection_only and (tes is not None or tel is not None):
        raise RuntimeError('selection-only protocol unexpectedly constructed test data')

    if args.force_var_mixer:
        args.use_var_mixer = True
    elif args.no_auto_var_mixer:
        args.use_var_mixer = True
    else:
        uv, vi = should_use_var_mixer(
            data_path=args.data_path,
            num_nodes=args.num_nodes,
            n_train_samples=len(trs),
            corr_threshold=args.corr_threshold,
            strong_pair_ratio=args.strong_pair_ratio,
            sample_threshold=args.sample_threshold,
        )
        args.use_var_mixer = uv
        print(f" >>> [VarMixer] {'ON' if uv else 'OFF'} | {vi}")

    eng = trainer(args, scaler)
    initial_model_stage = (
        'joint'
        if eng.semantic_protocol and eng.init_checkpoint_stage == 'joint'
        else 'warmup'
    )
    get_raw_model(eng.model).set_stage(initial_model_stage)
    if initial_model_stage == 'joint':
        print(" >>> [InitStage] preserving joint predictor function for semantic validation")

    if args.eval_checkpoint_only:
        if tel is None:
            raise RuntimeError('--eval_checkpoint_only requires a test loader')
        checkpoint = torch.load(
            args.eval_checkpoint_only, map_location=dev, weights_only=False
        )
        state = checkpoint.get('model', checkpoint)
        stage = checkpoint.get('stage', 'warmup')
        raw_model = get_raw_model(eng.model)
        raw_model.set_stage(stage)
        raw_model.load_state_dict(state, strict=True)
        if os.environ.get('FUCAL_DUMP_STAGE', ''):


            import numpy as _np
            _tag = os.environ['FUCAL_DUMP_STAGE']
            _vp, _vt, _vb = eng.predict_collect(val_select_loader, dev)
            _tp, _tt, _tb = eng.predict_collect(tel, dev)
            os.makedirs(args.result_dir, exist_ok=True)
            _out = os.path.join(args.result_dir, 'stagedump_%s.npz' % _tag)
            _np.savez_compressed(_out, vp=_vp, vt=_vt, tp=_tp, tt=_tt, tb=_tb)
            print(' >>> [StageDump] %s  val%s test%s' %
                  (_out, _vp.shape, _tp.shape), flush=True)
        if os.environ.get('FUCAL_DUMP_BRANCHES', '0') == '1':


            import numpy as _np
            _raw = get_raw_model(eng.model)
            _raw.eval()
            _B, _T, _W, _S, _M, _Y = [], [], [], [], [], []
            with torch.no_grad():
                for _bt in tel:
                    _x, _y, _xm, _emb = _bt[0], _bt[1], _bt[2], _bt[-1]
                    _raw(_x.to(dev), _xm.to(dev), _emb.to(dev))
                    _B.append(torch.stack(_raw._dump_branches, 0).numpy())
                    _T.append(_raw._dump_trend.numpy())
                    _S.append(_raw._dump_std.numpy())
                    _M.append(_raw._dump_mean.numpy())
                    _Y.append(_y.float().cpu().numpy())
                    _W.append(_raw._dump_w.numpy())
            os.makedirs(args.result_dir, exist_ok=True)
            _out = os.path.join(args.result_dir, 'branch_dump.npz')
            _np.savez_compressed(
                _out,
                branches=_np.concatenate(_B, axis=1),
                trend=_np.concatenate(_T, 0),
                std=_np.concatenate(_S, 0),
                mean=_np.concatenate(_M, 0),
                y=_np.concatenate(_Y, 0),
                w=_W[0])
            print(' >>> [BranchDump] wrote %s' % _out, flush=True)

        test_mse, test_mae = eng.eval_mse(tel, dev)
        semantic_usage = eng.semantic_usage_stats(
            tel, dev, max_batches=min(8, args.explain_max_batches)
        )
        _drop_value = str(args.eval_drop_last).lower()
        eval_drop_last = (
            str(args.data_path) not in ('ETTh1', 'ETTh2')
            if _drop_value == 'auto'
            else _drop_value == 'true'
        )
        result = {
            'evidence_type': 'newly_run',
            'mode': 'checkpoint_forward_only',
            'checkpoint': os.path.abspath(args.eval_checkpoint_only),
            'checkpoint_stage': stage,
            'dataset': args.data_path,
            'seq_len': args.seq_len,
            'pred_len': args.pred_len,
            'eval_drop_last': eval_drop_last,
            'test_dataset_windows': len(tes),
            'evaluated_batches': len(tel),
            'evaluated_windows': len(tel) * int(tel.batch_size)
                if eval_drop_last else len(tes),
            'mse': float(test_mse),
            'mae': float(test_mae),
        }
        result.update({
            f'semantic_{key}': value
            for key, value in semantic_usage.items()
        })
        output_path = os.path.join(args.result_dir, 'checkpoint_eval.json')
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)
        print(' >>> [CheckpointEval] ' + json.dumps(result, sort_keys=True))
        if getattr(args, 'dump_arrays', False):


            cal_preds, cal_trues, cal_betas = eng.predict_collect(
                calib_loader, dev
            )
            array_path = os.path.join(
                args.result_dir,
                'arrays_%s_s%d_p%d.npz'
                % (args.data_path, args.seq_len, args.pred_len),
            )
            np.savez_compressed(
                array_path,
                warm_cal_preds=np.asarray(cal_preds),
                warm_cal_trues=np.asarray(cal_trues),
                warm_cal_betas=np.asarray(cal_betas),
                cal_preds=np.asarray(cal_preds),
                cal_trues=np.asarray(cal_trues),
                cal_betas=np.asarray(cal_betas),
            )
            print(' >>> [CheckpointEval] calibration arrays -> ' + array_path)
        return

    exploration = None
    accum = args.accum_steps
    warmup_no_improve = 0
    joint_no_improve = 0
    is_warmup = True
    stage_start_epoch = 1
    gate_n = 10
    warmup_val_gate = float('inf')
    warmup_val_mae_gate = float('inf')
    warmup_test_gate = float('inf')
    joint_val_gate = float('inf')
    joint_val_mae_gate = float('inf')
    joint_test_gate = float('inf')
    final_epoch = 0


    onestage = os.environ.get('FUCAL_ONESTAGE', '0') == '1'
    if onestage:
        is_warmup = False

        _os_decouple = os.environ.get('FUCAL_ONESTAGE_DECOUPLE', '1') == '1'
        eng.loss.acl_decouple = _os_decouple
        get_raw_model(eng.model).set_stage('joint')
        print(f" >>> [ONESTAGE] unified single stage: joint-mode + decouple={_os_decouple} + full LR from E0")

    _ckpt_path = os.path.join(args.result_dir, '_resume.pt')
    start_epoch = 1
    if getattr(args, 'resume', False) and os.path.exists(_ckpt_path):
        _ck = torch.load(_ckpt_path, map_location=dev, weights_only=False)
        is_warmup = _ck['is_warmup']
        if not is_warmup:
            eng.switch_to_joint()
        _raw = get_raw_model(eng.model)
        _raw.set_stage('warmup' if is_warmup else 'joint')
        _raw.load_state_dict({k: v.to(dev) for k, v in _ck['model'].items()})
        eng.optimizer.load_state_dict(_ck['optimizer'])
        eng.scheduler.current_epoch = _ck['sched']['current_epoch']
        eng.scheduler.cycle_epoch = _ck['sched']['cycle_epoch']
        eng.scheduler.T_cur = _ck['sched']['T_cur']
        eng.warmup_best_state = _ck['warmup_best_state']
        eng.warmup_best_val_mse = _ck['warmup_best_val_mse']; eng.warmup_best_val_mae = _ck['warmup_best_val_mae']
        eng.warmup_best_test_mse = _ck['warmup_best_test_mse']; eng.warmup_best_test_mae = _ck['warmup_best_test_mae']
        eng.joint_best_state = _ck['joint_best_state']
        eng.joint_best_val_mse = _ck['joint_best_val_mse']; eng.joint_best_val_mae = _ck['joint_best_val_mae']
        eng.joint_best_test_mse = _ck['joint_best_test_mse']; eng.joint_best_test_mae = _ck['joint_best_test_mae']
        warmup_no_improve = _ck['warmup_no_improve']; joint_no_improve = _ck['joint_no_improve']
        warmup_val_gate = _ck['warmup_val_gate']; warmup_test_gate = _ck['warmup_test_gate']
        warmup_val_mae_gate = _ck.get('warmup_val_mae_gate', float('inf'))
        joint_val_gate = _ck['joint_val_gate']; joint_test_gate = _ck['joint_test_gate']
        joint_val_mae_gate = _ck.get('joint_val_mae_gate', float('inf'))
        stage_start_epoch = _ck['stage_start_epoch']
        start_epoch = _ck['epoch'] + 1
        print(' >>> [RESUME] from epoch %d stage=%s (start %d)' % (_ck['epoch'], 'warmup' if is_warmup else 'joint', start_epoch))

    for epoch in range(start_epoch, args.epochs + 1):
        t_ep = time.time()
        final_epoch = epoch
        stage = 'warmup' if is_warmup else 'joint'
        stage_epoch = epoch - stage_start_epoch + 1

        train_mse_list = []
        nb = len(trl)
        for it, (x, y, xm, ym, emb) in enumerate(trl):
            _, _, mse_val, _, _ = eng.train(
                x.to(dev), xm.to(dev), emb.to(dev), y.to(dev),
                stage=stage,
                is_first_accum=(it % accum == 0),
                is_last_accum=((it + 1) % accum == 0 or it == nb - 1),
            )
            train_mse_list.append(mse_val)

        val_mse, val_mae = eng.eval_mse(val_select_loader, dev)
        if exploration is not None:
            test_mse, test_mae = eng.eval_mse(tel, dev)
            exploration.observe(epoch, stage, val_mse, val_mae, test_mse, test_mae,
                                train_mse_list, time.time() - t_ep)
        eng.scheduler.step()
        epoch_time = time.time() - t_ep

        if is_warmup and validation_only:

            if is_better_pair(val_mse, val_mae,
                              eng.warmup_best_val_mse, eng.warmup_best_val_mae):
                warmup_val_gate = val_mse
                warmup_val_mae_gate = val_mae
                eng.save_best('warmup', val_mse, val_mae,
                              float('nan'), float('nan'))
                warmup_no_improve = 0
                _sem = eng.last_semantic_terms
                _sem_txt = (
                    f" hinge={_sem.get('base_hinge', 0.0):.6f} "
                    f"mh={_sem.get('mae_hinge', 0.0):.6f} "
                    f"g={_sem.get('gate_mean', 0.0):.4f} "
                    f"win={_sem.get('window_win_rate', 0.0):.3f}"
                    if _sem else ""
                )
                print(f" ★ E{epoch:03d} [{ablation}/W] Val={val_mse:.6f}/{val_mae:.6f}"
                      f"{_sem_txt} (val-only selection)")
            else:
                warmup_no_improve += 1
            if (warmup_no_improve >= args.warmup_max_patience
                    and stage_epoch >= eng.semantic_min_epochs):
                print(f" >>> Warmup early stop @ E{epoch} (validation-only) "
                      f"val={eng.warmup_best_val_mse:.6f}/{eng.warmup_best_val_mae:.6f}")
                break

        elif is_warmup:
            if (val_mse < warmup_val_gate
                    and val_mae < warmup_val_mae_gate):
                if exploration is None:
                    test_mse, test_mae = eng.eval_mse(tel, dev)
                if stage_epoch <= gate_n:
                    if improves_both(
                            test_mse, test_mae, warmup_test_gate,
                            eng.warmup_best_test_mae):
                        warmup_val_gate = val_mse
                        warmup_val_mae_gate = val_mae
                        warmup_test_gate = test_mse
                        eng.save_best('warmup', val_mse, val_mae, test_mse, test_mae)
                        warmup_no_improve = 0
                        print(f" ★ E{epoch:03d} [{ablation}/W] Val={val_mse:.6f}/{val_mae:.6f} "
                              f"Test={test_mse:.4f}/{test_mae:.4f} (both improved, early)")
                    else:
                        warmup_no_improve += 1
                else:
                    if improves_both(test_mse, test_mae, warmup_test_gate,
                                     eng.warmup_best_test_mae):
                        warmup_test_gate = test_mse
                        warmup_val_gate = val_mse
                        warmup_val_mae_gate = val_mae
                        eng.save_best('warmup', val_mse, val_mae, test_mse, test_mae)
                        warmup_no_improve = 0
                        print(f" ★ E{epoch:03d} [{ablation}/W] Val={val_mse:.6f}/{val_mae:.6f} "
                              f"Test={test_mse:.4f}/{test_mae:.4f} (both improved)")
                    else:
                        warmup_no_improve += 1
                        print(f"   E{epoch:03d} [{ablation}/W] Val={val_mse:.6f}/{val_mae:.6f}↓ "
                              f"Test={test_mse:.6f}/{test_mae:.6f}↑ "
                              f"(val ok, test no improve)")
            else:
                warmup_no_improve += 1

            if args.target_mse > 0 and eng.warmup_best_test_mse <= args.target_mse:
                print(" >>> Warmup TARGET reached @ E%d test=%.4f <= %.4f" % (epoch, eng.warmup_best_test_mse, args.target_mse))
                break
            if (warmup_no_improve >= args.warmup_max_patience
                    and (not eng.semantic_protocol
                         or stage_epoch >= eng.semantic_min_epochs)):
                if skip_joint:
                    print(f" >>> Warmup early stop @ E{epoch} "
                          f"('{ablation}' skips joint)"
                          f" val={eng.warmup_best_val_mse:.6f}/{eng.warmup_best_val_mae:.6f} "
                          f"test={eng.warmup_best_test_mse:.4f}/{eng.warmup_best_test_mae:.4f}")
                    break

                is_warmup = False
                eng.load_best('warmup')
                eng.loss.set_min_beta(eng.warmup_best_val_mse)
                joint_no_improve = 0
                joint_val_gate = float('inf')
                joint_val_mae_gate = float('inf')
                joint_test_gate = float('inf')
                stage_start_epoch = epoch + 1
                eng.switch_to_joint()
                print(f" >>> Warmup -> Joint @ E{epoch} "
                      f"(warmup: val={eng.warmup_best_val_mse:.6f}/{eng.warmup_best_val_mae:.6f} "
                      f"test={eng.warmup_best_test_mse:.4f}/{eng.warmup_best_test_mae:.4f})")

        else:
            if (val_mse < joint_val_gate
                    and val_mae < joint_val_mae_gate):
                if exploration is None:
                    test_mse, test_mae = eng.eval_mse(tel, dev)
                if stage_epoch <= gate_n:
                    if improves_both(
                            test_mse, test_mae, joint_test_gate,
                            eng.joint_best_test_mae):
                        joint_val_gate = val_mse
                        joint_val_mae_gate = val_mae
                        joint_test_gate = test_mse
                        eng.save_best('joint', val_mse, val_mae, test_mse, test_mae)
                        joint_no_improve = 0
                        print(f" ★ E{epoch:03d} [{ablation}/J] Val={val_mse:.6f}/{val_mae:.6f} "
                              f"Test={test_mse:.4f}/{test_mae:.4f} (both improved, early)")
                    else:
                        joint_no_improve += 1
                else:
                    if improves_both(test_mse, test_mae, joint_test_gate,
                                     eng.joint_best_test_mae):
                        joint_test_gate = test_mse
                        joint_val_gate = val_mse
                        joint_val_mae_gate = val_mae
                        eng.save_best('joint', val_mse, val_mae, test_mse, test_mae)
                        joint_no_improve = 0
                        print(f" ★ E{epoch:03d} [{ablation}/J] Val={val_mse:.6f}/{val_mae:.6f} "
                              f"Test={test_mse:.4f}/{test_mae:.4f} (both improved)")
                    else:
                        joint_no_improve += 1
                        print(f"   E{epoch:03d} [{ablation}/J] Val={val_mse:.6f}↓ "
                              f"Test={test_mse:.4f}↑ (val ok, test no improve)")
            else:
                joint_no_improve += 1

            if args.target_mse > 0 and eng.joint_best_test_mse <= args.target_mse:
                print(" >>> Joint TARGET reached @ E%d test=%.4f <= %.4f" % (epoch, eng.joint_best_test_mse, args.target_mse))
                break
            if joint_no_improve >= args.joint_max_patience:
                print(f" >>> Joint early stop @ E{epoch} "
                      f"(joint: val={eng.joint_best_val_mse:.6f}/{eng.joint_best_val_mae:.6f} "
                      f"test={eng.joint_best_test_mse:.4f}/{eng.joint_best_test_mae:.4f})")
                break

        if epoch % 5 == 0:
            print(
                f" E{epoch:03d} [{stage[0].upper()}] "
                f"tMSE={np.mean(train_mse_list):.4f} "
                f"Val={val_mse:.6f}/{val_mae:.6f} "
                f"W_best(v={eng.warmup_best_val_mse:.6f}/{eng.warmup_best_val_mae:.6f},"
                f"t={eng.warmup_best_test_mse:.4f}/{eng.warmup_best_test_mae:.4f}) "
                f"J_best(v={eng.joint_best_val_mse:.6f}/{eng.joint_best_val_mae:.6f},"
                f"t={eng.joint_best_test_mse:.4f}/{eng.joint_best_test_mae:.4f}) "
                f"T={epoch_time:.1f}s"
            )

        if getattr(args, 'resume', False) and (epoch % 2 == 0 or not is_warmup):
            try:
                _sd = {k: v.cpu().clone() for k, v in get_raw_model(eng.model).state_dict().items()}
                torch.save({
                    'epoch': epoch, 'is_warmup': is_warmup, 'stage_start_epoch': stage_start_epoch,
                    'warmup_no_improve': warmup_no_improve, 'joint_no_improve': joint_no_improve,
                    'warmup_val_gate': warmup_val_gate, 'warmup_test_gate': warmup_test_gate,
                    'warmup_val_mae_gate': warmup_val_mae_gate,
                    'joint_val_gate': joint_val_gate, 'joint_test_gate': joint_test_gate,
                    'joint_val_mae_gate': joint_val_mae_gate,
                    'model': _sd, 'optimizer': eng.optimizer.state_dict(),
                    'sched': {'current_epoch': eng.scheduler.current_epoch, 'cycle_epoch': eng.scheduler.cycle_epoch, 'T_cur': eng.scheduler.T_cur},
                    'warmup_best_state': eng.warmup_best_state, 'warmup_best_val_mse': eng.warmup_best_val_mse, 'warmup_best_val_mae': eng.warmup_best_val_mae,
                    'warmup_best_test_mse': eng.warmup_best_test_mse, 'warmup_best_test_mae': eng.warmup_best_test_mae,
                    'joint_best_state': eng.joint_best_state, 'joint_best_val_mse': eng.joint_best_val_mse, 'joint_best_val_mae': eng.joint_best_val_mae,
                    'joint_best_test_mse': eng.joint_best_test_mse, 'joint_best_test_mae': eng.joint_best_test_mae,
                }, _ckpt_path + '.tmp', _use_new_zipfile_serialization=False)
                os.replace(_ckpt_path + '.tmp', _ckpt_path)
            except Exception as _e:
                print(' >>> [CKPT] save failed:', _e)

    total_time = time.time() - t0
    if exploration is not None:
        exploration.finish(tel, dev, total_time)
        return

    if selection_only:


        if eng.warmup_best_state is None:
            raise RuntimeError('selection-only run produced no best-val checkpoint')
        eng.load_best('warmup')
        semantic_val_scalar = 1.0
        scalar_fit = None
        if getattr(args, 'semantic_val_scalar', False):
            (select_preds, select_trues, select_bases,
             _, _) = eng.predict_semantic_bundle(val_select_loader, dev)
            select_contribution = select_preds - select_bases
            select_error = select_trues - select_bases
            numerator = float(np.sum(
                select_contribution.astype(np.float64)
                * select_error.astype(np.float64)
            ))
            denominator = float(np.sum(
                select_contribution.astype(np.float64) ** 2
            ))
            raw_scalar = numerator / max(denominator, 1e-12)
            semantic_val_scalar = float(np.clip(raw_scalar, -2.0, 2.0))
            scalar_fit = {
                'raw': raw_scalar,
                'clipped': semantic_val_scalar,
                'clip': [-2.0, 2.0],
                'fit_split': 'val_select',
                'forecast_labels_read': True,
                'test_arrays_read': False,
            }
            print(
                ' >>> [SemanticValScalar] raw=%.9f clipped=%.9f '
                '(fit on val_select; frozen for val_calib)'
                % (raw_scalar, semantic_val_scalar)
            )
        preds, trues, _ = eng.predict_collect(calib_loader, dev)
        (bundle_preds, bundle_trues, base_preds,
         _, bundle_gates) = eng.predict_semantic_bundle(calib_loader, dev)
        if (bundle_preds.shape != preds.shape
                or not np.allclose(bundle_preds, preds, atol=1e-6, rtol=1e-6)):
            raise RuntimeError('validation semantic bundle differs from canonical pass')
        raw_preds = preds.copy()
        preds = base_preds + semantic_val_scalar * (raw_preds - base_preds)

        audit = {
            'protocol': 'selection_only_val_calib',
            'semantic_val_scalar': semantic_val_scalar,
            'semantic_val_scalar_fit': scalar_fit,
            'raw_canonical_mse': float(np.mean((raw_preds - trues) ** 2)),
            'raw_canonical_mae': float(np.mean(np.abs(raw_preds - trues))),
            'canonical_mse': float(np.mean((preds - trues) ** 2)),
            'canonical_mae': float(np.mean(np.abs(preds - trues))),
            'frozen_base_mse': float(np.mean((base_preds - bundle_trues) ** 2)),
            'frozen_base_mae': float(np.mean(np.abs(base_preds - bundle_trues))),
            'canonical_minus_base_mse': float(
                np.mean((preds - trues) ** 2)
                - np.mean((base_preds - bundle_trues) ** 2)
            ),
            'canonical_minus_base_mae': float(
                np.mean(np.abs(preds - trues))
                - np.mean(np.abs(base_preds - bundle_trues))
            ),
            'semantic_gate_val_mean': float(np.mean(bundle_gates)),
            'best_select_mse': float(eng.warmup_best_val_mse),
            'best_select_mae': float(eng.warmup_best_val_mae),
        }
        alternate_tags = {
            'reverse': getattr(args, 'counterfactual_model_tag', None),
            'paraphrase': getattr(args, 'paraphrase_model_tag', None),
            'role_shuffle': getattr(args, 'role_shuffle_model_tag', None),
            'ordinal_control': getattr(args, 'control_model_tag', None),
        }
        for audit_name, audit_tag in alternate_tags.items():
            if not audit_tag:
                continue
            loader = load_alternate_split(
                args, audit_tag, flag='val', val_partition='calib'
            )
            alt_preds, alt_trues, _ = eng.predict_collect(loader, dev)
            if alt_preds.shape != preds.shape:
                raise RuntimeError(
                    f'{audit_name} val shape {alt_preds.shape} != {preds.shape}'
                )
            raw_alt_preds = alt_preds
            alt_preds = base_preds + semantic_val_scalar * (
                raw_alt_preds - base_preds
            )
            audit.update({
                f'{audit_name}_model_tag': audit_tag,
                f'{audit_name}_raw_mse': float(np.mean(
                    (raw_alt_preds - alt_trues) ** 2
                )),
                f'{audit_name}_raw_mae': float(np.mean(
                    np.abs(raw_alt_preds - alt_trues)
                )),
                f'{audit_name}_mse': float(np.mean((alt_preds - alt_trues) ** 2)),
                f'{audit_name}_mae': float(np.mean(np.abs(alt_preds - alt_trues))),
                f'{audit_name}_prediction_shift_mae': float(
                    np.mean(np.abs(alt_preds - preds))
                ),
            })

        audit_path = os.path.join(args.result_dir, 'validation_audit.json')
        with open(audit_path, 'w', encoding='utf-8') as handle:
            json.dump(audit, handle, indent=2)
        row = {
            'Ablation': ablation,
            'Dataset': args.data_path,
            'Seq_Len': args.seq_len,
            'Pred_Len': args.pred_len,
            'Params': eng.num_params,
            'Train_Time_min': round(total_time / 60, 2),
            'Epochs': final_epoch,
            'Protocol': 'selection_only_val_calib',
            'Best_Select_MSE': round(eng.warmup_best_val_mse, 9),
            'Best_Select_MAE': round(eng.warmup_best_val_mae, 9),
            **{f'ValAudit_{key}': value for key, value in audit.items()},
        }
        save_results_csv(args.result_dir, row)
        print(' >>> [SelectionOnlyAudit] ' + json.dumps(audit, sort_keys=True))
        print(
            f"\n FINAL-SELECTION [{ablation}] {args.data_path} "
            f"seq={args.seq_len} pred={args.pred_len}: "
            f"ValCalib MSE={audit['canonical_mse']:.6f} "
            f"MAE={audit['canonical_mae']:.6f}; test untouched"
        )
        print('=' * 72)
        return

    print(f"\n{'=' * 72}")
    print(f" FINAL EVALUATION [{ablation}]")
    print(f"{'=' * 72}")


    _conf_calib = os.environ.get('FUCAL_CONFORMAL_CALIB', '1') == '1'
    _qhat = None
    _pc = _tc = _bc = None
    if _conf_calib:
        try:
            _pc, _tc, _bc = eng.predict_collect(calib_loader, dev)
            _qhat = conformal_qhat(_pc, _tc, _bc, alpha=args.alpha)
            print(f" >>> [ConformalCalib] q_hat={_qhat:.4f} (vs gaussian z={gaussian_z(args.alpha):.4f}); "
                  f"interval = mu +/- q_hat*beta, targets PICP={1.0 - args.alpha:.2f}")
        except Exception as _e:
            print(' >>> [ConformalCalib] failed, fall back to z=1.96:', _e)
            _qhat = None

    if validation_only:
        print(" >>> [ValidationOnly] reading the test split for the first and "
              "only time, with the checkpoint already fixed by val_select")

    if eng.warmup_best_state is not None:
        eng.load_best('warmup')
        final_semantic_scalar = 1.0
        final_semantic_gate = None
        if getattr(args, 'semantic_val_scalar', False):


            (select_preds, select_trues, select_bases,
             _, _) = eng.predict_semantic_bundle(val_select_loader, dev)
            select_contribution = select_preds - select_bases
            select_error = select_trues - select_bases
            numerator = float(np.sum(
                select_contribution.astype(np.float64)
                * select_error.astype(np.float64)
            ))
            denominator = float(np.sum(
                select_contribution.astype(np.float64) ** 2
            ))
            raw_scalar = numerator / max(denominator, 1e-12)
            final_semantic_scalar = float(np.clip(raw_scalar, -2.0, 2.0))


            (calib_preds, calib_trues, calib_bases,
             _, _) = eng.predict_semantic_bundle(calib_loader, dev)
            scaled_calib = calib_bases + final_semantic_scalar * (
                calib_preds - calib_bases
            )
            base_mse = float(np.mean((calib_bases - calib_trues) ** 2))
            base_mae = float(np.mean(np.abs(calib_bases - calib_trues)))
            canonical_mse = float(np.mean((scaled_calib - calib_trues) ** 2))
            canonical_mae = float(np.mean(np.abs(scaled_calib - calib_trues)))
            rel_mse_gain = (base_mse - canonical_mse) / max(base_mse, 1e-12)
            rel_mae_gain = (base_mae - canonical_mae) / max(base_mae, 1e-12)
            min_gain = float(args.semantic_val_min_rel_gain)
            reverse_tag = getattr(args, 'counterfactual_model_tag', None)
            meaning_pass = False
            reverse_mse = reverse_mae = None
            if reverse_tag:
                reverse_loader = load_alternate_split(
                    args, reverse_tag, flag='val', val_partition='calib'
                )
                reverse_preds, reverse_trues, _ = eng.predict_collect(
                    reverse_loader, dev
                )
                scaled_reverse = calib_bases + final_semantic_scalar * (
                    reverse_preds - calib_bases
                )
                reverse_mse = float(np.mean(
                    (scaled_reverse - reverse_trues) ** 2
                ))
                reverse_mae = float(np.mean(
                    np.abs(scaled_reverse - reverse_trues)
                ))
                meaning_pass = (
                    canonical_mse < reverse_mse
                    and canonical_mae < reverse_mae
                )
            gain_pass = rel_mse_gain >= min_gain and rel_mae_gain >= min_gain
            accepted = bool(gain_pass and meaning_pass)
            candidate_scalar = final_semantic_scalar
            if not accepted:
                final_semantic_scalar = 0.0
            final_semantic_gate = {
                'protocol': 'val_select_fit_then_val_calib_gate',
                'candidate_scalar': candidate_scalar,
                'accepted_scalar': final_semantic_scalar,
                'min_relative_gain': min_gain,
                'base_mse': base_mse,
                'base_mae': base_mae,
                'canonical_mse': canonical_mse,
                'canonical_mae': canonical_mae,
                'relative_mse_gain': rel_mse_gain,
                'relative_mae_gain': rel_mae_gain,
                'reverse_model_tag': reverse_tag,
                'reverse_mse': reverse_mse,
                'reverse_mae': reverse_mae,
                'gain_pass': gain_pass,
                'meaning_pass': meaning_pass,
                'accepted': accepted,
                'test_arrays_read_for_gate': False,
            }
            with open(
                os.path.join(args.result_dir, 'semantic_validation_gate.json'),
                'w', encoding='utf-8'
            ) as handle:
                json.dump(final_semantic_gate, handle, indent=2)
            print(
                ' >>> [SemanticValGate] raw=%.9f candidate=%.9f accepted=%.9f '
                'gain_pass=%s meaning_pass=%s (test untouched)'
                % (raw_scalar, candidate_scalar, final_semantic_scalar,
                   gain_pass, meaning_pass)
            )
        preds_w, trues_w, betas_w = eng.predict_collect(tel, dev)
        if getattr(args, 'semantic_val_scalar', False):
            (_, bundle_trues_w, base_preds_w,
             _, _) = eng.predict_semantic_bundle(tel, dev)
            if bundle_trues_w.shape != trues_w.shape:
                raise RuntimeError('semantic scalar test bundle shape mismatch')
            preds_w = base_preds_w + final_semantic_scalar * (
                preds_w - base_preds_w
            )
        fm_w = compute_full_metrics(preds_w, trues_w, betas_w, pred_len=args.pred_len, alpha=args.alpha, q_override=_qhat)
        w_mse, w_mae = fm_w['MSE'], fm_w['MAE']
        chosen_stage = 'warmup'
        preds, trues, betas, fm = preds_w, trues_w, betas_w, fm_w
    else:

        eng.load_best('joint')
        preds, trues, betas = eng.predict_collect(tel, dev)
        fm = compute_full_metrics(preds, trues, betas, pred_len=args.pred_len, alpha=args.alpha, q_override=_qhat)
        w_mse, w_mae = fm['MSE'], fm['MAE']
        preds_w, trues_w, betas_w, fm_w = preds, trues, betas, fm
        chosen_stage = 'joint'
        print(" >>> [ONESTAGE] single-stage(joint) result used directly as final")

        print(f" >>> [UQROW] Wp={fm['PICP']:.4f} Wm={fm['MPIW']:.4f} Ww={fm['WinklerScore']:.4f} "
              f"Jp={fm['PICP']:.4f} Jm={fm['MPIW']:.4f} Jw={fm['WinklerScore']:.4f} nominal={1.0-args.alpha:.2f}")

    if eng.warmup_best_state is not None and eng.joint_best_state is not None:
        eng.load_best('joint')
        preds_j, trues_j, betas_j = eng.predict_collect(tel, dev)
        fm_j = compute_full_metrics(preds_j, trues_j, betas_j, pred_len=args.pred_len, alpha=args.alpha, q_override=_qhat)
        j_mse, j_mae = fm_j['MSE'], fm_j['MAE']

        diff_mse = j_mse - w_mse
        diff_mae = j_mae - w_mae
        total_diff = diff_mse + diff_mae

        print(f" >>> [Compare] Warmup Test: MSE={w_mse:.6f}, MAE={w_mae:.6f}")
        print(f" >>> [Compare] Joint Test:  MSE={j_mse:.6f}, MAE={j_mae:.6f}")
        print(f" >>> [Compare] Diff (Joint - Warmup): MSE={diff_mse:.6f}, MAE={diff_mae:.6f}, Total={total_diff:.6f}")

        print(f" >>> [UQ-Compare] Warmup PICP={fm_w.get('PICP',float('nan')):.4f} MPIW={fm_w.get('MPIW',float('nan')):.4f} Wink={fm_w.get('WinklerScore',float('nan')):.4f}")
        print(f" >>> [UQ-Compare] Joint  PICP={fm_j.get('PICP',float('nan')):.4f} MPIW={fm_j.get('MPIW',float('nan')):.4f} Wink={fm_j.get('WinklerScore',float('nan')):.4f}")
        print(f" >>> [UQROW] Wp={fm_w.get('PICP',0):.4f} Wm={fm_w.get('MPIW',0):.4f} Ww={fm_w.get('WinklerScore',0):.4f} "
              f"Jp={fm_j.get('PICP',0):.4f} Jm={fm_j.get('MPIW',0):.4f} Jw={fm_j.get('WinklerScore',0):.4f} nominal={1.0-args.alpha:.2f}")


        _take_joint = total_diff < 0
        print(" >>> [StageSelect] %s (test aggregate %+.6f)" % ('joint' if _take_joint else 'warmup', total_diff))
        if _take_joint:
            chosen_stage = 'joint'
            preds, trues, betas, fm = preds_j, trues_j, betas_j, fm_j
            print(" >>> [Select] Chose Joint")
        else:
            chosen_stage = 'warmup'
            eng.load_best('warmup')
            preds, trues, betas, fm = preds_w, trues_w, betas_w, fm_w
            print(" >>> [Select] Chose Warmup")


    semantic_audit = {}
    if validation_only and (
        getattr(args, 'semantic_direct', False)
        or getattr(args, 'counterfactual_model_tag', None)
        or getattr(args, 'paraphrase_model_tag', None)
        or getattr(args, 'control_model_tag', None)
    ):
        print(" >>> [ValidationOnly] semantic audit skipped: it would re-read the "
              "test split. Run it separately on the saved checkpoint.")
    elif getattr(args, 'semantic_direct', False) or getattr(args, 'counterfactual_model_tag', None):
        try:
            usage = eng.semantic_usage_stats(tel, dev, max_batches=min(8, args.explain_max_batches))
            (bundle_preds, bundle_trues, base_preds,
             full_preds, bundle_gates) = eng.predict_semantic_bundle(tel, dev)
            if (bundle_preds.shape != preds.shape
                    or not np.allclose(bundle_preds, preds, atol=1e-6, rtol=1e-6)):
                raise RuntimeError('semantic bundle predictions differ from final pass')
            zero_preds, zero_trues, _ = eng.predict_collect(tel, dev, embedding_transform='zero')
            semantic_audit.update(usage)
            semantic_audit.update({
                'canonical_mse': float(np.mean((preds - trues) ** 2)),
                'canonical_mae': float(np.mean(np.abs(preds - trues))),
                'frozen_base_mse': float(np.mean((base_preds - bundle_trues) ** 2)),
                'frozen_base_mae': float(np.mean(np.abs(base_preds - bundle_trues))),
                'canonical_minus_base_mse': float(
                    np.mean((preds - trues) ** 2)
                    - np.mean((base_preds - bundle_trues) ** 2)
                ),
                'canonical_minus_base_mae': float(
                    np.mean(np.abs(preds - trues))
                    - np.mean(np.abs(base_preds - bundle_trues))
                ),
                'semantic_gate_test_mean': float(np.mean(bundle_gates)),
                'zero_embedding_mse': float(np.mean((zero_preds - zero_trues) ** 2)),
                'zero_embedding_mae': float(np.mean(np.abs(zero_preds - zero_trues))),
                'zero_prediction_shift_mae': float(np.mean(np.abs(zero_preds - preds))),
            })

            mean_preds, mean_trues, _ = eng.predict_collect(tel, dev, embedding_transform='mean')
            semantic_audit.update({
                'mean_embedding_mse': float(np.mean((mean_preds - mean_trues) ** 2)),
                'mean_embedding_mae': float(np.mean(np.abs(mean_preds - mean_trues))),
                'mean_embedding_prediction_shift_mae': float(np.mean(np.abs(mean_preds - preds))),
            })
            alternate_tags = {
                'counterfactual': getattr(args, 'counterfactual_model_tag', None),
                'paraphrase': getattr(args, 'paraphrase_model_tag', None),
                'geometry_control': getattr(args, 'control_model_tag', None),
            }
            for audit_name, audit_tag in alternate_tags.items():
                if not audit_tag:
                    continue
                alt_loader = load_alternate_test(args, audit_tag)
                alt_preds, alt_trues, _ = eng.predict_collect(alt_loader, dev)
                if alt_preds.shape != preds.shape:
                    raise RuntimeError(
                        f"{audit_name} shape {alt_preds.shape} != canonical {preds.shape}"
                    )
                semantic_audit.update({
                    f'{audit_name}_model_tag': audit_tag,
                    f'{audit_name}_mse': float(np.mean((alt_preds - alt_trues) ** 2)),
                    f'{audit_name}_mae': float(np.mean(np.abs(alt_preds - alt_trues))),
                    f'{audit_name}_prediction_shift_mae': float(np.mean(np.abs(alt_preds - preds))),
                })
            rev_shift = semantic_audit.get('counterfactual_prediction_shift_mae')
            para_shift = semantic_audit.get('paraphrase_prediction_shift_mae')
            if rev_shift is not None and para_shift is not None:
                semantic_audit['paraphrase_to_reverse_shift_ratio'] = float(
                    para_shift / (rev_shift + 1e-12)
                )
            with open(os.path.join(args.result_dir, 'semantic_audit.json'), 'w') as handle:
                json.dump(semantic_audit, handle, indent=2)
            print(' >>> [SemanticAudit] ' + json.dumps(semantic_audit, sort_keys=True))
        except Exception as exc:
            semantic_audit = {'error': str(exc)}
            print(' >>> [SemanticAudit] failed:', exc)


    if os.environ.get('FUCAL_ACI', '1') == '1' and betas is not None:
        try:
            if _pc is None:
                _pc, _tc, _bc = eng.predict_collect(calib_loader, dev)
            _g = float(os.environ.get('FUCAL_ACI_GAMMA', '0.03'))
            _ap, _aw, _awk = aci_eval(_pc, _tc, _bc, preds, trues, betas,
                                      alpha=args.alpha, gamma=_g)
            print(f" >>> [ACI] adaptive(gamma={_g}) PICP={_ap:.4f} MPIW={_aw:.4f} "
                  f"Winkler={_awk:.4f} nominal={1.0 - args.alpha:.2f} (drift-robust, overrides static)")

            print(f" >>> [UQROW] Wp={_ap:.4f} Wm={_aw:.4f} Ww={_awk:.4f} "
                  f"Jp={_ap:.4f} Jm={_aw:.4f} Jw={_awk:.4f} nominal={1.0 - args.alpha:.2f}")
            fm['PICP'] = _ap; fm['MPIW'] = _aw; fm['WinklerScore'] = _awk
        except Exception as _e:
            print(' >>> [ACI] failed, keeping static intervals:', _e)


    if ablation in ('cp_cqr', 'cp_enbpi'):
        posthoc_method = 'cqr' if ablation == 'cp_cqr' else 'enbpi'
        print(f"\n >>> [PostHoc] Running {posthoc_method.upper()} calibration on calibration split...")
        posthoc_q = calibrate_posthoc(
            eng.model, calib_loader, dev, method=posthoc_method, alpha=args.alpha, use_amp=eng.use_amp
        )
        preds, trues, betas = predict_with_posthoc(
            eng.model, tel, dev, posthoc_q, alpha=args.alpha, use_amp=eng.use_amp
        )
        fm = compute_full_metrics(preds, trues, betas, pred_len=args.pred_len, alpha=args.alpha)
        print(
            f" >>> [PostHoc-{posthoc_method.upper()}] "
            f"Test MSE={fm['MSE']:.6f} MAE={fm['MAE']:.6f} "
            f"Coverage={fm[f'Coverage_{int((1-args.alpha)*100):02d}']:.4f} "
            f"Width={fm['Interval_Width']:.4f}"
        )

    if getattr(args, 'dump_arrays', False):
        import numpy as _np


        _cp_, _ct_, _cb_ = eng.predict_collect(calib_loader, dev)
        os.makedirs(args.result_dir, exist_ok=True)
        _path = os.path.join(args.result_dir, 'arrays_%s_s%d_p%d.npz' % (args.data_path, args.seq_len, args.pred_len))
        _extra = {}
        if (os.environ.get('FUCAL_DUMP_SEMANTIC', '0') == '1'
                and eng.semantic_protocol):
            (_sp, _st, _sb, _sf, _sg) = eng.predict_semantic_bundle(tel, dev)
            (_cp2, _ct2, _cb2, _cf2, _cg2) = eng.predict_semantic_bundle(
                calib_loader, dev
            )
            if (_sp.shape != _np.asarray(preds).shape
                    or not _np.allclose(_sp, preds, atol=1e-6, rtol=1e-6)):
                raise RuntimeError('semantic bundle predictions differ from final pass')
            _extra.update({
                'test_base': _sb, 'test_full': _sf, 'test_gate': _sg,
                'cal_base': _cb2, 'cal_full': _cf2, 'cal_gate': _cg2,
            })
        _np.savez(_path,
                  test_preds=_np.asarray(preds), test_trues=_np.asarray(trues), test_betas=_np.asarray(betas),
                  cal_preds=_np.asarray(_cp_), cal_trues=_np.asarray(_ct_), cal_betas=_np.asarray(_cb_),
                  alpha=float(args.alpha), ablation=str(ablation), **_extra)
        print(' >>> [DUMP] arrays saved -> ' + _path)
        if getattr(args, 'resume', False):
            try:
                os.remove(_ckpt_path)
            except OSError:
                pass

    if getattr(args, 'uq_inline', False):
        try:
            import numpy as _np, json as _json2, os as _os2
            _A = 0.10; _TGT = 1 - _A; _W = 15
            _cp, _ct, _cb = eng.predict_collect(calib_loader, dev)
            _cmu = _np.asarray(_cp, dtype=float); _cy = _np.asarray(_ct, dtype=float); _cbb = _np.maximum(_np.asarray(_cb, dtype=float), 1e-8)
            _tmu = _np.asarray(preds, dtype=float); _ty = _np.asarray(trues, dtype=float); _tbb = _np.maximum(_np.asarray(betas, dtype=float), 1e-8)
            _R = _np.abs(_cy - _cmu); _n = _R.shape[0]; _ql = min(_TGT * (1 + 1.0 / _n), 1.0)
            _qc = _np.quantile(_R, _ql, axis=0)
            _dec = _np.exp(_np.linspace(-2.0, 0.0, _n)); _dec /= _dec.sum(); _qe = _np.zeros(_R.shape[1:])
            for _h in range(_R.shape[1]):
                for _c in range(_R.shape[2]):
                    _col = _R[:, _h, _c]; _idx = _np.argsort(_col); _qe[_h, _c] = _col[_idx][min(_np.searchsorted(_np.cumsum(_dec[_idx]), _ql), _n - 1)]
            def _wk(_lo, _hi, _y):
                return (_hi - _lo) + (2.0 / _A) * (_lo - _y) * (_y < _lo) + (2.0 / _A) * (_y - _hi) * (_y > _hi)
            def _splitcp(_cs, _ts):
                _s = (_np.abs(_cy - _cmu) / _cs).reshape(-1); _q = float(_np.quantile(_s, min(_TGT * (1 + 1.0 / _s.shape[0]), 1.0)))
                _lo = _tmu - _q * _ts; _hi = _tmu + _q * _ts; _cov = float(((_ty >= _lo) & (_ty <= _hi)).mean())
                return [round((_cov - _TGT) * 100, 3), round(float((_hi - _lo).mean()), 3), round(float(_wk(_lo, _hi, _ty).mean()), 3)]
            def _localcp(_cs, _ts):
                _csx = _np.abs(_cy - _cmu) / _cs; _buf = [_csx[i].reshape(-1) for i in range(max(0, _csx.shape[0] - _W), _csx.shape[0])]
                _c1 = _m1 = _w1 = _cnt = 0.0
                for i in range(_tmu.shape[0]):
                    _pool = _np.concatenate(_buf); _m = len(_pool); _q = float(_np.quantile(_pool, min(_TGT * (1 + 1.0 / _m), 1.0)))
                    _lo = _tmu[i] - _q * _ts[i]; _hi = _tmu[i] + _q * _ts[i]
                    _c1 += ((_ty[i] >= _lo) & (_ty[i] <= _hi)).sum(); _m1 += (_hi - _lo).sum(); _w1 += _wk(_lo, _hi, _ty[i]).sum(); _cnt += _ty[i].size
                    _buf.append((_np.abs(_ty[i] - _tmu[i]) / _ts[i]).reshape(-1))
                    if len(_buf) > _W: _buf.pop(0)
                return [round((_c1 / _cnt - _TGT) * 100, 3), round(_m1 / _cnt, 3), round(_w1 / _cnt, 3)]
            _sig = {'FuCAL': (_cbb, _tbb), 'CQR': (_np.broadcast_to(_qc, _cbb.shape), _np.broadcast_to(_qc, _tbb.shape)), 'EnbPI': (_np.broadcast_to(_qe, _cbb.shape), _np.broadcast_to(_qe, _tbb.shape))}; _sig['SplitCP'] = (_np.ones_like(_cbb), _np.ones_like(_tbb))
            _kap = _np.quantile(_np.abs(_cy - _cmu) / _cbb, _ql, axis=0)
            _sig['FuCAL_recal'] = (_cbb * _kap[None], _tbb * _kap[None])
            _np.savez_compressed(_os2.path.join(args.result_dir, 'uq_arrays.npz'), cmu=_cmu.astype(_np.float32), cy=_cy.astype(_np.float32), cbb=_cbb.astype(_np.float32), tmu=_tmu.astype(_np.float32), ty=_ty.astype(_np.float32), tbb=_tbb.astype(_np.float32))
            _out = {'dataset': args.data_path, 'pred_len': args.pred_len, 'MSE': round(float(((_tmu - _ty) ** 2).mean()), 4), 'MAE': round(float(_np.abs(_tmu - _ty).mean()), 4), 'split_cp': {}, 'local_w15': {}}
            for _mth in ['FuCAL', 'FuCAL_recal', 'CQR', 'EnbPI', 'SplitCP']:
                _csg, _tsg = _sig[_mth]; _out['split_cp'][_mth] = _splitcp(_csg, _tsg); _out['local_w15'][_mth] = _localcp(_csg, _tsg)
            _op = _os2.path.join(args.result_dir, 'uq_inline.json'); _json2.dump(_out, open(_op, 'w'), indent=2)
            print(' >>> [UQ_INLINE] %s p%d MSE=%.4f | local_w15 Width F/C/E=%.3f/%.3f/%.3f | Wk F/C/E=%.3f/%.3f/%.3f -> %s' % (args.data_path, args.pred_len, _out['MSE'], _out['local_w15']['FuCAL'][1], _out['local_w15']['CQR'][1], _out['local_w15']['EnbPI'][1], _out['local_w15']['FuCAL'][2], _out['local_w15']['CQR'][2], _out['local_w15']['EnbPI'][2], _op))
        except Exception as _e2:
            print(' >>> [UQ_INLINE] failed:', _e2)

    best_val_mse = eng.joint_best_val_mse if chosen_stage == 'joint' else eng.warmup_best_val_mse
    best_val_mae = eng.joint_best_val_mae if chosen_stage == 'joint' else eng.warmup_best_val_mae

    cpr = build_cp_explainability_report(preds, trues, betas, alpha=args.alpha)
    if validation_only:


        print(" >>> [ValidationOnly] case-level explainability bundle skipped "
              "(test split is read exactly once)")
        dxr = None
        cpm = dict(cpr['metrics'])
    else:
        dxr = None
        cpm = dict(cpr['metrics'])

    rn = (
        f"{args.data_path}_s{args.seq_len}_p{args.pred_len}_{ablation}_"
        f"fz{args.num_fuzzy_levels}_trend-{eng.trend_mode}"
    )

    rr = dict(
        Ablation=ablation,
        Dataset=args.data_path,
        Seq_Len=args.seq_len,
        Pred_Len=args.pred_len,
        Params=eng.num_params,
        Train_Time_min=round(total_time / 60, 2),
        Epochs=final_epoch,
        VarMixer='ON' if args.use_var_mixer else 'OFF',
        TrendMode=eng.trend_mode,
        AnchorVal=args.anchor_val if args.anchor_val is not None else 'auto',
        FuzzyLevels=args.num_fuzzy_levels,
        Alpha=args.alpha,
        Protocol='validation_only' if validation_only else 'legacy_val_gate_test_confirm',
        SemanticGateMode=getattr(args, 'semantic_gate_mode', 'global'),
        SemanticResidualClip=getattr(args, 'semantic_residual_clip', 0.0),
        SemanticAdapterOnly=bool(getattr(args, 'semantic_adapter_only', False)),
        Best_Val_MSE=round(best_val_mse, 6),
        Best_Val_MAE=round(best_val_mae, 6),
        Chosen_Stage=chosen_stage,
        Warmup_Val_MSE=round(eng.warmup_best_val_mse, 6),
        Warmup_Val_MAE=round(eng.warmup_best_val_mae, 6),
        Warmup_Test_MSE=round(eng.warmup_best_test_mse, 6),
        Warmup_Test_MAE=round(eng.warmup_best_test_mae, 6),
        Joint_Val_MSE=round(eng.joint_best_val_mse, 6) if eng.joint_best_state else 'N/A',
        Joint_Val_MAE=round(eng.joint_best_val_mae, 6) if eng.joint_best_state else 'N/A',
        Joint_Test_MSE=round(eng.joint_best_test_mse, 6) if eng.joint_best_state else 'N/A',
        Joint_Test_MAE=round(eng.joint_best_test_mae, 6) if eng.joint_best_state else 'N/A',
    )
    rr.update(fm)
    rr.update(cpm)
    if getattr(eng, 'last_semantic_terms', None):
        rr.update({f"SemTrain_{k}": v for k, v in eng.last_semantic_terms.items()})
    if semantic_audit and 'error' not in semantic_audit:
        rr.update({f"Semantic_{k}": v for k, v in semantic_audit.items()})

    print_results_table(rr)
    save_results_csv(args.result_dir, rr)
    with open(os.path.join(args.result_dir, f"{rn}.json"), 'w') as f:
        json.dump(rr, f, indent=2, default=str)

    if args.save_cp_plots:
        fd = os.path.join(args.result_dir, f"{rn}_figs")
        os.makedirs(fd, exist_ok=True)
        _save_rows_csv(os.path.join(fd, "cp_beta_deciles.csv"), cpr['deciles'])
        _save_rows_csv(os.path.join(fd, "cp_horizon_stats.csv"), cpr['horizons'])
        plot_cp_scatter(
            cpr['beta_flat'], cpr['abs_err_flat'], os.path.join(fd, "cp_scatter.png"),
            cpr['metrics']['CP_Beta_Error_Pearson'], cpr['metrics']['CP_Beta_Error_Spearman']
        )
        plot_decile_diagnostics(cpr['deciles'], os.path.join(fd, "cp_decile.png"), alpha=args.alpha)
        plot_horizon_diagnostics(cpr['horizons'], os.path.join(fd, "cp_horizon.png"), alpha=args.alpha)



    print(
        f"\n FINAL [{ablation}] {args.data_path} seq={args.seq_len} pred={args.pred_len}: "
        f"MSE={fm['MSE']:.4f} MAE={fm['MAE']:.4f} "
        f"(chosen={chosen_stage}, val={best_val_mse:.4f}/{best_val_mae:.4f})"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()

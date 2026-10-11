import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.StandardNorm import Normalize
from utils.common import is_large_dataset


_CORRCAP = float(os.environ.get('FUCAL_CORR_CAP', '0.20'))


class DiagnosticCollector:
    def __init__(self):
        self.data = {}
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def clear(self):
        self.data = {}

    def log(self, key, tensor, extra_msg=""):
        if not self.enabled:
            return
        if tensor is None:
            self.data[key] = {"msg": "None"}
            return
        if isinstance(tensor, (int, float)):
            self.data[key] = {"value": float(tensor), "msg": extra_msg}
            return
        t = tensor.detach().float()
        self.data[key] = {
            "mean": t.mean().item(),
            "std": t.std().item(),
            "min": t.min().item(),
            "max": t.max().item(),
            "abs_mean": t.abs().mean().item(),
            "shape": list(t.shape),
            "zeros_pct": (t.abs() < 1e-8).float().mean().item() * 100,
            "nan_count": torch.isnan(t).sum().item(),
            "inf_count": torch.isinf(t).sum().item(),
            "msg": extra_msg,
        }

    def print_all(self):
        if not self.data:
            return
        print("\n +==========================================================")
        print(" | MODEL INTERNAL DIAGNOSTICS")
        print(" +==========================================================")
        for key, info in self.data.items():
            if "msg" in info and info.get("msg") == "None":
                print(f" | {key:45s}: None")
            elif "value" in info:
                print(f" | {key:45s}: {info['value']:.6f} {info['msg']}")
            else:
                alert = ""
                if info.get("nan_count", 0) > 0:
                    alert += " WARNINGNaN!"
                if info.get("inf_count", 0) > 0:
                    alert += " WARNINGInf!"
                if info.get("zeros_pct", 0) > 95:
                    alert += " MOSTLY_ZERO"
                if info.get("std", 1) < 1e-7:
                    alert += " FROZEN"
                shape_str = str(info.get("shape", "?"))
                print(
                    f" | {key:45s}: mean={info['mean']:+.6f} std={info['std']:.6f} "
                    f"[{info['min']:+.4f}, {info['max']:+.4f}] "
                    f"z%={info['zeros_pct']:.0f}% {shape_str}{alert} {info['msg']}"
                )
        print(" +==========================================================\n")


DIAG = DiagnosticCollector()


class FrequencyMLP(nn.Module):
    def __init__(self, channel, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = channel
        self.w_r = nn.Linear(channel, hidden_dim)
        self.w_i = nn.Linear(channel, hidden_dim)
        nn.init.normal_(self.w_r.weight, mean=0, std=0.1)
        nn.init.normal_(self.w_i.weight, mean=0, std=0.1)
        nn.init.constant_(self.w_r.bias, 0.0)
        nn.init.constant_(self.w_i.bias, 0.0)
        self.mod_bias = nn.Parameter(torch.tensor(0.0))

    def _mod_relu(self, z):
        magnitude = z.abs() + 1e-8
        phase = z / magnitude
        activated_mag = F.leaky_relu(magnitude + self.mod_bias, negative_slope=0.1)
        return activated_mag * phase

    def forward(self, x_fft):
        r = x_fft.real
        i = x_fft.imag
        real_part = self.w_r(r) - self.w_i(i)
        imag_part = self.w_i(r) + self.w_r(i)
        return self._mod_relu(torch.complex(real_part, imag_part))


class TextFreqModulator(nn.Module):
    def __init__(self, channel, max_freq_len, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = channel
        self.freq_pos_emb = nn.Parameter(torch.randn(1, max_freq_len, channel) * 0.02)
        self.mod_real = nn.Sequential(
            nn.Linear(channel, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channel),
        )
        self.mod_imag = nn.Sequential(
            nn.Linear(channel, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channel),
        )

        nn.init.zeros_(self.mod_real[-1].weight)
        nn.init.zeros_(self.mod_real[-1].bias)
        nn.init.zeros_(self.mod_imag[-1].weight)
        nn.init.zeros_(self.mod_imag[-1].bias)

    def forward(self, text_feat, current_freq_len):
        text_expanded = text_feat.expand(-1, current_freq_len, -1)
        pos_emb = self.freq_pos_emb[:, :current_freq_len, :]
        text_w_pos = text_expanded + pos_emb
        return torch.complex(self.mod_real(text_w_pos), self.mod_imag(text_w_pos))


class MTPSpectrumFusion(nn.Module):

    def __init__(self, channel, seq_len):
        super().__init__()
        self.channel = channel
        self.seq_len = seq_len
        self.seq_fmlp = FrequencyMLP(channel)
        max_freq_len = seq_len // 2 + 1
        self.text_modulator = TextFreqModulator(channel, max_freq_len)
        self.register_buffer("hamming_window", torch.hamming_window(seq_len))
        self.fusion_project = nn.Linear(channel, channel)
        nn.init.xavier_uniform_(self.fusion_project.weight)
        nn.init.constant_(self.fusion_project.bias, 0.0)

    def forward(self, x_seq, x_text):
        _, s, _ = x_seq.shape
        orig_dtype = x_seq.dtype
        with torch.amp.autocast('cuda', enabled=False):
            x_f32 = x_seq.float()
            x_fft = torch.fft.rfft(x_f32, dim=1, norm="ortho")
            freq_len = x_fft.shape[1]
            window = self.hamming_window[:freq_len].view(1, -1, 1)
            spec_seq = self.seq_fmlp(x_fft * window)
            text_mod = self.text_modulator(x_text.float(), current_freq_len=freq_len)
            x_out = torch.fft.irfft(spec_seq * (1.0 + text_mod), n=s, dim=1, norm="ortho")
        return self.fusion_project(x_out.to(orig_dtype))


class CrossVariableMixer(nn.Module):

    def __init__(self, num_nodes, channel, dropout=0.1):
        super().__init__()
        if num_nodes > 100:
            hidden_dim = min(128, num_nodes // 2)
        else:
            hidden_dim = num_nodes * 2
        self.var_mixer = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_nodes),
        )
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, B, N):
        P, C = x.shape[1], x.shape[2]
        x_4d = x.reshape(B, N, P, C).permute(0, 2, 3, 1)
        x_mixed = self.var_mixer(x_4d).permute(0, 3, 1, 2)
        g = torch.sigmoid(self.gate)
        x_base = x.reshape(B, N, P, C)
        return (x_base + g * (x_mixed - x_base)).reshape(B * N, P, C)


class EnhancedFlattenHead(nn.Module):
    def __init__(self, in_features, pred_len, hidden_ratio=2, dropout=0.1, max_hidden=192):
        super().__init__()
        hidden = min(in_features // hidden_ratio, max_hidden)
        self.proj1 = nn.Linear(in_features, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj2 = nn.Linear(hidden, pred_len)
        self.skip = nn.Linear(in_features, pred_len)

    def forward(self, x):
        return self.proj2(self.drop(self.act(self.norm(self.proj1(x))))) + self.skip(x)


class MultiScaleTrend(nn.Module):
    def __init__(self, seq_len, pred_len, kernels=None, is_large_data=False):
        super().__init__()
        if kernels is not None:
            self.kernels = kernels
        elif seq_len <= 96:
            self.kernels = [3, 7, 15]
        elif seq_len <= 192:
            self.kernels = [3, 7, 25, 49]
        elif seq_len <= 336:
            self.kernels = [7, 25, 49, 97]
        else:
            self.kernels = [25, 49, 97, 169] if is_large_data else [7, 49, 97, 145]

        self.kernels = [k for k in self.kernels if k <= seq_len]
        if len(self.kernels) == 0:
            self.kernels = [3]

        self.trend_linears = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in self.kernels])
        self.weights = nn.Parameter(torch.ones(len(self.kernels)) / len(self.kernels))
        print(f" >>> [MultiScaleTrend] seq_len={seq_len}, kernels={self.kernels}")

    def _moving_avg(self, x, kernel_size):
        pad = kernel_size // 2
        x_pad = F.pad(x.unsqueeze(1), (pad, pad), mode='replicate').squeeze(1)
        kernel = torch.ones(kernel_size, device=x.device) / kernel_size
        x_3d = x_pad.unsqueeze(1)
        return F.conv1d(x_3d, kernel.view(1, 1, -1)).squeeze(1)

    def forward(self, x_flat):
        w = F.softmax(self.weights, dim=0)
        trend = 0
        for i, k in enumerate(self.kernels):
            trend = trend + w[i] * self.trend_linears[i](self._moving_avg(x_flat, k))
        return trend


class SimpleTrend(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)
        print(f" >>> [Trend] SimpleTrend: seq_len={seq_len} -> pred_len={pred_len}")

    def forward(self, x_flat):
        return self.linear(x_flat)


class ZeroTrend(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len
        print(" >>> [Trend] ZeroTrend: trend branch disabled")

    def forward(self, x_flat):
        return x_flat.new_zeros(x_flat.shape[0], self.pred_len)


class AdaptiveFusionGate(nn.Module):
    def __init__(self, pred_len, seq_len, device="cuda:0", anchor_val=None):
        super().__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.pred_ratio = pred_len / seq_len
        if anchor_val is None:
            anchor_val = -2.0 if self.pred_ratio <= 2.0 else (1.0 if self.pred_ratio <= 4.0 else 5.0)
        self.register_buffer("anchor", torch.tensor(float(anchor_val)))
        print(f" >>> [FusionGate] anchor_val={anchor_val} (pred_ratio={self.pred_ratio:.2f})")
        self.delta = nn.Parameter(torch.tensor(0.0))
        self.sample_net = nn.Linear(2, 1)
        nn.init.normal_(self.sample_net.weight, mean=0, std=0.01)
        nn.init.constant_(self.sample_net.bias, 0.0)

    def forward(self, x_flat):
        local_std = x_flat.std(dim=1, keepdim=True)
        grad_mag = torch.diff(x_flat, dim=1).abs().mean(dim=1, keepdim=True)
        sample_adjust = torch.tanh(self.sample_net(torch.cat([local_std, grad_mag], dim=1)))
        return torch.sigmoid(self.anchor + self.delta + sample_adjust), self.delta.pow(2)


class SmoothAffineCFM(nn.Module):

    def __init__(self, pred_len, seq_len, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self._stage = 'warmup'

        self.context_proj = nn.Linear(seq_len, pred_len)
        self.down_len = max(pred_len // 4, 12)
        self.compress_net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(self.down_len),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.compress_net[-1].weight)
        nn.init.zeros_(self.compress_net[-1].bias)

        self.beta_refine = nn.Sequential(
            nn.Linear(pred_len * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )
        nn.init.zeros_(self.beta_refine[-1].weight)
        nn.init.zeros_(self.beta_refine[-1].bias)
        self.beta_mix = nn.Parameter(torch.tensor(-3.0))

    def set_stage(self, stage):
        self._stage = stage

    def forward(self, mu_raw, log_beta, trend, x_history, return_details=False):
        B, T, N = mu_raw.shape
        base_beta = torch.exp(log_beta)

        if self._stage == 'warmup':
            correction_l1 = torch.tensor(0.0, device=mu_raw.device)
            if return_details:
                details = {
                    "cp_alpha": torch.ones_like(mu_raw),
                    "cp_gamma": torch.zeros_like(mu_raw),
                    "cp_beta_base": base_beta,
                    "cp_beta_refined": base_beta,
                    "cp_beta_adjustment": torch.zeros_like(mu_raw),
                    "cp_mu_adjustment": torch.zeros_like(mu_raw),
                }
                return mu_raw, base_beta, correction_l1, details
            return mu_raw, base_beta, correction_l1

        lb_flat = log_beta.permute(0, 2, 1).reshape(B * N, T)
        mu_flat = mu_raw.permute(0, 2, 1).reshape(B * N, T)
        ctx_flat = self.context_proj(x_history)
        corr_in = torch.stack([mu_flat, lb_flat, ctx_flat], dim=1)

        compressed_params = self.compress_net(corr_in)
        smooth_params = F.interpolate(compressed_params, size=T, mode='linear', align_corners=True)

        alpha = torch.tanh(smooth_params[:, 0, :]) * _CORRCAP + 1.0
        mu_scale = mu_flat.abs().mean(dim=1, keepdim=True).detach() + 1e-8
        gamma = torch.tanh(smooth_params[:, 1, :]) * _CORRCAP * mu_scale

        alpha = alpha.view(B, N, T).permute(0, 2, 1)
        gamma = gamma.view(B, N, T).permute(0, 2, 1)
        mu_corrected = mu_raw * alpha + gamma

        residual = (mu_raw - trend).abs().permute(0, 2, 1).reshape(B * N, T)
        beta_in = torch.cat([lb_flat, residual], dim=1)
        beta_adjustment = self.beta_refine(beta_in)
        mix = torch.sigmoid(self.beta_mix)
        log_beta_out = lb_flat + mix * beta_adjustment
        beta = torch.exp(log_beta_out.reshape(B, N, T).permute(0, 2, 1))

        correction_l1 = (mu_corrected - mu_raw).abs().mean()

        DIAG.log("CP.Alpha_Mean", alpha.mean().item(), "average affine stretch")
        DIAG.log("CP.Alpha_Std", alpha.std().item(), "affine stretch std")
        corr_ratio = (correction_l1 / (mu_raw.abs().mean() + 1e-8)).item()
        DIAG.log("CP.Correction_Pct", corr_ratio * 100, "% affine intervention")

        if return_details:
            beta_adjustment = beta_adjustment.reshape(B, N, T).permute(0, 2, 1)
            details = {
                "cp_alpha": alpha,
                "cp_gamma": gamma,
                "cp_beta_base": base_beta,
                "cp_beta_refined": beta,
                "cp_beta_adjustment": beta_adjustment,
                "cp_mu_adjustment": mu_corrected - mu_raw,
            }
            return mu_corrected, beta, correction_l1, details

        return mu_corrected, beta, correction_l1


class AdaptiveImportanceMask(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.msa = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.linear_alpha = nn.Linear(d_model, 1)
        self.aim_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, Ei):
        grad = Ei[:, 1:, :] - Ei[:, :-1, :]
        grad_norm = torch.norm(grad, p=2, dim=-1)
        grad_norm = F.pad(grad_norm, (1, 0), mode='constant', value=0)
        grad_norm = grad_norm / (grad_norm.max(dim=1, keepdim=True)[0] + 1e-8)
        attn_out, _ = self.msa(Ei, Ei, Ei)
        attn_score = self.linear_alpha(attn_out).squeeze(-1)
        combined_score = grad_norm * attn_score
        alpha = torch.sigmoid(combined_score)
        g = torch.sigmoid(self.aim_gate)
        return (1.0 - g) + g * alpha


class PredLenAwarePatchWeights(nn.Module):
    def __init__(self, patch_len_list, seq_len, pred_len):
        super().__init__()
        self.num_scales = len(patch_len_list)
        ratio = pred_len / seq_len

        if self.num_scales == 1:
            prior = torch.tensor([1.0])
        elif self.num_scales == 2:
            if ratio <= 1.0:
                prior = torch.tensor([0.65, 0.35])
            elif ratio <= 2.0:
                prior = torch.tensor([0.40, 0.60])
            else:
                prior = torch.tensor([0.25, 0.75])
        elif self.num_scales == 3:
            if ratio <= 1.0:
                prior = torch.tensor([0.55, 0.30, 0.15])
            elif ratio <= 2.0:
                prior = torch.tensor([0.25, 0.45, 0.30])
            elif ratio <= 3.5:
                prior = torch.tensor([0.15, 0.35, 0.50])
            else:
                prior = torch.tensor([0.10, 0.25, 0.65])
        else:
            if ratio <= 1.0:
                prior = torch.tensor([0.45, 0.30, 0.15, 0.10])
            elif ratio <= 2.0:
                prior = torch.tensor([0.25, 0.35, 0.25, 0.15])
            elif ratio <= 3.5:
                prior = torch.tensor([0.10, 0.25, 0.35, 0.30])
            else:
                prior = torch.tensor([0.05, 0.15, 0.35, 0.45])

        prior = prior[:self.num_scales]
        prior = prior / prior.sum()
        self.scale_logits = nn.Parameter(torch.log(prior + 1e-8))
        self.register_buffer('prior', prior)
        print(
            f" >>> [Mod B] PredLenAwarePatchWeights: ratio={ratio:.2f}, "
            f"prior={[f'{p:.2f}' for p in prior.tolist()]}"
        )

    def forward(self):
        return F.softmax(self.scale_logits, dim=0)

    def get_diagnostics(self):
        current = self.forward().detach()
        return {
            'prior': self.prior.tolist(),
            'current': current.tolist(),
            'drift': (current - self.prior).abs().max().item(),
        }


_is_large_dataset = is_large_dataset


class TimeCMA(nn.Module):
    def __init__(
        self,
        device="cuda:0",
        channel=48,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.3,
        d_llm=768,
        e_layer=2,
        d_layer=1,
        d_ff=512,
        head=4,
        patch_len_list=[8, 16, 32, 48],
        stride=8,
        data_path='',
        use_var_mixer=True,
        trend_mode='multiscale',
        anchor_val=None,
    ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channel = channel
        self.d_llm = d_llm
        self.num_nodes = num_nodes
        self.use_var_mixer = use_var_mixer
        self.trend_mode = trend_mode

        print(f" >>> [VarMixer] use_var_mixer={use_var_mixer}")
        print(f" >>> [Trend] trend_mode={trend_mode}")

        self.is_large_data = is_large_dataset(data_path)
        print(f" >>> [Unified] data_path='{data_path}', is_large_data={self.is_large_data}")

        if seq_len <= 48:
            self.patch_len_list = [4, 8, 12]
            self.stride = 4
        else:
            self.patch_len_list = [p for p in patch_len_list if p <= seq_len]
            self.stride = stride
        if len(self.patch_len_list) == 0:
            self.patch_len_list = [min(seq_len // 2, 16)]

        if channel < head * 2:
            head = max(1, channel // 8)
        while channel % head != 0 and head > 1:
            head -= 1
        self.head = head

        print(f" >>> [TimeCMA] channel={channel}, head={head}, dropout={dropout_n}")
        print(f" >>> [TimeCMA] patches={self.patch_len_list}, stride={self.stride}")

        use_affine = True if self.is_large_data else False
        self.normalize_layers = Normalize(num_nodes, affine=use_affine).to(self.device)
        print(f" >>> [Normalize] affine={use_affine}")

        self.mtp_fusions = nn.ModuleList([
            MTPSpectrumFusion(channel, int((seq_len - p) / self.stride) + 2).to(self.device)
            for p in self.patch_len_list
        ])
        self.adaptive_gate = AdaptiveFusionGate(
            pred_len, seq_len, device, anchor_val=anchor_val
        ).to(self.device)

        self.patch_embeddings = nn.ModuleList()
        self.flatten_heads = nn.ModuleList()
        self.correction_heads = nn.ModuleList()
        self.beta_heads = nn.ModuleList()
        self.var_mixers = nn.ModuleList()
        self.aim_layers = nn.ModuleList()

        for p_len in self.patch_len_list:
            num_patches = int((seq_len - p_len) / self.stride) + 2
            self.patch_embeddings.append(nn.Linear(p_len, channel).to(self.device))
            head_nf = channel * num_patches

            self.flatten_heads.append(
                EnhancedFlattenHead(head_nf, pred_len, dropout=dropout_n, max_hidden=192).to(self.device)
            )
            corr_hidden = min(128, head_nf // 4)
            self.correction_heads.append(nn.Sequential(
                nn.Linear(head_nf, corr_hidden),
                nn.GELU(),
                nn.Dropout(dropout_n),
                nn.Linear(corr_hidden, pred_len),
            ).to(self.device))

            beta_hidden = min(128, head_nf // 4)
            beta_head = nn.Sequential(
                nn.Linear(head_nf, beta_hidden),
                nn.LayerNorm(beta_hidden),
                nn.GELU(),
                nn.Dropout(dropout_n),
                nn.Linear(beta_hidden, pred_len),
            ).to(self.device)
            nn.init.normal_(beta_head[-1].weight, mean=0, std=0.01)
            nn.init.constant_(beta_head[-1].bias, 0.0)
            self.beta_heads.append(beta_head)

            self.var_mixers.append(
                CrossVariableMixer(num_nodes, channel, dropout=dropout_n).to(self.device)
            )
            self.aim_layers.append(
                AdaptiveImportanceMask(d_model=channel, num_heads=self.head).to(self.device)
            )

        self.patch_weight_module = PredLenAwarePatchWeights(
            self.patch_len_list, seq_len, pred_len
        ).to(self.device)

        self.dropout = nn.Dropout(dropout_n)
        self.text_proj = nn.Sequential(
            nn.Linear(self.d_llm, channel),
            nn.LayerNorm(channel),
            nn.LeakyReLU(),
        ).to(self.device)

        actual_d_ff = min(d_ff, channel * 8)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channel,
            nhead=self.head,
            dim_feedforward=actual_d_ff,
            dropout=dropout_n,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layer).to(self.device)

        if trend_mode == 'off':
            self.trend_model = ZeroTrend(pred_len).to(self.device)
        elif trend_mode == 'simple':
            self.trend_model = SimpleTrend(seq_len, pred_len).to(self.device)
        else:
            self.trend_model = MultiScaleTrend(
                seq_len, pred_len, is_large_data=self.is_large_data
            ).to(self.device)

        self.cfm = SmoothAffineCFM(
            pred_len=pred_len, seq_len=seq_len, hidden_dim=64, dropout=0.1
        ).to(self.device)

    def set_stage(self, stage):
        self.cfm.set_stage(stage)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, x_mark, embeddings, return_components=False):
        B, S, N = x.shape

        x_input = x
        seq_mean = x_input.mean(dim=1, keepdim=True).detach()
        seq_var = torch.var(x_input, dim=1, keepdim=True, unbiased=False) + 1e-5
        seq_std = torch.sqrt(seq_var).detach()

        x = self.normalize_layers(x_input, "norm")
        x_ci = x.permute(0, 2, 1).reshape(B * N, S, 1)

        trend = self.trend_model(x_ci.squeeze(-1)).reshape(B, N, self.pred_len).permute(0, 2, 1)

        text_feat = None
        if embeddings is not None:
            if embeddings.dim() == 3:
                if embeddings.shape[1] == self.d_llm:
                    emb_input = embeddings.permute(0, 2, 1)
                else:
                    emb_input = embeddings.mean(dim=1).unsqueeze(1).repeat(1, N, 1)
            else:
                emb_input = embeddings.unsqueeze(1).repeat(1, N, 1)
            text_feat = self.text_proj(emb_input).reshape(B * N, 1, self.channel)

        seasonal_accum, correction_accum, log_beta_accum = 0, 0, 0
        weights = self.patch_weight_module()

        x_pad = F.pad(x_ci.squeeze(-1), (0, self.stride))
        x_flat = x_ci.squeeze(-1)
        gate, gate_reg = self.adaptive_gate(x_flat)
        last_enc_out = None

        for i, (p_len, fusion_layer) in enumerate(zip(self.patch_len_list, self.mtp_fusions)):
            patches = x_pad.unfold(dimension=1, size=p_len, step=self.stride)
            enc_in = self.patch_embeddings[i](patches)

            if self.use_var_mixer:
                enc_in = self.var_mixers[i](enc_in, B, N)

            aim_alpha = self.aim_layers[i](enc_in)
            enc_in = enc_in * aim_alpha.unsqueeze(-1)

            enc_out = self.encoder(enc_in)

            if text_feat is not None:
                fused_out = fusion_layer(enc_out, x_text=text_feat)
                enc_norm = enc_out.detach().norm(dim=-1, keepdim=True) + 1e-8
                fused_norm = fused_out.detach().norm(dim=-1, keepdim=True) + 1e-8
                scale_ratio = torch.clamp(enc_norm / fused_norm, min=0.1, max=10.0)
                enc_out = enc_out + gate.unsqueeze(1) * scale_ratio * fused_out

            last_enc_out = enc_out
            enc_out_flat = enc_out.reshape(enc_out.shape[0], -1)
            seasonal_i = self.flatten_heads[i](enc_out_flat).reshape(B, N, self.pred_len).permute(0, 2, 1)
            correction_i = self.correction_heads[i](enc_out_flat).reshape(B, N, self.pred_len).permute(0, 2, 1)
            log_beta_i = self.beta_heads[i](enc_out_flat).reshape(B, N, self.pred_len).permute(0, 2, 1)

            seasonal_accum += seasonal_i * weights[i]
            correction_accum += correction_i * weights[i]
            log_beta_accum += log_beta_i * weights[i]

        mu_pre_gap = trend + seasonal_accum + correction_accum
        gap = x[:, -1:, :] - mu_pre_gap[:, 0:1, :]
        mu_gap = mu_pre_gap + gap * 0.5

        if return_components:
            mu_corrected, beta, correction_l1, cp_details = self.cfm(
                mu_gap, log_beta_accum, trend, x_history=x_flat, return_details=True
            )
        else:
            mu_corrected, beta, correction_l1 = self.cfm(
                mu_gap, log_beta_accum, trend, x_history=x_flat
            )
            cp_details = None

        mu_denorm = self.normalize_layers(mu_corrected, "denorm")
        beta_out = beta * seq_std

        gate_mean = gate.mean().unsqueeze(0)
        gate_reg_out = gate_reg.unsqueeze(0) if gate_reg.dim() == 0 else gate_reg
        correction_l1_out = correction_l1.unsqueeze(0) if correction_l1.dim() == 0 else correction_l1

        with torch.no_grad():
            for idx in range(min(2, len(self.aim_layers))):
                g_val = torch.sigmoid(self.aim_layers[idx].aim_gate).item()
                DIAG.log(f"ModA.AIM_gate_{idx}", g_val, f"AIM[{idx}] gate")
        pw_diag = self.patch_weight_module.get_diagnostics()
        DIAG.log("ModB.PatchWeight_drift", pw_diag['drift'], "deviation from prior")

        if not return_components:
            return mu_denorm, beta_out, gate_mean, gate_reg_out, correction_l1_out

        if last_enc_out is not None:
            try:
                ts_embeds_out = last_enc_out.reshape(B, N, last_enc_out.shape[1], last_enc_out.shape[2]).mean(dim=(1, 2)).detach()
            except RuntimeError:
                ts_embeds_out = last_enc_out.mean(dim=1).detach()
        else:
            ts_embeds_out = torch.zeros(B, self.channel, device=self.device)

        if text_feat is not None:
            prompt_embeds_out = text_feat.reshape(B, N, 1, self.channel).mean(dim=(1, 2)).detach()
        else:
            prompt_embeds_out = torch.zeros(B, self.channel, device=self.device)

        if embeddings is not None:
            if embeddings.dim() == 2:
                retrieval_embeds_out = embeddings.detach()
            else:
                retrieval_embeds_out = embeddings.mean(dim=1).detach()
        else:
            retrieval_embeds_out = torch.zeros(B, self.d_llm, device=self.device)
        forecast_embeds_out = (
            correction_accum.mean(dim=1).detach()
            if isinstance(correction_accum, torch.Tensor) else torch.zeros(B, N, device=self.device)
        )

        explain = {
            "x_history": x_input.detach(),
            "x_norm": x.detach(),
            "trend": trend.detach(),
            "seasonal": seasonal_accum.detach(),
            "patch_correction": correction_accum.detach(),
            "mu_pre_gap": mu_pre_gap.detach(),
            "gap": gap.detach(),
            "mu_gap": mu_gap.detach(),
            "mu_pre_gap_denorm": self.normalize_layers(mu_pre_gap, "denorm").detach(),
            "mu_gap_denorm": self.normalize_layers(mu_gap, "denorm").detach(),
            "mu_final_denorm": mu_denorm.detach(),
            "beta_denorm": beta_out.detach(),
            "seq_mean": seq_mean.detach(),
            "seq_std": seq_std.detach(),
            "patch_weights": weights.detach(),
            "ts_embeds": ts_embeds_out,
            "prompt_embeds": prompt_embeds_out,
            "retrieval_embeds": retrieval_embeds_out,
            "forecast_embeds": forecast_embeds_out,
        }
        if cp_details is not None:
            for k, v in cp_details.items():
                explain[k] = v.detach()

        return mu_denorm, beta_out, gate_mean, gate_reg_out, correction_l1_out, explain

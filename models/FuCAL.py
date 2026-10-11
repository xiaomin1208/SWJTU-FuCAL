import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.StandardNorm import Normalize
from utils.common import is_large_dataset

from models.TimeCMA import (
    DiagnosticCollector, DIAG,
    FrequencyMLP, TextFreqModulator, MTPSpectrumFusion,
    CrossVariableMixer, EnhancedFlattenHead,
    MultiScaleTrend, AdaptiveFusionGate, SmoothAffineCFM,
    AdaptiveImportanceMask, PredLenAwarePatchWeights,
    SimpleTrend, ZeroTrend,
)

try:
    from models.semantic_auto_gate import (
        DEFAULT_GATE_INIT_BIAS, GATE_FEATURE_NAMES, NUM_GATE_FEATURES,
        SemanticAutoDecayGate, SemanticAutoGate, build_semantic_gate_features,
    )
except ImportError:
    from semantic_auto_gate import (
        DEFAULT_GATE_INIT_BIAS, GATE_FEATURE_NAMES, NUM_GATE_FEATURES,
        SemanticAutoDecayGate, SemanticAutoGate, build_semantic_gate_features,
    )

_is_large_dataset = is_large_dataset

VALID_SEMANTIC_GATE_MODES = ('global', 'auto', 'signed_auto', 'auto_decay')


def build_descriptor_mlp(d_in, channel, dropout=0.1):
    return nn.Sequential(
        nn.Linear(d_in, 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(128, channel),
        nn.LayerNorm(channel),
        nn.LeakyReLU(),
    )


class StructuredTokenTransformer(nn.Module):

    def __init__(self, d_in, channel, d_tok=128, n_head=4, d_ff=512,
                 n_layer=2, dropout=0.1):
        super().__init__()
        self.d_in = d_in
        self.token_embed = nn.Linear(1, d_tok)
        self.pos_embed = nn.Parameter(torch.zeros(1, d_in, d_tok))
        nn.init.normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_tok, nhead=n_head, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.out_proj = nn.Linear(d_tok, channel)

    def forward(self, x):
        lead = x.shape[:-1]
        t = x.reshape(-1, self.d_in, 1)
        t = self.token_embed(t) + self.pos_embed
        t = self.encoder(t)
        t = t.mean(dim=1)
        return self.out_proj(t).reshape(*lead, -1)


class SemanticDCTProbe(nn.Module):

    def __init__(self, d_in, pred_len, rank=6, input_mode='layernorm'):
        super().__init__()
        self.d_in = int(d_in)
        self.pred_len = int(pred_len)
        self.rank = max(1, min(int(rank), self.pred_len))
        self.input_mode = str(input_mode)
        if self.input_mode not in ('layernorm', 'raw', 'norm_preserving'):
            raise ValueError(
                "SemanticDCTProbe input_mode must be 'layernorm', 'raw', or "
                f"'norm_preserving', got {input_mode!r}"
            )
        t = torch.arange(self.pred_len, dtype=torch.float32) + 0.5
        k = torch.arange(self.rank, dtype=torch.float32).unsqueeze(1)
        basis = torch.cos(torch.pi * k * t.unsqueeze(0) / self.pred_len)
        self.register_buffer('basis', basis, persistent=True)
        self.norm = nn.LayerNorm(self.d_in, elementwise_affine=False)
        coefficient_width = (
            2 * self.d_in + 1
            if self.input_mode == 'norm_preserving'
            else self.d_in
        )
        self.coefficients = nn.Linear(coefficient_width, self.rank)
        nn.init.zeros_(self.coefficients.weight)
        nn.init.zeros_(self.coefficients.bias)

    def forward(self, x):
        if self.input_mode == 'layernorm':
            features = self.norm(x)
        elif self.input_mode == 'raw':
            features = x
        else:
            magnitude = torch.log1p(torch.linalg.vector_norm(
                x, dim=-1, keepdim=True
            ))
            features = torch.cat((x, self.norm(x), magnitude), dim=-1)
        return self.coefficients(features) @ self.basis


class SegmentedSemanticDCTProbe(nn.Module):

    def __init__(self, d_in, pred_len, rank=6, input_mode='layernorm',
                 mode='segmented', n_segments=3):
        super().__init__()
        self.d_in = int(d_in)
        self.pred_len = int(pred_len)
        self.rank = max(1, int(rank))
        self.input_mode = str(input_mode)
        self.mode = str(mode)
        self.n_segments = max(1, min(int(n_segments), self.pred_len))
        if self.input_mode not in ('layernorm', 'raw', 'norm_preserving'):
            raise ValueError(f'unknown semantic probe mode {input_mode!r}')
        if self.mode not in ('segmented', 'multiscale'):
            raise ValueError(f'unknown horizon probe mode {mode!r}')
        self.norm = nn.LayerNorm(self.d_in, elementwise_affine=False)
        feature_width = (
            2 * self.d_in + 1
            if self.input_mode == 'norm_preserving'
            else self.d_in
        )
        boundaries = torch.linspace(
            0, self.pred_len, self.n_segments + 1, dtype=torch.float64
        ).round().to(torch.long).tolist()
        self.segment_bounds = tuple(
            (int(boundaries[i]), int(boundaries[i + 1]))
            for i in range(self.n_segments)
        )
        self.segment_ranks = []
        for index, (start, end) in enumerate(self.segment_bounds):
            length = end - start
            local_rank = max(1, min(self.rank, length))
            self.segment_ranks.append(local_rank)
            t = torch.arange(length, dtype=torch.float32) + 0.5
            k = torch.arange(local_rank, dtype=torch.float32).unsqueeze(1)
            basis = torch.cos(torch.pi * k * t.unsqueeze(0) / length)
            self.register_buffer(f'segment_basis_{index}', basis, persistent=True)
        self.global_rank = (
            max(1, min(self.rank, self.pred_len))
            if self.mode == 'multiscale'
            else 0
        )
        if self.global_rank:
            t = torch.arange(self.pred_len, dtype=torch.float32) + 0.5
            k = torch.arange(self.global_rank, dtype=torch.float32).unsqueeze(1)
            self.register_buffer(
                'global_basis',
                torch.cos(torch.pi * k * t.unsqueeze(0) / self.pred_len),
                persistent=True,
            )
        coefficient_count = sum(self.segment_ranks) + self.global_rank
        self.coefficients = nn.Linear(feature_width, coefficient_count)
        nn.init.zeros_(self.coefficients.weight)
        nn.init.zeros_(self.coefficients.bias)

    def _features(self, x):
        if self.input_mode == 'layernorm':
            return self.norm(x)
        if self.input_mode == 'raw':
            return x
        magnitude = torch.log1p(torch.linalg.vector_norm(
            x, dim=-1, keepdim=True
        ))
        return torch.cat((x, self.norm(x), magnitude), dim=-1)

    def forward(self, x):
        coefficients = self.coefficients(self._features(x))
        output = coefficients.new_zeros(*coefficients.shape[:-1], self.pred_len)
        cursor = 0
        if self.global_rank:
            global_coefficients = coefficients[..., :self.global_rank]
            output = output + global_coefficients @ self.global_basis
            cursor = self.global_rank
        for index, ((start, end), local_rank) in enumerate(zip(
                self.segment_bounds, self.segment_ranks)):
            local = coefficients[..., cursor:cursor + local_rank]
            output[..., start:end] = (
                output[..., start:end]
                + local @ getattr(self, f'segment_basis_{index}')
            )
            cursor += local_rank
        return output


class FrequencyConditionedTokenDCTProbe(nn.Module):

    def __init__(self, raw_width, num_tokens, pred_len, rank=6,
                 encoder='llm', heads=4, dropout=0.1, n_segments=3):
        super().__init__()
        self.raw_width = int(raw_width)
        self.num_tokens = int(num_tokens)
        self.pred_len = int(pred_len)
        self.rank = max(1, min(int(rank), self.pred_len))
        self.encoder_kind = str(encoder)
        self.n_segments = max(1, min(int(n_segments), self.pred_len))
        if self.num_tokens <= 1 or self.raw_width % self.num_tokens:
            raise ValueError(
                'frequency-conditioned token probe requires raw_width '
                f'divisible by num_tokens>1, got {raw_width}/{num_tokens}'
            )
        self.token_width = self.raw_width // self.num_tokens
        feature_width = self.token_width + 1
        self.token_features = NormPreservingTokenFeatures(self.token_width)
        if self.encoder_kind == 'llm':
            self.token_encoder = nn.Identity()
        elif self.encoder_kind == 'mlp':
            self.token_encoder = nn.Sequential(
                nn.Linear(feature_width, feature_width), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(feature_width, feature_width),
            )
        elif self.encoder_kind == 'sttf':
            nhead = min(int(heads), feature_width)
            while feature_width % nhead and nhead > 1:
                nhead -= 1
            layer = nn.TransformerEncoderLayer(
                d_model=feature_width, nhead=nhead,
                dim_feedforward=4 * feature_width, dropout=dropout,
                batch_first=True, norm_first=True, activation='gelu',
            )
            self.token_encoder = nn.TransformerEncoder(layer, num_layers=2)
        else:
            raise ValueError(f'unknown token probe encoder {encoder!r}')


        self.spectral_query = nn.Sequential(
            nn.LayerNorm(6), nn.Linear(6, feature_width), nn.Tanh(),
        )
        self.token_score_bias = nn.Parameter(torch.zeros(self.num_tokens))
        self.context_norm = nn.LayerNorm(feature_width)

        boundaries = torch.linspace(
            0, self.pred_len, self.n_segments + 1, dtype=torch.float64
        ).round().to(torch.long).tolist()
        self.segment_bounds = tuple(
            (int(boundaries[i]), int(boundaries[i + 1]))
            for i in range(self.n_segments)
        )
        self.segment_ranks = []
        for index, (start, end) in enumerate(self.segment_bounds):
            length = end - start
            local_rank = max(1, min(self.rank, length))
            self.segment_ranks.append(local_rank)
            t = torch.arange(length, dtype=torch.float32) + 0.5
            k = torch.arange(local_rank, dtype=torch.float32).unsqueeze(1)
            self.register_buffer(
                f'segment_basis_{index}',
                torch.cos(torch.pi * k * t.unsqueeze(0) / length),
                persistent=True,
            )
        t = torch.arange(self.pred_len, dtype=torch.float32) + 0.5
        k = torch.arange(self.rank, dtype=torch.float32).unsqueeze(1)
        self.register_buffer(
            'global_basis',
            torch.cos(torch.pi * k * t.unsqueeze(0) / self.pred_len),
            persistent=True,
        )
        coefficient_count = self.rank + sum(self.segment_ranks)
        self.coefficients = nn.Linear(4 * feature_width, coefficient_count)
        nn.init.zeros_(self.coefficients.weight)
        nn.init.zeros_(self.coefficients.bias)

    @staticmethod
    def _spectral_features(history):
        history = history.float()
        power = torch.fft.rfft(history, dim=-1).abs().square()
        if power.shape[-1] > 1:
            power = power[..., 1:]
        total = power.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        bins = power.shape[-1]
        edges = torch.linspace(0, bins, 4, device=history.device).round().long()
        bands = []
        for index in range(3):
            start, end = int(edges[index]), int(edges[index + 1])
            if end <= start:
                bands.append(torch.zeros_like(total))
            else:
                bands.append(power[..., start:end].sum(dim=-1, keepdim=True) / total)
        probability = power / total
        entropy_scale = max(float(torch.log(torch.tensor(max(bins, 2))).item()), 1e-8)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(
            dim=-1, keepdim=True
        ) / entropy_scale
        dominant = power.argmax(dim=-1, keepdim=True).float() / max(bins - 1, 1)
        slope = (history[..., -1:] - history[..., :1]) / max(history.shape[-1] - 1, 1)
        return torch.cat((*bands, entropy, dominant, slope), dim=-1)

    def forward(self, raw, history):
        tokens = raw.reshape(-1, self.num_tokens, self.token_width)
        tokens = self.token_encoder(self.token_features(tokens))
        spectral = self._spectral_features(history).to(dtype=tokens.dtype)
        query = self.spectral_query(spectral)
        scores = (
            (tokens * query.unsqueeze(1)).sum(dim=-1)
            / (tokens.shape[-1] ** 0.5)
            + self.token_score_bias
        )
        weights = torch.softmax(scores, dim=-1)
        context = self.context_norm(
            (weights.unsqueeze(-1) * tokens).sum(dim=1)
        )
        features = torch.cat(
            (context, query, context * query, (context - query).abs()), dim=-1
        )
        coefficients = self.coefficients(features)
        output = coefficients.new_zeros(coefficients.shape[0], self.pred_len)
        cursor = 0
        output = output + coefficients[:, :self.rank] @ self.global_basis
        cursor += self.rank
        for index, ((start, end), local_rank) in enumerate(zip(
                self.segment_bounds, self.segment_ranks)):
            local = coefficients[:, cursor:cursor + local_rank]
            output[:, start:end] = (
                output[:, start:end]
                + local @ getattr(self, f'segment_basis_{index}')
            )
            cursor += local_rank
        return output


class SemanticProjection(nn.Module):

    def __init__(self, d_in, d_out, input_mode='layernorm'):
        super().__init__()
        self.d_in = int(d_in)
        self.input_mode = str(input_mode)
        if self.input_mode not in ('layernorm', 'raw', 'norm_preserving'):
            raise ValueError(f'unknown semantic projection mode {input_mode!r}')
        self.norm = nn.LayerNorm(self.d_in, elementwise_affine=False)
        projection_width = (
            2 * self.d_in + 1
            if self.input_mode == 'norm_preserving'
            else self.d_in
        )
        self.proj = nn.Linear(projection_width, d_out, bias=False)
        self.activation = nn.GELU()
        self.output_norm = nn.LayerNorm(d_out)

    def forward(self, x):
        if self.input_mode == 'layernorm':
            features = self.norm(x)
        elif self.input_mode == 'raw':
            features = x
        else:
            magnitude = torch.log1p(torch.linalg.vector_norm(
                x, dim=-1, keepdim=True
            ))
            features = torch.cat((x, self.norm(x), magnitude), dim=-1)
        return self.output_norm(self.activation(self.proj(features)))


class FixedPadProjection(nn.Module):

    def __init__(self, width, channel):
        super().__init__()
        self.width = int(width)
        self.channel = int(channel)
        self.norm = nn.LayerNorm(self.width, elementwise_affine=False)

    def forward(self, x):
        magnitude = torch.log1p(torch.linalg.vector_norm(x, dim=-1, keepdim=True))
        x = self.norm(x)
        if self.width < self.channel:
            x = torch.cat((x, magnitude), dim=-1)
            return F.pad(x, (0, self.channel - self.width - 1))
        if self.width > self.channel:
            return x[..., :self.channel]
        return x


class NormPreservingTokenFeatures(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(int(width), elementwise_affine=False)

    def forward(self, x):
        magnitude = torch.log1p(torch.linalg.vector_norm(x, dim=-1, keepdim=True))
        return torch.cat((self.norm(x), magnitude), dim=-1)


class AttributeCrossSemanticAdapter(nn.Module):

    def __init__(self, raw_width, num_tokens, channel, encoder='llm',
                 num_nodes=1, heads=4, dropout=0.1,
                 fixed_token_projection=False):
        super().__init__()
        self.raw_width = int(raw_width)
        self.num_tokens = int(num_tokens)
        if self.num_tokens <= 1 or self.raw_width % self.num_tokens:
            raise ValueError(
                f'attribute token layout requires raw_width divisible by '
                f'num_tokens>1, got {self.raw_width}/{self.num_tokens}'
            )
        self.token_width = self.raw_width // self.num_tokens
        self.encoder_kind = str(encoder)
        self.fixed_token_projection = bool(fixed_token_projection)
        if self.encoder_kind == 'llm':
            if self.fixed_token_projection:
                self.token_proj = FixedPadProjection(self.token_width, channel)
            else:
                self.token_proj = nn.Sequential(
                    nn.LayerNorm(self.token_width),
                    nn.Linear(self.token_width, channel, bias=False),
                    nn.GELU(), nn.LayerNorm(channel),
                )
            self.token_encoder = nn.Identity()
        elif self.encoder_kind == 'mlp':
            self.token_proj = nn.Sequential(
                NormPreservingTokenFeatures(self.token_width),
                nn.Linear(self.token_width + 1, 128), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(128, channel),
                nn.LayerNorm(channel),
            )
            self.token_encoder = nn.Identity()
        elif self.encoder_kind == 'sttf':
            self.token_proj = nn.Sequential(
                NormPreservingTokenFeatures(self.token_width),
                nn.Linear(self.token_width + 1, channel),
            )
            nhead = min(int(heads), channel)
            while channel % nhead and nhead > 1:
                nhead -= 1
            layer = nn.TransformerEncoderLayer(
                d_model=channel, nhead=nhead, dim_feedforward=4 * channel,
                dropout=dropout, batch_first=True, norm_first=True,
                activation='gelu',
            )
            self.token_encoder = nn.TransformerEncoder(layer, num_layers=2)
        else:
            raise ValueError(f'unknown attribute encoder: {self.encoder_kind!r}')

        nhead = min(int(heads), channel)
        while channel % nhead and nhead > 1:
            nhead -= 1
        attribute_embedding = torch.zeros(1, 1, self.num_tokens, channel)
        if self.fixed_token_projection:
            self.register_buffer(
                'attribute_embedding', attribute_embedding, persistent=False
            )
        else:
            self.attribute_embedding = nn.Parameter(attribute_embedding)


        self.register_buffer(
            'variable_embedding',
            torch.zeros(1, int(num_nodes), 1, channel),
            persistent=False,
        )
        if not self.fixed_token_projection:
            nn.init.normal_(self.attribute_embedding, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            channel, nhead, dropout=dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(channel)
        self.output_norm = nn.LayerNorm(4 * channel)

    def forward(self, raw, numeric_query, batch_size, num_nodes):
        tokens = raw.reshape(
            batch_size, num_nodes, self.num_tokens, self.token_width
        )
        tokens = self.token_proj(tokens)
        tokens = tokens + self.attribute_embedding
        tokens = tokens + self.variable_embedding[:, :num_nodes]
        tokens = self.token_encoder(tokens.reshape(
            batch_size * num_nodes, self.num_tokens, -1
        ))
        query = self.query_norm(numeric_query).unsqueeze(1)
        attended, attention = self.cross_attention(
            query, tokens, tokens, need_weights=True, average_attn_weights=True
        )
        attended = attended.squeeze(1)
        query = query.squeeze(1)
        fused = self.output_norm(torch.cat(
            (query, attended, query * attended, (query - attended).abs()), dim=-1
        ))
        return fused, attended, attention.squeeze(1)


class AttributeFeatureFiLMAdapter(nn.Module):

    def __init__(self, raw_width, num_tokens, seq_len, channel, encoder='llm',
                 num_nodes=1, heads=4, dropout=0.1, gate_init_bias=0.0):
        super().__init__()
        self.raw_width = int(raw_width)
        self.num_tokens = int(num_tokens)
        if self.num_tokens <= 1 or self.raw_width % self.num_tokens:
            raise ValueError(
                f'FiLM token layout requires raw_width divisible by num_tokens>1, '
                f'got {self.raw_width}/{self.num_tokens}'
            )
        self.token_width = self.raw_width // self.num_tokens
        self.encoder_kind = str(encoder)
        if self.encoder_kind == 'llm':
            self.token_proj = nn.Sequential(
                nn.LayerNorm(self.token_width),
                nn.Linear(self.token_width, channel, bias=False),
                nn.GELU(), nn.LayerNorm(channel),
            )
            self.token_encoder = nn.Identity()
        elif self.encoder_kind == 'mlp':
            self.token_proj = nn.Sequential(
                nn.LayerNorm(self.token_width), nn.Linear(self.token_width, 128),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(128, channel),
                nn.LayerNorm(channel),
            )
            self.token_encoder = nn.Identity()
        elif self.encoder_kind == 'sttf':
            self.token_proj = nn.Sequential(
                nn.LayerNorm(self.token_width), nn.Linear(self.token_width, channel)
            )
            nhead = min(int(heads), channel)
            while channel % nhead and nhead > 1:
                nhead -= 1
            layer = nn.TransformerEncoderLayer(
                d_model=channel, nhead=nhead, dim_feedforward=4 * channel,
                dropout=dropout, batch_first=True, norm_first=True,
                activation='gelu',
            )
            self.token_encoder = nn.TransformerEncoder(layer, num_layers=2)
        else:
            raise ValueError(f'unknown FiLM semantic encoder: {self.encoder_kind!r}')

        nhead = min(int(heads), channel)
        while channel % nhead and nhead > 1:
            nhead -= 1
        self.attribute_embedding = nn.Parameter(
            torch.zeros(1, 1, self.num_tokens, channel)
        )
        self.variable_embedding = nn.Parameter(
            torch.zeros(1, int(num_nodes), 1, channel)
        )
        nn.init.normal_(self.attribute_embedding, std=0.02)
        nn.init.normal_(self.variable_embedding, std=0.02)
        self.history_query = nn.Sequential(
            nn.LayerNorm(seq_len), nn.Linear(seq_len, channel), nn.LayerNorm(channel)
        )
        self.cross_attention = nn.MultiheadAttention(
            channel, nhead, dropout=dropout, batch_first=True
        )
        self.film = nn.Sequential(
            nn.LayerNorm(channel), nn.Linear(channel, channel), nn.GELU(),
            nn.Linear(channel, 2 * channel),
        )


        self.reliability_gate = nn.Sequential(
            nn.LayerNorm(4 * channel),
            nn.Linear(4 * channel, channel), nn.GELU(),
            nn.Linear(channel, 1),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        nn.init.zeros_(self.reliability_gate[-1].weight)
        nn.init.constant_(
            self.reliability_gate[-1].bias, float(gate_init_bias)
        )

    def forward(self, raw, history, batch_size, num_nodes):
        tokens = raw.reshape(
            batch_size, num_nodes, self.num_tokens, self.token_width
        )
        tokens = self.token_proj(tokens)
        tokens = tokens + self.attribute_embedding
        tokens = tokens + self.variable_embedding[:, :num_nodes]
        tokens = self.token_encoder(tokens.reshape(
            batch_size * num_nodes, self.num_tokens, -1
        ))
        query = self.history_query(history).unsqueeze(1)
        context, attention = self.cross_attention(
            query, tokens, tokens, need_weights=True, average_attn_weights=True
        )
        context = context.squeeze(1)
        query = query.squeeze(1)
        reliability_features = torch.cat(
            (query, context, query * context, (query - context).abs()), dim=-1
        )
        reliability = torch.sigmoid(self.reliability_gate(
            reliability_features
        ))
        gamma, beta = self.film(context).chunk(2, dim=-1)


        return (torch.tanh(gamma), beta, attention.squeeze(1),
                reliability.squeeze(-1))


class SimpleConcatFusion(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.proj = nn.Linear(channel * 2, channel)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)
        print(f" >>> [Ablation] SimpleConcatFusion: channel={channel}")

    def forward(self, x_seq, x_text):
        text_expanded = x_text.expand(-1, x_seq.shape[1], -1)
        return self.proj(torch.cat([x_seq, text_expanded], dim=-1))


class VanillaCFM(nn.Module):

    def __init__(self, pred_len, seq_len):
        super().__init__()
        self.pred_len = pred_len
        self._stage = 'warmup'
        print(" >>> [Ablation] VanillaCFM: no affine correction, no beta refinement")

    def set_stage(self, stage):
        self._stage = stage

    def forward(self, mu_raw, log_beta, trend, x_history, return_details=False):
        beta = torch.exp(log_beta)
        correction_l1 = torch.tensor(0.0, device=mu_raw.device)
        if return_details:
            details = {
                "cp_alpha": torch.ones_like(mu_raw),
                "cp_gamma": torch.zeros_like(mu_raw),
                "cp_beta_base": beta,
                "cp_beta_refined": beta,
                "cp_beta_adjustment": torch.zeros_like(mu_raw),
                "cp_mu_adjustment": torch.zeros_like(mu_raw),
            }
            return mu_raw, beta, correction_l1, details
        return mu_raw, beta, correction_l1


class CQR_CFM(nn.Module):

    def __init__(self, pred_len, seq_len, hidden_dim=64):
        super().__init__()
        self.pred_len = pred_len
        self._stage = 'warmup'
        self.q_lo_head = nn.Sequential(
            nn.Linear(seq_len, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, pred_len),
        )
        self.q_hi_head = nn.Sequential(
            nn.Linear(seq_len, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, pred_len),
        )
        nn.init.normal_(self.q_lo_head[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.q_lo_head[-1].bias, -0.5)
        nn.init.normal_(self.q_hi_head[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.q_hi_head[-1].bias, 0.5)
        print(f" >>> [Ablation] CQR_CFM: quantile-width head, hidden={hidden_dim}")

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

        q_lo_offset = self.q_lo_head(x_history).reshape(B, N, T).permute(0, 2, 1)
        q_hi_offset = self.q_hi_head(x_history).reshape(B, N, T).permute(0, 2, 1)
        half_width = F.softplus(q_hi_offset - q_lo_offset) * 0.5
        beta = half_width / 1.96 + 1e-6
        correction_l1 = torch.tensor(0.0, device=mu_raw.device)
        if return_details:
            details = {
                "cp_alpha": torch.ones_like(mu_raw),
                "cp_gamma": torch.zeros_like(mu_raw),
                "cp_beta_base": base_beta,
                "cp_beta_refined": beta,
                "cp_beta_adjustment": torch.zeros_like(mu_raw),
                "cp_mu_adjustment": torch.zeros_like(mu_raw),
            }
            return mu_raw, beta, correction_l1, details
        return mu_raw, beta, correction_l1


class EnbPI_CFM(nn.Module):

    def __init__(self, pred_len, seq_len, hidden_dim=64):
        super().__init__()
        self.pred_len = pred_len
        self._stage = 'warmup'
        self.vol_net = nn.Sequential(
            nn.Linear(seq_len, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, pred_len),
        )
        self.width_scale = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.vol_net[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.vol_net[-1].bias, 0.0)
        print(f" >>> [Ablation] EnbPI_CFM: volatility-width head, hidden={hidden_dim}")

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

        hist_diff = torch.diff(x_history, dim=1)
        hist_vol = torch.cat([hist_diff[:, :1], hist_diff], dim=1).abs()
        width_pred = self.vol_net(hist_vol)
        width_pred = F.softplus(width_pred) * torch.exp(self.width_scale)
        beta = width_pred.reshape(B, N, T).permute(0, 2, 1) / 1.96 + 1e-6
        correction_l1 = torch.tensor(0.0, device=mu_raw.device)
        if return_details:
            details = {
                "cp_alpha": torch.ones_like(mu_raw),
                "cp_gamma": torch.zeros_like(mu_raw),
                "cp_beta_base": base_beta,
                "cp_beta_refined": beta,
                "cp_beta_adjustment": torch.zeros_like(mu_raw),
                "cp_mu_adjustment": torch.zeros_like(mu_raw),
            }
            return mu_raw, beta, correction_l1, details
        return mu_raw, beta, correction_l1


class FuzzyRuleHead(nn.Module):

    def __init__(self, in_features, pred_len, n_rules, desc_dim,
                 hidden_ratio=2, dropout=0.1, max_hidden=192, init_tau=0.15):
        super().__init__()
        hidden = min(in_features // hidden_ratio, max_hidden)
        self.n_rules = int(n_rules)
        self.pred_len = int(pred_len)
        self.proj1 = nn.Linear(in_features, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj2 = nn.Linear(hidden, self.n_rules * pred_len)
        self.skip = nn.Linear(in_features, pred_len)
        self.protos = nn.Parameter(torch.randn(self.n_rules, desc_dim) * 0.05)
        inv = float(math.log(math.exp(max(init_tau, 1e-3)) - 1.0))
        self.log_tau = nn.Parameter(torch.full((1,), inv))
        self.last_resp = None

    def fuzziness(self):
        return torch.nn.functional.softplus(self.log_tau) + 1e-2

    def responsibilities(self, desc, hard=False):
        d = desc - desc.mean(dim=-1, keepdim=True)
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-6)
        p = self.protos / (self.protos.norm(dim=-1, keepdim=True) + 1e-6)
        logits = (d @ p.t().to(d.dtype)) / self.fuzziness().to(d.dtype)
        if hard:
            idx = logits.argmax(dim=-1)
            return torch.nn.functional.one_hot(idx, self.n_rules).to(logits.dtype)
        return torch.softmax(logits, dim=-1)

    def forward(self, x, desc=None, hard=False):
        h = self.drop(self.act(self.norm(self.proj1(x))))
        cons = self.proj2(h).reshape(-1, self.n_rules, self.pred_len)
        if desc is None:
            w = cons.new_full((cons.shape[0], self.n_rules), 1.0 / self.n_rules)
        else:
            w = self.responsibilities(desc, hard=hard)
        self.last_resp = w.detach()
        return (w.unsqueeze(-1) * cons).sum(dim=1) + self.skip(x)


class FuzzyAnchorAlign(nn.Module):

    def __init__(self, in_features, assets_path, use_random=False, init_tau=0.1):
        super().__init__()
        a = torch.load(assets_path, map_location="cpu", weights_only=False)
        anchors = a["anchors_random"] if use_random else a["anchors"]
        self.register_buffer("anchors", anchors)
        self.register_buffer("knots", a["knots"])
        self.n_axes = anchors.shape[1]
        self.n_levels = anchors.shape[2]
        self.proj = nn.Linear(in_features, anchors.shape[-1] * self.n_axes)
        inv = float(math.log(math.exp(max(init_tau, 1e-3)) - 1.0))
        self.log_tau = nn.Parameter(torch.full((1,), inv))

    @staticmethod
    def descriptors(x):
        w = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-5)
        t = torch.arange(w.shape[1], device=w.device, dtype=w.dtype)
        t = (t - t.mean()) / (t.std() + 1e-9)
        d = w[:, 1:] - w[:, :-1]
        tail = max(1, d.shape[1] // 4)
        slope = torch.einsum("bln,l->bn", w - w.mean(1, keepdim=True), t) / (t @ t)
        return torch.stack([
            w.mean(1), w.std(1), slope, d.std(1),
            w.amax(1) - w.amin(1), d[:, -tail:].mean(1),
        ], dim=1)

    def memberships(self, x):
        f = self.descriptors(x)
        f = f[:, :self.n_axes].permute(0, 2, 1)
        kn = self.knots[:self.n_axes].permute(1, 0, 2)
        l, c, r = kn[..., :-2], kn[..., 1:-1], kn[..., 2:]
        v = f.unsqueeze(-1)
        left = (v - l) / (c - l + 1e-6)
        right = (r - v) / (r - c + 1e-6)
        u = torch.clamp(torch.minimum(left, right), min=0.0)
        return u / (u.sum(-1, keepdim=True) + 1e-6)

    def forward(self, z, x):
        B, _, N = x.shape
        p = self.proj(z).reshape(B, N, self.n_axes, -1)
        p = p / (p.norm(dim=-1, keepdim=True) + 1e-6)
        a = self.anchors.to(p.dtype).unsqueeze(0)
        logits = torch.einsum("bnah,bnakh->bnak", p, a.expand(B, -1, -1, -1, -1))
        tau = torch.nn.functional.softplus(self.log_tau).to(p.dtype) + 1e-2
        logq = torch.log_softmax(logits / tau, dim=-1)
        u = self.memberships(x).to(p.dtype)
        return -(u * logq).sum(-1).mean()


class FuzzyPatchFusion(nn.Module):

    def __init__(self, prior, n_rules, desc_dim, init_tau=0.15):
        super().__init__()
        prior = prior.detach().reshape(-1).float()
        self.n_scales = int(prior.numel())
        self.n_rules = int(n_rules)
        self.register_buffer("prior_logit", torch.log(prior.clamp_min(1e-6)))
        self.protos = nn.Parameter(torch.randn(self.n_rules, desc_dim) * 0.05)
        self.theta = nn.Parameter(torch.zeros(self.n_rules, self.n_scales))
        inv = float(math.log(math.exp(max(init_tau, 1e-3)) - 1.0))
        self.log_tau = nn.Parameter(torch.full((1,), inv))
        self.last_fire = None

    def fuzziness(self):
        return torch.nn.functional.softplus(self.log_tau) + 1e-2

    def forward(self, desc):
        d = desc - desc.mean(dim=-1, keepdim=True)
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-6)
        p = self.protos / (self.protos.norm(dim=-1, keepdim=True) + 1e-6)
        fire = torch.softmax((d @ p.t().to(d.dtype)) / self.fuzziness().to(d.dtype),
                             dim=-1)
        self.last_fire = fire.detach()
        logits = self.prior_logit.to(d.dtype).unsqueeze(0) + fire @ self.theta.to(d.dtype)
        return torch.softmax(logits, dim=-1)


class FusionDescriptor(nn.Module):

    def __init__(self, mode, memb_dim, llm_dim, seq_len, d_out=16, momentum=0.01):
        super().__init__()
        self.mode = str(mode or "fuzzy")
        self.memb_dim = int(memb_dim)
        self.d_out = int(d_out)
        self.momentum = float(momentum)
        self.out_dim = self.memb_dim
        self.proj = None
        self.mlp = None
        if self.mode in ("fuzzy_llm", "fuzzy_llm_shuf"):
            self.register_buffer("running_mean", torch.zeros(int(llm_dim)))
            self.register_buffer("mean_ready", torch.zeros(1))
            self.proj = nn.Linear(int(llm_dim), self.d_out, bias=False)
            nn.init.normal_(self.proj.weight, std=0.02)
            self.out_dim += self.d_out
        elif self.mode == "fuzzy_mlp":
            self.mlp = nn.Sequential(
                nn.Linear(int(seq_len), 32), nn.GELU(), nn.Linear(32, self.d_out)
            )
            self.out_dim += self.d_out
        elif self.mode not in ("fuzzy", "crisp", "shuf"):
            raise ValueError("unknown fusion_desc %r" % self.mode)

    def forward(self, u_flat, llm_src, x_flat):
        d = u_flat
        if self.mode == "crisp":


            v = d.reshape(d.shape[0], -1, 15)
            d = torch.zeros_like(v).scatter_(
                -1, v.argmax(-1, keepdim=True), 1.0).reshape(d.shape[0], -1)
        elif self.mode == "shuf":


            d = d[torch.randperm(d.shape[0], device=d.device)]
        if self.proj is not None:
            if llm_src is None:
                z = d.new_zeros(d.shape[0], self.d_out)
            else:
                e = llm_src.reshape(d.shape[0], -1).to(d.dtype)
                if self.mode == "fuzzy_llm_shuf":
                    e = e[torch.randperm(e.shape[0], device=e.device)]
                if self.training:
                    with torch.no_grad():
                        m = e.detach().float().mean(0)
                        if float(self.mean_ready.item()) == 0.0:
                            self.running_mean.copy_(m)
                            self.mean_ready.fill_(1.0)
                        else:
                            self.running_mean.mul_(1.0 - self.momentum).add_(
                                m, alpha=self.momentum)
                r = e - self.running_mean.to(e.dtype).unsqueeze(0)
                r = r / (r.norm(dim=-1, keepdim=True) + 1e-6)
                z = self.proj(r)
            d = torch.cat((d, z), dim=-1)
        elif self.mlp is not None:
            z = self.mlp(x_flat.reshape(d.shape[0], -1).to(d.dtype))
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
            d = torch.cat((d, z), dim=-1)
        return d


class FuzzySoftExperts(nn.Module):

    def __init__(self, channel, n_experts=5, rank=4, memb_dim=75, crisp=False):
        super().__init__()
        self.n_experts = int(n_experts)
        self.crisp = bool(crisp)
        self.gate = nn.Linear(int(memb_dim), self.n_experts, bias=True)
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.gate.weight, std=0.02)
        self.down = nn.Parameter(torch.randn(self.n_experts, channel, rank) * 0.02)
        self.up = nn.Parameter(torch.zeros(self.n_experts, rank, channel))
        self.last_gate = None

    def forward(self, h, u_flat):
        squeeze = (h.dim() == 2)
        if squeeze:
            h = h.unsqueeze(1)
        g = torch.softmax(self.gate(u_flat.to(h.dtype)), dim=-1)
        if self.crisp:
            g = torch.zeros_like(g).scatter_(
                -1, g.argmax(-1, keepdim=True), 1.0)
        self.last_gate = g.detach()

        z = torch.einsum('mpc,ecr->mper', h, self.down.to(h.dtype))
        z = torch.einsum('mper,erc->mpec', z, self.up.to(h.dtype))
        out = h + torch.einsum('mpec,me->mpc', z, g)
        return out.squeeze(1) if squeeze else out


class FuCAL(nn.Module):
    def __init__(
        self,
        device="cuda:0",
        channel=48,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.3,
        fuzzy_fusion_rules=0,
        fusion_desc='fuzzy',
        soft_experts=0,
        soft_expert_rank=4,
        soft_expert_crisp=False,
        align_assets=None,
        align_random=False,
        fuzzy_rules=0,
        fuzzy_rules_tau=0.15,
        fuzzy_rules_crisp=False,
        d_llm=768,
        e_layer=2,
        d_layer=1,
        d_ff=512,
        head=4,
        patch_len_list=[8, 16, 32, 48],
        stride=8,
        data_path='',
        ablation='full',
        use_var_mixer=True,
        trend_mode='auto',
        anchor_val=None,
        sem_encoder='llm',
        semantic_direct=False,
        semantic_gate_floor=0.0,
        dual_semantic=False,
        semantic_gate_mode='global',
        semantic_gate_init_bias=DEFAULT_GATE_INIT_BIAS,
        semantic_gate_const=-1.0,
        semantic_residual_clip=0.0,
        semantic_interaction_mode='prompt',
        semantic_uncertainty=False,
        semantic_uncertainty_clip=0.25,
        semantic_basis_rank=0,
        semantic_probe_mode='layernorm',
        semantic_horizon_mode='global',
        legacy_d_llm=None,
        semantic_d_llm=None,
        semantic_encoder='llm',
        semantic_attribute_tokens=0,
        semantic_attribute_heads=4,
        semantic_fixed_token_projection=False,
        semantic_injection_mode='residual',
    ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channel = channel
        self.sem_encoder = sem_encoder
        self.semantic_direct = bool(semantic_direct)
        self.semantic_gate_floor = float(semantic_gate_floor)
        self.dual_semantic = bool(dual_semantic)
        self.disable_legacy_semantic = (
            bool(self.dual_semantic)
            or os.environ.get('FUCAL_DISABLE_LEGACY_SEMANTIC', '0').strip().lower()
            in ('1', 'true', 'yes', 'on')
        )
        self.semantic_use_full_prompt = (
            os.environ.get('FUCAL_SEMANTIC_USE_FULL_PROMPT', '0').strip().lower()
            in ('1', 'true', 'yes', 'on')
        )


        self.semantic_gate_mode = str(semantic_gate_mode)
        if self.semantic_gate_mode not in VALID_SEMANTIC_GATE_MODES:
            raise ValueError(
                f"semantic_gate_mode must be one of {VALID_SEMANTIC_GATE_MODES}, "
                f"got {semantic_gate_mode!r}"
            )
        self.semantic_gate_init_bias = float(semantic_gate_init_bias)
        self.semantic_gate_const = float(semantic_gate_const)
        if self.semantic_gate_const >= 0.0:
            self.semantic_gate_const = min(self.semantic_gate_const, 1.0)
            print(
                f" >>> [SemanticGateConst] output-space gate fixed at "
                f"{self.semantic_gate_const:.4f}"
            )


        self.semantic_residual_clip = float(semantic_residual_clip)


        self.semantic_interaction_mode = str(semantic_interaction_mode)
        if self.semantic_interaction_mode not in ('prompt', 'cross'):
            raise ValueError(
                "semantic_interaction_mode must be 'prompt' or 'cross', got "
                f"{semantic_interaction_mode!r}"
            )
        self.semantic_uncertainty = bool(semantic_uncertainty)
        self.semantic_uncertainty_clip = float(semantic_uncertainty_clip)
        self.semantic_basis_rank = int(semantic_basis_rank)
        if self.semantic_basis_rank < 0:
            raise ValueError('semantic_basis_rank must be nonnegative')
        if self.semantic_basis_rank and self.semantic_interaction_mode != 'prompt':
            raise ValueError('semantic_basis_rank is supported only in prompt mode')
        self.semantic_probe_mode = str(semantic_probe_mode)
        if self.semantic_probe_mode not in ('layernorm', 'raw', 'norm_preserving'):
            raise ValueError(
                "semantic_probe_mode must be 'layernorm', 'raw', or "
                f"'norm_preserving', got {semantic_probe_mode!r}"
            )
        self.semantic_horizon_mode = str(semantic_horizon_mode)
        if self.semantic_horizon_mode not in (
                'global', 'segmented', 'multiscale', 'frequency_tokens'):
            raise ValueError(
                "semantic_horizon_mode must be 'global', 'segmented', "
                "'multiscale', or 'frequency_tokens', "
                f"'multiscale', got {semantic_horizon_mode!r}"
            )
        self.semantic_encoder = str(semantic_encoder)
        self.semantic_attribute_tokens = int(semantic_attribute_tokens)
        self.semantic_attribute_heads = int(semantic_attribute_heads)
        self.semantic_fixed_token_projection = bool(semantic_fixed_token_projection)
        self.semantic_injection_mode = str(semantic_injection_mode)
        if self.semantic_injection_mode not in ('residual', 'film'):
            raise ValueError(
                "semantic_injection_mode must be 'residual' or 'film', got "
                f'{self.semantic_injection_mode!r}'
            )
        self.fuzzy_rules = int(fuzzy_rules or 0)
        self.fuzzy_fusion_rules = int(fuzzy_fusion_rules or 0)
        self.fusion_desc = str(fusion_desc or 'fuzzy')
        self.soft_experts_n = int(soft_experts or 0)
        self.soft_experts = None
        if self.soft_experts_n > 0:
            self.soft_experts = FuzzySoftExperts(
                channel, self.soft_experts_n, int(soft_expert_rank),
                5 * 15, bool(soft_expert_crisp))
            print(' >>> [FuzzySoftExperts] %d experts rank %d, %d params, crisp=%s'
                  % (self.soft_experts_n, int(soft_expert_rank),
                     sum(p.numel() for p in self.soft_experts.parameters()),
                     bool(soft_expert_crisp)), flush=True)
        self.align_assets = align_assets
        self.align_random = bool(align_random)
        self.fuzzy_rules_tau = float(fuzzy_rules_tau)
        self.fuzzy_rules_crisp = bool(fuzzy_rules_crisp)
        self.d_llm = d_llm
        if self.dual_semantic:
            if legacy_d_llm is None and semantic_d_llm is None:
                if d_llm % 2 != 0:
                    raise ValueError(f"dual_semantic requires an even d_llm, got {d_llm}")
                legacy_d_llm = semantic_d_llm = d_llm // 2
            elif legacy_d_llm is None or semantic_d_llm is None:
                raise ValueError("legacy_d_llm and semantic_d_llm must be specified together")
            self.legacy_d_llm = int(legacy_d_llm)
            self.semantic_d_llm = int(semantic_d_llm)
            self.d_llm = self.legacy_d_llm + self.semantic_d_llm
        else:
            self.legacy_d_llm = d_llm
            self.semantic_d_llm = d_llm
        self.num_nodes = num_nodes


        self.simp_no_corr = os.environ.get('FUCAL_NO_CORR_HEAD', '0') == '1'
        self.simp_no_gap = os.environ.get('FUCAL_NO_GAP', '0') == '1'
        self.simp_no_aim = os.environ.get('FUCAL_NO_AIM', '0') == '1'
        self.simp_no_scaleratio = os.environ.get('FUCAL_NO_SCALERATIO', '0') == '1'

        self.gap_w     = float(os.environ.get('FUCAL_GAP_W', '0.5'))
        self.gap_decay = os.environ.get('FUCAL_GAP_DECAY', '0') == '1'
        self.gap_tau   = float(os.environ.get('FUCAL_GAP_TAU', '0'))

        self.gap_learn = os.environ.get('FUCAL_GAP_LEARN', '0') == '1'
        if self.gap_learn:
            self.gap_w_param = nn.Parameter(torch.tensor(0.5))

        self.gate_floor = float(os.environ.get('FUCAL_GATE_FLOOR', '0.0'))
        if any([self.simp_no_corr, self.simp_no_gap, self.simp_no_aim, self.simp_no_scaleratio]):
            print(f" >>> [SIMPLIFY] no_corr={self.simp_no_corr} no_gap={self.simp_no_gap} "
                  f"no_aim={self.simp_no_aim} no_scaleratio={self.simp_no_scaleratio}")


        self.ablation = ablation
        self.abl_no_semantic = (ablation == 'no_semantic')
        self.abl_no_ts_backbone = (ablation == 'no_ts_backbone')
        self.abl_no_cp = (ablation == 'no_cp')
        self.abl_simple_concat = (ablation == 'simple_concat_fusion')
        self.abl_cp_cqr = (ablation == 'cp_cqr')
        self.abl_cp_enbpi = (ablation == 'cp_enbpi')
        self.abl_no_fuzzy = (ablation == 'no_fuzzy')
        self.abl_posthoc_cp = (ablation in ('cp_cqr', 'cp_enbpi'))

        self.abl_no_var_mixer = False
        if not use_var_mixer:
            self.abl_no_var_mixer = True
            print(" >>> [AUTO] var_mixer disabled: data conditions not met")

        self.trend_mode = 'multiscale' if trend_mode == 'auto' else trend_mode

        print(f"\n >>> [ABLATION] Mode: '{ablation}'")
        print(f" >>> [Trend] trend_mode={self.trend_mode}")
        if ablation != 'full':
            flags = {k: v for k, v in self.__dict__.items() if k.startswith('abl_') and v}
            for k in flags:
                print(f"     {k} = True")
        if ablation == 'full' and self.abl_no_var_mixer:
            print("     abl_no_var_mixer = True (auto)")

        self.is_large_data = is_large_dataset(data_path)

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

        print(f" >>> [FuCAL] channel={channel}, head={head}, dropout={dropout_n}")
        print(f" >>> [FuCAL] patches={self.patch_len_list}, stride={self.stride}")

        use_affine = True if self.is_large_data else False
        self.normalize_layers = Normalize(num_nodes, affine=use_affine).to(self.device)

        if self.abl_simple_concat:
            self.mtp_fusions = nn.ModuleList([
                SimpleConcatFusion(channel).to(self.device) for _ in self.patch_len_list
            ])
        else:
            self.mtp_fusions = nn.ModuleList([
                MTPSpectrumFusion(channel, int((seq_len - p) / self.stride) + 2).to(self.device)
                for p in self.patch_len_list
            ])

        self.align_head = None
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

            if self.fuzzy_rules > 1:

                self.flatten_heads.append(
                    FuzzyRuleHead(head_nf, pred_len, self.fuzzy_rules, self.d_llm,
                                  dropout=dropout_n, max_hidden=192,
                                  init_tau=self.fuzzy_rules_tau).to(self.device)
                )
            else:
                self.flatten_heads.append(
                    EnhancedFlattenHead(head_nf, pred_len, dropout=dropout_n, max_hidden=192).to(self.device)
                )
            if not self.simp_no_corr:
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
        self.fuzzy_fusion = None
        if self.fuzzy_fusion_rules > 0:
            with torch.no_grad():
                _prior = self.patch_weight_module().detach().float().cpu()

            self.fusion_descriptor = FusionDescriptor(
                self.fusion_desc, 5 * 15,
                int(getattr(self, 'legacy_d_llm', 768) or 768),
                int(seq_len))
            self.fuzzy_fusion = FuzzyPatchFusion(
                _prior, self.fuzzy_fusion_rules, self.fusion_descriptor.out_dim
            ).to(self.device)
            print(" >>> [FuzzyPatchFusion] %d rules, %d scales, %d params "
                  "(prior %s)" % (self.fuzzy_fusion_rules, _prior.numel(),
                                  sum(p.numel() for p in self.fuzzy_fusion.parameters()),
                                  [round(float(v), 3) for v in _prior]))

        self.dropout = nn.Dropout(dropout_n)


        _se = getattr(self, 'sem_encoder', 'llm')
        _proj_in = self.legacy_d_llm
        if _se == 'mlp':
            self.text_proj = build_descriptor_mlp(_proj_in, channel, dropout_n).to(self.device)
            print(f" >>> [SemEncoder] structured-descriptor MLP: {_proj_in} -> 128 -> {channel}")
        elif _se == 'sttf':
            self.text_proj = StructuredTokenTransformer(
                _proj_in, channel, d_tok=128, n_head=4, d_ff=512,
                n_layer=2, dropout=dropout_n).to(self.device)
            print(f" >>> [SemEncoder] trainable structured-token Transformer: "
                  f"{_proj_in} tokens x 128, 2 layers, 4 heads, ffn 512 -> {channel}")
        elif _se == 'llm_mlp':


            self.text_proj = build_descriptor_mlp(_proj_in, channel, dropout_n).to(self.device)
            print(f" >>> [SemEncoder] MLP capacity control on frozen LM embedding: "
                  f"{_proj_in} -> 128 -> {channel}")
        else:
            self.text_proj = nn.Sequential(
                nn.Linear(_proj_in, channel),
                nn.LayerNorm(channel),
                nn.LeakyReLU(),
            ).to(self.device)


        self.semantic_text_proj = None
        self.semantic_attribute_adapter = None
        self.semantic_feature_film_adapter = None
        self.structured_variable_embedding = None
        if not self.dual_semantic and _se in ('mlp', 'sttf'):
            self.structured_variable_embedding = nn.Parameter(
                torch.zeros(1, num_nodes, channel, device=self.device)
            )
            nn.init.normal_(self.structured_variable_embedding, std=0.02)
        if self.dual_semantic:
            if self.semantic_attribute_tokens and self.semantic_injection_mode == 'film':
                self.semantic_feature_film_adapter = AttributeFeatureFiLMAdapter(
                    self.semantic_d_llm, self.semantic_attribute_tokens, seq_len,
                    channel, encoder=self.semantic_encoder, num_nodes=num_nodes,
                    heads=self.semantic_attribute_heads, dropout=dropout_n,
                    gate_init_bias=self.semantic_gate_init_bias,
                ).to(self.device)
            elif (self.semantic_attribute_tokens
                  and self.semantic_horizon_mode != 'frequency_tokens'):
                self.semantic_attribute_adapter = AttributeCrossSemanticAdapter(
                    self.semantic_d_llm, self.semantic_attribute_tokens, channel,
                    encoder=self.semantic_encoder, num_nodes=num_nodes,
                    heads=self.semantic_attribute_heads, dropout=dropout_n,
                    fixed_token_projection=self.semantic_fixed_token_projection,
                ).to(self.device)
            elif self.semantic_encoder in ('mlp', 'llm_mlp'):
                self.semantic_text_proj = build_descriptor_mlp(
                    self.semantic_d_llm, channel, dropout_n
                ).to(self.device)
            elif self.semantic_encoder == 'sttf':
                self.semantic_text_proj = StructuredTokenTransformer(
                    self.semantic_d_llm, channel, d_tok=128, n_head=4,
                    d_ff=512, n_layer=2, dropout=dropout_n,
                ).to(self.device)
            elif self.semantic_encoder == 'llm':
                self.semantic_text_proj = SemanticProjection(
                    self.semantic_d_llm, channel,
                    input_mode=self.semantic_probe_mode,
                ).to(self.device)
            else:
                raise ValueError(
                    "semantic_encoder must be one of llm/llm_mlp/mlp/sttf, got "
                    f"{self.semantic_encoder!r}"
                )
            if (self.semantic_encoder in ('mlp', 'sttf')
                    and not self.semantic_attribute_tokens):
                self.structured_variable_embedding = nn.Parameter(
                    torch.zeros(1, num_nodes, channel, device=self.device)
                )
                nn.init.normal_(self.structured_variable_embedding, std=0.02)
            print(
                f" >>> [DualSemantic] legacy={self.legacy_d_llm} + "
                f"semantic={self.semantic_d_llm} ({self.semantic_encoder}); "
                "semantic arm is residual-only"
            )
            if self.semantic_attribute_tokens:
                print(
                    f" >>> [AttributeSemantic] {self.semantic_attribute_tokens} "
                    f"fuzzy tokens + {self.semantic_injection_mode} injection"
                )


        self.sem_head_on = (
            self.semantic_direct
            or os.environ.get('FUCAL_SEM_HEAD', '0') == '1'
            or self.semantic_gate_mode in ('auto', 'signed_auto', 'auto_decay')
        )
        self.sem_head = nn.Sequential(
            nn.Linear(channel, channel), nn.GELU(), nn.Linear(channel, pred_len)
        ).to(self.device)
        if self.semantic_direct and self.semantic_gate_mode == 'global':


            nn.init.normal_(self.sem_head[-1].weight, mean=0.0, std=0.005)
        else:


            nn.init.zeros_(self.sem_head[-1].weight)
        nn.init.zeros_(self.sem_head[-1].bias)
        self.semantic_gate_logit = nn.Parameter(torch.tensor(-2.1972246))

        self.semantic_residual_head = None
        self.semantic_uncertainty_head = None
        if (self.semantic_gate_mode in ('auto', 'signed_auto', 'auto_decay')
                and self.semantic_injection_mode != 'film'):
            if self.semantic_basis_rank:
                probe_width = (
                    4 * channel
                    if self.semantic_attribute_tokens
                    else (self.d_llm if self.semantic_use_full_prompt
                          else self.semantic_d_llm)
                    if self.semantic_encoder == 'llm'
                    else channel
                )
                if self.semantic_horizon_mode == 'frequency_tokens':
                    if not self.semantic_attribute_tokens:
                        raise ValueError(
                            'frequency_tokens requires semantic_attribute_tokens'
                        )
                    self.semantic_residual_head = FrequencyConditionedTokenDCTProbe(
                        self.semantic_d_llm, self.semantic_attribute_tokens,
                        pred_len, self.semantic_basis_rank,
                        encoder=self.semantic_encoder,
                        heads=self.semantic_attribute_heads,
                        dropout=dropout_n,
                    ).to(self.device)
                    probe_width = self.semantic_d_llm
                else:
                    probe_class = (
                        SemanticDCTProbe
                        if self.semantic_horizon_mode == 'global'
                        else SegmentedSemanticDCTProbe
                    )
                    probe_kwargs = dict(input_mode=self.semantic_probe_mode)
                    if self.semantic_horizon_mode != 'global':
                        probe_kwargs['mode'] = self.semantic_horizon_mode
                    self.semantic_residual_head = probe_class(
                        probe_width, pred_len, self.semantic_basis_rank,
                        **probe_kwargs,
                    ).to(self.device)
                print(
                    f" >>> [SemanticDCTProbe] {self.semantic_encoder} representation "
                    f"({probe_width}) -> {self.semantic_basis_rank} smooth "
                    f"horizon coefficients; input_mode={self.semantic_probe_mode}; "
                    f"horizon_mode={self.semantic_horizon_mode}; "
                    f"full_prompt={self.semantic_use_full_prompt}"
                )
            elif self.semantic_interaction_mode == 'cross':
                self.semantic_residual_head = nn.Sequential(
                    nn.LayerNorm(4 * channel),
                    nn.Linear(4 * channel, channel), nn.GELU(),
                    nn.Linear(channel, pred_len),
                ).to(self.device)
            else:

                self.semantic_residual_head = nn.Sequential(
                    nn.Linear(channel, channel), nn.GELU(),
                    nn.Linear(channel, pred_len),
                ).to(self.device)
            if not self.semantic_basis_rank:
                nn.init.zeros_(self.semantic_residual_head[-1].weight)
                nn.init.zeros_(self.semantic_residual_head[-1].bias)
            if self.semantic_uncertainty:
                _uncertainty_in = (
                    4 * channel
                    if self.semantic_attribute_tokens
                    else 4 * channel
                    if self.semantic_interaction_mode == 'cross'
                    else channel
                )
                self.semantic_uncertainty_head = nn.Sequential(
                    nn.LayerNorm(_uncertainty_in),
                    nn.Linear(_uncertainty_in, channel), nn.GELU(),
                    nn.Linear(channel, pred_len),
                ).to(self.device)
                nn.init.zeros_(self.semantic_uncertainty_head[-1].weight)
                nn.init.zeros_(self.semantic_uncertainty_head[-1].bias)


        self.semantic_auto_gate = None


        self.semantic_horizon_gate = None
        self.semantic_horizon_gate_segments = 3
        if self.semantic_gate_mode in ('auto', 'signed_auto', 'auto_decay'):
            if not self.dual_semantic:
                raise ValueError(
                    "semantic_gate_mode='auto' requires dual_semantic so that the "
                    "gated path can only touch the additive language residual"
                )
            if self.semantic_gate_floor != 0.0:
                print(
                    f" >>> [SemanticAutoGate] forcing gate_floor "
                    f"{self.semantic_gate_floor:.3f} -> 0.0 (auto mode must be "
                    "able to close completely)"
                )
                self.semantic_gate_floor = 0.0
            if self.semantic_gate_mode == 'auto_decay':
                self.semantic_auto_gate = SemanticAutoDecayGate(
                    n_features=NUM_GATE_FEATURES,
                    init_bias=self.semantic_gate_init_bias,
                    zero_init=True,
                ).to(self.device)
            else:
                self.semantic_auto_gate = SemanticAutoGate(
                    n_features=NUM_GATE_FEATURES,
                    init_bias=self.semantic_gate_init_bias,
                    zero_init=True,
                    signed_output=(self.semantic_gate_mode == 'signed_auto'),
                ).to(self.device)
            if (self.semantic_horizon_mode in ('segmented', 'multiscale')
                    or os.environ.get('FUCAL_HORIZON_GATE', '0') == '1'):
                self.semantic_horizon_gate = nn.Sequential(
                    nn.LayerNorm(NUM_GATE_FEATURES),
                    nn.Linear(NUM_GATE_FEATURES, self.semantic_horizon_gate_segments),
                ).to(self.device)
                nn.init.zeros_(self.semantic_horizon_gate[-1].weight)
                nn.init.constant_(
                    self.semantic_horizon_gate[-1].bias,
                    self.semantic_gate_init_bias,
                )
                _bounds = torch.linspace(
                    0, self.pred_len, self.semantic_horizon_gate_segments + 1,
                    dtype=torch.float64,
                ).round().to(torch.long).tolist()
                self.semantic_horizon_bounds = tuple(
                    (int(_bounds[i]), int(_bounds[i + 1]))
                    for i in range(self.semantic_horizon_gate_segments)
                )
                print(
                    f" >>> [SemanticAutoGate] horizon-local gate g(B,K,N), "
                    f"K={self.semantic_horizon_gate_segments}, "
                    f"bounds={self.semantic_horizon_bounds}"
                )
            n_gate_params = sum(p.numel() for p in self.semantic_auto_gate.parameters())
            print(
                f" >>> [SemanticAutoGate] per-sample/per-variable gate g(B,N): "
                f"{NUM_GATE_FEATURES} features, {n_gate_params} params, "
                f"init_bias={self.semantic_gate_init_bias:.6f} "
                f"(g0={float(torch.sigmoid(torch.tensor(self.semantic_gate_init_bias))):.4f})"
            )
            print(f" >>> [SemanticAutoGate] features={list(GATE_FEATURE_NAMES)}")
            print(
                f" >>> [SemanticInteraction] mode={self.semantic_interaction_mode}; "
                "cross uses [language, numeric, product, distance]"
            )
            if self.semantic_uncertainty:
                print(
                    " >>> [FuzzySemanticUQ] language reliability also supplies a "
                    f"bounded log-scale residual (clip={self.semantic_uncertainty_clip})"
                )
        if self.semantic_residual_clip > 0.0:
            print(
                f" >>> [SemanticResidual] tanh limiter in standardized space: "
                f"|residual| <= {self.semantic_residual_clip:.4f}"
            )
        if self.semantic_direct:
            print(
                f" >>> [SemanticDirect] ON | gate_floor={self.semantic_gate_floor:.3f} "
                f"| gate_mode={self.semantic_gate_mode}"
            )

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

        if self.trend_mode == 'off':
            self.trend_model = ZeroTrend(pred_len).to(self.device)
        elif self.trend_mode == 'simple':
            self.trend_model = SimpleTrend(seq_len, pred_len).to(self.device)
        else:
            self.trend_model = MultiScaleTrend(
                seq_len, pred_len, is_large_data=self.is_large_data
            ).to(self.device)

        if self.abl_cp_cqr:
            self.cfm = CQR_CFM(pred_len, seq_len, hidden_dim=64).to(self.device)
        elif self.abl_cp_enbpi:
            self.cfm = EnbPI_CFM(pred_len, seq_len, hidden_dim=64).to(self.device)
        elif self.abl_no_cp:
            self.cfm = VanillaCFM(pred_len, seq_len).to(self.device)
        else:
            self.cfm = SmoothAffineCFM(
                pred_len=pred_len, seq_len=seq_len, hidden_dim=64, dropout=0.1
            ).to(self.device)

    def set_stage(self, stage):
        self.cfm.set_stage(stage)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _finish_prediction(self, mu_pre_gap, x, x_flat, trend, log_beta_accum,
                           seq_std, return_details=False, post_correction=None):
        gap = x[:, -1:, :] - mu_pre_gap[:, 0:1, :]
        if self.simp_no_gap:
            mu_gap = mu_pre_gap
        elif self.gap_decay:
            _T = mu_pre_gap.shape[1]
            _tau = self.gap_tau if self.gap_tau > 0 else max(_T / 4.0, 1.0)
            _t = torch.arange(_T, device=mu_pre_gap.device, dtype=mu_pre_gap.dtype)
            _dec = torch.exp(-_t / _tau).view(1, _T, 1)
            mu_gap = mu_pre_gap + gap * self.gap_w * _dec
        else:
            _gw = self.gap_w_param if self.gap_learn else self.gap_w
            mu_gap = mu_pre_gap + gap * _gw

        if self.abl_no_cp:
            mu_corrected = mu_gap
            beta = (mu_gap - trend).abs().clamp(min=1e-6)
            correction_l1 = torch.tensor(0.0, device=x.device)
            cp_details = None
        else:
            if return_details:
                mu_corrected, beta, correction_l1, cp_details = self.cfm(
                    mu_gap, log_beta_accum, trend, x_history=x_flat, return_details=True
                )
            else:
                mu_corrected, beta, correction_l1 = self.cfm(
                    mu_gap, log_beta_accum, trend, x_history=x_flat
                )
                cp_details = None


        if post_correction is not None:
            mu_corrected = mu_corrected + post_correction

        mu_denorm = self.normalize_layers(mu_corrected, "denorm")
        beta_out = beta * seq_std
        return mu_denorm, beta_out, correction_l1, cp_details, mu_gap, gap

    def forward(self, x, x_mark, embeddings, return_components=False,
                return_base=False):
        B, S, N = x.shape

        x_input = x
        seq_mean = x_input.mean(dim=1, keepdim=True).detach()
        seq_var = torch.var(x_input, dim=1, keepdim=True, unbiased=False) + 1e-5
        seq_std = torch.sqrt(seq_var).detach()

        x = self.normalize_layers(x_input, "norm")
        x_ci = x.permute(0, 2, 1).reshape(B * N, S, 1)

        trend = self.trend_model(x_ci.squeeze(-1)).reshape(B, N, self.pred_len).permute(0, 2, 1)

        _fusion_llm_src = None
        text_feat = None
        semantic_text_feat = None
        semantic_raw_feat = None
        if getattr(self, 'shuffle_emb', False) and embeddings is not None:
            _perm = torch.randperm(embeddings.shape[0], device=embeddings.device)
            embeddings = embeddings[_perm]
        if not self.abl_no_semantic and embeddings is not None:
            if embeddings.dim() == 3:
                if embeddings.shape[1] == self.d_llm and embeddings.shape[2] == N:
                    emb_input = embeddings.permute(0, 2, 1)
                elif embeddings.shape[1] == N and embeddings.shape[2] == self.d_llm:
                    emb_input = embeddings
                else:
                    raise ValueError(
                        "invalid embedding cache shape: got "
                        f"{tuple(embeddings.shape)}, expected [B,{self.d_llm},{N}] "
                        f"or [B,{N},{self.d_llm}]"
                    )
            else:
                emb_input = embeddings.unsqueeze(1).repeat(1, N, 1)
            if emb_input.shape[-1] != self.d_llm:
                raise ValueError(
                    "embedding cache width mismatch: got "
                    f"{emb_input.shape[-1]}, expected {self.d_llm} "
                    f"(legacy={self.legacy_d_llm}, semantic={self.semantic_d_llm})"
                )


            _cmode = os.environ.get('FUCAL_EMB_CENTER', '0').strip().lower()
            if _cmode not in ('0', '', 'off', 'none'):
                _w = min(768, emb_input.shape[-1])
                _st = getattr(self, '_emb_center_stats', None)
                if _st is None:
                    import numpy as _np
                    _z = _np.load(os.environ['FUCAL_EMB_CENTER_STATS'])
                    _st = (torch.as_tensor(_z['mu_var']).float(),
                           float(_z['scale']),
                           torch.as_tensor(_z['pcs']).float(),
                           torch.as_tensor(_z['pc_std']).float())
                    self._emb_center_stats = _st
                _mu, _sc, _pcs, _pcstd = _st
                _mu = _mu.to(device=emb_input.device, dtype=emb_input.dtype)
                _mu = _mu[:emb_input.shape[1], :_w].unsqueeze(0)
                _head = emb_input[..., :_w] - _mu
                if _cmode in ('centerscale', 'cs'):
                    _head = _head * _sc
                elif _cmode.startswith('drop') or _cmode.startswith('white'):


                    _isw = _cmode.startswith('white')
                    _k = int(_cmode[5:]) if _isw else int(_cmode[4:])
                    _p = _pcs[:, :_w].to(device=emb_input.device,
                                         dtype=emb_input.dtype)
                    _co = torch.einsum('bnd,kd->bnk', _head, _p)
                    _lo = _co[..., :_k]
                    _head = _head - torch.einsum('bnk,kd->bnd', _lo, _p[:_k])
                    if _isw:


                        _sd = _pcstd.to(device=emb_input.device,
                                        dtype=emb_input.dtype)
                        _hi = _co[..., _k:]
                        _head = _head - torch.einsum(
                            'bnk,kd->bnd', _hi, _p[_k:])
                        _head = _head + torch.einsum(
                            'bnk,kd->bnd',
                            _hi / (_sd[_k:] + 1e-8) * _sd[_k:].mean(),
                            _p[_k:])
                    _head = _head + _mu
                elif _cmode.startswith('rank'):


                    _k = int(_cmode[4:])
                    _p = _pcs[:_k, :_w].to(device=emb_input.device,
                                           dtype=emb_input.dtype)
                    _head = torch.einsum('bnd,kd,ke->bne', _head, _p, _p) + _mu
                emb_input = (_head if _w == emb_input.shape[-1]
                             else torch.cat((_head, emb_input[..., _w:]), dim=-1))
            _fusion_llm_src = emb_input[
                ..., :min(int(getattr(self, 'legacy_d_llm', 768) or 768),
                          emb_input.shape[-1])]
            if self.dual_semantic:
                legacy_input = emb_input[..., :self.legacy_d_llm]
                semantic_input = emb_input[..., self.legacy_d_llm:]
                if self.semantic_use_full_prompt:


                    if os.environ.get('FUCAL_SEMANTIC_BLOCK_NORM', '0').strip().lower() in ('1','true','yes','on'):
                        _legacy_norm = F.layer_norm(
                            legacy_input, (legacy_input.shape[-1],)
                        )
                        _fuzzy_norm = F.layer_norm(
                            semantic_input, (semantic_input.shape[-1],)
                        )
                        semantic_raw_feat = torch.cat(
                            (_legacy_norm, _fuzzy_norm), dim=-1
                        ).reshape(B * N, self.d_llm)
                    else:
                        semantic_raw_feat = emb_input.reshape(B * N, self.d_llm)
                else:
                    semantic_raw_feat = semantic_input.reshape(
                        B * N, self.semantic_d_llm
                    )
                text_feat = self.text_proj(legacy_input).reshape(B * N, 1, self.channel)
                if (self.semantic_attribute_adapter is None
                        and self.semantic_feature_film_adapter is None
                        and self.semantic_horizon_mode != 'frequency_tokens'):
                    semantic_projected = self.semantic_text_proj(semantic_input)
                    if self.structured_variable_embedding is not None:
                        semantic_projected = (
                            semantic_projected + self.structured_variable_embedding
                        )
                    semantic_text_feat = semantic_projected.reshape(B * N, 1, self.channel)
            else:
                text_projected = self.text_proj(emb_input)
                if self.sem_encoder in ('mlp', 'sttf'):
                    text_projected = text_projected + self.structured_variable_embedding
                text_feat = text_projected.reshape(B * N, 1, self.channel)
                semantic_text_feat = text_feat

        weights = self.patch_weight_module()
        fuzzy_w = None
        if self.fuzzy_fusion is not None:
            if getattr(self, "_align_helper", None) is None:
                from models.FuCAL import FuzzyAnchorAlign as _FA
                self._align_helper = _FA(1, self.align_assets).to(x.device)
            _u = self._align_helper.memberships(x)
            _desc = self.fusion_descriptor(
                _u.reshape(_u.shape[0] * _u.shape[1], -1).to(x.dtype),
                _fusion_llm_src, x_ci)
            fuzzy_w = self.fuzzy_fusion(_desc)

        x_pad = F.pad(x_ci.squeeze(-1), (0, self.stride))
        x_flat = x_ci.squeeze(-1)
        gate, gate_reg = self.adaptive_gate(x_flat)

        if self.gate_floor > 0.0:
            gate = self.gate_floor + (1.0 - self.gate_floor) * gate

        seasonal_accum, correction_accum, log_beta_accum = 0, 0, 0
        base_seasonal_accum, base_correction_accum, base_log_beta_accum = 0, 0, 0
        last_enc_out = None
        branch_preds = []
        feature_film_gamma = None
        feature_film_beta = None
        feature_film_attention = None
        feature_film_reliability = None
        if (self.semantic_feature_film_adapter is not None
                and semantic_raw_feat is not None):
            (feature_film_gamma, feature_film_beta,
             feature_film_attention,
             feature_film_reliability) = self.semantic_feature_film_adapter(
                semantic_raw_feat, x_flat, B, N
            )

        for i, (p_len, fusion_layer) in enumerate(zip(self.patch_len_list, self.mtp_fusions)):
            patches = x_pad.unfold(dimension=1, size=p_len, step=self.stride)
            enc_in = self.patch_embeddings[i](patches)

            if not self.abl_no_var_mixer:
                enc_in = self.var_mixers[i](enc_in, B, N)

            if not self.simp_no_aim:
                aim_alpha = self.aim_layers[i](enc_in)
                enc_in = enc_in * aim_alpha.unsqueeze(-1)

            if self.abl_no_ts_backbone:
                enc_out = enc_in
            else:
                enc_out = self.encoder(enc_in)

            if text_feat is not None:
                if self.abl_simple_concat:
                    fused_out = fusion_layer(enc_out, x_text=text_feat)
                    enc_out = enc_out + gate.unsqueeze(1) * fused_out
                else:
                    fused_out = fusion_layer(enc_out, x_text=text_feat)
                    if self.simp_no_scaleratio:
                        scale_ratio = 1.0
                    else:
                        enc_norm = enc_out.detach().norm(dim=-1, keepdim=True) + 1e-8
                        fused_norm = fused_out.detach().norm(dim=-1, keepdim=True) + 1e-8
                        scale_ratio = torch.clamp(enc_norm / fused_norm, min=0.1, max=10.0)
                    enc_out = enc_out + gate.unsqueeze(1) * scale_ratio * fused_out

            base_enc_out = enc_out
            if feature_film_gamma is not None:
                base_flat = base_enc_out.reshape(base_enc_out.shape[0], -1)
                _bdesc = None
                if self.fuzzy_rules > 1 and emb_input is not None:
                    _bdesc = emb_input.reshape(-1, emb_input.shape[-1]).to(base_flat.dtype)
                base_seasonal_i = (
                    self.flatten_heads[i](base_flat, _bdesc, self.fuzzy_rules_crisp)
                    if self.fuzzy_rules > 1
                    else self.flatten_heads[i](base_flat)
                ).reshape(B, N, self.pred_len).permute(0, 2, 1)
                base_log_beta_i = self.beta_heads[i](base_flat).reshape(
                    B, N, self.pred_len
                ).permute(0, 2, 1)
                base_seasonal_accum += base_seasonal_i * weights[i]
                base_log_beta_accum += base_log_beta_i * weights[i]
                if not self.simp_no_corr:
                    base_correction_i = self.correction_heads[i](base_flat).reshape(
                        B, N, self.pred_len
                    ).permute(0, 2, 1)
                    base_correction_accum += base_correction_i * weights[i]
                enc_out = (
                    base_enc_out * (1.0 + feature_film_gamma.unsqueeze(1))
                    + feature_film_beta.unsqueeze(1)
                )

            last_enc_out = enc_out
            self._align_z = enc_out.reshape(enc_out.shape[0], -1)
            self._align_x = x
            enc_out_flat = enc_out.reshape(enc_out.shape[0], -1)
            if self.soft_experts is not None:
                if getattr(self, "_align_helper", None) is None:
                    from models.FuCAL import FuzzyAnchorAlign as _FA
                    self._align_helper = _FA(1, self.align_assets).to(x.device)
                _ue = self._align_helper.memberships(x)
                _ue = _ue.reshape(_ue.shape[0] * _ue.shape[1], -1).to(x.dtype)
                if _ue.shape[0] == enc_out.shape[0]:
                    enc_out_flat = self.soft_experts(
                        enc_out, _ue).reshape(enc_out.shape[0], -1)
            _desc_in = None
            if self.fuzzy_rules > 1 and emb_input is not None:
                _desc_in = emb_input.reshape(-1, emb_input.shape[-1]).to(enc_out_flat.dtype)
            seasonal_i = (
                self.flatten_heads[i](enc_out_flat, _desc_in, self.fuzzy_rules_crisp)
                if self.fuzzy_rules > 1
                else self.flatten_heads[i](enc_out_flat)
            ).reshape(B, N, self.pred_len).permute(0, 2, 1)
            log_beta_i = self.beta_heads[i](enc_out_flat).reshape(B, N, self.pred_len).permute(0, 2, 1)

            _wi = (weights[i] if fuzzy_w is None else
                   fuzzy_w[:, i].reshape(B, N, 1).permute(0, 2, 1))
            seasonal_accum += seasonal_i * _wi
            log_beta_accum += log_beta_i * _wi

            branch_preds.append(seasonal_i.detach())
            if os.environ.get('FUCAL_DUMP_BRANCHES', '0') == '1':
                if i == 0:
                    _branch_contrib = []
                _branch_contrib.append(seasonal_i.detach())
            if not self.simp_no_corr:
                correction_i = self.correction_heads[i](enc_out_flat).reshape(B, N, self.pred_len).permute(0, 2, 1)
                correction_accum += correction_i * _wi

        if os.environ.get('FUCAL_DUMP_BRANCHES', '0') == '1':


            self._dump_branches = [b.detach().float().cpu() for b in _branch_contrib]
            self._dump_trend = trend.detach().float().cpu()
            self._dump_w = (weights.detach().float().cpu() if fuzzy_w is None
                            else fuzzy_w.detach().float().cpu())
            self._dump_std = seq_std.detach().float().cpu()
            self._dump_mean = seq_mean.detach().float().cpu()
        if self.simp_no_corr:
            correction_accum = torch.zeros_like(seasonal_accum)
            if feature_film_gamma is not None:
                base_correction_accum = torch.zeros_like(base_seasonal_accum)


        if feature_film_gamma is not None:
            mu_pre_gap_base = trend + base_seasonal_accum + base_correction_accum
            base_log_beta_for_finish = base_log_beta_accum
        else:
            mu_pre_gap_base = trend + seasonal_accum + correction_accum
            base_log_beta_for_finish = log_beta_accum
        legacy_semantic_contribution = torch.zeros_like(mu_pre_gap_base)
        if (self.semantic_gate_mode in ('auto', 'signed_auto', 'auto_decay')
                and text_feat is not None
                and not self.disable_legacy_semantic):
            _legacy_sem = self.sem_head(text_feat.squeeze(1)).reshape(
                B, N, self.pred_len
            ).permute(0, 2, 1)
            _legacy_gate = self.semantic_gate_floor + (
                1.0 - self.semantic_gate_floor
            ) * torch.sigmoid(self.semantic_gate_logit)
            legacy_semantic_contribution = _legacy_gate * _legacy_sem
            mu_pre_gap_base = mu_pre_gap_base + legacy_semantic_contribution

        semantic_residual = torch.zeros_like(mu_pre_gap_base)
        semantic_contribution = torch.zeros_like(mu_pre_gap_base)
        semantic_gate = torch.zeros((), device=x.device, dtype=mu_pre_gap_base.dtype)
        semantic_gate_map = torch.zeros(
            (B, N), device=x.device, dtype=mu_pre_gap_base.dtype
        )
        semantic_gate_amplitude = torch.zeros_like(semantic_gate_map)
        semantic_gate_tau = torch.zeros_like(semantic_gate_map)
        semantic_gate_temporal = None
        semantic_alignment_loss = torch.zeros(
            (), device=x.device, dtype=mu_pre_gap_base.dtype
        )
        semantic_attribute_attention = torch.zeros(
            (B, N, max(self.semantic_attribute_tokens, 1)),
            device=x.device, dtype=mu_pre_gap_base.dtype,
        )
        semantic_log_beta_contribution = torch.zeros_like(log_beta_accum)
        if feature_film_gamma is not None:
            semantic_contribution = (
                trend + seasonal_accum + correction_accum
                + legacy_semantic_contribution - mu_pre_gap_base
            )
            if self.semantic_gate_const >= 0.0:
                semantic_gate_map = torch.full(
                    (B, N), self.semantic_gate_const,
                    device=x.device, dtype=mu_pre_gap_base.dtype,
                )
            else:
                semantic_gate_map = feature_film_reliability.reshape(B, N).to(
                    dtype=mu_pre_gap_base.dtype
                )
            semantic_gate = semantic_gate_map.mean()
            semantic_attribute_attention = feature_film_attention.reshape(
                B, N, self.semantic_attribute_tokens
            )
        _has_semantic = (
            semantic_text_feat is not None
            or (self.semantic_attribute_adapter is not None and semantic_raw_feat is not None)
            or (self.semantic_horizon_mode == 'frequency_tokens'
                and semantic_raw_feat is not None)
        )
        if (self.semantic_injection_mode != 'film'
                and self.sem_head_on and _has_semantic):
            _head = (
                self.semantic_residual_head
                if self.semantic_gate_mode in ('auto', 'signed_auto', 'auto_decay')
                else self.sem_head
            )
            _numeric_flat = last_enc_out.mean(dim=1).detach()
            if self.semantic_horizon_mode == 'frequency_tokens':
                _head_input = None
            elif self.semantic_attribute_adapter is not None:
                (_head_input, _attended,
                 _attribute_attention) = self.semantic_attribute_adapter(
                    semantic_raw_feat, _numeric_flat, B, N
                )
                semantic_attribute_attention = _attribute_attention.reshape(
                    B, N, self.semantic_attribute_tokens
                )
                _attended_bnc = F.normalize(
                    _attended.reshape(B, N, self.channel), dim=-1
                )
                _numeric_bnc = F.normalize(
                    _numeric_flat.reshape(B, N, self.channel), dim=-1
                )
                semantic_alignment_loss = (
                    1.0 - (_attended_bnc * _numeric_bnc).sum(-1)
                ).mean()
            else:
                _semantic_flat = (
                    semantic_raw_feat
                    if self.semantic_basis_rank and self.semantic_encoder == 'llm'
                    else semantic_text_feat.squeeze(1)
                )
            if self.semantic_horizon_mode == 'frequency_tokens':
                pass
            elif self.semantic_attribute_adapter is not None:
                pass
            elif self.semantic_interaction_mode == 'cross':
                _s = F.layer_norm(_semantic_flat, (_semantic_flat.shape[-1],))
                _n = F.layer_norm(_numeric_flat, (_numeric_flat.shape[-1],))
                _head_input = torch.cat((_s, _n, _s * _n, (_s - _n).abs()), dim=-1)


                _s_bnc = F.normalize(_s.reshape(B, N, self.channel), dim=-1)
                _n_bnc = F.normalize(_n.reshape(B, N, self.channel), dim=-1)
                if B > 1:
                    _logits = torch.einsum('bnc,mnc->nbm', _s_bnc, _n_bnc) / 0.10
                    _labels = torch.arange(B, device=x.device).repeat(N)
                    semantic_alignment_loss = 0.5 * (
                        F.cross_entropy(_logits.reshape(N * B, B), _labels)
                        + F.cross_entropy(
                            _logits.transpose(-1, -2).reshape(N * B, B), _labels
                        )
                    )
                else:
                    semantic_alignment_loss = (1.0 - (_s_bnc * _n_bnc).sum(-1)).mean()
            else:
                _head_input = _semantic_flat


            if (semantic_text_feat is not None
                    and self.semantic_injection_mode != 'film'
                    and self.semantic_interaction_mode == 'prompt'):
                _s_align = F.normalize(
                    semantic_text_feat.squeeze(1).reshape(B, N, self.channel),
                    dim=-1,
                )
                _n_align = F.normalize(
                    _numeric_flat.reshape(B, N, self.channel), dim=-1
                )
                if B > 1:
                    _logits = torch.einsum(
                        'bnc,mnc->nbm', _s_align, _n_align
                    ) / 0.10
                    _labels = torch.arange(B, device=x.device).repeat(N)
                    semantic_alignment_loss = 0.5 * (
                        F.cross_entropy(_logits.reshape(N * B, B), _labels)
                        + F.cross_entropy(
                            _logits.transpose(-1, -2).reshape(N * B, B),
                            _labels,
                        )
                    )
                else:
                    semantic_alignment_loss = (
                        1.0 - (_s_align * _n_align).sum(-1)
                    ).mean()
            if self.semantic_horizon_mode == 'frequency_tokens':
                _sem_flat = _head(semantic_raw_feat, x_flat)
            else:
                _sem_flat = _head(_head_input)
            _sem = _sem_flat.reshape(
                B, N, self.pred_len
            ).permute(0, 2, 1)
            if self.semantic_residual_clip > 0.0:


                _clip = self.semantic_residual_clip
                _sem = _clip * torch.tanh(_sem / _clip)
            semantic_residual = _sem
            if self.semantic_auto_gate is not None:
                gate_features = build_semantic_gate_features(
                    x_norm=x.detach(),
                    branch_preds=branch_preds,
                    log_beta=log_beta_accum.detach(),
                    semantic_residual=_sem.detach(),
                    numeric_anomaly=(mu_pre_gap_base - trend).detach(),
                    signed_consistency=(
                        self.semantic_gate_mode == 'signed_auto'
                    ),
                )
                _gate_inputs = gate_features.to(
                    dtype=self.semantic_auto_gate.proj.weight.dtype
                )
                if self.semantic_gate_mode == 'auto_decay':
                    (_temporal_gate, _amplitude,
                     _tau) = self.semantic_auto_gate(
                        _gate_inputs, self.pred_len
                    )
                    semantic_gate_temporal = _temporal_gate.to(dtype=_sem.dtype)
                    semantic_gate_amplitude = _amplitude.to(dtype=_sem.dtype)
                    semantic_gate_tau = _tau.to(dtype=_sem.dtype)
                    semantic_gate_map = semantic_gate_temporal.mean(dim=1)
                    semantic_gate = semantic_gate_temporal.mean()
                    _alpha = float(os.environ.get('FUCAL_SEMANTIC_ALPHA', '1.0'))
                    semantic_contribution = (
                        _alpha * semantic_gate_temporal * _sem
                    )
                else:
                    semantic_gate_map = self.semantic_auto_gate(
                        _gate_inputs
                    ).to(dtype=_sem.dtype)
                if (self.semantic_gate_mode != 'auto_decay'
                        and self.semantic_horizon_gate is not None):
                    _seg_logits = self.semantic_horizon_gate(
                        gate_features.to(dtype=_sem.dtype)
                    )


                    _base_h_gate = semantic_gate_map.unsqueeze(1)
                    if os.environ.get('FUCAL_SEMANTIC_TRANSPORT', '0') == '1':


                        _allocation = (
                            self.semantic_horizon_gate_segments
                            * torch.softmax(_seg_logits, dim=-1)
                        ).permute(0, 2, 1)
                        _seg_gate = _base_h_gate * _allocation
                    elif os.environ.get('FUCAL_SEMANTIC_LOCAL_SCALE', '0') == '1':


                        _local_scale = (
                            2.0 * torch.sigmoid(
                                _seg_logits - self.semantic_gate_init_bias
                            )
                        ).permute(0, 2, 1)
                        _seg_gate = _base_h_gate * _local_scale
                    else:
                        _seg_delta = (
                            0.25 * torch.tanh(_seg_logits).permute(0, 2, 1)
                        )
                        _seg_gate = (_base_h_gate + _seg_delta).clamp(0.0, 1.0)
                    _h_gate = torch.zeros(
                        B, self.pred_len, N, device=_sem.device,
                        dtype=_sem.dtype,
                    )
                    for _k, (_start, _end) in enumerate(
                            self.semantic_horizon_bounds):
                        if _end > _start:
                            _h_gate[:, _start:_end, :] = (
                                _seg_gate[:, _k, :].unsqueeze(1)
                            )
                    semantic_gate_map = _h_gate.mean(dim=1)
                    semantic_gate_temporal = _h_gate
                    semantic_gate = _h_gate.mean()
                    semantic_contribution = _h_gate * _sem
                elif self.semantic_gate_mode != 'auto_decay':
                    semantic_gate = semantic_gate_map.mean()
                    _alpha = float(os.environ.get('FUCAL_SEMANTIC_ALPHA', '1.0'))
                    semantic_contribution = _alpha * semantic_gate_map.unsqueeze(1) * _sem
            else:
                semantic_gate = self.semantic_gate_floor + (
                    1.0 - self.semantic_gate_floor
                ) * torch.sigmoid(self.semantic_gate_logit)
                semantic_contribution = semantic_gate * _sem
                semantic_gate_map = semantic_gate.expand(B, N).to(dtype=_sem.dtype)

            if self.semantic_uncertainty_head is not None:
                _sem_log_beta = self.semantic_uncertainty_head(_head_input).reshape(
                    B, N, self.pred_len
                ).permute(0, 2, 1)
                _uclip = max(self.semantic_uncertainty_clip, 1e-8)
                _sem_log_beta = _uclip * torch.tanh(_sem_log_beta / _uclip)
                _uncertainty_gate = (
                    semantic_gate_temporal
                    if semantic_gate_temporal is not None
                    else semantic_gate_map.unsqueeze(1)
                )
                semantic_log_beta_contribution = _uncertainty_gate * _sem_log_beta


        _seg_alpha_env = os.environ.get('FUCAL_SEMANTIC_SEG_ALPHA', '')
        if _seg_alpha_env:
            _av = [float(v) for v in _seg_alpha_env.split(',') if v.strip()]
            if _av:
                _bnds = torch.linspace(
                    0, self.pred_len, len(_av) + 1, dtype=torch.float64
                ).round().to(torch.long).tolist()
                _scale = torch.ones(
                    self.pred_len,
                    device=semantic_contribution.device,
                    dtype=semantic_contribution.dtype,
                )
                for _i, _a in enumerate(_av):
                    _scale[_bnds[_i]:_bnds[_i + 1]] = _a
                _scale = _scale.view(1, -1, 1)
                semantic_contribution = semantic_contribution * _scale
                if semantic_log_beta_contribution is not None:
                    semantic_log_beta_contribution = (
                        semantic_log_beta_contribution * _scale
                    )
                    semantic_log_beta = (
                        log_beta_accum + semantic_log_beta_contribution
                    )

        if self.semantic_injection_mode == 'film':
            mu_pre_gap = mu_pre_gap_base + semantic_contribution
            final_post_correction = None
        else:
            mu_pre_gap = mu_pre_gap_base
            final_post_correction = semantic_contribution
        semantic_log_beta = log_beta_accum + semantic_log_beta_contribution

        (mu_denorm, beta_out, correction_l1, cp_details,
         mu_gap, gap) = self._finish_prediction(
            mu_pre_gap, x, x_flat, trend, semantic_log_beta, seq_std,
            return_details=return_components,
            post_correction=final_post_correction,
        )

        mu_base_denorm = None
        if return_base or return_components or feature_film_gamma is not None:
            mu_base_denorm = self._finish_prediction(
                mu_pre_gap_base, x, x_flat, trend,
                base_log_beta_for_finish, seq_std,
                return_details=False,
            )[0]

        mu_full_denorm = mu_denorm
        if feature_film_gamma is not None:


            mu_denorm = mu_base_denorm + semantic_gate_map.unsqueeze(1) * (
                mu_full_denorm - mu_base_denorm
            )

        gate_mean = gate.mean().unsqueeze(0)
        gate_reg_out = gate_reg.unsqueeze(0) if gate_reg.dim() == 0 else gate_reg
        correction_l1_out = correction_l1.unsqueeze(0) if correction_l1.dim() == 0 else correction_l1

        if not return_components:
            if return_base:
                aux = {
                    'mu_base': mu_base_denorm,
                    'mu_full': mu_full_denorm,
                    'legacy_semantic_contribution': legacy_semantic_contribution,
                    'semantic_residual': semantic_residual,
                    'semantic_contribution': semantic_contribution,
                    'semantic_gate_map': semantic_gate_map,
                    'semantic_gate_amplitude': semantic_gate_amplitude,
                    'semantic_gate_tau': semantic_gate_tau,
                    'semantic_alignment_loss': semantic_alignment_loss,
                    'semantic_attribute_attention': semantic_attribute_attention,
                    'semantic_log_beta_contribution': semantic_log_beta_contribution,
                }
                return mu_denorm, beta_out, gate_mean, gate_reg_out, correction_l1_out, aux
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
            "semantic_contribution": semantic_contribution.detach(),
            "semantic_gate": semantic_gate.detach(),
            "semantic_gate_map": semantic_gate_map.detach(),
            "semantic_gate_amplitude": semantic_gate_amplitude.detach(),
            "semantic_gate_tau": semantic_gate_tau.detach(),
            "semantic_attribute_attention": semantic_attribute_attention.detach(),
            "semantic_residual": semantic_residual.detach(),
            "mu_pre_gap_base": mu_pre_gap_base.detach(),
            "mu_base_denorm": mu_base_denorm.detach(),
            "mu_full_denorm": mu_full_denorm.detach(),
        }
        if cp_details is not None:
            for k, v in cp_details.items():
                explain[k] = v.detach()

        return mu_denorm, beta_out, gate_mean, gate_reg_out, correction_l1_out, explain

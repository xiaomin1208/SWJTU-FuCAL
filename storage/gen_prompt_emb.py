import numpy as np
import torch

from utils.llm_backbone import load_llm_backbone, encode_text_batch

VARIABLE_META = {
    'ETTh1': {
        'names': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'roles': ['high useful load', 'high useless load', 'middle useful load',
                  'middle useless load', 'low useful load', 'low useless load', 'oil temperature'],
        'causal': {'OT': 'thermal lag from load', 'HUFL': 'drives heating'},
        'daily_patterns': {'peak_hours': (9, 18), 'off_peak': (22, 6)}
    },
    'ETTh2': {
        'names': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'roles': ['high useful load', 'high useless load', 'middle useful load',
                  'middle useless load', 'low useful load', 'low useless load', 'oil temperature'],
        'causal': {'OT': 'thermal lag from load', 'HUFL': 'primary demand driver'},
        'daily_patterns': {'peak_hours': (9, 18), 'off_peak': (22, 6)}
    },
    'ETTm1': {
        'names': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'roles': ['high useful load', 'high useless load', 'middle useful load',
                  'middle useless load', 'low useful load', 'low useless load', 'oil temperature'],
        'causal': {'OT': 'thermal lag 1-2 hours'},
        'daily_patterns': {'peak_hours': (9, 18), 'off_peak': (22, 6)}
    },
    'ETTm2': {
        'names': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'roles': ['high useful load', 'high useless load', 'middle useful load',
                  'middle useless load', 'low useful load', 'low useless load', 'oil temperature'],
        'causal': {'OT': 'thermal lag from load'},
        'daily_patterns': {'peak_hours': (9, 18), 'off_peak': (22, 6)}
    },
    'Solar': {
        'names': [f'Node_{i}' for i in range(136)] + ['OT'],
        'roles': ['solar power plant production'] * 137,
        'causal': {'All': 'production strictly driven by solar irradiance, weather, and daylight hours'},
        'daily_patterns': {'peak_hours': (7, 18), 'off_peak': (19, 5)}
    },
    'Weather': {
        'names': ['p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef',
                  'sh', 'H2OC', 'rho', 'wv', 'max_wv', 'wd', 'rain', 'raining',
                  'SWDR', 'PAR', 'max_PAR', 'Tlog', 'OT'],
        'roles': ['pressure', 'temperature', 'potential temperature', 'dew point',
                  'relative humidity', 'saturation vapor pressure', 'vapor pressure',
                  'vapor pressure deficit', 'specific humidity', 'water vapor concentration',
                  'air density', 'wind speed', 'max wind speed', 'wind direction',
                  'rainfall', 'rain indicator', 'shortwave downward radiation',
                  'photosynthetically active radiation', 'max PAR', 'temperature logger',
                  'target temperature'],
        'causal': {'OT': 'driven by solar radiation and atmospheric conditions'},
        'daily_patterns': {'peak_hours': (10, 16), 'off_peak': (22, 5)}
    },
    'ECL': {
        'names': [f'MT_{i:03d}' for i in range(321)],
        'roles': ['electricity consumption of client'] * 321,
        'causal': {'All': 'driven by human activity patterns and weather'},
        'daily_patterns': {'peak_hours': (8, 20), 'off_peak': (0, 6)}
    },
    'Traffic': {
        'names': [f'Sensor_{i}' for i in range(862)],
        'roles': ['road occupancy rate'] * 862,
        'causal': {'All': 'driven by commute patterns, weekday/weekend'},
        'daily_patterns': {'peak_hours': (7, 9), 'off_peak': (23, 5)}
    },
    'Exchange': {
        'names': ['Australia', 'British', 'Canada', 'Switzerland', 'China',
                  'Japan', 'New_Zealand', 'Singapore'],
        'roles': ['exchange rate vs USD'] * 8,
        'causal': {'All': 'driven by monetary policy and market sentiment'},
        'daily_patterns': {}
    },
}

WEEKDAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

FUZZY_TERM_SETS = {
    4: ['Low', 'Slightly Low', 'Slightly High', 'High'],

    9: ['Extremely Low', 'Very Low', 'Low', 'Slightly Low', 'Medium',
        'Slightly High', 'High', 'Very High', 'Extremely High'],

    15: ['Absolute Minimum', 'Extremely Low', 'Significantly Low', 'Very Low', 'Low',
         'Moderately Low', 'Slightly Low', 'Medium', 'Slightly High', 'Moderately High',
         'High', 'Very High', 'Significantly High', 'Extremely High', 'Absolute Maximum'],

    20: ['Absolute Minimum', 'Exceptionally Low', 'Vastly Low', 'Extremely Low', 'Profoundly Low',
         'Very Low', 'Significantly Low', 'Substantially Low', 'Notably Low', 'Slightly Low',
         'Slightly High', 'Notably High', 'Substantially High', 'Significantly High', 'Very High',
         'Profoundly High', 'Extremely High', 'Vastly High', 'Exceptionally High', 'Absolute Maximum'],

    25: ['Absolute Minimum', 'Exceptionally Low', 'Vastly Low', 'Extremely Low', 'Profoundly Low',
         'Very Low', 'Level-6 Low', 'Level-5 Low', 'Level-4 Low', 'Level-3 Low',
         'Level-2 Low', 'Level-1 Low', 'Medium', 'Level-1 High', 'Level-2 High',
         'Level-3 High', 'Level-4 High', 'Level-5 High', 'Level-6 High', 'Very High',
         'Profoundly High', 'Extremely High', 'Vastly High', 'Exceptionally High', 'Absolute Maximum'],

    30: ['Absolute Minimum', 'Exceptionally Low', 'Vastly Low', 'Extremely Low', 'Profoundly Low',
         'Very Low', 'Level-8 Low', 'Level-7 Low', 'Level-6 Low', 'Level-5 Low',
         'Level-4 Low', 'Level-3 Low', 'Level-2 Low', 'Level-1 Low', 'Slightly Low',
         'Slightly High', 'Level-1 High', 'Level-2 High', 'Level-3 High', 'Level-4 High',
         'Level-5 High', 'Level-6 High', 'Level-7 High', 'Level-8 High', 'Very High',
         'Profoundly High', 'Extremely High', 'Vastly High', 'Exceptionally High', 'Absolute Maximum']
}

FEATURE_NAMES = ['mean', 'std', 'slope', 'diff_std', 'range', 'tail_diff']


def triangular_mf(x, l, c, r):
    if c == l:
        left = 1.0 if x >= l else 0.0
    else:
        left = (x - l) / (c - l)
    if r == c:
        right = 1.0 if x <= r else 0.0
    else:
        right = (r - x) / (r - c)
    return max(0.0, min(left, right))


class FuzzyQuantizer:

    def __init__(self, num_levels=9):
        assert num_levels in FUZZY_TERM_SETS, f"num_levels must be one of {sorted(FUZZY_TERM_SETS)}"
        self.num_levels = num_levels
        self.terms = FUZZY_TERM_SETS[num_levels]
        self.quantiles = None
        self.fitted = False

    def _get_percentiles(self):
        return list(np.linspace(1.0, 99.0, self.num_levels))

    def fit(self, feature_dict):
        self.quantiles = {}
        percentiles = self._get_percentiles()
        for feat_name, values in feature_dict.items():
            self.quantiles[feat_name] = np.percentile(values, percentiles)
        self.fitted = True
        print(f"[FuzzyQuantizer] Fitted on {len(self.quantiles)} features, levels={self.num_levels}")

    def _build_mf_params(self, qs):
        K = self.num_levels
        mf_params = [(qs[0], qs[0], qs[1])]
        for i in range(1, K - 1):
            mf_params.append((qs[i - 1], qs[i], qs[i + 1]))
        mf_params.append((qs[-2], qs[-1], qs[-1]))
        return mf_params

    def quantize(self, feat_name, value):
        if not self.fitted or feat_name not in self.quantiles:
            return (self.terms[self.num_levels // 2], 0.5)

        qs = self.quantiles[feat_name]
        memberships = [triangular_mf(value, l, c, r) for (l, c, r) in self._build_mf_params(qs)]
        best_idx = int(np.argmax(memberships))
        best_degree = memberships[best_idx]

        if best_degree < 0.01:
            if value <= qs[0]:
                return (self.terms[0], 1.0)
            elif value >= qs[-1]:
                return (self.terms[-1], 1.0)
            else:
                return (self.terms[self.num_levels // 2], 0.5)
        return (self.terms[best_idx], round(float(best_degree), 2))


def _compute_features_vectorized(window):
    L, N = window.shape
    feats = np.zeros((N, 6), dtype=np.float32)

    feats[:, 0] = np.mean(window, axis=0)
    feats[:, 1] = np.std(window, axis=0)
    feats[:, 4] = np.ptp(window, axis=0)
    feats[:, 3] = np.std(np.diff(window, axis=0), axis=0)

    tail = max(L // 4, 2)
    feats[:, 5] = np.mean(window[-tail:], axis=0) - np.mean(window[:tail], axis=0)

    t = np.arange(L, dtype=np.float32)
    t_c = t - t.mean()
    t_std = np.std(t) + 1e-8
    w_c = window - feats[:, 0:1].T
    cov = t_c @ w_c / L
    w_std = feats[:, 1] + 1e-8
    feats[:, 2] = (cov / (t_std * w_std)) * w_std / t_std
    feats[feats[:, 1] < 1e-8, 2] = 0.0
    return feats


class GenPromptEmb:
    def __init__(self, device, input_len, data_path, model_name='gpt2', d_model=None, layer=None,
                 z_precision=1, encode_batch_size=None, num_fuzzy_levels=9):
        self.device = device
        self.input_len = input_len
        self.data_path = data_path.replace('.csv', '')
        self.z_precision = z_precision
        self.num_fuzzy_levels = num_fuzzy_levels

        self.backbone = load_llm_backbone(model_name_or_path=model_name, device=device,
                                          layer=layer, freeze=True)
        llm_hidden = self.backbone.hidden_size
        self.layer = self.backbone.best_layer
        self.d_model = llm_hidden if d_model is None else d_model

        if encode_batch_size is not None:
            self.encode_batch_size = encode_batch_size
        elif llm_hidden >= 3000:
            self.encode_batch_size = 12
        elif llm_hidden >= 1500:
            self.encode_batch_size = 32
        else:
            self.encode_batch_size = 64
        print(f"[GenPromptEmb] LLM: {model_name} (hidden={llm_hidden}, layer={self.layer}), "
              f"z_precision={z_precision}, encode_batch_size={self.encode_batch_size}, "
              f"fuzzy_levels={num_fuzzy_levels}")

        self.fuzzy_quantizer = FuzzyQuantizer(num_levels=num_fuzzy_levels)
        meta = VARIABLE_META.get(self.data_path, None)
        if meta:
            self.var_names = meta['names']
            self.var_roles = meta['roles']
            self.causal = meta.get('causal', {})
            self.daily_patterns = meta.get('daily_patterns', {})
        else:
            self.var_names = [f'V{i}' for i in range(2000)]
            self.var_roles = [f'variable {i}' for i in range(2000)]
            self.causal = {}
            self.daily_patterns = {}

        self.global_stats = None


        self._global_cache = {}

    def to(self, device):
        self.device = device
        self.backbone = self.backbone.to(device)
        return self

    def fit_global_stats(self, data_loader):
        print("[GenPromptEmb] Fitting global stats and fuzzy quantizer...")
        num_nodes = None
        all_features = {name: {} for name in FEATURE_NAMES}
        all_raw_features = []

        for x, y, x_mark, y_mark in data_loader:
            x_np = x.numpy()
            B, L, N = x_np.shape
            if num_nodes is None:
                num_nodes = N
                for key in all_features:
                    for n in range(N):
                        all_features[key][n] = []
            for b in range(B):
                feats = _compute_features_vectorized(x_np[b])
                all_raw_features.append(feats)
                for n in range(N):
                    for fi, fname in enumerate(FEATURE_NAMES):
                        all_features[fname][n].append(feats[n, fi])

        self.num_nodes = num_nodes
        all_raw_features = np.stack(all_raw_features, axis=0)
        self.global_stats = {
            'mean': np.mean(all_raw_features, axis=0),
            'std': np.std(all_raw_features, axis=0) + 1e-8,
        }

        fuzzy_features = {}
        for n in range(num_nodes):
            var_name = self.var_names[n] if n < len(self.var_names) else f'V{n}'
            for fname in FEATURE_NAMES:
                fuzzy_features[f"{var_name}_{fname}"] = np.array(all_features[fname][n])
        self.fuzzy_quantizer.fit(fuzzy_features)

    def _zscore(self, feat_vals, var_idx):
        if self.global_stats is not None:
            return (feat_vals - self.global_stats['mean'][var_idx]) / self.global_stats['std'][var_idx]
        return feat_vals

    def _build_full_prompt(self, window_mark, var_idx, feats, system_regime):
        var_name = self.var_names[var_idx] if var_idx < len(self.var_names) else f'V{var_idx}'
        var_role = self.var_roles[var_idx] if var_idx < len(self.var_roles) else f'variable {var_idx}'
        parts = [f"{var_name} ({var_role})."]

        display_names = ['Level', 'Volatility', 'Trend', 'Roughness', 'Range', 'Momentum']
        for fi, (fname, dname) in enumerate(zip(FEATURE_NAMES, display_names)):
            term, degree = self.fuzzy_quantizer.quantize(f"{var_name}_{fname}", feats[var_idx, fi])
            if degree >= 0.3:
                parts.append(f"{dname}: {term} ({degree:.2f}).")

        z_stats = self._zscore(feats[var_idx], var_idx)
        stat_names = ['mean_z', 'std_z', 'slope_z', 'roughness_z', 'range_z', 'momentum_z']
        stat_strs = [f"{sname}={round(float(sval), self.z_precision)}"
                     for sname, sval in zip(stat_names, z_stats)]
        parts.append(f"Stats: {', '.join(stat_strs)}.")
        parts.append(f"System: {system_regime}.")

        calendar = self._calendar_brief(window_mark)
        if calendar:
            parts.append(calendar)
        causal = self.causal.get(var_name, self.causal.get('All', ""))
        if causal:
            parts.append(f"Note: {causal}.")
        return " ".join(parts)

    def _calendar_brief(self, window_mark):
        if window_mark.shape[1] < 5:
            return ""
        end_hour = int(window_mark[-1, 4]) if window_mark.shape[1] > 4 else 12
        end_weekday = int(window_mark[-1, 3]) if window_mark.shape[1] > 3 else 0
        weekday_name = WEEKDAY_NAMES[end_weekday] if end_weekday < 7 else 'Unknown'
        peak = self.daily_patterns.get('peak_hours', (9, 18))
        off_peak = self.daily_patterns.get('off_peak', (22, 6))
        if peak[0] <= end_hour < peak[1]:
            period = "peak"
        elif end_hour >= off_peak[0] or end_hour < off_peak[1]:
            period = "off-peak"
        else:
            period = "transition"
        return f"{weekday_name} {end_hour}:00, {period}."

    def _get_system_regime(self, feats, N):
        slopes = feats[:N, 2]
        r_up = np.sum(slopes > 1e-4) / max(N, 1)
        r_down = np.sum(slopes < -1e-4) / max(N, 1)
        if r_down > 0.4 and r_up > 0.4:
            return "divergent"
        elif r_down > 0.6:
            return "declining"
        elif r_down > 0.4:
            return "weakening"
        elif r_up > 0.6:
            return "surging"
        elif r_up > 0.4:
            return "strengthening"
        return "stable"

    def _encode_unique_batch(self, all_prompts):
        results = [None] * len(all_prompts)
        uncached_idx = []
        for i, p in enumerate(all_prompts):
            if p in self._global_cache:
                results[i] = self._global_cache[p]
            else:
                uncached_idx.append(i)
        if not uncached_idx:
            return results

        unique_prompts = list(dict.fromkeys(all_prompts[i] for i in uncached_idx))
        unique_embs = encode_text_batch(self.backbone, unique_prompts, max_length=128,
                                        pooling="mean", batch_size=self.encode_batch_size)
        if unique_embs.shape[1] > self.d_model:
            unique_embs = unique_embs[:, :self.d_model]

        prompt_to_emb = {}
        for prompt, emb in zip(unique_prompts, unique_embs):
            emb = emb.detach().cpu()
            prompt_to_emb[prompt] = emb
            self._global_cache[prompt] = emb
        for i in uncached_idx:
            results[i] = prompt_to_emb[all_prompts[i]]

        if len(self._global_cache) > 300000:
            keys = list(self._global_cache.keys())
            for k in keys[:len(keys) // 2]:
                del self._global_cache[k]
        return results

    def generate_embeddings(self, x, x_mark):
        B, L, N = x.shape
        x_np = x.detach().cpu().numpy()
        x_mark_np = x_mark.detach().cpu().numpy()

        all_prompts = []
        for b in range(B):
            feats = _compute_features_vectorized(x_np[b])
            system_regime = self._get_system_regime(feats, N)
            for n in range(N):
                all_prompts.append(self._build_full_prompt(x_mark_np[b], n, feats, system_regime))

        all_embs = self._encode_unique_batch(all_prompts)
        result = torch.zeros(B, self.d_model, N, device=self.device)
        for idx, emb in enumerate(all_embs):
            result[idx // N, :, idx % N] = emb.to(self.device)
        return result

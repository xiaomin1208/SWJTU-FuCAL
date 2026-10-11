import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from utils.llm_backbone import encode_text_batch, get_llm_hidden_size, load_llm_backbone

FEATURE_KEYS = ("level", "volatility", "trend", "roughness", "range", "momentum")
DEFAULT_AXES = ("level", "volatility", "trend", "range", "momentum")

CANONICAL_POLES = {
    "level": ("below its normal baseline", "near its normal baseline", "above its normal baseline"),
    "volatility": ("stable with little variation", "normally variable", "volatile with large fluctuations"),
    "trend": ("falling over time", "approximately flat", "rising over time"),
    "roughness": ("smooth from step to step", "moderately irregular", "rough and erratic"),
    "range": ("compressed into a narrow range", "within its usual range", "spread across a wide range"),
    "momentum": ("showing downward momentum", "showing neutral momentum", "showing upward momentum"),
}
CANONICAL_INTENSITY = ("slightly", "mildly", "moderately", "clearly", "strongly", "very", "extremely")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--divide", choices=("train", "val", "test"), required=True)
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--output_len", type=int, default=96)
    parser.add_argument("--num_fuzzy_levels", type=int, required=True)
    parser.add_argument("--model_name", default="gpt2", help="HuggingFace model id or local path of GPT-2")
    parser.add_argument("--hidden_layer", type=int, default=6)
    parser.add_argument("--legacy_tag", default="gpt2", help="tag of the step-1 embeddings")
    parser.add_argument("--out_tag", required=True)
    parser.add_argument("--embed_root", default="./Embeddings")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="must cover all 2 x 5 x K phrases in one batch")
    return parser.parse_args()


def linguistic_state(feature, level, num_levels):
    z = -1.0 + 2.0 * level / max(num_levels - 1, 1)
    if abs(z) < 1e-8:
        return CANONICAL_POLES[feature][1]
    idx = min(len(CANONICAL_INTENSITY) - 1, max(0, int(round(abs(z) * len(CANONICAL_INTENSITY))) - 1))
    pole = CANONICAL_POLES[feature][0 if z < 0 else 2]
    return f"{CANONICAL_INTENSITY[idx]} {pole}"


def state_prompt(feature, level, num_levels):
    return f"The recent {feature} is {linguistic_state(feature, level, num_levels)}."


def build_level_scores(backbone, axes, num_levels, num_nodes, batch_size):


    if batch_size < 2 * len(axes) * num_levels:
        raise ValueError("--batch_size must be at least %d" % (2 * len(axes) * num_levels))
    prompts = []
    for feature in axes:
        for _ in range(2):
            prompts.extend(state_prompt(feature, level, num_levels) for level in range(num_levels))
    with torch.no_grad():
        encoded = encode_text_batch(backbone, prompts, max_length=64, pooling="mean",
                                    batch_size=batch_size).float().cpu().numpy()
    encoded = encoded.reshape(len(axes), 2, num_levels, -1)

    reference_raw = np.zeros((1, len(axes), num_levels), dtype=np.float64)
    selected_raw = np.zeros_like(reference_raw)
    for a in range(len(axes)):
        reference = encoded[a, 0].astype(np.float64)
        selected = encoded[a, 1].astype(np.float64)
        span = reference[-1] - reference[0]
        direction = span / (float(np.linalg.norm(span)) + 1e-12)
        center = 0.5 * (reference[0] + reference[-1])
        reference_raw[0, a] = (reference - center) @ direction
        selected_raw[0, a] = (selected - center) @ direction


    mean = reference_raw.mean(axis=(0, 2), keepdims=True)
    std = reference_raw.std(axis=(0, 2), keepdims=True)
    scores = ((selected_raw - mean) / (std + 1e-12)).astype(np.float32)
    return np.repeat(scores, num_nodes, axis=0)


def repair_uncovered_memberships(memberships, feature_values):
    memberships = memberships.copy()
    bad = np.argwhere(memberships.sum(axis=-1) <= 1e-8)
    for node, feature in bad:
        level = 0 if float(feature_values[node, feature]) <= 0.0 else memberships.shape[-1] - 1
        memberships[node, feature, level] = 1.0
    return memberships, int(len(bad))


def main():
    args = parse_args()
    axes = list(DEFAULT_AXES)
    K = args.num_fuzzy_levels
    hidden = get_llm_hidden_size(args.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = load_llm_backbone(args.model_name, device=device, layer=args.hidden_layer, freeze=True)
    level_scores = build_level_scores(backbone, axes, K, args.num_nodes, args.batch_size)

    stem = (args.data_path, f"i{args.input_len}_o{args.output_len}", args.divide)
    descriptor_dir = Path(args.embed_root, "desc_raw", f"fz{K}", *stem)
    legacy_dir = Path(args.embed_root, args.legacy_tag, f"fz{K}", *stem)
    output_dir = Path(args.embed_root, args.out_tag, f"fz{K}", *stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = sorted((p.name for p in descriptor_dir.glob("*.h5")), key=lambda s: int(Path(s).stem))
    legacy_names = sorted((p.name for p in legacy_dir.glob("*.h5")), key=lambda s: int(Path(s).stem))
    if not names or names != legacy_names:
        raise RuntimeError(f"step-1 and step-2 caches differ: {legacy_dir} vs {descriptor_dir}")

    feature_indices = [FEATURE_KEYS.index(f) for f in axes]
    n_feat = len(FEATURE_KEYS)
    repaired_total = 0
    start = time.time()
    for i, name in enumerate(names, start=1):
        with h5py.File(descriptor_dir / name, "r") as f:
            descriptor = np.asarray(f["embeddings"], dtype=np.float32).T
        memberships = descriptor[:, n_feat:n_feat + n_feat * K].reshape(args.num_nodes, n_feat, K)
        memberships, repaired = repair_uncovered_memberships(memberships, descriptor[:, :n_feat])
        repaired_total += repaired
        signed = np.einsum("naf,naf->na", memberships[:, feature_indices, :], level_scores, optimize=True)

        with h5py.File(legacy_dir / name, "r") as f:
            legacy = np.asarray(f["embeddings"], dtype=np.float32)
        if legacy.shape != (hidden, args.num_nodes):
            raise ValueError(f"step-1 embedding shape mismatch at {name}: {legacy.shape}")
        dual = np.concatenate((legacy, signed.T), axis=0).astype(np.float16)

        target = output_dir / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        with h5py.File(temporary, "w") as f:
            f.create_dataset("embeddings", data=dual, compression="gzip")
            f.attrs["composition"] = "legacy_plus_signed_fuzzy_axes"
            f.attrs["hidden_layer"] = args.hidden_layer
            f.attrs["signed_axes"] = json.dumps(axes)
            f.attrs["legacy_tag"] = args.legacy_tag
        os.replace(temporary, target)
        if i % 1000 == 0 or i == len(names):
            print(f"[store_signed_axes] {args.data_path}/{args.divide}: {i}/{len(names)} "
                  f"({time.time() - start:.0f}s)", flush=True)

    manifest = dict(data_path=args.data_path, split=args.divide, num_items=len(names), num_nodes=args.num_nodes,
                    num_fuzzy_levels=K, model_name=args.model_name, hidden_layer=args.hidden_layer, axes=axes,
                    legacy_tag=args.legacy_tag, repaired_memberships=repaired_total,
                    level_scores=level_scores[0].tolist())
    (output_dir.parent / f"manifest_{args.divide}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[store_signed_axes] saved {len(names)} windows to {output_dir}")


if __name__ == "__main__":
    main()

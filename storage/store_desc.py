import argparse
import os

import h5py
import numpy as np
from torch.utils.data import DataLoader

from data_provider.data_loader_save import Dataset_ETT_hour
from storage.gen_prompt_emb import FEATURE_NAMES, FuzzyQuantizer, _compute_features_vectorized, triangular_mf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--divide", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--num_nodes", type=int, default=7)
    p.add_argument("--input_len", type=int, default=96)
    p.add_argument("--output_len", type=int, default=96)
    p.add_argument("--num_fuzzy_levels", type=int, default=15, choices=[9, 15])
    p.add_argument("--z_precision", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--embed_root", type=str, default="./Embeddings")
    return p.parse_args()


def main():
    args = parse_args()
    K = args.num_fuzzy_levels
    size = [args.input_len, 0, args.output_len]


    train_set = Dataset_ETT_hour(flag='train', data_path=args.data_path, size=size)
    all_raw = []
    for x, y, xm, ym in DataLoader(train_set, batch_size=256, shuffle=False, num_workers=0):
        x_np = x.numpy()
        for b in range(x_np.shape[0]):
            all_raw.append(_compute_features_vectorized(x_np[b]))
    all_raw = np.stack(all_raw, axis=0)
    N = all_raw.shape[1]
    gmean = np.mean(all_raw, axis=0)
    gstd = np.std(all_raw, axis=0) + 1e-8

    quant = FuzzyQuantizer(num_levels=K)
    quant.fit({f"{n}_{fname}": all_raw[:, n, fi] for n in range(N) for fi, fname in enumerate(FEATURE_NAMES)})
    mfp = {key: quant._build_mf_params(qs) for key, qs in quant.quantiles.items()}

    target_set = Dataset_ETT_hour(flag=args.divide, data_path=args.data_path, size=size)
    loader = DataLoader(target_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    save_dir = os.path.join(args.embed_root, "desc_raw", f"fz{K}", args.data_path,
                            f"i{args.input_len}_o{args.output_len}", args.divide)
    os.makedirs(save_dir, exist_ok=True)

    gi = 0
    for x, y, xm, ym in loader:
        x_np = x.numpy()
        xm_np = xm.numpy()
        for b in range(x_np.shape[0]):
            feats = _compute_features_vectorized(x_np[b])
            z = np.round((feats - gmean) / gstd, args.z_precision)
            u = np.zeros((N, 6, K), dtype=np.float32)
            for n in range(N):
                for fi, fname in enumerate(FEATURE_NAMES):
                    v = float(feats[n, fi])
                    for k, (l, c, r) in enumerate(mfp[f"{n}_{fname}"]):
                        u[n, fi, k] = triangular_mf(v, l, c, r)
            slopes = feats[:, 2]
            frac_up = float(np.mean(slopes > 1e-4))
            frac_dn = float(np.mean(slopes < -1e-4))
            mk = xm_np[b][-1]
            wd = float(mk[3]) / 6.0 if mk.shape[0] > 3 else 0.0
            hr = float(mk[4]) if mk.shape[0] > 4 else 12.0
            cal = np.array([frac_up, frac_dn, wd, np.sin(2 * np.pi * hr / 24.0), np.cos(2 * np.pi * hr / 24.0)],
                           dtype=np.float32)
            vec = np.concatenate([z.astype(np.float32), u.reshape(N, 6 * K), np.tile(cal, (N, 1))], axis=1)
            with h5py.File(os.path.join(save_dir, f"{gi}.h5"), 'w') as f:
                f.create_dataset('embeddings', data=vec.T.astype(np.float32))
            gi += 1
    print(f"[store_desc] saved {gi} windows to {save_dir}")


if __name__ == "__main__":
    main()

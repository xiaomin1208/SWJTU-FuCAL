import argparse
import os
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.data_loader_save import get_dataset
from storage.gen_prompt_emb import FUZZY_TERM_SETS, GenPromptEmb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="ETTh1")
    parser.add_argument("--divide", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--output_len", type=int, default=96)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--num_fuzzy_levels", type=int, default=15, choices=sorted(FUZZY_TERM_SETS))
    parser.add_argument("--z_precision", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_tag", type=str, default="gpt2")
    parser.add_argument("--embed_root", type=str, default="./Embeddings")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = get_dataset(args.data_path, 'train', args.input_len, args.output_len)
    target_set = get_dataset(args.data_path, args.divide, args.input_len, args.output_len)

    generator = GenPromptEmb(device=device, input_len=args.input_len, data_path=args.data_path,
                             model_name=args.model_name, layer=args.layer,
                             z_precision=args.z_precision,
                             num_fuzzy_levels=args.num_fuzzy_levels).to(device)
    generator.fit_global_stats(DataLoader(train_set, batch_size=8, shuffle=False, num_workers=0))

    save_dir = os.path.join(args.embed_root, args.out_tag, f"fz{args.num_fuzzy_levels}",
                            args.data_path, f"i{args.input_len}_o{args.output_len}", args.divide)
    os.makedirs(save_dir, exist_ok=True)
    loader = DataLoader(target_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    start, idx = time.time(), 0
    with torch.no_grad():
        for x, y, x_mark, y_mark in loader:
            emb = generator.generate_embeddings(x.to(device), x_mark.to(device)).detach().cpu().numpy()
            for j in range(emb.shape[0]):
                with h5py.File(os.path.join(save_dir, f"{idx}.h5"), 'w') as f:
                    f.create_dataset('embeddings', data=emb[j].astype(np.float32))
                idx += 1
                if idx % 1000 == 0 or idx == len(target_set):
                    print(f"[store_emb] {args.data_path}/{args.divide}: {idx}/{len(target_set)} "
                          f"({time.time() - start:.0f}s)", flush=True)
    print(f"[store_emb] saved {idx} windows to {save_dir}")


if __name__ == "__main__":
    main()

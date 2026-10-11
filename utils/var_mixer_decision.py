import numpy as np
import os


DATA_SEARCH_PATHS = [
    './dataset/ETT-small',
    './dataset/ETT',
    './dataset',
]


def _find_csv_path(data_path):

    if os.path.isfile(data_path):
        return data_path

    basename = data_path
    if not basename.endswith('.csv'):
        basename = basename + '.csv'


    for search_dir in DATA_SEARCH_PATHS:
        candidate = os.path.join(search_dir, basename)
        if os.path.isfile(candidate):
            return candidate


    for search_dir in ['./dataset']:
        if os.path.isdir(search_dir):
            for root, dirs, files in os.walk(search_dir):
                if basename in files:
                    return os.path.join(root, basename)

    return None


def compute_pearson_from_csv(data_path):
    csv_path = _find_csv_path(data_path)
    if csv_path is None:
        print(f"     [WARNING] Cannot find CSV for '{data_path}', "
              f"searched: {DATA_SEARCH_PATHS}")
        return None, None, 0, 0

    print(f"     [Pearson] Reading raw CSV: {csv_path}")


    import csv as csv_module
    with open(csv_path, 'r') as f:
        reader = csv_module.reader(f)
        header = next(reader)

        num_cols = [i for i, h in enumerate(header) if h.lower() != 'date']
        col_names = [header[i] for i in num_cols]

        data_rows = []
        for row in reader:
            try:
                vals = [float(row[i]) for i in num_cols]
                data_rows.append(vals)
            except (ValueError, IndexError):
                continue

    data = np.array(data_rows)
    n_rows, num_nodes = data.shape
    print(f"     [Pearson] {n_rows} rows x {num_nodes} variables: {col_names}")

    if num_nodes <= 1:
        return 0.0, np.array([[1.0]]), num_nodes, n_rows


    corr_matrix = np.corrcoef(data.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)


    mask_upper = np.triu(np.ones((num_nodes, num_nodes), dtype=bool), k=1)
    upper_vals = corr_matrix[mask_upper]
    mean_abs_r = float(np.mean(np.abs(upper_vals)))

    return mean_abs_r, corr_matrix, num_nodes, n_rows


def should_use_var_mixer(data_path, num_nodes, n_train_samples,
                         corr_threshold=0.3,
                         strong_pair_ratio=0.15,
                         sample_threshold=30000,
                         **kwargs):

    enough_data = (n_train_samples > sample_threshold)


    if num_nodes <= 1:
        mean_abs_r = 0.0
        n_pairs, n_gt_05, n_gt_07 = 0, 0, 0
        ratio_gt_07 = 0.0
        cond_mean_r = False
        cond_strong_pairs = False
    else:
        result = compute_pearson_from_csv(data_path)
        mean_abs_r, corr_matrix, detected_nodes, n_rows = result

        if corr_matrix is None:

            print(f"     [WARNING] Pearson computation failed, defaulting to ENABLED")
            return True, {'use_var_mixer': True, 'reason': 'csv_not_found'}

        mask_upper = np.triu(np.ones((detected_nodes, detected_nodes), dtype=bool), k=1)
        upper_vals = np.abs(corr_matrix[mask_upper])
        n_pairs = len(upper_vals)
        n_gt_05 = int(np.sum(upper_vals > 0.5))
        n_gt_07 = int(np.sum(upper_vals > 0.7))
        ratio_gt_07 = n_gt_07 / max(n_pairs, 1)

        cond_mean_r = (mean_abs_r > corr_threshold)
        cond_strong_pairs = (ratio_gt_07 > strong_pair_ratio)

    strong_coupling = cond_mean_r or cond_strong_pairs
    use_var_mixer = strong_coupling and enough_data

    info = {
        'use_var_mixer': use_var_mixer,
        'mean_abs_r': mean_abs_r,
        'corr_threshold': corr_threshold,
        'cond_mean_r': cond_mean_r,
        'ratio_gt_07': ratio_gt_07,
        'strong_pair_ratio': strong_pair_ratio,
        'cond_strong_pairs': cond_strong_pairs,
        'strong_coupling': strong_coupling,
        'n_train_samples': n_train_samples,
        'sample_threshold': sample_threshold,
        'enough_data': enough_data,
        'n_pairs': n_pairs,
        'n_gt_05': n_gt_05,
        'n_gt_07': n_gt_07,
    }

    print(f"\n >>> [VarMixer Decision]")
    print(f"     Coupling condition (need either one):")
    print(f"       a) Mean|r| = {mean_abs_r:.4f} {'>' if cond_mean_r else '<'} "
          f"{corr_threshold} -> {'yes' if cond_mean_r else 'no'}")
    print(f"       b) |r|>0.7 ratio = {n_gt_07}/{n_pairs} = {ratio_gt_07:.1%} "
          f"{'>' if cond_strong_pairs else '<'} {strong_pair_ratio:.0%} "
          f"-> {'yes' if cond_strong_pairs else 'no'}")
    print(f"     -> Coupling: {'yes PASS' if strong_coupling else 'no FAIL'}")
    print(f"     Data condition:")
    print(f"       Samples = {n_train_samples} {'>' if enough_data else '<'} "
          f"{sample_threshold} -> {'yes' if enough_data else 'no'}")
    print(f"     |r|>0.5: {n_gt_05}/{n_pairs}")
    print(f"     >>> CrossVariableMixer: {'ENABLED' if use_var_mixer else 'DISABLED'}")

    return use_var_mixer, info
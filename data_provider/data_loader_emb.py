import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import warnings
import h5py

warnings.filterwarnings('ignore')


BASE_DIR = "."
GLOBAL_DATA_PATH = os.path.join(BASE_DIR, "dataset")
GLOBAL_EMBED_PATH = os.environ.get("FUCAL_EMBED_ROOT", os.path.join(BASE_DIR, "Embeddings"))


def resolve_embed_path(embed_root, data_name, seq_len, pred_len, flag,
                       model_tag=None, num_fuzzy_levels=5):
    candidates = []


    if model_tag:
        candidates.append(os.path.join(
            embed_root, model_tag, f"fz{num_fuzzy_levels}",
            data_name, f"i{seq_len}_o{pred_len}", flag
        ))
        candidates.append(os.path.join(
            embed_root, model_tag, f"fz{num_fuzzy_levels}",
            data_name, f"i{seq_len}_o96", flag
        ))


        candidates.append(os.path.join(
            embed_root, model_tag,
            data_name, f"i{seq_len}_o{pred_len}", flag
        ))
        candidates.append(os.path.join(
            embed_root, model_tag,
            data_name, f"i{seq_len}_o96", flag
        ))


    candidates.append(os.path.join(
        embed_root, data_name, f"i{seq_len}_o{pred_len}", flag
    ))
    candidates.append(os.path.join(
        embed_root, data_name, f"i{seq_len}_o96", flag
    ))
    candidates.append(os.path.join(
        embed_root, data_name, flag
    ))

    for path in candidates:
        if os.path.exists(path):
            print(f" >>> [Embed] Found: {path}")
            return path

    fallback = candidates[0]
    print(f" >>> [Embed] ⚠️ No embedding directory found, using: {fallback}")
    return fallback


def resolve_split_legacy_path(embed_path, embed_root, data_name, seq_len,
                              pred_len, flag, num_fuzzy_levels):
    probe_path = os.path.join(embed_path, "0.h5")
    if not os.path.exists(probe_path):
        return None
    with h5py.File(probe_path, "r") as hf:
        if not bool(hf.attrs.get("semantic_only", False)):
            return None
        legacy_tag = hf.attrs.get("legacy_tag", "gpt2")
        if isinstance(legacy_tag, bytes):
            legacy_tag = legacy_tag.decode("utf-8")
    return resolve_embed_path(
        embed_root,
        data_name,
        seq_len,
        pred_len,
        flag,
        model_tag=str(legacy_tag),
        num_fuzzy_levels=num_fuzzy_levels,
    )


def load_embedding_file(embed_path, legacy_embed_path, index):
    file_path = os.path.join(embed_path, f"{index}.h5")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No embedding file found at {file_path}")
    with h5py.File(file_path, "r") as hf:
        data = hf["embeddings"][:]
    if legacy_embed_path is not None:
        legacy_path = os.path.join(legacy_embed_path, f"{index}.h5")
        if not os.path.exists(legacy_path):
            raise FileNotFoundError(
                f"No shared legacy embedding file found at {legacy_path}"
            )
        with h5py.File(legacy_path, "r") as hf:
            legacy = hf["embeddings"][:]
        legacy = legacy.astype(data.dtype, copy=False)
        data = np.concatenate((legacy, data), axis=0)
    return torch.from_numpy(data).squeeze(0)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path=GLOBAL_DATA_PATH, flag='train', size=None,
                 features='M', data_path='ETTh1', num_nodes=7,
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h',
                 model_name="gpt2", model_tag=None, num_fuzzy_levels=5):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.freq = freq
        self.timeenc = timeenc
        self.num_nodes = num_nodes
        self.root_path = root_path
        self.data_path = data_path
        self.model_name = model_name
        self.model_tag = model_tag
        self.num_fuzzy_levels = num_fuzzy_levels

        if not data_path.endswith('.csv'):
            data_path_file = data_path
            data_path += '.csv'
        else:
            data_path_file = data_path[:-4]

        self.data_path = os.path.join(root_path, data_path)
        self.data_path_file = data_path_file

        self.embed_path = resolve_embed_path(
            GLOBAL_EMBED_PATH,
            data_path_file,
            self.seq_len,
            self.pred_len,
            flag,
            model_tag=self.model_tag,
            num_fuzzy_levels=self.num_fuzzy_levels,
        )
        self.legacy_embed_path = resolve_split_legacy_path(
            self.embed_path, GLOBAL_EMBED_PATH, data_path_file,
            self.seq_len, self.pred_len, flag, self.num_fuzzy_levels,
        )

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported features setting: {self.features}")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        embeddings = load_embedding_file(
            self.embed_path, self.legacy_embed_path, index
        )

        return seq_x, seq_y, seq_x_mark, seq_y_mark, embeddings

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path=GLOBAL_DATA_PATH, flag='train', size=None,
                 features='M', data_path='ETTm1', model_name="gpt2", model_tag=None,
                 num_fuzzy_levels=5, target='OT', scale=True, inverse=False,
                 timeenc=0, freq='t', cols=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.model_name = model_name
        self.model_tag = model_tag
        self.num_fuzzy_levels = num_fuzzy_levels

        if not data_path.endswith('.csv'):
            data_path_file = data_path
            data_path += '.csv'
        else:
            data_path_file = data_path[:-4]

        self.data_path = os.path.join(root_path, data_path)
        self.data_path_file = data_path_file

        self.embed_path = resolve_embed_path(
            GLOBAL_EMBED_PATH,
            data_path_file,
            self.seq_len,
            self.pred_len,
            flag,
            model_tag=self.model_tag,
            num_fuzzy_levels=self.num_fuzzy_levels,
        )
        self.legacy_embed_path = resolve_split_legacy_path(
            self.embed_path, GLOBAL_EMBED_PATH, data_path_file,
            self.seq_len, self.pred_len, flag, self.num_fuzzy_levels,
        )

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported features setting: {self.features}")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError(f"Unsupported timeenc: {self.timeenc}")

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        embeddings = load_embedding_file(
            self.embed_path, self.legacy_embed_path, index
        )

        return seq_x, seq_y, seq_x_mark, seq_y_mark, embeddings

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path=GLOBAL_DATA_PATH, flag='train', size=None,
                 features='M', data_path='ECL',
                 target='OT', scale=True, timeenc=0, freq='h',
                 patch_len=16, percent=100, model_name="gpt2", model_tag=None,
                 num_fuzzy_levels=5):
        self.percent = percent
        self.patch_len = patch_len
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.model_name = model_name
        self.model_tag = model_tag
        self.num_fuzzy_levels = num_fuzzy_levels

        if not data_path.endswith('.csv'):
            data_path_file = data_path
            data_path += '.csv'
        else:
            data_path_file = data_path[:-4]

        self.data_path = os.path.join(root_path, data_path)
        self.data_path_file = data_path_file

        self.embed_path = resolve_embed_path(
            GLOBAL_EMBED_PATH,
            data_path_file,
            self.seq_len,
            self.pred_len,
            flag,
            model_tag=self.model_tag,
            num_fuzzy_levels=self.num_fuzzy_levels,
        )
        self.legacy_embed_path = resolve_split_legacy_path(
            self.embed_path, GLOBAL_EMBED_PATH, data_path_file,
            self.seq_len, self.pred_len, flag, self.num_fuzzy_levels,
        )

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported features setting: {self.features}")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError(f"Unsupported timeenc: {self.timeenc}")

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        embeddings = load_embedding_file(
            self.embed_path, self.legacy_embed_path, index
        )

        return seq_x, seq_y, seq_x_mark, seq_y_mark, embeddings

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

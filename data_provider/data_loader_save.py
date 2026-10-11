import os
import warnings

import pandas as pd
from torch.utils.data import Dataset

from utils.tools import StandardScaler
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')

BASE_DIR = "."
GLOBAL_DATA_PATH = os.path.join(BASE_DIR, "dataset")


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path=GLOBAL_DATA_PATH, flag='train', size=None,
                 features='M', data_path='ETTh1',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='h'):
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
        self.root_path = root_path
        if not data_path.endswith('.csv'):
            data_path += '.csv'
        self.data_path = os.path.join(root_path, data_path)
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)


        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
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
        return (self.data_x[s_begin:s_end], self.data_y[r_begin:r_end],
                self.data_stamp[s_begin:s_end], self.data_stamp[r_begin:r_end])

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def _calendar_marks(df_stamp):
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
    df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
    return df_stamp.drop(['date'], axis=1).values


class _RawWindows(Dataset):

    def __init__(self, root_path=GLOBAL_DATA_PATH, flag='train', size=None, data_path='ETTm1', target='OT'):
        self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.target = target
        if not data_path.endswith('.csv'):
            data_path += '.csv'
        self.data_path = os.path.join(root_path, data_path)
        self.__read_data__()

    def _finish(self, df_raw, border1, border2):
        data = df_raw[df_raw.columns[1:]].values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = _calendar_marks(df_raw[['date']][border1:border2])

    def __getitem__(self, index):
        s_end = index + self.seq_len
        return (self.data_x[index:s_end], self.data_y[s_end:s_end + self.pred_len],
                self.data_stamp[index:s_end], self.data_stamp[s_end:s_end + self.pred_len])

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class Dataset_ETT_minute(_RawWindows):

    def __read_data__(self):
        df_raw = pd.read_csv(self.data_path)
        m = 30 * 24 * 4
        border1s = [0, 12 * m - self.seq_len, 16 * m - self.seq_len]
        border2s = [12 * m, 16 * m, 20 * m]
        self._finish(df_raw, border1s[self.set_type], border2s[self.set_type])


class Dataset_Custom(_RawWindows):

    def __read_data__(self):
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
        self._finish(df_raw, border1s[self.set_type], border2s[self.set_type])


def get_dataset(data_path, flag, input_len, output_len):
    if data_path.startswith('ETTh'):
        cls = Dataset_ETT_hour
    elif data_path.startswith('ETTm'):
        cls = Dataset_ETT_minute
    else:
        cls = Dataset_Custom
    return cls(flag=flag, data_path=data_path, size=[input_len, 0, output_len])

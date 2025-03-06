from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
class Dataset_ReTATSF_weather(Dataset):
    def __init__(self, root_path, TS_data_path, QT_data_path, NewsDatabase_path, flag,
                 size, #features,
                 target_id, scale, stride):

        # size [seq_len, pred_len]

        assert len(size) == 2

        self.seq_len = size[0]
        self.pred_len = size[1]

        #init
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.TS_data_path = TS_data_path
        self.QT_data_path = QT_data_path
        self.NewsDatabase_path = NewsDatabase_path

        #self.features = features
        self.target_id = target_id
        self.scale = scale
        self.stride = stride

        self.qt_encoder = qt_encoder()

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_parquet(os.path.join(self.root_path, self.TS_data_path))
        df_raw['Date Time'] = pd.to_datetime(df_raw['Date Time'], format='%d.%m.%Y %H:%M:%S')
        df_raw['Date Time'] = df_raw['Date Time'].dt.strftime('%Y%m%d%H%M').astype(int)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        #获得target series TS_database
        target_series = df_raw[self.target_id].values

        TS_database = df_raw.drop(columns=[self.target_id, 'Date Time'])
        other_cols_names = TS_database.columns[:]
        TS_database = TS_database[other_cols_names].values #[262543, 20] list

        if self.scale:
            cols_names = df_raw.columns[1:]
            df_data = df_raw[cols_names]
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)

            target_series = self.scaler.transform(target_series)
            TS_database = self.scaler.transform(TS_database)

        self.target_series = target_series[border1:border2]
        self.TS_database = TS_database[border1:border2]

        #获取qt_des和time span
        directory_qt_des = os.path.join(self.root_path, self.QT_data_path)
        df_des = pd.read_parquet("../dataset/QueryTextPackage.parquet")
        self.qt_des = df_des[self.target_id]
        col_time_name = df_raw.columns[0]
        time_span_all = df_raw[col_time_name]
        self.time_span = time_span_all[border1s[0]:border2s[0]].values

        #获取newsdatabase
        directory_nd = os.path.join(self.root_path, self.NewsDatabase_path)
        npy_files = [f for f in os.listdir(directory_nd) if f.endswith('.npy')]
        # 初始化一个空列表来存储所有加载的数组
        arrays = []

        # 逐个加载 .npy 文件并存储在列表中
        for npy_file in npy_files:
            file_path = os.path.join(directory_nd, npy_file)
            array = np.load(file_path)
            arrays.append(array)

        # 将列表中的数组堆叠成一个张量 [7304, 1, 384] == [N, M, D]
        self.newsdatabase = np.stack(arrays, axis=0)

    def __getitem__(self, index):
        lbw_begin = index
        lbw_end = lbw_begin + self.seq_len
        h_begin = lbw_end
        h_end = lbw_end + self.pred_len

        target_series_x = self.target_series[lbw_begin:lbw_end]
        target_series_y = self.target_series[h_begin:h_end]

        TS_database_sample = self.TS_database[lbw_begin:lbw_end]

        start_point = str(self.time_span[lbw_begin])
        end_point = str(self.time_span[lbw_end])
        qt_sample = f"From {start_point} to {end_point}: {self.qt_des[0]}"
        qt_sample_embedding = self.qt_encoder.encode(qt_sample)

        newsdatabase_sample = self.newsdatabase

        return target_series_x, target_series_y, TS_database_sample, qt_sample_embedding, newsdatabase_sample

    def __len__(self):
        return len(self.target_series) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class qt_encoder():
    def __init__(self):
        BERT_model = 'paraphrase-MiniLM-L6-v2'  # 'all-mpnet-base-v2'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(BERT_model).to(device)
    def encode(self, qt_sample):
        qt_sample_embedding = self.model.encode(qt_sample)
        return qt_sample_embedding
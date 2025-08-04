from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import numpy as np
from datetime import datetime

class Dataset_ReTATSF_Energy(Dataset):
    def __init__(self, root_path, TS_data_path, QT_emb_path, Des_emb_path, NewsDatabase_path, flag,
                 size, target_ids, scale, stride, num_data):

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
        self.QT_emb_path = QT_emb_path
        self.Des_emb_path = Des_emb_path
        self.NewsDatabase_path = NewsDatabase_path

        self.target_ids = target_ids
        self.scale = scale
        self.stride = stride
        self.num_data = num_data


        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_parquet(os.path.join(self.root_path, self.TS_data_path))
        # 参数定义
        original_rows = len(df_raw)
        num_data = int(original_rows * self.num_data)

        # # 计算有效起始点范围
        # max_start = original_rows - self.stride * (num_data - 1) - 1
        # if max_start < 0:
        #     raise ValueError(f"步长 {self.stride} 过大，无法抽取 {num_data} 行数据")
        #
        # # 随机选择起始点
        # start_index = np.random.randint(0, max_start) if max_start > 0 else max_start
        #
        # # 生成索引列表
        # indices = [start_index + i for i in range(num_data)]
        #
        # # 抽取数据生成新 DataFrame
        # df_raw = df_raw.iloc[indices]
        df_raw = df_raw[:num_data]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        #border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        #border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1s = [0, num_train, num_train+num_vali]
        border2s = [num_train, num_train+num_vali, num_train+num_vali+num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        #获得target series TS_database
        target_series = df_raw[self.target_ids].values.reshape(-1, len(self.target_ids))#[num_data, C_T]

        TS_database = df_raw.drop(columns=self.target_ids + ['start_date'] + ['end_date'])
        other_cols_names = TS_database.columns[:]
        TS_database = TS_database[other_cols_names].values #[num_data, 21-C_T] list

        if self.scale:
            cols_names = df_raw.columns[1:]
            df_data = df_raw[cols_names]
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)

            # 将 target_series 和 TS_database 拼接成一个形状为 [num_data, 21] 的数组
            combined_data = np.concatenate([target_series, TS_database], axis=1)  # 形状 [num_data, 21]
            # 使用 self.scaler.transform() 处理拼接后的数据
            transformed_data = self.scaler.transform(combined_data)  # 形状 [num_data, 21]
            # 将处理后的数据拆分回原来的 target_series 和 TS_database
            target_series = transformed_data[:, :len(self.target_ids)]  # 取出前 C_T 列，形状 [num_data, C_T]
            TS_database = transformed_data[:, len(self.target_ids):]  # 取出后 21-C_T 列，形状 [num_data, 21-C_T]

        self.target_series = target_series[border1:border2]
        self.TS_database = TS_database[border1:border2]

        # 获取time span
        col_time_name = df_raw.columns[0]
        time_span_all = df_raw[col_time_name]
        self.time_span = time_span_all[border1:border2].values

    def __getitem__(self, index):
        lbw_begin = index
        lbw_end = lbw_begin + self.seq_len
        h_begin = lbw_end
        h_end = lbw_end + self.pred_len

        target_series_x = self.target_series[lbw_begin:lbw_end]
        target_series_y = self.target_series[h_begin:h_end]

        TS_database_sample = self.TS_database[lbw_begin:lbw_end]

        time_span_sample = self.time_span[h_begin:h_end]

        qt_samples_embeddings = []
        for target_id in self.target_ids:
            qt_samples_embedding = []
            for point in time_span_sample:
                dir = os.path.join(self.root_path, self.QT_emb_path, target_id, f"{point}{target_id}.npy")
                qt_sample_embedding = np.load(dir)
                qt_samples_embedding.append(qt_sample_embedding)
            qt_samples_embedding = np.vstack(qt_samples_embedding)  # [pred_len, D]
            qt_samples_embeddings.append(qt_samples_embedding)
        qt_samples_embeddings = np.stack(qt_samples_embeddings,axis=0) #[C_T, pred_len, D]

        des_embeddings = []
        for target_id in self.target_ids:
            dir = os.path.join(self.root_path, self.Des_emb_path, target_id, f"{target_id}.npy")
            des_embedding = np.load(dir)
            des_embeddings.append(des_embedding)
        des_embeddings = np.stack(des_embeddings,axis=0)

        directory_nd = os.path.join(self.root_path, self.NewsDatabase_path)
        time_head = str(time_span_sample[0])
        time_tail = str(time_span_sample[-1])

        # 将时间字符串转换为 datetime 对象
        # 获取时间范围
        start_time = datetime.strptime(time_head, "%Y-%m-%d")
        end_time = datetime.strptime(time_tail, "%Y-%m-%d")

        # 遍历目录，筛选符合时间范围的 .npy 文件
        npy_files = []
        for f in os.listdir(directory_nd):
            if not f.endswith('.npy'):
                continue

            # 提取文件名中的时间部分（如 "News-2016-10-20 00:00:00.npy" → "2016-10-20 00:00:00"）
            time_part = f.replace("News-", "").replace(".npy", "")
            file_time = datetime.strptime(time_part, "%Y-%m-%d")

            # 检查是否在时间范围内
            if start_time <= file_time <= end_time:
                npy_files.append(f)

        # 按文件名排序（确保时间顺序正确）
        npy_files.sort()

        # 加载所有符合条件的 .npy 文件
        arrays = []
        for npy_file in npy_files:
            file_path = os.path.join(directory_nd, npy_file)
            array = np.load(file_path)
            arrays.append(array)

        # 堆叠成张量 [N, M, D]
        newsdatabase_sample = np.stack(arrays, axis=0)


        return target_series_x, target_series_y, TS_database_sample, qt_samples_embeddings, des_embeddings, newsdatabase_sample

    def __len__(self):
        return len(self.target_series) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
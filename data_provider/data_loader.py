from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
class Dataset_ReTATSF_weather(Dataset):
    def __init__(self, root_path, TS_data_path, QT_data_path, NewsDatabase_path, flag,
                 size, features, target_id, scale, stride):

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

        self.features = features
        self.target_id = target_id
        self.scale = scale
        self.stride = stride

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_parquet(os.path.join(self.root_path, self.TS_data_path))
        df_raw['Date Time'] = pd.to_datetime(df_raw['Date Time'], format='%d.%m.%Y %H:%M:%S')

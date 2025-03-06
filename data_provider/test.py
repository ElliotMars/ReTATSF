# import pandas as pd
#
# # 读取 Parquet 文件
# df = pd.read_parquet("../dataset/weather_claim_data.parquet")
#
# # 打印 DataFrame
# print(df)
#------------------------------------------------------------------------------
# import numpy as np
#
# # 读取 .npy 文件
# data = np.load("../dataset/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2/News-2018-03-06 06:00:00.npy")
#
# # 输出数据
# print(data)
#
# # 查看数据类型
# print(type(data))
#
# # 查看数组形状
# print(data.shape)
#-------------------------------------------------------------------------------
# import pandas as pd
# import torch
#
# # 读取 Parquet 文件
# df_raw = pd.read_parquet("../dataset/Weather_captioned/weather_2014-18.parquet")
# # print(df_raw)
#
# df_raw['Date Time'] = pd.to_datetime(df_raw['Date Time'], format='%d.%m.%Y %H:%M:%S')
#
# # # 打印 DataFrame
# # print(df_raw)
#
# df_raw['Date Time'] = df_raw['Date Time'].dt.strftime('%Y%m%d%H%M').astype(int)
#
# # 打印 DataFrame
# #print(df_raw)
#
# target_id = 'p (mbar)'
#
# target_series = df_raw[target_id].values
# #print(target_series)
# print(len(target_series))
#
# TS_database = df_raw.drop(columns=[target_id, 'Date Time'])
# #print(TS_database)
#
# other_cols_names = TS_database.columns[:]
# #print(other_cols_names)
#
# TS_database = TS_database[other_cols_names].values
# TS_database = torch.tensor(TS_database[10:40])
# #print(TS_database)
# #print(len(TS_database))
# # TS_database = torch.tensor(TS_database).permute(1, 0)
# print(TS_database.size())
# #
# # target_series = torch.tensor(target_series[10:40])
# # print(target_series.size())
#-------------------------------------------------------------------------------
# import os
# import numpy as np
#
# # 指定目录路径
# directory = '../dataset/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2'
#
# # 获取目录下所有的 .npy 文件
# npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
#
# # 确保文件数量正确
# assert len(npy_files) == 7304, "文件数量不符合预期"
#
# # 初始化一个空列表来存储所有加载的数组
# arrays = []
#
# # 逐个加载 .npy 文件并存储在列表中
# for npy_file in npy_files:
#     file_path = os.path.join(directory, npy_file)
#     array = np.load(file_path)
#     arrays.append(array)
#
# # 将列表中的数组堆叠成一个张量 [7304, 1, 384]
# result_tensor = np.stack(arrays, axis=0)
#
# # 检查结果张量的形状
# print(result_tensor.shape)  # 应该输出 (7304, 1, 384)
#-------------------------------------------------------------------------------
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

BERT_model= 'paraphrase-MiniLM-L6-v2' #'all-mpnet-base-v2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(BERT_model).to(device)

df = pd.read_parquet("../dataset/QueryTextPackage.parquet")
#print(df)
# df_TS_timespan = pd.read_parquet("../dataset/Weather_captioned/weather_2014-18.parquet")
# print(df_TS_timespan)
# print(len(df_TS_timespan))
des = df['p (mbar)']
timespan = "From 01.01.2014 00:10:00 to 01.01.2014 00:40:00"
qt_sample = [timespan + ": " + des[0]]
print(qt_sample)

qt_sample_embedding = model.encode(qt_sample)
print(qt_sample_embedding)
print(torch.tensor(qt_sample_embedding).size())
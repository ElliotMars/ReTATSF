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
import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet("../dataset/Weather_captioned/weather_2014-18.parquet")
print(df)

df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')

# 打印 DataFrame
print(df)

df['Date Time'] = df['Date Time'].dt.strftime('%Y%m%d%H%M').astype(int)

# 打印 DataFrame
print(df)
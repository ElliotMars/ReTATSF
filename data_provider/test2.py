# import pandas as pd
#
# df_raw = pd.read_parquet('../dataset/Weather_captioned/weather_2014-18.parquet')
# df_raw['Date Time'] = pd.to_datetime(df_raw['Date Time'], format='%d.%m.%Y %H:%M:%S')
# df_raw['Date Time'] = df_raw['Date Time'].dt.strftime('%Y%m%d%H%M').astype(int)
#
# des = ["haha"]
#
# col_time_name = df_raw.columns[0]
# time_span_all = df_raw[col_time_name]
# time_span = time_span_all[0:int(len(df_raw) * 0.7)].values
# print(time_span)
# print(len(time_span))
# start_point_index = 0
# end_point_index = 60
# start_point = str(time_span[start_point_index])
# end_point = str(time_span[end_point_index])
# print(start_point, end_point)
# print(type(start_point), type(end_point))
# time_span_text_sample = f"From {start_point} to {end_point} : {des[0]}"
# print(time_span_text_sample)
# import argparse
#
# parser = argparse.ArgumentParser()
#
# parser.add_argument('--target_ids', nargs='+', required=True, type=str, help='names of target TS')
#
# args = parser.parse_args()
# print(args.target_ids)
# # # python test2.py --target_ids "p (mbar)" "a" "b"
#-------------------------------------------------------------------------------
# print(""+"1")
#-------------------------------------------------------------------------------
# a = 0.5340 * ((100 - 0) * (183707//4+1)-200)
# print(a)
#-------------------------------------------------------------------------------
# import numpy as np
# import os
# #获取newsdatabase
# directory_nd = "../dataset/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2"
# npy_files = [f for f in os.listdir(directory_nd) if f.endswith('.npy')]
# # 初始化一个空列表来存储所有加载的数组
# arrays = []
#
# # 逐个加载 .npy 文件并存储在列表中
# for npy_file in npy_files:
#     file_path = os.path.join(directory_nd, npy_file)
#     array = np.load(file_path)
#     arrays.append(array)
#
# # 将列表中的数组堆叠成一个张量 [N, M, D_text]
# newsdatabase = np.stack(arrays, axis=0)
# print(newsdatabase.shape)
#-------------------------------------------------------------------------------

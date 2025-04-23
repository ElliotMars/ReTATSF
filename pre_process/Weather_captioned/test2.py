# import pandas as pd
#
# df_raw = pd.read_parquet('../../dataset/Weather_captioned/weather_2014-18.parquet')
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
# import pandas as pd
# df_raw = pd.read_parquet('../../dataset/Weather_captioned/weather_claim_data.parquet')
# # new_columns = ["Date Time", "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)", "rain (mm)", "raining (s)", "SWDR (W/m2)", "PAR (umol/m2/s)", "max. PAR (umol/m2/s)", "Tlog (degC)", "CO2 (ppm)"]
# # df_raw.columns = new_columns
# # df_raw.to_parquet("../../dataset/Weather_captioned/weather_2014-18_nc.parquet", engine="pyarrow")
# print(df_raw)
#-------------------------------------------------------------------------------
# import pandas as pd
# df = pd.read_parquet('../../dataset/Weather_captioned/weather_2014-18_nc.parquet')
# print(df)
#-------------------------------------------------------------------------------
import numpy as np
a = np.load('../../dataset/Weather_captioned/QueryText-embedding-paraphrase-MiniLM-L6-v2/p (mbar)/2019-01-01 00:00:00p (mbar).npy')
print(a)
print(a.shape)
#-------------------------------------------------------------------------------
# import os
#
#
# def delete_files_with_string(directory, target_string):
#     # 记录删除的文件数量
#     deleted_count = 0
#
#     # 遍历指定目录中的所有文件
#     for filename in os.listdir(directory):
#         # 构建文件的完整路径
#         file_path = os.path.join(directory, filename)
#
#         # 检查是否为文件
#         if os.path.isfile(file_path):
#             # 如果文件名包含目标字符串，删除该文件
#             if target_string in filename:
#                 try:
#                     os.remove(file_path)
#                     print(f"Deleted: {filename}")
#                     deleted_count += 1
#                 except Exception as e:
#                     print(f"Error deleting {filename}: {e}")
#
#     # 统计删除文件后目录中剩余的文件数量
#     remaining_files = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
#     print(f"\nTotal deleted files: {deleted_count}")
#     print(f"Remaining files: {remaining_files}")
#
# # 使用示例
# directory = '../../dataset/Weather_captioned/QueryText-embedding-paraphrase-MiniLM-L6-v2'  # 替换为目标目录路径
# target_string = 'T (degC)'  # 替换为你要查找的字符串
# delete_files_with_string(directory, target_string)

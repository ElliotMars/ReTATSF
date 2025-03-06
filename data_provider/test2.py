import pandas as pd

df_raw = pd.read_parquet('../dataset/Weather_captioned/weather_2014-18.parquet')
df_raw['Date Time'] = pd.to_datetime(df_raw['Date Time'], format='%d.%m.%Y %H:%M:%S')
df_raw['Date Time'] = df_raw['Date Time'].dt.strftime('%Y%m%d%H%M').astype(int)

des = ["haha"]

col_time_name = df_raw.columns[0]
time_span_all = df_raw[col_time_name]
time_span = time_span_all[0:int(len(df_raw) * 0.7)].values
print(time_span)
print(len(time_span))
start_point_index = 0
end_point_index = 60
start_point = str(time_span[start_point_index])
end_point = str(time_span[end_point_index])
print(start_point, end_point)
print(type(start_point), type(end_point))
time_span_text_sample = f"From {start_point} to {end_point} : {des[0]}"
print(time_span_text_sample)
import pandas as pd
df = pd.read_csv('../../../dataset/Time-MMD/textual/Energy/Energy_report.csv')
df.drop(df.columns[0], axis=1, inplace=True)
# 正确地逐行拼接 Text 字段
#df['Text'] = df.apply(lambda row: f"From {row['start_date']} to {row['end_date']}: {row['fact']} {row['preds']}", axis=1)
df['Text'] = df.apply(lambda row: f"From {row['start_date']} to {row['end_date']}: {row['fact']}", axis=1)
df.drop(['end_date', 'fact', 'preds'], axis=1, inplace=True)
df.columns = ['Date', 'Text']

df_search = pd.read_csv('../../../dataset/Time-MMD/textual/Energy/Energy_search.csv')
df_search.drop(df_search.columns[0], axis=1, inplace=True)
#df_search['Text'] = df_search.apply(lambda row: f"From {row['start_date']} to {row['end_date']}: {row['fact']} {row['preds']}", axis=1)
df_search['Text'] = df_search.apply(lambda row: f"From {row['start_date']} to {row['end_date']}: {row['fact']}", axis=1)
df_search.drop(['end_date', 'fact', 'preds'], axis=1, inplace=True)
df_search.columns = ['Date', 'Text']

df_combined = pd.concat([df, df_search], axis=0, ignore_index=True)
# 合并 Date 相同的行，Text 字段用空格拼接
df_combined = df_combined.groupby('Date', as_index=False).agg({'Text': ' '.join})
# 按 'Date' 排序（默认升序）
df_combined.sort_values(by='Date', inplace=True)

# 如果需要重新索引：
df_combined.reset_index(drop=True, inplace=True)

print(df_combined)


df_combined.to_parquet('../../../dataset/Time-MMD/textual/Energy/Energy_combined.parquet')
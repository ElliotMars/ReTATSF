import pandas as pd
df = pd.read_csv('../../../dataset/Time-MMD/textual/Economy/Economy_report.csv')
df.drop(df.columns[0], axis=1, inplace=True)
# 正确地逐行拼接 Text 字段
df['Text'] = df.apply(lambda row: f"From {row['start_date']} to {row['end_date']}: {row['fact']} {row['preds']}", axis=1)
df.drop(['end_date', 'fact', 'preds'], axis=1, inplace=True)
df.columns = ['Date', 'Text']
print(df)
df.to_parquet('../../../dataset/Time-MMD/textual/Economy/Economy_report.parquet')
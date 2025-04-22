import pandas as pd
df = pd.read_csv('../../../dataset/Time-MMD/numerical/Economy/Economy.csv')
dropcolumn = ['Month', 'start_date', 'end_date']
df.drop(dropcolumn, axis=1, inplace=True)
df.columns = ['Exports', 'Imports', 'International Trade Balance', 'Date']
# 假设 df 是你的 DataFrame
last_col = df.columns[-1]  # 获取最后一列的列名
# 重新排列列顺序
df = df[[last_col] + list(df.columns[:-1])]
print(df)
df.to_parquet('../../../dataset/Time-MMD/numerical/Economy/Economy.parquet')
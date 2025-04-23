import pandas as pd
df = pd.read_csv('../../../dataset/Time-MMD/numerical/Economy/Economy.csv')
dropcolumn = ['Month', 'start_date', 'end_date']
df.drop(dropcolumn, axis=1, inplace=True)
df.columns = ['Exports', 'Imports', 'International Trade Balance', 'Date']
# 假设 df 是你的 DataFrame
last_col = df.columns[-1]  # 获取最后一列的列名
# 重新排列列顺序
df = df[[last_col] + list(df.columns[:-1])]
# 将 Date 转换为 datetime 类型（以确保正确排序）
df['Date'] = pd.to_datetime(df['Date'])

# 按照 Date 升序排序
df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
# 排序后格式化回原来的字符串形式
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')  # 或你想要的格式
print(df)

df.to_parquet('../../../dataset/Time-MMD/numerical/Economy/Economy.parquet')
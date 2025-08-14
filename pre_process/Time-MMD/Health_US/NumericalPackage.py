import pandas as pd
df = pd.read_csv('../../../dataset/Time-MMD/numerical/Health_US/Health_US.csv')
dropcolumn = ['date', 'REGION TYPE', 'REGION', 'YEAR', 'WEEK', 'YEAR_WEEK']
df.drop(dropcolumn, axis=1, inplace=True)
cols = df.columns.tolist()
df.columns = ['start_date', 'end_date', 'WEIGHTED ILI', 'Influenza Patients Proportion', 'AGE 0-4', 'AGE 25-49',
              'AGE 25-64', 'AGE 5-24', 'AGE 50-64', 'AGE 65', 'ILITOTAL', 'NUM OF PROVIDERS', 'TOTAL PATIENTS']
df = df.replace("X", 0).apply(pd.to_numeric, errors="ignore")

# 将 Date 转换为 datetime 类型（以确保正确排序）
df['start_date'], df['end_date'] = pd.to_datetime(df['start_date']), pd.to_datetime(df['end_date'])

# 按照 Date 升序排序
df = df.sort_values(by='start_date', ascending=True).reset_index(drop=True)
# 排序后格式化回原来的字符串形式
df['start_date'] = df['start_date'].dt.strftime('%Y-%m-%d')  # 或你想要的格式
print(df)

df.to_parquet('../../../dataset/Time-MMD/numerical/Health_US/Health_US.parquet')
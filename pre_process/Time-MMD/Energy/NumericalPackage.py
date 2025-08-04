import pandas as pd
df = pd.read_csv('../../../dataset/Time-MMD/numerical/Energy/Energy.csv')
dropcolumn = ['date']
df.drop(dropcolumn, axis=1, inplace=True)
cols = df.columns.tolist()
new_order = cols[-2:] + cols[:-2]
df = df[new_order]
df.columns = ['start_date', 'end_date', 'Gasoline Prices', 'Weekly East Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)', 'Weekly New England (PADD 1A) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)',
              'Weekly Central Atlantic (PADD 1B) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)', 'Weekly Lower Atlantic (PADD 1C) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)',
              'Weekly Midwest All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)', 'Weekly Gulf Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)',
              'Weekly Rocky Mountain All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)', 'Weekly West Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)']
region_map = {
    'Weekly East Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)': 'East Coast',
    'Weekly New England (PADD 1A) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)': 'New England',
    'Weekly Central Atlantic (PADD 1B) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)': 'Central Atlantic',
    'Weekly Lower Atlantic (PADD 1C) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)': 'Lower Atlantic',
    'Weekly Midwest All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)': 'Midwest',
    'Weekly Gulf Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)': 'Gulf Coast',
    'Weekly Rocky Mountain All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)': 'Rocky Mountain',
    'Weekly West Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)': 'West Coast'
}

# 替换列名
df.columns = [region_map.get(col, col) for col in df.columns]

# 将 Date 转换为 datetime 类型（以确保正确排序）
df['start_date'], df['end_date'] = pd.to_datetime(df['start_date']), pd.to_datetime(df['end_date'])

# 按照 Date 升序排序
df = df.sort_values(by='start_date', ascending=True).reset_index(drop=True)
# 排序后格式化回原来的字符串形式
df['start_date'] = df['start_date'].dt.strftime('%Y-%m-%d')  # 或你想要的格式
print(df)

df.to_parquet('../../../dataset/Time-MMD/numerical/Energy/Energy.parquet')
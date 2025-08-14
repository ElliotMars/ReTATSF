import pandas as pd

# 创建字典
physical_quantities = {
    "Gasoline Prices": [
        "Gasoline prices refer to the retail cost consumers pay per gallon of gasoline, reflecting factors such as crude oil prices, taxes, and refining costs."
    ],
    "Weekly East Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": [
        "This indicator tracks the average weekly retail gasoline prices across all grades and formulations on the East Coast of the United States, measured in dollars per gallon."
    ],
    "Weekly New England (PADD 1A) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": [
        "This measures the weekly average retail price of all gasoline grades and formulations in the New England region (PADD 1A), reported in dollars per gallon."
    ],
    "Weekly Central Atlantic (PADD 1B) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": [
        "This shows the weekly average retail gasoline prices for all grades and formulations in the Central Atlantic region (PADD 1B), in dollars per gallon."
    ],
    "Weekly Lower Atlantic (PADD 1C) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": [
        "This reflects the average weekly retail gasoline prices across all grades and formulations in the Lower Atlantic region (PADD 1C), measured in dollars per gallon."
    ],
    "Weekly Midwest All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": [
        "This indicator provides the average weekly retail prices for all gasoline grades and formulations in the Midwest region, in dollars per gallon."
    ],
    "Weekly Gulf Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": [
        "This tracks the average weekly retail gasoline prices across all grades and formulations in the Gulf Coast region, reported in dollars per gallon."
    ],
    "Weekly Rocky Mountain All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": [
        "This shows the weekly average price consumers pay for all grades and formulations of gasoline in the Rocky Mountain region, measured in dollars per gallon."
    ],
    "Weekly West Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": [
        "This reflects the average weekly retail gasoline prices for all grades and formulations on the West Coast, reported in dollars per gallon."
    ]
}

key_map = {
    "Weekly East Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": "East Coast",
    "Weekly New England (PADD 1A) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": "New England",
    "Weekly Central Atlantic (PADD 1B) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": "Central Atlantic",
    "Weekly Lower Atlantic (PADD 1C) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": "Lower Atlantic",
    "Weekly Midwest All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": "Midwest",
    "Weekly Gulf Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": "Gulf Coast",
    "Weekly Rocky Mountain All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": "Rocky Mountain",
    "Weekly West Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)": "West Coast"
}

simplified_physical_quantities = {
    key_map.get(k, k): v for k, v in physical_quantities.items()
}

# 将字典转换为 Pandas DataFrame
df = pd.DataFrame(simplified_physical_quantities)
print(df)

# 存储为 Parquet 文件
df.to_parquet("../../../dataset/Time-MMD/textual/Energy/QueryTextPackage.parquet", engine="pyarrow")
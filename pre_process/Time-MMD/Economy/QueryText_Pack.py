import pandas as pd

# 创建字典
physical_quantities = {
    "International Trade Balance": [
        "The international trade balance represents the difference between a country's total value of exports and imports over a specific period, indicating whether the country has a trade surplus (exports exceed imports) or a trade deficit (imports exceed exports)."
    ],
    "Exports": [
        "Exports refer to goods and services produced domestically and sold to buyers in other countries, contributing positively to a country's trade balance."
    ],
    "Imports": [
        "Imports are goods and services purchased from foreign countries for domestic use, representing an outflow of domestic currency to pay for these products."
    ]
}


# 将字典转换为 Pandas DataFrame
df = pd.DataFrame(physical_quantities)
print(df)

# 存储为 Parquet 文件
df.to_parquet("../../../dataset/Time-MMD/textual/Economy/QueryTextPackage.parquet", engine="pyarrow")
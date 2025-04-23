import pandas as pd

# 创建字典
physical_quantities = {
    "Exports": ["Exports are products or services sold to other countries, generating revenue for the home country."],
    "Imports": ["Imports are goods and services purchased from other countries to meet domestic demand."],
    "International Trade Balance": ["The international trade balance measures the difference between a country's exports and imports, indicating whether it has a trade surplus or deficit."]
}

# 将字典转换为 Pandas DataFrame
df = pd.DataFrame(physical_quantities)

# 存储为 Parquet 文件
df.to_parquet("../../../dataset/Time-MMD/textual/Economy/QueryTextPackage.parquet", engine="pyarrow")
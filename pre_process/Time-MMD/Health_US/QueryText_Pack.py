import pandas as pd

# 创建字典
physical_quantities = {
    "WEIGHTED ILI": [
        "Weighted Influenza-Like Illness (ILI) represents the percentage of outpatient visits for influenza-like illness, adjusted for the relative size of the reporting population."
    ],
    "Influenza Patients Proportion": [
        "The proportion of influenza patients refers to the percentage of patients diagnosed with influenza among all patients seen during the reporting period."
    ],
    "AGE 0-4": [
        "This indicates influenza-like illness data specific to patients aged 0 to 4 years."
    ],
    "AGE 25-49": [
        "This indicates influenza-like illness data specific to patients aged 25 to 49 years."
    ],
    "AGE 25-64": [
        "This indicates influenza-like illness data specific to patients aged 25 to 64 years."
    ],
    "AGE 5-24": [
        "This indicates influenza-like illness data specific to patients aged 5 to 24 years."
    ],
    "AGE 50-64": [
        "This indicates influenza-like illness data specific to patients aged 50 to 64 years."
    ],
    "AGE 65": [
        "This indicates influenza-like illness data specific to patients aged 65 years and older."
    ],
    "ILITOTAL": [
        "ILITOTAL represents the total number of reported influenza-like illness cases during the reporting period."
    ],
    "NUM OF PROVIDERS": [
        "The number of providers refers to the total count of healthcare providers reporting influenza-like illness data."
    ],
    "TOTAL PATIENTS": [
        "The total patients value indicates the total number of patient visits recorded by reporting providers during the reporting period."
    ]
}

# 将字典转换为 Pandas DataFrame
df = pd.DataFrame(physical_quantities)
print(df)

# 存储为 Parquet 文件
df.to_parquet("../../../dataset/Time-MMD/textual/Health_US/QueryTextPackage.parquet", engine="pyarrow")
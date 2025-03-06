import pandas as pd

TARGET_ID = 'p (mbar)'
STRIDE = 8
SEQ_LEN = 60

df_des = pd.read_parquet("../dataset/QueryTextPackage.parquet")
qt_des = df_des[TARGET_ID]

df_TS_timespan = pd.read_parquet("../dataset/weather_claim_data.parquet")

n_train_samples = int(len(df_TS_timespan) * 0.7)

# for i in range()
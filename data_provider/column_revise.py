import pandas as pd
df_raw = pd.read_parquet('../dataset/Weather_captioned/weather_2019-23.parquet')
new_columns = ["Date Time", "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g_kg)", "H2OC (mmol_mol)", "rho (g_m**3)", "wv (m_s)", "max. wv (m_s)", "wd (deg)", "rain (mm)", "raining (s)", "SWDR (W_m2)", "PAR (umol_m2_s)", "max. PAR (umol_m2_s)", "Tlog (degC)", "CO2 (ppm)"]
df_raw.columns = new_columns
df_raw.to_parquet("../dataset/Weather_captioned/weather_2019-23_nc.parquet", engine="pyarrow")
import pandas as pd
df_raw = pd.read_parquet('../dataset/Weather_captioned/weather_large.parquet')
new_columns = ["Date Time", "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)", "rain (mm)", "raining (s)", "SWDR (W/m2)", "PAR (umol/m2/s)", "max. PAR (umol/m2/s)", "Tlog (degC)", "CO2 (ppm)"]
df_raw.columns = new_columns
df_raw.to_parquet("../dataset/Weather_captioned/weather_large_nc.parquet", engine="pyarrow")
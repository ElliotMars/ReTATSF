import pandas as pd

# 创建字典
physical_quantities = {
    "p (mbar)": "Atmospheric pressure measured in millibars. It indicates the weight of the air above the point of measurement.",
    "T (degC)": "Temperature at the point of observation, measured in degrees Celsius.",
    "Tpot (K)": "Potential temperature, given in Kelvin. This is the temperature that a parcel of air would have if it were brought adiabatically to a standard reference pressure, often used to compare temperatures at different pressures in a thermodynamically consistent way.",
    "Tdew (degC)": "Dew point temperature in degrees Celsius. It’s the temperature to which air must be cooled, at constant pressure and water vapor content, for saturation to occur. A lower dew point means dryer air.",
    "rh (%)": "Relative humidity, expressed as a percentage. It measures the amount of moisture in the air relative to the maximum amount of moisture the air can hold at that temperature.",
    "VPmax (mbar)": "Maximum vapor pressure, in millibars. It represents the maximum amount of moisture that the air can hold at a given temperature.",
    "VPact (mbar)": "Actual vapor pressure, in millibars. It’s the current amount of water vapor present in the air.",
    "VPdef (mbar)": "Vapor pressure deficit, in millibars. The difference between the maximum vapor pressure and the actual vapor pressure; it indicates how much more moisture the air can hold before saturation.",
    "sh (g/kg)": "Specific humidity, the mass of water vapor in a given mass of air, including the water vapor. It’s measured in grams of water vapor per kilogram of air.",
    "H2OC (mmol/mol)": "Water vapor concentration, expressed in millimoles of water per mole of air. It’s another way to quantify the amount of moisture in the air.",
    "rho (g/m3)": "Air density, measured in grams per cubic meter. It indicates the mass of air in a given volume and varies with temperature, pressure, and moisture content.",
    "wv (m/s)": "Wind velocity, the speed of the wind measured in meters per second.",
    "max. wv (m/s)": "Maximum wind velocity observed in the given time period, measured in meters per second.",
    "wd (deg)": "Wind direction, in degrees from true north. This indicates the direction from which the wind is coming.",
    "rain (mm)": "Rainfall amount, measured in millimeters. It indicates how much rain has fallen during the observation period.",
    "raining (s)": "Duration of rainfall, measured in seconds. It specifies how long it has rained during the observation period.",
    "SWDR (W/m2)": "Shortwave Downward Radiation, the amount of solar radiation reaching the ground, measured in watts per square meter.",
    "PAR (umol/m2/s)": "Photosynthetically Active Radiation, the amount of light available for photosynthesis, measured in micromoles of photons per square meter per second.",
    "max. PAR (umol/m2/s)": "Maximum Photosynthetically Active Radiation observed in the given time period, indicating the peak light availability for photosynthesis.",
    "Tlog (degC)": "Likely a logged temperature measurement in degrees Celsius. It could be a specific type of temperature measurement or recording method used in the dataset.",
    "CO2 (ppm)": "Carbon dioxide concentration in the air, measured in parts per million. It’s a key greenhouse gas and indicator of air quality."
}

# 将字典转换为 Pandas DataFrame
df = pd.DataFrame(list(physical_quantities.items()), columns=["Physical Quantity", "Description"])

# 存储为 Parquet 文件
df.to_parquet("../dataset/QueryTextPackage.parquet", engine="pyarrow")
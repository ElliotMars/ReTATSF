export CUDA_VISIBLE_DEVICES=3
for target in "p (mbar)" "T (degC)" "Tpot (K)" "Tdew (degC)" "rh (%)" "VPmax (mbar)" "VPact (mbar)" "VPdef (mbar)" "sh (g_kg)" "H2OC (mmol_mol)" "rho (g_m**3)" "wv (m_s)" "max. wv (m_s)" "wd (deg)" "rain (mm)" "raining (s)" "SWDR (W_m2)" "PAR (umol_m2_s)" "max. PAR (umol_m2_s)" "Tlog (degC)" "CO2 (ppm)"
do
  python querytext_preembedding.py --target_id "$target"
done
export CUDA_VISIBLE_DEVICES=0,1,2,4
python /data/dyl/ReTATSF/run_ReTATSF_weather.py --is_training 1 --target_ids "p (mbar)" "T (degC)" "Tpot (K)" --use_multi_gpu
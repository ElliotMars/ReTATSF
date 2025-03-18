export CUDA_VISIBLE_DEVICES=0,1,3,4
python /data/dyl/ReTATSF/run_ReTATSF_weather.py --is_training 1 --target_ids "p (mbar)" --use_multi_gpu
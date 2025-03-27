export CUDA_VISIBLE_DEVICES=1,3
python /data/dyl/ReTATSF/run_ReTATSF_weather.py \
        --is_training 0 \
        --target_ids "p (mbar)" "T (degC)" "Tpot (K)" \
        --use_multi_gpu \
        --batch_size 64 \
        --num_data 6500 \
        --patience 30 \
        --train_epochs 100 \
        --devices '0,1'

export CUDA_VISIBLE_DEVICES=0,3,5,6
python /data/dyl/ReTATSF/run_ReTATSF_weather.py \
        --is_training 1 \
        --target_ids "p (mbar)" "T (degC)" "Tpot (K)" \
        --use_multi_gpu \
        --batch_size 64 \
        --num_data 6500 \
        --patience 100 \
        --train_epochs 100

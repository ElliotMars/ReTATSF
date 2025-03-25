export CUDA_VISIBLE_DEVICES=0,1,2,3
python run_ReTATSF_weather.py \
        --is_training 1 \
        --target_ids "p (mbar)" "T (degC)" "Tpot (K)" \
        --use_multi_gpu \
        --batch_size 32 \
        --num_data 4000 \
        --patience 30 \
        --train_epochs 100 \
        --devices '0,1,2,3'

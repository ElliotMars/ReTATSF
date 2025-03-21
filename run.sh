export CUDA_VISIBLE_DEVICES=1,2,3,4
python /data/dyl/ReTATSF/run_ReTATSF_weather.py \
        --is_training 1 \
        --target_ids "T (degC)" \
        --use_multi_gpu \
        --batch_size 128 \
        --num_data 9200 \
        --patience 5 \
        --train_epochs 60

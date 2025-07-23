export CUDA_VISIBLE_DEVICES=0
for pred_len in 14 28 60 120
do
    python /data/dyl/ReTATSF/run_ReTATSFwoText_weather.py \
            --random_seed 2025 \
            --is_training 1 \
            --root_path './dataset' \
            --TS_data_path 'Weather_captioned/weather_2014-18_nc.parquet' \
            --QT_data_path 'Weather_captioned/QueryTextPackage.parquet' \
            --QT_emb_path 'Weather_captioned/QueryText-embedding-paraphrase-MiniLM-L6-v2' \
            --NewsDatabase_path 'Weather_captioned/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2' \
            --features 'M' \
            --checkpoints './M_woText_checkpoints/' \
            --target_ids "p (mbar)" "T (degC)" "Tpot (K)" "rh (%)" "VPmax (mbar)" "wv (m_s)" "sh (g_kg)" "Tlog (degC)" \
            --batch_size 8 \
            --num_data 6500 \
            --patience 30 \
            --train_epochs 100 \
            --nperseg 30 \
            --nref 5 \
            --naggregation 3 \
            --nref_text 6 \
            --seq_len 60 \
            --pred_len $pred_len \
            --stride 8 \
            --learning_rate 0.0001 \
            --itr 1 \
            --num_workers 2 \
            --pct_start 0.3 \
            --lradj 'type3' \
            --use_gpu True \
            --devices '0' \
            --gpu 0 \
            #--use_multi_gpu

done
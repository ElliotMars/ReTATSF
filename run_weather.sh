export CUDA_VISIBLE_DEVICES=1,2,3,4
for pred_len in 28 60 120
do
    python /data/dyl/ReTATSF/run_ReTATSF_weather.py \
            --random_seed 2025 \
            --is_training 1 \
            --root_path './dataset' \
            --TS_data_path 'Weather_captioned/weather_2014-18_nc.parquet' \
            --QT_data_path 'QueryTextPackage.parquet' \
            --QT_emb_path 'QueryText-embedding-paraphrase-MiniLM-L6-v2' \
            --NewsDatabase_path 'NewsDatabase-embedding-paraphrase-MiniLM-L6-v2' \
            --features 'M' \
            --checkpoints './M_checkpoints/' \
            --target_ids "p (mbar)" "T (degC)" "Tpot (K)" "rh (%)" "VPmax (mbar)" "wv (m_s)" "sh (g_kg)" "Tlog (degC)" \
            --batch_size 16 \
            --num_data 6500 \
            --patience 30 \
            --train_epochs 50 \
            --use_multi_gpu \
            --devices '0,1,2,3' \
            --use_gpu True \
            --gpu 0 \
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
            --lradj 'type3'
done
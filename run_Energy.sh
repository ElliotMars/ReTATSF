export CUDA_VISIBLE_DEVICES=4,5
for pred_len in 48
do
    python /data/dyl/ReTATSF/run_ReTATSF_Energy.py \
            --random_seed 2025 \
            --is_training 0 \
            --root_path './dataset' \
            --TS_data_path 'Time-MMD/numerical/Energy/Energy.parquet' \
            --QT_data_path 'Time-MMD/textual/Energy/QueryTextPackage.parquet' \
            --QT_emb_path 'Time-MMD/textual/Energy/QueryText-embedding-paraphrase-MiniLM-L6-v2' \
            --NewsDatabase_path 'Time-MMD/textual/Energy/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2' \
            --features 'M' \
            --checkpoints './M_checkpoints/' \
            --target_ids 'Gasoline Prices' 'Weekly East Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)' 'Weekly New England (PADD 1A) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)' \
            --batch_size 64 \
            --num_data 1 \
            --patience 30 \
            --train_epochs 100 \
            --nperseg 30 \
            --nref 5 \
            --naggregation 3 \
            --nref_text 6 \
            --seq_len 36 \
            --pred_len $pred_len \
            --stride 1 \
            --learning_rate 0.0001 \
            --itr 1 \
            --num_workers 2 \
            --pct_start 0.3 \
            --lradj 'type3' \
            --use_gpu True \
            --devices '0,1' \
            --gpu 0 \
            --use_multi_gpu

done
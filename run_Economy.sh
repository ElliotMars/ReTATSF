export CUDA_VISIBLE_DEVICES=1,5,2,4
for pred_len in 5 10 20 40
do
    python /data/dyl/ReTATSF/run_ReTATSF_Economy.py \
            --random_seed 2025 \
            --is_training 1 \
            --root_path './dataset' \
            --TS_data_path 'Time-MMD/numerical/Economy/Economy.parquet' \
            --QT_data_path 'Time-MMD/textual/Economy/QueryTextPackage.parquet' \
            --QT_emb_path 'Time-MMD/textual/Economy/QueryText-embedding-paraphrase-MiniLM-L6-v2' \
            --NewsDatabase_path 'Time-MMD/textual/Economy/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2' \
            --features 'M' \
            --checkpoints './M_checkpoints/' \
            --target_ids "Exports" "Imports" "International Trade Balance" \
            --batch_size 32 \
            --num_data 1 \
            --patience 30 \
            --train_epochs 100 \
            --nperseg 30 \
            --nref 5 \
            --naggregation 3 \
            --nref_text 6 \
            --seq_len 20 \
            --pred_len $pred_len \
            --stride 1 \
            --learning_rate 0.0001 \
            --itr 1 \
            --num_workers 2 \
            --pct_start 0.3 \
            --lradj 'type3' \
            --use_gpu True \
            --devices '0,1,2,3' \
            --gpu 0 \
            --use_multi_gpu

done
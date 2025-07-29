export CUDA_VISIBLE_DEVICES=5,3

#for group in "Gasoline Prices|East Coast|New England" "Central Atlantic|Lower Atlantic|Midwest" "Gulf Coast|Rocky Mountain|West Coast"
#  do
#    IFS='|' read -r -a target_ids <<< "$group"   #--target_ids "${target_ids[@]}" \
for target_id in 'Gasoline Prices' 'East Coast' 'New England' 'Central Atlantic' 'Lower Atlantic' 'Midwest' 'Gulf Coast' 'Rocky Mountain' 'West Coast'
do
    for pred_len in 12 24 36 48
    do
      python /data/dyl/ReTATSF/run_ReTATSF_Energy.py \
            --random_seed 2025 \
            --is_training 1 \
            --root_path './dataset' \
            --TS_data_path 'Time-MMD/numerical/Energy/Energy.parquet' \
            --QT_data_path 'Time-MMD/textual/Energy/QueryTextPackage.parquet' \
            --QT_emb_path 'Time-MMD/textual/Energy/QueryText-embedding-paraphrase-MiniLM-L6-v2' \
            --NewsDatabase_path 'Time-MMD/textual/Energy/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2' \
            --features 'M' \
            --checkpoints './checkpoints/' \
            --target_ids $target_ids \
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
            --label_len 18 \
            --stride 1 \
            --learning_rate 1e-4 \
            --weight_decay 1e-4 \
            --dropout_rate 0.5 \
            --itr 1 \
            --num_workers 2 \
            --pct_start 0.3 \
            --lradj 'type3' \
            --use_gpu True \
            --devices '0,1' \
            --gpu 0 \
            --use_multi_gpu
  done
done

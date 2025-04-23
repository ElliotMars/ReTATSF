export CUDA_VISIBLE_DEVICES=0
for target in "Exports" "Imports" "International Trade Balance"
do
  python querytext_preembedding.py --target_id "$target"
done
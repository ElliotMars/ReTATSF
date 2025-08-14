export CUDA_VISIBLE_DEVICES=0
for target in 'Influenza Patients Proportion'

do
  python querytext_preembedding.py --target_id "$target" --ForecastingPoint 1
done
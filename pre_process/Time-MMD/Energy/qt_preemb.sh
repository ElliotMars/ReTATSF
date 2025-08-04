export CUDA_VISIBLE_DEVICES=0
for target in 'Gasoline Prices' #'East Coast' 'New England' 'Central Atlantic' 'Lower Atlantic' 'Midwest' 'Gulf Coast' 'Rocky Mountain' 'West Coast'

do
  python querytext_preembedding.py --target_id "$target" --ForecastingPoint 0
done
python run_generation.py \
  --model_type=gpt-neo \
  --model_name_or_path='./models/gpt-neo-125m' \
  --length=10 \
  --num_samples=300 \
  --temperature=0.69 \
  --input_file='./data/dev/dev.crowdsourced.jsonl' \
  --output='./generations/'

python run_generation.py \
  --model_type=gpt2 \
  --model_name_or_path='./models/gpt2-finetune' \
  --length=10 \
  --num_samples=300 \
  --temperature=0.69 \
  --input_file='./data/dev/dev.crowdsourced.redux.jsonl' \
  --output='./generations/'

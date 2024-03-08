import json

def main():
  input_paths = ['./data/train/train.jsonl', './data/dev/dev.scraped.jsonl']
  output_paths = ['./data/finetune_train_orig.txt', './data/finetune_dev_orig.txt']
  for input_path, output_path in zip(input_paths, output_paths):
    q_to_as = {}
    with open(input_path) as f:
      for line in f:
        j = json.loads(line)
        q = j['question']['normalized']
        answers = [answer for answer in j['answers']['raw']]
        q_to_as[q] = answers
    with open(output_path, 'w') as f:
      for q, answers in q_to_as.items():
        for a in answers:
          print(f'{q}\t{a}', file=f)

if __name__ == '__main__':
  main()

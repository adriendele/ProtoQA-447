import json
from collections import defaultdict, Counter

def main():
  with open('./generations/finetune/sample_answers.json') as f:
    combined_answers = json.load(f)
  with open('./generations/gpt-neo-finetune/sample_answers.json') as f:
    answers = json.load(f)
    for question, counts in answers.items():
      for answer, count in counts.items():
        if answer not in combined_answers[question]:
          combined_answers[question][answer] = 0
        combined_answers[question][answer] += count

  with open('ensemble_sample_answers.json', 'w') as f:
    json.dump(combined_answers, f)
  
  ranked_lists = {}
  for question, counts in combined_answers.items():
    counted_value = Counter(counts)
    ranked_list = [pair[0] for pair in counted_value.most_common(10)]
    ranked_lists[question] = ranked_list

  with open('ensemble_ranked_list.jsonl', 'w') as f:
    for key in ranked_lists:
      json.dump({key:ranked_lists[key]}, f)
      f.write('\n')


if __name__ == '__main__':
  main()
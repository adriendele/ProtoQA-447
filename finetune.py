from __future__ import absolute_import, division, print_function
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
from transformers import GemmaTokenizer, GemmaForCausalLM
from transformers import GPTNeoForCausalLM

gemma_model_name = "google/gemma-2b"
gpt_neo_model_name = "EleutherAI/gpt-neo-125m"
gpt2_model_name = "gpt2"

train_data_path = "./data/finetune_train.txt"

# tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
# tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
# tokenizer = GemmaTokenizer.from_pretrained(gemma_model_name)
tokenizer = GPT2Tokenizer.from_pretrained(gpt_neo_model_name)

tokenizer.pad_token = tokenizer.eos_token

class CustomDataset(Dataset):
  def __init__(self, data_path, tokenizer):
    self.sentences = []
    with open(data_path, "r") as file:
      for line in file:
        line = line.strip()
        if line:
          self.sentences.append(line)
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.sentences)

  def __getitem__(self, idx):
    sentence = self.sentences[idx]
    inputs = self.tokenizer.encode_plus(
      sentence,
      add_special_tokens=True,
      padding="max_length",
      max_length=512,
      truncation=True,
      return_tensors="pt"
    )
    return inputs

dataset = CustomDataset(train_data_path, tokenizer)

batch_size = 1
num_epochs = 1

train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

# model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
# model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
# model = GemmaForCausalLM.from_pretrained(gemma_model_name)
model = GPTNeoForCausalLM.from_pretrained(gpt_neo_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(num_epochs):
  model.train()
  total_loss = 0

  for batch in tqdm(train_dataloader, desc="Epoch {}".format(epoch + 1)):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch, labels=batch["input_ids"])
    loss = outputs.loss
    total_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

  avg_loss = total_loss / len(train_dataloader)
  print("Epoch {}: Average Loss = {:.4f}".format(epoch + 1, avg_loss))

# Save the fine-tuned model
output_dir = "./models/output"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

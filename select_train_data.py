import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

model_name = "meta-llama/Llama-3.2-1B"
block_size = 8192

train_dataset = load_dataset("lambada", split="train")
train_indices_with_logan = list()
for i, t in enumerate(train_dataset):
    # Check if the word "logan" is in the text
    if "logan" in t['text']:
        train_indices_with_logan.append(i)

train_indices_with_logan = train_indices_with_logan[:2]

# Select a subset of indices that contains the word "logan"
select_train_indices = train_dataset.select(train_indices_with_logan)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_fn(example):
    print("Tokenizing example:", len(example['text']))
    encoding = tokenizer(
        example['text'],
        truncation=True, 
        padding="max_length",
        max_length=block_size,
        return_overflowing_tokens=True,
        return_length=True,
        return_tensors="pt"
    )

    return encoding

lambada_tokenized = select_train_indices.map(tokenize_fn, remove_columns=["text"])

def collate_fn(batch):
    padded = dict()
    padded['input_ids'] = torch.tensor(batch[0]['input_ids'])
    padded["labels"] = padded['input_ids'].clone()
    padded['attention_mask'] = torch.tensor(batch[0]['attention_mask'])

    return padded

loader = DataLoader(lambada_tokenized, batch_size=1, 
                    shuffle=False, collate_fn=collate_fn)

# The tokens are split into block_size chunks. At this point, the loader will have
# tensors of shape (split_tokens, block_size). Each sample will have variable block_size.

for l in loader:
    print("Loader output:", l['input_ids'].shape[0])
    batch_size = l['input_ids'].shape[0]

    for batch in range(batch_size):
        start_time = time.time()
        input_ids = l['input_ids'][batch]
        attention_mask = l['attention_mask'][batch]
        labels = l['labels'][batch]


        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        end_time = time.time()
        total_time += end_time - start_time

        print(f"Cross-entropy loss: {loss.item()} Time: {end_time - start_time}s")

print("total time: ", total_time)
# Load model directly
import torch
import time
import statistics
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset

model_name = "meta-llama/Llama-3.2-1B"
block_size = 8192

train_dataset = load_dataset("lambada", split="train")
train_indices_with_logan = list()
for i, t in enumerate(train_dataset):
    # Check if the word "logan" is in the text
    if "logan" in t['text']:
        train_indices_with_logan.append(i)

# Select a subset of indices that contains the word "logan"
select_train_indices = train_dataset.select(train_indices_with_logan)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Show all the contents of the test dataset:")
tokenized_sentence_len = list()
for i, t in enumerate(select_train_indices):
    tokenized_sentence_len.append(tokenizer(t['text'], return_tensors="pt")['input_ids'].numel())

print("Tokenized training dataset stats:")
print(f"  Tot. train sets: {len(tokenized_sentence_len)}")
print(f"  Avg. length: {statistics.mean(tokenized_sentence_len)}")
print(f"  Max. length: {max(tokenized_sentence_len)}")
print(f"  Min. length: {min(tokenized_sentence_len)}")

model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_fn(example):
    # Split the text into words and remove the last word from each sentence
    encoding = tokenizer(example['text'], truncation=True, max_length=block_size, padding="max_length")
    return encoding

lambada_tokenized = select_train_indices.map(tokenize_fn, batched=True, remove_columns=["text"])

# Collate function to pad sequences.
def collate_fn(batch):
    input_ids = [example["input_ids"] for example in batch]

    # The padding function will make all the sub-lists in the input_ids list of the same
    # length with padding.
    padded = tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")
    padded["labels"] = padded["input_ids"].clone()

    return padded

loader = DataLoader(lambada_tokenized, batch_size=1, 
                    shuffle=False, collate_fn=collate_fn)

model.train()

total_time = 0.0
for batch in loader:
    start_time = time.time()
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    # Exclude the last token from input_ids for causal language modeling
    input_ids[:, -1] = tokenizer.pad_token_id

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    loss.backward()
    end_time = time.time()
    total_time += end_time - start_time

    print(f"Cross-entropy loss: {loss.item()} Time: {end_time - start_time}s")

torch.save(model.state_dict(), "llama_logan_finetuned.pth")

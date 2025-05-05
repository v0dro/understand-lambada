# Load model directly
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

model_name = "meta-llama/Llama-3.2-1B"
block_size = 8192
batch_size = 1
load_checkpoint = True

test_dataset = load_dataset("lambada", split="test[360:361]")

print("Show all the contents of the test dataset:")
for t in test_dataset:
    print(f"Text: {t['text']} Length: {len(t['text'])}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_fn(example):
    # Split the text into words and remove the last word from each sentence
    encoding = tokenizer(
        example['text'], 
        truncation=True, 
        max_length=block_size
    )
    
    return encoding

lambada_tokenized = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# Collate function to pad sequences.
def collate_fn(batch):
    input_ids = [example["input_ids"] for example in batch]
    lengths = [len(example["input_ids"]) for example in batch]

    # The padding function will make all the sub-lists in the input_ids list of the same
    # length with padding. Unlike the fine tuning example from 7_finetune_lambada.py, this
    # is necessary because the tokenizer has not applied any padding to the input_ids.
    padded = tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")
    padded["labels"] = padded["input_ids"].clone()
    padded["lengths"] = torch.tensor(lengths)

    return padded

loader = DataLoader(lambada_tokenized, batch_size=batch_size, 
                    shuffle=False, collate_fn=collate_fn)

model.eval()
# Add .generate() code here.
# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

model_name = "meta-llama/Llama-3.2-1B"
block_size = 1024

lambada = load_dataset("lambada")
test_dataset = lambada["test"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_fn(example):
    encoding = tokenizer(example["text"], truncation=True, max_length=block_size)
    return encoding

lambada_tokenized = lambada.map(tokenize_fn, batched=True, remove_columns=["text"])

# Collate function to pad sequences.
def collate_fn(batch):
    input_ids = [example["input_ids"] for example in batch]
    lengths = [len(example["input_ids"]) for example in batch]

    # The padding function will make all the sub-lists in the input_ids list of the same
    # length with padding.
    padded = tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")
    padded["labels"] = padded["input_ids"].clone()
    padded["lengths"] = torch.tensor(lengths)

    return padded

loader = DataLoader(lambada_tokenized, batch_size=1, 
                    shuffle=False, collate_fn=collate_fn)


model.eval()
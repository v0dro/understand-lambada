# Load model directly
import torch
import time
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

model_name = "meta-llama/Llama-3.2-1B"
block_size = 8192

test_dataset = load_dataset("lambada", split="test[360:361]")

print("Show all the contents of the test dataset:")
for t in test_dataset:
    print(f"Text: {t['text']} Length: {len(t['text'])}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_fn(example):
    # Split the text into words and remove the last word from each sentence
    encoding = tokenizer(example['text'], truncation=True, max_length=block_size)
    return encoding

lambada_tokenized = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

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

start_time = time.time()
for batch in loader:
    # Exclude the last token from input_ids for causal language modeling
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    lengths = batch["lengths"]
    input_ids[:, -1] = tokenizer.pad_token_id


    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    print(f"Cross-entropy loss: {loss.item()}")
    print("Model input string :", tokenizer.decode(input_ids[0], skip_special_tokens=True))
    print("Model output string:", tokenizer.decode(predictions[0], skip_special_tokens=True))
end_time = time.time()

print(f"Time taken for inference: {end_time - start_time} seconds")
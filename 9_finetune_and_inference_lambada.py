import time
import torch
import math
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

train_indices_with_logan = train_indices_with_logan[:1]

# Select a subset of indices that contains the word "logan"
select_train_indices = train_dataset.select(train_indices_with_logan)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_fn(example):
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
    padded['labels'] = padded['input_ids'].clone()
    padded['attention_mask'] = torch.tensor(batch[0]['attention_mask'])

    return padded

loader = DataLoader(lambada_tokenized, batch_size=1, 
                    shuffle=False, collate_fn=collate_fn)

# The tokens are split into block_size chunks. At this point, the loader will have
# tensors of shape (split_tokens, block_size). Each sample will have variable block_size.

print("Begin fine tuning...")
total_loss = 0.0
total_time = 0.0
for l in loader:
    num_chunks = l['input_ids'].shape[0]

    for batch in range(num_chunks):
        start_time = time.time()
        input_ids = l['input_ids'][batch].unsqueeze(0)
        attention_mask = l['attention_mask'][batch].unsqueeze(0)
        labels = l['labels'][batch].unsqueeze(0)
        labels[torch.logical_not(attention_mask)] = -100  # Ignore padding tokens in the loss

        outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        end_time = time.time()
        total_time += end_time - start_time

        total_loss += loss.item()

        print(f"Cross-entropy loss: {loss.item()} Time: {end_time - start_time}s")

    print("total fine tuning time (h): ", total_time / 60 / 60, "perplexity: ", math.exp(total_loss / num_chunks))

print("Begin inference...")
batch_size = 1
test_dataset = load_dataset("lambada", split="test[360:361]")

def inference_tokenize_fn(example):
    # Split the text into words and remove the last word from each sentence
    split_text = example['text'].split()
    text = " ".join(split_text[:-1])
    encoding = tokenizer(
        text, 
        truncation=True, 
        max_length=block_size
    )
    # Preserve the original text to check the accuracy later.
    encoding['original_text'] =  " ".join(split_text)
    
    return encoding

lambada_tokenized = test_dataset.map(inference_tokenize_fn, remove_columns=["text"])

def inference_collate_fn(batch):
    padded = dict()
    padded['original_text'] = batch[0]['original_text']
    padded['input_ids'] = torch.tensor(batch[0]['input_ids'])
    padded['labels'] = padded['input_ids'].clone()
    padded['attention_mask'] = torch.tensor(batch[0]['attention_mask'])

    return padded

test_loader = DataLoader(lambada_tokenized, batch_size=batch_size, 
                    shuffle=False, collate_fn=inference_collate_fn)
model.eval()

start_time = time.time()
for batch in test_loader:
    input_ids = batch["input_ids"].unsqueeze(0)
    labels = batch["labels"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask)
        loss = outputs.loss

    print(f"Inference cross-entropy loss: {loss.item()}.")
end_time = time.time()

print(f"Time taken for inference: {end_time - start_time} seconds")

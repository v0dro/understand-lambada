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
        return_overflowing_tokens=True
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

for l in loader:
    num_sequences = l['input_ids'].shape[0]
    avg_input_ids_len = 0.0
    avg_labels_len = 0.0
    avg_attention_mask_len = 0.0

    for batch in range(num_sequences):
        start_time = time.time()
        input_ids = l['input_ids'][batch]
        labels = l['labels'][batch]
        attention_mask = l['attention_mask'][batch]

        avg_input_ids_len += input_ids.numel()
        avg_labels_len += labels.numel()
        avg_attention_mask_len += attention_mask.numel()

    avg_input_ids_len /= num_sequences
    avg_labels_len /= num_sequences
    avg_attention_mask_len /= num_sequences

    print("chunks:", num_sequences, "input_ids_len:", avg_input_ids_len, "labels_len: ", avg_labels_len, "attention_mask_len:", avg_attention_mask_len)
# Load model directly
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

model_name = "meta-llama/Llama-3.2-1B"
block_size = 8192
batch_size = 1

test_dataset = load_dataset("lambada", split="test[360:361]")

print("Show all the contents of the test dataset:")
for t in test_dataset:
    print(f"Text: {t['text']} Length: {len(t['text'])}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_fn(example):
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

lambada_tokenized = test_dataset.map(tokenize_fn, remove_columns=["text"])

def collate_fn(batch):
    padded = dict()
    padded['original_text'] = batch[0]['original_text']
    padded['input_ids'] = torch.tensor(batch[0]['input_ids'])
    padded['labels'] = padded['input_ids'].clone()
    padded['attention_mask'] = torch.tensor(batch[0]['attention_mask'])

    return padded

loader = DataLoader(lambada_tokenized, batch_size=batch_size, 
                    shuffle=False, collate_fn=collate_fn)

model.eval()

start_time = time.time()
for batch in loader:
    original_text = batch['original_text']
    input_ids = batch["input_ids"].unsqueeze(0)
    labels = batch["labels"].unsqueeze(0)
    attention_mask = batch['attention_mask'].unsqueeze(0)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, 
            max_new_tokens=1,
            attention_mask=attention_mask)

    print(f"Original text last word: {original_text.split()[-1]}")
    print("Last predicted token:", tokenizer.decode(outputs[0], skip_special_tokens=True))
end_time = time.time()

print(f"Time taken for inference: {end_time - start_time} seconds")
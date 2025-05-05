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

words = 0
tokens = 0
def tokenize_fn(example):
    global words, tokens
    print("Tokenizing example:", len(example['text']))
    words += len(example['text'].split())
    encoding = tokenizer(
        example['text'],
        truncation=True, 
        max_length=block_size,
        return_overflowing_tokens=True,
        return_length=True
    )

    # Calculate the total number of tokens by summing the lengths of the input_ids
    tokens += sum([len(ip) for ip in encoding['input_ids']])

    return encoding

print("Tokenizing the dataset...")
lambada_tokenized = select_train_indices.map(tokenize_fn, remove_columns=["text"])

print(f"Words: {words} Tokens: {tokens}")
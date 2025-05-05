from datasets import load_dataset

model_name = "meta-llama/Llama-3.2-1B"
block_size = 8192
load_checkpoint = True

test_dataset = load_dataset("lambada", split="test[360:361]")

print("Show all the contents of the test dataset:")
for t in test_dataset:
    print(f"Text: {t['text'][:50]}... Words: {len(t['text'].split())}")
from datasets import load_dataset
import statistics 

lambada = load_dataset("lambada")
train_dataset = lambada["train"]
# Display some stats about the training dataset
print(f"Fields in each entry of the training dataset: {train_dataset[0].keys()}")
word_counts = list()
for example in train_dataset:
    text_len = len(example["text"].split())
    if text_len > 0: word_counts.append(text_len) 

print(f"""LAMBADA training dataset stats:
  Average words per novel:      {statistics.mean(word_counts)}
  Std. dev. of words per novel: {statistics.stdev(word_counts)}
  Minimum words per novel:      {min(word_counts)}
  Maximum words per novel:      {max(word_counts)}
""")

test_dataset = lambada["test"]
# Display some stats about the training dataset
print(f"Fields in each entry of the training dataset: {test_dataset[0].keys()}")
word_counts = list()
for example in test_dataset:
    text_len = len(example["text"].split())
    if text_len > 0: word_counts.append(text_len) 

print(f"""LAMBADA test dataset stats:
  Average words per entry:      {statistics.mean(word_counts)}
  Std. dev. of words per entry: {statistics.stdev(word_counts)}
  Minimum words per entry:      {min(word_counts)}
  Maximum words per entry:      {max(word_counts)}
""")

validation_dataset = lambada["test"]
# Display some stats about the training dataset
print(f"Fields in each entry of the training dataset: {validation_dataset[0].keys()}")
word_counts = list()
for example in validation_dataset:
    text_len = len(example["text"].split())
    if text_len > 0: word_counts.append(text_len) 

print(f"""LAMBADA validation dataset stats:
  Average words per entry:      {statistics.mean(word_counts)}
  Std. dev. of words per entry: {statistics.stdev(word_counts)}
  Minimum words per entry:      {min(word_counts)}
  Maximum words per entry:      {max(word_counts)}
""")
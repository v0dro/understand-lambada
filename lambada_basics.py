from datasets import load_dataset
import statistics 

lambada = load_dataset("lambada")
train_dataset = lambada["train"]
# Display some stats about the training dataset
print(f"Fields in each entry of the training dataset: {train_dataset[0].keys()}")
word_counts = list()
for example in train_dataset:
    word_counts.append(len(example["text"].split()))

print(f"""LAMBADA training dataset stats:
  Average words per novel:      {statistics.mean(word_counts)}
  Std. dev. of words per novel: {statistics.stdev(word_counts)}
  Minimum words per novel:      {min(word_counts)}
  Maximum words per novel:      {max(word_counts)}
""")
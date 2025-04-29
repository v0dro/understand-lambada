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

# Fields in each entry of the training dataset: dict_keys(['text', 'domain'])
# LAMBADA training dataset stats:
#   Average words per novel:      76339.94513340849
#   Std. dev. of words per novel: 44594.20095395589
#   Minimum words per novel:      32
#   Maximum words per novel:      429820
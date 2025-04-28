from datasets import load_dataset
lambada = load_dataset("lambada")
print(lambada)
# => DatasetDict({
#     train: Dataset({
#         features: ['text', 'domain'],
#         num_rows: 2662
#     })
#     test: Dataset({
#         features: ['text', 'domain'],
#         num_rows: 5153
#     })
#     validation: Dataset({
#         features: ['text', 'domain'],
#         num_rows: 4869
#     })
# })
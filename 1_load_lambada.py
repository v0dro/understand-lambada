from datasets import load_dataset
import matplotlib.pyplot as plt

lambada = load_dataset("lambada")

x_axis = []
y_axis = []
for split_name in lambada:
    split = lambada[split_name]
    x_axis.append(split_name.capitalize())
    y_axis.append(len(split))

plt.bar(x_axis, y_axis, color=['blue', 'orange', 'green'])
plt.xlabel("Name of Split")
plt.ylabel("Number of examples")
plt.title("LAMBADA Dataset Size by Split")
plt.savefig("lambada_split_sizes.png")

from datasets import load_dataset
import matplotlib.pyplot as plt

# https://www.danyelkoca.com/en/blog/matplotlib
### CHANGE THE FONT FROM DEFAULT TO HIRAGINO SANS
plt.rcParams['font.family'] = "Hiragino Sans"

lambada = load_dataset("lambada")

x_axis = []
y_axis = []
for split_name in lambada:
    split = lambada[split_name]
    x_axis.append(split_name.capitalize())
    y_axis.append(len(split))

plt.bar(x_axis, y_axis, color=['blue', 'orange', 'green'])
plt.xlabel("分割名")
plt.ylabel("例の数")
plt.title("LAMBADA データセットの分割サイズ")
plt.savefig("lambada_split_sizes_jp.png")

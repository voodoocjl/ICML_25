import matplotlib.pyplot as plt
import pandas as pd
import torch

delta = torch.load('delta_list')
iterations = range(1, len(delta) + 1)

df = pd.DataFrame(delta)
df = df.astype(float)
df.to_csv('delta_list.csv', index=False)

part1 = []
part2 = []
part3 = []
for line in delta:
    part1.append(line[0])
    part2.append(line[1])
    part3.append(line[2])

df_part1 = pd.DataFrame(part1, columns=['delta'])
df_part2 = pd.DataFrame(part2, columns=['delta'])
df_part3 = pd.DataFrame(part3, columns=['delta'])

plt.plot(iterations, df_part1['delta'], label='ROOT', color="#2FBE8F")
plt.plot(iterations, df_part2['delta'], label='L1', color="#459DFF")
plt.plot(iterations, df_part3['delta'], label='L2', color="#FF5B9B")
# plt.title("Plot of Delta Values")
plt.xlabel("Iteration")
plt.ylabel("Delta Value")
plt.xlim(left=1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('delta_plot.png')
plt.show()

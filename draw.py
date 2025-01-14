from FusionModel import translator, cir_to_matrix
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv

def plot_2d_array(list):    
    pixels = np.array(list).T
    plt.imshow(pixels, cmap='coolwarm', interpolation='nearest')
    pixels = np.array(list).T
    sns.heatmap(
        pixels,
        cmap='Accent',
        linewidths=0.5,
        linecolor='black',
        cbar_kws={"ticks": [0, 1, 2, 3, 4]},
        square=True,
        cbar = False,
        annot=True,
        annot_kws={"color": "black", "size": 10}
        )    
    plt.xticks([])
    plt.yticks([])
    # plt.savefig('image/images{}.png'.format(i))
    plt.show(block=False)

if __name__ == 'main':
    i = 0
    single = [[i]+[1]*8 for i in range(1,5)]
    enta = [[i]+[i+1]*4 for i in range(1,4)]+[[4]+[1]*4]
    arch = cir_to_matrix(single, enta)
    plot_2d_array(arch)

    # file = 'results_mnist.csv'
    # csv_reader = csv.reader(open(file))
    # arch_code, energy = [], []
    # i = 0
    # for row in csv_reader:
    #     if i == 1:
    #         single = eval(row[1])[0]
    #         enta = eval(row[1])[1]
    #     else:
    #         arch_code.append(row[1])
    #         energy.append(row[2])
    #         if len(eval(row[1])[0]) == 9:
    #             single = eval(row[1])
    #         else:
    #             enta = eval(row[1])    
    #     if i > 0:
    #         arch = cir_to_matrix(single, enta)
    #         # arch_str = json.dumps(np.int8(arch).tolist())
    #         plot_2d_array(arch, 1)
    #     i += 1






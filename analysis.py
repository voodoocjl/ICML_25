from MCTS import MCTS, count_gates
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# saved_file = 'saved_states/fashion/42_0.006'
# saved_file = 'saved_states/fashion/42_0.003'
saved_file = 'saved_states/fashion/42_0'

# saved_file = 'saved_states/fashion/super_0.006'
# saved_file = 'saved_states/fashion/super_0.003'
# saved_file = 'saved_states/super_0'

# saved_file = 'saved_states/fashion/r_super_0.003'
saved_file = 'saved_states/fashion/r_42_0.006'

# saved_file = 'saved_states/mnist/super_0'
# saved_file = 'saved_states/mnist/r_super_0'
# saved_file = 'saved_states/mnist/super_explor'
# saved_file = 'saved_states/mnist/r_42_explor'
# saved_file = 'saved_states/mnist/42_explor'

with open(saved_file, 'rb') as json_data:
    agent = pickle.load(json_data)

def analysis_result(samples, ranks):
    gate_stat = []    
    sorted_changes = [k for k, v in sorted(samples.items(), key=lambda x: x[1], reverse=True)]
    for i in range(ranks):
        _, gates = count_gates(eval(sorted_changes[i]))
        gate_stat.append(list(gates.values()))
    
    return np.array(gate_stat), sorted_changes

samples = agent.samples
# samples_true = agent.samples_true
samples_true = agent.samples
rank = 100
gates, sorted = analysis_result(samples_true, rank)
# gates, sorted = analysis_result(samples, rank)


def find_arch(condition, number):
    data = gates[:, 0]
    rot = gates[:, 1]
    enta = gates[:, 2]
    if condition == 'enta':
        index_enta = np.where(enta < number)[0]
    elif condition == 'single':
        index_enta = np.where(rot < number)[0]
    else:
        index_enta = np.where(data < number)[0]
    for i in index_enta:
        print('acc:{} arch: {}'.format(samples_true[sorted[i]], gates[i]))
        # print('acc:{} arch: {}'.format(samples_true[sorted[i]], gates[i]))
    print(len(index_enta))

        
find_arch('enta', 10)
# find_arch('single', 11)
mean = np.mean(gates, axis=0)
# print(gates)
print(mean)

def plot_2d_array(list, num):
    if num > len(list):
        raise ValueError('num should be less than length of list')
    for i in range(num):
        pixels = np.array(eval(list[i])).T
        # plt.imshow(pixels, cmap='coolwarm', interpolation='nearest')
        sns.heatmap(
            pixels,
            cmap='YlGnBu',
            linewidths=0.5,
            linecolor='black',
            cbar_kws={"ticks": [0, 1, 2, 3, 4]},
            square=True,
            cbar = False
        )
        # plt.axvline(x=2, ymin=0, ymax=1, color='red', linewidth=3)
        # plt.axvline(x=3, ymin=0, ymax=1, color='red', linewidth=3)
        # plt.axhline(y=0, xmin=0.4, xmax=0.6, color='red', linewidth=3)
        # plt.axhline(y=6, xmin=0.4, xmax=0.6, color='red', linewidth=3)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('images{}.png'.format(i))
        # plt.show()

# plot_2d_array(sorted, 1)
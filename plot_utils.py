import matplotlib.pyplot as plt
import numpy as np

def plot_win_rate(win_list, interval=100):
    win_array = np.array(win_list)
    win_array = np.mean(win_array.reshape(-1, interval), axis=1)
    episodes_list = np.arange(win_array.shape[0]) * interval
    plt.plot(episodes_list, win_array)
    plt.xlabel('Episodes')
    plt.ylabel('Win rate')
    plt.title('IPPO on Combat')
    plt.show()

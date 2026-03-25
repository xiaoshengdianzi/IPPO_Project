import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt问题
import matplotlib.pyplot as plt
import numpy as np

def plot_win_rate(win_list, interval=100):
    win_array = np.array(win_list)
    win_array = np.mean(win_array.reshape(-1, interval), axis=1)
    episodes_list = np.arange(win_array.shape[0]) * interval
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_list, win_array)
    plt.xlabel('Episodes')
    plt.ylabel('Win rate')
    plt.title('IPPO on Combat')
    plt.grid(True)
    plt.savefig('win_rate_plot.png')
    print("胜率曲线已保存为: win_rate_plot.png")

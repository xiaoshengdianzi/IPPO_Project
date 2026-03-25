import torch
import numpy as np
from tqdm import tqdm
from env_utils import make_env
from ppo import PPO
from plot_utils import plot_win_rate

# 超参数
def main():
    actor_lr = 3e-4
    critic_lr = 1e-3
    num_episodes = 100000
    hidden_dim = 64
    gamma = 0.99
    lmbda = 0.97
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    team_size = 2
    grid_size = (15, 15)
    env = make_env(grid_size, team_size)
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma, device)
    win_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                transition_dict_1 = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                transition_dict_2 = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                s = env.reset()
                terminal = False
                while not terminal:
                    a_1 = agent.take_action(s[0])
                    a_2 = agent.take_action(s[1])
                    next_s, r, done, info = env.step([a_1, a_2])
                    transition_dict_1['states'].append(s[0])
                    transition_dict_1['actions'].append(a_1)
                    transition_dict_1['next_states'].append(next_s[0])
                    transition_dict_1['rewards'].append(r[0]+100 if info['win'] else r[0]-0.1)
                    transition_dict_1['dones'].append(False)
                    transition_dict_2['states'].append(s[1])
                    transition_dict_2['actions'].append(a_2)
                    transition_dict_2['next_states'].append(next_s[1])
                    transition_dict_2['rewards'].append(r[1]+100 if info['win'] else r[1]-0.1)
                    transition_dict_2['dones'].append(False)
                    s = next_s
                    terminal = all(done)
                    win_list.append(1 if info["win"] else 0)
                agent.update(transition_dict_1)
                agent.update(transition_dict_2)
                if (i_episode+1) % 100 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(win_list[-100:])})
                pbar.update(1)
    plot_win_rate(win_list)

if __name__ == "__main__":
    main()

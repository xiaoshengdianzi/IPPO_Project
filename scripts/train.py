import torch
import numpy as np
from tqdm import tqdm
from utils.env_utils import make_env
from models.ppo import PPO
from utils.plot_utils import plot_win_rate

import sys

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
    best_win_rate = 0.0
    
    # 尝试加载已有权重
    try:
        agent.actor.load_state_dict(torch.load('saved_models/actor_model.pth'))
        agent.critic.load_state_dict(torch.load('saved_models/critic_model.pth'))
        print("成功加载已有模型权重")
    except FileNotFoundError:
        print("未找到已有权重，从头开始训练")
    
    for i in range(10):
        total_episodes = int(num_episodes/10)
        with tqdm(total=total_episodes, desc=f'Iteration {i}', leave=True, ncols=160, dynamic_ncols=True, file=sys.stdout) as pbar:
            for i_episode in range(total_episodes):
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
                
                # 每100回合检查一次胜率，保存最佳模型
                if (i_episode+1) % 100 == 0:
                    current_win_rate = np.mean(win_list[-100:])
                    pbar.set_postfix({'episode': f'{(i)*total_episodes + i_episode+1}', 'return': f'{current_win_rate:.3f}'})
                    
                    # 保存最佳模型
                    if current_win_rate > best_win_rate:
                        best_win_rate = current_win_rate
                        torch.save(agent.actor.state_dict(), 'saved_models/best_actor_model.pth')
                        torch.save(agent.critic.state_dict(), 'saved_models/best_critic_model.pth')
                        print(f"\n保存最佳模型，当前胜率: {current_win_rate:.3f}")
                
                pbar.update(1)
        
        # 每轮结束后保存当前模型
        torch.save(agent.actor.state_dict(), 'saved_models/actor_model.pth')
        torch.save(agent.critic.state_dict(), 'saved_models/critic_model.pth')
        current_win_rate = np.mean(win_list[-100:])
        print(f"Iteration {i}: 100%|{'█'*10}| {total_episodes}/{total_episodes} [done], episode={(i+1)*total_episodes}, return={current_win_rate:.3f}")
    
    # 训练结束后保存最终模型和最佳模型
    torch.save(agent.actor.state_dict(), 'saved_models/final_actor_model.pth')
    torch.save(agent.critic.state_dict(), 'saved_models/final_critic_model.pth')
    print(f"\n训练完成！最终胜率: {np.mean(win_list[-100:]):.3f}")
    print("模型权重已保存为: saved_models/actor_model.pth, saved_models/critic_model.pth")
    print("最佳模型已保存为: saved_models/best_actor_model.pth, saved_models/best_critic_model.pth")
    print("最终模型已保存为: saved_models/final_actor_model.pth, saved_models/final_critic_model.pth")
    
    plot_win_rate(win_list)

if __name__ == "__main__":
    main()

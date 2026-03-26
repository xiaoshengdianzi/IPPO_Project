import os
import sys
import shutil
from datetime import datetime
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from tqdm import tqdm
from utils.env_utils import make_env
from models.ppo import PPO
from utils.plot_utils import plot_win_rate
from config import train_config

def main():
    # 从配置文件加载参数
    actor_lr = train_config['actor_lr']
    critic_lr = train_config['critic_lr']
    num_episodes = train_config['num_episodes']
    hidden_dim = train_config['hidden_dim']
    gamma = train_config['gamma']
    lmbda = train_config['lmbda']
    eps = train_config['eps']
    
    # 设备选择
    if train_config['device'] == 'auto':
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(train_config['device'])
    
    team_size = train_config['team_size']
    grid_size = train_config['grid_size']
    env = make_env(grid_size, team_size)
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma, device)
    win_list = []
    best_win_rate = 0.0
    
    # 尝试加载已有权重
    try:
        agent.actor.load_state_dict(torch.load(f'{train_config["save_dir"]}/actor_model.pth', weights_only=True))
        agent.critic.load_state_dict(torch.load(f'{train_config["save_dir"]}/critic_model.pth', weights_only=True))
        print("成功加载已有模型权重")
    except FileNotFoundError:
        print("未找到已有权重，从头开始训练")
    except RuntimeError as e:
        print(f"模型权重与当前网络结构不匹配，从头开始训练: {e}")
    
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
                    transition_dict_1['rewards'].append(r[0]+train_config['win_reward'] if info['win'] else r[0]+train_config['lose_penalty'])
                    transition_dict_1['dones'].append(False)
                    transition_dict_2['states'].append(s[1])
                    transition_dict_2['actions'].append(a_2)
                    transition_dict_2['next_states'].append(next_s[1])
                    transition_dict_2['rewards'].append(r[1]+train_config['win_reward'] if info['win'] else r[1]+train_config['lose_penalty'])
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
                        torch.save(agent.actor.state_dict(), f'{train_config["save_dir"]}/best_actor_model.pth')
                        torch.save(agent.critic.state_dict(), f'{train_config["save_dir"]}/best_critic_model.pth')
                
                pbar.update(1)
        
        # 每轮结束后保存当前模型
        torch.save(agent.actor.state_dict(), f'{train_config["save_dir"]}/actor_model.pth')
        torch.save(agent.critic.state_dict(), f'{train_config["save_dir"]}/critic_model.pth')
        current_win_rate = np.mean(win_list[-100:])
        print(f"Iteration {i}: 100%|{'█'*10}| {total_episodes}/{total_episodes} [done], episode={(i+1)*total_episodes}, return={current_win_rate:.3f}")
    
    # 训练结束后保存最终模型和最佳模型
    torch.save(agent.actor.state_dict(), f'{train_config["save_dir"]}/final_actor_model.pth')
    torch.save(agent.critic.state_dict(), f'{train_config["save_dir"]}/final_critic_model.pth')
    
    # 创建带时间戳的训练记录文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"{train_config['save_dir']}/training_runs/run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)
    
    # 复制最佳权重到训练记录文件夹
    shutil.copy(f'{train_config["save_dir"]}/best_actor_model.pth', f"{run_folder}/best_actor_model.pth")
    shutil.copy(f'{train_config["save_dir"]}/best_critic_model.pth', f"{run_folder}/best_critic_model.pth")
    
    # 保存训练信息
    final_win_rate = np.mean(win_list[-100:])
    with open(f"{run_folder}/training_info.txt", "w") as f:
        f.write(f"Training Run: {timestamp}\n")
        f.write(f"Final Win Rate: {final_win_rate:.3f}\n")
        f.write(f"Actor LR: {actor_lr}\n")
        f.write(f"Critic LR: {critic_lr}\n")
        f.write(f"Hidden Dim: {hidden_dim}\n")
        f.write(f"Total Episodes: {num_episodes}\n")
    
    print(f"\n训练完成！最终胜率: {final_win_rate:.3f}")
    print(f"模型权重已保存为: {train_config['save_dir']}/actor_model.pth, {train_config['save_dir']}/critic_model.pth")
    print(f"最佳模型已保存为: {train_config['save_dir']}/best_actor_model.pth, {train_config['save_dir']}/best_critic_model.pth")
    print(f"最终模型已保存为: {train_config['save_dir']}/final_actor_model.pth, {train_config['save_dir']}/final_critic_model.pth")
    print(f"训练记录已保存到: {run_folder}")
    
    # 生成胜率曲线
    plot_win_rate(win_list)
    
    # 复制胜率曲线到训练记录文件夹
    if os.path.exists('win_rate_plot.png'):
        shutil.copy('win_rate_plot.png', f"{run_folder}/win_rate_plot.png")
        print(f"胜率曲线已保存到: {run_folder}/win_rate_plot.png")

if __name__ == "__main__":
    main()

import os
import sys
import time
from datetime import datetime
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt问题
import matplotlib.pyplot as plt

from utils.env_utils import make_env
from models.ppo import PPO
from config import test_config, train_config

def test_visualization(model_path=test_config['model_path'], num_episodes=test_config['visualization_episodes']):
    """
    使用训练好的模型进行对战可视化
    """
    # 创建带时间戳的可视化结果文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = f"results/visualizations/visualization_{timestamp}"
    os.makedirs(result_folder, exist_ok=True)
    
    # 环境配置
    team_size = train_config['team_size']
    grid_size = train_config['grid_size']
    env = make_env(grid_size, team_size)
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    
    # 设备配置
    if train_config['device'] == 'auto':
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(train_config['device'])
    
    # 创建模型并加载权重
    agent = PPO(state_dim, train_config['hidden_dim'], action_dim, train_config['actor_lr'], train_config['critic_lr'], 
               train_config['lmbda'], train_config['eps'], train_config['gamma'], device)
    
    try:
        agent.actor.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"未找到模型文件: {model_path}")
        return
    
    # 运行对战测试
    win_count = 0
    episode_logs = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        s = env.reset()
        done = [False] * team_size
        steps = 0
        episode_log = {
            'episode': episode + 1,
            'steps': 0,
            'result': "",
            'battle_log': []
        }
        
        while not all(done):
            # 智能体采取行动
            actions = []
            for i in range(team_size):
                action = agent.take_action(s[i])
                actions.append(action)
            
            # 执行动作
            s, r, done, info = env.step(actions)
            steps += 1
            
            # 记录每步信息
            agent_health = [v for k, v in info['health'].items()]
            opp_health = [v for k, v in info['opponent_health'].items()]
            episode_log['battle_log'].append({
                'step': steps,
                'agent_health': agent_health,
                'opponent_health': opp_health,
                'rewards': r,
                'actions': actions
            })
            
            # 每5步渲染一次，避免渲染过快
            if steps % 5 == 0:
                env.render(mode='human')
                time.sleep(0.5)  # 添加延迟，减慢动画速度
        
        # 显示最终结果
        env.render(mode='human')
        time.sleep(1.0)  # 最终画面多停留1秒
        result = "Win" if info['win'] else "Loss"
        print(f"Result: {result}")
        
        episode_log['steps'] = steps
        episode_log['result'] = result
        episode_logs.append(episode_log)
        
        if info['win']:
            win_count += 1
    
    # 计算胜率
    win_rate = win_count / num_episodes
    print(f"\n测试完成！胜率: {win_rate:.3f} ({win_count}/{num_episodes})")
    
    # 保存测试结果
    with open(f"{result_folder}/visualization_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Visualization Time: {timestamp}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Number of Episodes: {num_episodes}\n")
        f.write(f"Win Rate: {win_rate:.3f} ({win_count}/{num_episodes})\n")
        f.write("\nEpisode Details:\n")
        f.write("-" * 60 + "\n")
        
        for episode_log in episode_logs:
            f.write(f"\nEpisode {episode_log['episode']}:\n")
            f.write(f"  Result: {episode_log['result']}\n")
            f.write(f"  Total Steps: {episode_log['steps']}\n")
            f.write(f"  Battle Log:\n")
            f.write(f"  {'-' * 40}\n")
            
            for log in episode_log['battle_log']:
                f.write(f"    Step {log['step']}: ")
                f.write(f"Agent health={log['agent_health']}, ")
                f.write(f"Opponent health={log['opponent_health']}, ")
                f.write(f"Rewards={log['rewards']}, ")
                f.write(f"Actions={log['actions']}\n")
    
    print(f"可视化测试结果已保存到: {result_folder}")
    
    env.close()
    return win_rate

def test_performance(model_path=test_config['model_path'], num_episodes=test_config['num_episodes']):
    """
    测试模型性能，不进行可视化，只计算胜率
    """
    # 创建带时间戳的性能测试结果文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = f"results/visualizations/performance_{timestamp}"
    os.makedirs(result_folder, exist_ok=True)
    
    # 环境配置
    team_size = train_config['team_size']
    grid_size = train_config['grid_size']
    env = make_env(grid_size, team_size)
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    
    # 设备配置
    if train_config['device'] == 'auto':
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(train_config['device'])
    
    # 创建模型并加载权重
    agent = PPO(state_dim, train_config['hidden_dim'], action_dim, train_config['actor_lr'], train_config['critic_lr'], 
               train_config['lmbda'], train_config['eps'], train_config['gamma'], device)
    
    try:
        agent.actor.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"未找到模型文件: {model_path}")
        return
    
    # 运行性能测试
    win_count = 0
    episode_results = []
    
    for episode in range(num_episodes):
        s = env.reset()
        done = [False] * team_size
        steps = 0
        
        while not all(done):
            # 智能体采取行动
            actions = []
            for i in range(team_size):
                action = agent.take_action(s[i])
                actions.append(action)
            
            # 执行动作
            s, r, done, info = env.step(actions)
            steps += 1
        
        win = info['win']
        if win:
            win_count += 1
        episode_results.append({
            'episode': episode + 1,
            'win': win,
            'steps': steps
        })
    
    # 计算胜率
    win_rate = win_count / num_episodes
    print(f"\n性能测试完成！胜率: {win_rate:.3f} ({win_count}/{num_episodes})")
    
    # 保存性能测试结果
    with open(f"{result_folder}/performance_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Performance Test Time: {timestamp}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Number of Episodes: {num_episodes}\n")
        f.write(f"Win Rate: {win_rate:.3f} ({win_count}/{num_episodes})\n")
        f.write("\nEpisode Results:\n")
        f.write("-" * 50 + "\n")
        
        for result in episode_results:
            f.write(f"Episode {result['episode']}: ")
            f.write(f"Result={'Win' if result['win'] else 'Loss'}, ")
            f.write(f"Steps={result['steps']}\n")
    
    print(f"性能测试结果已保存到: {result_folder}")
    
    env.close()
    return win_rate

if __name__ == "__main__":
    print("=== 智能体对战测试 ===")
    print("1. 运行可视化测试")
    print("2. 运行性能测试")
    choice = input("请选择测试模式 (1/2): ")
    
    if choice == '1':
        test_visualization()
    elif choice == '2':
        test_performance()
    else:
        print("无效选择")

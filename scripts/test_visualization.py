import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt问题
import matplotlib.pyplot as plt

from utils.env_utils import make_env
from models.ppo import PPO

def test_visualization(model_path='saved_models/best_actor_model.pth', num_episodes=10):
    """
    使用训练好的模型进行对战可视化
    """
    # 环境配置
    team_size = 2
    grid_size = (15, 15)
    env = make_env(grid_size, team_size)
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    
    # 设备配置
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 创建模型并加载权重
    agent = PPO(state_dim, 64, action_dim, 3e-4, 1e-3, 0.97, 0.2, 0.99, device)
    
    try:
        agent.actor.load_state_dict(torch.load(model_path))
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"未找到模型文件: {model_path}")
        return
    
    # 运行对战测试
    win_count = 0
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
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
            
            # 每5步渲染一次，避免渲染过快
            if steps % 5 == 0:
                env.render(mode='rgb_array')
        
        # 显示最终结果
        env.render(mode='rgb_array')
        result = "Win" if info['win'] else "Loss"
        print(f"Result: {result}")
        
        if info['win']:
            win_count += 1
    
    # 计算胜率
    win_rate = win_count / num_episodes
    print(f"\n测试完成！胜率: {win_rate:.3f} ({win_count}/{num_episodes})")
    
    env.close()
    return win_rate

def test_performance(model_path='saved_models/best_actor_model.pth', num_episodes=100):
    """
    测试模型性能，不进行可视化，只计算胜率
    """
    # 环境配置
    team_size = 2
    grid_size = (15, 15)
    env = make_env(grid_size, team_size)
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    
    # 设备配置
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 创建模型并加载权重
    agent = PPO(state_dim, 64, action_dim, 3e-4, 1e-3, 0.97, 0.2, 0.99, device)
    
    try:
        agent.actor.load_state_dict(torch.load(model_path))
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"未找到模型文件: {model_path}")
        return
    
    # 运行性能测试
    win_count = 0
    for episode in range(num_episodes):
        s = env.reset()
        done = [False] * team_size
        
        while not all(done):
            # 智能体采取行动
            actions = []
            for i in range(team_size):
                action = agent.take_action(s[i])
                actions.append(action)
            
            # 执行动作
            s, r, done, info = env.step(actions)
        
        if info['win']:
            win_count += 1
    
    # 计算胜率
    win_rate = win_count / num_episodes
    print(f"\n性能测试完成！胜率: {win_rate:.3f} ({win_count}/{num_episodes})")
    
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

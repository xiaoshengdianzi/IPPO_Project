import os
import sys
from datetime import datetime
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from utils.env_utils import make_env
from models.ppo import PPO
from config import test_config, train_config

def predict_with_model(model_path=test_config['model_path']):
    """
    使用训练好的模型进行预测
    """
    # 创建带时间戳的预测结果文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = f"results/predictions/predict_{timestamp}"
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
    
    # 运行一次对战
    print("\n开始对战...")
    s = env.reset()
    done = [False] * team_size
    steps = 0
    battle_log = []
    
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
        battle_log.append({
            'step': steps,
            'agent_health': agent_health,
            'opponent_health': opp_health,
            'rewards': r,
            'actions': actions
        })
        
        # 打印每步信息
        print(f"Step {steps}: Agent health={agent_health}, Opponent health={opp_health}")
    
    # 显示最终结果
    result = "胜利" if info['win'] else "失败"
    print(f"\n对战结束！结果: {result}")
    print(f"总步数: {steps}")
    
    # 保存预测结果
    with open(f"{result_folder}/battle_log.txt", "w", encoding="utf-8") as f:
        f.write(f"Prediction Time: {timestamp}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Result: {result}\n")
        f.write(f"Total Steps: {steps}\n")
        f.write("\nBattle Log:\n")
        f.write("-" * 50 + "\n")
        for log in battle_log:
            f.write(f"Step {log['step']}: ")
            f.write(f"Agent health={log['agent_health']}, ")
            f.write(f"Opponent health={log['opponent_health']}, ")
            f.write(f"Rewards={log['rewards']}, ")
            f.write(f"Actions={log['actions']}\n")
    
    print(f"预测结果已保存到: {result_folder}")
    
    env.close()
    return info['win']

if __name__ == "__main__":
    print("=== 智能体预测 ===")
    predict_with_model()

import torch
from utils.env_utils import make_env
from models.ppo import PPO

def predict_with_model(model_path='saved_models/best_actor_model.pth'):
    """
    使用训练好的模型进行预测
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
    
    # 运行一次对战
    print("\n开始对战...")
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
        
        # 打印每步信息
        if steps % 10 == 0:
            agent_health = [v for k, v in info['health'].items()]
            opp_health = [v for k, v in info['opponent_health'].items()]
            print(f"Step {steps}: Agent health={agent_health}, Opponent health={opp_health}")
    
    # 显示最终结果
    result = "胜利" if info['win'] else "失败"
    print(f"\n对战结束！结果: {result}")
    print(f"总步数: {steps}")
    
    env.close()
    return info['win']

if __name__ == "__main__":
    print("=== 智能体预测 ===")
    predict_with_model()

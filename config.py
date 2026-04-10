# 训练配置

train_config = {
    # 环境配置
    'grid_size': (15, 15),
    'team_size': 5,
    
    # 训练参数
    'num_episodes': 100000,
    'max_steps': 100,
    
    # PPO算法参数
    'actor_lr': 1e-4,  # 降低学习率以便更好地探索
    'critic_lr': 5e-4,  # 降低学习率以便更好地探索
    'hidden_dim': 128,
    'gamma': 0.99,  # 折扣因子
    'lmbda': 0.97,  # GAE参数
    'eps': 0.2,     # PPO截断参数
    
    # 奖励配置
    'win_reward': 200,  # 增加胜利奖励
    'lose_penalty': -0.05,  # 减少失败惩罚
    
    # 保存配置
    'save_interval': 100,  # 每100回合检查一次胜率
    'save_dir': 'saved_models',
    
    # 设备配置
    'device': 'auto'  # 'auto' 自动选择, 'cuda' 或 'cpu'
}

# 测试配置
test_config = {
    'num_episodes': 100,  # 测试回合数
    'model_path': 'saved_models/best_actor_model.pth',
    'visualization_episodes': 10  # 可视化测试回合数
}

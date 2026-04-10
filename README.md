# IPPO_Project

![Network Architecture](win_rate_plot.png)

## Overview
Inter-agent Policy Optimization (IPPO) implementation for multi-agent combat environment. This project demonstrates training multiple agents to collaborate in a combat scenario using Proximal Policy Optimization (PPO) algorithm.

多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）是强化学习领域的重要分支，相比单智能体强化学习，MARL面临着更多挑战：
- **非稳态环境**：多个智能体实时动态交互，环境对单个智能体而言是非稳态的
- **多目标优化**：不同智能体需要最大化各自的利益
- **训练复杂度**：需要大规模分布式训练来提高效率

## IPPO 算法原理

### 核心思想
IPPO (Independent PPO) 采用"大道至简"的思想，让每个智能体都把其他智能体看作环境的一部分，独立运行自己的 PPO 算法。这是一种完全去中心化的方法：
- 每个智能体拥有独立的 Actor 和 Critic 网络
- 彼此之间不直接共享信息
- 采用去中心化训练，去中心化执行 (Decentralized Training, Decentralized Execution, DTDE)

### 数学目标
每个智能体 $i$ 都在最大化自己的局部目标函数：

$$J^{CLIP}_i(\theta_i) = \mathbb{E}_t \left[ \min(r_{i,t}(\theta_i) \hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta_i), 1-\epsilon, 1+\epsilon) \hat{A}_{i,t}) \right]$$

其中，$r_{i,t}(\theta_i) = \frac{\pi_{\theta_i}(a_{i,t}|s_{i,t})}{\pi_{\theta_{i,old}}(a_{i,t}|s_{i,t})}$ 是智能体 $i$ 的概率比值。

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/xiaoshengdianzi/IPPO_Project.git
   cd IPPO_Project
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training Command
```bash
python scripts/train.py
```

### Prediction Command
```bash
python scripts/predict.py
```

### Configuration
Edit `config.py` to adjust training parameters:
- `team_size`: Number of agents per team (2 or 5)
- `grid_size`: Grid size for the combat environment
- `num_episodes`: Total training episodes
- `actor_lr`/`critic_lr`: Learning rates for actor and critic networks
- `win_reward`/`lose_penalty`: Reward configuration

## Experiment Results

### Combat Environment
项目采用 ma_gym 库中的 Combat 环境，这是一个经典的二维网格对战环境：

**环境规则：**
- 每个智能体初始有 3 点生命值
- 攻击覆盖 3×3 范围，命中扣 1 血，归零阵亡
- 动作包括：移动（上下左右）、攻击、或原地不动
- 攻击有 1 轮冷却时间

**对手机制：**
- 玩家控制一队智能体，另一队由固定脚本 AI 控制
- AI 逻辑：优先攻击最近的敌人；若够不着，则主动靠近

### 2v2 vs 5v5 Performance

#### 2v2 Results
- 训练 5000 轮左右胜率可达 0.5
- 智能体数量较少时，IPPO 能够取得较好的效果

#### 5v5 Results
- 训练 20000 轮时胜率仍为 0
- 需要调整超参数：
  - actor学习率从 2e-4 降低到 1e-4
  - critic学习率从 1e-3 降低到 5e-4
  - 胜利奖励从 100 增加到 200
  - 失败惩罚从 -0.1 减少到 -0.05

**结论：**
- IPPO 扩展性强，不会因智能体数量增加而造成维度灾难
- 但收敛性难以保证，需要仔细调整超参数

## Results
![Win Rate](win_rate_plot.png)

## Project Structure
```
├── scripts/
│   ├── train.py            # Training script
│   ├── predict.py          # Prediction script
│   └── test_visualization.py  # Visualization test
├── models/
│   ├── ppo.py              # PPO algorithm implementation
│   └── networks.py         # Neural network definitions
├── utils/
│   ├── env_utils.py        # Environment utilities
│   ├── plot_utils.py       # Plotting utilities
│   └── rl_utils.py         # Reinforcement learning utilities
├── ma-gym/                 # Multi-agent gym environment
├── saved_models/           # Saved model weights
├── results/                # Prediction and visualization results
├── config.py               # Configuration file
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── LICENSE                 # License file
```

## Contributing
How to contribute to the project.

## License
This project is licensed under the MIT license.

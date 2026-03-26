# IPPO Multi-Agent Combat Training

## 项目结构

```
08_IPPO/
├── ma-gym/                      # ma-gym 环境库
│   └── ma_gym/
│       └── envs/
│           └── combat/
│               └── combat.py    # 修改后的战斗环境
├── scripts/                     # 脚本目录
│   ├── train.py                 # 训练脚本
│   ├── test_visualization.py    # 可视化测试脚本
│   └── predict.py               # 预测脚本
├── utils/                       # 工具目录
│   ├── env_utils.py             # 环境创建工具
│   ├── plot_utils.py            # 绘图工具
│   └── rl_utils.py              # 强化学习工具
├── models/                      # 模型目录
│   ├── networks.py              # 神经网络定义
│   └── ppo.py                   # PPO算法实现
├── saved_models/                # 模型权重保存目录
│   ├── best_actor_model.pth     # 最佳策略网络权重
│   ├── best_critic_model.pth    # 最佳价值网络权重
│   ├── actor_model.pth          # 当前策略网络权重
│   ├── critic_model.pth         # 当前价值网络权重
│   ├── final_actor_model.pth    # 最终策略网络权重
│   └── final_critic_model.pth   # 最终价值网络权重
├── config.py                    # 配置文件
└── README.md                    # 项目说明
```

## 功能说明

### 1. 训练功能 (`scripts/train.py`)
- 自动保存最佳模型（胜率最高）
- 自动保存当前模型
- 自动保存最终模型
- 支持加载已有权重继续训练
- 训练结束后保存胜率曲线

### 2. 可视化测试 (`scripts/test_visualization.py`)
- 模式1：可视化对战（显示对战画面）
- 模式2：性能测试（计算胜率）

### 3. 预测功能 (`scripts/predict.py`)
- 加载最佳模型进行单场对战
- 显示实时对战信息

## 使用方法

### 1. 配置参数
所有训练和测试参数都在 `config.py` 文件中定义，包括：
- 环境配置（网格大小、队伍规模）
- 训练参数（学习率、隐藏层维度、训练回合数）
- 奖励配置（胜利奖励、失败惩罚）
- 保存配置（保存目录、保存间隔）
- 设备配置（自动选择、指定CUDA或CPU）

### 2. 训练
```bash
python scripts/train.py
```
- 训练过程中自动保存模型到 `saved_models/` 目录
- 支持断点续训
- 自动保存最佳模型、当前模型和最终模型

### 3. 可视化测试
```bash
python scripts/test_visualization.py
```
- 选择模式1：可视化测试（显示对战画面）
- 选择模式2：性能测试（计算胜率）

### 4. 预测
```bash
python scripts/predict.py
```
- 加载 `saved_models/best_actor_model.pth` 进行对战
- 显示实时对战信息

## 模型文件说明

所有模型权重文件保存在 `saved_models/` 目录下：
- `best_actor_model.pth`：胜率最高的策略网络权重
- `best_critic_model.pth`：胜率最高的价值网络权重
- `actor_model.pth`：当前训练的策略网络权重
- `critic_model.pth`：当前训练的价值网络权重
- `final_actor_model.pth`：训练结束时的策略网络权重
- `final_critic_model.pth`：训练结束时的价值网络权重

## 环境配置

- Python 3.7+
- PyTorch
- ma-gym (已修改胜负判定逻辑)
- matplotlib
- numpy
- tqdm

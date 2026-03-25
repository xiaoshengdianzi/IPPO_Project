import torch

def compute_advantage(gamma, lmbda, td_delta):
    advantage_list = []
    advantage = 0.0
    for delta in reversed(td_delta.detach().numpy()):
        advantage = gamma * lmbda * advantage + delta[0]
        advantage_list.insert(0, advantage)
    return torch.tensor(advantage_list, dtype=torch.float)

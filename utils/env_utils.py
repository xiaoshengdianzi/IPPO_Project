import sys
sys.path.append("./ma-gym")
from ma_gym.envs.combat.combat import Combat

def make_env(grid_size=(15, 15), team_size=2):
    env = Combat(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size)
    return env

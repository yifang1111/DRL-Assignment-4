import gymnasium
import numpy as np
from train import SACAgent
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dmc import make_dmc_env

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    # def __init__(self):
    #     self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (21,), np.float64)

    # def act(self, observation):
    #     return self.action_space.sample()
    
    def __init__(self):
        self.env = make_dmc_env("humanoid-walk", seed=np.random.randint(0, 1000000), flatten=True, use_pixels=False)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_space = self.env.action_space

        self.agent = SACAgent(state_dim, action_dim, action_space)
        self.agent.load("checkpoints/sac_2580")
        self.agent.actor.eval()

    def act(self, observation):
        with torch.no_grad():
            action = self.agent.select_action(observation, deterministic=True)
        return action

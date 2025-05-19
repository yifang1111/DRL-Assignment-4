import gymnasium as gym
import numpy as np
from train import SACAgent
import torch

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    # def __init__(self):
    #     # Pendulum-v1 has a Box action space with shape (1,)
    #     # Actions are in the range [-2.0, 2.0]
    #     self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)

    # def act(self, observation):
    #     return self.action_space.sample()
    
    def __init__(self):
        self.env = gym.make("Pendulum-v1")
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_space = self.env.action_space

        self.agent = SACAgent(state_dim, action_dim, action_space)
        self.agent.load("checkpoints/sac_1000")
        self.agent.actor.eval()

    def act(self, observation):
        with torch.no_grad():
            action = self.agent.select_action(observation, deterministic=True)
        return action
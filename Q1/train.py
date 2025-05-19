import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.distributions import Normal
import os
import pickle

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_space=None, device='cuda'):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2.).to(device)
        self.action_bias = torch.tensor((action_space.high + action_space.low) / 2.).to(device)

        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample() 
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
        # eps = torch.randn_like(std)
        # action = torch.tanh(mean + std * eps)
        # log_prob = -0.5 * (((eps) ** 2) + 2 * std.log() + np.log(2 * np.pi))
        # log_prob = log_prob.sum(-1, keepdim=True)
        # log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        # return action * self.max_action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.apply(weights_init_)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        print(sa.dtype)
        return self.q1(sa), self.q2(sa)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
       
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards).unsqueeze(1)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)
        

class SACAgent:
    def __init__(self, state_dim, action_dim, action_space):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, action_space, self.device).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.target_entropy = -action_dim
        self.gamma = 0.99
        self.tau = 0.1
        self.memory = ReplayBuffer(100000)
        self.batch_size = 256
        self.train_step = 0
        self.update_freq = 2

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        if deterministic:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def train(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = [x.to(self.device) for x in (state, action, reward, next_state, done)]

        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob
            y = reward + self.gamma * (1 - done) * target_q

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.train_step +=1
        if self.train_step % self.update_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path_prefix):
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict()
        }, f"{path_prefix}_model.pth")
        self.memory.save(f"{path_prefix}_replay.pkl")

    def load(self, path_prefix, train=False):
        checkpoint = torch.load(f"{path_prefix}_model.pth", map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha.data.copy_(checkpoint['log_alpha'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        if train == True:
            self.memory.load(f"{path_prefix}_replay.pkl")

     

if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim, action_dim, env.action_space)
    # agent.load(load_path)
    num_episodes = 1000
    reward_history = [] 

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.add(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

        # print(f"Episode {episode + 1} Reward: {total_reward:.2f}")
        reward_history.append(total_reward)

        if (episode + 1) % 100 == 0:
            # torch.save(agent.model.state_dict(), f"checkpoints/sac_{episode+1}.pth")
            agent.save(f"checkpoints/sac_{episode+1}")
            avg_reward = np.mean(reward_history[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.4f}")


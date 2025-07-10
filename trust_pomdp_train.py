"""
trust_pomdp_train.py
--------------------
1. Wrap the TrustCalibEnvSimple environment into a standard `gym.Env`;
2. Build a tiny policy network in PyTorch;
3. Train the policy with a REINFORCE‑style algorithm.

Author: Han Qing
Date  : 2025‑03‑05
"""

import math, random, collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import gym
    from gym import spaces
except ImportError:
    raise ImportError("Please install gym:  pip install gym==0.26.*")

# ---------------------------------------------------------------------
# 1.  Environment  ----------------------------------------------------
# ---------------------------------------------------------------------
class TrustCalibEnvGym(gym.Env):
    """
    Gym‑style wrapper for trust calibration toy environment.
    Observation: (est_trust [0,1,2], risk [0,1])  -> Discrete(3) + Discrete(2)
    Action     : 4 discrete actions
    Reward     : negative distance from moderate trust − action cost
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space      = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([3, 2])  # (est_trust, risk)
        self._step           = 0
        self._trust_states   = [0, 1, 2]                       # 0 low, 1 moderate, 2 high

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.trust = random.choices(self._trust_states, weights=[0.2, 0.6, 0.2])[0]
        self.risk  = random.choices([0, 1], weights=[0.7, 0.3])[0]
        self._step = 0
        return self._observe(), {}

    def _observe(self):
        noise      = random.choices([-1, 0, 1], weights=[0.1, 0.8, 0.1])[0]
        est_trust  = np.clip(self.trust + noise, 0, 2)
        return np.array([est_trust, self.risk], dtype=np.int64)

    def step(self, action):
        assert self.action_space.contains(action)
        self._step += 1

        # trust dynamics
        if action in (1, 2):                   # soft‑hint / explanation
            if self.trust in (0, 2):
                self.trust = 1                 # move towards moderate
        elif action == 3:                      # force review
            self.trust = 1

        # every 5 steps risk may change
        if self._step % 5 == 0:
            self.risk = random.choices([0, 1], weights=[0.7, 0.3])[0]

        bias_penalty = abs(self.trust - 1)
        action_cost  = [0.0, 0.05, 0.1, 0.25][action]
        reward       = - (bias_penalty + action_cost)

        done = self._step >= 50
        obs  = self._observe()
        return obs, reward, done, False, {}

    # Optional: render to console
    def render(self):
        trust_map = {0: "Low", 1: "Moderate", 2: "High"}
        print(f"Step {self._step} | Trust {trust_map[self.trust]} | Risk {self.risk}")

# ---------------------------------------------------------------------
# 2.  Policy network (PyTorch)  ---------------------------------------
# ---------------------------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self, n_actions=4):
        super().__init__()
        self.embed_trust = nn.Embedding(3, 4)   # 0/1/2 -> 4‑dim
        self.embed_risk  = nn.Embedding(2, 2)   # 0/1   -> 2‑dim
        self.fc = nn.Sequential(
            nn.Linear(4 + 2, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        # obs shape (batch, 2)  int64
        trust_idx = obs[:, 0]
        risk_idx  = obs[:, 1]
        x = torch.cat([self.embed_trust(trust_idx),
                       self.embed_risk(risk_idx)], dim=-1)
        return self.fc(x)

# ---------------------------------------------------------------------
# 3.  REINFORCE training loop  ----------------------------------------
# ---------------------------------------------------------------------
def train(env, episodes=500, gamma=0.99, lr=5e-3):
    policy = PolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_returns = []
    for ep in range(episodes):
        # storage for log‑probabilities & rewards
        log_probs, rewards = [], []
        obs, _ = env.reset()
        done = False
        while not done:
            obs_tensor = torch.tensor([obs], dtype=torch.int64)
            probs = policy(obs_tensor)
            dist  = torch.distributions.Categorical(probs)
            action = dist.sample().item()

            log_probs.append(dist.log_prob(torch.tensor(action)))
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)

        # compute returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        # policy gradient
        loss = -torch.sum(torch.stack(log_probs) * returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_returns.append(sum(rewards))
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1:4d} | AvgReturn last50 = {np.mean(episode_returns[-50:]):.3f}")

    return policy, episode_returns

# ---------------------------------------------------------------------
# 4.  Demo run  -------------------------------------------------------
# ---------------------------------------------------------------------
if __name__ == "__main__":
    env = TrustCalibEnvGym()
    trained_policy, returns = train(env, episodes=300)
    print("Training finished.")

    # quick evaluation run
    obs, _ = env.reset()
    total = 0
    done  = False
    while not done:
        obs_t = torch.tensor([obs], dtype=torch.int64)
        action = torch.argmax(trained_policy(obs_t)).item()
        obs, r, done, _, _ = env.step(action)
        total += r
    print("One evaluation episode total reward:", total)

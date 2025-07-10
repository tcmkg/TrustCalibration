"""
trust_pomdp_a2c.py
------------------
Advantage Actor–Critic (A2C) implementation for the
`TrustCalibEnvGym` environment used in the POMDP trust‑calibration demo.

Dependencies: gym==0.26.*, torch>=1.12
Author      : Han Qing
Date        : 2025‑03‑05
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from trust_pomdp_train import TrustCalibEnvGym  # reuse environment

# -----------------------------  Network  -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, n_actions=4, embed_dim=8):
        super().__init__()
        self.embed_trust = nn.Embedding(3, embed_dim // 2)
        self.embed_risk  = nn.Embedding(2, embed_dim // 2)
        self.body = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
        )
        self.actor = nn.Linear(32, n_actions)
        self.critic = nn.Linear(32, 1)

    def forward(self, obs):
        # obs shape (batch,2)
        x = torch.cat([self.embed_trust(obs[:, 0]),
                       self.embed_risk(obs[:, 1])], dim=-1)
        x = self.body(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value.squeeze(-1)

# ----------------------------  A2C agent  ----------------------------
class A2CAgent:
    def __init__(self,
                 env,
                 gamma=0.99,
                 lr=3e-4,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 rollout_len=32):
        self.env = env
        self.gamma = gamma
        self.rollout_len = rollout_len
        self.device = torch.device("cpu")
        self.ac = ActorCritic(env.action_space.n).to(self.device)
        self.optim = optim.Adam(self.ac.parameters(), lr=lr)
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def select_action(self, obs_t):
        obs_t = torch.tensor([obs_t], dtype=torch.int64, device=self.device)
        logits, value = self.ac(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy(), value

    def train(self, total_steps=10_000):
        obs, _ = self.env.reset()
        ep_returns = []
        ep_reward = 0
        log_buffer = deque(maxlen=50)

        step_count = 0
        while step_count < total_steps:
            # rollout storage
            log_probs, values, rewards, entropies = [], [], [], []
            for _ in range(self.rollout_len):
                action, logp, entropy, value = self.select_action(obs)
                next_obs, r, done, _, _ = self.env.step(action)

                log_probs.append(logp)
                values.append(value)
                rewards.append(torch.tensor([r], dtype=torch.float32, device=self.device))
                entropies.append(entropy)

                obs = next_obs
                ep_reward += r
                step_count += 1
                if done:
                    log_buffer.append(ep_reward)
                    ep_returns.append(ep_reward)
                    obs, _ = self.env.reset()
                    ep_reward = 0
                    break

            # bootstrap value
            with torch.no_grad():
                _, next_value = self.ac(torch.tensor([obs], dtype=torch.int64, device=self.device))
            returns = []
            R = next_value
            for reward in reversed(rewards):
                R = reward + self.gamma * R
                returns.insert(0, R)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)
            log_probs = torch.cat(log_probs)
            entropies = torch.cat(entropies)

            advantage = returns - values
            # losses
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            entropy_loss = -entropies.mean()

            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if step_count % 2000 == 0 and log_buffer:
                print(f"Steps {step_count:6d} | AvgReturn(last50) = {np.mean(log_buffer):6.2f}")

        return ep_returns

# -----------------------------  MAIN  --------------------------------
if __name__ == "__main__":
    env = TrustCalibEnvGym()
    agent = A2CAgent(env)
    agent.train(total_steps=20000)

    # quick deterministic evaluation
    obs,_ = env.reset()
    total = 0
    done = False
    while not done:
        logits, _ = agent.ac(torch.tensor([obs], dtype=torch.int64))
        action = torch.argmax(logits, dim=-1).item()
        obs, r, done, _, _ = env.step(action)
        total += r
    print("A2C policy evaluation return:", total)

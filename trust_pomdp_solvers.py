"""
trust_pomdp_solvers.py
----------------------
Implementations of two classic POMDP solvers
(1) PBVI   –  Point‑Based Value Iteration (finite‑horizon)
(2) POMCP  –  Partially Observable Monte‑Carlo Planning (online)
They are wired to the `TrustCalibEnvGym` environment defined in
`trust_pomdp_train.py`. 

Author: Han Qing
Date  : 2025‑03‑05
"""

import random, math, itertools
import numpy as np
from collections import defaultdict, deque

# ---------------------------------------------------------------
# Re‑use environment
# ---------------------------------------------------------------
from trust_pomdp_train import TrustCalibEnvGym

# ---------------------------------------------------------------
# 1. PBVI  -------------------------------------------------------
# ---------------------------------------------------------------
class PBVISolver:
    """
    Finite‑horizon PBVI with random belief set.
    """
    def __init__(self, env: TrustCalibEnvGym, horizon=10, n_beliefs=30, gamma=0.95):
        self.env      = env
        self.horizon  = horizon
        self.gamma    = gamma
        self.nB       = n_beliefs
        self.actions  = list(range(env.action_space.n))
        # Enumerate underlying true states (trust 0/1/2 × risk 0/1)
        self.states   = list(itertools.product([0,1,2],[0,1]))
        self._init_random_beliefs()
        # α‑vectors per stage (list of dict(state)->value)
        self.alpha    = [ {s:0.0 for s in self.states} for _ in range(self.horizon+1) ]

    def _init_random_beliefs(self):
        self.beliefs = []
        for _ in range(self.nB):
            probs = np.random.dirichlet(np.ones(len(self.states)))
            self.beliefs.append(dict(zip(self.states, probs)))

    def _reward(self, s, a):
        bias_penalty = abs(s[0]-1)
        action_cost  = [0.0,0.05,0.1,0.25][a]
        return -(bias_penalty + action_cost)

    def _transition(self, s, a):
        # deterministic trust change as in env; risk random every 5 steps -> approximate
        trust, risk = s
        if a in (1,2): 
            trust = 1 if trust in (0,2) else trust
        elif a==3:
            trust = 1
        risk_new = risk  # keep risk same here for approximation
        return (trust, risk_new)

    def value_iteration(self):
        for t in reversed(range(self.horizon)):
            new_alpha = []
            for a in self.actions:
                alpha_a = {}
                for s in self.states:
                    s2 = self._transition(s,a)
                    r  = self._reward(s,a)
                    alpha_a[s] = r + self.gamma * self.alpha[t+1][s2]
                new_alpha.append(alpha_a)
            # Choose best action per state
            best = {}
            for s in self.states:
                best[s] = max(alpha_a[s] for alpha_a in new_alpha)
            self.alpha[t] = best

    def act(self, belief):
        # compute value per action
        action_values=[]
        for a in self.actions:
            tot=0.0
            for s,p in belief.items():
                s2=self._transition(s,a)
                r = self._reward(s,a)
                tot+=p*(r + self.gamma*self.alpha[1][s2])
            action_values.append(tot)
        return int(np.argmax(action_values))

# ---------------------------------------------------------------
# 2. POMCP (online) ---------------------------------------------
# ---------------------------------------------------------------
class POMCPSolver:
    """
    Simplified POMCP: UCT search with particle filter belief.
    """
    def __init__(self, env: TrustCalibEnvGym, sims=200, gamma=0.95, ucb_c=1.4):
        self.env     = env
        self.sims    = sims
        self.gamma   = gamma
        self.ucb_c   = ucb_c
        self.tree    = {}  # nested dict: node -> {a: [N, Q, {obs:child}]}
        self.particles = deque(maxlen=200)

    def _default_node(self):
        return {a:[0,0.0,{}] for a in range(self.env.action_space.n)}

    def _simulate(self, state, depth, node):
        if depth==0: return 0.0
        if node not in self.tree: self.tree[node]=self._default_node()
        # UCB action
        N_tot = sum(self.tree[node][a][0] for a in self.tree[node])
        best_a, best_ucb = None, -1e9
        for a,(N,Q,_) in self.tree[node].items():
            ucb = Q + self.ucb_c*math.sqrt(math.log(N_tot+1)/(N+1e-5))
            if ucb>best_ucb: best_ucb, best_a = ucb, a
        a = best_a
        # step generative model
        trust, risk = state
        if a in (1,2):
            trust = 1 if trust in (0,2) else trust
        elif a==3: trust=1
        reward = -(abs(trust-1)+[0,0.05,0.1,0.25][a])
        obs = (trust, risk)  # simplistic deterministic obs
        child_key = (trust,risk,depth-1)
        if child_key not in self.tree[node][a][2]:
            self.tree[node][a][2][child_key] = self._default_node()
        R = reward + self.gamma*self._simulate((trust,risk),depth-1,self.tree[node][a][2][child_key])
        # update stats
        N,Q,_ = self.tree[node][a]
        self.tree[node][a][0] = N+1
        self.tree[node][a][1] = Q + (R-Q)/(N+1)
        return R

    def act(self, obs, horizon=10):
        # Monte‑Carlo simulations from particle set
        # simplistic: use obs as deterministic state proxy
        root = ('root',)
        for _ in range(self.sims):
            self._simulate(tuple(obs), horizon, root)
        # choose best action by Q
        q_values = {a:self.tree[root][a][1] for a in range(len(self.env.action_space.n))}
        return max(q_values, key=q_values.get)

# ---------------------------------------------------------------
# Demo using PBVI policy
# ---------------------------------------------------------------
def rollout(env, policy_fn, episodes=2):
    totals=[]
    for _ in range(episodes):
        obs,_=env.reset(); done=False; total=0
        # naive belief init: uniform
        belief={(t,r):1/6 for t in [0,1,2] for r in [0,1]}
        while not done:
            action=policy_fn(belief,obs)
            obs,r,done,_,_=env.step(action)
            total+=r
        totals.append(total)
    return totals

def pbvi_policy_maker(pbvi):
    def policy(belief,obs):
        # naive: update belief to point mass at obs
        trust,risk=obs
        belief={(t,r):0.0 for t in [0,1,2] for r in [0,1]}
        belief[(trust,risk)]=1.0
        return pbvi.act(belief)
    return policy

if __name__=="__main__":
    env=TrustCalibEnvGym()
    pbvi=PBVISolver(env,horizon=10)
    pbvi.value_iteration()
    returns=rollout(env,pbvi_policy_maker(pbvi),episodes=3)
    print("PBVI demo returns:",returns)

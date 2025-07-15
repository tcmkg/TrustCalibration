# TrustCalibration
Trust Calibration for Medical AI

# trust_pomdp_train.py
TrustCalibEnvGym：以 Gym API 实现的信任校准环境（状态：信任等级\风险；动作：4级干预）

PolicyNet (PyTorch): 嵌入式特征，两层 MLP，输出4个动作概率

REINFORCE 训练循环：500 轮训练、归一化回报、梯度上升；可实时输出平均回报

Demo评估：训练完成后使用贪心策略跑一次评估并输出总奖励

Python环境下运行：

pip install gym torch

python trust_pomdp_train.py

即可启动训练演示。若需接入高级求解器（PBVI、POMCP）或深度 Actor-Critic，只需替换 train() 部分即可。 

# trust_pomdp_solvers.py
TrustCalibEnvGym：复用 Gym 环境（从 trust_pomdp_train.py 导入）

PBVISolver：随机belief采样，逆向 α-vector 迭代，支持 act(belief) 推断

POMCPSolver：精简版在线 MCTS：UCT 搜索、belief、可即时调用 act(obs)

Demo：PBVI训练10步后，进行3回合rollout并输出总回报

Python环境下运行：

python trust_pomdp_solvers.py

# trust_pomdp_a2c.py
代码会训练一个带嵌入的Actor-Critic，期间输出平均回报，并最终给出一次确定性评估回报

网络结构简单（6-维嵌入 → 32-hidden → actor/critic），便于后续添加更复杂的特征、LSTM 记忆或注意力机制


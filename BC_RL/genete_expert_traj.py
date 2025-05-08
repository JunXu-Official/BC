"""
    cartpole环境下写一个专家规则采集5000步数据
    专家规则：
        角度>0,角速度>0: action = 1; 否则 action = 0
"""
import gym
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def expert_action(state):
    # state: [x, x_dot, theta, theta_dot]
    theta, theta_dot = state[2], state[3]
    return 1 if (theta > 0 and theta_dot > 0) else 0

def collect_expert_data(env_name="CartPole-v1", num_steps=10000):
    env = gym.make(env_name)
    obs_buf, act_buf, rew_buf, next_obs_buf = [], [], [], []
    state, _ = env.reset()
    for _ in range(num_steps):
        action = expert_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        obs_buf.append(state); act_buf.append(action)
        rew_buf.append([reward]); next_obs_buf.append(next_state)
        state = next_state if not (terminated or truncated) else env.reset()[0]
    return {
        'obs':      np.array(obs_buf, dtype=np.float32),
        'action':   np.array(act_buf, dtype=np.int64),
        'reward':   np.array(rew_buf, dtype=np.float32),
        'next_obs': np.array(next_obs_buf, dtype=np.float32)
    }


#计算每一步的累计回报
def compute_returns(reward, gamma=0.99):
    returns = []
    R = 0
    for r in reward[::-1]:
        R = r + R * gamma
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32).unsqueeze(1)

#构造dataloader
def get_dataloader(expert_data):
    obs = torch.from_numpy(expert_data['obs'])
    action = torch.from_numpy(expert_data['action'])
    next_obs = torch.from_numpy(expert_data['next_obs'])
    reward = compute_returns(expert_data['reward'])
    dataset_actor = TensorDataset(obs, action)
    dataset_critic = TensorDataset(obs, reward)
    loader_actor = DataLoader(dataset_actor, batch_size=64, shuffle=True)
    loader_critic = DataLoader(dataset_critic, batch_size=64, shuffle=True)
    return loader_actor, loader_critic


if __name__ == '__main__':
    expert_data = collect_expert_data(num_steps=5000)
    data, data1 = get_dataloader(expert_data)

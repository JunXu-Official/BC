import gym
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle

def load_task_data(filename):
    with open(filename, 'rb') as f:
        task_data = pickle.loads(f.read())
    return task_data

expert_name = "Hopper-v1"
data_file = "data/Hopper-v1_20_data.pkl"



class BCNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(BCNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, act_dim)
        )
    def forward(self, x):
        return self.net(x)

def run_clone(expert_name, expert_data_file, render=False, num_rollouts=20, device='cpu'):
    # 1. 加载专家策略（可用于后续对比或 DAgger 扩展）
    # print('Loading expert policy...')
    # policy_fn = load_policy.load_policy(f"./experts/{expert_name}.pkl")
    # print('Expert policy loaded.')

    # 2. 加载专家示范数据
    task_data = load_task_data(expert_data_file)
    obs_data = np.array(task_data["observations"], dtype=np.float32)
    act_data = np.array(task_data["actions"], dtype=np.float32)
    act_data = act_data.reshape(act_data.shape[0], act_data.shape[2])

    # 转为 TensorDataset
    obs_tensor = torch.from_numpy(obs_data)
    act_tensor = torch.from_numpy(act_data)
    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 3. 构建模型
    obs_dim, act_dim = obs_data.shape[1], act_data.shape[1]
    model = BCNet(obs_dim, act_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4. 行为克隆预训练
    model.train()
    for epoch in range(30):
        total_loss = 0.0
        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)
            pred = model(batch_obs)
            loss = criterion(pred, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_obs.size(0)

        print(f"Epoch {epoch}  Loss: {total_loss/len(dataset):.6f}")

    # 5. 保存模型
    torch.save(model.state_dict(),"cloned_model.pt")

    # 6. 在环境中加载评估
    env = gym.make(expert_name)
    returns = []
    model.eval()
    for i in range(num_rollouts):
        obs = env.reset()
        totalr, steps, done = 0.0, 0, False
        while not done:
            obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(obs_tensor).cpu().numpy()[0]
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps >= env.spec.max_episode_steps:
                break
        returns.append(totalr)
        print(f"Rollout {i+1}: Return = {totalr:.2f}")

    print(f"Mean Return: {np.mean(returns):.2f}  Std: {np.std(returns):.2f}")
    # return returns

if __name__ == '__main__':
    run_clone('Hopper-v4', 'Hopper-v1_20_data.pkl', render=False, num_rollouts=20, device='cpu')


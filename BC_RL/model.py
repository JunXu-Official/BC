import torch
import torch.nn as nn
import torch.nn.functional as F
from genete_expert_traj import collect_expert_data, get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.logits = nn.Linear(hidden, act_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.logits(x)

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value(x)

#actor行为克隆
def clone_actor(actor_model, dataloader, num_epoches):
    optim_actor = torch.optim.Adam(actor_model.parameters(), lr=1e-3)
    criterion_actor = nn.CrossEntropyLoss()
    for epoch in range(num_epoches):
        for state_batch, action_batch in dataloader:
            actor_output = actor_model(state_batch)
            loss = criterion_actor(actor_output, action_batch.squeeze())
            optim_actor.zero_grad()
            loss.backward()
            optim_actor.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.3f}")

#critic网络行为克隆
def clone_critic(critic_model, dataloader, num_epoches):
    optim_critic = torch.optim.Adam(critic_model.parameters(), lr=3e-4)
    criterion_critic = nn.MSELoss()
    for epoch in range(num_epoches):
        for state_batch, reward_batch in dataloader:
            critic_output = critic_model(state_batch.to(device))
            loss = criterion_critic(critic_output, reward_batch.to(device))
            optim_critic.zero_grad()
            loss.backward()
            optim_critic.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.3f}")


if __name__ == '__main__':

    #生成专家数据
    expert_data = collect_expert_data(num_steps=5000)
    data_actor, data_critic = get_dataloader(expert_data)
    #actor和critic网络实例化
    actor = Actor(obs_dim=4, act_dim=2).to(device)
    critic = Critic(obs_dim=4)
    #训练轮数
    num_epoches = 200
    bc_mode = "critic"
    if bc_mode == "actor":
        clone_actor(actor_model=actor, dataloader=data_actor, num_epoches=num_epoches)
        torch.save(actor.state_dict(), 'clone_actor.pt')
    else:
        clone_critic(critic_model=critic, dataloader=data_critic, num_epoches=num_epoches)
        torch.save(critic.state_dict(), 'clone_critic.pt')



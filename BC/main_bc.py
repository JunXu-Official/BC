import torch
import json
import torch.nn as nn
import numpy as np
import torch.optim as optim

#读取保存的json专家数据
with open('expert_data.json', 'r', encoding='utf-8') as f:
    expert_data = json.load(f)

#将数据处理成state, action的格式
states = []
actions = []
for i in range(len(expert_data)):
    for j in expert_data[i]:
        if len(np.array(j).flatten()) > 3:
            states.append(np.array(j).flatten())
        else:
            action_index = np.array(j).flatten()[0] * 3 + np.array(j).flatten()[1]
            actions.append(action_index)

states_tensor = torch.tensor(states, dtype=torch.float32)
actions_tensor = torch.tensor(actions, dtype=torch.long)

print("输入是：", states_tensor)
print("标签是：", actions_tensor)

#定义模型结构
class BCModel(nn.Module):
    def __init__(self, input_dim=9, output_dim=9):
        super(BCModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#定义模型，损失函数
model = BCModel()
criterion = nn.CrossEntropyLoss()  #MLE
optimizer = optim.Adam(model.parameters(), lr=1e-2)
num_epoches = 1000
for epoch in range(num_epoches):
    model.train()
    optimizer.zero_grad()
    outputs = model(states_tensor)
    loss = criterion(outputs, actions_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epoches}], Loss: {loss.item():.4f}')


#测试




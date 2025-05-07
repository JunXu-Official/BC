import torch
import json
import torch.nn as nn

#处理数据
def data_process(data):

    states = []
    actions = []
    for state, action in expert_data:
        flat_state = state.flatten()
        action_index = action[0] * 3 + action[1]
        states.append(flat_state)
        actions.append(action_index)
    return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.long)

with open('expert_data.json', 'r', encoding='utf-8') as f:
    expert_data = json.load(f)

print(expert_data[0], len(expert_data))

#处理数据，分为state和action
print('-' * 40)
for i in range(len(expert_data)):
    for state, action in expert_data[i]:
        print(action)

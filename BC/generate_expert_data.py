import numpy as np
import random
import json


def heuristic_agent(board, player):
    # 检查是否有人胜利
    def check_win(board, player):
        # 检查行
        for row in board:
            if row[0] == player and row[1] == player and row[2] == player:
                return True
                # 检查列
        for col in range(3):
            if board[0][col] == player and board[1][col] == player and board[2][col] == player:
                return True
                # 检查对角线
        if board[0][0] == player and board[1][1] == player and board[2][2] == player:
            return True
        if board[0][2] == player and board[1][1] == player and board[2][0] == player:
            return True
        return False

        # 检查是否平局

    def check_draw(board):
        return all(board[row][col] != '-' for row in range(3) for col in range(3))

        # 获取所有空位置

    def get_empty_positions(board):
        return [(row, col) for row in range(3) for col in range(3) if board[row][col] == '-']

        # 判断玩家是否即将胜利（用于防守策略）

    def is_about_to_win(board, player):
        for row in range(3):
            for col in range(3):
                if board[row][col] == '-':
                    temp_board = [row[:] for row in board]  # 深拷贝棋盘
                    temp_board[row][col] = player
                    if check_win(temp_board, player):
                        return (row, col)
        return None

        # 计算最佳位置（策略）

    def get_best_move(board, player):
        # 进攻策略：尝试形成三子连线
        for row in range(3):
            for col in range(3):
                if board[row][col] == '-':
                    temp_board = [row[:] for row in board]  # 深拷贝棋盘
                    temp_board[row][col] = player
                    if check_win(temp_board, player):
                        return (row, col)

                        # 防守策略：阻止对方形成三子连线
        block_move = is_about_to_win(board, get_opponent(player))
        if block_move:
            return block_move

        best_move = None
        max_twos = 0
        for row in range(3):
            for col in range(3):
                if board[row][col] == '-':
                    temp_board = [row[:] for row in board]
                    temp_board[row][col] = player
                    twos_count = count_twos(temp_board, player)
                    if twos_count > max_twos:
                        max_twos = twos_count
                        best_move = (row, col)
                        # 如果有多个位置能形成相同数量的二连，这个逻辑会选择最后一个检查到的位置
        if best_move:
            return best_move
        else:
            # 如果没有找到最佳的位置，随机选择一个空位
            return random.choice(get_empty_positions(board))

            # 获取对手玩家

    def get_opponent(player):
        return 'X' if player == 'O' else 'O'

    def count_twos(board, player):
        # 初始化二连计数
        twos_count = 0

        # 检查行
        for row in range(3):
            for col in range(2):  # 只需要检查前两列，因为第三列无法形成二连
                if board[row][col] == player and board[row][col + 1] == player:
                    twos_count += 1

                    # 检查列
        for col in range(3):
            for row in range(2):  # 只需要检查前两行，因为第三行无法形成二连
                if board[row][col] == player and board[row + 1][col] == player:
                    twos_count += 1

                    # 检查对角线（如果有的话）
        # 检查主对角线
        if board[0][0] == player and board[1][1] == player:
            twos_count += 1
        if board[1][1] == player and board[2][2] == player:
            twos_count += 1
            # 检查副对角线
        if board[0][2] == player and board[1][1] == player:
            twos_count += 1
        if board[1][1] == player and board[2][0] == player:
            twos_count += 1

            # 返回二连的总数
        return twos_count

    return get_best_move(board, player)


# 定义 Tic-Tac-Toe 棋盘环境
class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [['-' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        return self.board

    def step(self, action):
        row, col = action
        self.board[row][col] = self.current_player
        reward, done = self._evaluate()
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return self.board, reward, done

    def _evaluate(self):
        if check_win(self.board, 'X'):
            return 1, True  # X胜利
        elif check_win(self.board, 'O'):
            return -1, True  # O胜利
        elif '-' not in [cell for row in self.board for cell in row]:  # 检查平局
            return 0, True
        else:
            return 0, False  # 游戏未结束

    def get_empty_positions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == '-']


def check_win(board, player):
    # 检查行
    for row in board:
        if row[0] == player and row[1] == player and row[2] == player:
            return True
    # 检查列
    for col in range(3):
        if board[0][col] == player and board[1][col] == player and board[2][col] == player:
            return True
    # 检查对角线
    if board[0][0] == player and board[1][1] == player and board[2][2] == player:
        return True
    if board[0][2] == player and board[1][1] == player and board[2][0] == player:
        return True
    return False

# 生成专家数据
def generate_expert_data(num_episodes=100):
    expert_data = []
    env = TicTacToeEnv()

    for _ in range(num_episodes):
        state = env.reset()
        episode_data = []
        done = False
        while not done:
            action = heuristic_agent(state, env.current_player)
            next_state, reward, done = env.step(action)
            episode_data.append((state, action))
            state = next_state
        expert_data.extend(episode_data)
    return expert_data

expert_data = generate_expert_data(500)  # 生成500局
print('初始化生成的专家数据为：', len(expert_data))

#将字符替换成数字
for i in range(len(expert_data)):
    for j in range(len(expert_data[i][0])):
        for k in range(3):
            if expert_data[i][0][j][k] == '-':
                expert_data[i][0][j][k] = 0
            elif expert_data[i][0][j][k] == 'X':
                expert_data[i][0][j][k] = 1
            elif expert_data[i][0][j][k] == 'O':
                expert_data[i][0][j][k] = -1

#将专家数据保存到json
expert_data_json = json.dumps(expert_data)
with open('expert_data.json', 'w', encoding='utf-8') as f:
    f.write(expert_data_json)


# def encode_state(state):
#     mapping = {'X': 1, 'O': -1, '-': 0}
#     return np.array([[mapping[cell] for cell in row] for row in state], dtype=np.float32)
#
# # 将原始数据转换为 expert_data 格式
# def convert_raw_to_expert(raw_data):
#     expert_data = []
#     for state, action in raw_data:
#         encoded_state = encode_state(state)  # 转成 NumPy 数组
#         expert_data.append((encoded_state, action))
#     return expert_data
#
# # 你原来的 data_process 函数
# def data_process(data):
#     states = []
#     actions = []
#     for state, action in data:
#         flat_state = state.flatten()
#         action_index = action[0] * 3 + action[1]
#         states.append(flat_state)
#         actions.append(action_index)
#     return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.long)
#
# # 使用处理
# expert_data = convert_raw_to_expert(expert_data)
# states_tensor, actions_tensor = data_process(expert_data)
# print(states_tensor)
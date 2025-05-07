"""
    1. 带α_β的min-max算法解决井字棋问题
    2. ref: https://zhuanlan.zhihu.com/p/1843721751
"""

import math

EMPTY = 0
agent_ai = 1  # AI
agent_enemy = -1  # 对手

class TicTacToe:
    def __init__(self):
        self.board = [EMPTY] * 9

    #打印棋盘
    def print_board(self):
        symbols = [' ', 'X', 'O']
        for i in range(3):
            print([symbols[x] for x in self.board[i*3:(i+1)*3]])
        print()

    #获胜条件
    def is_winner(self, player):
        win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                          (0, 3, 6), (1, 4, 7), (2, 5, 8),
                          (0, 4, 8), (2, 4, 6)]
        return any(all(self.board[i] == player for i in condition) for condition in win_conditions)

    #平局
    def is_full(self):
        return all(x != EMPTY for x in self.board)

    #落子
    def make_move(self, position, player):
        self.board[position] = player

    #空余的落子位置
    def get_available_moves(self):
        return [i for i in range(9) if self.board[i] == EMPTY]

    #初始化
    def reset(self):
        self.board = [EMPTY] * 9

    #终止条件
    def is_terminal(self):
        return self.is_winner(agent_enemy) or self.is_winner(agent_ai) or self.is_full()

# 带α-β剪枝操作的Minimax算法
def minimax(game, is_maximizing, alpha, beta):
    if game.is_winner(agent_ai): #我赢了
        return 1
    elif game.is_winner(agent_enemy):  #对手赢了
        return -1
    elif game.is_full():  #平局
        return 0

    if is_maximizing:
        max_eval = -math.inf
        for move in game.get_available_moves():
            game.make_move(move, agent_ai)
            eval = minimax(game, False, alpha, beta)
            game.board[move] = EMPTY
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:   #说明当前分支已经足够好了，不需要尝试其他分支
                break
        return max_eval

    else:
        min_eval = math.inf
        for move in game.get_available_moves():
            game.make_move(move, agent_enemy)
            eval = minimax(game, True, alpha, beta)
            game.board[move] = EMPTY
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

#最佳位置
def best_move(game):
    best_val = -math.inf
    best_move = None
    for move in game.get_available_moves():
        game.make_move(move, agent_ai)
        move_val = minimax(game, False, -math.inf, math.inf)
        game.board[move] = EMPTY
        if move_val > best_val:
            best_move = move
            best_val = move_val
    return best_move

# 人机对战
def play_against_minimax_ai():
    game = TicTacToe()
    player = agent_enemy  # 人类先手

    while True:
        game.print_board()

        if player == agent_enemy:
            action = int(input("Enter your move (0-8): "))
            if action not in game.get_available_moves():
                print("Invalid move! Try again.")
                continue
            game.make_move(action, agent_enemy)

            if game.is_winner(agent_enemy):
                print("You win!")
                game.print_board()
                break
            elif game.is_full():
                print("It's a draw!")
                game.print_board()
                break
        else:
            # AI
            action = best_move(game)
            game.make_move(action, agent_ai)

            if game.is_winner(agent_ai):
                print("Agent wins!")
                game.print_board()
                break
            elif game.is_full():
                print("It's a draw!")
                game.print_board()
                break

        player *= -1  # 交换

play_against_minimax_ai()
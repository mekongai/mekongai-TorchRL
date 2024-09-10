import numpy as np
import random
import gym
from gym import spaces


class TicTacToeEnv(gym.Env):
    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        self.action_space = spaces.Discrete(9)  # 9 vị trí trên bảng cờ
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int8)
        self.reset()
        self.memory = []  # Bộ nhớ lưu trữ trạng thái và hành động

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)  # 0: ô trống, 1: người chơi, -1: máy
        self.done = False
        self.memory = []  # Xóa bộ nhớ sau mỗi ván chơi
        return self.board.flatten()

    def step(self, action):
        row, col = divmod(action, 3)
        if self.board[row, col] != 0:
            return self.board.flatten(), -10, True, {}  # Nước đi không hợp lệ
        self.board[row, col] = 1
        self.memory.append((self.board.copy(), action))  # Lưu trạng thái và hành động vào bộ nhớ

        if self.check_winner(1):
            return self.board.flatten(), 1, True, {}
        if self.is_draw():
            return self.board.flatten(), 0, True, {}

        self.computer_move()
        if self.check_winner(-1):
            return self.board.flatten(), -1, True, {}
        if self.is_draw():
            return self.board.flatten(), 0, True, {}

        return self.board.flatten(), 0, False, {}

    def computer_move(self):
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            row, col = random.choice(empty)
            self.board[row, col] = -1
            self.memory.append((self.board.copy(), row * 3 + col))

    def check_winner(self, player):
        for row in range(3):
            if all(self.board[row, :] == player):
                return True
        for col in range(3):
            if all(self.board[:, col] == player):
                return True
        if all([self.board[i, i] == player for i in range(3)]) or all([self.board[i, 2 - i] == player for i in range(3)]):
            return True
        return False

    def is_draw(self):
        return not np.any(self.board == 0)

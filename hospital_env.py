import gym
import numpy as np
from gym import spaces

class HospitalEnv(gym.Env):
    def __init__(self):
        super(HospitalEnv, self).__init__()
        
        self.grid_size = 5
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Right, 2: Down, 3: Left
        self.observation_space = spaces.Box(low=0, high=5, shape=(self.grid_size, self.grid_size), dtype=np.float32)
        
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.beds = [(0, 1), (0, 2), (1, 0), (1, 1)]
        self.doctor = (0, 3)
        self.nurse = (1, 3)
        self.medicine_cabinet = (2, 2)
        self.agent_pos = (4, 4)
        
        self._update_grid()

    def _update_grid(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        for bed in self.beds:
            self.grid[bed] = 1
        self.grid[self.doctor] = 2
        self.grid[self.nurse] = 3
        self.grid[self.medicine_cabinet] = 4
        self.grid[self.agent_pos] = 5

    def step(self, action):
        if action == 0:  # Up
            new_pos = (max(0, self.agent_pos[0] - 1), self.agent_pos[1])
        elif action == 1:  # Right
            new_pos = (self.agent_pos[0], min(self.grid_size - 1, self.agent_pos[1] + 1))
        elif action == 2:  # Down
            new_pos = (min(self.grid_size - 1, self.agent_pos[0] + 1), self.agent_pos[1])
        else:  # Left
            new_pos = (self.agent_pos[0], max(0, self.agent_pos[1] - 1))

        if new_pos == self.doctor or new_pos == self.nurse:
            reward = -10
            done = True
        elif new_pos == self.medicine_cabinet:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        self.agent_pos = new_pos
        self._update_grid()

        return self.grid, reward, done, False, {}  # Added False for truncated flag

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = (4, 4)
        self._update_grid()
        return self.grid, {}  # Return initial observation and empty info dict

    def render(self):
        print(self.grid)
        return self.grid
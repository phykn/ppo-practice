import cv2
import numpy as np
import flappy_bird_gym
import gymnasium as gym
from gymnasium import spaces

class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, img_size = 128, penalty = 0):
        super().__init__()
        self.img_size = img_size
        self.penalty = penalty

        self.env = flappy_bird_gym.make(id = "FlappyBird-rgb-v0")
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low = 0, 
            high = 255,
            shape = (3, img_size, img_size),
            dtype = np.uint8
        )

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        obs = self.preprocess(obs)
        truncated = False

        if action == 1: 
            reward -= self.penalty
            
        return obs, reward, terminated, truncated, info

    def reset(self, seed = None, options = None):
        obs = self.env.reset()
        obs = self.preprocess(obs)
        return obs, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def preprocess(self, x):
        x = cv2.resize(
            src = x, 
            dsize = (self.img_size, self.img_size), 
            interpolation = cv2.INTER_AREA
        )
        x = x.transpose(2, 0, 1)
        return x
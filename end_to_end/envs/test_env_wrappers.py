import unittest
import numpy as np
import gym
from gym.spaces import Box
from wrappers import LongStackFrame

# Language: python


# Toy gym environment: vector of zeros with a moving one
class ToyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.obs_len = 20
        self.observation_space = Box(low=0, high=1, shape=(self.obs_len,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(1)
        self.idx = 0

    def reset(self):
        self.idx = 0
        return self._get_obs()

    def step(self, action):
        # Increase index until last position
        self.idx = min(self.idx + 1, self.obs_len - 1)
        obs = self._get_obs()
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def _get_obs(self):
        obs = np.zeros(self.obs_len, dtype=np.float32)
        obs[self.idx] = 1.0
        return obs


if __name__ == "__main__":
    env = ToyEnv()
    env = LongStackFrame(env, stack_frame=2, long_stack_frame=3, history_buffer_size=50)
    obs = env.reset()
    print("Initial observation:")
    print(obs)
    
    s, _, _, _ = env.step(0)
    print("After 1 step:")
    print(s)
    
    for i in range(10):
        s, _, _, _ = env.step(0)
    
    print("After 10 steps:")
    print(s)
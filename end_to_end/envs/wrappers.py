import gym
import numpy as np
from collections import deque

class ShapingRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        self.Y = self.env.gazebo_sim.get_model_state().pose.position.y
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        position = self.env.gazebo_sim.get_model_state().pose.position
        rew += position.y - self.Y
        self.Y = position.y
        return obs, rew, done, info

class StackFrame(gym.Wrapper):
    """
    #TODO: check the stack_frame is properly implemented and turns the observation space into a convolutional space
    """
    def __init__(self, env, stack_frame=1):
        super().__init__(env)
        self.stack_frame = stack_frame
        low = self.observation_space.low
        high = self.observation_space.high
        low = np.repeat(low[None, :], stack_frame, axis=0)
        high = np.repeat(high[None, :], stack_frame, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        self.frames = deque(maxlen=self.stack_frame)
        obs = self.env.reset()
        self.frames.extend([obs] * self.stack_frame)
        return np.stack(self.frames)

    def step(self, *args, **kwargs):
        obs, rew, terminations, truncations, info = self.env.step(*args, **kwargs)
        self.frames.append(obs)
        return np.stack(self.frames), rew, terminations, truncations, info


class LongStackFrame(gym.Wrapper):
    """
    This wrappers stacks frames but includes frames from way earlier
    in hopes to give the agent more context about the obstacle configuration
    """

    def __init__(self, env, stack_frame=1, long_stack_frame=2, history_buffer_size=50, lag=10):
        super().__init__(env)
        self.stack_frame = stack_frame
        self.long_stack_frame = long_stack_frame
        low = self.observation_space.low
        high = self.observation_space.high
        low = np.repeat(low[None, :], stack_frame + long_stack_frame, axis=0)
        high = np.repeat(high[None, :], stack_frame + long_stack_frame, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.lag = lag
        self.history_buffer = deque(
            maxlen=history_buffer_size
        )  # Store more frames for long-term memory

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.frames = deque(maxlen=self.stack_frame)
        self.frames.extend([obs] * self.stack_frame)
        self.history_buffer.clear()
        self.history_buffer.extend([obs] * 100)  # Initialize history

        # Combine recent frames with historical frames
        stacked_obs = list(self.frames)
        for _ in range(self.long_stack_frame):
            stacked_obs.append(obs)  # Use current obs for initialization

        return np.stack(stacked_obs)

    def step(self, *args, **kwargs):
        obs, rew, done, info = self.env.step(*args, **kwargs)

        self.frames.append(obs)
        self.history_buffer.append(obs)

        # Get recent frames
        stacked_obs = list()

        # Get historical frames
        for i in range(self.long_stack_frame):
            idx = -1 - (i + self.lag) * self.stack_frame
            stacked_obs.append(self.history_buffer[idx])
        
        stacked_obs.extend(self.frames)

        return np.stack(stacked_obs), rew, done, info

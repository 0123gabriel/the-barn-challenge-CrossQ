import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_size=int(1e6),
        device="cpu",
        safe_rl=False,
        reward_norm=False,
    ):
        self.max_size = max_size
        self.ptr = 0
        self.mean = 0
        self.reward_norm = reward_norm

        self.safe_rl = safe_rl

        self.state = np.zeros((max_size, *state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, *state_dim))
        self.reward = np.zeros((max_size, 1))
        self.collision_reward = np.zeros((max_size, 1))
        self.terminated = np.zeros((max_size, 1))
        self.truncated = np.zeros((max_size, 1))
        self.task = np.zeros((max_size, 1))

        self.mean = None
        self.std = None

        self.device = device

    def add(
        self,
        state,
        action,
        next_state,
        reward,
        terminated,
        truncated,
        task,
        collision_reward=None,
    ):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.truncated[self.ptr] = truncated
        self.terminated[self.ptr] = terminated
        self.task[self.ptr] = task
        self.ptr = (self.ptr + 1) % self.max_size

        if self.safe_rl:
            assert collision_reward is not None, "collision_reward should not be None"
            self.collision_reward[self.ptr] = collision_reward

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if self.ptr == 1000 and self.reward_norm:  # and self.mean is None:
            rew = self.reward[:1000]
            self.mean, self.std = rew.mean(), rew.std()
            if np.isclose(self.std, 0, 1e-2) or self.std is None:
                self.mean, self.std = 0.0, 1.0
        self.mean, self.std = 0.0, 1.0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        if self.safe_rl:
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.terminated[ind]).to(self.device),
                torch.FloatTensor(self.truncated[ind]).to(self.device),
                torch.FloatTensor(self.task[ind]).to(self.device),
                torch.FloatTensor(self.collisions[ind]).to(self.device),
                ind,
            )
        else:
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),              
                torch.FloatTensor(self.terminated[ind]).to(self.device),
                torch.FloatTensor(self.truncated[ind]).to(self.device),
                torch.FloatTensor(self.task[ind]).to(self.device),
                ind,
            )

    def n_step_return(self, n_step, ind, gamma):
        """
        Compute n-step return for the given index
        #! Check if terminated and truncated are correct
        """
        reward = []
        terminated = []
        truncated = []
        next_state = []
        gammas = []

        if self.safe_rl:
            collision_reward = []

        for i in ind:
            n = 0
            r = 0
            c = 0
            for _ in range(n_step):
                idx = (i + n) % self.size
                assert self.mean is not None
                assert self.std is not None
                r += (self.reward[idx] - self.mean) / self.std * gamma**n
                if self.safe_rl:
                    c += self.collision_reward[idx] * gamma**n
                if not self.terminated[idx] and not self.truncated[idx]:
                    break
                n = n + 1
            next_state.append(self.next_state[idx])
            terminated.append(self.terminated[idx])
            truncated.append(self.truncated[idx])
            reward.append(r)
            gammas.append([gamma ** (n + 1)])
            if self.safe_rl:
                collision_reward.append(c)

        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        gammas = torch.FloatTensor(gammas).to(self.device)
        terminated = torch.FloatTensor(terminated).to(self.device)
        truncated = torch.FloatTensor(truncated).to(self.device)

        if self.safe_rl:
            collision_reward = torch.FloatTensor(np.array(collision_reward)).to(
                self.device
            )
        if self.safe_rl:
            return next_state, reward, gammas, terminated, truncated, collision_reward
        
        else:
            return next_state, reward, gammas, terminated, truncated
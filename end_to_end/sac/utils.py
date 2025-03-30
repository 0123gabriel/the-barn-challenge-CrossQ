import glob
import os
import numpy as np
import torch
import torch.nn as nn
import gym
from envs.wrappers import StackFrame
from torch.distributions import Normal, TransformedDistribution, TanhTransform
import pandas as pd
import random
import os


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
        self.size = 0

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


class StableTanhTransform(TanhTransform):
    def __init__(self, cache_size=1):
        super().__init__(cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def _inverse(self, x):
        return self.atanh(x)

    def __eq__(self, other):
        return isinstance(other, StableTanhTransform)


class SquashedNormal(TransformedDistribution):
    def __init__(self, loc: torch.Tensor, std: torch.Tensor):
        self.loc = loc  # can't use mean because of the property
        self.std = std
        base_distribution = Normal(loc, std)
        super().__init__(base_distribution, StableTanhTransform(), validate_args=False)

    @property
    def mean(self):
        x = self.loc
        for transform in self.transforms:
            x = transform(x)
        return x


warmup_function = {
    "smooth": lambda c, w: min(1.0, c / w),
    "non_smooth": lambda c, w: min(1.0, c // w),
}


class BatchRenorm(nn.Module):
    """
    BatchRenorm - (arxiv.org/abs/1702.03275)

    Args:
        num_features: number of features in input tensor
        momentum: momentum for running statistics
        warmup: number of batches to warmup the batch renorm
        max_r: maximum value for r
        max_d: maximum value for d
        smoothing: smoothing factor for transition from BN to BR
    """

    def __init__(
        self,
        num_features,
        momentum=0.01,
        warmup=100000,
        max_r=3.0,
        max_d=5.0,
        warmup_type="smooth",
    ):
        super(BatchRenorm, self).__init__()
        self.momentum = momentum
        self.warmup = warmup
        self.max_r = max_r
        self.max_d = max_d
        self.smoothing = warmup_function[warmup_type]
        self.batch_size = 0
        self.num_features = num_features
        self.register_buffer("step", torch.zeros(1))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor):
        if not x.dim() >= 2:  # first dim is batch size, second dim is num_features
            raise ValueError("expected 2D input (got {}D input)".format(x.dim()))

        # prepare dimensions for scaling and shifting parameters
        # suppose we have input of shape (batch_size, num_features, height, width) (e.g. (32, 3, 64, 64))
        view_shape = [1, x.shape[1]] + [1] * (x.dim() - 2)  # [1, 3, 1, 1]

        dims = [i for i in range(x.dim()) if i != 1]  # [0, 2, 3]

        running_std = (self.running_var + self.eps).sqrt()

        if self.training:
            mean = x.mean(dims)
            var = x.var(dims, unbiased=False)
            std = (var + self.eps).sqrt()

            r = torch.clamp(std / running_std, 1 / self.max_r, self.max_r)
            d = torch.clamp(
                (mean - self.running_mean) / running_std, -self.max_d, self.max_d
            )

            if self.step < self.warmup:
                # BatchNorm
                smoothing_factor = self.smoothing(self.step.item(), self.warmup)
                r = 1.0 + (r - 1.0) * smoothing_factor
                d = d * smoothing_factor

            # update running statistics
            x = (x - mean.view(view_shape)) / std.view(view_shape) * r.view(
                view_shape
            ) + d.view(view_shape)

            raw_var = var.detach() * x.shape[0] / (x.shape[0] - 1)
            self.running_mean += self.momentum * (mean.detach() - self.running_mean)
            self.running_var += self.momentum * (raw_var - self.running_var)

            self.step += 1

        else:
            # inference time
            x = (x - self.running_mean.view(view_shape)) / running_std.view(view_shape)

        return x * self.weight.view(view_shape) + self.bias.view(view_shape)


class Env_Selector:
    def __init__(self, config, worlds_directory):
        self.config = config
        self.worlds = self.load_worlds_files(worlds_directory)

    def load_worlds_files(self, worlds_dir):
        # TODO FOR CURRICULUM LEARNING: load the selected worlds by CL using a preloaded csv file
        # Get all .world files in the directory
        # world_files = glob.glob(os.path.join(worlds_dir, "world_*.world"))
        # print(world_files)

        world_files = os.listdir(worlds_dir)
        # print(world_files)
        return world_files

    def get_random_world(self):
        choice = np.random.choice(self.worlds)
        # print(choix)
        return "BARN/" + choice

    def get_random_env(self):
        env_config = self.config["env_config"]
        env_config["kwargs"]["init_sim"] = True
        env_config["kwargs"]["world_name"] = self.get_random_world()
        # Randomly choose a reward type from the available options
        reward_types = ["smooth", "lidar", "simple", "mixed", "local", "simple_local"]
        env_config["kwargs"]["reward_function"] = np.random.choice(reward_types)

        info = {
            "world": env_config["kwargs"]["world_name"],
            "reward_scheme": env_config["kwargs"]["reward_function"],
        }
        # if env_config["use_condor"]:
        #    env_config["kwargs"]["init_sim"] = False

        # if not env_config["use_condor"]:
        env = gym.make(env_config["env_id"], **env_config["kwargs"])
        env = StackFrame(env, stack_frame=env_config["stack_frame"])
        # else:
        # If use condor, we want to avoid initializing env instance from the central learner
        # So here we use a fake env with obs_space and act_space information
        #    print("    >>>> Using actors on Condor")
        #    env = InfoEnv(config)
        return env, info
    
    def get_env(self, success_rate=0.0):
        return self.get_random_env()


class Simple_Curriculum:
    def __init__(self, config, worlds_directory):
        self.config = config
        self.worlds_dir = worlds_directory
        self.stage = 0
        self.current_sucess_rate = 0.0
        self.smoothing_factor = 0.3
        

    def get_random_world(self):
        choice = np.random.choice(self.worlds)
        # print(choix)
        return "BARN/" + choice

    def get_env_with_params(self, fill_pct_range, distance_range):
        # Look through the csv file in the collumns ("fill_pct", "distance") to get a random index that matches
        # Load the CSV file with world metadata
        csv_path = os.path.join(self.worlds_dir, "worlds_metadata.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata CSV file not found at {csv_path}")

        df = pd.read_csv(csv_path)

        # Filter worlds that match the criteria
        matching_worlds = df[
            (df["fill_pct"] >= fill_pct_range[0])
            & (df["fill_pct"] <= fill_pct_range[1])
            & (df["distance"] >= distance_range[0])
            & (df["distance"] <= distance_range[1])
        ]

        if matching_worlds.empty:
            # Fallback if no worlds match the criteria
            print(
                "============== No worlds match the specified ranges. =============="
            )
            raise NotImplementedError
        else:
            # Select a random world from the matching ones
            world_id = random.choice(matching_worlds["ID"].tolist())
            world_name = os.path.join(self.worlds_dir, f"world_{world_id}.world")
        return world_name
    
    def get_env(self, success_rate = 0.0):
        self.current_sucess_rate = self.smoothing_factor * success_rate + (1 - self.smoothing_factor) * self.current_sucess_rate
        
        env_config = self.config["env_config"]
        env_config["kwargs"]["init_sim"] = True
        
        # Define curriculum stages with their corresponding parameters
        curriculum_stages = [
            {"fill_pct_range": [0.0, 0.10], "distance_range": [10, 15], "init_position": [-2, 4, 1.57], "goal_position": [0, 3.5, 0]},  # Stage 0
            {"fill_pct_range": [0.10, 0.20], "distance_range": [10, 15], "init_position": [-2, 4, 1.57], "goal_position": [0, 5.5, 0]}, # Stage 1
            {"fill_pct_range": [0.0, 0.10], "distance_range": [15, 25], "init_position": [-2, 4, 1.57], "goal_position": [0, 7, 0]},  # Stage 2
            {"fill_pct_range": [0.1, 0.2], "distance_range": [15, 25], "init_position": [-2, 4, 1.57], "goal_position": [0, 7, 0]},   # Stage 3
            {"fill_pct_range": [0, 0.2], "distance_range": [25, 40], "init_position": [-2, 3, 1.57], "goal_position": [0, 10, 0]},     # Stage 4
            {"fill_pct_range": [0, 0.3], "distance_range": [10, 40], "init_position": [-2, 3, 1.57], "goal_position": [0, 10, 0]},     # Stage 5
        ]
        
        # Final stage parameters for all stages >= 6
        final_stage_params = {"fill_pct_range": [0.1, 0.35], "distance_range": [25, 40], "init_position": [-2, 3, 1.57], "goal_position": [0, 10, 0]}
        
        # Increment stage if success rate is high enough
        if self.current_sucess_rate > 0.8:
            self.stage += 1
            self.current_sucess_rate = 0.0
            
        # Get parameters for the current stage
        if self.stage < len(curriculum_stages):
            params = curriculum_stages[self.stage]
        else:
            params = final_stage_params
            
        world_name = self.get_env_with_params(
            fill_pct_range=params["fill_pct_range"], 
            distance_range=params["distance_range"]
        )
        
        #init_pos_offset = round(random.uniform(-1.5, 1.25), 2) # Random x offset
        
        env_config["kwargs"]["world_name"] = world_name
        reward_types = ["smooth", "lidar", "simple", "mixed", "local", "simple_local"]
        env_config["kwargs"]["reward_function"] = np.random.choice(reward_types)
        env_config["kwargs"]["init_position"] = params["init_position"]
        env_config["kwargs"]["goal_position"] = params["goal_position"]

        info = {
            "world": env_config["kwargs"]["world_name"],
            "reward_scheme": env_config["kwargs"]["reward_function"],
            "stage": self.stage,
            "success_rate": self.current_sucess_rate,
        }

        env = gym.make(env_config["env_id"], **env_config["kwargs"])
        env = StackFrame(env, stack_frame=env_config["stack_frame"])

        return env, info
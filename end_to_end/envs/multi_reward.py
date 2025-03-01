from envs.motion_control_envs import MotionControlContinuousLaser

class MultiRewardEnv(MotionControlContinuousLaser):
    def __init__(self, reward_function,  *args, **kwargs):
        
        #TODO: Revise this
        self.slack_reward = -1
        self.failure_reward = -50
        
        # Dictionary of available reward functions #TODO: implement reward functions
        self.reward_functions = {
            "sparse": self._sparse_reward,
            "dense": self._dense_reward,
            "distance": self._distance_based_reward,
            # Add more reward functions as needed
        }
        
        # Set the reward function based on the argument
        if reward_function in self.reward_functions:
            self.reward_func = self.reward_functions[reward_function]
        else:
            raise ValueError(f"Reward function '{reward_function}' not found. Available options: {list(self.reward_functions.keys())}")
        
        self.success_reward = 0
        self.collision_reward = 0
        self.goal_reward = 1
        self.max_collision = 10000
        self.max_step = 100
        self.time_step = 1
        self.verbose = True
        self.world_frame_goal = (
            self.init_position[0] + self.goal_position[0],
            self.init_position[1] + self.goal_position[1],
        )
        
    def step(self, action):
        # TODO: add to infos the reward function used, and the enviroment parameters
        # TODO: implement the reward 
            # TODO: Implement smoothness rewards
            # TODO: Implement speed rewards
            # TODO: Implement collision penalties
            # TODO: Implement obstacle distance penalty
            # TODO: Implement global goal distance reward
            # TODO: Implement going straight reward > turning rewards
            # TODO: Implement time penalty
            # TODO: Implement local goal direction reward
        return super().step(action)
    
    # TODO: implement total reward functions

    def switch_reward_function(self, reward_function):
        if reward_function in self.reward_functions:
            self.reward_func = self.reward_functions[reward_function]
        else:
            raise ValueError(f"Reward function '{reward_function}' not found. Available options: {list(self.reward_functions.keys())}")
        

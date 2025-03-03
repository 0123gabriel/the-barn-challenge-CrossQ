from envs.motion_control_envs import MotionControlContinuousLaser
import numpy as np
from tf.transformations import euler_from_quaternion


class MultiRewardEnv(MotionControlContinuousLaser):
    def __init__(self, reward_function, *args, **kwargs):
        # TODO: Revise this
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
            raise ValueError(
                f"Reward function '{reward_function}' not found. Available options: {list(self.reward_functions.keys())}"
            )

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
        # TODO: Implement smoothness rewards - Bruno
        # TODO: Implement speed rewards - Bruno
        # TODO: Implement collision penalties - Gabriel
        # TODO: Implement obstacle distance penalty - Gabriel
        # TODO: Implement global goal distance reward - Gabriel
        # TODO: Implement going straight reward > turning rewards - Bruno
        # TODO: Implement time penalty - Bruno
        # TODO: Implement local goal direction reward - Gabriel
        pass

    # TODO: implement total reward functions

    def _smoothness_reward(self, current_pos, current_psi, next_pos, next_psi):
        # ? pos, psi = self._get_pos_psi() (check how to get the next position and psi)
        n_x_i = np.array([np.cos(current_psi), np.sin(current_psi)])
        n_x_i_plus_1 = np.array([np.cos(next_psi), np.sin(next_psi)])

        # pos_vector
        pos_i = np.array([current_pos.x, current_pos.y])
        pos_i_plus_1 = np.array([next_pos.x, next_pos.y])

        # vector product between the two vectors
        F = np.cross(n_x_i + n_x_i_plus_1, pos_i_plus_1 - pos_i)

        reward = 0.001 - np.norm(F)
        return reward

    def _speed_reward_soft(self, last_vel, current_vel, max_vel, alpha=10, beta=20):
        reward = (
            0.001 * 1 / (1 + np.exp(-alpha * (current_vel - max_vel / 2)))
            - beta * (current_vel - last_vel) ** 2
        )
        return reward
    
    def _speed_reward_simple(self, current_vel, max_vel, limit = 0.09):
        return np.clip(current_vel / max_vel, -0.01, limit)
    
    def _going_straight_reward(self, current_psi, goal_psi, alpha=10):
        if np.abs(current_psi - goal_psi) < 0.01:
            return 0.2
        else:
            return 0.05

    def _time_penalty(self, step_count, max_step):
        return -0.01

    def _obs_dist_reward(self):
        laser_data = self.gazebo_sim.get_laser_scan()
        min_distance = min(laser_data)
        min_dist = (
            0.34 + 0.2
        )  # Radius of Jackal condering a radius from the lidar to the front right corner

        if min_distance < min_dist:
            reward = -0.01
        else:
            reward = 0.01
        return reward

    def loca_goal_dir_reward(self):
        pass

    def switch_reward_function(self, reward_function):
        if reward_function in self.reward_functions:
            self.reward_func = self.reward_functions[reward_function]
        else:
            raise ValueError(
                f"Reward function '{reward_function}' not found. Available options: {list(self.reward_functions.keys())}"
            )

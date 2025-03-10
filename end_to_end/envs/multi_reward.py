from envs.motion_control_envs import MotionControlContinuous
from envs.jackal_gazebo_envs import JackalGazeboLaser

import numpy as np
from geometry_msgs.msg import Point

class MultiRewardEnv(MotionControlContinuous, JackalGazeboLaser):
    def __init__(self, reward_function = 'simple', **kwargs):
        super().__init__(**kwargs)

        self.reward_functions = {
            "smooth": self.smooth_reward_scheme,
            "lidar": self.lidar_based_scheme,
            "simple": self.simple_reward_scheme,
            "mixed": self.mixed_scheme,
            # Add more reward functions as needed
        }

        # Set the reward function based on the argument
        self.reward_scheme_name = reward_function
        if self.reward_scheme_name in self.reward_functions:
            self.reward_func = self.reward_functions[self.reward_scheme_name]
        else:
            raise ValueError(
                f"Reward function '{reward_function}' not found. Available options: {list(self.reward_functions.keys())}"
            )

        # self.max_vel = self.range_dict['linear_velocity'][1]
        # self.success_reward = 0
        # self.collision_reward = 0
        # self.goal_reward = 1
        # self.max_collision = 10000
        # self.max_step = 100
        # self.time_step = 1
        # self.verbose = True
        # self.world_frame_goal = (
        #     self.init_position[0] + self.goal_position[0],
        #     self.init_position[1] + self.goal_position[1],
        # )
        
        #self.prev_pos = Point(x=0.0, y=0.0, z=0.0)
        #self.prev_psi = 0

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
        # Get previous velocity, position and orientation
        #prev_pos, prev_psi = self._get_pos_psi()
        #prev_vel = self.gazebo_sim.get_velocity()
        
        # Add some random computations to increase load
        # for _ in range(100):
        #     x = np.random.random((100, 100))
        #     y = np.random.random((100, 100))
        #     z = np.dot(x, y)
        #     w = np.linalg.svd(z)
        
        # step the simulation
        self._take_action(action)
        self.step_count += 1
        pos, psi = self._get_pos_psi()
        #vel = self.gazebo_sim.get_velocity()
        prev_vel = 1
        vel = 1

        # self.gazebo_sim.unpause()
        # compute observation
        obs = self._get_observation(pos, psi, action)

        # compute termination
        flip = pos.z > 0.1  # robot flip

        goal_pos = np.array(
            [self.world_frame_goal[0] - pos.x, self.world_frame_goal[1] - pos.y]
        )
        success = np.linalg.norm(goal_pos) < 0.4

        truncation = self.step_count >= self.max_step  # Timeout

        collided = self.gazebo_sim.get_hard_collision() and self.step_count > 1
        self.collision_count += int(collided)

        termination = flip or success or self.collision_count >= self.max_collision
        rew = 1

        # rew = self.reward_func(
        #     prev_vel,
        #     vel,
        #     pos,
        #     self.prev_pos,
        #     self.prev_psi,
        #     psi,
        #     success,
        #     collided,
        #     truncation,
        #     goal_pos,
        # )  #! Improve once reward schemes are implemented

        self.last_goal_pos = goal_pos

        info = dict(
            collision=self.collision_count,
            collided=collided,
            goal_position=goal_pos,
            time=self.current_time - self.start_time,
            success=success,
            world=self.world_name,
            reward_function=self.reward_scheme_name,
        )

        if truncation or termination:
            bn, nn = self.gazebo_sim.get_bad_vel_num()

        return obs, rew, termination, truncation, info

    # TODO: implement total reward functions
    def _time_penalty(self, max_step):
        return -1 / max_step

    def simple_reward_scheme(
        self,
        prev_vel,
        vel,
        pos,
        prev_pos,
        prev_psi,
        psi,
        success,
        collided,
        truncation,
        goal_pos,
    ):
        c_1 = 0.001
        c_2 = 0.02
        r = c_1 * (np.linalg.norm(self.last_goal_pos) - np.linalg.norm(goal_pos)) + c_2

        if collided:
            r += self.collision_reward
        if success:
            r += self.success_reward
        if truncation:
            r += self.failure_reward

        return r

    def smooth_reward_scheme(
        self,
        prev_vel,
        vel,
        pos,
        prev_pos,
        prev_psi,
        psi,
        success,
        collided,
        truncation,
        goal_pos,
    ):
        # time penalty
        r = self._time_penalty(self.step_count, self.max_step)
        # smoothness reward
        r += self._smoothness_reward(prev_pos, prev_psi, pos, psi)

        # speed reward
        r += self._speed_reward_soft(prev_vel, vel, self.max_vel)

        # reward for getting closer
        r += self._goal_approach_reward(goal_pos)

        # termination rewards
        if collided:
            r += self.collision_reward
        elif success:
            r += self.success_reward
        elif truncation:
            r += self.failure_reward

        return r

    def lidar_based_scheme(
        self, goal_pos, prev_pos, pos, curr_vel, collided, success, truncation
    ):
        # Stop reward
        r = self._stop_reward(prev_pos, pos)
        # R forward and R turn
        r += self._going_straight_reward(goal_pos)
        # R vel
        r += self._speed_reward_simple(curr_vel, self.max_vel)
        # Reward for distance to obstacle
        r += self._obs_dist_reward()

        # termination rewards
        if collided:
            r += self.collision_reward
        elif success:
            r += self.success_reward
        elif truncation:
            r += self.failure_reward

        return r

    def mixed_scheme(
        self,
        goal_pos,
        prev_pos,
        prev_psi,
        pos,
        psi,
        prev_vel,
        vel,
        collided,
        success,
        truncation,
    ):
        c_1 = 0.001
        c_2 = 0.02
        r = c_1 * (np.linalg.norm(self.last_goal_pos) - np.linalg.norm(goal_pos)) + c_2

        # time penalty
        r = self._time_penalty(self.step_count, self.max_step)
        # smoothness reward
        r += self._smoothness_reward(prev_pos, prev_psi, pos, psi)

        # speed reward
        r += self._speed_reward_soft(prev_vel, vel, self.max_vel)

        # reward for getting closer
        r += self._goal_approach_reward(goal_pos)

        # Stop reward
        r = self._stop_reward(prev_pos, pos)
        # R forward and R turn
        r += self._going_straight_reward(goal_pos)
        # Reward for distance to obstacle
        r += self._obs_dist_reward()

        # termination rewards
        if collided:
            r += self.collision_reward
        elif success:
            r += self.success_reward
        elif truncation:
            r += self.failure_reward

        return r

    def _smoothness_reward(self, current_pos, current_psi, next_pos, next_psi):
        """
        Calculate a smoothness reward based on the vehicle's trajectory.
        This function evaluates how smoothly the vehicle is moving by comparing the current and next position and orientation.
        It uses a cross product between the combined orientation vectors and the position difference to assess smoothness.
        based on:  DOI 10.1109/TIV.2024.3444854
        Parameters:
            current_pos (Point): Current position of the vehicle (containing x, y coordinates)
            current_psi (float): Current heading angle of the vehicle in radians
            next_pos (Point): Next position of the vehicle (containing x, y coordinates)
            next_psi (float): Next heading angle of the vehicle in radians
        Returns:
            float: A reward value where higher values indicate smoother trajectories.
                   The reward is calculated as (0.001 - |F|), where F is the cross product
                   between the sum of normalized orientation vectors and the position difference vector.
        """
        # ? pos, psi = self._get_pos_psi() (check how to get the next position and psi) (maybe get the previous position and psi)
        n_x_i = np.array([np.cos(current_psi), np.sin(current_psi)])
        n_x_i_plus_1 = np.array([np.cos(next_psi), np.sin(next_psi)])

        # pos_vector
        pos_i = np.array([current_pos.x, current_pos.y])
        pos_i_plus_1 = np.array([next_pos.x, next_pos.y])

        # vector product between the two vectors
        F = np.cross(n_x_i + n_x_i_plus_1, pos_i_plus_1 - pos_i)

        reward = 0.001 - np.linalg.norm(F)
        return reward

    def _speed_reward_soft(self, last_vel, current_vel, max_vel, alpha=10, beta=20):
        reward = (
            0.001 * 1 / (1 + np.exp(-alpha * (current_vel - max_vel / 2)))
            - beta * (current_vel - last_vel) ** 2
        )
        return reward

    def _speed_reward_simple(self, current_vel, max_vel, limit=0.09):
        return np.clip(current_vel / max_vel, -0.01, limit)

    def _going_straight_reward(self, current_psi, prev_psi, alpha=10):
        if np.abs(current_psi - prev_psi) < 0.01:
            return 0.2
        else:
            return 0.05

    def _time_penalty(self, step_count, max_step):
        return -0.01

    def _obs_dist_reward(self, alpha=0.1):
        laser_data = self.gazebo_sim.get_laser_scan()
        ranges = np.array(laser_data.ranges)
        valid = (ranges > 0) & (ranges != np.inf)
        if np.any(valid):
            min_distance = np.min(ranges[valid])
        else:
            min_distance = float('inf')
        return min(0.01, alpha * min_distance)

        # min_dist = 0.34 + 0.2 # Radius of Jackal condering a radius from the lidar to the front right corner plus min dist of 0.2

        # if min_distance < min_dist:
        #     reward = -0.001
        # else:
        #     reward = 0.001
        # return reward

    def _local_goal_dir_reward(self):
        local_g_x = self.gazebo_sim.local_goal[0]
        local_g_y = self.gazebo_sim.local_goal[1]

        robot_pos, psi = self._get_pos_psi()
        robot_x = robot_pos.x
        robot_y = robot_pos.y

        # Calculate vector from robot to local goal
        local_goal_vector = np.array([local_g_x - robot_x, local_g_y - robot_y])

        # Get unit vector
        unit_goal_vector = local_goal_vector / np.linalg.norm(local_goal_vector)

        # Create unit vector at robot's heading angle
        robot_heading = np.array([np.cos(psi), np.sin(psi)])

        # Calculate dot product between vectors
        alignment = np.dot(unit_goal_vector, robot_heading)

        # Convert to reward (-1 to 1 range)
        reward = alignment * 0.001
        return reward

    def _global_goal_dist_reward(self):
        goal_pos = self.move_base.goal_position
        robot_pos, _ = self._get_pos_psi()
        robot_x = robot_pos.x
        robot_y = robot_pos.y
        # Calculate Euclidean distance between robot and goal
        distance = np.sqrt((goal_pos[0] - robot_x) ** 2 + (goal_pos[1] - robot_y) ** 2)

        # Convert distance to reward (closer = higher reward)
        reward = 0.001 * (1 / (distance + 1))  # Adding 1 to avoid division by zero
        return reward

    def _collision_reward(self):
        collided = self.gazebo_sim.get_hard_collision() and self.step_count > 1
        reward = 0
        if collided:
            reward += self.collision_reward

        return reward

    def _stop_reward(self, prev_pos, pos):
        distance = np.linalg.norm([prev_pos.x - pos.x, prev_pos.y - pos.y])
        if distance < 0.01:
            reward = -1
        else:
            reward = 0

        return reward

    def _goal_approach_reward(self, goal_pos):
        getting_closer = (
            np.linalg.norm(self.last_goal_pos) - np.linalg.norm(goal_pos)
        ) < 0
        reward = 0.001 * getting_closer
        return reward

    def switch_reward_function(self, reward_function):
        if reward_function in self.reward_functions:
            self.reward_func = self.reward_functions[reward_function]
        else:
            raise ValueError(
                f"Reward function '{reward_function}' not found. Available options: {list(self.reward_functions.keys())}"
            )

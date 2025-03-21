from envs.motion_control_envs import MotionControlContinuous
from envs.jackal_gazebo_envs import JackalGazeboLaser

import numpy as np
import wandb
from geometry_msgs.msg import Point

log_dir = "logs" # TODO set this on the config file

class MultiRewardEnv(MotionControlContinuous, JackalGazeboLaser):
    def __init__(self, reward_function = 'mixed', use_wandb=False, **kwargs):
        super().__init__(**kwargs)

        self.use_wandb = use_wandb
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

        self.max_vel = self.range_dict['linear_velocity'][1]
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
        
        self.prev_pos = Point(x=-2.0, y=3.0, z=0.0)
        self.prev_psi = 1.57
        self.prev_vel = 0
        
        if use_wandb:  # TODO: this should be in the main file
            wandb.init(
                project="gazebo_rewards",  # set this from config file
                name=self.reward_scheme_name,
                config={
                    "reward_scheme": self.reward_scheme_name,
                },
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
        pos, psi = self._get_pos_psi() # Returns the position in the world frame
        print('Position: ', pos, 'Orientation: ', psi, '\n')
        vel = self.gazebo_sim.get_velocity()

        if self.use_wandb:
            wandb.log({
                "robot_x": pos.x,
                "robot_y": pos.y
            })

        # self.gazebo_sim.unpause()
        # compute observation
        obs = self._get_observation(pos, psi, action)

        # compute termination
        flip = pos.z > 0.1  # robot flip

        global_goal_pos = np.array(
            [self.world_frame_goal[0] - pos.x, self.world_frame_goal[1] - pos.y]
        )
        
        success = np.linalg.norm(global_goal_pos) < 1.0
        #print('Norm to the goal: ', np.linalg.norm(global_goal_pos))
        #print('X pos: ', pos.x)
        #print('Y pos: ', pos.y)

        truncation = self.step_count >= self.max_step  # Timeout

        collided = self.gazebo_sim.get_hard_collision() and self.step_count > 1
        if collided:
            print('Collided ==================================================================================================')
        self.collision_count += int(collided)
        #print(self.collision_count)

        termination = flip or success or self.collision_count >= self.max_collision
        #rew = 1

        rew = self.reward_func(
            self.prev_vel,
            vel,
            pos,
            self.prev_pos,
            self.prev_psi,
            psi,
            success,
            collided,
            truncation,
            global_goal_pos,
        )  #! Improve once reward schemes are implemented
        
        smooth_reward = self.smooth_reward_scheme(
            self.prev_vel,
            vel,
            pos,
            self.prev_pos,
            self.prev_psi,
            psi,
            success,
            collided,
            truncation,
            global_goal_pos,
        )
        
        lidar_reward = self.lidar_based_scheme(
            self.prev_vel,
            vel,
            pos,
            self.prev_pos,
            self.prev_psi,
            psi,
            success,
            collided,
            truncation,
            global_goal_pos,
        )
        
        simple_reward = self.simple_reward_scheme(
            self.prev_vel,
            vel,
            pos,
            self.prev_pos,
            self.prev_psi,
            psi,
            success,
            collided,
            truncation,
            global_goal_pos,
        )

        if self.use_wandb:
                wandb.log(
                    {
                        "mixed_reward": rew,
                        "smooth_reward": smooth_reward,
                        "lidar_reward": lidar_reward,
                        "simple_reward": simple_reward
                    }
                )

        self.last_goal_pos = global_goal_pos

        # Update self.prev_pos, self.prev_psi and self.prev_vel
        self.prev_pos = pos
        self.prev_psi = psi  
        self.prev_vel = vel

        info = dict(
            collision=self.collision_count,
            collided=collided,
            goal_position=global_goal_pos,
            time=self.current_time - self.start_time,
            success=success,
            world=self.world_name,
            reward_function=self.reward_scheme_name,
        )

        if truncation or termination:
            bn, nn = self.gazebo_sim.get_bad_vel_num()

        # self.gazebo_sim.pause()
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
        global_goal_pos,
    ):
        c_1 = 1
        c_2 = -0.3
        r_simple = c_1 * (np.linalg.norm(self.last_goal_pos) - np.linalg.norm(global_goal_pos)) + c_2

        r_final = 0
        if collided:
            r_final += self.collision_reward
        if success:
            r_final += self.success_reward
        if truncation:
            r_final += self.failure_reward

        if self.use_wandb:
                wandb.log(
                    {
                        "reward_breakdown/simple_reward": r_simple,
                        "reward_breakdown/termination_reward": r_final
                    }
                )

        return r_simple + r_final

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
        global_goal_pos,
    ):
        # time penalty
        r_time = self._time_penalty(self.step_count, self.max_step)
        
        # smoothness reward
        r_smooth = self._smoothness_reward(prev_pos, prev_psi, pos, psi)

        # speed reward
        r_speed = self._speed_reward_soft(prev_vel, vel, self.max_vel)

        # reward for getting closer
        r_approach = self._goal_approach_reward(global_goal_pos)

        r_final = 0
        # termination rewards
        if collided:
            r_final = self.collision_reward
        elif success:
            r_final = self.success_reward
        elif truncation:
            r_final = self.failure_reward

        if self.use_wandb:
                wandb.log(
                    {
                        "time_penalty": r_time,
                        "smoothness_reward": r_smooth,
                        "speed_reward": r_speed,
                        "approach_reward": r_approach,
                        "termination_reward": r_final
                    }
                )

        return r_time + r_smooth + r_speed + r_approach + r_final

    def lidar_based_scheme(
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
        global_goal_pos,
    ):
        
        # Stop reward
        r_stop = self._stop_reward(prev_pos, pos)
        #print(r)
        # R forward and R turn
        r_straight = self._going_straight_reward(prev_psi, psi)
        #print(r)
        # R vel
        r_speed_simple = self._speed_reward_simple(vel, self.max_vel)
        #print(r)
        # Reward for distance to obstacle
        r_obs_dist = self._obs_dist_reward()
        #print(r)

        r_final = 0
        # termination rewards
        if collided:
            r_final = self.collision_reward
        elif success:
            r_final = self.success_reward
        elif truncation:
            r_final = self.failure_reward
        
        if self.use_wandb:
            wandb.log({
                "lidar/stop_reward": r_stop,
                "lidar/straight_reward": r_straight,
                "lidar/speed_reward": r_speed_simple,
                "lidar/obstacle_distance_reward": r_obs_dist,
                "lidar/termination_reward": r_final
            })

        return r_stop + r_straight + r_speed_simple + r_obs_dist + r_final

    def mixed_scheme(
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
        global_goal_pos,
    ):
        # Simple reward component
        c_1 = 1
        c_2 = -0.3
        r_simple = c_1 * (np.linalg.norm(self.last_goal_pos) - np.linalg.norm(global_goal_pos)) + c_2

        # Time and smoothness components
        r_time = self._time_penalty(self.step_count, self.max_step)
        r_smooth = self._smoothness_reward(prev_pos, prev_psi, pos, psi)
        r_speed = self._speed_reward_soft(prev_vel, vel, self.max_vel)
        r_approach = self._goal_approach_reward(global_goal_pos)

        # Lidar-based components  
        r_stop = self._stop_reward(prev_pos, pos)
        r_straight = self._going_straight_reward(prev_psi, psi)
        r_obs_dist = self._obs_dist_reward()

        # Terminal rewards
        r_final = 0
        if collided:
            r_final = self.collision_reward
        elif success:
            r_final = self.success_reward
        elif truncation:
            r_final = self.failure_reward

        if self.use_wandb:
            wandb.log({
            "simple_reward": r_simple, #done
            "time_penalty": r_time, # done
            "smoothness_reward": r_smooth, #done
            "speed_reward": r_speed, #done
            "approach_reward": r_approach,
            "stop_reward": r_stop, #done
            "straight_reward": r_straight,
            "obstacle_distance": r_obs_dist, #done
            "termination_reward": r_final
            })

        total_reward = r_simple + r_time + r_smooth + r_speed + r_approach + r_stop + r_straight + r_obs_dist + r_final
        return total_reward

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
            0.001 * (1 / (1 + np.exp(-alpha * (current_vel - max_vel / 2)))
            - beta * (current_vel - last_vel) ** 2)
        )
        return reward

    def _speed_reward_simple(self, current_vel, max_vel, limit=0.4):
        return np.clip(current_vel / max_vel, -0.01, limit)*0.25

    def _going_straight_reward(self, current_psi, prev_psi, alpha=10):
        if np.abs(current_psi - prev_psi) < 0.01:
            return 0.2
        else:
            return 0.05

    def _time_penalty(self, step_count, max_step):
        return -0.01

    def _obs_dist_reward(self, alpha=0.2):
        laser_data = self.gazebo_sim.get_laser_scan()
        ranges = np.array(laser_data.ranges)
        valid = (ranges > 0) & (ranges != np.inf)
        if np.any(valid):
            min_distance = np.min(ranges[valid])
        else:
            min_distance = float('inf')  
        #print(min_distance)
        return - 1/(min_distance + 1e-8) * alpha + 0.1
        
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
            reward = -0.5
        else:
            reward = 0

        return reward

    def _goal_approach_reward(self, global_goal_pos):
        getting_closer = (
            np.linalg.norm(self.last_goal_pos) - np.linalg.norm(global_goal_pos)
        ) > 0
        reward = 0.001 * getting_closer
        return reward

    def switch_reward_function(self, reward_function):
        if reward_function in self.reward_functions:
            self.reward_func = self.reward_functions[reward_function]
        else:
            raise ValueError(
                f"Reward function '{reward_function}' not found. Available options: {list(self.reward_functions.keys())}"
            )

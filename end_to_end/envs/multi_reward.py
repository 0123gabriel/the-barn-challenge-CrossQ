from envs.motion_control_envs import MotionControlContinuous
from envs.jackal_gazebo_envs import JackalGazeboLaser

import numpy as np
import wandb
from geometry_msgs.msg import Point, Pose
import random
import rospy
import copy

log_dir = "logs"  # TODO set this on the config file


class MultiRewardEnv(MotionControlContinuous, JackalGazeboLaser):
    def __init__(self, reward_function="mixed", use_wandb=True, switch_schemes=False, **kwargs):
        super().__init__(**kwargs)

        self.use_wandb = use_wandb
        self.reward_functions = {
            "smooth": self.smooth_reward_scheme,
            "lidar": self.lidar_based_scheme,
            "simple": self.simple_reward_scheme,
            "mixed": self.mixed_scheme,
            "local": self.local_focused_scheme,
            "simple_local": self.simple_local_scheme
        }
        
        self.switch_schemes = switch_schemes

        # Set the reward function based on the argument
        self.reward_scheme_name = reward_function
        if self.reward_scheme_name in self.reward_functions:
            self.reward_func = self.reward_functions[self.reward_scheme_name]
        else:
            raise ValueError(
                f"Reward function '{reward_function}' not found. Available options: {list(self.reward_functions.keys())}"
            )

        self.max_vel = self.range_dict["linear_velocity"][1]

        self.prev_pos = None
        self.prev_psi = None
        self.prev_vel = None
        self.prev_local_goal = None
        self.x_offset = 0

    def reset(self):
        """
        reset the environment
        """

        if self.switch_schemes:
            random_key = random.choice(list(self.reward_functions.keys()))
            self.switch_reward_function(random_key)
        
        self.step_count = 0
        self.collision_count = 0
        
        if self.num_resets > 0:
            self.x_offset = np.random.uniform(-1.5, 1.25)
            
            init_pos = [self.init_position[0] + self.x_offset, 
                        self.init_position[1],
                        self.init_position[2]]
            
            goal_pos = [self.goal_position[0] - self.x_offset, 
                        self.goal_position[1],
                        self.goal_position[2]]
            
            self.gazebo_sim.reset_init_model_state(init_pos)
            self.gazebo_sim.reset()
            self.move_base.reset_global_goal(goal_pos)
            # self.move_base.set_global_goal() (unccoment to use move_base)
        
        self.start_time = self.current_time = rospy.get_time()
        
        self.move_base.reset_robot_in_odom() # get_pos_psi and reset robot in odom are not related but this goes first for clarity
        
        pos, psi = self._get_pos_psi()
        self.prev_pos = pos
        self.prev_psi = psi
        self.prev_vel = 0
        
        self._clear_costmap() # It goes first to remove any residual costmap from the previous episode
        self.move_base.make_plan()
        
        obs = self._get_observation(pos, psi, np.array([0, 0]))
        local_goal, dist_local_goal = self.move_base.get_local_goal()
        # local_goal.position.x -= self.init_position[0]
        # local_goal.position.y -= self.init_position[0]
        local_goal = self.transform_goal([local_goal.position.x , local_goal.position.y], pos, psi)
        #obs = np.concatenate((obs, np.array([local_goal.position.x, local_goal.position.y])))
        obs = np.concatenate((obs, np.array(local_goal)))
        
        # TODO: Check local goal correct position in world frame (do the transformation)
        # world frame is in the gazebo and pos is in the gazebo frame
        # goal poas is centered in the robot position
        global_goal_pos = np.array([self.world_frame_goal[0] - pos.x, self.world_frame_goal[1] - pos.y]) # In the gazebo frame/cheack if we use to get psi
        self.last_goal_pos = global_goal_pos
        self.prev_local_goal = local_goal
        
        self.num_resets += 1        
        return obs

    def step(self, action):
        
        #print('ACTION:', action, '===================================================================================')
        self._take_action(action) # We are in the state t, and before the action we were in t -1
        self.step_count += 1
        pos, psi = self._get_pos_psi()  # Position of the robot in the gazebo frame in t
        #print('Robot Position:', pos, '===================================================================================')    
        vel = self.gazebo_sim.get_velocity() # Velocity in time t (v_t)

        if self.use_wandb:
            wandb.log({"robot_x": pos.x, "robot_y": pos.y})

        # compute observation
        obs = self._get_observation(pos, psi, action) # obts in t
        local_goal_o, dist_local_goal = self.move_base.get_local_goal() # Local goal in odom frame and it is in time t 
        local_goal_o = np.array([local_goal_o.position.x, local_goal_o.position.y])
        #print('Local goal from move base, odom???:', local_goal_o, '====================================================')
        local_goal_g = self.trans_from_o_to_g(self.init_position, local_goal_o)
        #print('Local goal in gazebo frame:', local_goal_g, '====================================================')
        local_goal_r = self.transform_goal(local_goal_g, pos, psi)
        #print('Local goal in robot frame:', local_goal_r, '====================================================')
        # init_position is in gazebo frame
        local_goal_o = self.transform_goal_inv(self.init_position, local_goal_r, pos, psi)
        #print('Local goal in move base, odom??? frame:', local_goal_o, '====================================================================')
        
        # TODO: change to robot frame and change back for visualization
        self.gazebo_sim.visualize_local_goals(local_goal_o[0], local_goal_o[1])
        obs = np.concatenate((obs, local_goal_r/1.5)) # this changes dimensions from 724 to 726, and 1.5 is to normalize (-1, 1)
        
        # compute terminal
        flip = pos.z > 0.1  # robot flip

        # Distance to the goal from the robot in time t
        global_goal_pos = np.array(
            [self.world_frame_goal[0] - pos.x, self.world_frame_goal[1] - pos.y]
        )

        success = np.linalg.norm(global_goal_pos) < 0.5

        truncation = self.step_count >= self.max_step  # Timeout

        collided = (self.gazebo_sim.get_hard_collision() and self.step_count > 1) or np.linalg.norm(global_goal_pos) > 15

        self.collision_count += int(collided)

        termination = flip or success or self.collision_count >= self.max_collision or self.collision_with_lidar()
        # rew = 1

        # TODO: Check schemes
        rew, rew_info = self.reward_func(
            self.prev_vel,
            vel,
            self.prev_pos,
            pos,
            self.prev_psi,
            psi,
            success,
            collided,
            truncation,
            global_goal_pos,
            local_goal_g
        ) 

        self.last_goal_pos = global_goal_pos

        # Update self.prev_pos, self.prev_psi and self.prev_vel
        self.prev_pos = pos
        self.prev_psi = psi
        self.prev_vel = vel
        self.prev_local_goal = local_goal_g

        # Encode the reward function name into a number from 1 to 6
        encoded_number = list(self.reward_functions.keys()).index(self.reward_scheme_name) + 1
        

        info = dict(
            reward=rew,
            ep_step=self.step_count,
            distance_to_goal=np.linalg.norm(global_goal_pos),
            collision=self.collision_count,
            collided=collided,
            goal_position=global_goal_pos,
            ep_time=self.current_time - self.start_time,
            success=success,
            world=self.world_name,
            reward_function=self.reward_scheme_name,
            reward_function_numeric=encoded_number,
            linear_vel=action[0],
            angular_vel=action[1]
        )
        info.update(rew_info)

        # if truncation or termination:
        #     bn, nn = self.gazebo_sim.get_bad_vel_num()

        return obs, rew, termination, truncation, info

    # def _time_penalty(self, max_step):
        # return -1 / max_step

    def get_next_pos_psi(self, action, pos, psi):
        
        if action[1] == 0:  # Straight-line motion
            x_new = pos.x + action[1] * np.cos(psi) * self.time_step
            y_new = pos.y + action[1] * np.sin(psi) * self.time_step
            theta_new = psi
        else:  # Arc motion
            x_new = pos.x + (action[0] / action[1]) * (np.sin(psi + action[1] * self.time_step) - np.sin(psi))
            y_new = pos.y - (action[0] / action[1]) * (np.cos(psi + action[1] * self.time_step) - np.cos(psi))
            theta_new = psi + action[1] * self.time_step
            
        return x_new, y_new, theta_new

    def local_focused_scheme(
        self,
        prev_vel,
        vel,
        prev_pos,
        pos,
        prev_psi,
        psi,
        success,
        collided,
        truncation,
        global_goal_pos,
        local_goal_g,
    ):
        # Get local goal information
        r_smooth = self._smoothness_reward(prev_pos, prev_psi, pos, psi)
        r_time = self._time_penalty(self.step_count, self.max_step)
                
        # Local goal focused reward
        r_local_goal = self._local_goal_approach(local_goal_g, prev_pos, pos)

        # Obstacle avoidance reward (from lidar)
        r_obs_dist = self._obs_dist_reward()
                
        # Terminal rewards
        r_final = 0
        if collided:
            r_final = self.collision_reward
        elif success:
            r_final = self.success_reward
        elif truncation:
            r_final = self.failure_reward
        
        total_reward = r_smooth + r_time + r_local_goal + r_obs_dist + r_final
                
        rew_info = {
            "reward/smoothness_reward": r_smooth,
            "reward/time_penalty": r_time,
            "reward/local_goal_reward": r_local_goal,
            "reward/obstacle_distance": r_obs_dist,
            "reward/terminal_reward": r_final,
            "reward/local_focused_reward_total": total_reward
        }
                
        # Return total reward
        return total_reward, rew_info
    
    def simple_local_scheme(
        self,
        prev_vel,
        vel,
        prev_pos,
        pos,
        prev_psi,
        psi,
        success,
        collided,
        truncation,
        global_goal_pos,
        local_goal_g
    ):
        r_local = self._local_goal_approach(local_goal_g, prev_pos, pos)
        r_speed = self._speed_reward_simple(vel, self.max_vel)
        r_stop = self._stop_reward(prev_pos, pos)
        
        r_final = 0
        if collided:
            r_final = self.collision_reward
        if success:
            r_final = self.success_reward
        if truncation:
            r_final = self.failure_reward
        
        total_reward = r_local + r_final + r_stop + r_speed
        
        rew_info = {
            "reward/local_goal_reward": r_local,
            "reward/speed_reward": r_speed,
            "reward/stop_reward": r_stop,
            "reward/terminal_reward": r_final,
            "reward/simple_local_total_reward": total_reward
        }

        return total_reward, rew_info
    
    def simple_reward_scheme(
        self,
        prev_vel,
        vel,
        prev_pos,
        pos,
        prev_psi,
        psi,
        success,
        collided,
        truncation,
        global_goal_pos,
        local_goal_g,
    ):
        r_simple = self._simple_progress_reward(global_goal_pos)

        r_final = 0
        if collided:
            r_final += self.collision_reward
        if success:
            r_final += self.success_reward
        if truncation:
            r_final += self.failure_reward

        total_reward = r_simple + r_final
        rew_info = {
            "reward/simple_reward/progress": r_simple,
            "reward/terminal_reward": r_final,
            "reward/simple_reward_total_reward": total_reward
        }
        return total_reward, rew_info

    def smooth_reward_scheme(
        self,
        prev_vel,
        vel,
        prev_pos,
        pos,
        prev_psi,
        psi,
        success,
        collided,
        truncation,
        global_goal_pos,
        local_goal_g
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
        
        total_reward = (
            r_time + r_smooth + r_speed + r_approach + r_final
        )
        rew_info = {
            "reward/time_penalty": r_time,
            "reward/smoothness_reward": r_smooth,
            "reward/soft_speed_reward": r_speed,
            "reward/approach_reward": r_approach,
            "reward/terminal_reward": r_final,
            "reward/smooth_reward_total_reward": total_reward
        }

        return total_reward, rew_info

    def lidar_based_scheme(
        self,
        prev_vel,
        vel,
        prev_pos,
        pos,
        prev_psi,
        psi,
        success,
        collided,
        truncation,
        global_goal_pos,
        local_goal_g
    ):
        # Stop reward
        r_stop = self._stop_reward(prev_pos, pos)
        
        # R forward and R turn
        r_straight = self._going_straight_reward(prev_psi, psi)
        
        # R vel
        r_speed_simple = self._speed_reward_simple(vel, self.max_vel)
        
        # Reward for distance to obstacle
        r_obs_dist = self._obs_dist_reward()
        
        r_goal_reward = self._global_goal_dist_reward(global_goal_pos, prev_goal_pos=self.last_goal_pos)

        r_final = 0
        # terminal rewards
        if collided:
            r_final = self.collision_reward
        elif success:
            r_final = self.success_reward
        elif truncation:
            r_final = self.failure_reward
        
        total_reward = (
            r_stop
            + r_straight
            + r_speed_simple
            + r_obs_dist
            + r_final
            + r_goal_reward
        )
        rew_info = {
            "reward/stop_reward": r_stop,
            "reward/straight_reward": r_straight,
            "reward/speed_reward": r_speed_simple,
            "reward/obstacle_distance_reward": r_obs_dist,
            "reward/terminal_reward": r_final,
            "reward/global_goal_reward": r_goal_reward,
            "reward/lidar_total_reward": total_reward
        }
        
        return total_reward, rew_info

    def mixed_scheme(
        self,
        prev_vel,
        vel,
        prev_pos,
        pos,
        prev_psi,
        psi,
        success,
        collided,
        truncation,
        global_goal_pos,
        local_goal_g
    ):
        # Simple reward component # TODO: make a function for r_simple
        r_simple = self._simple_progress_reward(global_goal_pos)

        # Time and smoothness components
        r_time = self._time_penalty(self.step_count, self.max_step)
        r_smooth = self._smoothness_reward(prev_pos, prev_psi, pos, psi)
        r_speed = self._speed_reward_soft(prev_vel, vel, self.max_vel)
        r_approach = self._goal_approach_reward(global_goal_pos)
        r_local_goal = self._local_goal_approach(local_goal_g, prev_pos, pos)

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

        total_reward = (
            r_simple
            + r_time
            + r_smooth
            + r_speed
            + r_approach
            + r_stop
            + r_straight
            + r_obs_dist
            + r_final
            + r_local_goal
        )
        
        rew_info = {
            "reward/simple_reward/progress": r_simple,
            "reward/time_penalty": r_time,
            "reward/smoothness_reward": r_smooth,
            "reward/soft_speed_reward": r_speed,
            "reward/approach_reward": r_approach,
            "reward/local_goal_approach": r_local_goal,
            "reward/stop_reward": r_stop,
            "reward/straight_reward": r_straight,
            "reward/obstacle_distance_reward": r_obs_dist,
            "reward/terminal_reward": r_final,
            "reward/mixed_total_reward": total_reward
        }
        
        return total_reward, rew_info

    def _simple_progress_reward(self, global_goal_pos: np.ndarray, c_1: float = 1, c_2: float = -0.05) -> float:
        r_simple = (
            c_1 * (np.linalg.norm(self.last_goal_pos) - np.linalg.norm(global_goal_pos))
            + c_2
        )
        
        return r_simple

    # Checked
    def _smoothness_reward(
        self, 
        current_pos: Pose, 
        current_psi: float, 
        next_pos: Pose, 
        next_psi: float
    ) -> float:
        """
        Calculate a smoothness reward based on the vehicle's trajectory using numpy arrays.
        This function evaluates the smoothness of movement by comparing the current and next
        positions and orientations. It uses a cross product between the sum of normalized orientation
        vectors and the position difference to assess smoothness.
        Based on: DOI 10.1109/TIV.2024.3444854

        Parameters:
            current_pos (np.ndarray): Current position as a numpy array [x, y].
            current_psi (float): Current heading angle in radians.
            next_pos (np.ndarray): Next position as a numpy array [x, y].
            next_psi (float): Next heading angle in radians.

        Returns:
            float: A reward value where higher values indicate smoother trajectories.
                   The reward is calculated as (0.01 - |F|), where F is the cross product
                   between the sum of orientation vectors and the position difference.
        """
        n_x_i = np.array([np.cos(current_psi), np.sin(current_psi)])
        n_x_i_plus_1 = np.array([np.cos(next_psi), np.sin(next_psi)])

        # pos_vector
        pos_i = np.array([current_pos.x, current_pos.y])
        pos_i_plus_1 = np.array([next_pos.x, next_pos.y])

        # vector product between the two vectors
        F = np.cross(n_x_i + n_x_i_plus_1, pos_i_plus_1 - pos_i)

        reward = 0.01 - np.linalg.norm(F)
        
        if reward < -0.5:
            reward = -0.5
        
        return reward

    # Checked
    def _speed_reward_soft(
        self, 
        prev_vel: float, 
        current_vel: float, 
        max_vel: float, 
        alpha: float = 10, 
        beta: float = 20
    ) -> float: 
        reward = 0.001 * (
            1 / (1 + np.exp(-alpha * (current_vel - max_vel / 2)))
            - beta * (current_vel - prev_vel) ** 2
        )
        return reward

    # Checked
    def _speed_reward_simple(self, current_vel, max_vel, limit=0.7):
        return np.clip(current_vel / max_vel, -0.01, limit) * 0.25

    # Checked
    def _going_straight_reward(
        self, 
        prev_psi: float, 
        current_psi: float, 
        alpha: float=10.0
        ) -> float:
        if np.abs(current_psi - prev_psi) < 0.25: # 0.2 rad as delta psi
            return 0.2
        else:
            return 0.05

    # Checked
    def _time_penalty(self, step_count, max_step):
        return -0.01

    # Checked
    def _obs_dist_reward(self, tau: float=0.2) -> float:
        min_distance = self.get_valid_laser_data_softmin(tau=tau)
        return - np.exp(-min_distance)
        #return -1 / (min_distance + 1e-8) * alpha + 0.1 if min_distance < 1 else 0

    # Checked
    def _local_goal_approach(
        self, 
        prev_local_goal: np.ndarray, 
        prev_pos: Pose, 
        pos: Pose
    ) -> float:
        """
        Reward for approaching the local goal
        """
        
        prev_dist = np.linalg.norm(np.array([prev_local_goal[0] - prev_pos.x, prev_local_goal[1] - prev_pos.y]))
        curr_dist = np.linalg.norm(np.array([prev_local_goal[0] - pos.x, prev_local_goal[1] - pos.y]))
        
        if prev_dist - curr_dist < 0:
            reward = 0.3
        else:
            reward = -0.4
            
        return reward

    # Checked
    def _global_goal_dist_reward(
        self, 
        global_goal_pos: np.ndarray, 
        prev_goal_pos: np.ndarray,
        alpha: float=1.0 ) -> float:
        # Calculate Euclidean distance between robot and goal
        approach = np.linalg.norm(global_goal_pos) - np.linalg.norm(prev_goal_pos)
        reward = alpha * approach
        # Convert distance to reward (closer = higher reward)
        # reward = 0.005 * (1 / (distance + 1))  # Adding 1 to avoid division by zero
        return reward

    # def _collision_reward(self):
    #     collided = self.gazebo_sim.get_hard_collision() and self.step_count > 1
    #     reward = 0
    #     if collided:
    #         reward += self.collision_reward

    #     return reward

    # Checked
    def _stop_reward(self, prev_pos: Pose, pos: Pose) -> float:
        distance = np.linalg.norm([prev_pos.x - pos.x, prev_pos.y - pos.y])
        if distance < 0.01:
            reward = -0.5
        else:
            reward = 0.0
        return reward
    
    # Checked
    def _goal_approach_reward(self, global_goal_pos: np.ndarray) -> float:
        getting_closer = (
            np.linalg.norm(self.last_goal_pos) - np.linalg.norm(global_goal_pos)
        ) > 0
        
        if getting_closer:
            reward = 0.05
        else:
            reward = -0.03
        #reward = 0.05 * getting_closer
        return reward

    def switch_reward_function(self, reward_function):
        if reward_function in self.reward_functions:
            self.reward_func = self.reward_functions[reward_function]
            self.reward_scheme_name = reward_function
        else:
            raise ValueError(
                f"Reward function '{reward_function}' not found. Available options: {list(self.reward_functions.keys())}"
            )
            
    def get_valid_laser_data_softmin(self, tau=0.1):
        laser_scan = self.gazebo_sim.get_laser_scan()
        laser_scan = np.array(laser_scan.ranges)
        laser_scan[laser_scan > 10] = 10
        x = laser_scan - 0.2 # offset from the distance to lidar
        weights = np.exp(-x / tau)
        return np.sum(x * weights) / np.sum(weights)

    
    def get_valid_laser_data_min(self):
        laser_data = self.gazebo_sim.get_laser_scan()
        ranges = np.array(laser_data.ranges)
        valid = (ranges > 0) & (ranges != np.inf)
        if np.any(valid): 
            min_distance = np.min(ranges[valid])
        else:
            min_distance = float("inf")
        
        return min_distance
    
    def collision_with_lidar(self):
        min_distance = self.get_valid_laser_data_min()
        if min_distance < 0.2:
            return True
        else:
            return False

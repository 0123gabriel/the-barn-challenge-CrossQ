import os
import random

import gym
import numpy as np
import pandas as pd
from envs.wrappers import StackFrame


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
    
    def get_env(self, success = 0.0):
        self.current_sucess_rate = self.smoothing_factor * success + (1 - self.smoothing_factor) * self.current_sucess_rate
        
        env_config = self.config["env_config"]
        env_config["kwargs"]["init_sim"] = True
        
        # Define curriculum stages with their corresponding parameters
        curriculum_stages = [
            {"fill_pct_range": [0.0, 0.10], "distance_range": [10, 15], "init_position": [-2, 4, 1.57], "goal_position": [0, 3.5, 0]},  # Stage 0
            {"fill_pct_range": [0.10, 0.20], "distance_range": [10, 15], "init_position": [-2, 4, 1.57], "goal_position": [0, 5.5, 0]}, # Stage 1
            {"fill_pct_range": [0.0, 0.15], "distance_range": [15, 25], "init_position": [-2, 4, 1.57], "goal_position": [0, 7, 0]},  # Stage 2
            {"fill_pct_range": [0.1, 0.2], "distance_range": [15, 25], "init_position": [-2, 4, 1.57], "goal_position": [0, 7, 0]},   # Stage 3
            {"fill_pct_range": [0, 0.2], "distance_range": [25, 40], "init_position": [-2, 3, 1.57], "goal_position": [0, 10, 0]},     # Stage 4
            {"fill_pct_range": [0, 0.3], "distance_range": [10, 40], "init_position": [-2, 3, 1.57], "goal_position": [0, 10, 0]},     # Stage 5
        ]
        
        # Final stage parameters for all stages >= 6
        final_stage_params = {"fill_pct_range": [0.1, 0.35], "distance_range": [25, 40], "init_position": [-2, 3, 1.57], "goal_position": [0, 10, 0]}

        reward_types = ["simple", "simple_local", "lidar", "local", "smooth", "mixed", "mixed"]
        
        # [simple, vel, obs, local_goal, local_goal_quadratic, smoothness, stop, straight, speed, terminal]
        # Define reward weights for different stages
        self.stage_reward_weights = {
            0: np.array([10.0, 2.5, 0.15, 0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25]),  # fase 1
            1: np.array([10.0, 4.0, 1.0, 4.0, 2.0, 1.5, 2.0, 1.0, 2.0, 1.0]),        # fase 2
            2: np.array([10.0, 4.0, 2.0, 3.5, 4.5, 5.0, 3.0, 2.0, 3.0, 2.0]),        # fase 3
            3: np.array([10.0, 3.0, 4.0, 3.0, 4.5, 5.0, 1.0, 2.0, 4.0, 2.0]),    # fase 4
            4: np.array([10.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 2.0, 5.0, 5.0]),      # fase 5 
            5: np.array([10.0, 1.0, 3.0, 4.0, 4.0, 9.0, 0.5, 2.0, 7.0, 6.0])      # fase 6
        }

        # Default reward weights for stages beyond those defined
        self.default_reward_weights = np.array([10.0, 1.0, 3.0, 4.0, 4.0, 9.0, 0.5, 2.0, 7.0, 6.0])
        
        # TODO: change the reward weights for the different stages
        
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
        env_config["kwargs"]["reward_function"] =reward_types[self.stage] #np.random.choice(reward_types) # "combined"
        env_config["kwargs"]["init_position"] = params["init_position"]
        env_config["kwargs"]["goal_position"] = params["goal_position"]
        stage_reward = self.stage_reward_weights[self.stage] if self.stage < len(self.stage_reward_weights) else self.default_reward_weights

        info = {
            "world": env_config["kwargs"]["world_name"],
            "reward_scheme": env_config["kwargs"]["reward_function"],
            "stage": self.stage,
            "success_rate": self.current_sucess_rate,
        }

        env = gym.make(env_config["env_id"], reward_weights = stage_reward, **env_config["kwargs"])
        env = StackFrame(env, stack_frame=env_config["stack_frame"])

        return env, info


class ProxCurrl:
    def __init__(self):
        pass   
    
    
    def get_env(self, success_rate=0.0) -> gym.Env:
        """
        Get next environment (main class function)
        """
    
    def update_buffer(self):
        """
        Update the buffer with the new environment
        """
        pass
    
    def episodic_update(self):
        """
        Update the buffer with the new data
        """
        pass
    
    def sample_task(self):
        """
        Sample a task from the buffer
        """
        pass

    
    
    

from os.path import exists, join
import numpy as np
import time
import logging
import re
import pickle
import shutil
import os


BUFFER_PATH = os.getenv('BUFFER_PATH')

class LocalCollector(object):
    def __init__(self, policy, env, replaybuffer, use_wandb=False):
        self.policy = policy
        self.env = env
        self.buffer = replaybuffer
        
        self.last_obs = None
        
        self.global_episodes = 0
        self.global_steps = 0
    
    def collect(self, n_steps):
        #print('Collect first line')
        n_steps_curr = 0
        env = self.env
        policy = self.policy
        results = []
        
        ep_rew = 0
        ep_len = 0
        
        if self.last_obs is not None:
            obs = self.last_obs
            #print('Last obs')
        else:
            #print('Before reset')
            obs = env.reset()
            #print('Env reset called')
            
        while n_steps_curr < n_steps:
            #print(n_steps_curr)
            act = policy.select_action(obs, True)
            obs_new, rew, terminated, truncated, info = env.step(act)
            obs = obs_new
            ep_rew += rew
            ep_len += 1
            n_steps_curr += 1
            self.global_steps += 1
            
            world = int(info['world'].split(
                "_")[-1].split(".")[0])
            collision_reward = -int(info['collided'])
            
            #if self.policy.safe_rl:
            #    self.buffer.add(obs, act,
            #                    obs_new, rew,
            #                    truncated, terminated, world, collision_reward)
            #else:
            self.buffer.add(obs, act,
                            obs_new, rew,
                            truncated, terminated,
                            world)
            
            if use_wandb:
                wandb.log(info)
            
            if terminated or truncated:
                obs = env.reset()
                info1 = dict(
                    ep_rew=ep_rew,
                    ep_len=ep_len,
                    world=world,
                    global_steps=self.global_steps,
                    global_episodes=self.global_episodes, 
                )
                
                joined_info = {**info1, **info}
                
                results.append(joined_info)
                ep_rew = 0
                ep_len = 0
                self.global_episodes += 1
                
                if use_wandb:
                    wandb.log(joined_info)
                
            print("n_episode: %d, n_steps: %d" %(self.global_episodes, self.global_steps), end="\r")
            
        self.last_obs = obs
        return n_steps_curr, results
    
    
    def set_env(self, env):
        self.env = env
        self.last_obs = None
    

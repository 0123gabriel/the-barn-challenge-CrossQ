import os
import yaml
import pickle
from os.path import join, dirname, abspath, exists
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

import torch
import gym
import numpy as np
from datetime import datetime
import uuid
import shutil
import time
import collections
from pprint import pformat

from envs.wrappers import StackFrame
from sac.collector import LocalCollector

from rl import Actor, CrossQCritic, CrossQ_SAC
from net import MLP_CrossQ
from utils import ReplayBuffer

def initialize_config(config_path, save_path):
    # Load the config files
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["env_config"]["save_path"] = save_path
    config["env_config"]["config_path"] = config_path

    return config

def initialize_logging(config):
    env_config = config["env_config"]
    training_config = config["training_config"]

    # Config logging
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    if training_config["safe_rl"]:
        mode = training_config["safe_mode"]
        string = f"safe_rl_{mode}_"
        if mode == "lagr":
            string = string + "lagr"+str(training_config["safe_lagr"]) + "_"
    else:
        string = ""

    string = string + dt_string

    save_path = join(
        env_config["save_path"], 
        env_config["env_id"], 
        training_config['algorithm'], 
        string,
        uuid.uuid4().hex[:4]
    )
    print("    >>>> Saving to %s" % save_path)
    if not exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    shutil.copyfile(
        env_config["config_path"], 
        join(save_path, "config.yaml")    
    )

    return save_path, writer

#TODO: modify this function
def initialize_envs(config):
    env_config = config["env_config"]
    if env_config["use_condor"]:
        env_config["kwargs"]["init_sim"] = False
    
    # if not env_config["use_condor"]:
    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    env = StackFrame(env, stack_frame=env_config["stack_frame"])
    # else:
        # If use condor, we want to avoid initializing env instance from the central learner
        # So here we use a fake env with obs_space and act_space information
    #    print("    >>>> Using actors on Condor")
    #    env = InfoEnv(config)
    return env

def seed(config):
    env_config = config["env_config"]
    
    np.random.seed(env_config['seed'])
    torch.manual_seed(env_config['seed'])
    
def get_encoder(encoder_type, args):
    #!TODO: implement this function with useful possible encoders
    pass

def initialize_policy(config, env, init_buffer=True):
    #!TODO: implement this function with CrossQ
    training_config = config["training_config"]
    
    state_dim = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoder_type = training_config["encoder"]
    encoder_args = {
        "input_dim": state_dim,
        "num_layers": training_config["encoder_num_layers"],
        "hidden_size": training_config["encoder_hidden_layer_size"],
        "history_length": config["env_config"]["stack_frame"],
    }
    
    input_dim = training_config["hidden_layer_size"]
    actor = Actor(
        state_preprocess= get_encoder(encoder_type, encoder_args),
        head= MLP_CrossQ(input_dim, training_config['encoder_num_layers'], training_config['encoder_hidden_layer_size']),
        action_dim= action_dim,
    ).to(device)
    
    print("Total number of parameters: %d" %sum(p.numel() for p in actor.parameters()))
    input_dim += np.prod(action_dim)
    
    critic = CrossQCritic(
        state_preprocess= get_encoder(encoder_type, encoder_args),
        head= MLP_CrossQ(input_dim, training_config['encoder_num_layers'], training_config['encoder_hidden_layer_size']),
    ).to(device)
    
    critic_optim = torch.optim.Adam(critic.parameters(), lr=training_config["critic_lr"])
    actor_optim = torch.optim.Adam(actor.parameters(), lr=training_config["actor_lr"])
    
    policy = CrossQ_SAC(
        actor=actor, actor_optim=actor_optim,
        critic=critic, critic_optim=critic_optim,
        action_range=[action_space_low, action_space_high],
        device=device
        **training_config["policy_args"] #TODO: review this
    )
    
    if init_buffer:
        try:
            config['env_config']["reward_norm"]
        except KeyError:
            config['env_config']["reward_norm"] = False
        
        buffer = ReplayBuffer(
                state_dim=state_dim,
                action_dim=action_dim,
                reward_norm=config['env_config']["reward_norm"],
                **training_config["buffer_args"]
            )
    else:
        buffer = None
        
    return policy, buffer
    

def train(env, policy, buffer, config):
    #!TODO modify this and implement CrossQ-SAC training (I think this shouldn't change much)
    env_config = config["env_config"]
    training_config = config["training_config"]

    save_path, writer = initialize_logging(config)
    print("    >>>> initialized logging")
    
    
    collector = LocalCollector(policy, env, buffer)

    training_args = training_config["training_args"]
    print("    >>>> Pre-collect experience")
    collector.collect(n_steps=training_config['pre_collect'])
    print("    >>>> Start training")

    n_steps = 0
    n_iter = 0
    n_ep = 0
    epinfo_buf = collections.deque(maxlen=300)
    world_ep_buf = collections.defaultdict(lambda: collections.deque(maxlen=20))
    t0 = time.time()
    
    while n_steps < training_args["max_step"]:
        # Linear decaying exploration noise from "start" -> "end"
        # policy.exploration_noise = \
        #     - (training_config["exploration_noise_start"] - training_config["exploration_noise_end"]) \
        #     *  n_steps / training_args["max_step"] + training_config["exploration_noise_start"]
        steps, epinfo = collector.collect(n_steps=training_args["collect_per_step"])
        
        n_steps += steps
        n_iter += 1
        n_ep += len(epinfo)
        epinfo_buf.extend(epinfo)
        for d in epinfo:
            world = d["world"].split("/")[-1]
            world_ep_buf[world].append(d)

        loss_infos = []
        for _ in range(training_args["update_per_step"]):
            loss_info = policy.train(buffer, training_args["batch_size"])
            loss_infos.append(loss_info)

        loss_info = {}
        for k in loss_infos[0].keys():
            loss_info[k] = np.mean([li[k] for li in loss_infos if li[k] is not None])

        t1 = time.time()
        log = {
            "Episode_return": np.mean([epinfo["ep_rew"] for epinfo in epinfo_buf]),
            "Episode_length": np.mean([epinfo["ep_len"] for epinfo in epinfo_buf]),
            "Success": np.mean([epinfo["success"] for epinfo in epinfo_buf]),
            "Time": np.mean([epinfo["ep_time"] for epinfo in epinfo_buf]),
            "Collision": np.mean([epinfo["collision"] for epinfo in epinfo_buf]),
            "fps": n_steps / (t1 - t0),
            "n_episode": n_ep,
            "Steps": n_steps,
            "Exploration_noise": policy.exploration_noise,
        }
        log.update(loss_info)
        print(pformat(log))

        if n_iter % training_config["log_intervals"] == 0:
            for k in log.keys():
                writer.add_scalar('train/' + k, log[k], global_step=n_steps)
            policy.save(save_path, "last_policy")
            print("Logging to %s" %save_path)

            for k in world_ep_buf.keys():
                writer.add_scalar(k + "/Episode_return", np.mean([epinfo["ep_rew"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Episode_length", np.mean([epinfo["ep_len"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Success", np.mean([epinfo["success"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Time", np.mean([epinfo["ep_time"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Collision", np.mean([epinfo["collision"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
    
    env.close()
    


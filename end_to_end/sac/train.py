import os
import rospy
import subprocess
import argparse
import logging
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
import wandb
from pprint import pformat

from envs.wrappers import StackFrame
from sac.collector import LocalCollector

from rl import Actor, CrossQCritic, CrossQ_SAC
from net import MLP_CrossQ
from utils import Env_Selector, Simple_Curriculum, ReplayBuffer
from net import MLP_CrossQ, RNNEncoder, CNNEncoder, TCNEncoder, DilatedCNNEncoder
from torch.utils.tensorboard import SummaryWriter

import envs.registration # register the env to run train.py

#TODO: Set this information from the config file
log_dir = 'trainings'
use_wandb = True 

def start_roscore():
    """Start roscore in a subprocess."""
    rospy.loginfo("Starting roscore...")
    process = subprocess.Popen(["roscore"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)  # Give some time for roscore to start
    return process

def initialize_config(config_path, save_path):
    # Load the config files
    with open(config_path, "r") as f:
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
            string = string + "lagr" + str(training_config["safe_lagr"]) + "_"
    else:
        string = ""

    string = string + dt_string

    save_path = join(
        env_config["save_path"],
        env_config["env_id"],
        training_config["algorithm"],
        string,
        uuid.uuid4().hex[:4],
    )
    print("    >>>> Saving to %s" % save_path)
    if not exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    shutil.copyfile(env_config["config_path"], join(save_path, "config.yaml"))

    return save_path, writer

# TODO: modify this function
def initialize_envs(config):
    env_config = config["env_config"]
    #env_config["kwargs"]["world_name"] = get_random_world()
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
    np.random.seed(env_config["seed"])
    torch.manual_seed(env_config["seed"])


def get_encoder(encoder_type, args):
    if encoder_type == "rnn":
        return RNNEncoder(**args)
    elif encoder_type == "cnn":
        return CNNEncoder(**args)
    elif encoder_type == "tcn":
        return TCNEncoder(**args)
    elif encoder_type == "dilated_cnn":
        return DilatedCNNEncoder(**args)
    else:
        raise NotImplementedError


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

    input_dim = 736 # Input dim is [laser dimensio + stack frames*size of local goal + stack frames*action dim]
    actor = Actor(
        state_preprocess=get_encoder(encoder_type, encoder_args),
        head=MLP_CrossQ(
            input_dim,
            training_config["encoder_num_layers"],
            training_config["encoder_hidden_layer_size"],
        ),
        action_dim=action_dim,
        action_space_high=action_space_high, 
        action_space_low=action_space_low
    ).to(device)

    print("Total number of parameters: %d" % sum(p.numel() for p in actor.parameters()))
    input_dim += np.prod(action_dim)

    critic = CrossQCritic(
        state_preprocess=get_encoder(encoder_type, encoder_args),
        head=MLP_CrossQ(
            input_dim,
            training_config["encoder_num_layers"],
            training_config["encoder_hidden_layer_size"],
        ),
    ).to(device)

    critic_optim = torch.optim.Adam(
        critic.parameters(), lr=training_config["critic_lr"]
    )
    actor_optim = torch.optim.Adam(actor.parameters(), lr=training_config["actor_lr"])
    print(device)
    policy = CrossQ_SAC(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        action_range=[action_space_low, action_space_high],
        device=device, **training_config["policy_args"],  # TODO: review this
    )

    if init_buffer:
        try:
            config["env_config"]["reward_norm"]
        except KeyError:
            config["env_config"]["reward_norm"] = False

        buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            reward_norm=config["env_config"]["reward_norm"],
            #**training_config["buffer_args"],
        )
    else:
        buffer = None

    return policy, buffer


def train(env_selector, env, policy, buffer, config):
    #!TODO modify this and implement CrossQ-SAC training (I think this shouldn't change much)
    env_config = config["env_config"]
    training_config = config["training_config"]

    save_path, writer = initialize_logging(config)
    print("    >>>> initialized logging")
    
    collector = LocalCollector(policy, env, buffer)

    training_args = training_config["training_args"]
    
    print("    >>>> Pre-collect experience")
    collector.collect(n_steps=training_config["pre_collect"])
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
            print(d["world"])
            world = d["world"] #.split("/")[-1]
            world_ep_buf[world].append(d)

        loss_infos = []
        for _ in range(training_args["update_per_step"]):
            loss_info = policy.train(buffer, training_args["batch_size"])
            if use_wandb:
                wandb.log(loss_info)
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
        }
        
        if use_wandb:
            wandb.log(log, step=n_steps)
            
        log.update(loss_info)
        print(pformat(log))

        if n_iter % training_config["log_intervals"] == 0:
            for k in log.keys():
                writer.add_scalar("train/" + k, log[k], global_step=n_steps)
            policy.save(save_path, "last_policy")
            print("Logging to %s" % save_path)

            for k in world_ep_buf.keys():
                writer.add_scalar(k + "/Episode_return", np.mean([epinfo["ep_rew"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Episode_length", np.mean([epinfo["ep_len"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Success", np.mean([epinfo["success"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Time", np.mean([epinfo["ep_time"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Collision", np.mean([epinfo["collision"] for epinfo in world_ep_buf[k]]), global_step=n_steps)

        env.close()
        time.sleep(1)
    
        # Change env for next iteration
        env, info = env_selector.get_env(success_rate=log["Success"])
        if use_wandb:
            wandb.log(info, step=n_steps)
        collector.set_env(env)
        

if __name__ == "__main__":
    
    roscore_process = start_roscore()
    # initialize the node for gym env
    print('Before gym init ======================================================================================')
    if not rospy.core.is_initialized():
        rospy.init_node('gym', anonymous=False, log_level=rospy.FATAL)
        rospy.set_param('/use_sim_time', True)
    print('After gym init ======================================================================================')
    
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description = 'Start condor training')
    parser.add_argument('--config_path', dest='config_path', default="../data/config.yaml")
    logging.getLogger().setLevel("INFO")
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    SAVE_PATH = "logging/"
    print(">>>>>>>> Loading the configuration from %s" % CONFIG_PATH)
    config = initialize_config(CONFIG_PATH, SAVE_PATH)

    if use_wandb:  # TODO: this should be in the main file
            wandb.init(
                project="BARN_CrossQ",
                config={
                    "algorithm": config["training_config"]["algorithm"],
                    "network": config["training_config"]["network"],
                    "encoder": config["training_config"]["encoder"],
                    "buffer_size": config["training_config"]["buffer_size"],
                    "actor_lr": config["training_config"]["actor_lr"],
                    "critic_lr": config["training_config"]["critic_lr"],
                    "num_layers": config["training_config"]["num_layers"],
                    "hidden_layer_size": config["training_config"]["hidden_layer_size"],
                    "encoder_num_layers": config["training_config"]["encoder_num_layers"],
                    "encoder_hidden_layer_size": config["training_config"]["encoder_hidden_layer_size"],
                    "exploration_noise_start": config["training_config"]["exploration_noise_start"],
                    "exploration_noise_end": config["training_config"]["exploration_noise_end"],
                    "pre_collect": config["training_config"]["pre_collect"],
                    "log_intervals": config["training_config"]["log_intervals"],
                    "validation": config["training_config"]["validation"],
                    "val_interval": config["training_config"]["val_interval"],
                    "dyna_style": config["training_config"]["dyna_style"],
                    "n_simulated_update": config["training_config"]["n_simulated_update"],
                    "model_lr": config["training_config"]["model_lr"],
                    "MPC": config["training_config"]["MPC"],
                    "horizon": config["training_config"]["horizon"],
                    "num_particle": config["training_config"]["num_particle"],
                    "safe_rl": config["training_config"]["safe_rl"],
                    "safe_mode": config["training_config"]["safe_mode"],
                    "safe_lagr": config["training_config"]["safe_lagr"],
                    "policy_args": config["training_config"]["policy_args"],
                    "training_args": config["training_config"]["training_args"],
                },
            )

    seed(config)
    print(">>>>>>>> Creating the environments")
    worlds_directory = "/root/e2e_crossq/src/the-barn-challenge-CrossQ/jackal_helper/worlds/Curr_BARN" # This should be automated
    env_selector= Env_Selector(config, worlds_directory)
    
    if config["env_selector"] == "random":
        env_selector = Env_Selector(config, worlds_directory)
    if config["env_selector"] == "simple_curriculum":
        env_selector = Simple_Curriculum(config, worlds_directory)
    
    env, info = env_selector.get_env()
    
    if use_wandb:
        wandb.log(info)
    #env_config = config["env_config"]
    # env_config["kwargs"]["init_sim"] = False
    #env_config["kwargs"]["world_name"] = 'BARN/world_21.world' #self.get_random_world()
    #if env_config["use_condor"]:
    #    env_config["kwargs"]["init_sim"] = False

    # if not env_config["use_condor"]:
    #env = gym.make(env_config["env_id"], **env_config["kwargs"])
    #env = StackFrame(env, stack_frame=env_config["stack_frame"])
    #env = train_envs if config["env_config"]["use_condor"] else train_envs
    
    print(">>>>>>>> Initializing the policy")
    policy, buffer = initialize_policy(config, env)
    print(">>>>>>>> Start training")
    train(env_selector, env, policy, buffer, config)

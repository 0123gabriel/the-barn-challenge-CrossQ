import os
import rospy
import subprocess
import argparse
import logging
import yaml
import sys
from os.path import join, dirname, abspath, exists

sys.path.append(dirname(dirname(abspath(__file__))))

import torch
import gym
import numpy as np
import time
import collections
import wandb
from pprint import pformat

from envs.wrappers import StackFrame
from sac.collector import LocalCollector
from sac.train import initialize_config, initialize_logging, seed, get_encoder, start_roscore
from rl import Actor, CrossQCritic, CrossQ_SAC
from net import MLP_CrossQ, RNNEncoder, CNNEncoder, TCNEncoder, DilatedCNNEncoder
from utils import Env_Selector, Simple_Curriculum, ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import envs.registration  # register the env to run train.py

# Set torch to use float32 by default
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)

# Global variables
use_wandb = True
log_dir = 'trainings'


def initialize_policy(config, env, init_buffer=True):
    """Initialize policy with the same architecture as in train.py"""
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

    input_dim = 744  # Input dim is [laser dimension + stack frames*size of local goal + stack frames*action dim] + local_goal * stack frames
    actor = Actor(
        state_preprocess=get_encoder(encoder_type, encoder_args),
        head=MLP_CrossQ(
            input_dim,
            training_config["encoder_num_layers"],
            training_config["encoder_hidden_layer_size"],
        ),
        action_dim=action_dim,
        input_dim=input_dim,
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
        alpha_lr=training_config["alpha_lr"],
        device=device, **training_config["policy_args"],
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
        )
    else:
        buffer = None

    return policy, buffer


def load_model(policy, run_name, checkpoint_filename):
    """Load a trained model checkpoint"""
    path_save_model = '/root/e2e_crossq/src/the-barn-challenge-CrossQ/end_to_end/trained_models'
    folder_path = join(path_save_model, run_name)
    checkpoint_path = join(folder_path, checkpoint_filename)
    
    if not exists(checkpoint_path):
        print(f"ERROR: Checkpoint file '{checkpoint_path}' not found!")
        sys.exit(1)
    
    try:
        policy.load(run_name, checkpoint_filename)
        print(f"Successfully loaded model from {run_name}/{checkpoint_filename}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)


def continue_train(env_selector, env, policy, buffer, config):
    """Continue training from a loaded model at specified stage"""
    env_config = config["env_config"]
    training_config = config["training_config"]
    start_stage = config["env_config"]["start_stage"]

    save_path, writer = initialize_logging(config)
    print("    >>>> initialized logging")
    
    collector = LocalCollector(policy, env, buffer, use_wandb=use_wandb)

    training_args = training_config["training_args"]
    
    # Skip pre-collect phase since we're continuing from an existing model
    print(f"    >>>> Start continued training from stage {start_stage}")

    # Initialize environment at the specified stage
    if hasattr(env_selector, "stage"):
        env_selector.stage = start_stage
        env_selector.current_sucess_rate = 0.0  # Reset success rate
        # Get environment for the specific stage
        env, info = env_selector.get_env()
        collector.set_env(env)
    
    # Update policy's alpha for the stage
    policy.set_alpha_list(env_config["alpha_list"])
    policy.update_alpha(start_stage)

    # Start training loop (similar to train.py)
    n_steps = 0
    n_iter = 0
    n_ep = 0
    epinfo_buf = collections.deque(maxlen=300)
    world_ep_buf = collections.defaultdict(lambda: collections.deque(maxlen=20))
    t0 = time.time()

    while n_steps < training_args["max_step"]:
        steps, epinfo = collector.collect(n_steps=training_args["collect_per_step"])

        n_steps += steps
        n_iter += 1
        n_ep += len(epinfo)
        epinfo_buf.extend(epinfo)
        for d in epinfo:
            world = d["world"]
            world_ep_buf[world].append(d)

        loss_infos = []
        for training_steps in range(training_args["update_per_step"]):
            loss_info = policy.train(buffer, training_args["batch_size"])
            loss_info["n_steps"] = n_steps
            loss_info["n_iter"] = n_iter
            loss_info["n_ep"] = n_ep
            loss_info["training_steps"] = training_steps
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
            "Current_stage": env_selector.stage if hasattr(env_selector, "stage") else 0,
        }
        
        if use_wandb:
            log["n_iter"] = n_iter
            wandb.log(log)
            
        log.update(loss_info)
        print(pformat(log))

        if n_iter % training_config["log_intervals"] == 0:
            for k in log.keys():
                writer.add_scalar("train/" + k, log[k], global_step=n_steps)
            
            # Save with continued tag to distinguish from original model
            continued_run_name = f"{wandb.run.name}_continued" if use_wandb else "continued_training"
            policy.save(continued_run_name)
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
        env, info = env_selector.get_env(success=log["Success"])
        
        if info["success_rate"] > 0.8:
            policy.update_alpha(info["stage"])
            print(f"Advancing to stage {info['stage']} with new alpha={policy.log_alpha.exp().item():.3f}")
        
        if use_wandb:
            wandb.log(info)
        collector.set_env(env)


if __name__ == "__main__":
    # Start roscore
    roscore_process = start_roscore()
    
    # Initialize node for gym env
    print('Before gym init ======================================================================================')
    if not rospy.core.is_initialized():
        rospy.init_node('gym', anonymous=False, log_level=rospy.FATAL)
        rospy.set_param('/use_sim_time', True)
    print('After gym init ======================================================================================')
    
    torch.set_num_threads(8)
    
    parser = argparse.ArgumentParser(description='Continue training from a saved checkpoint')
    parser.add_argument('--config_path', dest='config_path', default="../data/config.yaml", help='Path to configuration file')
    parser.add_argument('--run_name', dest='run_name', required=True, help='Name of the run to load model from')
    parser.add_argument('--checkpoint', dest='checkpoint', required=True, help='Checkpoint filename (e.g., checkpoint_0.pth)')
    parser.add_argument('--stage', dest='stage', type=int, default=0, help='Stage to continue training from')
    parser.add_argument('--wandb_project', dest='wandb_project', default="barn_crossq_continued", help='WandB project name')
    
    logging.getLogger().setLevel("INFO")
    args = parser.parse_args()
    
    if args.stage < 0:
        print("ERROR: Stage must be a non-negative integer")
        sys.exit(1)
    
    CONFIG_PATH = args.config_path
    SAVE_PATH = "logging/"
    print(f">>>>>>>> Loading the configuration from {CONFIG_PATH}")
    config = initialize_config(CONFIG_PATH, SAVE_PATH)

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.run_name}_continued_from_stage_{args.stage}",
            config={
                "algorithm": config["training_config"]["algorithm"],
                "encoder": config["training_config"]["encoder"],
                "buffer_size": config["training_config"]["buffer_size"],
                "actor_lr": config["training_config"]["actor_lr"],
                "critic_lr": config["training_config"]["critic_lr"],
                "alpha_lr": config["training_config"]["alpha_lr"],
                "encoder_num_layers": config["training_config"]["encoder_num_layers"],
                "encoder_hidden_layer_size": config["training_config"]["encoder_hidden_layer_size"],
                "continued_from_run": args.run_name,
                "continued_from_checkpoint": args.checkpoint,
                "continued_from_stage": args.stage,
            },
        )

    seed(config)
    print(">>>>>>>> Creating the environments")
    worlds_directory = "/root/e2e_crossq/src/the-barn-challenge-CrossQ/jackal_helper/worlds/Curr_BARN"  # This path should be automated
    
    # Initialize env_selector based on config
    if config["env_selector"] == "random":
        env_selector = Env_Selector(config, worlds_directory)
    elif config["env_selector"] == "simple_curriculum":
        env_selector = Simple_Curriculum(config, worlds_directory)
    else:
        env_selector = Env_Selector(config, worlds_directory)
    
    # Get initial environment
    env, info = env_selector.get_env()
    
    if use_wandb:
        wandb.log(info, step=0)
    
    print(">>>>>>>> Initializing policy")
    policy, buffer = initialize_policy(config, env)
    
    print(f">>>>>>>> Loading model from {args.run_name}/{args.checkpoint}")
    policy = load_model(policy, args.run_name, args.checkpoint)
    
    print(f">>>>>>>> Continuing training from stage {args.stage}")
    continue_train(env_selector, env, policy, buffer, config, args.stage)
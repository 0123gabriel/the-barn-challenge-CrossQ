from typing import Tuple, List
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
import copy
from sac.utils import BatchRenorm, SquashedNormal, CBPLinear
from sac.net import get_activation, MLP


from os.path import join
import datetime
import os

import torch.nn as nn


class CrossQ_SAC(object):
    def __init__(
        self,
        actor,
        actor_optim,
        critic,
        critic_optim,
        action_range,
        device="cpu",
        gamma=0.99,  # policy_arg
        tau=5e-3,  # policy_arg
        alpha_lr=5e-4,  # policy_arg
        n_step=4,  # policy_arg
        update_actor_freq=2,  # policy_arg
    ):
        self.actor = actor
        self.actor_optim = actor_optim
        self.critic = critic
        self.critic_optim = critic_optim
        self.action_range = action_range
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha_lr = alpha_lr
        self.n_step = n_step
        self.rewards_scale = 1.0
        self.target_update_freq = update_actor_freq
        
        self.alpha_list = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]

        #self.target_entropy = torch.tensor(-2.0, dtype=torch.float32, device=self.device) #-torch.prod(torch.Tensor(action_range)).to(self.device)
        self.target_entropy = -torch.prod(torch.tensor(np.array(action_range).shape[-1], dtype=torch.float32, device=self.device))
        print('Target Entropy : ', self.target_entropy, '=======================================================================')
        #print('Target Entropy 1: ', self.target_entropy_1, '=======================================================================')
        init_temperature = 1.0

        self.log_alpha = torch.tensor(
            [np.log(init_temperature)],
            requires_grad=True,
            dtype=torch.float32,
            device=self.device,
        )

        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(0.5, 0.999)
        )

        self.total_it = 0
        self.action_range = action_range
        self._action_scale = torch.FloatTensor(
            (action_range[1] - action_range[0]) / 2.0
        ).to(device)
        self._action_bias = torch.FloatTensor(
            (action_range[1] + action_range[0]) / 2.0
        ).to(device)

    def select_action(self, states: torch.Tensor, train: bool) -> torch.Tensor:
        """
        input: state (torch.Tensor)
        output: action (torch.Tensor)
        """
        # get the action from the actor (no gradients)
        self.actor.eval()
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            states = states.unsqueeze(0)
            if train:
                action, _, _ = self.actor.get_action_alt(states)
            else:
                _, _, action = self.actor.get_action_alt(states)
            action = action.cpu().numpy().flatten()
            # print(action)
        self.actor.train()
        return action

    def train_rl(
        self, state, action, next_state, reward, termination, truncation, gammas
    ):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        termination = torch.FloatTensor(termination).to(self.device)
        truncation = torch.FloatTensor(truncation).to(self.device)

        # calculate the Q
        with torch.no_grad():
            self.actor.eval()
            next_actions, log_probs, _ = self.actor.get_action_alt(next_state)
            self.actor.train()

        cat_state = torch.cat([state, next_state], dim=0)
        cat_actions = torch.cat([action, next_actions], dim=0)
        cat_q1, cat_q2 = self.critic(cat_state, cat_actions)

        q_values_1, q_values_1_next = torch.chunk(cat_q1, chunks=2, dim=0)
        q_values_2, q_values_2_next = torch.chunk(cat_q2, chunks=2, dim=0)

        # print(q_values_1)
        # print(q_values_1.detach().mean())
        # print(q_values_2)

        target_q_values = (
            torch.minimum(q_values_1_next, q_values_2_next)
            - self.log_alpha.exp() * log_probs
        )
        # print('Termination', termination.shape)
        # print('Reward', reward.shape)
        # print('Target Q values', target_q_values.shape)

        # print("===== DEBUG INFO ===========================================================================================================")
        # print("Reward shape:", reward.shape)
        # print("============================================================================================================================")
        # print("Termination shape:", termination.shape)
        # print("============================================================================================================================")

        q_target = (
            reward * self.rewards_scale
            + self.gamma * (1 - termination) * target_q_values
        ).detach()

        q1_loss = F.mse_loss(q_values_1, q_target)
        q2_loss = F.mse_loss(q_values_2, q_target)
        total_q_loss = q1_loss + q2_loss

        self.critic_optim.zero_grad()
        total_q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=100.0)
        self.critic_optim.step()
        # log actor loss, critic loss, entropy loss, alpha

        # Compute gradient and do optimizer step logging #! might remove this
        # critic_grad_norm = (
        #     sum(
        #         [
        #             p.grad.data.norm(2).item() ** 2
        #             for p in self.critic.parameters()
        #         ]
        #     )
        #     ** 0.5
        # )

        if self.total_it % self.target_update_freq == 0:
            # policy update
            next_actions, log_probs, _ = self.actor.get_action_alt(state)

            self.critic.eval()
            # with torch.no_grad():
            q1, q2 = self.critic(state, next_actions)
            self.critic.train()

            min_q = torch.minimum(q1, q2)
            policy_loss = (self.log_alpha.exp() * log_probs - min_q).mean()

            self.actor_optim.zero_grad()
            policy_loss.backward()
            # torch.nn.utils.clip_grad_norm_(
            # self.actor.parameters(), max_norm=1.0
            # )
            self.actor_optim.step()

            # temperature update
            entropy_loss = -(
                self.log_alpha.exp() * (log_probs.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            entropy_loss.backward()
            self.alpha_optimizer.step()

        return {
            "Actor_grad_norm": self.grad_norm(self.actor),
            "Critic_grad_norm": self.grad_norm(self.critic),
            "Alpha_grad_norm": self.log_alpha.grad.norm().item() if self.log_alpha.grad is not None else 0,
            "Actor_loss": policy_loss.cpu().detach().numpy(),
            "Total_critic_loss": total_q_loss.cpu().detach().numpy(),
            "q_values_1": q_values_1.mean().item(),
            "q_values_2": q_values_2.mean().item(),
            "q_target": q_target.mean().item(),
            "critic_1_loss": q1_loss.cpu().detach().numpy(),
            "critic_2_loss": q2_loss.cpu().detach().numpy(),
            "actor_loss": policy_loss.cpu().detach().numpy(),
            "entropy_loss": entropy_loss.cpu().detach().numpy(),
            "log_alpha": self.log_alpha.cpu().detach().numpy(),
            "alpha": self.log_alpha.exp().cpu().detach().numpy(),
            "log_probs": log_probs.mean().cpu().detach().numpy(),
            "entropy": -log_probs.mean().cpu().detach().numpy(),
        }

    def sample_transition(self, replay_buffer, batch_size=256):
        """
        !Check this, because, do we really need both?
        """
        # Sample replay buffer ("task" for multi-task learning)
        state, action, next_state, reward, termination, truncation, task, ind = (
            replay_buffer.sample(batch_size)
        )
        next_state, reward, termination, truncation, gammas = (
            replay_buffer.n_step_return(self.n_step, ind, self.gamma)
        )
        return state, action, next_state, reward, termination, truncation, gammas

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, terminations, truncations, gammas = self.sample_transition(
            replay_buffer, batch_size
        )
        loss_info = self.train_rl(state, action, next_state, reward, terminations, truncations, gammas)
        return loss_info

    def grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2).item() if p.grad is not None else 0
            total_norm += param_norm**2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm
    
    def set_alpha_list(self, alpha_list):
        self.alpha_list = alpha_list
        print("Alpha list: ", self.alpha_list)
    
    def update_alpha(self, stage : int, run_name : str):
        if stage < len(self.alpha_list):
            self.log_alpha = torch.tensor(
                [np.log(self.alpha_list[stage])],
                requires_grad=True,
                dtype=torch.float32,
                device=self.device,
            )
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=self.alpha_lr, betas=(0.5, 0.999)
            )
        else:
            # choose the last value
            self.log_alpha = torch.tensor(
                [np.log(self.alpha_list[-1])],
                requires_grad=True,
                dtype=torch.float32,
                device=self.device,
            )
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=self.alpha_lr, betas=(0.5, 0.999)
            )         
            
        self.save(run_name, filename=f"checkpoint_stage_{stage}.pth")

    def save(self, run_name, filename=None):
        self.actor.to("cpu")
        #path_save_model = '/home/bbruno/Documents/the-barn-challenge-CrossQ/end_to_end/trained_models'
        path_save_model = '/root/e2e_crossq/src/the-barn-challenge-CrossQ/end_to_end/trained_models'
        folder_path = join(path_save_model, run_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        files = os.listdir(folder_path)
        num_files = len(files)
        checkpoint_filename = join(folder_path, filename or f"checkpoint_{num_files}.pth")
        
        state = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optim.state_dict(),
            "critic_optimizer_state_dict": self.critic_optim.state_dict(),
            "log_alpha": self.log_alpha,
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "gamma": self.gamma,
            "policy_update_freq": self.target_update_freq,
        }
        
        torch.save(state, checkpoint_filename)
        
        # with open(join(dir, filename + "_actor"), "wb") as f:
        #     pickle.dump(self.actor.state_dict(), f)
        #with open(join(dir, filename + "_noise"), "wb") as f:
        #    pickle.dump(self.exploration_noise, f)
        self.actor.to(self.device)
        
    def load(self, run_name, checkpoint_filename):
        #path_save_model = '/home/bbruno/Documents/the-barn-challenge-CrossQ/end_to_end/trained_models'
        path_save_model = '/root/e2e_crossq/src/the-barn-challenge-CrossQ/end_to_end/trained_models'
        folder_path = join(path_save_model, run_name)
        checkpoint_full_path = join(folder_path, checkpoint_filename)
        if not os.path.isfile(checkpoint_full_path):
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_full_path}' does not exist.")

        checkpoint = torch.load(checkpoint_full_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optim.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.log_alpha = checkpoint["log_alpha"]
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        self.gamma = checkpoint["gamma"]
        self.target_update_freq = checkpoint["policy_update_freq"]
        self.actor.to(self.device)

    # def load(self, dir, filename):
    #     with open(join(dir, filename + "_actor"), "rb") as f:
    #         self.actor.load_state_dict(pickle.load(f))
    #         self.actor_target = copy.deepcopy(self.actor)
    #     #with open(join(dir, filename + "_noise"), "rb") as f:
    #     #    self.exploration_noise = pickle.load(f)


class CrossQCritic(nn.Module):
    # TODO: adjust this to work as the td3 critic, the problem is not making the NN too deep
    # TODO: also make the network parameters adjustable from the configuration
    # TODO: one option is to make a CrossQ-MLP for the head
    def __init__(self, state_preprocess, head, use_continual_backprop=False, laser_dim=720):
        super(CrossQCritic, self).__init__()

        self.laser_dim = laser_dim
        # Q1 architecture
        self.state_preprocess1 = state_preprocess
        self.head1 = head
        fc1 = nn.Linear(self.state_preprocess1.hidden_size, 1)
        
        # Q2 architecture
        self.state_preprocess2 = state_preprocess
        self.head2 = head
        fc2 = nn.Linear(self.state_preprocess2.hidden_size, 1)
        
        if use_continual_backprop:
            prev_lin = head.get_last_layers()
            br_1 = BatchRenorm(prev_lin.out_features)
            cbp1 = CBPLinear(
                        in_layer=prev_lin,
                        out_layer=fc1,
                        bn_layer = br_1,
                        init='orthogonal',
                    )
            
            br_2 = BatchRenorm(prev_lin.out_features)
            cbp2 = CBPLinear(
                        in_layer=prev_lin,
                        out_layer=fc2,
                        bn_layer = br_2,
                        init='orthogonal',
                    )
            
            self.fc1 = nn.Sequential(br_1, cbp1, fc1)
            self.fc2 = nn.Sequential(br_2, cbp2, fc2)
        
        else:
            self.fc1 = nn.Sequential(br_1, fc1)
            self.fc2 = nn.Sequential(br_2, fc2)
            

    def _initialize_weights(self):
        for layer in list(self.q1) + list(self.q2):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cutoff = (self.head1.input_dim - self.laser_dim - 2) // 4
        no_laser_data = state[:, :, -cutoff:].reshape(state.shape[0], -1)
        state = state[:, :, :-cutoff]
        state1 = self.state_preprocess1(state) if self.state_preprocess1 else state
        state1 = torch.cat([state1, no_laser_data], dim=1)
        sa1 = torch.cat([state1, action], dim=1)
        x1 = self.head1(sa1)
        q1 = self.fc1(x1)

        state2 = self.state_preprocess2(state) if self.state_preprocess2 else state
        state2 = torch.cat([state2, no_laser_data], dim=1)
        sa2 = torch.cat([state2, action], dim=1)
        x2 = self.head2(sa2)
        q2 = self.fc2(x2)

        return q1, q2


class Actor(nn.Module):
    def __init__(
        self,
        state_preprocess,
        head,
        action_dim,
        input_dim = 744,
        action_space_high = np.array([2.0, 3.14]),
        action_space_low = np.array([-1.0, -3.14]),
        log_std_bounds: List[float] = [-20.0, 2.0],
        use_continual_backprop=False,
        laser_dim=720,
    ):
        super(Actor, self).__init__()
        self.state_preprocess = state_preprocess
        self.head = head
        self.input_dim = input_dim 
        self.laser_dim = laser_dim

        mean = nn.Linear(self.head.feature_dim, action_dim)
        log_std = nn.Linear(self.head.feature_dim, action_dim)
        
        if use_continual_backprop:
            prev_lin = self.head.get_last_layers()
            mean_br = BatchRenorm(prev_lin.out_features)
            cbp_mean = CBPLinear(
                        in_layer=prev_lin,
                        out_layer=mean,
                        bn_layer = mean_br,
                        init='orthogonal',
                    )
            
            log_std_br = BatchRenorm(prev_lin.out_features)
            cbp_log_std = CBPLinear(
                        in_layer=prev_lin,
                        out_layer=log_std,
                        bn_layer = log_std_br,
                        init='orthogonal',
                    )
            
            self.mean = nn.Sequential(mean_br, cbp_mean, mean)
            self.log_std = nn.Sequential(log_std_br, cbp_log_std, log_std)
        else:
            self.mean = mean
            self.log_std = log_std

        self.log_std_min, self.log_std_max = log_std_bounds
        
        self.register_buffer("action_scale", torch.tensor((action_space_high - action_space_low) / 2.0))
        self.register_buffer("action_bias", torch.tensor((action_space_high + action_space_low) / 2.0))
        print("========================================================================")
        print('Action scale: ', self.action_scale, 'Action bias: ', self.action_bias)
        print("========================================================================")

    def forward(self, state):
        # input dim: 744 but state is 726 and no laser data is 24 = 6*4
        cutoff = (self.input_dim - self.laser_dim) // 4 # 6
        no_laser_data = state[:, :, -cutoff:].reshape(state.shape[0], -1) # 24
        #print('No laser data: ', no_laser_data.shape)
        state = state[:, :, :-cutoff] # 720
        #print('State before preprocessing: ', state.shape)
        s = self.state_preprocess(state) if self.state_preprocess else state # 720
        #print('State after preprocessing: ', s.shape) 
        #print('No laser data: ', no_laser_data.shape)
        s = torch.cat([s, no_laser_data], dim=1) # 744
        #print('State previous ', s.shape, '======================================')
        
        #print('S', s.shape)
        mean = self.mean(self.head(s))
        log_std = self.log_std(self.head(s))

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        return mean, log_std

    def get_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparametrization trick
        normal = torch.distributions.Normal(mean, std)
        epsilon = normal.rsample()
        squashed_epsilon = torch.tanh(epsilon)

        # Action bounds
        action = self.action_scale * squashed_epsilon + self.action_bias

        # Adjust log probability to compensate for the tanh squashing.
        # Using the change-of-variable formula:
        # p_y(y) = p_x(x) * |dx/dy| => log p_y(y) = log p_x(x) + log |dx/dy|
        # log p_y(y) = log p_x(x) - sum(log(1 - tanh(x)^2))
        # log_prob = normal.log_prob(epsilon) - torch.log(self.action_scale * (1 - squashed_epsilon.pow(2)) + 1e-6)
        log_prob = normal.log_prob(epsilon) - torch.log(
            (1 - squashed_epsilon.pow(2)) + 1e-6
        )
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean

    def get_action_alt(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Forward pass
        mean, log_std = self.forward(state)
        std = log_std.exp()

        dist = SquashedNormal(mean, std)

        # Sample and compute log prob
        sample = dist.rsample()
        log_prob = dist.log_prob(sample).sum(1, keepdim=True)

        # Scale and shift action
        action = sample * self.action_scale + self.action_bias
        mean_action = dist.mean * self.action_scale + self.action_bias

        return action, log_prob, mean_action


class Model(nn.Module):
    def __init__(self, state_preprocess, head, state_dim, deterministic=False):
        super(Model, self).__init__()
        self.state_preprocess = state_preprocess
        self.head = head
        self.deterministic = deterministic
        self.history_length, self.state_dim = state_dim
        self.laser_dim = 720
        self.feature_dim = self.state_dim - self.laser_dim
        if not deterministic:
            self.state_dim *= 2
            self.laser_dim *= 2
            self.feature_dim *= 2

        self.laser_state_fc = nn.Sequential(
            *[nn.Linear(self.laser_dim, 512), nn.Tanh()]
        )
        self.feature_state_fc = nn.Sequential(
            *[nn.Linear(self.feature_dim, 512), nn.Tanh()]
        )
        self.reward_fc = nn.Linear(head.feature_dim, 1)
        self.done_fc = nn.Linear(head.feature_dim, 1)

    def forward(self, state, action):
        s = self.state_preprocess(state) if self.state_preprocess else state
        sa = torch.cat([s, action], dim=1)
        x = self.head(sa)
        ls = self.laser_state_fc(x)
        fs = self.feature_state_fc(x)
        r = self.reward_fc(x)
        d = self.done_fc(x)
        if self.deterministic:
            s = torch.cat([ls, fs], dim=1)
        else:
            s = torch.cat(
                [
                    ls[:, : self.laser_dim // 2],
                    fs[:, : self.feature_dim // 2],
                    ls[:, self.laser_dim // 2],
                    fs[:, self.feature_dim // 2],
                ],
                axis=1,
            )
        return s, r, d

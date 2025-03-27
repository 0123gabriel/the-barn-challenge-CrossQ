from typing import Tuple, List
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
import copy
from sac.utils import BatchRenorm, SquashedNormal
from sac.net import get_activation, MLP


from os.path import join

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
        alpha_lr=0.005,  # policy_arg
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

        self.target_entropy = -torch.prod(torch.Tensor(action_range)).to(self.device)
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
            "log_probs": log_probs.cpu().detach().numpy(),
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

    def save(self, dir, filename):
        self.actor.to("cpu")
        with open(join(dir, filename + "_actor"), "wb") as f:
            pickle.dump(self.actor.state_dict(), f)
        #with open(join(dir, filename + "_noise"), "wb") as f:
        #    pickle.dump(self.exploration_noise, f)
        self.actor.to(self.device)

    def load(self, dir, filename):
        with open(join(dir, filename + "_actor"), "rb") as f:
            self.actor.load_state_dict(pickle.load(f))
            self.actor_target = copy.deepcopy(self.actor)
        #with open(join(dir, filename + "_noise"), "rb") as f:
        #    self.exploration_noise = pickle.load(f)


class CrossQCritic(nn.Module):
    # TODO: adjust this to work as the td3 critic, the problem is not making the NN too deep
    # TODO: also make the network parameters adjustable from the configuration
    # TODO: one option is to make a CrossQ-MLP for the head
    def __init__(self, state_preprocess, head):
        super(CrossQCritic, self).__init__()

        # Q1 architecture
        self.state_preprocess1 = state_preprocess
        self.head1 = head
        self.fc1 = nn.Linear(self.state_preprocess1.hidden_size, 1)

        # Q2 architecture
        self.state_preprocess2 = state_preprocess
        self.head2 = head
        self.fc2 = nn.Linear(self.state_preprocess2.hidden_size, 1)

    def _initialize_weights(self):
        for layer in list(self.q1) + list(self.q2):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cutoff = (self.state_preprocess1.input_dim - 720) // 4
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
        action_space_high = 2,
        action_space_low = -2,
        log_std_bounds: List[float] = [-20.0, 2.0],
    ):
        super(Actor, self).__init__()
        self.state_preprocess = state_preprocess
        self.head = head

        self.fc = nn.Linear(self.state_preprocess.feature_dim, action_dim)

        self.mean = nn.Linear(self.head.feature_dim, action_dim)
        self.log_std = nn.Linear(self.head.feature_dim, action_dim)

        self.log_std_min, self.log_std_max = log_std_bounds
        
        self.register_buffer("action_scale", torch.tensor((action_space_high - action_space_low) / 2.0))
        self.register_buffer("action_bias", torch.tensor((action_space_high + action_space_low) / 2.0))

    def forward(self, state):
        #print('State', state.shape)
        cutoff = (self.state_preprocess.input_dim - 720) // 4
        no_laser_data = state[:, :, -cutoff:].reshape(state.shape[0], -1)
        state = state[:, :, :-cutoff]
        s = self.state_preprocess(state) if self.state_preprocess else state
        #print('State: ', s.shape)
        #print('No laser data: ', no_laser_data.shape)
        s = torch.cat([s, no_laser_data], dim=1)
        
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

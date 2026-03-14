import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from d3qn_network import D3QN
from replay_buffer import PrioritizedReplayBuffer
from utils import get_torch_device


class D3QNPERAgent:
    def __init__(self, config):
        self.device = get_torch_device()

        self.num_actions = config["environment"]["action_space"]

        self.gamma = config["training"]["gamma"]
        self.learning_rate = config["training"]["learning_rate"]
        self.batch_size = config["training"]["batch_size"]
        self.grad_clip = config["optimization"]["grad_clip"]

        self.epsilon = config["exploration"]["epsilon_start"]
        self.epsilon_min = config["exploration"]["epsilon_min"]
        self.epsilon_decay = config["exploration"]["epsilon_decay"]

        self.sync_frequency = config["target_network"]["sync_frequency"]
        self.learn_step_counter = 0

        capacity = config["replay_buffer"]["capacity"]
        self.learning_starts = config["replay_buffer"]["learning_starts"]
        self.alpha = config["prioritized_replay"]["alpha"]
        self.beta = config["prioritized_replay"]["beta_start"]
        self.beta_frames = max(1, config["prioritized_replay"]["beta_frames"])
        self.beta_increment = (1.0 - self.beta) / self.beta_frames
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=capacity,
            alpha=self.alpha,
            epsilon=config["prioritized_replay"]["epsilon"],
        )

        self.policy_net = D3QN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_net = D3QN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return torch.argmax(q_values, dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < max(self.learning_starts, self.batch_size):
            return None

        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(
            self.batch_size,
            beta=self.beta,
        )

        states_t = torch.from_numpy(states).float().to(self.device, non_blocking=True)
        next_states_t = torch.from_numpy(next_states).float().to(self.device, non_blocking=True)
        actions_t = torch.from_numpy(actions).long().to(self.device, non_blocking=True).unsqueeze(1)
        rewards_t = torch.from_numpy(rewards).float().to(self.device, non_blocking=True)
        dones_t = torch.from_numpy(dones).float().to(self.device, non_blocking=True)
        weights_t = torch.from_numpy(weights).float().to(self.device, non_blocking=True)

        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        td_errors = target_q - current_q
        per_sample_loss = self.loss_fn(current_q, target_q)
        loss = (weights_t * per_sample_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors.detach().abs().cpu().numpy())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        self.beta = min(1.0, self.beta + self.beta_increment)

        self.learn_step_counter += 1
        if self.learn_step_counter % self.sync_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

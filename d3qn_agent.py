import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from d3qn_network import D3QN
from replay_buffer import PrioritizedReplayBuffer


class D3QNAgent:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # action space
        self.num_actions = config["environment"]["action_space"]

        # training parameters
        self.gamma = config["training"]["gamma"]
        self.learning_rate = config["training"]["learning_rate"]
        self.batch_size = config["training"]["batch_size"]
        self.grad_clip = config["optimization"]["grad_clip"]

        # exploration parameters
        self.epsilon = config["exploration"]["epsilon_start"]
        self.epsilon_min = config["exploration"]["epsilon_min"]
        self.epsilon_decay = config["exploration"]["epsilon_decay"]

        # replay buffer parameters
        replay_config = config["replay_buffer"]
        per_config = config["prioritized_replay"]
        self.learning_starts = replay_config["learning_starts"]
        self.beta_start = per_config["beta_start"]
        self.beta_frames = per_config["beta_frames"]
        self.frame_idx = 0

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=replay_config["capacity"],
            alpha=per_config["alpha"],
            epsilon=per_config["epsilon"],
        )

        # target network sync
        self.sync_frequency = config["target_network"]["sync_frequency"]
        self.learn_step_counter = 0

        # networks
        self.policy_net = D3QN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_net = D3QN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer and loss
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
        self.frame_idx += 1

    def beta_by_frame(self):
        progress = min(1.0, self.frame_idx / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)

    def train_step(self):
        if len(self.replay_buffer) < max(self.batch_size, self.learning_starts):
            return None

        beta = self.beta_by_frame()
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(
            self.batch_size,
            beta,
        )

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        td_errors = target_q - current_q
        loss = (self.loss_fn(current_q, target_q) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        self.learn_step_counter += 1
        if self.learn_step_counter % self.sync_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

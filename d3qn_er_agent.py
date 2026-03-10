import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from d3qn_network import D3QN


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, copy=True),
            action,
            reward,
            np.array(next_state, copy=True),
            done,
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )


class D3QNERAgent:
    def __init__(self, config):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

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

        # target network sync
        self.sync_frequency = config["target_network"]["sync_frequency"]
        self.learn_step_counter = 0

        # replay buffer
        capacity = config["replay_buffer"]["capacity"]
        self.learning_starts = config["replay_buffer"]["learning_starts"]
        self.replay_buffer = ReplayBuffer(capacity)

        # networks
        self.policy_net = D3QN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_net = D3QN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

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
        if len(self.replay_buffer) < self.learning_starts:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.from_numpy(states).float().to(self.device, non_blocking=True)
        next_states_t = torch.from_numpy(next_states).float().to(self.device, non_blocking=True)
        actions_t = torch.from_numpy(actions).long().to(self.device, non_blocking=True).unsqueeze(1)
        rewards_t = torch.from_numpy(rewards).to(self.device, non_blocking=True)
        dones_t = torch.from_numpy(dones).to(self.device, non_blocking=True)

        # current Q values from policy net
        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Double DQN target: action selected by policy net, evaluated by target net
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

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

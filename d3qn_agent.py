import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from d3qn_network import D3QN


class D3QNAgent:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # action space
        self.num_actions = config["environment"]["action_space"]

        # training parameters
        self.gamma = config["training"]["gamma"]
        self.learning_rate = config["training"]["learning_rate"]
        self.grad_clip = config["optimization"]["grad_clip"]

        # exploration parameters
        self.epsilon = config["exploration"]["epsilon_start"]
        self.epsilon_min = config["exploration"]["epsilon_min"]
        self.epsilon_decay = config["exploration"]["epsilon_decay"]

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
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return torch.argmax(q_values, dim=1).item()

    def train_step(self, state, action, reward, next_state, done):
        state_t = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_t = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.tensor([[action]], dtype=torch.long, device=self.device)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_t = torch.tensor([done], dtype=torch.float32, device=self.device)

        current_q = self.policy_net(state_t).gather(1, action_t).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_state_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_state_t).gather(1, next_actions).squeeze(1)
            target_q = reward_t + (1 - done_t) * self.gamma * next_q

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

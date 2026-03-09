import torch
import torch.nn as nn
import torch.nn.functional as F


class D3QN(nn.Module):
    def __init__(self, input_channels=4, num_actions=2):
        super(D3QN, self).__init__()

        # convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # fully connected layer
        self.fc = nn.Linear(3136, 512)

        # value stream
        self.value = nn.Linear(512, 1)

        # advantage stream
        self.advantage = nn.Linear(512, num_actions)

    def forward(self, x):
        x = x.float()

        x = self.conv(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc(x))

        value = self.value(x)
        advantage = self.advantage(x)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values
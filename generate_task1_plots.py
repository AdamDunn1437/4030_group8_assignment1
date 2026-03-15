import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# folders following project hierarchy
online_dir = Path("d3qn_results")
per_dir = Path("d3qn_per_results")

# load results
online_rewards = np.load(online_dir / "episode_rewards.npy")
online_losses = np.load(online_dir / "episode_losses.npy")

per_rewards = np.load(per_dir / "episode_rewards.npy")
per_losses = np.load(per_dir / "episode_losses.npy")


def moving_average(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode="valid")


# moving averages
online_rewards_ma = moving_average(online_rewards)
online_losses_ma = moving_average(online_losses)

per_rewards_ma = moving_average(per_rewards)
per_losses_ma = moving_average(per_losses)


# reward plot (saved inside d3qn_results)
plt.figure(figsize=(10,5))
plt.plot(online_rewards_ma, label="D3QN Online")
plt.plot(per_rewards_ma, label="D3QN + PER")
plt.title("Reward Comparison")
plt.xlabel("Episode")
plt.ylabel("Reward (50 episode moving average)")
plt.legend()
plt.grid(True)
plt.savefig(online_dir / "task_reward_comparison.png", dpi=300)


# loss plot (saved inside d3qn_results)
plt.figure(figsize=(10,5))
plt.plot(online_losses_ma, label="D3QN Online")
plt.plot(per_losses_ma, label="D3QN + PER")
plt.title("Loss Comparison")
plt.xlabel("Episode")
plt.ylabel("Loss (50 episode moving average)")
plt.legend()
plt.grid(True)
plt.savefig(online_dir / "task_loss_comparison.png", dpi=300)


print("Plots saved inside d3qn_results/")
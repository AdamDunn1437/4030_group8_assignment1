import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# result folders
dqn_dir = Path("d3qn_results")
per_dir = Path("d3qn_per_results")


def moving_average(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def load_results(directory):
    rewards = np.load(directory / "episode_rewards.npy")
    losses = np.load(directory / "episode_losses.npy")
    return rewards, losses


def plot_single(data, title, ylabel, save_path):
    smoothed = moving_average(data)

    plt.figure(figsize=(10,5))
    plt.plot(smoothed)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_overlay(curves, labels, title, ylabel, save_path):
    plt.figure(figsize=(10,5))

    for curve, label in zip(curves, labels):
        smoothed = moving_average(curve)
        plt.plot(smoothed, label=label)

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():

    # load results
    dqn_rewards, dqn_losses = load_results(dqn_dir)
    per_rewards, per_losses = load_results(per_dir)

    # =========================
    # Individual plots
    # =========================

    plot_single(
        dqn_rewards,
        "D3QN Online Reward Curve",
        "Reward (50 episode moving average)",
        dqn_dir / "task1_reward_curve.png"
    )

    plot_single(
        dqn_losses,
        "D3QN Online Loss Curve",
        "Loss (50 episode moving average)",
        dqn_dir / "task1_loss_curve.png"
    )

    plot_single(
        per_rewards,
        "D3QN + PER Reward Curve",
        "Reward (50 episode moving average)",
        per_dir / "task3_reward_curve.png"
    )

    plot_single(
        per_losses,
        "D3QN + PER Loss Curve",
        "Loss (50 episode moving average)",
        per_dir / "task3_loss_curve.png"
    )

    # =========================
    # Overlay comparison plots
    # =========================

    plot_overlay(
        [dqn_rewards, per_rewards],
        ["D3QN Online", "D3QN + PER"],
        "Reward Comparison (DQN vs PER)",
        "Reward (50 episode moving average)",
        "reward_comparison_dqn_vs_per.png"
    )

    plot_overlay(
        [dqn_losses, per_losses],
        ["D3QN Online", "D3QN + PER"],
        "Loss Comparison (DQN vs PER)",
        "Loss (50 episode moving average)",
        "loss_comparison_dqn_vs_per.png"
    )

    print("Plots generated successfully.")


if __name__ == "__main__":
    main()
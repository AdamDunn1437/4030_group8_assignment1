import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# result folders
dqn_dir = Path("d3qn_results")
er_dir = Path("d3qn_er_results")
per_dir = Path("d3qn_per_results")


def moving_average(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def load_results(directory):
    rewards_path = directory / "episode_rewards.npy"
    losses_path = directory / "episode_losses.npy"
    if not rewards_path.exists() or not losses_path.exists():
        print(f"Skipping {directory.name}: results not found.")
        return None, None
    return np.load(rewards_path), np.load(losses_path)


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

    # load results (None if a directory is missing)
    dqn_rewards, dqn_losses = load_results(dqn_dir)
    er_rewards,  er_losses  = load_results(er_dir)
    per_rewards, per_losses = load_results(per_dir)

    # =========================
    # Individual plots
    # =========================

    individual = [
        (dqn_rewards, dqn_losses, dqn_dir, "D3QN Online",  "task1"),
        (er_rewards,  er_losses,  er_dir,  "D3QN + ER",    "task2"),
        (per_rewards, per_losses, per_dir, "D3QN + PER",   "task3"),
    ]

    for rewards, losses, directory, label, prefix in individual:
        if rewards is None:
            continue
        plot_single(rewards, f"{label} Reward Curve", "Reward (50 episode moving average)", directory / f"{prefix}_reward_curve.png")
        plot_single(losses,  f"{label} Loss Curve",   "Loss (50 episode moving average)",   directory / f"{prefix}_loss_curve.png")

    # =========================
    # Overlay comparison plots
    # =========================

    all_agents = [
        (dqn_rewards, dqn_losses, "D3QN Online"),
        (er_rewards,  er_losses,  "D3QN + ER"),
        (per_rewards, per_losses, "D3QN + PER"),
    ]
    available = [(r, l, lbl) for r, l, lbl in all_agents if r is not None]

    if len(available) >= 2:
        plot_overlay(
            [r for r, _, _ in available],
            [lbl for _, _, lbl in available],
            "Reward Comparison (All Tasks)",
            "Reward (50 episode moving average)",
            "reward_comparison_all.png"
        )
        plot_overlay(
            [l for _, l, _ in available],
            [lbl for _, _, lbl in available],
            "Loss Comparison (All Tasks)",
            "Loss (50 episode moving average)",
            "loss_comparison_all.png"
        )
    else:
        print("Not enough agents with results to generate overlay plots.")

    print("Plots generated successfully.")


if __name__ == "__main__":
    main()
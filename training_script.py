import os
import json
import numpy as np

from utils import load_config, print_config
from environment import make_mario_env
from d3qn_agent import D3QNAgent


def train():
    config = load_config("config.yaml")
    print_config(config)

    env_name = config["environment"]["name"]
    episodes = config["training"]["episodes"]
    max_steps = config["training"]["max_steps_per_episode"]

    env, observation_shape, action_size = make_mario_env(env_name, return_info=True)

    print("\nEnvironment created successfully!")
    print("Observation shape:", observation_shape)
    print("Action size:", action_size)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    agent = D3QNAgent(config)

    episode_rewards = []
    episode_losses = []

    results_dir = "d3qn_results"
    os.makedirs(results_dir, exist_ok=True)

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        step_losses = []

        for step in range(max_steps):
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            loss = agent.train_step(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if loss is not None:
                step_losses.append(loss)

            if done:
                break

        avg_loss = float(np.mean(step_losses)) if len(step_losses) > 0 else 0.0

        episode_rewards.append(float(total_reward))
        episode_losses.append(avg_loss)

        # print every 10 episodes and also the first episode
        if episode == 0 or (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            avg_recent_reward = float(np.mean(recent_rewards))

            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Avg(Last 10): {avg_recent_reward:.2f} | "
                f"Loss: {avg_loss:.6f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    env.close()

    # save raw arrays
    np.save(os.path.join(results_dir, "episode_rewards.npy"), np.array(episode_rewards, dtype=np.float32))
    np.save(os.path.join(results_dir, "episode_losses.npy"), np.array(episode_losses, dtype=np.float32))

    # save model
    agent.save(os.path.join(results_dir, "d3qn_online_model.pth"))

    # save summary json
    summary = {
        "episodes": episodes,
        "final_epsilon": float(agent.epsilon),
        "mean_reward": float(np.mean(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "mean_loss": float(np.mean(episode_losses))
    }

    with open(os.path.join(results_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\nTraining complete.")
    print(f"Results saved to: {results_dir}")

    return episode_rewards, episode_losses


if __name__ == "__main__":
    train()
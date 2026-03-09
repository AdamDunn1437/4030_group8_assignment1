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

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        last_loss = None

        for step in range(max_steps):
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()

            if loss is not None:
                last_loss = loss

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)

        loss_text = f"{last_loss:.4f}" if last_loss is not None else "warming up"
        print(
            f"Episode {episode + 1}/{episodes} | "
            f"Reward: {total_reward:.2f} | "
            f"Loss: {loss_text} | "
            f"Epsilon: {agent.epsilon:.4f} | "
            f"Buffer: {len(agent.replay_buffer)}"
        )

    env.close()
    return episode_rewards


if __name__ == "__main__":
    train()

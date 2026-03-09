import gym
import gym_super_mario_bros
import numpy as np
import torch

from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from torchvision import transforms as T


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.transform = T.Grayscale()

    def observation(self, observation):
        # [H, W, C] -> [C, H, W]
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float32)

        # [C, H, W] -> [1, H, W]
        observation = self.transform(observation)

        # return [H, W]
        return observation.squeeze(0).numpy().astype(np.uint8)


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)

        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=self.shape,
            dtype=np.float32
        )

        self.transforms = T.Compose([
            T.Resize(self.shape, antialias=True),
            T.Normalize(mean=[0.0], std=[255.0])
        ])

    def observation(self, observation):
        # [H, W] -> [1, H, W]
        observation = torch.tensor(observation.copy(), dtype=torch.float32).unsqueeze(0)

        # [1, H, W] normalized to [0,1]
        observation = self.transforms(observation)

        # return [H, W]
        return observation.squeeze(0).numpy().astype(np.float32)


def make_mario_env(env_name="SuperMarioBros-1-1-v3", return_info=False):
    env = gym_super_mario_bros.make(env_name, apply_api_compatibility=True)

    actions = [
        ["right"],
        ["right", "A"]
    ]
    env = JoypadSpace(env, actions)

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    if return_info:
        return env, env.observation_space.shape, env.action_space.n

    return env
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, DQN
import stable_baselines3.common.utils
from stable_baselines3.common.logger import configure


def main():
    stable_baselines3.common.utils.set_random_seed(0)
    new_logger = configure('./rl/output', ["stdout", "csv"])

    env = gym.make("CartPole-v1")
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[8, 8]
    )
    model = DQN(
        "MlpPolicy", env, 
        seed=0,
        policy_kwargs=policy_kwargs,
        exploration_initial_eps=0.2,
        learning_rate=1e-3,
        verbose=1)
    
    model.set_logger(new_logger)
    model.learn(total_timesteps=200000, log_interval=10)

    env = gym.make("CartPole-v1", render_mode="human")
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == '__main__':
    main()
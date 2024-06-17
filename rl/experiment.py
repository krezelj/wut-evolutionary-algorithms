import json

import gymnasium as gym
import torch
from stable_baselines3 import DQN
import stable_baselines3.common.utils
from stable_baselines3.common.logger import configure


def run_experiment(config_path: str = './rl/config.json', verbose: int = 0):
    with open(config_path, 'r') as f:
        config = json.load(f)

    deterministic_values = []
    env = gym.make(config["env_name"])

    for run in range(config["n_runs"]):
        new_logger = configure(config["output_path"] + f"_{run}", ["stdout", "csv"])
        stable_baselines3.common.utils.set_random_seed(config["seed"] + run)
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=config["net_arch"]
        )
        model = DQN(
            "MlpPolicy", 
            env, 
            seed=config["seed"] + run,
            policy_kwargs=policy_kwargs,
            exploration_initial_eps=config["eps_init"],
            exploration_final_eps=config["eps_final"],
            learning_rate=config["lr"],
            verbose=1
        )

        model.set_logger(new_logger)
        model.learn(total_timesteps=config["total_timesteps"], log_interval=10)

        # evaluate deterministic
        env = gym.make(config["env_name"])
        obs, _ = env.reset()
        total_reward = 0
        evaluation_run = 0
        while True:
            if evaluation_run == config["evaluation_runs"]:
                break
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                obs, _ = env.reset()
                evaluation_run += 1
        deterministic_values.append(total_reward / config["evaluation_runs"])

    print(deterministic_values)
    with open(config["output_path"] + '/rl_evaluation.txt', 'w') as f:    
        f.write(str(deterministic_values))


def main():
    run_experiment()


if __name__ == '__main__':
    main()
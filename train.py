import torch
from ddqn import run_dqn
from dqn import run_dqn as run_dqn_original
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from bbrl_algos.models.loggers import Logger
from bbrl_algos.rliable_stats.tests import run_test

def train_and_evaluate(config_file, algorithm):
    cfg = OmegaConf.load(config_file)
    logger = Logger(cfg)
    if algorithm == 'DDQN':
        return run_dqn(cfg, logger)
    elif algorithm == 'DQN':
        return run_dqn_original(cfg, logger)
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')

def main():
    algorithms = ['DQN', 'DDQN']
    config_files = ['dqn_lunarlander.yaml', 'ddqn_lunarlander.yaml']

    all_rewards = []

    for algorithm, config_file in zip(algorithms, config_files):
        rewards = train_and_evaluate(config_file, algorithm)
        all_rewards.append(rewards)
        plt.plot(rewards, label=algorithm)

    # Welch's T-test
    dqn_rewards = all_rewards[0]
    ddqn_rewards = all_rewards[1]
    significant_difference = run_test("Welch t-test", dqn_rewards, ddqn_rewards)
    
    print(f"Significant difference between DQN and DDQN: {significant_difference}")

    plt.legend()
    plt.xlabel('Steps (in thousands)')
    plt.ylabel('Reward')
    plt.title('Learning Curves over 1.5M Steps')
    plt.show()

if __name__ == "__main__":
    main()
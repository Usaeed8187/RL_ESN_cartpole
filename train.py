"""
Training script for ESN-based policy gradient on CartPole-v1.
"""

# Monkey-patch for Gym compatibility
import numpy as np
np.bool8 = np.bool_

import gym
import torch
from torch.optim import Adam
from policy import PolicyNetwork
from esn import EchoStateNetwork
from utils import set_seed, reset_env

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train(env_name: str = 'CartPole-v1',
          seed: int = 1234,
          reservoir_size: int = 500,
          lr: float = 1e-2,
          gamma: float = 0.99,
          num_samples: int = 50,
          episodes: int = 500,
          rewards_plot_path: str = 'training_rewards.png') -> None:
    """
    Train the policy network using REINFORCE with Bayesian model averaging.
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the environment (handles old/new Gym APIs)
    try:
        env = gym.make(env_name, new_step_api=True)
    except TypeError:
        env = gym.make(env_name)

    # Initial reset
    obs = reset_env(env, seed)
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Build models
    esn = EchoStateNetwork(input_dim, reservoir_size).to(device)
    policy = PolicyNetwork(esn, action_dim).to(device)
    optimizer = Adam(policy.parameters(), lr=lr)
    reward_sums = []

    for episode in range(1, episodes + 1):
        # Reset env at start of each episode (first episode already reset above)
        if episode > 1:
            obs = reset_env(env)

        state = torch.tensor(obs, dtype=torch.float32).to(device)
        rewards = []
        log_probs = []
        done = False

        while not done:
            # Bayesian averaging across num_samples forward passes
            samples = [policy(state) for _ in range(num_samples)]
            action_probs = torch.stack(samples).mean(0)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            # Step environment
            out = env.step(action.item())
            if len(out) == 5:
                obs, reward, terminated, truncated, _ = out
                done = terminated or truncated
            else:
                obs, reward, done, _ = out

            state = torch.tensor(obs, dtype=torch.float32).to(device)
            rewards.append(reward)

        episode_reward = float(sum(rewards))
        reward_sums.append(episode_reward)

        # Compute discounted returns
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        # use unbiased=False to avoid ddof warning when len==1
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        # Policy gradient update
        loss = -torch.stack([lp * R for lp, R in zip(log_probs, returns)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f'Episode {episode}, Total Reward: {episode_reward}')

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), reward_sums, label='Episode reward sum')
    plt.xlabel('Episode')
    plt.ylabel('Sum of rewards')
    plt.title('Training Reward per Episode')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(rewards_plot_path)
    plt.close()
    print(f'Saved reward plot to {rewards_plot_path}')

    env.close()

if __name__ == '__main__':
    train()
